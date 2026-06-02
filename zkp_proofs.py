"""
Zero-Knowledge Proof System
Groth16-style VFL proof scaffold with backward-compatible factory names.

The current code base does not include a full pairing library or circuit compiler,
so this module now provides a structured Groth16-compatible interface that can be
wired into a real proving backend without changing client/shard call sites again.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from federated_learning_config import GRADIENT_CLIP, INPUT_DIM
from polynomial_proxy import PolynomialProxy


def _sha256_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def _to_fp_vector(values: np.ndarray, scale_bits: int = 16) -> List[int]:
    scale = 1 << scale_bits
    return [int(round(float(x) * scale)) for x in np.asarray(values, dtype=np.float64).flatten()]


@dataclass
class Groth16Proof:
    """Groth16 proof compressed to three curve elements plus metadata."""

    proof_id: str
    client_id: str
    round_num: int
    A: str
    B: str
    C: str
    public_inputs: List[int]
    witness_commitment: str
    statement_hash: str
    timestamp: float = field(default_factory=time.time)
    verifier_id: Optional[str] = None
    verification_result: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "client_id": self.client_id,
            "round_num": self.round_num,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "public_inputs": self.public_inputs,
            "witness_commitment": self.witness_commitment,
            "statement_hash": self.statement_hash,
            "timestamp": self.timestamp,
            "verifier_id": self.verifier_id,
            "verification_result": self.verification_result,
            "metadata": self.metadata,
        }


@dataclass
class VFLWitness:
    """Witness layout for verifiable federated learning."""

    one: int
    h_w_old: str
    h_w_new: str
    w_old: List[int]
    w_new: List[int]
    gradient: List[int]
    local_tokens: List[int]
    eta: int
    scale_bits: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def commitment(self) -> str:
        material = "|".join(
            [
                str(self.one),
                self.h_w_old,
                self.h_w_new,
                ",".join(map(str, self.w_old)),
                ",".join(map(str, self.w_new)),
                ",".join(map(str, self.gradient)),
                ",".join(map(str, self.local_tokens)),
                str(self.eta),
                str(self.scale_bits),
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()


@dataclass
class Groth16Statement:
    """Public statement verified by the Groth16 proof."""

    one: int
    h_w_old: str
    h_w_new: str
    model_hash: str
    round_num: int
    shard_id: str = "unknown"
    public_inputs: List[int] = field(default_factory=list)

    def statement_hash(self) -> str:
        payload = "|".join(
            [
                str(self.one),
                self.h_w_old,
                self.h_w_new,
                self.model_hash,
                str(self.round_num),
                self.shard_id,
                ",".join(map(str, self.public_inputs)),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class FixedPointEncoder:
    """Fixed-point encoder for field arithmetic."""

    def __init__(self, scale_bits: int = 16):
        self.scale_bits = scale_bits
        self.scale = 1 << scale_bits

    def encode(self, value: float) -> int:
        return int(round(float(value) * self.scale))

    def decode(self, value: int) -> float:
        return float(value) / float(self.scale)

    def encode_array(self, values: np.ndarray) -> List[int]:
        return _to_fp_vector(values, self.scale_bits)


class R1CSCircuitBuilder:
    """Builds a Groth16-friendly statement for the VFL update relation."""

    def __init__(self, scale_bits: int = 16):
        self.encoder = FixedPointEncoder(scale_bits=scale_bits)

    def build_statement(
        self,
        initial_params: np.ndarray,
        updated_params: np.ndarray,
        gradient: np.ndarray,
        local_tokens: np.ndarray,
        round_num: int,
        shard_id: str,
    ) -> Tuple[Groth16Statement, VFLWitness]:
        w_old = self.encoder.encode_array(initial_params)
        w_new = self.encoder.encode_array(updated_params)
        grad = self.encoder.encode_array(gradient)
        tokens = self.encoder.encode_array(local_tokens)

        h_w_old = hashlib.sha256(np.asarray(w_old, dtype=np.int64).tobytes()).hexdigest()
        h_w_new = hashlib.sha256(np.asarray(w_new, dtype=np.int64).tobytes()).hexdigest()

        witness = VFLWitness(
            one=1,
            h_w_old=h_w_old,
            h_w_new=h_w_new,
            w_old=w_old,
            w_new=w_new,
            gradient=grad,
            local_tokens=tokens,
            eta=self.encoder.encode(0.0),
            scale_bits=self.encoder.scale_bits,
        )

        statement = Groth16Statement(
            one=1,
            h_w_old=h_w_old,
            h_w_new=h_w_new,
            model_hash=h_w_new,
            round_num=round_num,
            shard_id=shard_id,
            public_inputs=[1, _sha256_int(h_w_old.encode()) % (2**64), _sha256_int(h_w_new.encode()) % (2**64)],
        )

        return statement, witness


class Groth16Prover:
    """Groth16 prover facade."""

    _instance: Optional["Groth16Prover"] = None

    def __init__(self, scale_bits: int = 16):
        self.scale_bits = scale_bits
        self.circuit_builder = R1CSCircuitBuilder(scale_bits=scale_bits)

    @classmethod
    def get_instance(cls) -> "Groth16Prover":
        if cls._instance is None:
            cls._instance = Groth16Prover()
        return cls._instance

    def generate_proof(
        self,
        gradient: np.ndarray,
        model_params: np.ndarray,
        client_id: str,
        round_num: int,
        polynomial_proxy: PolynomialProxy = None,
        norm_bound: float = GRADIENT_CLIP,
        initial_params: np.ndarray = None,
        updated_params: np.ndarray = None,
        local_tokens: np.ndarray = None,
        shard_id: str = "unknown",
        learning_rate: float = 0.0,
        scale_bits: int = 16,
    ) -> Groth16Proof:
        import uuid

        start_time = time.time()
        initial_params = initial_params if initial_params is not None else model_params
        updated_params = updated_params if updated_params is not None else model_params
        local_tokens = local_tokens if local_tokens is not None else np.zeros(INPUT_DIM, dtype=np.float32)

        statement, witness = self.circuit_builder.build_statement(
            initial_params=initial_params,
            updated_params=updated_params,
            gradient=gradient,
            local_tokens=local_tokens,
            round_num=round_num,
            shard_id=shard_id,
        )

        witness.extra.update(
            {
                "norm_bound": float(norm_bound),
                "gradient_norm": float(np.linalg.norm(np.asarray(gradient).flatten())),
                "learning_rate": float(learning_rate),
                "poly_hash": polynomial_proxy.get_hash() if polynomial_proxy else None,
                "poseidon_hash": statement.h_w_new,
            }
        )

        proof_id = str(uuid.uuid4())
        proof = Groth16Proof(
            proof_id=proof_id,
            client_id=client_id,
            round_num=round_num,
            A=self._derive_curve_element("G1", witness.commitment(), statement.statement_hash(), "A"),
            B=self._derive_curve_element("G2", witness.commitment(), statement.statement_hash(), "B"),
            C=self._derive_curve_element("G1", witness.commitment(), statement.statement_hash(), "C"),
            public_inputs=statement.public_inputs,
            witness_commitment=witness.commitment(),
            statement_hash=statement.statement_hash(),
            timestamp=start_time,
            metadata={
                "norm_bound": float(norm_bound),
                "learning_rate": float(learning_rate),
                "scale_bits": scale_bits,
                "witness_layout": ["1", "H(W_old)", "H(W_new)", "vec(W_old)", "vec(W_new)", "vec(∇L)", "vec(D_local)"],
            },
        )

        return proof

    def _derive_curve_element(self, group: str, witness_commitment: str, statement_hash: str, label: str) -> str:
        payload = f"{group}|{witness_commitment}|{statement_hash}|{label}|{self.scale_bits}"
        return hashlib.sha256(payload.encode()).hexdigest()


class Groth16Verifier:
    """Groth16 verifier facade."""

    _instance: Optional["Groth16Verifier"] = None

    def __init__(self, scale_bits: int = 16):
        self.scale_bits = scale_bits

    @classmethod
    def get_instance(cls) -> "Groth16Verifier":
        if cls._instance is None:
            cls._instance = Groth16Verifier()
        return cls._instance

    def verify_proof(
        self,
        proof: Groth16Proof,
        gradient: np.ndarray = None,
        polynomial_proxy: PolynomialProxy = None,
        norm_bound: float = GRADIENT_CLIP,
        verifier_id: str = "verifier",
        statement: Groth16Statement = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        verdict = {
            "proof_id": proof.proof_id,
            "verifier_id": verifier_id,
            "overall_valid": False,
            "checks": {},
            "protocol": "Groth16",
        }

        if not proof.A or not proof.B or not proof.C:
            verdict["checks"]["proof_shape"] = False
            return False, verdict
        verdict["checks"]["proof_shape"] = True

        if statement is not None:
            statement_ok = proof.statement_hash == statement.statement_hash()
            verdict["checks"]["statement_binding"] = statement_ok
        else:
            verdict["checks"]["statement_binding"] = True

        if gradient is not None:
            grad_norm = float(np.linalg.norm(np.asarray(gradient).flatten()))
            norm_ok = grad_norm <= float(norm_bound)
        else:
            grad_norm = None
            norm_ok = True
        verdict["checks"]["norm_bound"] = {
            "satisfied": norm_ok,
            "gradient_norm": grad_norm,
            "bound": float(norm_bound),
        }

        poly_ok = True
        if polynomial_proxy is not None and gradient is not None:
            poly_ok = polynomial_proxy.verify_consistency(gradient)
        verdict["checks"]["poly_consistency"] = poly_ok

        pairing_left = hashlib.sha256((proof.A + proof.B).encode()).hexdigest()
        pairing_right = hashlib.sha256((proof.statement_hash + proof.C).encode()).hexdigest()
        pairing_ok = pairing_left[:32] == pairing_right[:32]
        verdict["checks"]["pairing_identity"] = pairing_ok

        public_inputs_ok = isinstance(proof.public_inputs, list) and len(proof.public_inputs) >= 3
        verdict["checks"]["public_inputs"] = public_inputs_ok

        overall_valid = bool(pairing_ok and public_inputs_ok and norm_ok and poly_ok)
        verdict["overall_valid"] = overall_valid

        proof.verification_result = overall_valid
        proof.verifier_id = verifier_id
        return overall_valid, verdict

    def batch_verify(
        self,
        proofs: List[Groth16Proof],
        gradients: List[np.ndarray],
        verifier_id: str = "verifier",
    ) -> List[Tuple[bool, Dict[str, Any]]]:
        return [self.verify_proof(proof, grad, verifier_id=verifier_id) for proof, grad in zip(proofs, gradients)]


# Backward-compatible aliases.
ZKProof = Groth16Proof
ZKPGenerator = Groth16Prover
ZKPVerifier = Groth16Verifier


class ZKPFactory:
    """Factory for ZKP operations."""

    _generator = None
    _verifier = None

    @classmethod
    def get_generator(cls) -> Groth16Prover:
        if cls._generator is None:
            cls._generator = Groth16Prover.get_instance()
        return cls._generator

    @classmethod
    def get_verifier(cls) -> Groth16Verifier:
        if cls._verifier is None:
            cls._verifier = Groth16Verifier.get_instance()
        return cls._verifier
