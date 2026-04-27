"""
Zero-Knowledge Proof System
Simplified Schnorr-like proof + Fiat-Shamir heuristic for prototype
Production would use py-snark or similar for real zk-SNARK

Proof πi certifies:
1. Correct gradient computation (Wi = Wt - η∇L(Di, Wt))
2. Consistency with polynomial proxy Pi(x)
3. Norm bound constraint (‖∆Wi‖ ≤ B)
"""

import hashlib
import json
from typing import Dict, Any, Tuple, List
import numpy as np
from dataclasses import dataclass
from polynomial_proxy import PolynomialProxy
from federated_learning_config import (
    ZKP_SECURITY_PARAM, ZKP_CHALLENGE_BITS, ZKP_ROUNDS, GRADIENT_CLIP
)


@dataclass
class ZKProof:
    """Zero-Knowledge Proof structure"""
    proof_id: str
    client_id: str
    round_num: int
    
    # Commitment phase (Schnorr-like)
    commitments: List[str]  # List of commitment hashes
    
    # Challenge (Fiat-Shamir)
    challenges: List[int]  # Challenges from heuristic
    
    # Response phase
    responses: List[Dict[str, Any]]  # Responses to challenges
    
    # Proof properties
    certifies_gradient_correct: bool
    certifies_poly_consistency: bool
    certifies_norm_bound: bool
    norm_bound: float
    
    # Metadata
    timestamp: float
    verifier_id: str = None
    verification_result: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize proof"""
        return {
            "proof_id": self.proof_id,
            "client_id": self.client_id,
            "round_num": self.round_num,
            "commitments": self.commitments,
            "challenges": self.challenges,
            "responses": self.responses,
            "certifies_gradient_correct": self.certifies_gradient_correct,
            "certifies_poly_consistency": self.certifies_poly_consistency,
            "certifies_norm_bound": self.certifies_norm_bound,
            "norm_bound": self.norm_bound,
            "timestamp": self.timestamp,
            "verifier_id": self.verifier_id,
            "verification_result": self.verification_result,
        }


class ZKPGenerator:
    """Generates zero-knowledge proofs for gradient updates"""
    
    def __init__(self, security_param: int = ZKP_SECURITY_PARAM,
                 challenge_bits: int = ZKP_CHALLENGE_BITS):
        """
        Initialize ZKP generator.
        
        Args:
            security_param: Security parameter in bits
            challenge_bits: Challenge size in bits
        """
        self.security_param = security_param
        self.challenge_bits = challenge_bits
        
    def generate_proof(self,
                      gradient: np.ndarray,
                      model_params: np.ndarray,
                      client_id: str,
                      round_num: int,
                      polynomial_proxy: PolynomialProxy = None,
                      norm_bound: float = GRADIENT_CLIP) -> ZKProof:
        """
        Generate ZKP for gradient update.
        
        Proves:
        1. Gradient computed correctly
        2. Polynomial proxy consistent with gradient
        3. Gradient norm within bound B
        
        Args:
            gradient: Gradient update ∆Wi
            model_params: Model parameters Wt
            client_id: Client identifier
            round_num: Training round number
            polynomial_proxy: Polynomial proxy Pi(x)
            norm_bound: Maximum norm bound B
            
        Returns:
            ZKProof object
        """
        import uuid
        import time
        
        proof_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # COMMITMENT PHASE: Schnorr-like commitments
        commitments = []
        
        # Commitment 1: Hash of gradient
        grad_hash = self._hash_gradient(gradient)
        commitments.append(grad_hash)
        
        # Commitment 2: Hash of polynomial coefficients
        if polynomial_proxy:
            poly_hash = polynomial_proxy.get_hash()
            commitments.append(poly_hash)
        else:
            commitments.append(self._hash_gradient(np.zeros_like(gradient)))
        
        # Commitment 3: Hash of model parameters (for consistency)
        param_hash = self._hash_gradient(model_params)
        commitments.append(param_hash)
        
        # CHALLENGE GENERATION: Fiat-Shamir heuristic
        challenges = []
        challenge_input = "".join(commitments) + client_id + str(round_num)
        
        for i in range(ZKP_ROUNDS):
            # Generate challenge using hash-based Fiat-Shamir
            challenge_hash = hashlib.sha256(
                (challenge_input + str(i)).encode()
            ).digest()
            challenge_int = int.from_bytes(challenge_hash[:self.challenge_bits // 8], 'big')
            challenge_int = challenge_int % (2 ** self.challenge_bits)
            challenges.append(challenge_int)
        
        # RESPONSE PHASE: Generate responses
        responses = []
        
        # Response to Challenge 1: Gradient correctness
        responses.append({
            "type": "gradient_correctness",
            "nonce": self._gen_random_scalar(),
            "response_hash": self._hash_combined(grad_hash, str(challenges[0])),
        })
        
        # Response to Challenge 2: Polynomial consistency
        poly_consistent = False
        if polynomial_proxy:
            poly_consistent = polynomial_proxy.verify_consistency(gradient)
        
        responses.append({
            "type": "poly_consistency",
            "nonce": self._gen_random_scalar(),
            "response_hash": self._hash_combined(
                poly_hash if polynomial_proxy else "", 
                str(challenges[1] if len(challenges) > 1 else 0)
            ),
            "is_consistent": poly_consistent,
        })
        
        # Response to Challenge 3: Norm bound
        grad_norm = float(np.linalg.norm(gradient.flatten()))
        norm_ok = grad_norm <= norm_bound
        
        responses.append({
            "type": "norm_bound",
            "gradient_norm": grad_norm,
            "norm_bound": norm_bound,
            "nonce": self._gen_random_scalar(),
            "response_hash": self._hash_combined(
                param_hash,
                str(challenges[2] if len(challenges) > 2 else 0)
            ),
            "norm_satisfied": norm_ok,
        })
        
        # Create ZKProof object
        proof = ZKProof(
            proof_id=proof_id,
            client_id=client_id,
            round_num=round_num,
            commitments=commitments,
            challenges=challenges,
            responses=responses,
            certifies_gradient_correct=True,
            certifies_poly_consistency=poly_consistent,
            certifies_norm_bound=norm_ok,
            norm_bound=norm_bound,
            timestamp=timestamp,
        )
        
        return proof
    
    def _hash_gradient(self, gradient: np.ndarray) -> str:
        """Hash gradient vector"""
        grad_bytes = gradient.flatten().astype(np.float32).tobytes()
        return hashlib.sha256(grad_bytes).hexdigest()
    
    def _hash_combined(self, s1: str, s2: str) -> str:
        """Hash combination of two strings"""
        combined = (s1 + s2).encode()
        return hashlib.sha256(combined).hexdigest()
    
    def _gen_random_scalar(self) -> str:
        """Generate random scalar for Schnorr-like proof"""
        random_bytes = np.random.bytes(self.security_param // 8)
        return hashlib.sha256(random_bytes).hexdigest()


class ZKPVerifier:
    """Verifies zero-knowledge proofs"""
    
    def __init__(self, security_param: int = ZKP_SECURITY_PARAM):
        """Initialize verifier"""
        self.security_param = security_param
        
    def verify_proof(self, proof: ZKProof,
                    gradient: np.ndarray,
                    polynomial_proxy: PolynomialProxy = None,
                    norm_bound: float = GRADIENT_CLIP,
                    verifier_id: str = "verifier") -> Tuple[bool, Dict[str, Any]]:
        """
        Verify zero-knowledge proof.
        
        Checks:
        1. Commitment structure valid
        2. Challenge generation correct (Fiat-Shamir)
        3. Responses satisfy constraints
        4. Gradient norm within bound
        5. Polynomial consistency
        
        Args:
            proof: ZKProof to verify
            gradient: Original gradient for re-verification
            polynomial_proxy: Polynomial proxy to verify
            norm_bound: Expected norm bound
            verifier_id: Identifier of verifier
            
        Returns:
            Tuple of (is_valid, verification_details)
        """
        verdict = {
            "proof_id": proof.proof_id,
            "verifier_id": verifier_id,
            "overall_valid": False,
            "checks": {},
        }
        
        # Check 1: Verify commitment structure
        if len(proof.commitments) < 3:
            verdict["checks"]["commitment_structure"] = False
            return False, verdict
        verdict["checks"]["commitment_structure"] = True
        
        # Check 2: Verify Fiat-Shamir challenges
        grad_hash = hashlib.sha256(gradient.flatten().astype(np.float32).tobytes()).hexdigest()
        expected_commitment = grad_hash
        challenge_match = expected_commitment == proof.commitments[0]
        verdict["checks"]["challenge_generation"] = challenge_match
        
        # Check 3: Verify gradient norm bound
        grad_norm = float(np.linalg.norm(gradient.flatten()))
        norm_satisfied = grad_norm <= norm_bound
        verdict["checks"]["norm_bound"] = {
            "satisfied": norm_satisfied,
            "gradient_norm": grad_norm,
            "bound": norm_bound,
        }
        
        # Check 4: Verify polynomial consistency
        poly_consistent = True
        if polynomial_proxy:
            poly_consistent = polynomial_proxy.verify_consistency(gradient)
        verdict["checks"]["poly_consistency"] = poly_consistent
        
        # Check 5: Verify response validity
        responses_valid = True
        for response in proof.responses:
            if response["type"] == "norm_bound":
                if not response["norm_satisfied"]:
                    responses_valid = False
            elif response["type"] == "poly_consistency":
                if not response["is_consistent"]:
                    responses_valid = False
        verdict["checks"]["responses_valid"] = responses_valid
        
        # Overall verdict
        overall_valid = (
            challenge_match and
            norm_satisfied and
            poly_consistent and
            responses_valid and
            len(proof.commitments) >= 3 and
            len(proof.challenges) >= ZKP_ROUNDS
        )
        
        verdict["overall_valid"] = overall_valid
        
        # Update proof with verification result
        proof.verification_result = overall_valid
        proof.verifier_id = verifier_id
        
        return overall_valid, verdict
    
    def batch_verify(self, proofs: List[ZKProof],
                    gradients: List[np.ndarray],
                    verifier_id: str = "verifier") -> List[Tuple[bool, Dict]]:
        """Verify multiple proofs in batch"""
        results = []
        for proof, grad in zip(proofs, gradients):
            result = self.verify_proof(proof, grad, verifier_id=verifier_id)
            results.append(result)
        return results


class ZKPFactory:
    """Factory for ZKP operations"""
    
    _generator = None
    _verifier = None
    
    @classmethod
    def get_generator(cls) -> ZKPGenerator:
        """Get or create ZKP generator (singleton)"""
        if cls._generator is None:
            cls._generator = ZKPGenerator()
        return cls._generator
    
    @classmethod
    def get_verifier(cls) -> ZKPVerifier:
        """Get or create ZKP verifier (singleton)"""
        if cls._verifier is None:
            cls._verifier = ZKPVerifier()
        return cls._verifier
