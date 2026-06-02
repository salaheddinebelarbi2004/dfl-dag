# Groth16 VFL Refactor Blueprint

This document maps the current prototype ZKP flow onto a Groth16-based Verifiable Federated Learning design for edge fine-tuning of an ArabBERT-style transformer model.

## Current State

The existing implementation in `zkp_proofs.py` is a prototype:
- Hash commitments
- Fiat-Shamir-style challenges
- Response objects with consistency checks
- Verification performed with direct access to the raw gradient

That is useful as a correctness scaffold, but it is not a real zk-SNARK and does not satisfy the Groth16 pairing-check architecture.

## Target Architecture

Replace the current proof object with a Groth16 proof over an R1CS circuit that proves:
- the local fine-tuning step was executed correctly,
- the public hashes match the private witness,
- the update satisfies the expected arithmetic constraints,
- the revealed statement is bound to the hidden witness without exposing raw weights, gradients, or text tokens.

The proof should be exactly three curve elements:

```text
pi = { [A]G1, [B]G2, [C]G1 }
```

The verifier must check a single pairing equation of the Groth16 form:

```text
e([A]G1, [B]G2) = e(g1^alpha, g2^beta) * e([I]G1, g2^gamma) * e([C]G1, g2^delta)
```

## File-by-File Refactor Plan

### 1. `zkp_proofs.py`

Replace the prototype generator/verifier with a Groth16 API wrapper.

Recommended structure:

```python
@dataclass
class Groth16Proof:
    A: bytes  # or a curve point object in G1
    B: bytes  # curve point in G2
    C: bytes  # curve point in G1
    public_inputs: List[int]
    proof_id: str
    client_id: str
    round_num: int
    verifier_id: Optional[str] = None
    verification_result: bool = False
```

Add:
- `Groth16Prover`
- `Groth16Verifier`
- `R1CSCircuitBuilder`
- `FixedPointEncoder`

Keep the existing `ZKPFactory` names for backward compatibility, but make them return Groth16 implementations behind the same interface.

Suggested compatibility approach:

```python
class ZKPFactory:
    @classmethod
    def get_generator(cls):
        return Groth16Prover.get_instance()

    @classmethod
    def get_verifier(cls):
        return Groth16Verifier.get_instance()
```

### 2. `client.py`

The client must stop generating proof hashes over raw gradients and instead produce a circuit witness.

Required change in `create_update()`:
- keep local training unchanged,
- encode the witness vector,
- hash the old and new classification-head weights,
- package public inputs and private witness separately,
- call the Groth16 prover.

Add a helper layer:

```python
def build_vfl_witness(self, initial_params, updated_params, gradient, local_tokens):
    ...
```

### 3. `dag_system.py`

No structural rewrite is required.

Keep DAG nodes, sibling agreement, trust scoring, and finalize logic intact, but switch the verification flag to use Groth16 proof verification results instead of the current prototype proof checks.

### 4. `shard_chief.py`

The shard chief should verify Groth16 proofs using only public inputs and the proof tuple `(A, B, C)`.

Do not pass raw gradients into verification unless you are explicitly in a debug mode.

### 5. `global_aggregator.py`

No major change is required unless you want to bind global aggregation to the VFL proof state as well.

## R1CS Witness Layout

Use the witness order you specified:

```text
s = [1, H(W_old), H(W_new), vec(W_old)^T, vec(W_new)^T, vec(∇L)^T, vec(D_local)^T]^T
```

### Public Inputs
- `1`
- `H(W_old)`
- `H(W_new)`

### Private Witness
- `vec(W_old)`
- `vec(W_new)`
- `vec(∇L)`
- `vec(D_local)`

If you want Groth16 to remain practical, avoid putting the entire ArabBERT model into the circuit. Restrict the proof to the fine-tuned classification head or a low-dimensional adapter layer.

## Fixed-Point Field Encoding

Groth16 circuits operate over a prime field `F_p`, not floating-point numbers.

Add a fixed-point helper:

```python
class FixedPointEncoder:
    def __init__(self, scale_bits: int = 16):
        self.scale = 1 << scale_bits

    def encode(self, x: float) -> int:
        return int(round(x * self.scale))

    def decode(self, x: int) -> float:
        return x / self.scale
```

Use it for:
- learning rate `η`,
- gradients,
- embeddings,
- intermediate softmax and loss terms.

## Circuit Constraints

The circuit should enforce the training step:

```text
W_new = W_old - floor(η * ∇L * 2^-b)
```

Implement this as field arithmetic with fixed-point scaling.

The R1CS rows should conceptually correspond to:
- Matrix A: coefficient terms for `η * ∇L`
- Matrix B: pass-through scalar `1`
- Matrix C: subtraction relation `W_old - W_new`

This is a modeling choice for circuit compilation, not a literal matrix API requirement at runtime.

## In-Circuit Hash Binding

Use Poseidon as the in-circuit hash because it is SNARK-friendly.

Add constraints that prove:

```text
Poseidon(vec(W_new_raw)) == H(W_new_public)
```

Do the same for the old weights if needed.

Do not use SHA-256 inside the circuit unless you are willing to pay a very large constraint cost.

## Softmax / Cross-Entropy Approximation

Replace transcendental functions with a 3rd-order Taylor approximation:

```text
exp(x) ≈ 1 + x + x^2/2 + x^3/6
```

Flatten the multiplications into separate quadratic constraints:

```text
x * x = v1
v1 * x = v2
(1 + x + v1/2 + v2/6) * 1 = v_out
```

If you include the loss in-circuit, also flatten the log term or replace it with a bounded surrogate loss. For Groth16, the most practical route is to prove the parameter-update relation and a norm-bound, not full transformer inference.

## Recommended Scope for ArabBERT VFL

To keep the circuit tractable:
- prove only the classification head update, or
- prove an adapter/LoRA layer update, or
- prove a compressed projection of the transformer update.

Do not attempt to place full ArabBERT forward propagation into Groth16 unless you are prepared for a very large circuit and heavy proving time.

## Backward-Compatible Migration Strategy

Use a two-phase rollout:

### Phase 1: Interface Preservation
- Keep `ZKPFactory`, `ClientUpdate`, and shard verification call sites.
- Replace internal proof internals with Groth16.
- Keep existing DAG and aggregation logic.

### Phase 2: Circuit Upgrade
- Add circuit builder and trusted setup artifacts.
- Add fixed-point quantization.
- Bind text tokens and weights to public hashes.
- Verify only the Groth16 proof tuple at shard level.

## Minimal Implementation Checklist

1. Add `Groth16Proof` dataclass.
2. Add `FixedPointEncoder`.
3. Add `R1CSCircuitBuilder` for the VFL statement.
4. Replace prototype `ZKPGenerator` with a Groth16 prover wrapper.
5. Replace prototype `ZKPVerifier` with a pairing-check verifier wrapper.
6. Update client update packaging to carry public inputs and witness serialization.
7. Keep DAG and shard logic unchanged, but consume Groth16 verification results.

## Practical Recommendation

If you want this to be usable soon, implement Groth16 for:
- update correctness,
- hash binding,
- norm bound.

Leave full text-model execution and full cross-entropy inside the circuit for a later iteration.

That gives you a real VFL trust statement while keeping proving costs manageable.