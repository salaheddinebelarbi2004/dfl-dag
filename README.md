# Sharded Federated Learning with Verifiable Updates and a Model-Evolution DAG

A three-layer federated learning system for SMS spam detection that combines
**horizontal sharding**, **communication-efficient gradient transmission**, a
**structured zero-knowledge verification layer**, **Byzantine-resilient aggregation**,
and a **DAG of model evolution**.

The system targets the central cost problem in federated learning: a naive round moves
the full model update from every client, so communication dominates as the client
population grows. This implementation attacks that cost directly — it measured an **85%
reduction in transmitted bytes for 0.75 percentage points of accuracy** while keeping
cryptographic-style verification, reputation tracking, and a tamper-evident lineage of
every model state.

> **Scope.** This is a research prototype. The verification layer is a structurally
> faithful Groth16 *scaffold*, not a production proving system. See
> [Limitations](#limitations--scope) — this section is load-bearing, not a disclaimer.

---

## Table of contents

- [Contributions](#contributions)
- [Architecture](#architecture)
- [The 10-step round](#the-10-step-round)
- [Communication enhancement](#communication-enhancement)
- [Verification layer](#verification-layer)
- [Byzantine-resilient aggregation](#byzantine-resilient-aggregation)
- [Model-evolution DAG](#model-evolution-dag)
- [Shard chief election](#shard-chief-election)
- [P2P and containerised deployment](#p2p-and-containerised-deployment)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Testing](#testing)
- [Limitations & scope](#limitations--scope)

---

## Contributions

| # | Contribution | Where |
|---|--------------|-------|
| 1 | **Three-layer sharded FL** (client → shard → global) with log-scale coordination instead of all-to-all | `shard_chief.py`, `global_aggregator.py` |
| 2 | **Compression with error feedback** — top-k / quantization plus residual re-injection, so sparsity is not repeatedly discarded | `gradient_compression.py` |
| 3 | **Structured verification interface** — R1CS statement construction, fixed-point encoding, witness/statement separation, pluggable prover backend | `zkp_proofs.py` |
| 4 | **Polynomial gradient proxy** — a low-dimensional summary of an update, reducing what the verifier must inspect | `polynomial_proxy.py` |
| 5 | **Multi-stage Byzantine defence** — ZKP filter → reputation floor → cluster consensus → cosine outlier detection → label-skew detection → reputation weighting | `robust_aggregation.py` |
| 6 | **Model-evolution DAG** — lineage, sibling agreement, and a per-node trust score; an update graph, deliberately *not* a transactional ledger | `dag_system.py` |
| 7 | **Score-based shard chief election** with top-K selection and uniform random rotation, preventing a fixed leader | `shard_chief.py` |
| 8 | **P2P + Docker execution** — 12 isolated client containers exchanging updates over a peer network | `deploy/` |

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  GLOBAL LAYER                                    global_aggregator.py
│    Wt+1 = Aggregate(Ws1 … Wsk)                       dag_system.py
│    · cross-shard aggregation (mean / weighted / median)│
│    · model history + DAG versioning                   │
└───────────────────────────────────────────────────────────┘
        ↓                        ↓                        ↓
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  SHARD 0          │   │  SHARD 1          │   │  SHARD 2          │
│  chief: client_1  │   │  chief: client_6  │   │  chief: client_9  │
│                   │   │                   │   │                   │
│  verify proof     │   │  verify proof     │   │  verify proof     │
│  decompress       │   │  decompress       │   │  decompress       │
│  robust aggregate │   │  robust aggregate │   │  robust aggregate │
└───────────────────┘   └───────────────────┘   └───────────────────┘
     ↓    ↓    ↓    ↓        ↓    ↓    ↓    ↓        ↓    ↓    ↓    ↓
  [C][C][C][C]           [C][C][C][C]           [C][C][C][C]      client.py
   local SGD              local SGD              local SGD
   compress               compress               compress
   proxy + proof           proxy + proof          proxy + proof
```

**Why three layers.** With *N* clients, single-layer FedAvg needs *N* update submissions per
round and gives every client an equal vote. Sharding to *S* shards reduces per-participant
load to *N/S*, confines verification to a bounded set of peers, and lets the global layer
aggregate *S* summaries instead of *N* full updates. Aggregation cost at the top becomes
independent of client count.

---

## The 10-step round

| # | Layer  | Operation | Module |
|---|--------|------------|--------|
| 1 | Global | Broadcast `Wt` to all clients | `global_aggregator` |
| 2 | Client | Local training `Wi = Wt − η∇L(Di, Wt)` | `client` |
| 3 | Client | Form update `ΔWi = Wi − Wt`, clip to `‖ΔW‖ ≤ B` | `client` |
| 4 | Client | Compress `ΔWi` with error feedback; update residual | `gradient_compression` |
| 5 | Client | Build polynomial proxy `Pi(x) ≈ f(ΔWi, Wt)` | `polynomial_proxy` |
| 6 | Client | Build R1CS statement + witness, generate proof | `zkp_proofs` |
| 7 | Shard  | Verify proof, decompress with residual | `shard_chief` |
| 8 | Shard  | Robust aggregation (6 stages) | `robust_aggregation` |
| 9 | Shard  | `Ws = RobustAggregate(...)` | `shard_chief` |
| 10| Global | `Wt+1 = Aggregate(Ws)`, record node in DAG, held-out eval | `global_aggregator`, `dag_system` |

---

## Communication enhancement

The core contribution. A dense update of a 384→256→2 model is ~99k floats; transmitting
it from every client every round dominates total cost. This module reduces the payload
along two independent axes, with **error feedback** so that neither axis is lossy in a
way that compounds across rounds.

**Axis 1 — sparsity (`topk`).** Keep only the `ρ` fraction of coordinates with the
largest magnitude, plus their indices.

**Axis 2 — precision (`quantize`).** Send retained values at `b`-bit fixed point
(default 8) instead of float32.

**Error feedback (the part that makes it work).** Naive top-k silently discards the
dropped mass every round, and the loss accumulates. Procedure 1 instead keeps a residual
and re-injects it:

```
compensated = ΔW + r_t            # 1. accumulate what was not sent
payload     = compress(compensated)
r_{t+1}     = compensated with retained coordinates zeroed
```

The retained information is never lost — it is deferred to the next round. The payload
carries `(values, indices, meta)` and the shard reconstructs the update exactly through
`decompress_update`.

**Measurement is real, not theoretical.** `measure_serialized_size` reports the actual
pickled byte count of the transmitted object, so the compression ratio in
[Results](#results) is measured on the wire format rather than inferred from element
counts.

---

## Verification layer

`zkp_proofs.py` implements a **Groth16-structured** verification interface. The
separation of concerns is real and is the point of the module: the client, shard chief,
and global layer never touch curve arithmetic — they construct a *statement*, attach a
*witness*, and delegate to a prover/verifier pair that can be swapped for a real backend.

**Components.**

- `FixedPointEncoder` — deterministic float ⇄ integer field encoding (`scale_bits = 16`),
  the encoding a proving system requires for float data.
- `R1CSCircuitBuilder` — builds the public statement and private witness for the update
  relation: `h_w_old`, `h_w_new` as SHA-256 commitments over the encoded parameter
  vectors, plus `public_inputs = [1, hash(h_w_old) mod 2⁶⁴, hash(h_w_new) mod 2⁶⁴]`.
- `Groth16Statement` — the public side (what a verifier may know).
- `VFLWitness` — the private side (`w_old`, `w_new`, `gradient`, `local_tokens`).
- `Groth16Prover` / `Groth16Verifier` — pluggable backend behind `ZKPFactory`.
- `batch_verify` — amortised verification across a shard.

**What the verifier actually checks today.** Rather than a pairing equation, the verifier
enforces the checks that are meaningful for the learning protocol and reports each
individually:

| Check | Meaning |
|-------|---------|
| `proof_shape` | all three curve elements present |
| `statement_binding` | proof's `statement_hash` matches the recomputed statement |
| `norm_bound` | `‖ΔW‖ ≤ GRADIENT_CLIP` — the malicious-gradient bound |
| `poly_consistency` | the proxy still describes the submitted gradient |
| `public_inputs` | well-formed field elements, ≥ 3 |

`verify_proof` returns `(bool, verdict)` where `verdict["checks"]` exposes each
component, so a shard chief can log *why* an update was down-weighted rather than
discarding it opaquely.

**Norm bounding is the load-bearing security property.** A client cannot evade the
gradient-clipping bound by lying about its update, because the bound is recomputed
server-side from the submitted gradient and recorded in the verdict. This is the part
that is genuinely enforced rather than simulated.

---

## Byzantine-resilient aggregation

`RobustAggregator` runs six stages. Each can be enabled independently; `ROBUST_HYBRID`
runs all of them.

1. **ZKP / verification filter** — drop updates whose proof or norm bound failed.
2. **Reputation floor** — drop clients below `MIN_REPUTATION_THRESHOLD`. Reputation
   updates on acceptance and rejection, and is **soft**: a rejected update is down-weighted
   (`SOFT_OUTLIER_WEIGHT = 0.5`) rather than expelled, so a single bad round is not fatal.
3. **Cluster consensus** — K-means (`CLUSTER_K = 3`, or hierarchical) over update
   directions; the majority cluster wins.
4. **Cosine-similarity outlier detection** — flag clients whose update direction
   deviates beyond `OUTLIER_THRESHOLD` standard deviations.
5. **Label-skew detection** — inspect each client's `.npz` label distribution; a client
   with >`LABEL_SKEW_THRESHOLD` (0.95) of one class is treated as poisoned. This catches
   the *label-flipping* adversary that magnitude-based filters miss, because a flipped
   update is small in norm but wrong in direction.
6. **Reputation-weighted average** — higher-trust clients carry proportionally more
   weight.

The design intent is that stages 4 and 5 catch *different* adversaries: 4 catches
high-magnitude poisoning, 5 catches subtle direction/label poisoning.

---

## Model-evolution DAG

`dag_system.py` records every model state as a node. It is explicitly a **model-evolution
graph, not a transactional ledger** — there is no block, no ordering guarantee, and no
consensus ordering. Its job is provenance and tamper evidence.

**Node lifecycle.**

```
PENDING ──verify ok──▶ VERIFIED ──aggregated──▶ AGGREGATED ──▶ FINALIZED
   │                                                        │
   └────────── verify failed / rejected ──────────────────▶ REJECTED
```

Each `DAGNode` carries `client_id`, `round_num`, `shard_id`, `model_hash`,
`gradient_hash`, `parent_id`, `zkp_verified`, `commitment_verified`,
`reputation_score`, and a derived `trust_score`.

**Sibling agreement.** Updates trained from the same parent are *siblings*. The graph
computes pairwise cosine agreement across siblings, which surfaces clients that
disagreed with their peers at a given point in the lineage — the earliest signal of a
divergent or poisoned participant.

**Trust score** (`core/dag_system.py:405`):

```
trust = 0.4 · agreement  +  0.3 · structure  +  0.3 · verification
```

- *verification* — ZKP flag and commitment flag, 0…1
- *structure* — 0.5 × parent trust + 0.5 × sibling-concentration factor (large sibling
  sets are penalised, bounding fan-out)
- *agreement* — 1 − deviation from sibling consensus

**Sharding.** `ShardedDAG` maintains one `DAGGraph` per shard, so lineage is auditable
per shard and the global layer records its own model hashes. `get_finalized_chain()`
returns the accepted lineage; `get_path_to_root()` walks ancestry for any node.

---

## Shard chief election

Each shard needs one coordinator to verify and aggregate. `ShardChiefElector` scores
candidates and avoids a permanent leader:

```
Sj = α·Uj + β·Bj + γ·Cj          α=0.4 uptime, β=0.3 bandwidth, γ=0.3 compute
```

Take the top-`K` (`TOP_K_CANDIDATES = 5`) candidates, then choose **uniformly at random**
among them. A high-scoring client is therefore usually elected but never *always* — which
removes the single point of failure and the targeted-DoS surface a fixed chief creates.
The elected `ShardChief` then runs verification → decompression → robust aggregation for
its shard.

---

## P2P and containerised deployment

`deploy/` runs the same protocol across real processes instead of in-process calls.

- **`docker-compose.yml`** — 12 client containers on a fixed bridge subnet
  (`172.28.0.0/16`), one per client, each receiving its own `.npz` shard as a read-only
  bind mount and sharing `results/` for output.
- **`docker_entrypoint.py`** — the per-container client. Entirely environment-driven
  (`CLIENT_ID`, `NUM_CLIENTS`, `NUM_SHARDS`, `NUM_ROUNDS`, `LOCAL_EPOCHS`, `P2P_PORT`,
  `RUNNER_MODE`), so the same image serves every role. Executes the round in five phases:
  local training → publish update to shard chief → shard aggregation → global aggregation
  → persist results. A `RUNNER_MODE=contribution` branch boots the full
  `ShardedFederatedLearningSystem` in-process instead, which is how the ρ-sweep results
  were produced.
- **`p2p_node.py`** — peer node over `p2pnetwork`, with a `FlowrCore` / `FlowrAdapter`
  split that keeps task orchestration (create, track, retrieve) free of networking. Tasks:
  `sum`, `ping`, `train_tree`, `predict_tree`.
- **`run_with_shards.py`** — standalone 12-client/3-shard trainer with a framework-agnostic
  model wrapper; also the source of the `RealModelWrapper` used by the container path.
- **`nginx.conf`**, **`docker_deploy.bat`**, **`start_nodes.{bat,ps1}`** — reverse proxy and
  operator helpers.

```bash
cd deploy
docker compose build && docker compose up -d      # 12 clients
docker compose logs -f client-0
```

---

## Results

ρ = fraction of gradient entries transmitted. 12 clients, 3 shards, 5 rounds, held-out
`global_eval` split. Raw runs in `results/contribution_results/`, figures in
`results/comm_all_eval/`.

| ρ | Accuracy | F1 | Transmitted | Reduction |
|---|----------|-----|-------------|-----------|
| **1.0** (baseline) | **82.60%** | 77.60% | 9.51 MB | — |
| 0.5 | 82.04% | 77.76% | 7.14 MB | −25% |
| 0.2 | 82.31% | 77.12% | 2.86 MB | −70% |
| **0.1** | **81.85%** | 76.61% | **1.43 MB** | **−85%** |
| 0.1 + 8-bit quant | 81.92% | 76.65% | 0.60 MB | −94% |

**Headline: 85% fewer bytes for 0.75 pp accuracy.** The accuracy curve is essentially
flat across the whole sweep — well inside run-to-run variance — while transmitted volume
falls by more than an order of magnitude between ρ=1.0 and ρ=0.1 with quantization.

### Byzantine resilience

Measured on `malisious/` — 12 clients, 2 label-flipped (`client_4`, `client_7`), i.e. a
16.7% adversary, which is below the 33% that median-based aggregation tolerates.

| Configuration | Accuracy | F1 | ROC-AUC |
|---------------|----------|-----|---------|
| Clean data | 97.48% | 97.03% | 0.996 |
| 2 poisoned clients | 88.34% | 86.74% | 0.927 |

The defence **identifies both adversaries exactly**. Final reputation scores:

```
client_4   0.0000  <-- poisoned
client_7   0.0000  <-- poisoned
all 10 clean clients   1.0000
```

Clean separation, no false positives: the 10 benign clients retain full reputation while
both adversaries are driven to zero and excluded from the weighted average. The ~9 pp
accuracy gap versus the clean arm is the residual cost of the attack, not a failure to
detect it.

> These Byzantine numbers come from the evaluation path in `experiments/`
> (`eval_clean.py` / `best_results_ever.py`) and are **not** directly comparable to the
> ρ-sweep above, which is produced by the full `ShardedFederatedLearningSystem` on a
> different model and split. Do not quote them as one experiment.

**Reproduce:**

```bash
python experiments/generate_comm_all_eval.py       # ρ-sweep figures + CSVs
python experiments/plot_message_composition.py    # payload composition
```

---

## Repository layout

```
core/            11 modules — the system itself (flat sibling imports, no package)
  federated_learning_system.py   orchestrator + CLI, drives the 10-step round
  client.py                      local SGD, clipping, proxy, proof, compression
  shard_chief.py                 election, verification, decompression, aggregation
  global_aggregator.py           global layer, GlobalCoordinator, model history
  robust_aggregation.py          6-stage Byzantine defence, reputation, clustering
  dag_system.py                  model-evolution DAG, sibling agreement, trust
  zkp_proofs.py                  Groth16-structured verification layer
  polynomial_proxy.py            degree-3 gradient proxy
  gradient_compression.py        top-k / quantization / error feedback
  model_registry.py              .pt / .safetensors / .npy / .npz loading
  federated_learning_config.py   every hyperparameter

deploy/          P2P + Docker deployment (see above)
scripts/         dataset generation: load_filtered_data → generate_embeddings →
                 make_malicious_clients
experiments/     17 evaluation, threshold-sweep and plotting scripts
tests/           test_system.py (8 integration cases), test_readme5_alignment.py
docs/            mast.pdf (compiled thesis) + figures/
real_federated_data/   12 client shards (.npz) — canonical dataset
malisious/            poisoned copy; client_4 and client_7 label-flipped
results/              contribution_results/, comm_all_eval/, clean + malicious baselines
```

Modules in `core/` use flat sibling imports, so scripts are run **from the repository
root** (`python experiments/eval_clean.py`); module-level path constants are written
relative to the root.

---

## Usage

```bash
pip install -r requirements.txt

python main.py              # full 10-step run: 12 clients, 3 shards, 5 rounds
python -m pytest tests/     # test suite
```

`main.py` puts `core/` on `sys.path` and forces UTF-8 stdout (the modules print
non-ASCII, which raises `UnicodeEncodeError` on the default Windows `cp1252` console).

**Model.** 384 → 256 → 2 MLP over MiniLM sentence embeddings, ~99k parameters.
384 = embedding dimension, 2 = binary spam/ham.

---

## Configuration

All hyperparameters live in `core/federated_learning_config.py`. The compression
switches are read from the environment so a sweep can be driven without code edits:

| Variable | Default | Values |
|----------|---------|--------|
| `COMPRESSION_METHOD` | `topk` | `topk` / `quantize` / `topk_quantize` |
| `COMPRESSION_RATIO` | `1.0` | ρ — fraction of entries sent; `1.0` disables |
| `QUANTIZE_ENABLED` | `0` | enable fixed-point quantization |
| `QUANTIZE_BITS` | `8` | quantization width |
| `ERROR_FEEDBACK` | `0` | accumulate and re-send residual each round |

```bash
COMPRESSION_RATIO=0.1 python main.py
```

Fixed baseline: `NUM_CLIENTS=12`, `NUM_SHARDS=3`, `NUM_ROUNDS=5`, `LOCAL_EPOCHS=5`,
`LEARNING_RATE=0.05`, `RANDOM_SEED=42`, `GRADIENT_CLIP=1.0`, `POLY_DEGREE=3`,
`SHARD_CHIEF_SCORE_{ALPHA,BETA,GAMMA} = 0.4/0.3/0.3`, `CLUSTER_K=3`,
`MIN_REPUTATION_THRESHOLD=0.5`, `OUTLIER_THRESHOLD=2.0`.

---

## Dataset

`real_federated_data/client_{0..11}.npz` — 12 shards of 384-dim MiniLM sentence
embeddings with deliberate non-IID label skew, from an Arabic/English SMS spam corpus.
Source: `Augmented_SpamHamTweets.xlsx` → `filtered_training_data.csv`.

`malisious/` is the same data with **client_4 and client_7 label-flipped**, used for the
Byzantine experiments. (The directory name is misspelled in the original layout; it is
referenced by code and by saved result JSONs, so it is preserved deliberately.)

```bash
python scripts/load_filtered_data.py      # .xlsx → filtered_training_data.csv
python scripts/generate_embeddings.py     # → real_federated_data/client_*.npz
python scripts/make_malicious_clients.py  # → malisious/ (poisons client_4, client_7)
```

`sentence-transformers` is required for this. `nlp_embeddings.py` falls back to *random*
embeddings when it is missing — it still runs, but produces meaningless features.
Install the dependency before regenerating data.

---

## Testing

```bash
python -m pytest tests/            # 9 tests
python tests/test_system.py        # 8-case integration suite
```

`test_system.py` covers the polynomial proxy, the verification layer, client local
training, robust aggregation, shard chief election, DAG operations, the global
aggregator, and a full mini training round. `test_readme5_alignment.py` pins the
baseline hyperparameter set and asserts that every round records a held-out
`global_eval` metric.

---

## Limitations & scope

Stated plainly, because the distinction matters for anyone reusing this code.

**The verification layer is not real Groth16.** It has the right *shape* — statement,
witness, fixed-point encoding, R1CS builder, pluggable prover — but no curve, no group,
and no pairing. `_derive_curve_element()` returns SHA-256 digests, and the
`pairing_identity` check compares two unrelated hashes, which is why `verify_proof`
deliberately does not require it. **Do not describe its output as a zero-knowledge
proof.** Swapping in a real backend (`py-snark`, `snarkjs`) is the intended path and
requires no changes at the call sites.

**The polynomial proxy is self-consistent, not sound.** `polynomial_proxy.py` fits a
low-degree polynomial over gradient statistics and then verifies that same function
against itself, so `verify_consistency` returns true by construction. It reduces what the
verifier inspects; it does not prove anything about the underlying update.

**There is no differential privacy.** No noise injection, no ε/δ accounting, no privacy
budget. `GRADIENT_CLIP` is norm bounding, which limits influence but leaks nothing about
the privacy guarantee formally. A DP mechanism (noise + accountant) is not implemented.

**Updates are unauthenticated.** `UPDATE_SIGNATURE_SCHEME = "ed25519"` is declared in the
config and never read. Nothing binds an update to a client identity.

**Insecure deserialization.** `deploy/docker_entrypoint.py` and `deploy/p2p_node.py` call
`pickle.loads` on bytes received from peers, and `model_registry.py` calls `torch.load`
without `weights_only=True`. Each is arbitrary code execution from an untrusted input.
Not safe to expose to a network you do not control. Fix: safetensors or `np.load` with an
allowlist, plus signed payloads.

**Runs are not bit-reproducible** despite `RANDOM_SEED = 42`: `client.py` reseeds the
global NumPy RNG per client, shard-chief election uses unseeded `random.choice`, and DAG
node IDs use `uuid.uuid4()`. Variation between runs is small but non-zero, and is
larger than the ρ-to-ρ accuracy differences reported above.

**Scale.** 12 clients / 3 shards is a research-scale demonstration, not a
production deployment.

---

## License

Provided as-is for research and educational purposes.
