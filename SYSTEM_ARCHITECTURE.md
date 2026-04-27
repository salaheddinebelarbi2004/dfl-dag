# Sharded Federated Learning System with Polynomial Proxy + ZKP + DAG

Complete, production-grade prototype implementation of a horizontally-scalable federated learning system with advanced cryptographic and aggregation mechanisms.

## System Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│              GLOBAL LAYER                                   │
│   Global Aggregator: Wt+1 = Aggregate(Ws1, ..., Wsk)       │
│   - Weighted shard aggregation                              │
│   - Model evolution tracking (DAG)                          │
└─────────────────────────────────────────────────────────────┘
              ↓                    ↓                    ↓
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   SHARD 1   Layer    │ │   SHARD 2   Layer    │ │   SHARD 3   Layer    │
│ (Shard Chief)        │ │ (Shard Chief)        │ │ (Shard Chief)        │
│ - ZKP verification   │ │ - ZKP verification   │ │ - ZKP verification   │
│ - Robust aggregation │ │ - Robust aggregation │ │ - Robust aggregation │
│ - Ws = RobustAgg(..) │ │ - Ws = RobustAgg(..) │ │ - Ws = RobustAgg(..) │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
    ↓               ↓      ↓               ↓      ↓               ↓
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│  CLIENT 1  │ │  CLIENT 2  │ │  CLIENT 3  │ │  CLIENT 4  │ │  CLIENT 5  │
│ (Shard 1)  │ │ (Shard 1)  │ │ (Shard 2)  │ │ (Shard 2)  │ │ (Shard 3)  │
└────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘
```

### Three-Layer Operations

#### CLIENT LAYER
- **Local Dataset**: Each client Di holds private training data
- **Local Training**: Wi = Wt - η∇L(Di, Wt)
- **Update Computation**: ∆Wi = Wi - Wt
- **Polynomial Proxy**: Pi(x) ≈ f(∆Wi, Wt) - lightweight update encoding
- **Cryptographic Commitment**: Ci = Commit(∆Wi)
- **ZKP Generation**: πi proving correct computation, consistency, and norm bounds

#### SHARD LAYER
- **Shard Chief Election**: Score-based (uptime, bandwidth, compute) + rotation
- **ZKP Verification**: Verify πi proofs before aggregation
- **Robust Aggregation**: Multi-stage filtering and weighting
  - ZKP-filtered updates
  - Reputation-based filtering
  - Cluster-consensus aggregation
  - Reputation-weighted averaging
- **Shard Model**: Ws = RobustAggregate(∆Wi_valid)
- **DAG Updates**: Record verified updates in shard-local DAG

#### GLOBAL LAYER
- **Model Aggregation**: Wt+1 = Aggregate(Ws1, ..., Wsk)
- **Global Learning Rate**: Step size for global update
- **Model Evolution Tracking**: Maintain global DAG for model history

## 10-Step Training Workflow

### Step 1: Global Model Broadcast
```
GlobalCoordinator → All Shards → All Clients
Broadcast Wt to initialize local training
```

### Step 2: Client Local Training
```
foreach client:
    Wi = LocalTrain(Wt, Di, η, Local_Epochs)
```

### Step 3: Update Computation
```
foreach client:
    ∆Wi = Wi - Wt
    clip to ‖∆Wi‖ ≤ B
```

### Step 4: Polynomial Proxy Construction
```
foreach client:
    Pi(x) = a0 + a1*x + ... + an*x^n
    fit from ∆Wi properties
```

### Step 5: ZKP Generation
```
foreach client:
    πi ← ZKProve(∆Wi, Pi(x), B)
    certifies: ∇ correctness, Pi consistency, norm bound
```

### Step 6: Shard Submission
```
foreach client:
    send (∆Wi, Pi, Ci, πi) → Shard Chief
```

### Step 7: Shard Chief Verification
```
foreach shard chief:
    verify πi ∀ received updates
    check Ci match ∆Wi
    verify ‖∆Wi‖ ≤ B
```

### Step 8: Robust Aggregation
```
foreach shard:
    1. ZKP-filter verified updates
    2. Reputation-filter (score ≥ threshold)
    3. Cluster-consensus aggregation
    4. Reputation-weighted average
```

### Step 9: Shard Update
```
foreach shard:
    Ws ← RobustAggregate(∆Wi_valid)
    update local DAG
```

### Step 10: Global Aggregation
```
GlobalCoordinator:
    Wt+1 ← WeightedAverage(Ws from all shards)
    broadcast Wt+1 for next round
```

## Key Components

### 1. Polynomial Proxy (`polynomial_proxy.py`)
- **Purpose**: Lightweight encoding of gradient properties
- **Representation**: Pi(x) = a0 + a1*x + ... + an*x^n
- **Evaluation Points**: Stratified sampling for fitting
- **Consistency Verification**: Check Pi matches original gradient
- **ZKP Integration**: Reduce proof complexity via algebraic verification

**Example Usage**:
```python
proxy = PolynomialFactory.create_from_update(gradient, model_params)
is_consistent = proxy.verify_consistency(gradient)
coefficients = proxy.get_coefficients_ordered()  # [a0, a1, ..., an]
```

### 2. Zero-Knowledge Proofs (`zkp_proofs.py`)
- **Scheme**: Schnorr-like proof + Fiat-Shamir heuristic
- **Proves**:
  1. **Gradient Correctness**: ∇ computed correctly from Di
  2. **Poly Consistency**: Pi(x) consistent with ∆Wi
  3. **Norm Bound**: ‖∆Wi‖ ≤ B (prevents model poisoning)
- **Structure**:
  - Commitment phase: Hash-based commitments
  - Challenge generation: Fiat-Shamir heuristic
  - Response phase: Schnorr-like responses

**Example Usage**:
```python
generator = ZKPFactory.get_generator()
proof = generator.generate_proof(gradient, model_params, client_id, round_num, poly)

verifier = ZKPFactory.get_verifier()
is_valid, details = verifier.verify_proof(proof, gradient, poly)
```

### 3. DAG System (`dag_system.py`)
- **NOT a blockchain ledger** - a model evolution graph
- **Nodes**: Validated model updates
- **Edges**: A → B means "B trained from state A"
- **Properties**:
  - Supports concurrent sibling updates
  - Sharded sub-graphs per shard
  - Only Shard-Chief-approved updates added
  - Tracks model lineage and versioning

**DAG Structure**:
```
root (initial model)
  ├── client_1_update (Round 1)
  ├── client_2_update (Round 1)  [siblings]
  │    └── client_3_update (Round 2)
  └── client_4_update (Round 1)
```

**Example Usage**:
```python
dag = ShardedDAG(num_shards=3, initial_model_hash="...")
node_id = dag.add_shard_update(shard_id, client_id, round_num, ...)
dag.verify_update(node_id, zkp_verified=True)
dag.finalize_update(node_id)
```

### 4. Robust Aggregation (`robust_aggregation.py`)
- **Multi-Stage Filtering**:
  1. ZKP verification - only verified updates
  2. Reputation filtering - minimum trustworthiness
  3. Cluster consensus - similarity-based groups
  4. Reputation weighting - higher-reputation clients weighted higher
- **Methods**: Mean, Median, Cluster-Consensus, Reputation-Weighted, Robust-Hybrid
- **Reputation Tracking**: Dynamic scores (accepted/rejected history)

**Example Usage**:
```python
aggregator = RobustAggregator()
# Single method
agg, meta = aggregator.aggregate(
    updates, verification_results, client_ids,
    method=AggregationMethod.ROBUST_HYBRID
)
# Track reputation
aggregator.register_client(client_id)
rep_score = aggregator.get_reputation(client_id)
```

### 5. Shard Chief (`shard_chief.py`)
- **Election**: Score-based (Sj = α·Uj + β·Bj + γ·Cj)
  - Uj: Uptime score
  - Bj: Bandwidth score
  - Cj: Compute capacity score
- **Top-K Selection**: Top K candidates
- **Random Rotation**: Uniform random choice from top K
- **Functions**:
  - Verify ZKP proofs
  - Execute robust aggregation
  - Maintain local DAG
  - Track statistics

**Example Usage**:
```python
elector = ShardChiefElector()
chief, election_meta = elector.elect_for_shard(
    candidates, shard_id, top_k=5
)
shard_agg, process_meta = chief.process_round(
    gradients, proofs, client_ids
)
```

### 6. Client (`client.py`)
- **Neural Network Simulator**: Enables training without DL framework
- **Local Training**: SGD-based with mini-batches
- **Update Generation**: All 6 client-side steps
- **Synthetic Data**: Non-IID distribution support

**Example Usage**:
```python
client = FederatedLearningClient(client_id="client_1", shard_id="shard_0")
client.set_local_dataset(X, y)
client.receive_global_model(global_params)
client.local_training_round(local_epochs=2)
update = client.create_update(global_params, round_num=1)
```

### 7. Global Aggregator (`global_aggregator.py`)
- **Shard Aggregation**: Weighted average of shard models
- **Global Update**: Apply learning rate to update
- **Model Tracking**: History of global models

**Example Usage**:
```python
coordinator = GlobalCoordinator(num_shards=3, initial_model)
coordinator.broadcast_global_model()
new_model, meta = coordinator.run_training_round(shard_updates)
```

### 8. Complete System (`federated_learning_system.py`)
- **Orchestration**: Manages all components
- **Training Loop**: Executes full 10-step workflow
- **Statistics**: Comprehensive metrics collection
- **DAG Integration**: Records all training steps

**Example Usage**:
```python
system = ShardedFederatedLearningSystem(
    num_clients=12,
    num_shards=3,
    num_rounds=5
)
results = system.execute_full_training()
system.save_results("training_results.json")
```

## Configuration (`federated_learning_config.py`)

All system parameters in one place:

```python
# System Parameters
NUM_CLIENTS = 12
NUM_SHARDS = 3
NUM_ROUNDS = 5

# Network Parameters
INPUT_DIM = 784  # MNIST-like
OUTPUT_DIM = 10
LEARNING_RATE = 0.01

# Polynomial Proxy
POLY_DEGREE = 3
POLY_EVAL_POINTS = 5

# ZKP
ZKP_SECURITY_PARAM = 128
ZKP_ROUNDS = 3

# Shard Chief Election
SHARD_CHIEF_SCORE_ALPHA = 0.4  # uptime weight
SHARD_CHIEF_SCORE_BETA = 0.3   # bandwidth weight
SHARD_CHIEF_SCORE_GAMMA = 0.3  # compute weight

# Robust Aggregation
CLUSTER_K = 3
OUTLIER_THRESHOLD = 2.0
MIN_REPUTATION_THRESHOLD = 0.5

# Security
GRADIENT_CLIP = 1.0
COMMITMENT_HASH_ALG = "sha256"
```

## Running the System

### 1. Install Dependencies
```bash
cd "d:\p2p network"
pip install -r requirements.txt
```

### 2. Run Complete System
```bash
python federated_learning_system.py
```

### 3. Output
- Console output showing all 10 steps per round
- `training_results.json` with complete statistics
- DAG structure per shard
- Client and Shard Chief metrics

### Example Output:
```
============================================================
SHARDED FEDERATED LEARNING SYSTEM
Clients: 12, Shards: 3, Rounds: 5
============================================================

============================================================
ROUND 1 / 5
============================================================
Step 1: Broadcasting global model...
Steps 2-6: Client local training (12 clients)...
  Client Updates: 12/12 successful
Steps 7-9: Shard verification and aggregation (3 shards)...
  shard_0: 4 updates, Verification: 100.0%
  shard_1: 4 updates, Verification: 100.0%
  shard_2: 4 updates, Verification: 100.0%
Step 10: Global aggregation...
  Global Update: Successful
  Round Time: 2.34s

...

============================================================
TRAINING COMPLETE
============================================================
Total Time: 12.45 seconds
Rounds: 5
Clients: 12
Shards: 3

Global Coordinator:
  Aggregations: 5
  Success Rate: 100.0%

Shard Chiefs:
  shard_0 (client_1): 12 verified, 0 rejected
  shard_1 (client_3): 12 verified, 0 rejected
  shard_2 (client_9): 12 verified, 0 rejected

DAG Status:
  Global Model History: 6 snapshots
  shard_0: 5 nodes, 5 finalized
  shard_1: 5 nodes, 5 finalized
  shard_2: 5 nodes, 5 finalized
============================================================
```

## Security Properties

### 1. Gradient Verification
- ZKP proves correct local training
- Commitment verification ensures update integrity
- Norm bounds prevent poisoning attacks

### 2. Client Reputation
- Track accept/reject history
- Dynamic reputation scores
- Minimum threshold for aggregation inclusion

### 3. Malicious Update Detection
- ZKP verification filters invalid proofs
- Reputation filtering removes unreliable clients
- Cluster consensus isolates outliers
- Multi-stage robust aggregation

### 4. Model Integrity
- DAG tracks all model updates
- Merkle-like hashing for verification
- Clear lineage from initial to final model

## Performance Characteristics

### Computational Complexity
- **Local Training**: O(n_samples × n_epochs × model_params)
- **ZKP Generation**: O(ZKP_ROUNDS × vector_size)
- **Shard Aggregation**: O(num_clients × model_params) + clustering
- **Global Aggregation**: O(num_shards × model_params)

### Communication
- Model broadcast: O(num_clients × model_size)
- Update submission: O(num_clients × update_size)
- Shard aggregation: O(1) per shard
- Global aggregation: O(num_shards × update_size)

### Storage
- Per client: model + data
- Per shard chief: model + received updates + reputation scores
- Global: model history + DAG + statistics

## Extension Points

### Adding New Aggregation Methods
1. Implement in `RobustAggregator.aggregate()`
2. Add to `AggregationMethod` enum

### Custom ZKP Schemes
1. Replace `ZKPGenerator` with new scheme
2. Update `ZKPVerifier` to match
3. Maintain interface compatibility

### DAG Extensions
1. Add metadata fields to `DAGNode`
2. Implement new query methods
3. Support cross-shard DAG merges for consensus

### Client Strategies
1. Extend `FederatedLearningClient`
2. Implement custom training loops
3. Add Byzantine-resilient local training

## Testing & Validation

### Unit Tests (can be added)
```python
# Test polynomial proxy consistency
def test_poly_consistency():
    gradient = np.random.randn(100)
    proxy = PolynomialProxy()
    proxy.fit_from_gradient(gradient)
    assert proxy.verify_consistency(gradient)

# Test ZKP generation and verification
def test_zkp_verification():
    gradient = np.random.randn(784)
    generator = ZKPFactory.get_generator()
    verifier = ZKPFactory.get_verifier()
    proof = generator.generate_proof(...)
    is_valid, _ = verifier.verify_proof(proof, gradient)
    assert is_valid

# Test robust aggregation
def test_robust_aggregation():
    aggregator = RobustAggregator()
    updates = [np.random.randn(100) for _ in range(10)]
    verifications = [True] * 10
    agg, meta = aggregator.aggregate(updates, verifications)
    assert agg.shape == (100,)
```

## Production Considerations

### Deployment
1. **Horizontal Scaling**: Add more shards as clients grow
2. **Fault Tolerance**: Shard chief election for failover
3. **Security**: Use proper cryptographic libraries (py-snark)
4. **Monitoring**: Comprehensive logging and metrics

### Optimization
1. **Model Compression**: Quantization before transmission
2. **Gradient Compression**: Sparse updates
3. **ZKP Optimization**: Circuit-specific optimizations
4. **Batch Aggregation**: Process multiple rounds together

### Compliance
1. **Privacy**: Differential privacy on aggregation
2. **Audit Trail**: Full DAG history
3. **Data Governance**: Client data location tracking
4. **Transparency**: Shard chief election logging

## References

- Federated Learning: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Robust Aggregation: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
- Zero-Knowledge Proofs: Schnorr zero-knowledge proofs, Fiat-Shamir heuristic
- DAG Model: Version control systems (Git), blockchain consensus mechanisms

## License

This prototype implementation is provided as-is for research and educational purposes.
