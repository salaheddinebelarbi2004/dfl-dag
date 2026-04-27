"""
QUICK START GUIDE
Sharded Federated Learning System
"""

# ============================================================
# INSTALLATION
# ============================================================

# 1. Navigate to project directory
cd "d:\p2p network"

# 2. Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, scipy, sklearn; print('✓ Dependencies installed')"


# ============================================================
# RUNNING THE SYSTEM
# ============================================================

# Option 1: Run complete training system
python federated_learning_system.py

# This will:
# - Initialize 12 clients in 3 shards
# - Run 5 training rounds
# - Execute all 10 workflow steps per round
# - Save results to training_results.json
# - Print detailed progress to console

# Option 2: Run test suite
python test_system.py

# This will:
# - Test polynomial proxy
# - Test ZKP generation/verification
# - Test client training
# - Test robust aggregation
# - Test Shard Chief election
# - Test DAG system
# - Test global aggregator
# - Run integration test

# Option 3: Run custom training
# See CUSTOM_EXAMPLES.md


# ============================================================
# OUTPUT FILES
# ============================================================

# training_results.json
# - Complete system statistics
# - Per-round metrics
# - Client and shard chief stats
# - DAG information
# - Aggregation results


# ============================================================
# CONFIGURATION
# ============================================================

# Edit federated_learning_config.py to change:

# System Scale
NUM_CLIENTS = 12  # Total clients
NUM_SHARDS = 3    # Number of shards

# Training Parameters
NUM_ROUNDS = 5         # Training rounds
LOCAL_EPOCHS = 2      # Local training epochs
LEARNING_RATE = 0.01  # Learning rate

# Security Parameters
ZKP_SECURITY_PARAM = 128  # ZKP security bits
GRADIENT_CLIP = 1.0       # Gradient norm bound

# Aggregation
CLUSTER_K = 3                    # Clustering parameter
MIN_REPUTATION_THRESHOLD = 0.5   # Min reputation score

# See federated_learning_config.py for all options


# ============================================================
# UNDERSTANDING THE WORKFLOW
# ============================================================

# 10-Step Training Workflow (per round):

# Step 1: Global Model Broadcast
# - Server sends Wt to all clients

# Steps 2-6: Client Local Training & Update
# - Local training: Wi = Wt - η∇L(Di, Wt)
# - Update: ∆Wi = Wi - Wt
# - Polynomial proxy: Pi(x) ≈ f(∆Wi)
# - ZKP proof: πi
# - Commitment: Ci = Commit(∆Wi)

# Steps 7-9: Shard Verification & Aggregation
# - Verify ZKP proofs
# - Filter by reputation
# - Robust aggregation → Ws

# Step 10: Global Aggregation
# - Combine shard models: Wt+1 = Aggregate(Ws)


# ============================================================
# KEY COMPONENTS
# ============================================================

# federated_learning_config.py
#   └─ System configuration (all parameters in one place)

# client.py
#   └─ FederatedLearningClient: local training
#   └─ NeuralNetworkSimulator: network simulation
#   └─ ClientUpdate: update package

# polynomial_proxy.py
#   └─ PolynomialProxy: gradient encoding
#   └─ Lightweight update representation

# zkp_proofs.py
#   └─ ZKPGenerator: proof generation (Schnorr-like)
#   └─ ZKPVerifier: proof verification
#   └─ Proves: gradient correctness, consistency, norm bound

# robust_aggregation.py
#   └─ RobustAggregator: multi-stage aggregation
#   └─ Methods: mean, median, cluster-consensus, reputation-weighted, hybrid

# shard_chief.py
#   └─ ShardChief: shard-level coordinator
#   └─ ShardChiefElector: election mechanism
#   └─ Score: Sj = α·Uj + β·Bj + γ·Cj

# dag_system.py
#   └─ DAGGraph: model evolution tracking per shard
#   └─ ShardedDAG: multi-shard DAG system
#   └─ Tracks: valid updates, lineage, versioning

# global_aggregator.py
#   └─ GlobalAggregator: global model aggregation
#   └─ GlobalCoordinator: round orchestration

# federated_learning_system.py
#   └─ ShardedFederatedLearningSystem: complete integration
#   └─ Executes: full 10-step workflow per round


# ============================================================
# EXAMPLE: CUSTOM TRAINING
# ============================================================

from federated_learning_system import ShardedFederatedLearningSystem

# Create system
system = ShardedFederatedLearningSystem(
    num_clients=20,      # 20 clients
    num_shards=4,        # 4 shards
    num_rounds=10        # 10 rounds
)

# Run training
results = system.execute_full_training()

# Access results
print(f"Rounds: {results['num_rounds']}")
print(f"Training time: {results['total_training_time_seconds']:.2f}s")

# Save results
system.save_results("my_results.json")


# ============================================================
# EXAMPLE: SHARD CHIEF ELECTION
# ============================================================

from shard_chief import ShardChiefElector, ShardChiefCandidate, NodeCapabilities

# Create candidates
candidates = [
    ShardChiefCandidate(
        node_id=f"node_{i}",
        capabilities=NodeCapabilities(
            node_id=f"node_{i}",
            uptime=0.95,           # 95% uptime
            bandwidth_mbps=100,    # 100 Mbps
            compute_capacity=1.0   # 1x compute
        )
    )
    for i in range(5)
]

# Elect chief
elector = ShardChiefElector()
chief, election_meta = elector.elect_for_shard(
    candidates, "shard_0", top_k=3
)

print(f"Elected: {chief.chief_id}, Score: {election_meta['score']:.3f}")


# ============================================================
# EXAMPLE: ROBUST AGGREGATION
# ============================================================

from robust_aggregation import RobustAggregator, AggregationMethod
import numpy as np

# Create aggregator
aggregator = RobustAggregator()

# Create updates
updates = [np.random.randn(100) for _ in range(10)]
client_ids = [f"client_{i}" for i in range(10)]
verification_results = [True] * 10

# Aggregate with robust hybrid method
agg, metadata = aggregator.aggregate(
    updates,
    verification_results=verification_results,
    client_ids=client_ids,
    method=AggregationMethod.ROBUST_HYBRID
)

print(f"Aggregated update shape: {agg.shape}")
print(f"Stages: {metadata['stages'].keys()}")


# ============================================================
# EXAMPLE: ZKP GENERATION & VERIFICATION
# ============================================================

from zkp_proofs import ZKPFactory
from polynomial_proxy import PolynomialProxyFactory
import numpy as np

# Create gradient
gradient = np.random.randn(784)
model_params = np.random.randn(784) * 0.01

# Generate proof
generator = ZKPFactory.get_generator()
poly = PolynomialProxyFactory.create_from_update(gradient)
proof = generator.generate_proof(
    gradient, model_params,
    client_id="client_0",
    round_num=1,
    polynomial_proxy=poly
)

# Verify proof
verifier = ZKPFactory.get_verifier()
is_valid, details = verifier.verify_proof(proof, gradient, poly)

print(f"Proof valid: {is_valid}")
print(f"Checks: {details['checks'].keys()}")


# ============================================================
# MONITORING & METRICS
# ============================================================

# Access system metrics after training
stats = system._get_final_stats()

# Global coordinator stats
gc_stats = stats['global_coordinator']
print(f"Aggregations: {gc_stats['total_aggregations']}")
print(f"Success rate: {gc_stats['aggregation_success_rate']:.1%}")

# Shard chief stats
for shard_id, chief_stats in stats['shard_chiefs'].items():
    print(f"{shard_id}: {chief_stats['updates_verified']} verified")

# DAG stats
dag_stats = stats['dag_stats']
print(f"Global model snapshots: {dag_stats['global_model_history_length']}")

# Client stats
for client_stat in stats['client_stats'][:3]:  # First 3 clients
    print(f"{client_stat['client_id']}: {client_stat['rounds_completed']} rounds")


# ============================================================
# TROUBLESHOOTING
# ============================================================

# Q: ModuleNotFoundError: No module named 'numpy'
# A: Run: pip install -r requirements.txt

# Q: System running very slowly
# A: Reduce NUM_CLIENTS or NUM_SHARDS in config, or reduce NUM_ROUNDS

# Q: ZKP verification failing
# A: Check gradient norm doesn't exceed GRADIENT_CLIP
# A: Verify polynomial proxy consistency

# Q: Aggregation failing
# A: Check at least one update has verification_result=True
# A: Reduce OUTLIER_THRESHOLD if many legitimate updates filtered

# Q: Shard chief election empty
# A: Ensure NUM_CLIENTS >= NUM_SHARDS, provide local node list

# Q: DAG nodes not finalizing
# A: Verify ZKP proofs are passing verification
# A: Check commitment verification is True

# Q: Out of memory with large datasets
# A: Reduce LOCAL_BATCH_SIZE in config
# A: Use fewer clients (NUM_CLIENTS)


# ============================================================
# NEXT STEPS
# ============================================================

# 1. Run the demo system
#    python federated_learning_system.py

# 2. Review architecture documentation
#    Read SYSTEM_ARCHITECTURE.md

# 3. Run tests
#    python test_system.py

# 4. Customize parameters
#    Edit federated_learning_config.py

# 5. Add extensions
#    Implement custom aggregation methods
#    Add differential privacy
#    Implement Byzantine-resilient training

# 6. Deploy at scale
#    Add distributed communication
#    Implement fault tolerance
#    Add monitoring/logging


# ============================================================
# RESOURCES
# ============================================================

# Documentation
# - SYSTEM_ARCHITECTURE.md: Complete system design
# - CUSTOM_EXAMPLES.md: Code examples
# - federated_learning_config.py: Configuration reference

# Code Examples
# - test_system.py: Component tests
# - federated_learning_system.py: Complete system example

# Academic References
# - McMahan et al.: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
# - Blanchard et al.: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
# - ZKP Schemes: Schnorr proofs, Fiat-Shamir heuristic

# ============================================================
