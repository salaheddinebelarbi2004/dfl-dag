"""
Configuration for Sharded Federated Learning System
Defines all hyperparameters, constants, and system settings
"""

# ============ SYSTEM PARAMETERS ============
NUM_CLIENTS = 12  # Total number of clients
NUM_SHARDS = 3  # Number of geographic/network shards
SHARD_SIZE = NUM_CLIENTS // NUM_SHARDS  # Clients per shard
NUM_ROUNDS = 5  # Training rounds
LOCAL_EPOCHS = 2  # Local training epochs per client per round

# ============ NEURAL NETWORK PARAMETERS ============
# NLP Model: Spam Detection with MiniLM embeddings
INPUT_DIM = 384  # MiniLM sentence embedding dimension
HIDDEN_DIM = 256  # Match your PyTorch model (model.pt has 256)
OUTPUT_DIM = 2  # Binary classification: Spam (1) / Ham (0)
LEARNING_RATE = 0.01
LOCAL_BATCH_SIZE = 32
GRADIENT_CLIP = 1.0  # Norm bound B for gradients

# ============ POLYNOMIAL PROXY PARAMETERS ============
POLY_DEGREE = 3  # Polynomial proxy degree
POLY_EVAL_POINTS = 5  # Number of evaluation points for ZKP

# ============ ZERO-KNOWLEDGE PROOF PARAMETERS ============
ZKP_SECURITY_PARAM = 128  # Security parameter (bits)
ZKP_CHALLENGE_BITS = 64  # Challenge size
ZKP_ROUNDS = 3  # Interactive proof rounds (simplified)

# ============ SHARD CHIEF ELECTION PARAMETERS ============
SHARD_CHIEF_SCORE_ALPHA = 0.4  # Weight for uptime
SHARD_CHIEF_SCORE_BETA = 0.3  # Weight for bandwidth
SHARD_CHIEF_SCORE_GAMMA = 0.3  # Weight for compute capacity
TOP_K_CANDIDATES = 5  # Top K candidates for Shard Chief election

# ============ ROBUST AGGREGATION PARAMETERS ============
CLUSTER_K = 3  # Number of clusters for cluster-consensus
CLUSTERING_METHOD = "kmeans"  # "kmeans" or "hierarchical"
OUTLIER_THRESHOLD = 2.0  # Std devs from mean to flag as outlier
MIN_REPUTATION_THRESHOLD = 0.5  # Min reputation to be included
MAX_REPUTATION = 1.0
INIT_REPUTATION = 0.8

# ============ DAG PARAMETERS ============
DAG_VALIDATION_REQUIRED = True  # Require ZKP validation for DAG addition
DAG_MAX_CONCURRENT_SIBLINGS = 5  # Max concurrent updates from same parent

# ============ SECURITY PARAMETERS ============
COMMITMENT_HASH_ALG = "sha256"  # Hash algorithm for commitments
UPDATE_SIGNATURE_SCHEME = "ed25519"  # Digital signature scheme
MALICIOUS_CLIENT_PENALTY = 0.3  # Reputation penalty for malicious clients

# ============ LOGGING & MONITORING ============
VERBOSE = True  # Detailed logging
METRICS_ENABLED = True  # Track metrics
