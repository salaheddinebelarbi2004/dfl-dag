"""
Configuration for Sharded Federated Learning System
Defines all hyperparameters, constants, and system settings
"""

import os

# ============ SYSTEM PARAMETERS ============
NUM_CLIENTS = 12  # Total number of clients
NUM_SHARDS = 3  # Number of geographic/network shards
SHARD_SIZE = NUM_CLIENTS // NUM_SHARDS  # Clients per shard
NUM_ROUNDS = 5  # Training rounds
LOCAL_EPOCHS = 5  # Local training epochs per client per round
RANDOM_SEED = 42

# ============ NEURAL NETWORK PARAMETERS ============
# NLP Model: Spam Detection with MiniLM embeddings
INPUT_DIM = 384  # MiniLM sentence embedding dimension
HIDDEN_DIM = 256  # Match your PyTorch model (model.pt has 256)
OUTPUT_DIM = 2  # Binary classification: Spam (1) / Ham (0)
LEARNING_RATE = 0.05
LOCAL_BATCH_SIZE = 32
GRADIENT_CLIP = 1.0  # Norm bound B for gradients

# ============ COMMUNICATION / COMPRESSION (Option A) ============
# Compression method: "topk" | "quantize" | "topk_quantize"
COMPRESSION_METHOD = os.environ.get("COMPRESSION_METHOD", "topk")
# Baseline of the README: compression is disabled when comparing against the uncompressed system.
COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", "1.0"))
QUANTIZE_ENABLED = bool(int(os.environ.get("QUANTIZE_ENABLED", "0")))
ERROR_FEEDBACK = bool(int(os.environ.get("ERROR_FEEDBACK", "0")))
# Bit-width for quantization routines
QUANTIZE_BITS = int(os.environ.get("QUANTIZE_BITS", "8"))

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
# Soft weight to apply to suspected outliers instead of removing them entirely
# Increased to 0.5 to reduce aggressive downweighting that made the model conservative
SOFT_OUTLIER_WEIGHT = 0.5

# Label-skew detection: if a client has > LABEL_SKEW_THRESHOLD fraction of one class,
# consider it suspicious (likely poisoned) and apply SKEW_OUTLIER_WEIGHT.
# Raise threshold to 0.95 so only extreme skews are flagged; use milder downweight 0.2
LABEL_SKEW_THRESHOLD = 0.95
SKEW_OUTLIER_WEIGHT = 0.2

# ============ LOGGING & MONITORING ============
VERBOSE = True  # Detailed logging
METRICS_ENABLED = True  # Track metrics
