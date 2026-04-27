# Decision Tree Training Example
# Real federated learning task using scikit-learn

import json

# Sample training data (Iris-like dataset)
TRAINING_DATA = {
    "X": [
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
        [5.0, 3.4, 1.5, 0.2],
        [6.5, 3.0, 5.5, 1.8],
        [5.5, 2.5, 4.0, 1.3],
        [6.7, 3.3, 5.7, 2.5],
        [4.9, 3.0, 1.4, 0.2]
    ],
    "y": [0, 1, 2, 0, 2, 1, 2, 0],  # Class labels
    "max_depth": 4
}

TEST_DATA = {
    "test_X": [
        [5.2, 3.6, 1.5, 0.3],
        [6.8, 3.1, 4.8, 1.5]
    ],
    "test_y": [0, 1]
}

# ===================================
# COMMAND TO RUN IN NODE TERMINAL
# ===================================

# Copy the payload as JSON:
payload = {
    "X": [
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
        [5.0, 3.4, 1.5, 0.2],
        [6.5, 3.0, 5.5, 1.8],
        [5.5, 2.5, 4.0, 1.3],
        [6.7, 3.3, 5.7, 2.5],
        [4.9, 3.0, 1.4, 0.2]
    ],
    "y": [0, 1, 2, 0, 2, 1, 2, 0],
    "test_X": [[5.2, 3.6, 1.5, 0.3], [6.8, 3.1, 4.8, 1.5]],
    "test_y": [0, 1],
    "max_depth": 4
}

# In any node terminal, execute:
# >> task train_tree <JSON_PAYLOAD>

# Example (copy-paste this):
# >> task train_tree {"X": [[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5], [5.0, 3.4, 1.5, 0.2], [6.5, 3.0, 5.5, 1.8], [5.5, 2.5, 4.0, 1.3], [6.7, 3.3, 5.7, 2.5], [4.9, 3.0, 1.4, 0.2]], "y": [0, 1, 2, 0, 2, 1, 2, 0], "test_X": [[5.2, 3.6, 1.5, 0.3], [6.8, 3.1, 4.8, 1.5]], "test_y": [0, 1], "max_depth": 4}

# ===================================
# EXPECTED RESULT
# ===================================

# >> task train_tree {...}
# 📋 Task created: abc-123
#
# >> result abc-123
# {
#   "status": "success",
#   "train_accuracy": 1.0,           ← Training accuracy
#   "test_accuracy": 1.0,            ← Test accuracy
#   "tree_depth": 3,                 ← Max depth reached
#   "n_nodes": 11,                   ← Number of internal nodes
#   "feature_importance": [0.5, 0.3, 0.2, 0.0],  ← Feature importance scores
#   "model": "gANjc2tsZWFybmluZy50cmVl...",      ← Encoded model (for predictions)
#   "n_samples": 8,                  ← Training samples
#   "n_features": 4                  ← Feature dimensions
# }

# ===================================
# FEDERATED LEARNING WORKFLOW
# ===================================

# Scenario: Train tree on 3 peers with different datasets

# Terminal 1 (Node 1) - Train on Dataset A:
task train_tree {"X": [[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5], [5.0, 3.4, 1.5, 0.2]], "y": [0, 1, 2, 0], "max_depth": 4}

# Terminal 2 (Node 2) - Train on Dataset B:
task train_tree {"X": [[6.5, 3.0, 5.5, 1.8], [5.5, 2.5, 4.0, 1.3], [6.7, 3.3, 5.7, 2.5], [4.9, 3.0, 1.4, 0.2]], "y": [2, 1, 2, 0], "max_depth": 4}

# Terminal 3 (Node 3) - Train on Dataset C:
task train_tree {"X": [[5.2, 3.6, 1.5, 0.3], [6.8, 3.1, 4.8, 1.5], [5.9, 3.0, 4.2, 1.5], [6.0, 2.7, 5.1, 1.6]], "y": [0, 1, 1, 2], "max_depth": 4}

# Each node trains locally on its own data
# Results show accuracies for each node
# Models are returned in encoded format for aggregation

# ===================================
# SIMPLE TEST (Small dataset)
# ===================================

# Minimal test:
task train_tree {"X": [[1, 2], [2, 3], [3, 4], [4, 5]], "y": [0, 0, 1, 1], "max_depth": 2}

# Expected result:
# {
#   "status": "success",
#   "train_accuracy": 1.0,
#   "test_accuracy": null,
#   "tree_depth": 2,
#   "n_nodes": 5,
#   "feature_importance": [0.5, 0.5],
#   "model": "...",
#   "n_samples": 4,
#   "n_features": 2
# }

# ===================================
# REAL-WORLD SCENARIO
# ===================================

# Use Case: Distributed Customer Classification
# Different clients train trees on local customer data

# Client 1 (Node 1) - Northern region data
task train_tree {
  "X": [
    [25, 50000, 1, 0],  # age, income, years_customer, has_loan
    [35, 75000, 3, 1],
    [45, 100000, 5, 1],
    [28, 55000, 2, 0]
  ],
  "y": [0, 1, 1, 0],  # 0=low_risk, 1=high_value
  "max_depth": 5
}

# Client 2 (Node 2) - Southern region data
task train_tree {
  "X": [
    [30, 60000, 2, 1],
    [40, 85000, 4, 0],
    [50, 120000, 6, 1],
    [32, 70000, 3, 1]
  ],
  "y": [1, 0, 1, 1],
  "max_depth": 5
}

# Each learns local patterns
# Accuracies indicate data quality per region
# Models can be aggregated for global insights

print("Ready to train decision trees across your P2P network!")
