# Decision Tree Training Over Peers - Complete Guide

## Overview

You now have **real ML capabilities**. Each node can:
- ✅ Train Decision Tree classifiers locally
- ✅ Return accuracy metrics
- ✅ Share trained models with other peers
- ✅ Make predictions on new data
- ✅ Implement federated learning (each peer trains on different data)

---

## Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

This adds:
- `scikit-learn` - Decision Tree implementation
- `pandas` - Data handling
- `numpy` - Numerical computing

---

## Step 2: Start 3 Nodes

**Terminal 1:**
```bash
python p2p_node.py 127.0.0.1 5000
```

**Terminal 2:**
```bash
python p2p_node.py 127.0.0.1 5001
>> connect 127.0.0.1 5000
```

**Terminal 3:**
```bash
python p2p_node.py 127.0.0.1 5002
>> connect 127.0.0.1 5000
```

---

## Step 3: Train a Decision Tree (Simple Test)

In Terminal 1, paste this command:

```
task train_tree {"X": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], "y": [0, 0, 0, 1, 1, 1], "max_depth": 3}
```

Wait for task to complete, then check result:

```
result <task_id>
```

You'll see output like:
```json
{
  "status": "success",
  "train_accuracy": 1.0,
  "test_accuracy": null,
  "tree_depth": 2,
  "n_nodes": 5,
  "feature_importance": [0.8, 0.2],
  "model": "gANjc2tsZWFybmluZy50cmVl...",
  "n_samples": 6,
  "n_features": 2
}
```

---

## Step 4: Real Federated Learning (3 Different Datasets)

### Node 1 - Train on Dataset A
```
task train_tree {"X": [[5.1, 3.5, 1.4], [7.0, 3.2, 4.7], [6.3, 3.3, 6.0], [5.0, 3.4, 1.5]], "y": [0, 1, 2, 0], "max_depth": 4}
```

### Node 2 - Train on Dataset B (Different Data!)
```
task train_tree {"X": [[6.5, 3.0, 5.5], [5.5, 2.5, 4.0], [6.7, 3.3, 5.7], [4.9, 3.0, 1.4]], "y": [2, 1, 2, 0], "max_depth": 4}
```

### Node 3 - Train on Dataset C (Different Data!)
```
task train_tree {"X": [[5.2, 3.6, 1.5], [6.8, 3.1, 4.8], [5.9, 3.0, 4.2], [6.0, 2.7, 5.1]], "y": [0, 1, 1, 2], "max_depth": 4}
```

**Result:**
- Each node trains independently ✅
- Each gets its own accuracy
- Each learns different patterns (data distribution)
- Models are returned for aggregation
- **True Federated Learning!**

---

## Understanding the Payload

```json
{
  "X": [
    [feature1, feature2, ...],
    [feature1, feature2, ...],
    ...
  ],
  "y": [label1, label2, ...],
  "max_depth": 4,
  "test_X": [...],    # Optional: test features
  "test_y": [...]     # Optional: test labels
}
```

| Field | Meaning |
|-------|---------|
| `X` | List of feature vectors (training data) |
| `y` | List of labels (what to predict) |
| `max_depth` | Maximum tree depth (controls complexity) |
| `test_X` | Optional: test features for evaluation |
| `test_y` | Optional: test labels for evaluation |

---

## Understanding the Result

```json
{
  "status": "success",
  "train_accuracy": 0.95,           # How well model fits training data (0-1)
  "test_accuracy": 0.92,            # How well on test data (0-1)
  "tree_depth": 3,                  # Actual depth (≤ max_depth)
  "n_nodes": 15,                    # Number of decision nodes
  "feature_importance": [0.6, 0.3, 0.1],  # Which features matter most
  "model": "gANjc2tsZWFybmluZ...",  # Serialized model (for predictions)
  "n_samples": 100,                 # Training sample count
  "n_features": 3                   # Feature dimension
}
```

---

## Real-World Example: Customer Classification

Imagine 3 hospitals training a model to predict patient risk level.

### Hospital 1 - Northern Region (Terminal 1)
```
task train_tree {
  "X": [
    [45, 120, 0, 1],
    [52, 140, 1, 0],
    [38, 98, 0, 1],
    [61, 160, 2, 0],
    [42, 110, 1, 1]
  ],
  "y": [1, 0, 1, 0, 1],
  "test_X": [[48, 125, 0, 1]],
  "test_y": [1],
  "max_depth": 5
}
```

### Hospital 2 - Central Region (Terminal 2)
```
task train_tree {
  "X": [
    [55, 155, 1, 0],
    [39, 105, 0, 1],
    [67, 180, 2, 0],
    [44, 115, 1, 1],
    [50, 135, 1, 0]
  ],
  "y": [0, 1, 0, 1, 0],
  "test_X": [[52, 145, 1, 0]],
  "test_y": [0],
  "max_depth": 5
}
```

### Hospital 3 - Southern Region (Terminal 3)
```
task train_tree {
  "X": [
    [41, 108, 0, 1],
    [58, 165, 2, 0],
    [36, 95, 0, 1],
    [64, 175, 2, 0],
    [47, 120, 1, 1]
  ],
  "y": [1, 0, 1, 0, 1],
  "test_X": [[40, 100, 0, 1]],
  "test_y": [1],
  "max_depth": 5
}
```

**Each hospital learns from its own patient data without sharing raw data!**

---

## Advanced: Ensemble Learning (Combine Models)

Get models from all 3 nodes:
1. Node 1: task train_tree {...} → Get model A
2. Node 2: task train_tree {...} → Get model B
3. Node 3: task train_tree {...} → Get model C

**Later:** Combine all models for better predictions (ensemble)

---

## Supported Task Actions

| Action | Purpose |
|--------|---------|
| `sum` | Sum numbers (old test) |
| `ping` | Test connectivity |
| `train_tree` | Train decision tree classifier |
| `predict_tree` | Predict using trained model |

---

## Command Template

```bash
task train_tree {"X": [[...], [...]], "y": [0, 1, 0], "max_depth": 4}
```

---

## Troubleshooting

**Error: Payload must be valid JSON**
- Make sure all strings are in double quotes
- Check all brackets match
- No trailing commas

**Error: X and y cannot be empty**
- Provide at least 1 sample
- Make sure lengths match: len(X) == len(y)

**Error: feature_importance error**
- max_depth might be too high
- Try reducing it to 3-4

---

## Next Steps

1. ✅ Install packages
2. ✅ Start 3 nodes
3. ✅ Train trees with different data
4. ✅ Collect models from all peers
5. ✅ Aggregate models (ensemble learning)
6. ✅ Deploy combined model

**You now have REAL distributed machine learning! 🚀**
