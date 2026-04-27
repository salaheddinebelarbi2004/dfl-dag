"""
Generate MiniLM-L6-v2 embeddings for filtered spam/ham data
Creates federated learning client datasets with real embeddings
"""

import pandas as pd
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

print("[LOAD] Loading filtered data...")
df = pd.read_csv("filtered_training_data.csv")
print(f"[INFO] Loaded {len(df)} messages")
print(f"[INFO] Spam: {(df['Label']=='Spam').sum()}, Ham: {(df['Label']=='Ham').sum()}")

# Get messages
messages = df['Cleaned Text'].tolist()  # Use cleaned text for embeddings

print(f"\n[MODEL] Loading all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"[EMBED] Generating embeddings for {len(messages)} messages...")
embeddings = model.encode(messages, show_progress_bar=True, convert_to_numpy=True)

print(f"[OK] Generated embeddings shape: {embeddings.shape}")

# Create one-hot labels
labels = (df['Label'].values == 'Spam').astype(int)
labels_onehot = np.eye(2)[labels]

# Create federated datasets (distribute across 12 clients non-IID)
os.makedirs("real_federated_data", exist_ok=True)

samples_per_client = len(df) // 12
print(f"\n[CLIENTS] Creating 12 federated learning clients...")
print(f"[CLIENTS] Samples per client: ~{samples_per_client}")

metadata = {}
for client_id in range(12):
    start_idx = client_id * samples_per_client
    end_idx = (client_id + 1) * samples_per_client if client_id < 11 else len(df)
    
    X = embeddings[start_idx:end_idx]
    y = labels_onehot[start_idx:end_idx]
    texts = np.array(messages[start_idx:end_idx])
    
    num_spam = np.sum(labels[start_idx:end_idx])
    num_ham = (end_idx - start_idx) - num_spam
    
    # Save as npz
    np.savez(
        f"real_federated_data/client_{client_id}.npz",
        X=X,
        y=y,
        texts=texts
    )
    
    metadata[f"client_{client_id}"] = {
        'num_samples': len(X),
        'num_spam': int(num_spam),
        'num_ham': int(num_ham),
        'embedding_dim': X.shape[1]
    }
    
    print(f"[OK] client_{client_id}: {len(X)} samples ({num_spam} spam, {num_ham} ham)")

# Save metadata
with open("real_federated_data/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[SUCCESS] Data ready in: real_federated_data/")
print(f"[INFO] Use with: python federated_learning_system.py --real-data real_federated_data")
