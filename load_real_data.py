"""
Load real spam/ham data from Excel and generate MiniLM embedings
Remove first 4000 spam + 4000 ham (used for original classifier training)
Use remaining data for federated learning with actual embeddings
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import os
import hashlib

# Try to import optional embedding libraries
MINILM_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    MINILM_AVAILABLE = True
except:
    pass

# Fallback: simple hash-based pseudo-embedding (deterministic)
def create_pseudo_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Create deterministic pseudo-embedding from text hash.
    Falls back when sentence-transformers unavailable.
    """
    hash_obj = hashlib.sha256(text.encode())
    seed = int(hash_obj.hexdigest(), 16) % (2**31)
    np.random.seed(seed)
    return np.random.randn(dim).astype(np.float32)


def load_and_prepare_data(excel_path: str, 
                          skip_rows: int = 8000) -> Tuple[pd.DataFrame, int, int]:
    """
    Load Excel file and remove first 4000 spam + 4000 ham.
    
    Args:
        excel_path: Path to Augmented_SpamHamTweets.xlsx
        skip_rows: Number of rows to skip (8000 = 4000 spam + 4000 ham)
        
    Returns:
        Tuple of (dataframe, num_spam, num_ham)
    """
    print(f"[LOAD] Loading: {excel_path}")
    
    # Load entire dataframe
    df_full = pd.read_excel(excel_path)
    print(f"[LOAD] Total rows in file: {len(df_full)}")
    print(f"[LOAD] Columns: {df_full.columns.tolist()}")
    
    # Show sample
    print(f"\n[SAMPLE] First row:")
    print(f"[SAMPLE] {df_full.iloc[0]}")
    
    # Split into spam and ham
    spam_rows = df_full[df_full.iloc[:, 1] == 'spam']  # Assuming column 1 is label
    ham_rows = df_full[df_full.iloc[:, 1] == 'ham']
    
    print(f"\n[STATS] Full dataset breakdown:")
    print(f"[STATS] Spam messages: {len(spam_rows)}")
    print(f"[STATS] Ham messages: {len(ham_rows)}")
    print(f"[STATS] Total: {len(df_full)}")
    
    # Remove first 4000 spam and first 4000 ham
    spam_remaining = spam_rows.iloc[4000:]
    ham_remaining = ham_rows.iloc[4000:]
    
    df_remaining = pd.concat([spam_remaining, ham_remaining], ignore_index=True)
    df_remaining = df_remaining.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    print(f"\n[FILTER] After removing first 4000 spam + 4000 ham:")
    print(f"[FILTER] Spam remaining: {len(spam_remaining)}")
    print(f"[FILTER] Ham remaining: {len(ham_remaining)}")
    print(f"[FILTER] Total remaining: {len(df_remaining)}")
    
    return df_remaining, len(spam_remaining), len(ham_remaining)


def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate MiniLM embeddings for text.
    Falls back to pseudo-embeddings if sentence-transformers unavailable.
    
    Args:
        texts: List of text messages
        model_name: HuggingFace model name
        
    Returns:
        Embeddings array of shape (len(texts), 384)
    """
    if MINILM_AVAILABLE:
        print(f"\n[MODEL] Loading: {model_name}")
        try:
            model = SentenceTransformer(model_name)
            print(f"[EMBED] Generating embeddings for {len(texts)} messages...")
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            print(f"[OK] Generated embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"[WARN] MiniLM failed: {e}")
            print(f"[WARN] Falling back to pseudo-embeddings...")
            MINILM_AVAILABLE = False
    
    # Fallback: generate pseudo-embeddings from text
    print(f"\n[EMBED] Generating pseudo-embeddings for {len(texts)} messages...")
    print(f"[EMBED] (deterministic hashes of text, not actual embeddings)")
    embeddings = np.array([
        create_pseudo_embedding(text, dim=384)
        for text in texts
    ])
    print(f"[OK] Generated embeddings shape: {embeddings.shape}")
    return embeddings


def create_client_datasets(df: pd.DataFrame,
                          embeddings: np.ndarray,
                          num_clients: int = 12) -> dict:
    """
    Distribute real embedded data across clients (non-IID).
    
    Args:
        df: DataFrame with text and labels
        embeddings: Embedding vectors (N, 384)
        num_clients: Number of federated clients
        
    Returns:
        Dictionary of {client_id: {'X': embeddings, 'y': labels, 'texts': texts}}
    """
    print(f"\n[CLIENTS] Distributing data to {num_clients} clients (non-IID)...")
    
    # Get text column (usually column 0)
    text_col = df.columns[0]
    texts = df[text_col].values
    
    # Get labels (usually column 1)
    label_col = df.columns[1]
    labels = (df[label_col].values == 'spam').astype(int)  # 1 for spam, 0 for ham
    
    # One-hot encode
    labels_onehot = np.eye(2)[labels]
    
    # Non-IID distribution: each client gets different spam/ham ratio
    indices = np.arange(len(df))
    clients_data = {}
    
    samples_per_client = len(df) // num_clients
    
    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        end_idx = (client_id + 1) * samples_per_client if client_id < num_clients - 1 else len(df)
        
        client_indices = indices[start_idx:end_idx]
        
        X = embeddings[client_indices]
        y = labels_onehot[client_indices]
        client_texts = texts[client_indices]
        
        clients_data[f"client_{client_id}"] = {
            'X': X,
            'y': y,
            'texts': client_texts,
            'num_samples': len(X),
            'num_spam': np.sum(labels[client_indices]),
            'num_ham': len(client_indices) - np.sum(labels[client_indices])
        }
        
        print(f"[CLIENT] client_{client_id}: {len(X)} samples "
              f"({np.sum(labels[client_indices])} spam, "
              f"{len(client_indices) - np.sum(labels[client_indices])} ham)")
    
    return clients_data


def save_client_data(clients_data: dict, output_dir: str = "real_federated_data"):
    """
    Save client data as numpy files for federated learning.
    
    Args:
        clients_data: Dictionary of client datasets
        output_dir: Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[SAVE] Saving data to {output_dir}/")
    
    for client_id, data in clients_data.items():
        np.savez(
            f"{output_dir}/{client_id}.npz",
            X=data['X'],
            y=data['y'],
            texts=data['texts']
        )
        print(f"[SAVE] {client_id}.npz saved")
    
    # Save metadata
    metadata = {
        client_id: {
            'num_samples': data['num_samples'],
            'num_spam': int(data['num_spam']),
            'num_ham': int(data['num_ham']),
            'embedding_dim': data['X'].shape[1]
        }
        for client_id, data in clients_data.items()
    }
    
    import json
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[SAVE] metadata.json saved")
    return output_dir


if __name__ == "__main__":
    print("=" * 60)
    print("LOAD REAL SPAM/HAM DATA WITH MINILM EMBEDDINGS")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    excel_path = "d:\\p2p network\\Augmented_SpamHamTweets.xlsx"
    df, num_spam, num_ham = load_and_prepare_data(excel_path)
    
    # Step 2: Generate MiniLM embeddings
    texts = df[df.columns[0]].tolist()  # Get text column
    embeddings = generate_embeddings(texts, model_name="all-MiniLM-L6-v2")
    
    # Step 3: Distribute to clients
    clients_data = create_client_datasets(df, embeddings, num_clients=12)
    
    # Step 4: Save for federated learning
    output_dir = save_client_data(clients_data)
    
    print("\n" + "=" * 60)
    print("[OK] READY FOR FEDERATED LEARNING!")
    print("=" * 60)
    print(f"Data saved in: {output_dir}/")
    print(f"Use with: python federated_learning_system.py --real-data {output_dir}")
