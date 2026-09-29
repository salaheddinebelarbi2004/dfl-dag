"""
Create adversarial versions of client_4 and client_7 in malisious/.
- Adds large noise to embeddings and flips labels to targeted class.
- Saves updated .npz files in-place under malisious/.
Usage: python make_malicious_clients.py
"""
import os
import numpy as np

target_dir = "malisious"
clients = [4,7]

for cid in clients:
    path = os.path.join(target_dir, f"client_{cid}.npz")
    data = np.load(path)
    # Support multiple key names
    if 'X' in data.files:
        X = data['X'].astype(np.float32)
    elif 'embeddings' in data.files:
        X = data['embeddings'].astype(np.float32)
    else:
        raise RuntimeError("Unknown embeddings key in npz")

    if 'y' in data.files:
        y = data['y']
    elif 'labels' in data.files:
        y = data['labels']
    else:
        raise RuntimeError("Unknown labels key in npz")

    # Make adversarial: add large directional noise and set majority to class 1
    rng = np.random.default_rng(123 + cid)
    noise = rng.normal(loc=0.0, scale=5.0, size=X.shape).astype(np.float32)
    X_adv = X + noise

    # Flip labels to spam (1) or invert
    if y.ndim > 1:
        # one-hot -> set to class 1
        y_adv = np.zeros_like(y, dtype=np.float32)
        y_adv[:, 1] = 1.0
    else:
        y_adv = np.ones_like(y, dtype=np.int64)

    # Save back using same key names
    save_kwargs = {}
    if 'X' in data.files:
        save_kwargs['X'] = X_adv
    else:
        save_kwargs['embeddings'] = X_adv

    if 'y' in data.files:
        save_kwargs['y'] = y_adv
    else:
        save_kwargs['labels'] = y_adv

    np.savez(path, **save_kwargs)
    print(f"Patched malicious client: {path}")
