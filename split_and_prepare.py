"""Split `clean_spam_dataset.csv` into per-client .npz files with embeddings.

Saves files into `real_federated_data/client_{i}.npz` with arrays `X` and `y`.

Usage: python split_and_prepare.py --clients 12 --seed 42
"""
import argparse
import csv
import os
from pathlib import Path
import numpy as np

from nlp_embeddings import MiniLMEmbedder


def read_csv(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def label_to_int(label: str) -> int:
    if label is None:
        return 0
    l = label.strip().lower()
    if l.startswith('spam') or l == 'spam' or l == '1':
        return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='clean_spam_dataset.csv')
    p.add_argument('--clients', type=int, default=12)
    p.add_argument('--outdir', default='real_federated_data')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = read_csv(csv_path)
    texts = []
    labels = []
    # Prefer columns 'Cleaned Text' or 'Cleaned_Text' or 'Tweet Text'
    for r in rows:
        txt = r.get('Cleaned Text') or r.get('Cleaned_Text') or r.get('Tweet Text') or r.get('Text')
        lab = r.get('Label') or r.get('label')
        if not txt:
            continue
        texts.append(txt)
        labels.append(label_to_int(lab))

    print(f"Loaded {len(texts)} examples from {csv_path}")

    # Embed all texts
    embedder = MiniLMEmbedder(model_name='all-MiniLM-L6-v2')
    embeddings = []
    batch = []
    for t in texts:
        emb = embedder.embed_text(t)
        embeddings.append(emb.reshape(-1))

    X = np.vstack(embeddings).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Split into roughly equal chunks
    n = len(X)
    sizes = [n // args.clients] * args.clients
    for i in range(n % args.clients):
        sizes[i] += 1

    start = 0
    for cid in range(args.clients):
        size = sizes[cid]
        if size == 0:
            arrX = np.empty((0, X.shape[1]), dtype=np.float32)
            arry = np.empty((0,), dtype=np.int64)
        else:
            arrX = X[start:start+size]
            arry = y[start:start+size]
        start += size

        fname = outdir / f'client_{cid}.npz'
        np.savez_compressed(fname, X=arrX, y=arry)
        print(f"Saved {fname} (samples={size})")


if __name__ == '__main__':
    main()
