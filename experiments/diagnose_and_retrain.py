import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from lightweight_classifier import LightweightClassifier


DATA_DIR = Path("real_federated_data")
MAL_DIR = Path("malisious")
OUT_DIR = Path("results/malicious_baseline")
OUT_DIR.mkdir(exist_ok=True)


def summarize_label_distribution(data_dir: Path):
    summary = {}
    total_y = []
    for path in sorted(data_dir.glob("client_*.npz")):
        cid = path.stem
        data = np.load(path)
        y = data["y"]
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        vals, counts = np.unique(y, return_counts=True)
        dist = {int(int(k)): int(int(v)) for k, v in zip(vals, counts)}
        summary[cid] = dist
        total_y.append(y)
    all_y = np.concatenate(total_y)
    vals, counts = np.unique(all_y, return_counts=True)
    summary["global"] = {int(int(k)): int(int(v)) for k, v in zip(vals, counts)}
    return summary


def centralized_baseline(data_dir: Path, epochs: int = 20, lr: float = 0.05):
    X_parts, y_parts = [], []
    for path in sorted(data_dir.glob("client_*.npz")):
        data = np.load(path)
        X = data["X"].astype(np.float32)
        y = data["y"]
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        X_parts.append(X)
        y_parts.append(y)

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.20, random_state=42, stratify=y_all)

    clf = LightweightClassifier(input_dim=X_train.shape[1], output_dim=2, learning_rate=lr)
    for epoch in range(epochs):
        clf.train_step(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    loss = float(clf.compute_loss(X_test, y_test))
    return {"accuracy": acc, "f1": f1, "loss": loss}


def main():
    # Summarize clean data
    real_summary = summarize_label_distribution(DATA_DIR)
    mal_summary = summarize_label_distribution(MAL_DIR)
    with open(OUT_DIR / "data_label_summary.json", "w") as fh:
        json.dump({"real": real_summary, "malisious": mal_summary}, fh, indent=2)

    # Centralized baseline on real_federated_data
    baseline = centralized_baseline(DATA_DIR, epochs=30, lr=0.03)
    with open(OUT_DIR / "centralized_baseline_metrics.json", "w") as fh:
        json.dump(baseline, fh, indent=2)

    print("Diagnostics complete. Outputs:")
    print(" -", OUT_DIR / "data_label_summary.json")
    print(" -", OUT_DIR / "centralized_baseline_metrics.json")


if __name__ == "__main__":
    main()
