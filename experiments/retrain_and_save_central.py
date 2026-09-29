import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from client import NeuralNetworkSimulator

DATA_DIR = Path("real_federated_data")
OUT_DIR = Path("results/malicious_baseline")
OUT_DIR.mkdir(exist_ok=True)


def load_all_data(data_dir: Path):
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
    return X_all, y_all


def train_and_save(epochs: int = 50, lr: float = 0.05):
    X_all, y_all = load_all_data(DATA_DIR)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all)

    model = NeuralNetworkSimulator(input_dim=X_train.shape[1], hidden_dim=256, output_dim=2)
    for epoch in range(epochs):
        model.update_params(model.backward(X_train, y_train), learning_rate=lr)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Save flattened params in same format as system
    params = model.get_params()
    np.save(OUT_DIR / "global_model_malicious.npy", params)
    metrics = {"accuracy": acc, "f1": f1}
    with open(OUT_DIR / "centralized_retrain_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("Retrain complete. Saved model and metrics to:")
    print(" -", OUT_DIR / "global_model_malicious.npy")
    print(" -", OUT_DIR / "centralized_retrain_metrics.json")


if __name__ == "__main__":
    train_and_save()
