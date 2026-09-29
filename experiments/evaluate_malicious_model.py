import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from lightweight_classifier import LightweightClassifier
from client import NeuralNetworkSimulator
from federated_learning_config import INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM


OUTPUT_DIR = Path("results/malicious_baseline")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_real_federated_dataset(data_dir: Path):
    X_parts = []
    y_parts = []
    for path in sorted(data_dir.glob("client_*.npz")):
        data = np.load(path)
        X = data["X"].astype(np.float32)
        y = data["y"]
        if y.ndim > 1:
            y = np.argmax(y, axis=1).astype(np.int64)
        else:
            y = y.astype(np.int64).reshape(-1)
        X_parts.append(X)
        y_parts.append(y)

    X_all = np.vstack(X_parts).astype(np.float32)
    y_all = np.concatenate(y_parts).astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham", "Spam"])
    ax.set_yticklabels(["Ham", "Spam"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"), ha="center", va="center", color="black"
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc(y_true, y_score, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.12)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bar(metrics: dict, out_path: Path):
    labels = ["accuracy", "f1", "loss"]
    values = [
        metrics.get("accuracy", 0.0),
        metrics.get("f1", 0.0),
        metrics.get("loss", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, values, color=["#2c7fb8", "#66c2a5", "#d95f02"])
    ax.set_ylim(0, 1 if max(values) <= 1 else max(values) * 1.1)
    ax.set_title("Evaluation Metrics")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01 * max(1.0, v), f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    model_path = OUTPUT_DIR / "global_model_malicious.npy"
    if not model_path.exists():
        print("Model file not found:", model_path)
        return

    weights = np.load(model_path)
    # If the flattened weights match the main MLP model expected size, use
    # the NeuralNetworkSimulator which the system uses. Otherwise, fall back
    # to LightweightClassifier when dimensions look like that model.
    expected_size = (
        INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    )
    if len(weights) == expected_size:
        model = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model.set_params(weights)
        use_nn = True
    else:
        # fallback: try to infer lightweight classifier dims (output_dim=2)
        out_dim = 2
        if (len(weights) - out_dim) % out_dim != 0:
            input_dim = 384
        else:
            input_dim = int((len(weights) - out_dim) / out_dim)
        clf = LightweightClassifier(input_dim=input_dim, output_dim=out_dim)
        clf.set_weights(weights)
        use_nn = False

    X_train, X_test, y_train, y_test = load_real_federated_dataset(
        Path("real_federated_data")
    )

    if use_nn:
        y_pred = model.predict(X_test)
        loss = float(model.compute_loss(X_test, y_test))
        probs = model.forward(X_test)[0][:, 1]
    else:
        y_pred = clf.predict(X_test)
        loss = float(clf.compute_loss(X_test, y_test))
        probs = clf.forward(X_test)[1][:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    metrics = {"loss": loss, "accuracy": acc, "f1": f1}

    # save metrics json
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # plots
    save_confusion_matrix(y_test, y_pred, OUTPUT_DIR / "confusion_matrix.png")
    save_roc(y_test, probs, OUTPUT_DIR / "roc_curve.png")
    save_metrics_bar(metrics, OUTPUT_DIR / "metrics_bar.png")

    print("Evaluation complete. Artifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
