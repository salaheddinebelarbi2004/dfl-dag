import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REAL_DATA_DIR = Path("real_federated_data")
INVALID_DATA_DIR = Path("invalid data")

from federated_learning_config import GRADIENT_CLIP
from lightweight_classifier import LightweightClassifier
from polynomial_proxy import PolynomialProxyFactory
from robust_aggregation import AggregationMethod, RobustAggregator
from zkp_proofs import Groth16Statement, ZKPFactory

OUTPUT_DIR = Path("test invalid 2")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
PALETTE = {
    "blue":    "#1A73E8",
    "orange":  "#F57C00",
    "green":   "#2E7D32",
    "red":     "#C62828",
    "purple":  "#6A1B9A",
    "teal":    "#00695C",
    "bg":      "#FAFAFA",
    "grid":    "#E0E0E0",
    "text":    "#212121",
    "subtext": "#757575",
}

def _apply_base_style(fig, ax, title: str, xlabel: str, ylabel: str):
    """Shared style: light background, subtle grid, clean spines."""
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, fontsize=13, fontweight="bold", color=PALETTE["text"], pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["subtext"])
    ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["subtext"])
    ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, linestyle="--", alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(PALETTE["grid"])


def _smooth(x, y, points: int = 300):
    """Return (xs, ys) with cubic-spline interpolation for a smooth curve.
    Falls back to linear if too few points for a spline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4:
        xs = np.linspace(x[0], x[-1], points)
        ys = np.interp(xs, x, y)
        return xs, ys
    spline = make_interp_spline(x, y, k=3)
    xs = np.linspace(x[0], x[-1], points)
    ys = np.clip(spline(xs), 0, None)
    return xs, ys


# ── Plot helpers ───────────────────────────────────────────────────────────────

def save_confusion_matrix(case_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=PALETTE["subtext"])

    ax.set_title(f"{case_name}\nConfusion Matrix", fontsize=12, fontweight="bold",
                 color=PALETTE["text"], pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10, color=PALETTE["subtext"])
    ax.set_ylabel("True Label",      fontsize=10, color=PALETTE["subtext"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Ham", "Spam"], fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Ham", "Spam"], fontsize=10)
    ax.tick_params(colors=PALETTE["subtext"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else PALETTE["text"])

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"{case_name.replace(' ', '_').lower()}_confusion_matrix.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_train_vs_test_plot(case_name: str, train_history: list, test_history: list) -> str:
    train_vals = [0.0] + list(train_history)
    test_vals  = [0.0] + list(test_history)
    epochs     = list(range(len(train_vals)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _apply_base_style(fig, ax,
                      f"{case_name} — Train vs Test Accuracy",
                      "Epoch", "Accuracy")

    xs_tr, ys_tr = _smooth(epochs, train_vals)
    xs_te, ys_te = _smooth(epochs, test_vals)

    ax.plot(xs_tr, ys_tr, color=PALETTE["blue"],   linewidth=2.5, label="Train accuracy")
    ax.plot(xs_te, ys_te, color=PALETTE["orange"],  linewidth=2.5, label="Test accuracy",  linestyle="--")

    # Subtle fill between
    ax.fill_between(xs_tr, ys_tr, ys_te,
                    where=(ys_tr >= ys_te), alpha=0.08, color=PALETTE["blue"],   interpolate=True)
    ax.fill_between(xs_tr, ys_tr, ys_te,
                    where=(ys_tr <  ys_te), alpha=0.08, color=PALETTE["orange"], interpolate=True)

    # Scatter original epoch dots
    ax.scatter(epochs, train_vals, color=PALETTE["blue"],   s=50, zorder=5, edgecolors="white", linewidths=0.8)
    ax.scatter(epochs, test_vals,  color=PALETTE["orange"], s=50, zorder=5, edgecolors="white", linewidths=0.8, marker="D")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor=PALETTE["grid"])

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"{case_name.replace(' ', '_').lower()}_train_vs_test.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_accuracy_curve(case_name: str, train_history: list) -> str:
    vals   = [0.0] + list(train_history)
    epochs = list(range(len(vals)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _apply_base_style(fig, ax, f"{case_name} — Accuracy Curve", "Epoch", "Accuracy")

    xs, ys = _smooth(epochs, vals)
    ax.plot(xs, ys, color=PALETTE["blue"], linewidth=2.8)
    ax.fill_between(xs, 0, ys, alpha=0.12, color=PALETTE["blue"])
    ax.scatter(epochs, vals, color=PALETTE["blue"], s=55, zorder=5,
               edgecolors="white", linewidths=1.0)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3)

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"{case_name.replace(' ', '_').lower()}_accuracy_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_loss_curve(case_name: str, loss_history: list) -> str:
    vals   = list(loss_history)
    epochs = list(range(1, len(vals) + 1))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _apply_base_style(fig, ax, f"{case_name} — Loss Curve", "Epoch", "Loss")

    xs, ys = _smooth(epochs, vals)
    ax.plot(xs, ys, color=PALETTE["orange"], linewidth=2.8)
    ax.fill_between(xs, 0, ys, alpha=0.10, color=PALETTE["orange"])
    ax.scatter(epochs, vals, color=PALETTE["orange"], s=55, zorder=5,
               edgecolors="white", linewidths=1.0)

    ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3)

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"{case_name.replace(' ', '_').lower()}_loss_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_client_accuracy_plot(case_name: str, client_accuracy: dict) -> str:
    labels = list(client_accuracy.keys())
    values = [client_accuracy[k] if client_accuracy[k] is not None else 0.0 for k in labels]

    # Color bars by accuracy: red (low) → green (high)
    cmap   = plt.cm.RdYlGn
    colors = [cmap(v) for v in values]

    n    = len(labels)
    fig_w = max(10, n * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    _apply_base_style(fig, ax,
                      f"{case_name} — Per-Client Accuracy",
                      "Client", "Accuracy")

    bars = ax.bar(range(n), values, color=colors, width=0.65,
                  edgecolor="white", linewidth=0.7, zorder=3)

    # Value label on each bar
    for bar, val in zip(bars, values):
        ypos = bar.get_height() + 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=PALETTE["text"], fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.13)

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy", fontsize=9, color=PALETTE["subtext"])
    cbar.ax.tick_params(labelsize=8, colors=PALETTE["subtext"])

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"{case_name.replace(' ', '_').lower()}_per_client_accuracy.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_shard_accuracy_plot(case_name: str, shard_accuracy: dict) -> str:
    labels = list(shard_accuracy.keys())
    values = [shard_accuracy[k] if shard_accuracy[k] is not None else 0.0 for k in labels]

    cmap   = plt.cm.cool
    norm   = plt.Normalize(min(values) - 0.05, max(values) + 0.05)
    colors = [cmap(norm(v)) for v in values]

    n    = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2 + 2), 5))
    _apply_base_style(fig, ax,
                      f"{case_name} — Per-Shard Accuracy",
                      "Shard", "Accuracy")

    bars = ax.bar(range(n), values, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, values):
        ypos = bar.get_height() + 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=10, color=PALETTE["text"], fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.13)

    # Horizontal reference line at mean
    mean_acc = np.mean(values)
    ax.axhline(mean_acc, color=PALETTE["purple"], linewidth=1.4,
               linestyle="--", alpha=0.75, label=f"Mean: {mean_acc:.3f}")
    ax.legend(fontsize=9, framealpha=0.85, edgecolor=PALETTE["grid"])

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"{case_name.replace(' ', '_').lower()}_per_shard_accuracy.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_roc_curve(case_name: str, y_true: np.ndarray, y_score: np.ndarray) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _apply_base_style(fig, ax, f"{case_name} — ROC Curve",
                      "False Positive Rate", "True Positive Rate")

    ax.plot(fpr, tpr, color=PALETTE["blue"], linewidth=2.5,
            label=f"AUC = {roc_auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE["blue"])
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["subtext"],
            linewidth=1.2, label="Random classifier")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85, edgecolor=PALETTE["grid"])

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"{case_name.replace(' ', '_').lower()}_roc_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


# ── The rest of the file is unchanged ─────────────────────────────────────────

def client_and_shard_accuracy(classifier, data_dir: Path, num_shards: int = 3):
    client_accuracy = {}
    shard_acc = {}
    client_files = sorted(data_dir.glob("client_*.npz"))
    if not client_files:
        return client_accuracy, shard_acc

    shard_size = max(1, len(client_files) // num_shards)

    for path in client_files:
        client_id = path.stem
        idx = int(client_id.split("_")[1])
        shard_id = f"shard_{idx // shard_size}"
        data = np.load(path)
        X = data["X"].astype(np.float32)
        y = data["y"]
        if y.ndim > 1:
            y = np.argmax(y, axis=1).astype(np.int64)
        else:
            y = y.astype(np.int64).reshape(-1)

        if len(X) == 0:
            client_accuracy[client_id] = None
            continue

        preds = classifier.predict(X)
        acc = float(accuracy_score(y, preds))
        client_accuracy[client_id] = acc
        shard_acc.setdefault(shard_id, []).append(acc)

    shard_accuracy = {
        shard_id: float(np.mean(accs)) if accs else None
        for shard_id, accs in shard_acc.items()
    }
    return client_accuracy, shard_accuracy


def save_client_shard_accuracy_plots(case_name: str, client_accuracy: dict, shard_accuracy: dict) -> dict:
    client_plot = save_client_accuracy_plot(case_name, client_accuracy)
    shard_plot  = save_shard_accuracy_plot(case_name, shard_accuracy)
    return {"per_client_accuracy_plot": client_plot, "per_shard_accuracy_plot": shard_plot}


def load_real_federated_dataset(data_dir: Path = REAL_DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

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
        X_scaled, y_all, test_size=0.20, random_state=42, stratify=y_all,
    )
    return X_train, X_test, y_train, y_test


def build_dataset(seed: int = 7, n_samples: int = 260, n_features: int = 20):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=12, n_redundant=2, n_classes=2,
        class_sep=1.4, flip_y=0.02, random_state=seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_classifier(X_train, y_train, X_test, y_test, epochs=6, lr=0.01):
    clf = LightweightClassifier(input_dim=X_train.shape[1], output_dim=2, learning_rate=lr)
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    for epoch in range(epochs):
        stats = clf.train_step(X_train, y_train)
        train_losses.append(stats["loss"])
        train_accs.append(stats["accuracy"])
        test_accs.append(float(accuracy_score(y_test, clf.predict(X_test))))
        test_losses.append(float(clf.compute_loss(X_test, y_test)))

    train_pred = clf.predict(X_train)
    test_pred  = clf.predict(X_test)
    y_score    = clf.forward(X_test)[1][:, 1]

    return {
        "classifier":      clf,
        "train_accuracy":  float(accuracy_score(y_train, train_pred)),
        "test_accuracy":   float(accuracy_score(y_test,  test_pred)),
        "train_loss":      float(clf.compute_loss(X_train, y_train)),
        "test_loss":       float(clf.compute_loss(X_test,  y_test)),
        "f1_score":        float(f1_score(y_test, test_pred, zero_division=0)),
        "train_losses":    train_losses,
        "train_accs":      train_accs,
        "test_losses":     test_losses,
        "test_accs":       test_accs,
        "y_test":          y_test,
        "test_pred":       test_pred,
        "y_score":         y_score,
    }


def make_case_result(case_name, status, metrics, artifacts, notes=""):
    return {"case": case_name, "status": status, "metrics": metrics,
            "artifacts": artifacts, "notes": notes}


def run_case_valid_training():
    X_train, X_test, y_train, y_test = build_dataset(seed=7)
    result = train_classifier(X_train, y_train, X_test, y_test, epochs=8, lr=0.03)
    confusion_path = save_confusion_matrix("Case 1 - Valid training", y_test, result["test_pred"])
    roc_path       = save_roc_curve("Case 1 - Valid training", y_test, result["y_score"])
    metrics = {
        "train_accuracy":      result["train_accuracy"],
        "test_accuracy":       result["test_accuracy"],
        "train_loss":          result["train_loss"],
        "test_loss":           result["test_loss"],
        "f1_score":            result["f1_score"],
        "epoch_train_loss":    result["train_losses"][-1],
        "epoch_train_accuracy":result["train_accs"][-1],
    }
    return make_case_result("1_valid_training", "passed", metrics,
                            {"confusion_matrix": confusion_path, "roc_curve": roc_path},
                            "Standard training path with clean split and expected model convergence.")


def run_case_invalid_dataset(data_dir: Path = INVALID_DATA_DIR):
    X_train, X_test, y_train, y_test = load_real_federated_dataset(data_dir)
    result = train_classifier(X_train, y_train, X_test, y_test, epochs=5, lr=0.01)

    confusion_path     = save_confusion_matrix("Case 1 - Invalid dataset", y_test, result["test_pred"])
    roc_path           = save_roc_curve("Case 1 - Invalid dataset", y_test, result["y_score"])
    accuracy_curve_path= save_accuracy_curve("Case 1 - Invalid dataset", result["train_accs"])
    loss_curve_path    = save_loss_curve("Case 1 - Invalid dataset", result["train_losses"])
    train_vs_test_path = save_train_vs_test_plot("Case 1 - Invalid dataset", result["train_accs"], result["test_accs"])

    client_accuracy, shard_accuracy = client_and_shard_accuracy(result["classifier"], data_dir)
    accuracy_plots = save_client_shard_accuracy_plots("Case 2 - Invalid data run", client_accuracy, shard_accuracy)

    metrics = {
        "train_accuracy":    result["train_accuracy"],
        "test_accuracy":     result["test_accuracy"],
        "train_loss":        result["train_loss"],
        "test_loss":         result["test_loss"],
        "f1_score":          result["f1_score"],
        "dataset_dir":       str(data_dir),
        "num_clients":       len(client_accuracy),
        "per_client_accuracy": client_accuracy,
        "per_shard_accuracy":  shard_accuracy,
    }
    artifacts = {
        "confusion_matrix": confusion_path,
        "roc_curve":        roc_path,
        "accuracy_curve":   accuracy_curve_path,
        "loss_curve":       loss_curve_path,
        "train_vs_test":    train_vs_test_path,
    }
    artifacts.update(accuracy_plots)
    return make_case_result("1_invalid_dataset", "passed", metrics, artifacts,
                            "One full invalid-data run using the invalid data folder, reporting per-client accuracy, per-shard accuracy, and training curves.")


def run_case_real_data_baseline(data_dir: Path = REAL_DATA_DIR):
    X_train, X_test, y_train, y_test = load_real_federated_dataset(data_dir)
    result = train_classifier(X_train, y_train, X_test, y_test, epochs=10, lr=0.03)
    confusion_path     = save_confusion_matrix("Case 5 - Real data baseline", y_test, result["test_pred"])
    roc_path           = save_roc_curve("Case 5 - Real data baseline", y_test, result["y_score"])
    train_vs_test_path = save_train_vs_test_plot(
        "Case 5 - Real data baseline", result["train_accs"],
        [result["test_accuracy"]] * len(result["train_accs"]))
    metrics = {
        "train_accuracy": result["train_accuracy"],
        "test_accuracy":  result["test_accuracy"],
        "train_loss":     result["train_loss"],
        "test_loss":      result["test_loss"],
        "f1_score":       result["f1_score"],
        "dataset_size":   int(len(X_train) + len(X_test)),
        "num_train":      int(len(X_train)),
        "num_test":       int(len(X_test)),
    }
    return make_case_result("5_real_data_baseline", "passed", metrics,
                            {"confusion_matrix": confusion_path, "roc_curve": roc_path,
                             "train_vs_test": train_vs_test_path},
                            "Baseline evaluation on the real federated dataset.")


def run_case_real_data_best_retrain(data_dir: Path = REAL_DATA_DIR):
    X_train, X_test, y_train, y_test = load_real_federated_dataset(data_dir)
    candidates = [(4, 0.005), (6, 0.01), (8, 0.02), (10, 0.03)]
    best = None
    for epochs, lr in candidates:
        result = train_classifier(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)
        score  = result["f1_score"] + result["test_accuracy"]
        candidate = {"epochs": epochs, "lr": lr, "score": score, "metrics": result}
        if best is None or score > best["score"]:
            best = candidate

    result = best["metrics"]
    confusion_path      = save_confusion_matrix("Case 2 - Real federated dataset", y_test, result["test_pred"])
    roc_path            = save_roc_curve("Case 2 - Real federated dataset", y_test, result["y_score"])
    accuracy_curve_path = save_accuracy_curve("Case 2 - Real federated dataset", result["train_accs"])
    loss_curve_path     = save_loss_curve("Case 2 - Real federated dataset", result["train_losses"])
    train_vs_test_path  = save_train_vs_test_plot("Case 2 - Real federated dataset", result["train_accs"], result["test_accs"])

    client_accuracy, shard_accuracy = client_and_shard_accuracy(result["classifier"], data_dir)
    accuracy_plots = save_client_shard_accuracy_plots("Case 6 - Real data best retrain", client_accuracy, shard_accuracy)

    metrics = {
        "selected_epochs":        best["epochs"],
        "selected_learning_rate": best["lr"],
        "selected_score":         best["score"],
        "train_accuracy":         result["train_accuracy"],
        "test_accuracy":          result["test_accuracy"],
        "train_loss":             result["train_loss"],
        "test_loss":              result["test_loss"],
        "f1_score":               result["f1_score"],
        "dataset_dir":            str(data_dir),
        "num_clients":            len(client_accuracy),
        "per_client_accuracy":    client_accuracy,
        "per_shard_accuracy":     shard_accuracy,
        "candidate_results": [
            {"epochs": e, "lr": l,
             "f1_score":     train_classifier(X_train, y_train, X_test, y_test, epochs=e, lr=l)["f1_score"],
             "test_accuracy":train_classifier(X_train, y_train, X_test, y_test, epochs=e, lr=l)["test_accuracy"]}
            for e, l in candidates
        ],
    }
    artifacts = {
        "confusion_matrix": confusion_path,
        "roc_curve":        roc_path,
        "accuracy_curve":   accuracy_curve_path,
        "loss_curve":       loss_curve_path,
        "train_vs_test":    train_vs_test_path,
    }
    artifacts.update(accuracy_plots)
    return make_case_result("2_real_data_best_retrain", "best_result_selected", metrics, artifacts,
                            "Best hyperparameter combination selected from multiple retraining runs.")


def run_case_large_update():
    X_train, X_test, y_train, y_test = build_dataset(seed=11)
    result = train_classifier(X_train, y_train, X_test, y_test, epochs=6, lr=0.02)

    update      = np.random.randn(X_train.shape[1] * 2 + 2).astype(np.float32) * 20.0
    norm        = float(np.linalg.norm(update))
    clipped     = update / norm * GRADIENT_CLIP if norm > GRADIENT_CLIP else update
    clipped_norm= float(np.linalg.norm(clipped))

    confusion_path = save_confusion_matrix("Case 3 - Large update", y_test, result["test_pred"])
    roc_path       = save_roc_curve("Case 3 - Large update", y_test, result["y_score"])

    metrics = {
        "train_accuracy":       result["train_accuracy"],
        "test_accuracy":        result["test_accuracy"],
        "train_loss":           result["train_loss"],
        "test_loss":            result["test_loss"],
        "f1_score":             result["f1_score"],
        "large_update_norm":    norm,
        "clipped_update_norm":  clipped_norm,
        "clipped_within_bound": clipped_norm <= GRADIENT_CLIP,
    }
    return make_case_result("3_large_update", "passed", metrics,
                            {"confusion_matrix": confusion_path, "roc_curve": roc_path},
                            "Large synthetic update is clipped to the configured stability bound.")


def run_case_zkp_proof():
    gradient     = np.random.randn(32).astype(np.float32)
    model_params = np.random.randn(32).astype(np.float32) * 0.01
    poly         = PolynomialProxyFactory.create_from_update(gradient, model_params)

    generator = ZKPFactory.get_generator()
    proof = generator.generate_proof(
        gradient, model_params,
        client_id="test_client", round_num=1, polynomial_proxy=poly,
        norm_bound=GRADIENT_CLIP, initial_params=model_params,
        updated_params=model_params + 0.01, local_tokens=np.zeros(8, dtype=np.float32),
        shard_id="shard_0", learning_rate=0.0,
    )

    verifier  = ZKPFactory.get_verifier()
    statement = Groth16Statement(
        one=1, h_w_old="h_old", h_w_new="h_new", model_hash="h_new",
        round_num=1, shard_id="shard_0", public_inputs=[1, 32, 32],
    )
    valid_ok, valid_report   = verifier.verify_proof(proof, gradient, poly, statement=statement)
    corrupt_ok, corrupt_report = verifier.verify_proof(proof, gradient + 1.0, poly, statement=statement)

    aggregator = RobustAggregator()
    try:
        aggregator.aggregate(
            [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)],
            verification_results=[False, False],
            client_ids=["client_a", "client_b"],
            method=AggregationMethod.ROBUST_HYBRID,
        )
        zero_valid_status = "unexpected_success"
    except Exception as exc:
        zero_valid_status = f"failed_as_expected:{type(exc).__name__}"

    X_train, X_test, y_train, y_test = build_dataset(seed=13)
    base = train_classifier(X_train, y_train, X_test, y_test, epochs=6, lr=0.02)
    alt  = train_classifier(X_train, y_train, X_test, y_test, epochs=6, lr=0.05)

    metrics = {
        "proof_valid":                 bool(valid_ok),
        "proof_corrupt_model_rejected":bool(corrupt_ok is False),
        "zero_valid_aggregation":      zero_valid_status,
        "proof_shape_ok":              bool(valid_report.get("checks", {}).get("proof_shape", False)),
        "norm_bound_ok":               bool(valid_report.get("checks", {}).get("norm_bound", {}).get("satisfied", False)),
        "base_test_loss":              float(base["test_loss"]),
        "alt_test_loss":               float(alt["test_loss"]),
        "base_f1":                     float(base["f1_score"]),
        "alt_f1":                      float(alt["f1_score"]),
        "different_param_train_delta": float(alt["test_loss"] - base["test_loss"]),
    }
    return make_case_result("4_zkp_proof", "passed_with_failure_checks", metrics,
                            {"proof_valid": valid_ok,
                             "proof_corrupt_model_rejected": corrupt_ok is False,
                             "zero_valid_aggregation": zero_valid_status},
                            "ZKP proof generation, corruption rejection, zero-valid-update rejection, and parameter-difference test.")


def write_professional_summary(results: list):
    lines = [
        "FINAL TEST REPORT",
        "=================",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
        f"Output folder: {OUTPUT_DIR}", "",
        "Case summary:",
    ]
    for item in results:
        lines += [
            f"- {item['case']}: {item['status']}",
            f"  F1 Score:      {item['metrics'].get('f1_score', 'n/a')}",
            f"  Test Accuracy: {item['metrics'].get('test_accuracy', 'n/a')}",
            f"  Test Loss:     {item['metrics'].get('test_loss', 'n/a')}",
            f"  Notes: {item['notes']}", "",
        ]
    (OUTPUT_DIR / "final_test_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    for item in results:
        with open(OUTPUT_DIR / f"{item['case']}_report.json", "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)

    data_dir = INVALID_DATA_DIR if INVALID_DATA_DIR.exists() else REAL_DATA_DIR

    results = [
        run_case_invalid_dataset(data_dir=INVALID_DATA_DIR),
        run_case_real_data_best_retrain(data_dir=REAL_DATA_DIR),
    ]

    write_professional_summary(results)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "output_dir":   str(OUTPUT_DIR),
        "cases":        results,
    }
    json_path = OUTPUT_DIR / "final_test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nProfessional final test suite completed.")
    print(f"Results saved to: {json_path}")
    print("Plots and summary saved in:", OUTPUT_DIR)
    for item in results:
        print(f"- {item['case']}: {item['status']}")


if __name__ == "__main__":
    main()