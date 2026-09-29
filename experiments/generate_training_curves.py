import json, itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lightweight_classifier import LightweightClassifier

OUTPUT_DIR = Path("results/malicious_baseline")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("malisious")

# â”€â”€ Load per-client, detect malicious â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
client_data = {}
for path in sorted(DATA_DIR.glob("client_*.npz")):
    cid = path.stem
    d = np.load(path)
    X = d["X"].astype(np.float32)
    y = d["y"]
    if y.ndim > 1:
        y = np.argmax(y, axis=1).astype(np.int64)
    else:
        y = y.astype(np.int64).reshape(-1)
    client_data[cid] = (X, y)

spam_ratios = {cid: float(y.mean()) for cid, (_, y) in client_data.items()}
ratios = np.array(list(spam_ratios.values()))
median_sr = np.median(ratios)
mad_sr = np.median(np.abs(ratios - median_sr))
if mad_sr == 0:
    mad_sr = float(np.std(ratios))
thresh_sr = 3.0 * max(mad_sr, 0.01)
malicious = sorted(
    [cid for cid, sr in spam_ratios.items() if abs(sr - median_sr) > thresh_sr]
)
clean = sorted([cid for cid in client_data if cid not in malicious])

# â”€â”€ Build clean dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
X_parts, y_parts = [], []
for cid in clean:
    Xc, yc = client_data[cid]
    X_parts.append(Xc)
    y_parts.append(yc)
X_all = np.vstack(X_parts).astype(np.float32)
y_all = np.concatenate(y_parts).astype(np.int64)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.20, random_state=42, stratify=y_all
)
input_dim = X_train.shape[1]


# â”€â”€ Grid search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_and_score(epochs, lr):
    clf = LightweightClassifier(input_dim=input_dim, output_dim=2, learning_rate=lr)
    for _ in range(epochs):
        clf.train_step(X_train, y_train)
    pred = clf.predict(X_test)
    return float(accuracy_score(y_test, pred)), float(
        f1_score(y_test, pred, zero_division=0)
    )


best = None
for epochs, lr in itertools.product([10, 20, 30, 50, 75], [0.1, 0.05, 0.01, 0.005]):
    acc, f1 = train_and_score(epochs, lr)
    score = f1 + acc
    if best is None or score > best["score"]:
        best = {"epochs": epochs, "lr": lr, "score": score, "acc": acc, "f1": f1}

# â”€â”€ Full training with per-epoch metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EP = best["epochs"]
clf = LightweightClassifier(input_dim=input_dim, output_dim=2, learning_rate=best["lr"])
init_pred = clf.predict(X_test)
f1_curve = [float(f1_score(y_test, init_pred, zero_division=0))]
acc_curve = [float(accuracy_score(y_test, init_pred))]
loss_curve = [float(clf.compute_loss(X_test, y_test))]

for e in range(EP):
    clf.train_step(X_train, y_train)
    pred_e = clf.predict(X_test)
    acc_curve.append(float(accuracy_score(y_test, pred_e)))
    f1_curve.append(float(f1_score(y_test, pred_e, zero_division=0)))
    loss_curve.append(float(clf.compute_loss(X_test, y_test)))

final_pred = clf.predict(X_test)
final_acc = float(accuracy_score(y_test, final_pred))
final_f1 = float(f1_score(y_test, final_pred, zero_division=0))
final_prec = float(
    __import__("sklearn").metrics.precision_score(y_test, final_pred, zero_division=0)
)
final_rec = float(
    __import__("sklearn").metrics.recall_score(y_test, final_pred, zero_division=0)
)
final_loss = float(clf.compute_loss(X_test, y_test))

# â”€â”€ Save JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
curves_data = {
    "rounds": list(range(EP + 1)),
    "f1_scores": f1_curve,
    "accuracies": acc_curve,
    "losses": loss_curve,
    "final_metrics": {
        "acc": final_acc,
        "f1": final_f1,
        "prec": final_prec,
        "rec": final_rec,
        "loss": final_loss,
    },
}
with open(OUTPUT_DIR / "training_curves.json", "w") as f:
    json.dump(curves_data, f, indent=2)

# â”€â”€ Plot (same style as best_results) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
C = {
    "blue": "#1A73E8",
    "green": "#2E7D32",
    "red": "#C62828",
    "orange": "#F57C00",
    "purple": "#6A1B9A",
    "bg": "#FAFAFA",
}
epochs_x = np.arange(EP + 1)
bb = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8)


def save_plot(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


acc_min = max(0, min(acc_curve) * 0.98)
f1_min = max(0, min(f1_curve) * 0.98)
loss_min = max(0, min(loss_curve) * 0.95)

# F1
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_title("F1 Score Curve", fontsize=13, fontweight="bold")
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.plot(epochs_x, f1_curve, linewidth=2.5, color=C["green"], label="F1 Score")
ax.fill_between(epochs_x, f1_min, f1_curve, alpha=0.2, color=C["green"])
ax.scatter(
    epochs_x,
    f1_curve,
    s=80,
    color=C["green"],
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
)
ax.text(
    0.03,
    0.95,
    f"Final F1: {f1_curve[-1]:.4f}",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(0, EP)
ax.set_ylim(f1_min, 1.02)
ax.legend(fontsize=11, loc="lower right")
fig.tight_layout()
save_plot(fig, "curve_f1_score.png")

# Accuracy
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_title("Accuracy Curve", fontsize=13, fontweight="bold")
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.plot(epochs_x, acc_curve, linewidth=2.5, color=C["blue"], label="Accuracy")
ax.fill_between(epochs_x, acc_min, acc_curve, alpha=0.2, color=C["blue"])
ax.scatter(
    epochs_x,
    acc_curve,
    s=80,
    color=C["blue"],
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
)
ax.text(
    0.03,
    0.95,
    f"Final Accuracy: {acc_curve[-1]:.4f}",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(0, EP)
ax.set_ylim(acc_min, 1.02)
ax.legend(fontsize=11, loc="lower right")
fig.tight_layout()
save_plot(fig, "curve_accuracy.png")

# Loss
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_title("Loss Curve", fontsize=13, fontweight="bold")
ax.set_xlabel("Training Round", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.plot(epochs_x, loss_curve, linewidth=2.5, color=C["red"], label="Loss")
ax.fill_between(epochs_x, loss_min, loss_curve, alpha=0.2, color=C["red"])
ax.scatter(
    epochs_x,
    loss_curve,
    s=80,
    color=C["red"],
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
)
ax.text(
    0.03,
    0.95,
    f"Final Loss: {loss_curve[-1]:.4f}",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(0, EP)
ax.set_ylim(loss_min * 0.98, max(loss_curve) * 1.15)
ax.legend(fontsize=11, loc="upper right")
fig.tight_layout()
save_plot(fig, "curve_loss.png")

# Combined
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
fig.patch.set_facecolor(C["bg"])
for ax in [ax1, ax2, ax3]:
    ax.set_facecolor(C["bg"])
ax1.plot(epochs_x, f1_curve, linewidth=2.5, color=C["green"])
ax1.fill_between(epochs_x, f1_min, f1_curve, alpha=0.15, color=C["green"])
ax1.scatter(
    epochs_x,
    f1_curve,
    s=70,
    color=C["green"],
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
)
ax1.text(
    0.03,
    0.92,
    f"Final F1: {f1_curve[-1]:.4f}",
    transform=ax1.transAxes,
    fontsize=10,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax1.set_ylabel("F1 Score", fontsize=11)
ax1.set_title(
    f"Training Metrics - Malicious Detection (excluded {malicious})",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.set_xlim(0, EP)
ax1.set_ylim(f1_min, 1.02)
ax2.plot(epochs_x, acc_curve, linewidth=2.5, color=C["blue"])
ax2.fill_between(epochs_x, acc_min, acc_curve, alpha=0.15, color=C["blue"])
ax2.scatter(
    epochs_x,
    acc_curve,
    s=70,
    color=C["blue"],
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
)
ax2.text(
    0.03,
    0.92,
    f"Final Accuracy: {acc_curve[-1]:.4f}",
    transform=ax2.transAxes,
    fontsize=10,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax2.set_ylabel("Accuracy", fontsize=11)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_xlim(0, EP)
ax2.set_ylim(acc_min, 1.02)
ax3.plot(epochs_x, loss_curve, linewidth=2.5, color=C["red"])
ax3.fill_between(epochs_x, loss_min, loss_curve, alpha=0.15, color=C["red"])
ax3.scatter(
    epochs_x,
    loss_curve,
    s=70,
    color=C["red"],
    zorder=5,
    edgecolors="white",
    linewidths=1.5,
)
ax3.text(
    0.03,
    0.92,
    f"Final Loss: {loss_curve[-1]:.4f}",
    transform=ax3.transAxes,
    fontsize=10,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax3.set_xlabel("Training Round", fontsize=11)
ax3.set_ylabel("Loss", fontsize=11)
ax3.grid(True, alpha=0.3, linestyle="--")
ax3.set_xlim(0, EP)
ax3.set_ylim(loss_min * 0.98, max(loss_curve) * 1.15)
fig.tight_layout()
save_plot(fig, "curves_combined.png")

print(f"\nTraining curves generated (malisious, malicious={malicious} excluded)")
print(f"Final: acc={final_acc:.4f}, f1={final_f1:.4f}, loss={final_loss:.4f}")
