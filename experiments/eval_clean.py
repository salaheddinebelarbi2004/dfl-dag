import json, time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from client import NeuralNetworkSimulator
from federated_learning_config import (
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    GRADIENT_CLIP,
)

DATA_DIR = Path("real_federated_data")
OUTPUT_DIR = Path("results/clean_baseline")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("CLEAN FEDERATED LEARNING EVALUATION")
print("=" * 80)

# 1. Load pre-trained global model
print("\n[1/8] Loading pre-trained global model...")
pretrained_weights = np.load("global_model.npy")
nn = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
nn.set_params(pretrained_weights)
print(f"  global_model.npy ({len(pretrained_weights)} params, {type(nn).__name__})")

# 2. Load clean data per client
print("\n[2/8] Loading real federated data per client...")
client_data = {}
for path in sorted(DATA_DIR.glob("client_*.npz")):
    cid = path.stem
    data = np.load(path)
    X, y = data["X"].astype(np.float64), data["y"].astype(np.float64)
    spam_pct = y.mean() * 100
    print(f"  {cid}: {len(X)} samples, spam={spam_pct:.1f}%")
    client_data[cid] = (X, y)

# 3. Prepare evaluation set (all clients are clean)
print("\n[3/8] Preparing evaluation set...")
all_X, all_y = [], []
for cid, (Xc, yc) in client_data.items():
    all_X.append(Xc)
    all_y.append(yc)
all_X = np.concatenate(all_X)
all_y = np.concatenate(all_y)

scaler = StandardScaler()
all_X_s = scaler.fit_transform(all_X)

X_train, X_test, y_train, y_test = train_test_split(
    all_X_s, all_y, test_size=0.2, random_state=42, stratify=all_y
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# 4. Initial evaluation
print("\n[4/8] Initial evaluation (round 0)...")


def evaluate(model, X_eval, y_eval):
    probs, _ = model.forward(X_eval)
    preds = np.argmax(probs, axis=1)
    loss = -np.mean(
        np.log(probs[np.arange(len(y_eval)), y_eval.astype(int).ravel()] + 1e-8)
    )
    return {
        "accuracy": float(accuracy_score(y_eval, preds)),
        "f1": float(f1_score(y_eval, preds)),
        "loss": float(loss),
        "preds": preds,
        "probs": probs,
    }


local = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
local.set_params(pretrained_weights)
init_eval = evaluate(local, X_test, y_test)
print(
    f"  Accuracy: {init_eval['accuracy']:.4f} | F1: {init_eval['f1']:.4f} | Loss: {init_eval['loss']:.4f}"
)

# 5. Training history
NUM_ROUNDS = 5
rounds_x = list(range(NUM_ROUNDS + 1))
acc_curve = [init_eval["accuracy"]]
f1_curve = [init_eval["f1"]]
loss_curve = [init_eval["loss"]]
reputation_history = {cid: [0.8] for cid in client_data}

current_weights = pretrained_weights.copy()

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n  === Round {rnd} ===")
    updates = {}
    update_norms = {}

    for cid, (Xc, yc) in client_data.items():
        local = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        local.set_params(current_weights)
        Xc_s = scaler.transform(Xc)

        n = len(Xc_s)
        batch_size = 32
        n_batches = (n + batch_size - 1) // batch_size
        for epoch in range(5):
            idx = np.random.permutation(n)
            Xs, ys = Xc_s[idx], yc[idx]
            for b in range(n_batches):
                s, e = b * batch_size, min((b + 1) * batch_size, n)
                grads = local.backward(Xs[s:e], ys[s:e])
                local.update_params(grads, learning_rate=0.05)

        update = local.get_params() - current_weights
        updates[cid] = update
        update_norms[cid] = float(np.linalg.norm(update))

    # Simple averaging (all clients are clean)
    all_cids = sorted(client_data.keys())
    update_arr = np.array([updates[cid] for cid in all_cids])
    avg_update = np.mean(update_arr, axis=0)
    current_weights = current_weights + avg_update

    # Evaluate
    global_model = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    global_model.set_params(current_weights)
    eval_res = evaluate(global_model, X_test, y_test)

    acc_curve.append(eval_res["accuracy"])
    f1_curve.append(eval_res["f1"])
    loss_curve.append(eval_res["loss"])

    print(
        f"    Accuracy: {eval_res['accuracy']:.4f} | F1: {eval_res['f1']:.4f} | Loss: {eval_res['loss']:.4f}"
    )

# 7. Final evaluation
print("\n[7/8] Final evaluation...")
final_probs, _ = global_model.forward(X_test)
final_preds = np.argmax(final_probs, axis=1)
final_acc = accuracy_score(y_test, final_preds)
final_prec = precision_score(y_test, final_preds)
final_rec = recall_score(y_test, final_preds)
final_f1 = f1_score(y_test, final_preds)
final_cm = confusion_matrix(y_test, final_preds)
fpr, tpr, _ = roc_curve(y_test, final_probs[:, 1])
final_auc = auc(fpr, tpr)

print(f"  Accuracy:  {final_acc:.4f}")
print(f"  Precision: {final_prec:.4f}")
print(f"  Recall:    {final_rec:.4f}")
print(f"  F1 Score:  {final_f1:.4f}")
print(f"  ROC-AUC:   {final_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"          Pred Ham  Pred Spam")
print(f"  True Ham  {final_cm[0, 0]:5d}     {final_cm[0, 1]:5d}")
print(f"  True Spam  {final_cm[1, 0]:5d}     {final_cm[1, 1]:5d}")

# 8. Generate plots
print("\n[8/8] Generating plots...")
C = {
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "orange": "#ff7f0e",
    "bg": "#f8f9fa",
    "text": "#2c3e50",
}
bb = dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.2)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", color=C["text"], pad=15)
    ax.set_xlabel(xlabel, fontsize=12, color=C["text"])
    ax.set_ylabel(ylabel, fontsize=12, color=C["text"])
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors=C["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_plot(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved {name}")


EP = NUM_ROUNDS
acc_lo = max(0, min(acc_curve) * 0.98)
f1_lo = max(0, min(f1_curve) * 0.98)
loss_hi = max(loss_curve) * 1.15

# Accuracy
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(ax, "Global Model Accuracy - Clean Federated Learning", "Round", "Accuracy")
ax.plot(
    rounds_x,
    acc_curve,
    linewidth=2.5,
    marker="o",
    markersize=8,
    color=C["blue"],
    label="Accuracy",
)
ax.fill_between(rounds_x, acc_lo, acc_curve, alpha=0.12, color=C["blue"])
for x, y in zip(rounds_x, acc_curve):
    ax.text(
        x, y - 0.012, f"{y:.4f}", ha="center", va="top", fontsize=9, fontweight="bold"
    )
ax.set_xlim(-0.3, EP + 0.3)
ax.set_ylim(acc_lo, 1.0)
ax.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
ax.text(
    0.02,
    0.95,
    f"Pre-trained broadcast ({init_eval['accuracy']:.4f})",
    transform=ax.transAxes,
    fontsize=9,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax.legend(loc="lower left", fontsize=10)
save_plot(fig, "clean_accuracy_curve.png")

# F1
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(ax, "Global Model F1 Score - Clean Federated Learning", "Round", "F1 Score")
ax.plot(
    rounds_x,
    f1_curve,
    linewidth=2.5,
    marker="s",
    markersize=8,
    color=C["green"],
    label="F1 Score",
)
ax.fill_between(rounds_x, f1_lo, f1_curve, alpha=0.12, color=C["green"])
for x, y in zip(rounds_x, f1_curve):
    ax.text(
        x, y - 0.012, f"{y:.4f}", ha="center", va="top", fontsize=9, fontweight="bold"
    )
ax.set_xlim(-0.3, EP + 0.3)
ax.set_ylim(f1_lo, 1.0)
ax.legend(loc="lower left", fontsize=10)
save_plot(fig, "clean_f1_curve.png")

# Loss
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(ax, "Global Model Loss - Clean Federated Learning", "Round", "Loss")
ax.plot(
    rounds_x,
    loss_curve,
    linewidth=2.5,
    marker="^",
    markersize=8,
    color=C["red"],
    label="Loss",
)
loss_lo = max(0, min(loss_curve) * 0.95)
ax.fill_between(rounds_x, loss_lo, loss_curve, alpha=0.10, color=C["red"])
for x, y in zip(rounds_x, loss_curve):
    ax.text(
        x,
        y + loss_hi * 0.02,
        f"{y:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
ax.set_xlim(-0.3, EP + 0.3)
ax.set_ylim(loss_lo, loss_hi)
ax.legend(loc="upper right", fontsize=10)
save_plot(fig, "clean_loss_curve.png")

# Combined
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(C["bg"])
for ax in (ax1, ax2, ax3):
    ax.set_facecolor(C["bg"])

style(ax1, "Accuracy", "Round", "Accuracy")
ax1.plot(rounds_x, acc_curve, linewidth=2.5, marker="o", color=C["blue"])
ax1.fill_between(rounds_x, acc_lo, acc_curve, alpha=0.12, color=C["blue"])
ax1.set_xlim(-0.3, EP + 0.3)
ax1.set_ylim(acc_lo, 1.0)

style(ax2, "F1 Score", "Round", "F1")
ax2.plot(rounds_x, f1_curve, linewidth=2.5, marker="s", color=C["green"])
ax2.fill_between(rounds_x, f1_lo, f1_curve, alpha=0.12, color=C["green"])
ax2.set_xlim(-0.3, EP + 0.3)
ax2.set_ylim(f1_lo, 1.0)

style(ax3, "Loss", "Round", "Loss")
ax3.plot(rounds_x, loss_curve, linewidth=2.5, marker="^", color=C["red"])
ax3.fill_between(rounds_x, loss_lo, loss_curve, alpha=0.10, color=C["red"])
ax3.set_xlim(-0.3, EP + 0.3)
ax3.set_ylim(loss_lo, loss_hi)

fig.suptitle(
    "Clean Federated Learning - Training Curves",
    fontsize=16,
    fontweight="bold",
    color=C["text"],
)
plt.tight_layout()
save_plot(fig, "clean_curves_combined.png")

# ROC curve
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(
    ax,
    f"ROC Curve (AUC = {final_auc:.4f}) - Clean FL",
    "False Positive Rate",
    "True Positive Rate",
)
ax.plot(fpr, tpr, linewidth=2.5, color=C["blue"], label=f"ROC (AUC = {final_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
ax.fill_between(fpr, tpr, alpha=0.12, color=C["blue"])
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower right", fontsize=11)
save_plot(fig, "clean_roc_curve.png")

# Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
im = ax.imshow(final_cm, cmap="Blues", vmin=0, vmax=final_cm.max() * 1.1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Ham", "Pred Spam"], fontsize=11)
ax.set_yticklabels(["True Ham", "True Spam"], fontsize=11)
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(final_cm[i, j]),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="white" if final_cm[i, j] > final_cm.max() / 2 else "black",
        )
ax.set_title(
    "Confusion Matrix - Clean FL",
    fontsize=14,
    fontweight="bold",
    color=C["text"],
    pad=15,
)
plt.colorbar(im, ax=ax, shrink=0.8)
save_plot(fig, "clean_confusion_matrix.png")

# Metrics bar chart
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
metrics_vals = [final_acc, final_prec, final_rec, final_f1, final_auc]
colors = [C["blue"], C["green"], C["orange"], C["purple"], C["red"]]
bars = ax.bar(
    metrics_names,
    metrics_vals,
    color=colors,
    edgecolor="white",
    linewidth=1.2,
    width=0.6,
)
for bar, val in zip(bars, metrics_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=C["text"],
    )
ax.set_ylim(0, 1.08)
ax.set_title(
    "Final Metrics - Clean Federated Learning",
    fontsize=14,
    fontweight="bold",
    color=C["text"],
    pad=15,
)
ax.set_ylabel("Score", fontsize=12, color=C["text"])
ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save_plot(fig, "clean_metrics_bar.png")

# Save metrics JSON
metrics_data = {
    "initial": {
        "accuracy": init_eval["accuracy"],
        "f1": init_eval["f1"],
        "loss": init_eval["loss"],
    },
    "final": {
        "accuracy": float(final_acc),
        "f1": float(final_f1),
        "precision": float(final_prec),
        "recall": float(final_rec),
        "roc_auc": float(final_auc),
        "loss": float(loss_curve[-1]),
        "confusion_matrix": {
            "tn": int(final_cm[0, 0]),
            "fp": int(final_cm[0, 1]),
            "fn": int(final_cm[1, 0]),
            "tp": int(final_cm[1, 1]),
        },
    },
    "rounds": [
        {"round": i, "accuracy": acc_curve[i], "f1": f1_curve[i], "loss": loss_curve[i]}
        for i in range(NUM_ROUNDS + 1)
    ],
    "config": {
        "num_rounds": NUM_ROUNDS,
        "model": "global_model.npy",
        "data": "real_federated_data",
        "aggregation": "simple_averaging (clean data)",
        "clients": len(client_data),
    },
}
with open(OUTPUT_DIR / "clean_metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)

# Summary text
summary = f"""
{"=" * 80}
CLEAN FEDERATED LEARNING EVALUATION - RESULTS
{"=" * 80}

DATA: real_federated_data/ ({len(client_data)} clients, all clean)
MODEL: Pre-trained global_model.npy

RESULTS:
  Metric           Round 0 (broadcast)    Round 5 (final)    Change
  -------          -------------------    -----------------  --------
  Accuracy         {init_eval["accuracy"]:.4f}                 {final_acc:.4f}                  {final_acc - init_eval["accuracy"]:+.4f}
  Precision        N/A                     {final_prec:.4f}                  N/A
  Recall           N/A                     {final_rec:.4f}                  N/A
  F1 Score         {init_eval["f1"]:.4f}                 {final_f1:.4f}                  {final_f1 - init_eval["f1"]:+.4f}
  Loss             {init_eval["loss"]:.4f}                 {loss_curve[-1]:.4f}                  {loss_curve[-1] - init_eval["loss"]:+.4f}
  ROC-AUC          N/A                     {final_auc:.4f}                  N/A

FINAL CONFUSION MATRIX:
          Pred Ham  Pred Spam
True Ham  {final_cm[0, 0]:5d}       {final_cm[0, 1]:5d}
True Spam  {final_cm[1, 0]:5d}       {final_cm[1, 1]:5d}

{"=" * 80}
"""
with open(OUTPUT_DIR / "CLEAN_RESULTS_SUMMARY.txt", "w") as f:
    f.write(summary)
print(summary)

print(f"\nAll files saved to: {OUTPUT_DIR.resolve()}")
