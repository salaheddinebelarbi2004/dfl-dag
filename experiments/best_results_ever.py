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
    MALICIOUS_CLIENT_PENALTY,
    LABEL_SKEW_THRESHOLD,
)

DATA_DIR = Path("malisious")
OUTPUT_DIR = Path("results/malicious_baseline")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("MALICIOUS EVALUATION TEST - FULL SYSTEM")
print("=" * 80)

# â”€â”€ 1. Load pre-trained global model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[1/8] Loading pre-trained global model...")
pretrained_weights = np.load("global_model.npy")
nn = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
nn.set_params(pretrained_weights)
print(f"  global_model.npy ({len(pretrained_weights)} params, {type(nn).__name__})")

# â”€â”€ 2. Load malisious data per client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[2/8] Loading malisious data per client...")
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
    flag = "  <<< MALICIOUS" if y.mean() > 0.9 else ""
    print(f"  {cid}: {len(y)} samples, spam={y.mean() * 100:.1f}%{flag}")

malicious_ground_truth = ["client_4", "client_7"]

# â”€â”€ 3. Evaluation set from clean clients â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[3/8] Preparing evaluation set (clean clients only)...")
clean_clients = sorted(
    [cid for cid in client_data if cid not in malicious_ground_truth]
)
X_eval_parts, y_eval_parts = [], []
for cid in clean_clients:
    Xc, yc = client_data[cid]
    X_eval_parts.append(Xc)
    y_eval_parts.append(yc)
X_eval_all = np.vstack(X_eval_parts).astype(np.float32)
y_eval_all = np.concatenate(y_eval_parts).astype(np.int64)
scaler = StandardScaler()
X_eval_scaled = scaler.fit_transform(X_eval_all)
_, X_eval, _, y_eval = train_test_split(
    X_eval_scaled, y_eval_all, test_size=0.3, random_state=42, stratify=y_eval_all
)
print(f"  {len(clean_clients)} clean clients, {len(X_eval)} eval samples")

# â”€â”€ 4. Initial evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[4/8] Initial evaluation (round 0)...")


def evaluate(model, Xs, y):
    probs = model.forward(Xs)[0][:, 1]
    preds = (probs >= 0.5).astype(int)
    return (
        float(accuracy_score(y, preds)),
        float(f1_score(y, preds, zero_division=0)),
        float(model.compute_loss(Xs, y)),
        probs,
        preds,
    )


init_acc, init_f1, init_loss, _, _ = evaluate(nn, X_eval, y_eval)
print(f"  Accuracy: {init_acc:.4f} | F1: {init_f1:.4f} | Loss: {init_loss:.4f}")

# â”€â”€ 5. Initialize reputation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[5/8] Initializing reputation system...")
reputations = {cid: 0.8 for cid in client_data}
reputation_history = {cid: [0.8] for cid in client_data}

# â”€â”€ 6. Federated rounds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NUM_ROUNDS = 5
print(f"\n[6/8] Simulating {NUM_ROUNDS} federated rounds...")

round_metrics = [
    {
        "round": 0,
        "accuracy": init_acc,
        "f1": init_f1,
        "loss": init_loss,
        "reputations": reputations.copy(),
        "detected": [],
        "stage": "Pre-trained model broadcast",
    }
]

current_weights = pretrained_weights.copy()

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n  === Round {rnd} ===")
    updates = {}
    update_norms = {}

    for cid, (Xc, yc) in client_data.items():
        local = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        local.set_params(current_weights)
        Xc_s = scaler.transform(Xc)

        # Mini-batch SGD for meaningful updates
        n = len(Xc_s)
        batch_size = 32
        n_batches = (n + batch_size - 1) // batch_size
        for epoch in range(5):
            idx = np.random.permutation(n)
            Xs, ys = Xc_s[idx], yc[idx]
            for b in range(n_batches):
                s, e = b * batch_size, min((b + 1) * batch_size, n)
                grads = local.backward(Xs[s:e], ys[s:e], weight_decay=0.001)
                local.update_params(grads, learning_rate=0.05)

        update = local.get_params() - current_weights
        updates[cid] = update
        update_norms[cid] = float(np.linalg.norm(update))

    # â”€â”€ Malicious Detection Engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_cids = sorted(client_data.keys())
    all_updates_arr = np.array([updates[cid] for cid in all_cids])
    all_norms = np.array([update_norms[cid] for cid in all_cids])

    # 1. Label skew detection
    detected_skew = [
        cid for cid in all_cids if client_data[cid][1].mean() > LABEL_SKEW_THRESHOLD
    ]

    # 2. Gradient norm outlier detection
    norm_median = np.median(all_norms)
    norm_mad = np.median(np.abs(all_norms - norm_median))
    if norm_mad < 1e-8:
        norm_mad = float(np.std(all_norms))
    norm_threshold = 3.0 * max(norm_mad, 0.01)
    detected_norm = [
        cid
        for cid, n in zip(all_cids, all_norms)
        if abs(n - norm_median) > norm_threshold
    ]

    # 3. Cosine similarity outlier detection (Byzantine detection)
    norms_for_sim = np.where(all_norms == 0, 1.0, all_norms)
    normalized = all_updates_arr / norms_for_sim[:, np.newaxis]
    sim_matrix = normalized @ normalized.T
    np.fill_diagonal(sim_matrix, 0)
    avg_sim = np.mean(sim_matrix, axis=1)
    sim_median = np.median(avg_sim)
    sim_mad = np.median(np.abs(avg_sim - sim_median))
    if sim_mad < 1e-8:
        sim_mad = float(np.std(avg_sim))
    sim_threshold = 3.0 * max(sim_mad, 0.01)
    detected_sim = [
        cid for cid, s in zip(all_cids, avg_sim) if abs(s - sim_median) > sim_threshold
    ]

    detected_all = list(set(detected_skew + detected_norm + detected_sim))
    print(f"    Label skew: {detected_skew}")
    print(f"    Norm outliers: {detected_norm}")
    print(f"    Similarity outliers: {detected_sim}")
    print(f"    Total detected: {detected_all}")

    # â”€â”€ Robust Aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    weights = np.array([reputations[cid] for cid in all_cids])

    # Apply downweight for detected malicious
    for cid in detected_all:
        idx = all_cids.index(cid)
        weights[idx] *= 0.1  # 90% weight reduction

    # Exclude clients with very low reputation
    eligible = [i for i, cid in enumerate(all_cids) if reputations[cid] >= 0.5]
    if not eligible:
        eligible = list(range(len(all_cids)))

    eligible_weights = weights[eligible]
    eligible_weights = eligible_weights / eligible_weights.sum()
    eligible_updates = all_updates_arr[eligible]

    # Weighted aggregation
    if len(eligible_updates) > 1:
        aggregated_update = np.average(
            eligible_updates, axis=0, weights=eligible_weights
        )
    else:
        aggregated_update = eligible_updates[0]

    current_weights = current_weights + aggregated_update

    # â”€â”€ Reputation Update â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for cid in all_cids:
        if cid in detected_all:
            reputations[cid] = max(0.0, reputations[cid] - MALICIOUS_CLIENT_PENALTY)
        else:
            reputations[cid] = min(1.0, reputations[cid] + 0.05)
        reputation_history[cid].append(reputations[cid])

    # â”€â”€ Evaluate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    nn.set_params(current_weights)
    acc, f1_val, loss, _, _ = evaluate(nn, X_eval, y_eval)
    detected_low_rep = [cid for cid in all_cids if reputations[cid] < 0.5]

    print(f"    Aggregation: {len(eligible)}/{len(all_cids)} eligible updates")
    print(f"    Accuracy: {acc:.4f} | F1: {f1_val:.4f} | Loss: {loss:.4f}")
    for cid in all_cids[:3] + malicious_ground_truth:
        print(f"      {cid}: rep={reputations[cid]:.3f}, norm={update_norms[cid]:.4f}")

    round_metrics.append(
        {
            "round": rnd,
            "accuracy": acc,
            "f1": f1_val,
            "loss": loss,
            "reputations": reputations.copy(),
            "detected": detected_all,
            "detected_skew": detected_skew,
            "detected_norm": detected_norm,
            "detected_sim": detected_sim,
            "eligible_count": len(eligible),
            "total_count": len(all_cids),
            "stage": "Robust aggregation (skew+norm+sim detection, reputation weighting)",
        }
    )

# â”€â”€ 7. Final evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[7/8] Final evaluation...")
final_acc, final_f1, final_loss, final_probs, final_preds = evaluate(nn, X_eval, y_eval)
cm = confusion_matrix(y_eval, final_preds)
tn, fp, fn, tp = cm.ravel()
fpr, tpr, _ = roc_curve(y_eval, final_probs)
roc_auc = auc(fpr, tpr)
final_prec = float(precision_score(y_eval, final_preds, zero_division=0))
final_rec = float(recall_score(y_eval, final_preds, zero_division=0))

print(f"  Accuracy:  {final_acc:.4f}")
print(f"  Precision: {final_prec:.4f}")
print(f"  Recall:    {final_rec:.4f}")
print(f"  F1 Score:  {final_f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# â”€â”€ Save metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
metrics = {
    "initial": {"accuracy": init_acc, "f1": init_f1, "loss": init_loss},
    "final": {
        "accuracy": final_acc,
        "f1": final_f1,
        "precision": final_prec,
        "recall": final_rec,
        "roc_auc": float(roc_auc),
        "loss": final_loss,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    },
    "rounds": [
        {
            "round": m["round"],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "loss": m["loss"],
            "detected": m["detected"],
            "stage": m["stage"],
        }
        for m in round_metrics
    ],
    "reputation_evolution": reputation_history,
    "config": {
        "num_rounds": NUM_ROUNDS,
        "model": "global_model.npy",
        "data": "malisious",
        "malicious_clients": malicious_ground_truth,
        "aggregation": "robust_weighted (norm_outlier + similarity_outlier + label_skew + reputation_weighted)",
        "reputation_penalty": MALICIOUS_CLIENT_PENALTY,
        "final_reputations": {cid: round(reputations[cid], 3) for cid in reputations},
    },
}
with open(OUTPUT_DIR / "best_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# â”€â”€ Curves JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rounds_x = [m["round"] for m in round_metrics]
acc_curve = [m["accuracy"] for m in round_metrics]
f1_curve = [m["f1"] for m in round_metrics]
loss_curve = [m["loss"] for m in round_metrics]
curves = {
    "rounds": rounds_x,
    "accuracies": acc_curve,
    "f1_scores": f1_curve,
    "losses": loss_curve,
    "final_metrics": {"acc": final_acc, "f1": final_f1, "loss": final_loss},
}
with open(OUTPUT_DIR / "training_curves.json", "w") as f:
    json.dump(curves, f, indent=2)

# â”€â”€ 8. PLOTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[8/8] Generating plots...")


def save_plot(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


C = {
    "blue": "#1A73E8",
    "green": "#2E7D32",
    "red": "#C62828",
    "orange": "#F57C00",
    "purple": "#6A1B9A",
    "teal": "#00695C",
    "bg": "#FAFAFA",
}
bb = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)


def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


EP = NUM_ROUNDS
acc_lo = max(0, min(acc_curve) * 0.98)
f1_lo = max(0, min(f1_curve) * 0.98)
loss_hi = max(loss_curve) * 1.15

# Accuracy
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(
    ax,
    "Global Model Accuracy - Malicious Attack with Robust Defense",
    "Round",
    "Accuracy",
)
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
    f"Pre-trained broadcast ({init_acc:.4f})",
    transform=ax.transAxes,
    fontsize=9,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax.legend(loc="lower left", fontsize=10)
save_plot(fig, "best_accuracy_curve.png")

# F1
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(
    ax,
    "Global Model F1 Score - Malicious Attack with Robust Defense",
    "Round",
    "F1 Score",
)
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
save_plot(fig, "best_f1_curve.png")

# Loss
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(ax, "Global Model Loss - Malicious Attack with Robust Defense", "Round", "Loss")
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
save_plot(fig, "best_loss_curve.png")

# Combined
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
fig.patch.set_facecolor(C["bg"])
for ax in [ax1, ax2, ax3]:
    ax.set_facecolor(C["bg"])
ax1.plot(rounds_x, acc_curve, linewidth=2.5, marker="o", color=C["blue"])
ax1.fill_between(rounds_x, acc_lo, acc_curve, alpha=0.12, color=C["blue"])
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.set_title(
    "Malicious Evaluation - Robust Aggregation + Reputation + Byzantine Detection",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.set_xlim(-0.3, EP + 0.3)
ax1.set_ylim(acc_lo, 1.0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.text(
    0.02,
    0.93,
    f"Init: {acc_curve[0]:.4f} â†’ Final: {acc_curve[-1]:.4f}",
    transform=ax1.transAxes,
    fontsize=10,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax2.plot(rounds_x, f1_curve, linewidth=2.5, marker="s", color=C["green"])
ax2.fill_between(rounds_x, f1_lo, f1_curve, alpha=0.12, color=C["green"])
ax2.set_ylabel("F1 Score", fontsize=12)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_xlim(-0.3, EP + 0.3)
ax2.set_ylim(f1_lo, 1.0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.text(
    0.02,
    0.93,
    f"Init: {f1_curve[0]:.4f} â†’ Final: {f1_curve[-1]:.4f}",
    transform=ax2.transAxes,
    fontsize=10,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
ax3.plot(rounds_x, loss_curve, linewidth=2.5, marker="^", color=C["red"])
ax3.fill_between(rounds_x, loss_lo, loss_curve, alpha=0.10, color=C["red"])
ax3.set_xlabel("Round", fontsize=12)
ax3.set_ylabel("Loss", fontsize=12)
ax3.grid(True, alpha=0.3, linestyle="--")
ax3.set_xlim(-0.3, EP + 0.3)
ax3.set_ylim(loss_lo, loss_hi)
ax3.spines[["top", "right"]].set_visible(False)
ax3.text(
    0.02,
    0.93,
    f"Init: {loss_curve[0]:.4f} â†’ Final: {loss_curve[-1]:.4f}",
    transform=ax3.transAxes,
    fontsize=10,
    fontweight="bold",
    verticalalignment="top",
    bbox=bb,
)
fig.tight_layout()
save_plot(fig, "best_curves_combined.png")

# ROC
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(
    ax, f"ROC Curve (AUC = {roc_auc:.4f})", "False Positive Rate", "True Positive Rate"
)
ax.plot(fpr, tpr, linewidth=2.5, color=C["blue"], label=f"AUC = {roc_auc:.4f}")
ax.fill_between(fpr, tpr, alpha=0.12, color=C["blue"])
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Random")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower right", fontsize=11)
save_plot(fig, "best_roc_curve.png")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix (Robust Aggregation)", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Ham (0)", "Spam (1)"], fontsize=11)
ax.set_yticklabels(["Ham (0)", "Spam (1)"], fontsize=11)
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
save_plot(fig, "best_confusion_matrix.png")

# Reputation evolution
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
style(
    ax,
    "Client Reputation Scores (Penalty = -0.3 for detected malicious)",
    "Round",
    "Reputation",
)
rep_colors = {
    cid: ("#C62828" if cid in malicious_ground_truth else "#2E7D32")
    for cid in client_data
}
for cid in sorted(client_data.keys()):
    ax.plot(
        rounds_x,
        reputation_history[cid],
        linewidth=2,
        marker="o",
        markersize=6,
        color=rep_colors[cid],
        label=cid,
    )
ax.axhline(
    y=0.5,
    color="gray",
    linewidth=1,
    linestyle="--",
    alpha=0.5,
    label="Eligibility threshold (0.5)",
)
ax.set_xlim(-0.3, EP + 0.3)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", fontsize=8, ncol=3)
save_plot(fig, "best_reputation_curve.png")

# Metrics bar
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
vals = [final_acc, final_prec, final_rec, final_f1, roc_auc]
bcols = [C["blue"], C["orange"], C["green"], C["purple"], C["teal"]]
bars = ax.bar(names, vals, color=bcols, width=0.6, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{v:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax.set_ylim(0, 1.08)
ax.set_title(
    "Final Evaluation Metrics (Robust Defense)", fontsize=14, fontweight="bold", pad=12
)
ax.grid(True, alpha=0.3, axis="y", linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save_plot(fig, "best_metrics_bar.png")

# â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
summary_lines = [
    "=" * 80,
    "MALICIOUS EVALUATION TEST - FULL SYSTEM RESULTS",
    "=" * 80,
    "",
    "SYSTEM ARCHITECTURE:",
    "  Pre-trained global model (global_model.npy) broadcast to all clients at round 0.",
    "  Each client trains locally on their malisious/ data partition.",
    "  Malicious clients (4, 7) have 100% spam labels (label-flipping attack).",
    "",
    "MALICIOUS DETECTION (3-layer):",
    f"  1. Label Skew Detection:  Clients with >{LABEL_SKEW_THRESHOLD:.0%} of one class",
    "  2. Norm Outlier Detection: Gradients with norm > 3 MAD from median",
    "  3. Similarity Outlier:     Updates with low cosine similarity to consensus (Byzantine detection)",
    "",
    "ROBUST AGGREGATION:",
    "  - Detected clients receive 90% weight reduction",
    "  - Clients with reputation < 0.5 excluded entirely",
    "  - Weighted averaging by reputation scores",
    "",
    "REPUTATION SYSTEM:",
    f"  - Initial score: 0.8 for all clients",
    f"  - Detected malicious: -{MALICIOUS_CLIENT_PENALTY:.1f} penalty",
    "  - Honest behavior: +0.05 reward per round",
    "",
    "BYZANTINE FAULT TOLERANCE:",
    "  - Voting via cosine similarity consensus",
    "  - Updates diverging from group median are downweighted",
    "  - Requires majority honest for correct consensus",
    "",
    "RESULTS:",
    f"  Metric           Round 0 (broadcast)    Round {NUM_ROUNDS} (final)    Change",
    "  -------          -------------------    ----------------------    --------",
    f"  Accuracy         {init_acc:.4f}                 {final_acc:.4f}                  {final_acc - init_acc:+.4f}",
    f"  Precision        N/A                     {final_prec:.4f}                  N/A",
    f"  Recall           N/A                     {final_rec:.4f}                  N/A",
    f"  F1 Score         {init_f1:.4f}                 {final_f1:.4f}                  {final_f1 - init_f1:+.4f}",
    f"  Loss             {init_loss:.4f}                 {final_loss:.4f}                  {final_loss - init_loss:+.4f}",
    "",
    "REPUTATION EVOLUTION:",
]
for cid in sorted(client_data.keys()):
    rep_start = reputation_history[cid][0]
    rep_end = reputation_history[cid][-1]
    marker = " <<< MALICIOUS" if cid in malicious_ground_truth else ""
    summary_lines.append(f"    {cid}: {rep_start:.1f} -> {rep_end:.3f}{marker}")
summary_lines += [
    "",
    "FINAL CONFUSION MATRIX:",
    f"          Pred Ham  Pred Spam",
    f"True Ham  {cm[0, 0]:<6}    {cm[0, 1]:<6}",
    f"True Spam {cm[1, 0]:<6}    {cm[1, 1]:<6}",
    "",
    "=" * 80,
]
summary = "\n".join(summary_lines)
for cid in sorted(client_data.keys()):
    rep_start = reputation_history[cid][0]
    rep_end = reputation_history[cid][-1]
    marker = " <<< MALICIOUS" if cid in malicious_ground_truth else ""
    summary += f"    {cid}: {rep_start:.1f} -> {rep_end:.3f}{marker}\n"

summary += f"""
FINAL CONFUSION MATRIX:
          Pred Ham  Pred Spam
True Ham  {cm[0, 0]:<6}    {cm[0, 1]:<6}
True Spam {cm[1, 0]:<6}    {cm[1, 1]:<6}

{"=" * 80}
"""
print(summary)
with open(OUTPUT_DIR / "BEST_RESULTS_SUMMARY.txt", "w", encoding="utf-8") as f:
    f.write(summary)
print(f"All files saved to: {OUTPUT_DIR.resolve()}")
