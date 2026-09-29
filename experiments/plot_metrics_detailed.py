import numpy as np
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from evaluate_malicious_model import (
    load_real_federated_dataset,
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
)
from client import NeuralNetworkSimulator
from lightweight_classifier import LightweightClassifier
from federated_learning_config import SHARD_SIZE

OUTPUT_DIR = Path("results/malicious_baseline")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESH = 0.9


def load_model():
    model_path = OUTPUT_DIR / "global_model_malicious.npy"
    weights = np.load(model_path)
    expected_size = (
        INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    )
    if len(weights) == expected_size:
        model = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model.set_params(weights)
        return model, "nn"
    else:
        out_dim = 2
        if (len(weights) - out_dim) % out_dim != 0:
            input_dim = 384
        else:
            input_dim = int((len(weights) - out_dim) / out_dim)
        clf = LightweightClassifier(input_dim=input_dim, output_dim=out_dim)
        clf.set_weights(weights)
        return clf, "clf"


def plot_confusion_matrix(cm, labels, outpath, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
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
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    model, mtype = load_model()
    X_train, X_test, y_train, y_test = load_real_federated_dataset(
        Path("real_federated_data")
    )

    # Global metrics & ROC
    # compute probabilities and apply decision threshold for final predictions
    if mtype == "nn":
        probs = model.forward(X_test)[0][:, 1]
    else:
        probs = model.forward(X_test)[1][:, 1]
    preds = (probs >= THRESH).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    with open(OUTPUT_DIR / "metrics_global.json", "w") as fh:
        json.dump({"accuracy": acc, "f1": f1}, fh, indent=2)

    plot_confusion_matrix(
        cm,
        ["Ham", "Spam"],
        OUTPUT_DIR / "confusion_matrix_global.png",
        title="Confusion Matrix (Global)",
    )

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("ROC Curve (Global)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_global.png", dpi=150)
    plt.close(fig)

    # Per-client metrics
    client_files = sorted(Path("real_federated_data").glob("client_*.npz"))
    client_ids = []
    accs = []
    f1s = []
    for p in client_files:
        client_id = p.stem
        client_ids.append(client_id)
        d = np.load(p)
        Xc = d["X"].astype(np.float32)
        yc = d["y"]
        if yc.ndim > 1:
            yc = np.argmax(yc, axis=1)
        else:
            yc = yc.reshape(-1)
        # scale using global scaler? We used StandardScaler in dataset loader; reuse by fitting on real_federated_data whole set
        # to keep things simple, reuse model's predict interface
        if mtype == "nn":
            probs_c = model.forward(Xc)[0][:, 1]
        else:
            probs_c = model.forward(Xc)[1][:, 1]
        pc = (probs_c >= THRESH).astype(int)
        ac = float(accuracy_score(yc, pc))
        f1c = float(f1_score(yc, pc, zero_division=0))
        accs.append(ac)
        f1s.append(f1c)
        cmc = confusion_matrix(yc, pc)
        plot_confusion_matrix(
            cmc,
            ["Ham", "Spam"],
            OUTPUT_DIR / f"confusion_{client_id}.png",
            title=f"Confusion ({client_id})",
        )

    # Save per-client metrics
    per_client = {
        cid: {"accuracy": a, "f1": f} for cid, a, f in zip(client_ids, accs, f1s)
    }
    with open(OUTPUT_DIR / "metrics_per_client.json", "w") as fh:
        json.dump(per_client, fh, indent=2)

    # Plot per-client accuracy & f1 (bars)
    x = range(len(client_ids))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, accs, width=0.4, label="accuracy")
    ax.bar([i + 0.4 for i in x], f1s, width=0.4, label="f1")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(client_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Per-client Accuracy and F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_client_acc_f1.png", dpi=150)
    plt.close(fig)

    # Per-shard aggregation
    shard_map = {}
    for cid in client_ids:
        idx = int(cid.split("_")[1])
        shard_id = idx // SHARD_SIZE
        shard_map.setdefault(shard_id, []).append(cid)

    per_shard = {}
    for shard_id, cids in shard_map.items():
        # aggregate metrics across clients in this shard
        a_list = [per_client[c]["accuracy"] for c in cids]
        f_list = [per_client[c]["f1"] for c in cids]
        per_shard[f"shard_{shard_id}"] = {
            "accuracy_mean": float(np.mean(a_list)),
            "f1_mean": float(np.mean(f_list)),
            "members": cids,
        }
        # combined confusion matrix across members
        cms = None
        for cid in cids:
            d = np.load(Path("real_federated_data") / f"{cid}.npz")
            Xc = d["X"].astype(np.float32)
            yc = d["y"]
            if yc.ndim > 1:
                yc = np.argmax(yc, axis=1)
            else:
                yc = yc.reshape(-1)
            if mtype == "nn":
                probs_c = model.forward(Xc)[0][:, 1]
            else:
                probs_c = model.forward(Xc)[1][:, 1]
            pc = (probs_c >= THRESH).astype(int)
            cm_part = confusion_matrix(yc, pc)
            if cms is None:
                cms = cm_part
            else:
                cms = cms + cm_part
        plot_confusion_matrix(
            cms,
            ["Ham", "Spam"],
            OUTPUT_DIR / f"confusion_shard_{shard_id}.png",
            title=f"Confusion (shard_{shard_id})",
        )

    with open(OUTPUT_DIR / "metrics_per_shard.json", "w") as fh:
        json.dump(per_shard, fh, indent=2)

    # Round-wise curves: gradient avg and model change norms
    tr = json.load(open(OUTPUT_DIR / "training_results_malicious.json"))
    rounds = tr.get("total_round_metrics", [])
    rnums = [r["round_num"] for r in rounds]
    avg_grad = [r["gradient_stats"]["avg_gradient_norm"] for r in rounds]
    model_change = [r["model_changes"]["weight_change_norm"] for r in rounds]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rnums, avg_grad, marker="o", label="avg_gradient_norm", linewidth=2)
    ax.plot(rnums, model_change, marker="s", label="weight_change_norm", linewidth=2)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Norm", fontsize=11)
    ax.set_title(
        "Round-wise Gradient and Model-change Norms", fontsize=12, fontweight="bold"
    )
    ax.set_xlim(0, max(rnums) + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "roundwise_gradient_modelchange.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print("Saved detailed plots and metrics to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
