"""Plot measured message composition for the baseline and rho=0.1 run."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "contribution_results"
OUTPUT = ROOT / "results" / "comm_all_eval"

RUNS = {
    "Baseline (rho=1.0)": RESULTS / "ratio_1" / "training_results.json",
    "Compressed (rho=0.1)": RESULTS / "ratio_0_1" / "training_results.json",
}

COLORS = {
    "update": "#24527a",
    "protocol": "#d46b32",
}


def mean_payload_bytes(path: Path) -> float:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    payload_sizes = [
        client["compressed_size_bytes"]
        for metric in data["metrics"]
        for client in metric["steps"]["2-6_client_updates"]["clients"]
    ]
    return float(np.mean(payload_sizes))


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)

    labels = list(RUNS)
    update_bytes = [mean_payload_bytes(path) for path in RUNS.values()]

    # Protocol component sizes are not stored separately in training_results.json.
    # Keep the zero-height segment so the figure remains a composition chart
    # without fabricating proof, commitment, or polynomial-proxy measurements.
    protocol_bytes = np.zeros(len(update_bytes))
    x = np.arange(len(labels))
    width = 0.58

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(
        x,
        update_bytes,
        width=width,
        color=COLORS["update"],
        label="Serialized update payload",
    )
    ax.bar(
        x,
        protocol_bytes,
        width=width,
        bottom=update_bytes,
        color=COLORS["protocol"],
        label="Protocol overhead",
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Mean serialized bytes per client update")
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline\n(rho=1.0)", "Compressed\n(rho=0.1)"])
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "05_message_composition_clean.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(OUTPUT / "05_message_composition_clean.svg", bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(OUTPUT / "05_message_composition_clean.png")
    print(OUTPUT / "05_message_composition_clean.svg")


if __name__ == "__main__":
    main()
