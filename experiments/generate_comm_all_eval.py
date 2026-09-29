"""Generate multi-rho communication evaluation artifacts from saved runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "comm_all_eval"
RUN_PATHS = {
    1.0: ROOT / "results/contribution_results/ratio_1/training_results.json",
    0.5: ROOT / "results/contribution_results/ratio_0_5/training_results.json",
    0.2: ROOT / "results/contribution_results/ratio_0_2/training_results.json",
    0.1: ROOT / "results/contribution_results/ratio_0_1/training_results.json",
}
QUANTIZED_PATH = (
    ROOT / "results/contribution_results/ratio_0_1_quant8/training_results.json"
)
COLORS = {1.0: "#24527a", 0.5: "#d46b32", 0.2: "#2a9d8f", 0.1: "#8d5a97"}


def load_run(rho: float, path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = []
    for metric in data["metrics"]:
        clients = metric["steps"]["2-6_client_updates"]["clients"]
        global_aggregation = metric["steps"]["10_global_aggregation"]["aggregation"]
        stats = metric["communication_stats"]
        rows.append(
            {
                "rho": rho,
                "round": metric["round_num"],
                "original_bytes": stats["total_bytes_original"],
                "compressed_bytes": stats["total_bytes_compressed"],
                "measured_ratio": stats["avg_compression_ratio"],
                "bytes_saved": stats["bytes_saved"],
                "accuracy": metric["global_eval"]["test_accuracy"],
                "f1": metric["global_eval"]["test_f1"],
                "update_norm": global_aggregation["model_update"]["update_norm"],
                "model_change_ratio": metric["model_changes"]["model_change_ratio"],
                "elapsed_seconds": metric["elapsed_seconds"],
                "global_success": global_aggregation["success"],
                "verification_rates": [
                    item["verification_rate"]
                    for item in metric["steps"]["7-9_shard_aggregation"][
                        "shard_results"
                    ].values()
                ],
                "clients": len(clients),
            }
        )
    return {"rho": rho, "rows": rows, "data": data}


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def generate_plots(runs: list[dict]) -> None:
    rounds = np.array([row["round"] for row in runs[0]["rows"]])

    # Requested: global accuracy versus round, one curve per rho.
    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        ax.plot(
            rounds,
            [row["accuracy"] * 100 for row in run["rows"]],
            marker="o",
            linewidth=2.2,
            label=f"rho={run['rho']}",
            color=COLORS[run["rho"]],
        )
    ax.set_title("Global held-out accuracy versus round")
    ax.set_xlabel("Training round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(rounds)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, "01_accuracy_by_rho")

    # Requested: communication volume per round, per rho.
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.18
    for index, run in enumerate(runs):
        values = np.array([row["compressed_bytes"] for row in run["rows"]]) / 1_000_000
        positions = rounds + (index - (len(runs) - 1) / 2) * width
        ax.bar(
            positions,
            values,
            width,
            label=f"rho={run['rho']}",
            color=COLORS[run["rho"]],
        )
    ax.set_title("Communication volume per round")
    ax.set_xlabel("Training round")
    ax.set_ylabel("Serialized update volume (MB)")
    ax.set_xticks(rounds)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, "02_communication_by_rho")

    # Requested: trade-off, total bytes on log scale versus final accuracy.
    fig, ax = plt.subplots(figsize=(8, 5))
    totals = []
    for run in runs:
        total = sum(row["compressed_bytes"] for row in run["rows"])
        final_accuracy = run["rows"][-1]["accuracy"] * 100
        totals.append(total)
        ax.scatter(
            total,
            final_accuracy,
            s=150,
            color=COLORS[run["rho"]],
            label=f"rho={run['rho']}",
            zorder=3,
        )
        ax.annotate(
            f"rho={run['rho']}",
            (total, final_accuracy),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_title("Communication-accuracy trade-off")
    ax.set_xlabel("Total communication over five rounds (bytes, log scale)")
    ax.set_ylabel("Final global accuracy (%)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    save(fig, "03_tradeoff_log_bytes")

    # Requested: measured versus theoretical compression ratio.
    fig, ax = plt.subplots(figsize=(8, 5))
    rhos = np.array([run["rho"] for run in runs])
    measured = np.array(
        [np.mean([row["measured_ratio"] for row in run["rows"]]) for run in runs]
    )
    order = np.argsort(rhos)
    ax.plot(
        rhos[order],
        rhos[order] * 100,
        "k--",
        linewidth=1.8,
        label="Theoretical rho x 100",
    )
    ax.plot(
        rhos[order],
        measured[order] * 100,
        "o-",
        linewidth=2.2,
        color="#d46b32",
        label="Measured serialized ratio x 100",
    )
    for rho, value in zip(rhos, measured):
        ax.annotate(
            f"{value * 100:.1f}%",
            (rho, value * 100),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Measured versus theoretical compression ratio")
    ax.set_xlabel("Target compression ratio rho")
    ax.set_ylabel("Transmitted / original (%)")
    ax.set_xlim(-0.02, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "04_measured_vs_theoretical_ratio")

    # Requested: available message composition. The raw JSON has no proof/proxy byte fields,
    # so this chart reports the measured serialized update and the unmeasured protocol fields explicitly.
    fig, ax = plt.subplots(figsize=(9, 5))
    selected = [run for run in runs if run["rho"] in (1.0, 0.1)]
    labels = [f"rho={run['rho']}" for run in selected]
    update_bytes = [
        np.mean([row["compressed_bytes"] for row in run["rows"]]) for run in selected
    ]
    protocol_bytes = [0 for _ in selected]
    bars = ax.bar(
        labels,
        update_bytes,
        label="Measured serialized update payload",
        color="#24527a",
    )
    ax.bar(
        labels,
        protocol_bytes,
        bottom=update_bytes,
        label="Proof + commitment + polynomial proxy (not stored)",
        color="#d46b32",
    )
    for bar, value in zip(bars, update_bytes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(update_bytes) * 0.02,
            f"{value:,.0f} B",
            ha="center",
            fontsize=9,
        )
    ax.set_title("Message composition: measured payload and missing protocol breakdown")
    ax.set_ylabel("Mean serialized bytes per client update")
    ax.text(
        0.5,
        0.92,
        "Proof, commitment, and polynomial-proxy bytes are not recorded separately in training_results.json",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#8a3b12",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "05_message_composition_available")

    # Additional useful plot: final accuracy against communication saving.
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_total = sum(row["compressed_bytes"] for row in runs[0]["rows"])
    for run in runs:
        total = sum(row["compressed_bytes"] for row in run["rows"])
        saving = (baseline_total - total) / baseline_total * 100
        accuracy = run["rows"][-1]["accuracy"] * 100
        ax.scatter(
            saving, accuracy, s=150, color=COLORS[run["rho"]], label=f"rho={run['rho']}"
        )
        ax.annotate(
            f"rho={run['rho']}",
            (saving, accuracy),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_title("Accuracy retained versus communication saving")
    ax.set_xlabel("Communication saving relative to rho=1 (%)")
    ax.set_ylabel("Final global accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "06_accuracy_vs_saving")

    # Additional useful plot: robust verification and timing.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in runs:
        mean_verification = [
            np.mean(row["verification_rates"]) * 100 for row in run["rows"]
        ]
        elapsed = [row["elapsed_seconds"] for row in run["rows"]]
        axes[0].plot(
            rounds,
            mean_verification,
            marker="o",
            linewidth=2,
            label=f"rho={run['rho']}",
            color=COLORS[run["rho"]],
        )
        axes[1].plot(
            rounds,
            elapsed,
            marker="o",
            linewidth=2,
            label=f"rho={run['rho']}",
            color=COLORS[run["rho"]],
        )
    axes[0].set_title("Mean shard verification rate")
    axes[0].set_ylabel("Verified updates (%)")
    axes[1].set_title("Round wall-clock time")
    axes[1].set_ylabel("Seconds")
    for axis in axes:
        axis.set_xlabel("Training round")
        axis.set_xticks(rounds)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, "07_verification_and_runtime")

    if QUANTIZED_PATH.exists():
        quantized = load_run(0.1, QUANTIZED_PATH)
        sparse = next(run for run in runs if run["rho"] == 0.1)
        labels = ["Top-k\n(float)", "Top-k + 8-bit\nquantization"]
        accuracy = [
            sparse["rows"][-1]["accuracy"] * 100,
            quantized["rows"][-1]["accuracy"] * 100,
        ]
        bytes_total = [
            sum(row["compressed_bytes"] for row in sparse["rows"]) / 1_000_000,
            sum(row["compressed_bytes"] for row in quantized["rows"]) / 1_000_000,
        ]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        bars = axes[0].bar(labels, accuracy, color=["#24527a", "#d46b32"])
        axes[0].set_title("Final accuracy at rho=0.1")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim(min(accuracy) - 2, max(accuracy) + 1)
        for bar, value in zip(bars, accuracy):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.1,
                f"{value:.2f}%",
                ha="center",
                fontsize=9,
            )
        bars = axes[1].bar(labels, bytes_total, color=["#24527a", "#d46b32"])
        axes[1].set_title("Cumulative communication at rho=0.1")
        axes[1].set_ylabel("Serialized bytes (MB)")
        for bar, value in zip(bars, bytes_total):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.05,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
            )
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
        save(fig, "08_quantization_effect_rho_0_1")


def write_outputs(runs: list[dict]) -> None:
    row_fields = [
        "rho",
        "round",
        "original_bytes",
        "compressed_bytes",
        "measured_ratio",
        "bytes_saved",
        "accuracy",
        "f1",
        "update_norm",
        "model_change_ratio",
        "elapsed_seconds",
        "global_success",
    ]
    rows = [row for run in runs for row in run["rows"]]
    with (OUTPUT / "per_round_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in row_fields} for row in rows])

    baseline_total = sum(row["compressed_bytes"] for row in runs[0]["rows"])
    summaries = []
    for run in runs:
        total = sum(row["compressed_bytes"] for row in run["rows"])
        first, last = run["rows"][0], run["rows"][-1]
        summaries.append(
            {
                "rho": run["rho"],
                "total_original_bytes": sum(
                    row["original_bytes"] for row in run["rows"]
                ),
                "total_compressed_bytes": total,
                "saving_vs_original_percent": (
                    sum(row["original_bytes"] for row in run["rows"]) - total
                )
                / sum(row["original_bytes"] for row in run["rows"])
                * 100,
                "saving_vs_baseline_percent": (baseline_total - total)
                / baseline_total
                * 100,
                "measured_ratio_mean": np.mean(
                    [row["measured_ratio"] for row in run["rows"]]
                ),
                "final_accuracy": last["accuracy"],
                "final_f1": last["f1"],
                "accuracy_gain_points": (last["accuracy"] - first["accuracy"]) * 100,
                "mean_verification_percent": np.mean(
                    [np.mean(row["verification_rates"]) for row in run["rows"]]
                )
                * 100,
                "all_global_rounds_successful": all(
                    row["global_success"] for row in run["rows"]
                ),
            }
        )
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    report_lines = [
        "# Multi-rho Communication Optimization Evaluation",
        "",
        "Source runs: `ratio_1`, `ratio_0_5`, `ratio_0_2`, and `ratio_0_1`.",
        "",
        "## Summary",
        "",
        "| rho | Total transmitted bytes | Saving vs baseline | Measured ratio | Final accuracy | Final F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        report_lines.append(
            f"| {item['rho']} | {item['total_compressed_bytes']:,} | {item['saving_vs_baseline_percent']:+.2f}% | {item['measured_ratio_mean'] * 100:.2f}% | {item['final_accuracy'] * 100:.2f}% | {item['final_f1'] * 100:.2f}% |"
        )
    report_lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `01_accuracy_by_rho`: global accuracy versus round for every rho.",
            "- `02_communication_by_rho`: serialized communication volume per round.",
            "- `03_tradeoff_log_bytes`: total communication versus final accuracy on a log x-axis.",
            "- `04_measured_vs_theoretical_ratio`: target rho compared with measured serialized ratio.",
            "- `05_message_composition_available`: measured update payload; protocol components are marked unavailable because the JSON does not store separate byte counts.",
            "- `06_accuracy_vs_saving`: additional accuracy-retention view relative to the baseline.",
            "- `07_verification_and_runtime`: additional robustness and runtime diagnostics.",
            "- `08_quantization_effect_rho_0_1`: exploratory comparison of float and 8-bit quantized payloads at fixed rho=0.1.",
            "",
            "## Quantization effect",
            "",
            "An exploratory fixed-rho=0.1 8-bit quantized run is available as `ratio_0_1_quant8`. It is plotted separately from the four primary ratio cases because quantization changes the compression method in addition to the target sparsity ratio.",
            "",
            "## Interpretation",
            "",
            "The figures use post-aggregation held-out accuracy (`global_eval`) and serialized communication totals. Theoretical rho is the target retained-coordinate ratio; measured ratio includes metadata serialization overhead, so it is expected to be larger than the target.",
        ]
    )
    (OUTPUT / "evaluation_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    runs = [
        load_run(rho, path) for rho, path in sorted(RUN_PATHS.items(), reverse=True)
    ]
    generate_plots(runs)
    write_outputs(runs)
    print(f"Generated multi-rho evaluation artifacts in {OUTPUT}")


if __name__ == "__main__":
    main()
