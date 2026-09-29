# Multi-rho Communication Optimization Evaluation

Source runs: `ratio_1`, `ratio_0_5`, `ratio_0_2`, and `ratio_0_1`.

## Summary

| rho | Total transmitted bytes | Saving vs baseline | Measured ratio | Final accuracy | Final F1 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 42,814,548 | +0.00% | 100.01% | 82.60% | 77.60% |
| 0.5 | 33,310,644 | +22.20% | 80.03% | 82.04% | 77.76% |
| 0.2 | 13,336,788 | +68.85% | 32.05% | 82.31% | 77.12% |
| 0.1 | 6,678,744 | +84.40% | 16.05% | 81.85% | 76.61% |

## Figures

- `01_accuracy_by_rho`: global accuracy versus round for every rho.
- `02_communication_by_rho`: serialized communication volume per round.
- `03_tradeoff_log_bytes`: total communication versus final accuracy on a log x-axis.
- `04_measured_vs_theoretical_ratio`: target rho compared with measured serialized ratio.
- `05_message_composition_available`: measured update payload; protocol components are marked unavailable because the JSON does not store separate byte counts.
- `06_accuracy_vs_saving`: additional accuracy-retention view relative to the baseline.
- `07_verification_and_runtime`: additional robustness and runtime diagnostics.
- `08_quantization_effect_rho_0_1`: exploratory comparison of float and 8-bit quantized payloads at fixed rho=0.1.

## Quantization effect

An exploratory fixed-rho=0.1 8-bit quantized run is available as `ratio_0_1_quant8`. It is plotted separately from the four primary ratio cases because quantization changes the compression method in addition to the target sparsity ratio.

## Interpretation

The figures use post-aggregation held-out accuracy (`global_eval`) and serialized communication totals. Theoretical rho is the target retained-coordinate ratio; measured ratio includes metadata serialization overhead, so it is expected to be larger than the target.
