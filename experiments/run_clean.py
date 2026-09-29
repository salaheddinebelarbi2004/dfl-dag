import os, json
from pathlib import Path
from federated_learning_system import main

out_dir = Path("results/clean_baseline")
out_dir.mkdir(exist_ok=True)

system, results = main(
    pretrained_model_path="global_model.npy",
    use_real_data=True,
    real_data_dir="real_federated_data",
)

results_path = out_dir / "training_results_clean.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

model_path = out_dir / "global_model_clean.npy"
system.save_model(str(model_path))

print(f"Results saved to {results_path}")
print(f"Model saved to {model_path}")
