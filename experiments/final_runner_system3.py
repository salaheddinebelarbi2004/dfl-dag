"""
Run the ShardedFederatedLearningSystem using `malisious/` as the data source
and save outputs into `results/malicious_baseline/`.
Usage: python final_runner_system3.py
"""
import os
from pathlib import Path
from federated_learning_system import main

out_dir = Path("results/malicious_baseline")
out_dir.mkdir(exist_ok=True)

# Start from scratch: no pretrain, random initialization
system, results = main(pretrained_model_path=None, use_real_data=True, real_data_dir="malisious")

# Save outputs
results_path = out_dir / "training_results_malicious.json"
with open(results_path, 'w') as f:
    import json
    json.dump(results, f, indent=2, default=str)

# Save model
model_path = out_dir / "global_model_malicious.npy"
system.save_model(str(model_path))

print(f"Results saved to {results_path}")
print(f"Model saved to {model_path}")
