"""
Sharded Federated Learning System - Complete Implementation
Integrates all components for full end-to-end training
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import time
import json

# PyTorch support (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from federated_learning_config import (
    NUM_CLIENTS, NUM_SHARDS, SHARD_SIZE, NUM_ROUNDS,
    LOCAL_EPOCHS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
    VERBOSE
)
from client import FederatedLearningClient, create_synthetic_client_data
from model_registry import registry

def load_real_client_data(client_id: int, data_dir: str = "real_federated_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real embeddings and labels for a specific client.

    Args:
        client_id: Client index (0-11)
        data_dir: Directory containing client npz files

    Returns:
        Tuple of (embeddings, one_hot_labels)
    """
    import os

    npz_path = os.path.join(data_dir, f"client_{client_id}.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Real data not found: {npz_path}")

    data = np.load(npz_path)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)

    return X, y
from shard_chief import ShardChief, ShardChiefCandidate, ShardChiefElector, NodeCapabilities
from global_aggregator import GlobalCoordinator, ShardModelUpdate
from dag_system import ShardedDAG
from zkp_proofs import ZKPFactory
from polynomial_proxy import PolynomialProxy


class ShardedFederatedLearningSystem:
    """
    Complete Sharded Federated Learning System
    Implements all three layers: Client, Shard, Global
    """

    def __init__(self,
                 num_clients: int = NUM_CLIENTS,
                 num_shards: int = NUM_SHARDS,
                 num_rounds: int = NUM_ROUNDS,
                 pretrained_model_path: str = None,
                 use_real_data: bool = False,
                 real_data_dir: str = "real_federated_data"):
        """
        Initialize the complete system.
        
        Args:
            num_clients: Total number of clients
            num_shards: Number of shards
            num_rounds: Number of training rounds
            pretrained_model_path: Path to pre-trained model file (.npy), or None for random init
            use_real_data: If True, load real embeddings from real_federated_data/
            real_data_dir: Directory containing real client data
        """
        self.num_clients = num_clients
        self.num_shards = num_shards
        self.num_rounds = num_rounds
        self.shard_size = num_clients // num_shards
        self.use_real_data = use_real_data
        self.real_data_dir = real_data_dir

        # Initialize global model (load from file or create random)
        if pretrained_model_path:
            self.global_model = self._load_pretrained_model(pretrained_model_path)
            if VERBOSE:
                print(f"[OK] Loaded pre-trained model from {pretrained_model_path}")
        else:
            self.global_model = np.random.randn(
                INPUT_DIM * HIDDEN_DIM +
                HIDDEN_DIM +
                HIDDEN_DIM * OUTPUT_DIM +
                OUTPUT_DIM
            ).astype(np.float32) * 0.01
            if VERBOSE:
                print(f"[OK] Initialized random global model")

        # Initialize components
        self.clients: List[FederatedLearningClient] = []
        self.shard_chiefs: Dict[str, ShardChief] = {}
        self.global_coordinator = GlobalCoordinator(num_shards, self.global_model)
        self.sharded_dag = ShardedDAG(num_shards, self._hash_model(self.global_model))

        # System state
        self.current_round = 0
        self.system_metrics: List[Dict] = []

        self._initialize_clients()
        self._elect_shard_chiefs()

        if VERBOSE:
            print(f"[OK] System initialized: {num_clients} clients in {num_shards} shards")

    def _initialize_clients(self):
        """Create and initialize all clients"""
        for i in range(self.num_clients):
            shard_id = f"shard_{i // self.shard_size}"
            client = FederatedLearningClient(
                client_id=f"client_{i}",
                shard_id=shard_id
            )

            # Assign local dataset (real or synthetic)
            if self.use_real_data:
                X, y = load_real_client_data(i, self.real_data_dir)
                if VERBOSE:
                    print(f"[LOAD] client_{i}: {len(X)} real samples from {self.real_data_dir}")
            else:
                X, y = create_synthetic_client_data(i, num_samples=500)

            client.set_local_dataset(X, y)
            self.clients.append(client)

        data_source = "real embeddings" if self.use_real_data else "synthetic data"
        if VERBOSE:
            print(f"[OK] Created {len(self.clients)} clients with {data_source}")

    def _elect_shard_chiefs(self):
        """Elect Shard Chief for each shard"""
        for shard_idx in range(self.num_shards):
            shard_id = f"shard_{shard_idx}"

            # Create candidates from shard clients
            shard_clients = [
                c for c in self.clients
                if c.shard_id == shard_id
            ]

            candidates = [
                ShardChiefCandidate(
                    node_id=client.client_id,
                    capabilities=NodeCapabilities(
                        node_id=client.client_id,
                        uptime=0.95 + np.random.rand() * 0.05,
                        bandwidth_mbps=50 + np.random.rand() * 100,
                        compute_capacity=0.8 + np.random.rand() * 0.4,
                    )
                )
                for client in shard_clients[:3]  # Top 3 candidates
            ]

            # Elect chief
            chief, election_meta = ShardChiefElector.elect_for_shard(
                candidates, shard_id
            )

            self.shard_chiefs[shard_id] = chief

            if VERBOSE:
                print(f"[OK] Elected Shard Chief for {shard_id}: {chief.chief_id} (score: {election_meta['score']:.3f})")

    def execute_training_round(self) -> Dict[str, Any]:
        """
        Execute one complete training round.
        Implements all 10 steps of the workflow.

        Returns:
            Round metadata with all statistics
        """
        self.current_round += 1
        round_start = time.time()

        # Store model before round (for tracking changes)
        model_before = self.global_model.copy()

        round_meta = {
            "round_num": self.current_round,
            "timestamp": round_start,
            "steps": {},
            "gradient_stats": {},  # Track gradient magnitudes
            "shard_contributions": {},  # Track shard updates
            "model_changes": {}  # Track weight changes
        }

        # STEP 1: Global model broadcast
        if VERBOSE:
            print(f"Step 1: Broadcasting global model...")
        broadcast_meta = self.global_coordinator.broadcast_global_model()
        round_meta["steps"]["1_broadcast"] = broadcast_meta

        # STEPS 2-6: Client local training and update generation
        if VERBOSE:
            print(f"Steps 2-6: Client local training ({len(self.clients)} clients)...")

        client_updates = []
        step6_meta = {
            "successful_submissions": 0,
            "failed_submissions": 0,
            "clients": []
        }

        for client in self.clients:
            try:
                # Step 2: Local training
                client.receive_global_model(self.global_model)
                training_meta = client.local_training_round(
                    local_epochs=LOCAL_EPOCHS
                )

                # Steps 3-6: Compute update + polynomial proxy + ZKP + commitment
                update = client.create_update(self.global_model, self.current_round)
                client_updates.append(update)

                step6_meta["successful_submissions"] += 1
                step6_meta["clients"].append({
                    "client_id": client.client_id,
                    "gradient_norm": float(np.linalg.norm(update.gradient.flatten())),
                })
            except Exception as e:
                step6_meta["failed_submissions"] += 1
                if VERBOSE:
                    import traceback
                    print(f"  [WARN] Client {client.client_id} submission failed: {str(e)}")
                    # Get the full traceback
                    tb_lines = traceback.format_exc().split('\n')
                    for line in tb_lines[-5:]:
                        if line.strip():
                            print(f"    {line}")

        round_meta["steps"]["2-6_client_updates"] = step6_meta

        # Calculate average gradient magnitude
        if step6_meta["clients"]:
            gradient_norms = [c["gradient_norm"] for c in step6_meta["clients"]]
            round_meta["gradient_stats"] = {
                "avg_gradient_norm": float(np.mean(gradient_norms)),
                "max_gradient_norm": float(np.max(gradient_norms)),
                "min_gradient_norm": float(np.min(gradient_norms)),
                "std_gradient_norm": float(np.std(gradient_norms)),
                "total_clients_trained": len(gradient_norms)
            }

        # STEPS 7-9: Shard verification and aggregation
        if VERBOSE:
            print(f"Steps 7-9: Shard verification and aggregation ({len(self.shard_chiefs)} shards)...")

        shard_aggregates = []
        step9_meta = {
            "shard_results": {}
        }

        for shard_id, chief in self.shard_chiefs.items():
            # Get updates for this shard
            shard_client_ids = [c.client_id for c in self.clients if c.shard_id == shard_id]
            shard_updates = [u for u in client_updates if u.client_id in shard_client_ids]

            if shard_updates:
                try:
                    # Steps 7-9: Verify + aggregate
                    gradients = [u.gradient for u in shard_updates]
                    proofs = [u.zkp for u in shard_updates]
                    client_ids = [u.client_id for u in shard_updates]

                    shard_aggregate, process_meta = chief.process_round(
                        gradients, proofs, client_ids, self.global_model
                    )

                    # Record in DAG
                    for i, (update, gradient) in enumerate(zip(shard_updates, gradients)):
                        node_id = self.sharded_dag.add_shard_update(
                            shard_id,
                            update.client_id,
                            self.current_round,
                            update.model_hash,
                            update.commitment,
                            "root"
                        )
                        self.sharded_dag.shard_dags[shard_id].verify_update(
                            node_id,
                            zkp_verified=update.zkp.certifies_gradient_correct
                        )

                    shard_model_update = ShardModelUpdate(
                        shard_id=shard_id,
                        shard_model=shard_aggregate,
                        timestamp=time.time(),
                        num_updates_aggregated=len(shard_updates),
                        verification_success_rate=process_meta["verification"]["results"].count(True) / len(shard_updates) if shard_updates else 0,
                    )
                    shard_aggregates.append(shard_model_update)

                    shard_contribution_norm = float(np.linalg.norm(shard_aggregate.flatten()))

                    step9_meta["shard_results"][shard_id] = {
                        "updates_aggregated": len(shard_updates),
                        "verification_rate": shard_model_update.verification_success_rate,
                        "aggregation_norm": shard_contribution_norm,
                    }

                    round_meta["shard_contributions"][shard_id] = {
                        "num_updates": len(shard_updates),
                        "verification_rate": float(shard_model_update.verification_success_rate),
                        "update_norm": shard_contribution_norm,
                    }

                except Exception as e:
                    step9_meta["shard_results"][shard_id] = {
                        "error": str(e),
                        "aggregation_failed": True
                    }
                    if VERBOSE:
                        print(f"  [WARN] Shard {shard_id} aggregation failed: {str(e)[:50]}")

        round_meta["steps"]["7-9_shard_aggregation"] = step9_meta

        # STEP 10: Global aggregation
        if VERBOSE:
            print(f"Step 10: Global aggregation...")

        if shard_aggregates:
            try:
                new_global_model, global_meta = self.global_coordinator.run_training_round(
                    shard_aggregates,
                    global_learning_rate=1.0
                )

                self.global_model = new_global_model
                self.sharded_dag.record_global_update(self._hash_model(new_global_model))

                # Track model changes
                model_change = self.global_model - model_before
                model_change_norm = float(np.linalg.norm(model_change.flatten()))

                round_meta["model_changes"] = {
                    "weight_change_norm": model_change_norm,
                    "global_model_norm": float(np.linalg.norm(self.global_model.flatten())),
                    "model_change_ratio": model_change_norm / (float(np.linalg.norm(model_before.flatten())) + 1e-8),
                }

                round_meta["steps"]["10_global_aggregation"] = global_meta

            except Exception as e:
                round_meta["steps"]["10_global_aggregation"] = {
                    "error": str(e),
                    "aggregation_failed": True
                }
                if VERBOSE:
                    print(f"  ⚠ Global aggregation failed: {str(e)}")

        # Round statistics
        round_elapsed = time.time() - round_start
        round_meta["elapsed_seconds"] = round_elapsed

        self.system_metrics.append(round_meta)

        if VERBOSE:
            self._print_round_summary(round_meta)

        return round_meta

    def execute_full_training(self) -> Dict[str, Any]:
        """
        Execute complete training across all rounds.

        Returns:
            System-wide statistics
        """
        if VERBOSE:
            print(f"\n{'='*60}")
            print(f"SHARDED FEDERATED LEARNING SYSTEM")
            print(f"Clients: {self.num_clients}, Shards: {self.num_shards}, Rounds: {self.num_rounds}")
            print(f"{'='*60}\n")

        training_start = time.time()

        for round_num in range(self.num_rounds):
            self.execute_training_round()

        training_elapsed = time.time() - training_start

        system_stats = {
            "total_training_time_seconds": training_elapsed,
            "num_rounds": self.current_round,
            "num_clients": self.num_clients,
            "num_shards": self.num_shards,
            "clients_per_shard": self.shard_size,
            "total_round_metrics": self.system_metrics,
            "final_system_state": self._get_final_stats(),
        }

        if VERBOSE:
            self._print_final_summary(system_stats)

        return system_stats

    def _get_final_stats(self) -> Dict[str, Any]:
        """Gather final system statistics"""
        return {
            "global_coordinator": self.global_coordinator.get_stats(),
            "shard_chiefs": {
                shard_id: chief.get_stats()
                for shard_id, chief in self.shard_chiefs.items()
            },
            "dag_stats": self.sharded_dag.get_stats(),
            "client_stats": [c.get_stats() for c in self.clients],
        }

    def _print_round_summary(self, round_meta: Dict):
        """Print round summary with detailed gradient and weight tracking"""
        steps = round_meta.get("steps", {})
        client_updates = steps.get("2-6_client_updates", {})
        shard_results = steps.get("7-9_shard_aggregation", {}).get("shard_results", {})

        print(f"\n[CLIENT TRAINING]")
        print(f"  Successful: {client_updates.get('successful_submissions', 0)}/{self.num_clients}")

        # Gradient statistics
        grad_stats = round_meta.get("gradient_stats", {})
        if grad_stats:
            print(f"\n[GRADIENT MAGNITUDES]")
            print(f"  Avg Norm: {grad_stats.get('avg_gradient_norm', 0):.6f}")
            print(f"  Max: {grad_stats.get('max_gradient_norm', 0):.6f}")
            print(f"  Min: {grad_stats.get('min_gradient_norm', 0):.6f}")
            print(f"  StdDev: {grad_stats.get('std_gradient_norm', 0):.6f}")

        # Shard aggregation details
        print(f"\n[SHARD AGGREGATION]")
        for shard_id, result in shard_results.items():
            if "error" not in result:
                print(f"  {shard_id}:")
                print(f"    Updates: {result.get('updates_aggregated', 0)}")
                print(f"    Verification: {result.get('verification_rate', 0):.1%}")
                print(f"    Update Norm: {result.get('aggregation_norm', 0):.6f}")

        # Model weight changes
        model_changes = round_meta.get("model_changes", {})
        if model_changes:
            print(f"\n[MODEL IMPROVEMENTS]")
            print(f"  Weight Change Norm: {model_changes.get('weight_change_norm', 0):.6f}")
            print(f"  Global Model Norm: {model_changes.get('global_model_norm', 0):.6f}")
            print(f"  Change Ratio: {model_changes.get('model_change_ratio', 0):.4%}")

        print(f"  Round Time: {round_meta.get('elapsed_seconds', 0):.2f}s")

    def _print_final_summary(self, stats: Dict):
        """Print final training summary with aggregate metrics"""
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total Time: {stats['total_training_time_seconds']:.2f} seconds")
        print(f"Rounds: {stats['num_rounds']}")
        print(f"Clients: {stats['num_clients']}")
        print(f"Shards: {stats['num_shards']}")

        # Aggregate metrics across all rounds
        all_metrics = stats.get("total_round_metrics", [])

        if all_metrics:
            # Gradient evolution
            grad_norms = []
            model_change_norms = []

            for metric in all_metrics:
                grad_stat = metric.get("gradient_stats", {})
                if grad_stat:
                    grad_norms.append(grad_stat.get("avg_gradient_norm", 0))

                model_change = metric.get("model_changes", {})
                if model_change:
                    model_change_norms.append(model_change.get("weight_change_norm", 0))

            print(f"\n[GRADIENT EVOLUTION]")
            if grad_norms:
                print(f"  Initial Avg Gradient: {grad_norms[0]:.6f}")
                print(f"  Final Avg Gradient: {grad_norms[-1]:.6f}")
                print(f"  Min Avg Gradient: {min(grad_norms):.6f}")
                print(f"  Max Avg Gradient: {max(grad_norms):.6f}")

            print(f"\n[WEIGHT CHANGES]")
            if model_change_norms:
                print(f"  Total Weight Updates: {sum(model_change_norms):.6f}")
                print(f"  Avg Per Round: {np.mean(model_change_norms):.6f}")
                print(f"  Largest Update: {max(model_change_norms):.6f}")

        final_state = stats["final_system_state"]

        print(f"\n[SHARD PERFORMANCE]")
        for shard_id, chief_stats in final_state["shard_chiefs"].items():
            print(f"  {shard_id} ({chief_stats['chief_id']}):")
            print(f"    Verified: {chief_stats['updates_verified']}")
            print(f"    Rejected: {chief_stats['updates_rejected']}")

        print(f"\n[DAG STATUS]")
        dag_stats = final_state["dag_stats"]
        print(f"  Model History: {dag_stats['global_model_history_length']} snapshots")
        for shard_id, shard_stat in dag_stats["shard_stats"].items():
            print(f"  {shard_id}: {shard_stat['total_nodes']} nodes, {shard_stat['finalized_chain_length']} finalized")

        print(f"\n{'='*60}\n")

    def _hash_model(self, model: np.ndarray) -> str:
        """Compute model hash"""
        import hashlib
        model_bytes = model.astype(np.float32).tobytes()
        return hashlib.sha256(model_bytes).hexdigest()

    def _load_pretrained_model(self, filepath: str) -> np.ndarray:
        """
        Load pre-trained model from file using model_registry.
        
        Supports: .npy, .npz, .pt (PyTorch), .safetensors

        Args:
            filepath: Path to model file

        Returns:
            Model weights as numpy array (flattened to 1D)
        """
        import os

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Determine file type and get appropriate loader
        if filepath.endswith('.npy'):
            loader_name = 'numpy'
        elif filepath.endswith('.npz'):
            loader_name = 'numpy_compressed'
        elif filepath.endswith('.pt'):
            loader_name = 'pytorch'
        elif filepath.endswith('.safetensors'):
            loader_name = 'safetensors'
        else:
            raise ValueError(f"Unsupported file format: {filepath}. "
                           "Use .npy, .npz, .pt, or .safetensors")

        try:
            # Load model using registry
            loaded_model = registry.get(loader_name, model_path=filepath, map_location='cpu')
            
            # Extract flattened weights
            weights = self._extract_weights_from_loaded_model(loaded_model, loader_name)
            
            if VERBOSE:
                print(f"[OK] Extracted weights from model (shape: {weights.shape})")

            return weights.astype(np.float32)

        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {e}")

    def _extract_weights_from_loaded_model(self, loaded_model: Any, model_type: str) -> np.ndarray:
        """
        Extract flattened weights from loaded model based on type.

        Args:
            loaded_model: Model object returned by registry
            model_type: Type of model ('numpy', 'numpy_compressed', 'pytorch', 'safetensors')

        Returns:
            Flattened weights as numpy array
        """
        if model_type in ('numpy', 'numpy_compressed'):
            # Already a numpy array
            if isinstance(loaded_model, np.ndarray):
                return loaded_model.flatten()
            else:
                raise ValueError(f"Expected numpy array, got {type(loaded_model)}")

        elif model_type == 'pytorch':
            # PyTorch checkpoint or model object
            return self._extract_weights_from_pytorch(loaded_model)

        elif model_type == 'safetensors':
            # Dictionary of torch tensors from safetensors
            return self._extract_weights_from_dict(loaded_model)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _extract_weights_from_pytorch(self, checkpoint) -> np.ndarray:
        """
        Extract weights from PyTorch model/checkpoint.

        Args:
            checkpoint: Either a PyTorch model, state_dict, or checkpoint dict

        Returns:
            Flattened weights as numpy array
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not available for model extraction")

        # Check if it's a state_dict (from model.state_dict()) or full model
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # It's a full checkpoint
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            # It's already a state_dict
            state_dict = checkpoint
        else:
            # It's a model object
            state_dict = checkpoint.state_dict()

        # Extract all weights and flatten
        weights_list = []

        for layer_name, param in state_dict.items():
            if VERBOSE:
                print(f"  Layer: {layer_name}, shape: {param.shape}")

            # Convert to numpy and flatten
            weights = param.detach().cpu().numpy().flatten()
            weights_list.append(weights)

        # Concatenate all weights
        all_weights = np.concatenate(weights_list)

        if VERBOSE:
            print(f"[OK] Total parameters extracted: {len(all_weights)}")

        return all_weights

    def _extract_weights_from_dict(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extract weights from a dictionary of tensors (e.g., from safetensors).

        Args:
            state_dict: Dict mapping layer names to torch tensors or numpy arrays

        Returns:
            Flattened weights as numpy array
        """
        weights_list = []

        for layer_name, param in state_dict.items():
            if VERBOSE:
                print(f"  Layer: {layer_name}, shape: {param.shape}")

            # Convert to numpy and flatten
            if hasattr(param, 'detach'):
                # torch.Tensor
                weights = param.detach().cpu().numpy().flatten()
            else:
                # Already numpy
                weights = np.asarray(param).flatten()

            weights_list.append(weights)

        # Concatenate all weights
        all_weights = np.concatenate(weights_list)

        if VERBOSE:
            print(f"[OK] Total parameters extracted: {len(all_weights)}")

        return all_weights

    def save_model(self, filepath: str = "global_model.npy"):
        """
        Save current global model to file
        
        Args:
            filepath: Path to save (.npy or .npz)
        """
        if filepath.endswith('.npy'):
            np.save(filepath, self.global_model)
        elif filepath.endswith('.npz'):
            np.savez(filepath, model=self.global_model)
        else:
            np.save(filepath, self.global_model)

        if VERBOSE:
            print(f"[OK] Model saved to {filepath}")

    def save_results(self, filepath: str = "training_results.json"):
        """Save training results to file"""
        results = {
            "system_config": {
                "num_clients": self.num_clients,
                "num_shards": self.num_shards,
                "num_rounds": self.num_rounds,
                "shard_size": self.shard_size,
            },
            "metrics": self.system_metrics,
            "final_stats": self._get_final_stats(),
        }

        # Convert numpy types for JSON serialization
        results_json = json.dumps(results, indent=2, default=str)

        with open(filepath, 'w') as f:
            f.write(results_json)

        if VERBOSE:
            print(f"[OK] Results saved to {filepath}")


def main(pretrained_model_path: str = None, use_real_data: bool = False, real_data_dir: str = "real_federated_data"):
    """
    Run complete system demonstration
    
    Args:
        pretrained_model_path: Optional path to pre-trained model file
                              (e.g., "global_model.npy" or previous training results)
        use_real_data: Whether to load real embeddings instead of synthetic data
        real_data_dir: Directory containing real client data (.npz files)
    """

    # Create and run system
    system = ShardedFederatedLearningSystem(
        num_clients=NUM_CLIENTS,
        num_shards=NUM_SHARDS,
        num_rounds=NUM_ROUNDS,
        pretrained_model_path=pretrained_model_path,
        use_real_data=use_real_data,
        real_data_dir=real_data_dir
    )

    results = system.execute_full_training()

    # Save results
    system.save_results("training_results.json")

    # Save final model for next run
    system.save_model("global_model.npy")

    return system, results


if __name__ == "__main__":
    import sys
    import os

    # Parse command line arguments
    pretrained_model = None
    use_real_data = False
    real_data_dir = "real_federated_data"

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.endswith(".pt") or arg.endswith(".npy") or "model" in arg.lower():
                pretrained_model = arg
                print(f"[MODEL] Using pre-trained model: {pretrained_model}")
            elif arg == "--real-data" and len(sys.argv) > sys.argv.index(arg) + 1:
                real_data_dir = sys.argv[sys.argv.index(arg) + 1]
                use_real_data = True
                print(f"[DATA] Using real embeddings from: {real_data_dir}")
            elif arg.startswith("--real-data="):
                real_data_dir = arg.split("=")[1]
                use_real_data = True
                print(f"[DATA] Using real embeddings from: {real_data_dir}")

    # Check if real data directory exists when requested
    if use_real_data and not os.path.exists(real_data_dir):
        print(f"[ERROR] Real data directory not found: {real_data_dir}")
        print(f"[ERROR] Please run: python generate_embeddings.py")
        sys.exit(1)

    if use_real_data:
        print(f"\n[INFO] Real Data Mode:")
        print(f"  - Loading embeddings from: {real_data_dir}")
        print(f"  - Model: {pretrained_model if pretrained_model else 'random init'}")
        print()

    system, results = main(pretrained_model_path=pretrained_model, use_real_data=use_real_data, real_data_dir=real_data_dir)
