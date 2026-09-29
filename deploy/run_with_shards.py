"""
Federated Learning with Shard Architecture
- Real model.pt (384 → 256 → 2)
- Real embeddings from real_federated_data/
- Each client gets DIFFERENT data slice (non-IID)
- SHARD ARCHITECTURE:yes
  * Divide clients into shards
  * Each shard has a chief (elected)
  * Shard-level aggregation (local)
  * Global aggregation (across shards)
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Model registry for framework-agnostic loading
from model_registry import registry


class RealModelWrapper(nn.Module):
    """Wrapper for real pretrained model"""

    def __init__(self, model_path: str = 'model.pt', learning_rate: float = 0.001, state_dict: Dict = None):
        super().__init__()

        # Load real model state either from provided state_dict or from disk
        if state_dict is None:
            # Use map_location='cpu' to be conservative and avoid weights_only quirks
            state_dict = torch.load(model_path, map_location='cpu')

        # Build neural network matching the expected state_dict
        self.layers = nn.Sequential(
            nn.Linear(384, 256),  # Layer 0: 384 → 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)     # Layer 3: 256 → 2
        )

        # Load weights from real model using helper method
        self.load_real_weights(state_dict)

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        print("✅ Model loaded from model.pt (384 → 256 → 2)")

    def load_real_weights(self, state_dict: Dict):
        """Safely load weights from pretrained model using .data.copy_()"""
        with torch.no_grad():
            self.layers[0].weight.data.copy_(state_dict['0.weight'])
            self.layers[0].bias.data.copy_(state_dict['0.bias'])
            self.layers[3].weight.data.copy_(state_dict['3.weight'])
            self.layers[3].bias.data.copy_(state_dict['3.bias'])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        was_training = self.layers.training
        self.layers.eval()
        x_tensor = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = self.layers(x_tensor)
            probs = torch.softmax(logits, dim=1)
        if was_training:
            self.layers.train()
        return probs.numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute loss"""
        was_training = self.layers.training
        self.layers.eval()
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.LongTensor(y)

        with torch.no_grad():
            logits = self.layers(x_tensor)
            loss = self.criterion(logits, y_tensor)
        if was_training:
            self.layers.train()
        return loss.item()

    def train_step(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Single training step"""
        self.layers.train()
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.LongTensor(y)

        self.optimizer.zero_grad()
        logits = self.layers(x_tensor)
        loss = self.criterion(logits, y_tensor)
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        predictions = torch.argmax(logits, dim=1).numpy()
        accuracy = np.mean(predictions == y)

        return {'loss': float(loss.item()), 'accuracy': float(accuracy)}

    def get_weights(self) -> Dict:
        """Get model weights for aggregation"""
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data.cpu().numpy().copy()
        return weights

    def set_weights(self, weights: Dict):
        """Set model weights from aggregation"""
        for name, param in self.named_parameters():
            if name in weights:
                param.data.copy_(torch.from_numpy(weights[name]))


class ShardManager:
    """Manages shards of clients"""

    def __init__(self, num_clients: int, num_shards: int):
        self.num_clients = num_clients
        self.num_shards = num_shards
        self.shards = {}
        self.shard_chiefs = {}
        self.shard_accuracies = {}

    def create_shards(self, client_list: List[str]):
        """Divide clients into shards"""
        clients_per_shard = len(client_list) // self.num_shards

        for shard_id in range(self.num_shards):
            start_idx = shard_id * clients_per_shard
            if shard_id == self.num_shards - 1:
                # Last shard gets remaining clients
                end_idx = len(client_list)
            else:
                end_idx = start_idx + clients_per_shard

            self.shards[shard_id] = client_list[start_idx:end_idx]
            print(f"  Shard {shard_id}: {len(self.shards[shard_id])} clients")

    def elect_shard_chief(self, shard_id: int, client_accuracies: Dict[str, float]):
        """Elect shard chief based on accuracy"""
        shard_clients = self.shards[shard_id]

        # Find best performer in shard
        best_client = None
        best_accuracy = -1

        for client_id in shard_clients:
            if client_id in client_accuracies:
                acc = client_accuracies[client_id]
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_client = client_id

        self.shard_chiefs[shard_id] = best_client
        return best_client

    def aggregate_shard(self, shard_id: int, client_models: Dict,
                       client_data: Dict) -> Tuple[Dict, float]:
        """Aggregate models within a shard (FedAvg) using test data for metrics"""
        shard_clients = self.shards[shard_id]

        # Calculate total samples in shard
        total_samples = sum(client_data[cid]['num_samples'] for cid in shard_clients)

        # Weighted average
        agg_weights = None
        shard_test_loss = 0

        for client_id in shard_clients:
            num_samples = client_data[client_id]['num_samples']
            weight = num_samples / total_samples

            client_weights = client_models[client_id].get_weights()

            if agg_weights is None:
                agg_weights = {k: v * weight for k, v in client_weights.items()}
            else:
                for k in agg_weights:
                    agg_weights[k] += client_weights[k] * weight

            # Accumulate TEST loss for shard metric
            X_test = client_data[client_id]['X_test']
            y_test = client_data[client_id]['y_test']
            shard_test_loss += client_models[client_id].compute_loss(X_test, y_test)

        avg_shard_test_loss = shard_test_loss / len(shard_clients)
        self.shard_accuracies[shard_id] = avg_shard_test_loss

        return agg_weights, avg_shard_test_loss

    def broadcast_to_shard(self, shard_id: int, shard_weights: Dict,
                          client_models: Dict):
        """Broadcast aggregated weights to all clients in shard"""
        shard_clients = self.shards[shard_id]

        for client_id in shard_clients:
            client_models[client_id].set_weights(shard_weights)


class FederatedLearnerWithShardsAndRealModel:
    """Federated learning with SHARD ARCHITECTURE + real model + non-IID data"""

    def __init__(self, data_dir: str = "real_federated_data",
                 num_clients: int = 12, num_shards: int = 3):
        self.data_dir = Path(data_dir)
        self.num_clients = num_clients
        self.num_shards = num_shards
        self.clients_data = {}
        self.client_models = {}
        self.training_history = []
        self.shard_manager = ShardManager(num_clients, num_shards)

    def load_real_data(self) -> bool:
        """Load real data per client file and create 80/20 train-test split for each client."""
        print("\n" + "="*70)
        print("📚 LOADING REAL DATA (Client Files + 80/20 Train-Test Split)")
        print("="*70)

        if not self.data_dir.exists():
            print(f"❌ Directory not found: {self.data_dir}")
            return False

        client_files = sorted(self.data_dir.glob("client_*.npz"))
        if not client_files:
            print(f"❌ No client_*.npz files found")
            return False

        if len(client_files) < self.num_clients:
            print(f"❌ Requested {self.num_clients} clients, but only found {len(client_files)} files")
            return False

        if len(client_files) > self.num_clients:
            print(f"⚠️  Found {len(client_files)} client files, using first {self.num_clients}")
            client_files = client_files[:self.num_clients]

        print(f"\n🔍 Found {len(client_files)} client files")

        # Load each client file directly to avoid cross-client leakage.
        total_samples = 0

        for client_file in client_files:
            data = np.load(client_file)
            X = data['X'] if 'X' in data.files else data['embeddings']
            y = data['y'] if 'y' in data.files else data['labels']

            # Convert one-hot to class indices if needed
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
            else:
                y = y.flatten()

            if len(X) < 2:
                print(f"❌ Not enough samples in {client_file.stem} (need at least 2)")
                return False

            # Deterministic split per client file for reproducibility.
            client_id = client_file.stem.replace("client_", "")
            rng = np.random.default_rng(42 + int(client_id))
            indices = rng.permutation(len(X))

            train_size = int(len(X) * 0.8)
            train_size = max(1, min(train_size, len(X) - 1))
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]

            X_train = X[train_idx].astype(np.float32)
            y_train = y[train_idx].astype(int)
            X_test = X[test_idx].astype(np.float32)
            y_test = y[test_idx].astype(int)

            class_ratio = float(np.mean(y_train == 0))

            self.clients_data[str(client_id)] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'num_samples': len(X_train),
                'class_ratio': class_ratio
            }

            total_samples += len(X)

            train_ham = int(np.sum(y_train == 0))
            train_spam = int(np.sum(y_train == 1))
            print(f"  ✓ {client_file.stem}: total={len(X)}, train={len(X_train)}, test={len(X_test)}, "
                  f"train_ham={train_ham}, train_spam={train_spam}")

        print(f"\n📊 Total data: {total_samples} samples")
        print(f"\n✅ Data distributed to {len(self.clients_data)} clients (Non-IID)")
        print("✅ Each client has 80% training data + 20% test data")
        return True

    def initialize_clients(self) -> bool:
        """Initialize real model for each client"""
        print("\n" + "="*70)
        print("🤖 INITIALIZING REAL MODELS FOR EACH CLIENT")
        print("="*70)

        try:
            # Prefer safetensors model if present, otherwise fallback to model.pt
            preferred_path = Path('model.safetensors') if Path('model.safetensors').exists() else Path('model.pt')
            loader_name = 'safetensors' if preferred_path.suffix == '.safetensors' else 'pytorch'

            for client_id in self.clients_data.keys():
                # Use model registry to instantiate model; allows different backends
                try:
                    self.client_models[client_id] = registry.get(
                        loader_name,
                        model_path=str(preferred_path),
                        wrapper_class=RealModelWrapper,
                        learning_rate=0.001
                    )
                except Exception as e:
                    print(f"⚠️  Loader '{loader_name}' failed for client {client_id}: {e}")
                    print("   Falling back to 'pytorch' loader with model.pt")
                    self.client_models[client_id] = registry.get(
                        'pytorch',
                        model_path='model.pt',
                        wrapper_class=RealModelWrapper,
                        learning_rate=0.001
                    )

            print(f"\n✅ Initialized {len(self.client_models)} real models")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def create_shards(self):
        """Create shard structure"""
        print("\n" + "="*70)
        print("🔀 CREATING SHARD STRUCTURE")
        print("="*70)
        print(f"\nDividing {self.num_clients} clients into {self.num_shards} shards:")

        client_list = list(self.clients_data.keys())
        self.shard_manager.create_shards(client_list)

    def train_round(self, round_num: int, local_epochs: int = 2) -> Dict:
        """Execute federated learning round with shard architecture"""
        print(f"\n{'='*70}")
        print(f"🔄 FEDERATED ROUND {round_num}")
        print(f"{'='*70}")

        round_start = time.time()
        round_metrics = {
            'round': round_num,
            'client_metrics': {},
            'shard_metrics': {},
            'aggregated_metrics': {}
        }

        # Phase 1: Local training on each client (using 80% training data)
        print("\n[Phase 1: Local Client Training (80% Training Data)]")
        print("-" * 70)

        client_test_accuracies = {}

        for client_id, data in self.clients_data.items():
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            model = self.client_models[client_id]

            # Train locally on training data
            for epoch in range(local_epochs):
                indices = np.random.permutation(len(X_train))

                for i in range(0, len(X_train), 32):
                    batch_idx = indices[i:min(i+32, len(X_train))]
                    X_batch = X_train[batch_idx]
                    y_batch = y_train[batch_idx]

                    model.train_step(X_batch, y_batch)

            # Compute metrics on TEST data (20% held-out)
            test_loss = model.compute_loss(X_test, y_test)
            test_predictions = model.predict(X_test)
            test_accuracy = np.mean(test_predictions == y_test)

            # Also track training metrics for reference
            train_loss = model.compute_loss(X_train, y_train)
            train_predictions = model.predict(X_train)
            train_accuracy = np.mean(train_predictions == y_train)

            client_test_accuracies[client_id] = test_accuracy

            round_metrics['client_metrics'][client_id] = {
                'train_loss': float(train_loss),
                'train_accuracy': float(train_accuracy),
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'num_train_samples': len(X_train),
                'num_test_samples': len(X_test)
            }

            print(f"  Client {client_id:2s}: "
                  f"Train(L={train_loss:.4f},Acc={train_accuracy:.4f}) | "
                  f"Test(L={test_loss:.4f},Acc={test_accuracy:.4f})")

        # Phase 2: Shard-level aggregation
        print("\n[Phase 2: Shard-Level Aggregation (using TEST accuracy)]")
        print("-" * 70)

        shard_weights_dict = {}
        shard_test_losses = {}

        for shard_id in range(self.num_shards):
            # Elect shard chief based on TEST accuracy
            chief_id = self.shard_manager.elect_shard_chief(shard_id, client_test_accuracies)

            # Aggregate shard models
            shard_weights, avg_loss = self.shard_manager.aggregate_shard(
                shard_id, self.client_models, self.clients_data
            )

            shard_weights_dict[shard_id] = shard_weights
            shard_test_losses[shard_id] = avg_loss

            # Broadcast to shard
            self.shard_manager.broadcast_to_shard(
                shard_id, shard_weights, self.client_models
            )

            shard_clients_count = len(self.shard_manager.shards[shard_id])
            print(f"  Shard {shard_id}: {shard_clients_count} clients, "
                  f"chief={chief_id}, test_loss={avg_loss:.6f}")

            round_metrics['shard_metrics'][shard_id] = {
                'chief': chief_id,
                'num_clients': shard_clients_count,
                'avg_test_loss': float(avg_loss)
            }

        # Phase 3: Global aggregation
        print("\n[Phase 3: Global Aggregation]")
        print("-" * 70)

        # Aggregate all shard models globally
        global_weights = None

        for shard_id, shard_weights in shard_weights_dict.items():
            if global_weights is None:
                global_weights = {k: v / self.num_shards for k, v in shard_weights.items()}
            else:
                for k in global_weights:
                    global_weights[k] += shard_weights[k] / self.num_shards

        # Broadcast global model back to all clients
        for model in self.client_models.values():
            model.set_weights(global_weights)

        print("  ✓ Global model aggregated from all shards")
        print("  ✓ Global model broadcast to all clients")

        # Compute final metrics
        avg_test_loss = np.mean(list(shard_test_losses.values()))
        avg_test_acc = np.mean(list(client_test_accuracies.values()))

        round_metrics['aggregated_metrics'] = {
            'avg_test_loss': float(avg_test_loss),
            'avg_test_accuracy': float(avg_test_acc),
            'num_clients': self.num_clients,
            'num_shards': self.num_shards
        }

        round_time = time.time() - round_start

        print(f"\n📊 Round {round_num} Results:")
        print(f"   Average Test Loss:     {avg_test_loss:.6f}")
        print(f"   Average Test Accuracy: {avg_test_acc:.4f}")
        print(f"   Time:                  {round_time:.2f}s")

        return round_metrics

    def run_training(self, num_rounds: int = 5, local_epochs: int = 2):
        """Run complete training"""
        print("\n" + "="*70)
        print("🚀 STARTING FEDERATED TRAINING WITH SHARD ARCHITECTURE")
        print("="*70)

        for round_num in range(1, num_rounds + 1):
            round_metrics = self.train_round(round_num, local_epochs)
            self.training_history.append(round_metrics)
            time.sleep(0.5)

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)

    def save_results(self, output_file: str = "shard_model_results.json"):
        """Save results"""
        print(f"\n💾 Saving to {output_file}...")

        results = {
            'system_config': {
                'model': 'Real model.pt (384 → 256 → 2)',
                'num_clients': self.num_clients,
                'num_shards': self.num_shards,
                'architecture': 'Shard-based hierarchy',
                'distribution': 'Non-IID (direct client files)',
                'real_embeddings': True,
                'total_samples': sum(d['num_samples'] for d in self.clients_data.values()),
                'embedding_dim': 384
            },
            'training_history': self.training_history
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ Saved!")

    def save_final_model(self, output_model_file: str = "trained_model.pt", update_base_model: bool = True):
        """Persist final global model weights for downstream inference."""
        if not self.client_models:
            print("⚠️  No trained client models found; skipping model save")
            return

        reference_model = next(iter(self.client_models.values()))
        state_dict = {
            '0.weight': reference_model.layers[0].weight.detach().cpu(),
            '0.bias': reference_model.layers[0].bias.detach().cpu(),
            '3.weight': reference_model.layers[3].weight.detach().cpu(),
            '3.bias': reference_model.layers[3].bias.detach().cpu(),
        }

        torch.save(state_dict, output_model_file)
        print(f"✅ Saved final trained model to {output_model_file}")

        if update_base_model:
            torch.save(state_dict, "model.pt")
            print("✅ Updated model.pt with final trained global weights")
        # Attempt to save a safetensors copy if the library is available
        try:
            import safetensors.torch as st
            # safetensors expects plain tensors or numpy arrays
            st_state = {k: v.cpu() for k, v in state_dict.items()}
            st.save_file(st_state, "trained_model.safetensors")
            print("✅ Saved final trained model to trained_model.safetensors")
            if update_base_model:
                st.save_file(st_state, "model.safetensors")
                print("✅ Updated model.safetensors with final trained global weights")
        except Exception:
            # If safetensors isn't installed or saving fails, continue silently
            pass

    def print_summary(self):
        """Print final summary"""
        if not self.training_history:
            return

        print("\n" + "="*70)
        print("📈 FINAL SUMMARY (Test Set Evaluation)")
        print("="*70)

        first_round = self.training_history[0]['aggregated_metrics']
        last_round = self.training_history[-1]['aggregated_metrics']

        loss_improve = first_round['avg_test_loss'] - last_round['avg_test_loss']
        acc_improve = last_round['avg_test_accuracy'] - first_round['avg_test_accuracy']
        loss_pct = (loss_improve / first_round['avg_test_loss']) * 100

        print(f"\nImproved over {len(self.training_history)} rounds:")
        print(f"  Test Loss:     {first_round['avg_test_loss']:.6f} → {last_round['avg_test_loss']:.6f} "
              f"(↓ {loss_improve:.6f}, -{loss_pct:.1f}%)")
        print(f"  Test Accuracy: {first_round['avg_test_accuracy']:.4f} → {last_round['avg_test_accuracy']:.4f} "
              f"(↑ {acc_improve:.4f})")
        print(f"\nArchitecture: {self.num_clients} clients → {self.num_shards} shards → Global")
        print(f"Data Split:   80% training per client, 20% test per client (held-out)")
        print(f"Results:      shard_model_results.json\n")

    def run(self, num_rounds: int = 5, local_epochs: int = 2):
        """Execute complete pipeline"""
        try:
            if not self.load_real_data():
                return

            if not self.initialize_clients():
                return

            self.create_shards()
            self.run_training(num_rounds, local_epochs)
            self.save_results()
            self.save_final_model()
            self.print_summary()

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    learner = FederatedLearnerWithShardsAndRealModel(
        num_clients=12,
        num_shards=3
    )
    learner.run(num_rounds=5, local_epochs=2)
