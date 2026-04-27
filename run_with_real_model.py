"""
Federated Learning with REAL Model.pt
- Uses actual trained model (384 → 256 → 2 classes)
- Real embeddings from real_federated_data/
- Each client gets DIFFERENT data slice (non-IID)
- Fine-tunes on client-specific data
- Aggregates using FedAvg
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


class RealModelWrapper(nn.Module):
    """Wrapper for real pretrained model"""
    
    def __init__(self, model_path: str = 'model.pt', learning_rate: float = 0.001):
        super().__init__()
        
        # Load real model state
        state_dict = torch.load(model_path)
        
        # Build neural network matching the state_dict
        self.layers = nn.Sequential(
            nn.Linear(384, 256),  # Layer 0: 384 → 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)     # Layer 3: 256 → 2
        )
        
        # Load weights from real model
        self.layers[0].weight = nn.Parameter(state_dict['0.weight'])
        self.layers[0].bias = nn.Parameter(state_dict['0.bias'])
        self.layers[3].weight = nn.Parameter(state_dict['3.weight'])
        self.layers[3].bias = nn.Parameter(state_dict['3.bias'])
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        print("✅ Model loaded from model.pt (384 → 256 → 2)")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        x_tensor = torch.FloatTensor(x)
        with torch.no_grad():
            logits = self.layers(x_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
    
    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute loss"""
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.LongTensor(y)
        
        logits = self.layers(x_tensor)
        loss = self.criterion(logits, y_tensor)
        return loss.item()
    
    def train_step(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Single training step"""
        x_tensor = torch.FloatTensor(x)
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
        for name, param in self.layers.named_parameters():
            weights[name] = param.data.cpu().numpy().copy()
        return weights
    
    def set_weights(self, weights: Dict):
        """Set model weights from aggregation"""
        for name, param in self.layers.named_parameters():
            if name in weights:
                param.data = torch.FloatTensor(weights[name])


class FederatedLearnerWithRealModel:
    """Federated learning with real model and non-IID data distribution"""
    
    def __init__(self, data_dir: str = "real_federated_data", num_clients: int = 12):
        self.data_dir = Path(data_dir)
        self.num_clients = num_clients
        self.clients_data = {}
        self.client_models = {}
        self.training_history = []
        
    def load_real_data(self) -> bool:
        """Load real data and create NON-IID splits per client"""
        print("\n" + "="*70)
        print("📚 LOADING REAL DATA (Non-IID Distribution)")
        print("="*70)
        
        if not self.data_dir.exists():
            print(f"❌ Directory not found: {self.data_dir}")
            return False
        
        client_files = sorted(self.data_dir.glob("client_*.npz"))
        if not client_files:
            print(f"❌ No client_*.npz files found")
            return False
        
        print(f"\n🔍 Found {len(client_files)} client files")
        
        # Load all data first
        all_X = []
        all_y = []
        total_samples = 0
        
        for client_file in client_files:
            data = np.load(client_file)
            X = data['X']
            y = data['y']
            
            # Convert one-hot to class indices if needed
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
            else:
                y = y.flatten()
            
            all_X.append(X)
            all_y.append(y)
            total_samples += len(X)
            print(f"  ✓ {client_file.stem}: {len(X)} samples")
        
        # Concatenate all data
        all_X = np.vstack(all_X)  # (total, 384)
        all_y = np.concatenate(all_y)  # (total,)
        
        print(f"\n📊 Total data: {len(all_X)} samples")
        
        # Create NON-IID splits: each client gets specific class distribution
        print("\n🔀 Creating Non-IID distribution (each client has different mix):")
        
        # Separate by class
        class_0_idx = np.where(all_y == 0)[0]
        class_1_idx = np.where(all_y == 1)[0]
        
        samples_per_client = len(all_X) // self.num_clients
        
        for client_id in range(self.num_clients):
            # Vary the class ratio for each client (0%-100% to 100%-0%)
            class_ratio = client_id / (self.num_clients - 1) if self.num_clients > 1 else 0.5
            
            # Sample indices
            num_class_0 = int(samples_per_client * class_ratio)
            num_class_1 = samples_per_client - num_class_0
            
            idx_0 = np.random.choice(class_0_idx, min(num_class_0, len(class_0_idx)), replace=False)
            idx_1 = np.random.choice(class_1_idx, min(num_class_1, len(class_1_idx)), replace=False)
            
            client_idx = np.concatenate([idx_0, idx_1])
            np.random.shuffle(client_idx)
            
            X_client = all_X[client_idx]
            y_client = all_y[client_idx]
            
            self.clients_data[str(client_id)] = {
                'X': X_client.astype(np.float32),
                'y': y_client.astype(int),
                'num_samples': len(X_client),
                'class_ratio': class_ratio
            }
            
            # Count classes
            n_class_0 = np.sum(y_client == 0)
            n_class_1 = np.sum(y_client == 1)
            print(f"  Client {client_id:2d}: {len(X_client):4d} samples (class 0: {n_class_0}%, class 1: {n_class_1}%)")
        
        print(f"\n✅ Data distributed to {len(self.clients_data)} clients (Non-IID)")
        return True
    
    def initialize_clients(self) -> bool:
        """Initialize real model for each client"""
        print("\n" + "="*70)
        print("🤖 INITIALIZING REAL MODEL (model.pt) FOR EACH CLIENT")
        print("="*70)
        
        try:
            for client_id in self.clients_data.keys():
                self.client_models[client_id] = RealModelWrapper(
                    model_path='model.pt',
                    learning_rate=0.001
                )
            
            print(f"\n✅ Initialized {len(self.client_models)} real models")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def train_round(self, round_num: int, local_epochs: int = 2) -> Dict:
        """Execute federated learning round"""
        print(f"\n{'='*70}")
        print(f"🔄 FEDERATED ROUND {round_num}")
        print(f"{'='*70}")
        
        round_start = time.time()
        round_metrics = {
            'round': round_num,
            'client_metrics': {},
            'aggregated_metrics': {}
        }
        
        print("\n[Local Training Phase]")
        print("-" * 70)
        
        total_loss = 0
        total_acc = 0
        
        # Local training on each client
        for client_id, data in self.clients_data.items():
            X = data['X']
            y = data['y']
            model = self.client_models[client_id]
            
            # Train for multiple epochs on client data
            for epoch in range(local_epochs):
                indices = np.random.permutation(len(X))
                
                for i in range(0, len(X), 32):
                    batch_idx = indices[i:min(i+32, len(X))]
                    X_batch = X[batch_idx]
                    y_batch = y[batch_idx]
                    
                    model.train_step(X_batch, y_batch)
            
            # Compute metrics
            final_loss = model.compute_loss(X, y)
            predictions = model.predict(X)
            accuracy = np.mean(predictions == y)
            
            round_metrics['client_metrics'][client_id] = {
                'loss': float(final_loss),
                'accuracy': float(accuracy),
                'samples': len(X),
                'class_ratio': data['class_ratio']
            }
            
            total_loss += final_loss
            total_acc += accuracy
            
            print(f"  Client {client_id:2s}: loss={final_loss:.6f}, acc={accuracy:.4f}")
        
        # Aggregate models (FedAvg)
        print("\n[Model Aggregation Phase (FedAvg)]")
        print("-" * 70)
        
        self.aggregate_models()
        
        # Compute aggregated metrics
        num_clients = len(self.clients_data)
        avg_loss = total_loss / num_clients
        avg_acc = total_acc / num_clients
        
        round_metrics['aggregated_metrics'] = {
            'avg_loss': float(avg_loss),
            'avg_accuracy': float(avg_acc),
            'num_clients': num_clients
        }
        
        round_time = time.time() - round_start
        
        print(f"\n📊 Round {round_num} Results:")
        print(f"   Average Loss:     {avg_loss:.6f}")
        print(f"   Average Accuracy: {avg_acc:.4f}")
        print(f"   Time:             {round_time:.2f}s")
        
        return round_metrics
    
    def aggregate_models(self):
        """FedAvg aggregation"""
        if not self.client_models:
            return
        
        # Calculate total samples for weighting
        total_samples = sum(d['num_samples'] for d in self.clients_data.values())
        
        # Initialize aggregated weights
        agg_weights = None
        
        # Weighted averaging
        for client_id, model in self.client_models.items():
            num_samples = self.clients_data[client_id]['num_samples']
            weight = num_samples / total_samples
            
            client_weights = model.get_weights()
            
            if agg_weights is None:
                agg_weights = {k: v * weight for k, v in client_weights.items()}
            else:
                for k in agg_weights:
                    agg_weights[k] += client_weights[k] * weight
        
        # Update all clients with aggregated weights
        for model in self.client_models.values():
            model.set_weights(agg_weights)
        
        print("  ✓ Models aggregated using FedAvg")
    
    def run_training(self, num_rounds: int = 5, local_epochs: int = 2):
        """Run complete training"""
        print("\n" + "="*70)
        print("🚀 STARTING FEDERATED TRAINING")
        print("="*70)
        
        for round_num in range(1, num_rounds + 1):
            round_metrics = self.train_round(round_num, local_epochs)
            self.training_history.append(round_metrics)
            time.sleep(0.5)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
    
    def save_results(self, output_file: str = "real_model_results.json"):
        """Save results"""
        print(f"\n💾 Saving to {output_file}...")
        
        results = {
            'system_config': {
                'model': 'Real model.pt (384 → 256 → 2)',
                'num_clients': len(self.clients_data),
                'distribution': 'Non-IID (each client different class mix)',
                'real_embeddings': True,
                'total_samples': sum(d['num_samples'] for d in self.clients_data.values()),
                'embedding_dim': 384
            },
            'training_history': self.training_history
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved!")
    
    def print_summary(self):
        """Print final summary"""
        if not self.training_history:
            return
        
        print("\n" + "="*70)
        print("📈 FINAL SUMMARY")
        print("="*70)
        
        first_round = self.training_history[0]['aggregated_metrics']
        last_round = self.training_history[-1]['aggregated_metrics']
        
        loss_improve = first_round['avg_loss'] - last_round['avg_loss']
        acc_improve = last_round['avg_accuracy'] - first_round['avg_accuracy']
        loss_pct = (loss_improve / first_round['avg_loss']) * 100
        
        print(f"\nImproved over {len(self.training_history)} rounds:")
        print(f"  Loss:     {first_round['avg_loss']:.6f} → {last_round['avg_loss']:.6f} "
              f"(↓ {loss_improve:.6f}, -{loss_pct:.1f}%)")
        print(f"  Accuracy: {first_round['avg_accuracy']:.4f} → {last_round['avg_accuracy']:.4f} "
              f"(↑ {acc_improve:.4f})")
        print(f"\nClients: {first_round['num_clients']}")
        print(f"Data distribution: Non-IID (each client has different class mix)")
        print(f"Results: real_model_results.json\n")
    
    def run(self, num_rounds: int = 5, local_epochs: int = 2):
        """Execute complete pipeline"""
        try:
            if not self.load_real_data():
                return
            
            if not self.initialize_clients():
                return
            
            self.run_training(num_rounds, local_epochs)
            self.save_results()
            self.print_summary()
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    learner = FederatedLearnerWithRealModel()
    learner.run(num_rounds=5, local_epochs=2)
