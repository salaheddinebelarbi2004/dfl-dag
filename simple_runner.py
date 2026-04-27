"""
SIMPLE SYSTEM RUNNER - Run Complete Federated Learning Pipeline
================================================================

The easiest way to run everything without manual node management.
Just run: python simple_runner.py
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

# Import local modules
from lightweight_classifier import LightweightClassifier
from federated_learning_config import NUM_ROUNDS, LOCAL_EPOCHS, LEARNING_RATE


class SimpleFederatedRunner:
    """Simple runner for federated learning with real data"""
    
    def __init__(self, data_dir: str = "real_federated_data"):
        self.data_dir = Path(data_dir)
        self.clients_data = {}
        self.client_models = {}
        self.training_history = []
        
    def load_real_data(self) -> bool:
        """Load real federated data"""
        print("\n" + "="*70)
        print("📚 LOADING REAL FEDERATED DATA")
        print("="*70)
        
        if not self.data_dir.exists():
            print(f"❌ Directory not found: {self.data_dir}")
            return False
        
        client_files = sorted(self.data_dir.glob("client_*.npz"))
        if not client_files:
            print(f"❌ No client_*.npz files found in {self.data_dir}")
            return False
        
        print(f"\n🔍 Found {len(client_files)} client datasets")
        
        total_samples = 0
        for client_file in client_files:
            client_id = client_file.stem.replace("client_", "")
            
            try:
                data = np.load(client_file)
                
                # Handle different naming conventions
                if 'embeddings' in data.files:
                    embeddings = data['embeddings']
                elif 'X' in data.files:
                    embeddings = data['X']
                else:
                    # Try first array
                    embeddings = data[data.files[0]]
                
                if 'labels' in data.files:
                    labels = data['labels']
                elif 'y' in data.files:
                    labels = data['y']
                else:
                    # Try second array
                    labels = data[data.files[1]]
                
                # Convert one-hot to class indices if needed
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    labels = np.argmax(labels, axis=1)
                else:
                    labels = labels.flatten()
                
                self.clients_data[client_id] = {
                    'X': embeddings.astype(np.float32),
                    'y': labels.astype(int),
                    'num_samples': len(embeddings)
                }
                
                total_samples += len(embeddings)
                dim = embeddings.shape[1]
                print(f"  ✓ Client {client_id:2s}: {len(embeddings):5d} samples, dim={dim}")
                
            except Exception as e:
                print(f"  ⚠️  Could not load {client_file}: {e}")
        
        print(f"\n📊 Total data loaded: {total_samples} samples")
        return len(self.clients_data) > 0
    
    def initialize_classifiers(self) -> bool:
        """Initialize classifiers"""
        print("\n" + "="*70)
        print("🤖 INITIALIZING LIGHTWEIGHT CLASSIFIERS")
        print("="*70)
        
        if not self.clients_data:
            print("❌ No client data loaded!")
            return False
        
        # Get embedding dimension
        first_client_id = list(self.clients_data.keys())[0]
        input_dim = self.clients_data[first_client_id]['X'].shape[1]
        output_dim = 2  # Binary classification (spam/ham)
        
        print(f"\n🔧 Creating classifiers:")
        print(f"   Input dimension:  {input_dim}")
        print(f"   Output classes:   {output_dim}")
        print(f"   Learning rate:    {LEARNING_RATE}")
        print(f"   Local epochs:     {LOCAL_EPOCHS}")
        
        # Create classifier for each client
        for client_id in self.clients_data.keys():
            self.client_models[client_id] = LightweightClassifier(
                input_dim=input_dim,
                output_dim=output_dim,
                learning_rate=LEARNING_RATE
            )
        
        print(f"\n✅ Created {len(self.client_models)} classifiers")
        return True
    
    def train_round(self, round_num: int) -> Dict:
        """Execute one federated learning round"""
        print(f"\n{'='*70}")
        print(f"🔄 FEDERATED LEARNING ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'='*70}")
        
        round_start = time.time()
        round_metrics = {
            'round': round_num,
            'client_metrics': {},
            'aggregated_metrics': {}
        }
        
        # ========== LOCAL TRAINING ==========
        print("\n[Local Training Phase]")
        print("-" * 70)
        
        total_loss = 0
        total_acc = 0
        num_clients = len(self.clients_data)
        
        for client_id, data in self.clients_data.items():
            X = data['X']
            y = data['y']
            model = self.client_models[client_id]
            
            # Train locally
            client_losses = []
            
            for epoch in range(LOCAL_EPOCHS):
                # Mini-batch training
                indices = np.random.permutation(len(X))
                batch_size = 32
                
                for i in range(0, len(X), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    X_batch = X[batch_idx]
                    y_batch = y[batch_idx]
                    
                    # Single training step
                    result = model.train_step(X_batch, y_batch)
                    client_losses.append(result['loss'])
            
            # Final metrics
            final_loss = model.compute_loss(X, y)
            predictions = model.predict(X)
            accuracy = np.mean(predictions == y)
            
            round_metrics['client_metrics'][client_id] = {
                'loss': float(final_loss),
                'accuracy': float(accuracy),
                'samples': len(X)
            }
            
            total_loss += final_loss
            total_acc += accuracy
            
            print(f"  Client {client_id}: loss={final_loss:.6f}, acc={accuracy:.4f}")
        
        # ========== MODEL AGGREGATION ==========
        print("\n[Model Aggregation Phase (FedAvg)]")
        print("-" * 70)
        
        self.aggregate_models_fedavg()
        
        # ========== METRICS ==========
        avg_loss = total_loss / num_clients
        avg_acc = total_acc / num_clients
        
        round_metrics['aggregated_metrics'] = {
            'avg_loss': float(avg_loss),
            'avg_accuracy': float(avg_acc),
            'num_clients': num_clients
        }
        
        round_time = time.time() - round_start
        
        print(f"\n📊 Round {round_num} Summary:")
        print(f"   Average Loss:     {avg_loss:.6f}")
        print(f"   Average Accuracy: {avg_acc:.4f}")
        print(f"   Time:             {round_time:.2f}s")
        
        return round_metrics
    
    def aggregate_models_fedavg(self):
        """Aggregate models using FedAvg (weighted average)"""
        if not self.client_models:
            return
        
        # Calculate total samples for weighting
        total_samples = sum(d['num_samples'] for d in self.clients_data.values())
        
        # Initialize aggregated weights
        avg_W = None
        avg_b = None
        
        # Weighted average of all client models
        for client_id, model in self.client_models.items():
            num_samples = self.clients_data[client_id]['num_samples']
            weight = num_samples / total_samples
            
            if avg_W is None:
                avg_W = model.W * weight
                avg_b = model.b * weight
            else:
                avg_W += model.W * weight
                avg_b += model.b * weight
        
        # Update all clients with aggregated model
        for model in self.client_models.values():
            model.W = avg_W.copy()
            model.b = avg_b.copy()
        
        print("  ✓ Aggregated models using FedAvg")
    
    def run_training(self) -> List[Dict]:
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print("🚀 STARTING FEDERATED TRAINING")
        print("="*70)
        
        training_start = time.time()
        
        for round_num in range(1, NUM_ROUNDS + 1):
            round_metrics = self.train_round(round_num)
            self.training_history.append(round_metrics)
            time.sleep(0.5)
        
        total_time = time.time() - training_start
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\n⏱️  Total training time: {total_time:.2f}s")
        
        return self.training_history
    
    def save_results(self, output_file: str = "federated_learning_results.json"):
        """Save results to file"""
        print(f"\n💾 Saving results to {output_file}...")
        
        results = {
            'system_config': {
                'num_clients': len(self.clients_data),
                'num_rounds': NUM_ROUNDS,
                'local_epochs': LOCAL_EPOCHS,
                'learning_rate': LEARNING_RATE,
                'classifier': 'LightweightClassifier (Logistic Regression)',
                'aggregation_method': 'FedAvg (Weighted Average)',
                'total_samples': sum(d['num_samples'] for d in self.clients_data.values()),
                'embedding_dim': self.clients_data[list(self.clients_data.keys())[0]]['X'].shape[1]
            },
            'training_history': self.training_history
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved!")
    
    def print_summary(self):
        """Print final summary"""
        if not self.training_history:
            return
        
        print("\n" + "="*70)
        print("📈 FINAL SUMMARY")
        print("="*70)
        
        first_round = self.training_history[0]['aggregated_metrics']
        last_round = self.training_history[-1]['aggregated_metrics']
        
        loss_improvement = first_round['avg_loss'] - last_round['avg_loss']
        acc_improvement = last_round['avg_accuracy'] - first_round['avg_accuracy']
        
        print(f"\nMetric Improvements ({NUM_ROUNDS} rounds):")
        print(f"  Loss:     {first_round['avg_loss']:.6f} → {last_round['avg_loss']:.6f} "
              f"(↓ {loss_improvement:.6f})")
        print(f"  Accuracy: {first_round['avg_accuracy']:.4f} → {last_round['avg_accuracy']:.4f} "
              f"(↑ {acc_improvement:.4f})")
        
        print(f"\nClients Trained: {first_round['num_clients']}")
        print(f"Total Training Rounds: {NUM_ROUNDS}")
        print(f"Results saved to: federated_learning_results.json\n")
    
    def run(self):
        """Execute complete pipeline"""
        try:
            # Load data
            if not self.load_real_data():
                print("❌ Failed to load data")
                return
            
            # Initialize classifiers
            if not self.initialize_classifiers():
                print("❌ Failed to initialize classifiers")
                return
            
            # Train
            self.run_training()
            
            # Save results
            self.save_results()
            
            # Print summary
            self.print_summary()
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    runner = SimpleFederatedRunner()
    runner.run()
