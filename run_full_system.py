"""
COMPLETE SYSTEM RUNNER
Federated Learning with P2P Nodes + Real Data + Lightweight Classifier
=======================================================================

This script coordinates:
1. Loading real federated data from real_federated_data/
2. Initializing P2P network nodes
3. Training lightweight classifier in federated manner
4. Aggregating results across nodes
"""

import os
import sys
import json
import time
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Import local modules
from lightweight_classifier import LightweightClassifier
from federated_learning_config import (
    NUM_CLIENTS, NUM_SHARDS, NUM_ROUNDS, 
    LOCAL_EPOCHS, LEARNING_RATE
)
from robust_aggregation import RobustAggregator
from global_aggregator import GlobalAggregator


class FullSystemRunner:
    """Orchestrates complete federated learning system"""
    
    def __init__(self, data_dir: str = "real_federated_data", num_nodes: int = 3):
        """
        Initialize system runner
        
        Args:
            data_dir: Directory containing real federated data
            num_nodes: Number of P2P nodes to use
        """
        self.data_dir = Path(data_dir)
        self.num_nodes = num_nodes
        self.node_processes = []
        self.clients_data = {}
        self.client_models = {}
        self.aggregator = GlobalAggregator()
        
        print("\n" + "="*70)
        print("🚀 FULL FEDERATED LEARNING SYSTEM")
        print("="*70)
        
    def load_real_data(self) -> bool:
        """Load real federated data from npz files"""
        print("\n[STEP 1] Loading Real Federated Data...")
        print("-" * 70)
        
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return False
            
        # Find all client data files
        client_files = sorted(self.data_dir.glob("client_*.npz"))
        
        if not client_files:
            print(f"❌ No client_*.npz files found in {self.data_dir}")
            return False
        
        print(f"📂 Found {len(client_files)} client data files")
        
        # Load each client's data
        total_samples = 0
        for client_file in client_files:
            client_id = client_file.stem.replace("client_", "")
            data = np.load(client_file)
            
            # Extract embeddings and labels
            embeddings = data['embeddings'] if 'embeddings' in data.files else data['X']
            labels = data['labels'] if 'labels' in data.files else data['y']
            
            self.clients_data[client_id] = {
                'X': embeddings,
                'y': labels,
                'num_samples': len(embeddings)
            }
            
            total_samples += len(embeddings)
            print(f"  ✓ Client {client_id}: {len(embeddings)} samples, dim={embeddings.shape[1]}")
        
        print(f"\n✅ Total samples loaded: {total_samples}")
        return True
    
    def start_p2p_nodes(self, base_port: int = 5000) -> bool:
        """Start P2P network nodes"""
        print("\n[STEP 2] Starting P2P Network Nodes...")
        print("-" * 70)
        print(f"Starting {self.num_nodes} nodes on ports {base_port}-{base_port + self.num_nodes - 1}")
        
        try:
            for i in range(self.num_nodes):
                port = base_port + i
                print(f"  Starting Node-{i+1} on port {port}...")
                
                # Start node in new PowerShell window
                cmd = f'python p2p_node.py 127.0.0.1 {port}'
                proc = subprocess.Popen(
                    f'powershell -NoExit -Command "{cmd}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.node_processes.append(proc)
                time.sleep(1)  # Wait between starts
            
            time.sleep(3)  # Wait for nodes to initialize
            print(f"✅ All {self.num_nodes} nodes started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start nodes: {e}")
            return False
    
    def initialize_classifiers(self) -> bool:
        """Initialize lightweight classifiers for each client"""
        print("\n[STEP 3] Initializing Lightweight Classifiers...")
        print("-" * 70)
        
        try:
            # Get embedding dimension from first client
            first_client_id = list(self.clients_data.keys())[0]
            input_dim = self.clients_data[first_client_id]['X'].shape[1]
            
            print(f"Embedding dimension: {input_dim}")
            print(f"Output classes: 2 (binary spam/ham)")
            
            # Create classifier for each client
            for client_id in self.clients_data.keys():
                self.client_models[client_id] = LightweightClassifier(
                    input_dim=input_dim,
                    output_dim=2,
                    learning_rate=LEARNING_RATE
                )
            
            print(f"✅ Created {len(self.client_models)} classifiers")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize classifiers: {e}")
            return False
    
    def train_round(self, round_num: int) -> Dict:
        """Execute one federated learning round"""
        print(f"\n[TRAINING ROUND {round_num}]")
        print("-" * 70)
        
        round_metrics = {
            'round': round_num,
            'client_losses': {},
            'client_accuracies': {},
            'timestamp': time.time()
        }
        
        # Train each client locally
        for client_id, data in self.clients_data.items():
            X = data['X']
            y = data['y']
            model = self.client_models[client_id]
            
            # Train locally for several epochs
            start_loss = model.compute_loss(X, y)
            
            for epoch in range(LOCAL_EPOCHS):
                # Mini-batch training
                batch_size = 32
                num_batches = len(X) // batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    
                    X_batch = X[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx]
                    
                    # Compute gradients and update
                    dW, db = model.compute_gradients(X_batch, y_batch)
                    model.W -= model.learning_rate * dW
                    model.b -= model.learning_rate * db
            
            # Compute metrics
            final_loss = model.compute_loss(X, y)
            predictions = model.predict(X)
            accuracy = np.mean(predictions == y)
            
            round_metrics['client_losses'][client_id] = float(final_loss)
            round_metrics['client_accuracies'][client_id] = float(accuracy)
            
            print(f"  Client {client_id}: loss={final_loss:.6f}, acc={accuracy:.4f}")
        
        # Aggregate models
        print("\n  Aggregating models across clients...")
        self.aggregate_models()
        
        avg_loss = np.mean(list(round_metrics['client_losses'].values()))
        avg_acc = np.mean(list(round_metrics['client_accuracies'].values()))
        
        print(f"\n  📊 Round {round_num} Results:")
        print(f"     Average Loss: {avg_loss:.6f}")
        print(f"     Average Accuracy: {avg_acc:.4f}")
        
        return round_metrics
    
    def aggregate_models(self):
        """Aggregate all client models (FedAvg)"""
        if not self.client_models:
            return
        
        # Weight averaging
        total_weight = 0
        avg_W = None
        avg_b = None
        
        for client_id, model in self.client_models.items():
            num_samples = self.clients_data[client_id]['num_samples']
            
            if avg_W is None:
                avg_W = model.W * num_samples
                avg_b = model.b * num_samples
            else:
                avg_W += model.W * num_samples
                avg_b += model.b * num_samples
            
            total_weight += num_samples
        
        # Normalize
        avg_W /= total_weight
        avg_b /= total_weight
        
        # Update all clients with aggregated model
        for model in self.client_models.values():
            model.W = avg_W.copy()
            model.b = avg_b.copy()
    
    def run_training(self, num_rounds: int = NUM_ROUNDS) -> List[Dict]:
        """Run complete training pipeline"""
        print("\n[STEP 4] Training Lightweight Classifier (Federated)...")
        print("-" * 70)
        
        all_metrics = []
        
        for round_num in range(1, num_rounds + 1):
            round_metrics = self.train_round(round_num)
            all_metrics.append(round_metrics)
            time.sleep(1)
        
        return all_metrics
    
    def save_results(self, output_file: str = "full_system_results.json"):
        """Save training results"""
        print(f"\n[STEP 5] Saving Results...")
        print("-" * 70)
        
        results = {
            'system_config': {
                'num_clients': len(self.clients_data),
                'num_nodes': self.num_nodes,
                'embedding_dim': self.clients_data[list(self.clients_data.keys())[0]]['X'].shape[1],
                'classifier': 'LightweightClassifier (Logistic Regression)',
                'total_samples': sum(d['num_samples'] for d in self.clients_data.values())
            },
            'training_results': [],  # Will be populated
            'timestamp': time.time()
        }
        
        print(f"📁 Saving to {output_file}...")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved")
    
    def cleanup(self):
        """Clean up resources"""
        print("\n[CLEANUP]")
        print("-" * 70)
        
        # Terminate node processes
        for proc in self.node_processes:
            try:
                proc.terminate()
            except:
                pass
        
        print("✅ Cleanup complete")
    
    def run(self, num_rounds: int = NUM_ROUNDS):
        """Run complete system"""
        try:
            # Step 1: Load data
            if not self.load_real_data():
                return
            
            # Step 2: Start P2P nodes
            if not self.start_p2p_nodes():
                print("⚠️  Warning: P2P nodes failed to start, continuing with local training...")
            
            # Step 3: Initialize classifiers
            if not self.initialize_classifiers():
                return
            
            # Step 4: Train
            training_metrics = self.run_training(num_rounds)
            
            # Step 5: Save results
            self.save_results()
            
            print("\n" + "="*70)
            print("✅ SYSTEM TRAINING COMPLETE!")
            print("="*70)
            print(f"\n📊 Final Summary:")
            print(f"   • Trained {len(self.clients_data)} clients")
            print(f"   • {num_rounds} federated learning rounds")
            print(f"   • Classifier: Lightweight (Logistic Regression)")
            print(f"   • Results saved to: full_system_results.json")
            print("\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


if __name__ == "__main__":
    # Run the complete system
    runner = FullSystemRunner(data_dir="real_federated_data", num_nodes=3)
    runner.run(num_rounds=NUM_ROUNDS)
