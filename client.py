"""
Federated Learning Client
Implements local training and update generation
Performs Step 2-6 of training workflow
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import time
import hashlib

from federated_learning_config import (
    LEARNING_RATE, LOCAL_EPOCHS, LOCAL_BATCH_SIZE,
    INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, GRADIENT_CLIP
)
from polynomial_proxy import PolynomialProxy, PolynomialProxyFactory
from zkp_proofs import ZKProof, ZKPGenerator, ZKPFactory


@dataclass
class ClientUpdate:
    """Client model update for submission"""
    client_id: str
    round_num: int
    
    # Update data
    gradient: np.ndarray  # ∆Wi
    model_hash: str  # Hash of local model Wi
    
    # Polynomial proxy
    polynomial_proxy: PolynomialProxy  # Pi(x)
    
    # Cryptographic commitment
    commitment: str  # Ci = Commit(∆Wi)
    
    # Zero-knowledge proof
    zkp: ZKProof  # πi
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    local_accuracy: float = 0.0
    num_local_samples: int = 0


class NeuralNetworkSimulator:
    """
    Simulates a neural network for local training.
    Enables gradient computation without actual DL framework.
    """
    
    def __init__(self, input_dim: int = INPUT_DIM,
                hidden_dim: int = HIDDEN_DIM,
                output_dim: int = OUTPUT_DIM):
        """
        Initialize network parameters.
        
        W1: input_dim × hidden_dim
        b1: hidden_dim
        W2: hidden_dim × output_dim
        b2: output_dim
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize parameters
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
        
        self.params_history = [self.get_params()]
    
    def get_params(self) -> np.ndarray:
        """Get flattened parameters"""
        params = np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
        return params
    
    def set_params(self, params: np.ndarray) -> None:
        """Set parameters from flattened vector"""
        offset = 0
        
        # W1
        w1_size = self.input_dim * self.hidden_dim
        self.W1 = params[offset:offset + w1_size].reshape(self.input_dim, self.hidden_dim)
        offset += w1_size
        
        # b1
        b1_size = self.hidden_dim
        self.b1 = params[offset:offset + b1_size]
        offset += b1_size
        
        # W2
        w2_size = self.hidden_dim * self.output_dim
        self.W2 = params[offset:offset + w2_size].reshape(self.hidden_dim, self.output_dim)
        offset += w2_size
        
        # b2
        self.b2 = params[offset:]
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass.
        Returns logits and cache for backprop.
        """
        batch_size = X.shape[0]
        
        # Hidden layer: Z1 = X @ W1 + b1
        Z1 = X @ self.W1 + self.b1
        # Activation: A1 = ReLU(Z1)
        A1 = np.maximum(0, Z1)
        
        # Output layer: Z2 = A1 @ W2 + b2
        Z2 = A1 @ self.W2 + self.b2
        
        # Softmax for probabilities
        exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
        
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        """
        A2, _ = self.forward(X)
        m = X.shape[0]
        
        # Cross-entropy loss
        loss = -np.mean(np.sum(y * np.log(A2 + 1e-8), axis=1))
        return loss
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradients via backpropagation.
        """
        A2, cache = self.forward(X)
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = A2 - y
        dW2 = (cache["A1"].T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        
        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0)  # ReLU derivative
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m
        
        gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        
        return gradients
    
    def update_params(self, gradients: Dict[str, np.ndarray],
                     learning_rate: float = LEARNING_RATE) -> None:
        """Update parameters via gradient descent"""
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]
        
        self.params_history.append(self.get_params())
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        logits, _ = self.forward(X)
        return np.argmax(logits, axis=1)


class FederatedLearningClient:
    """
    Federated Learning Client.
    Implements Steps 2-6 of training workflow.
    """
    
    def __init__(self, client_id: str, shard_id: str,
                 input_dim: int = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = OUTPUT_DIM):
        """
        Initialize client.
        
        Args:
            client_id: Unique client identifier
            shard_id: Shard this client belongs to
            input_dim, hidden_dim, output_dim: Network dimensions
        """
        self.client_id = client_id
        self.shard_id = shard_id
        
        # Local training
        self.local_data_X: Optional[np.ndarray] = None
        self.local_data_y: Optional[np.ndarray] = None
        self.local_model = NeuralNetworkSimulator(input_dim, hidden_dim, output_dim)
        
        # Statistics
        self.round_num = 0
        self.total_samples_trained = 0
        self.training_history: Dict[int, Dict[str, Any]] = {}
        
    def set_local_dataset(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Set local dataset.
        
        Args:
            X: Feature matrix (n_samples, input_dim)
            y: Label matrix one-hot encoded (n_samples, output_dim)
        """
        self.local_data_X = X
        self.local_data_y = y
    
    def receive_global_model(self, global_params: np.ndarray) -> None:
        """
        Receive global model from aggregator.
        Equivalent to Step 1 (broadcast).
        
        Args:
            global_params: Flattened global model parameters
        """
        self.local_model.set_params(global_params.copy())
    
    def local_training_round(self,
                            local_epochs: int = LOCAL_EPOCHS,
                            batch_size: int = LOCAL_BATCH_SIZE) -> Dict[str, Any]:
        """
        Perform local training.
        Implements Step 2 of workflow: Wi = Wt - η∇L(Di, Wt)
        
        Args:
            local_epochs: Number of local epochs
            batch_size: Training batch size
            
        Returns:
            Training metadata
        """
        if self.local_data_X is None:
            raise ValueError("Local dataset not set")
        
        metadata = {
            "client_id": self.client_id,
            "local_epochs": local_epochs,
            "total_samples": len(self.local_data_X),
            "epoch_losses": [],
        }
        
        n_samples = self.local_data_X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Local training loop
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = self.local_data_X[indices]
            y_shuffled = self.local_data_y[indices]
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Compute loss and gradients
                loss = self.local_model.compute_loss(X_batch, y_batch)
                epoch_loss += loss
                
                gradients = self.local_model.backward(X_batch, y_batch)
                self.local_model.update_params(gradients)
            
            avg_loss = epoch_loss / n_batches
            metadata["epoch_losses"].append(float(avg_loss))
        
        self.total_samples_trained += n_samples * local_epochs
        return metadata
    
    def compute_update(self, initial_params: np.ndarray) -> np.ndarray:
        """
        Compute model update.
        Implements Step 3 of workflow: ∆Wi = Wi - Wt
        
        Args:
            initial_params: Initial global model parameters Wt
            
        Returns:
            Update vector ∆Wi
        """
        current_params = self.local_model.get_params()
        update = current_params - initial_params
        return update
    
    def create_update(self,
                     initial_params: np.ndarray,
                     round_num: int,
                     global_model_params: np.ndarray = None) -> ClientUpdate:
        """
        Create complete client update package.
        Implements Steps 3-6 of workflow:
        3. ∆Wi = Wi - Wt
        4. Pi(x) ≈ f(∆Wi, Wt)
        5. πi = ZKP(...)
        6. Submit (∆Wi, Pi, Ci, πi)
        
        Args:
            initial_params: Initial global model Wt
            round_num: Training round
            global_model_params: Optional model params for ZKP
            
        Returns:
            ClientUpdate object
        """
        self.round_num = round_num
        
        # Step 3: Compute update
        gradient = self.compute_update(initial_params)
        
        # Clip gradient norm
        grad_norm = float(np.linalg.norm(gradient.flatten()))
        if grad_norm > GRADIENT_CLIP:
            gradient = gradient / grad_norm * GRADIENT_CLIP
        
        # Step 4: Build polynomial proxy
        polynomial_proxy = PolynomialProxyFactory.create_from_update(
            gradient, initial_params
        )
        
        # Compute commitment: Ci = Commit(∆Wi)
        commitment = self._compute_commitment(gradient)
        
        # Step 5: Generate ZKP
        zkp_gen = ZKPFactory.get_generator()
        zkp = zkp_gen.generate_proof(
            gradient,
            initial_params if initial_params is not None else self.local_model.get_params(),
            self.client_id,
            round_num,
            polynomial_proxy,
            norm_bound=GRADIENT_CLIP
        )
        
        # Create model hash
        model_params = self.local_model.get_params()
        model_hash = self._compute_hash(model_params)
        
        # Create update object
        update = ClientUpdate(
            client_id=self.client_id,
            round_num=round_num,
            gradient=gradient,
            model_hash=model_hash,
            polynomial_proxy=polynomial_proxy,
            commitment=commitment,
            zkp=zkp,
            local_accuracy=0.0,  # Could compute from validation set
            num_local_samples=len(self.local_data_X) if self.local_data_X is not None else 0,
        )
        
        # Store in history
        self.training_history[round_num] = {
            "update": update,
            "gradient_norm": float(grad_norm),
        }
        
        return update
    
    def _compute_commitment(self, gradient: np.ndarray) -> str:
        """
        Compute cryptographic commitment to gradient.
        Ci = Commit(∆Wi)
        
        Args:
            gradient: The gradient vector
            
        Returns:
            Commitment hash
        """
        grad_bytes = gradient.astype(np.float32).tobytes()
        commitment = hashlib.sha256(grad_bytes).hexdigest()
        return commitment
    
    def _compute_hash(self, params: np.ndarray) -> str:
        """Compute hash of parameters"""
        param_bytes = params.astype(np.float32).tobytes()
        return hashlib.sha256(param_bytes).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "client_id": self.client_id,
            "shard_id": self.shard_id,
            "round_num": self.round_num,
            "total_samples_trained": self.total_samples_trained,
            "rounds_completed": len(self.training_history),
        }


# Helper function for loading real embeddings from federated data
def load_real_client_embeddings(client_id: int, data_dir: str = "real_federated_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real MiniLM embeddings for a client from pre-generated files.
    
    Args:
        client_id: Client identifier (0-11)
        data_dir: Directory containing real_federated_data files
        
    Returns:
        Tuple of (X, y) where X is embeddings, y is one-hot labels
    """
    import os
    
    file_path = os.path.join(data_dir, f"client_{client_id}.npz")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real data not found: {file_path}")
    
    data = np.load(file_path)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    
    return X, y


# Helper function for creating synthetic datasets
def create_synthetic_client_data(client_id: int,
                                num_samples: int = 500,
                                input_dim: int = INPUT_DIM,
                                output_dim: int = OUTPUT_DIM,
                                data_distribution: str = "iid") -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data for a client.
    
    Args:
        client_id: Client identifier
        num_samples: Number of samples
        input_dim: Feature dimension
        output_dim: Number of classes
        data_distribution: "iid" or "non_iid"
        
    Returns:
        Tuple of (X, y) where y is one-hot encoded
    """
    np.random.seed(42 + client_id)  # Deterministic but different per client
    
    # Generate features
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # Generate labels based on distribution
    if data_distribution == "iid":
        # IID: uniform distribution
        y = np.random.randint(0, output_dim, num_samples)
    else:  # non_iid
        # Non-IID: each client focuses on subset of classes
        num_classes_per_client = 2
        primary_classes = np.arange(client_id * num_classes_per_client, 
                                   (client_id + 1) * num_classes_per_client) % output_dim
        y = np.random.choice(primary_classes, num_samples)
    
    # One-hot encode labels
    y_onehot = np.eye(output_dim)[y].astype(np.float32)
    
    return X, y_onehot
