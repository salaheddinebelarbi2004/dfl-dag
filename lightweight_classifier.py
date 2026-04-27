"""
Lightweight Classifier for Federated Learning
Simple logistic regression for spam/ham classification
Fast training, efficient gradient computation
"""

import numpy as np
from typing import Tuple, Dict, Any
import time


class LightweightClassifier:
    """
    Simple logistic regression classifier
    Efficient for federated learning
    """

    def __init__(self, input_dim: int = 384, output_dim: int = 2, learning_rate: float = 0.01):
        """
        Initialize lightweight classifier
        
        Args:
            input_dim: Embedding dimension (384 for MiniLM)
            output_dim: Number of classes (2 for binary spam/ham)
            learning_rate: Learning rate for gradient descent
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Initialize weights: (input_dim, output_dim)
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
        self.b = np.zeros(output_dim, dtype=np.float32)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: logistic regression

        Args:
            X: Input embeddings (batch_size, input_dim)

        Returns:
            Tuple of (logits, probabilities)
        """
        # Linear transformation
        logits = X @ self.W + self.b  # (batch_size, output_dim)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return logits, probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Input embeddings

        Returns:
            Predicted labels (0 or 1)
        """
        _, probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Cross-entropy loss

        Args:
            X: Input embeddings (batch_size, input_dim)
            y: True labels (batch_size,)

        Returns:
            Scalar loss
        """
        _, probs = self.forward(X)

        # One-hot encode labels
        batch_size = len(y)
        one_hot = np.zeros((batch_size, self.output_dim))
        one_hot[np.arange(batch_size), y] = 1

        # Cross-entropy loss
        loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-10), axis=1))
        return loss

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients using backpropagation

        Args:
            X: Input embeddings (batch_size, input_dim)
            y: True labels (batch_size,)

        Returns:
            Tuple of (dW, db) gradients
        """
        batch_size = len(y)

        # Forward pass
        logits, probs = self.forward(X)

        # One-hot encode labels
        one_hot = np.zeros((batch_size, self.output_dim))
        one_hot[np.arange(batch_size), y] = 1

        # Gradient of loss w.r.t. logits
        dloss = (probs - one_hot) / batch_size

        # Gradient w.r.t. weights and bias
        dW = X.T @ dloss  # (input_dim, output_dim)
        db = np.sum(dloss, axis=0)  # (output_dim,)

        return dW, db

    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Single training step

        Args:
            X: Input embeddings (batch_size, input_dim)
            y: True labels (batch_size,)

        Returns:
            Dict with loss and accuracy
        """
        # Compute loss before update
        loss = self.compute_loss(X, y)

        # Compute gradients
        dW, db = self.compute_gradients(X, y)

        # Gradient clipping (for stability)
        dW_norm = np.linalg.norm(dW)
        if dW_norm > 1.0:
            dW = dW / dW_norm

        # Update weights
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

        # Compute accuracy after update
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "gradient_norm": float(dW_norm)
        }

    def get_weights(self) -> np.ndarray:
        """Get flattened weights for distribution"""
        # Flatten: [W (input_dim*output_dim), b (output_dim)]
        return np.concatenate([self.W.flatten(), self.b])

    def set_weights(self, weights: np.ndarray):
        """Set weights from flattened array"""
        W_size = self.input_dim * self.output_dim
        self.W = weights[:W_size].reshape(self.input_dim, self.output_dim).astype(np.float32)
        self.b = weights[W_size:].astype(np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "total_parameters": self.input_dim * self.output_dim + self.output_dim,
            "W_norm": float(np.linalg.norm(self.W)),
            "b_norm": float(np.linalg.norm(self.b))
        }


# For compatibility with existing system
class NeuralNetworkSimulator:
    """Wrapper to replace old NeuralNetworkSimulator with lightweight classifier"""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 2, learning_rate: float = 0.01):
        """Initialize (hidden_dim is ignored, uses lightweight model)"""
        self.classifier = LightweightClassifier(input_dim, output_dim, learning_rate)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

    def get_params(self) -> np.ndarray:
        """Get model parameters"""
        return self.classifier.get_weights()

    def set_params(self, params: np.ndarray):
        """Set model parameters"""
        self.classifier.set_weights(params)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass"""
        return self.classifier.forward(X)

    def train_batch(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> Dict[str, Any]:
        """Train on batch"""
        metrics = {"loss_history": [], "accuracy_history": []}

        for epoch in range(epochs):
            stats = self.classifier.train_step(X, y)
            metrics["loss_history"].append(stats["loss"])
            metrics["accuracy_history"].append(stats["accuracy"])

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.classifier.predict(X)


if __name__ == "__main__":
    print("Testing lightweight classifier...\n")

    # Create classifier
    clf = LightweightClassifier(input_dim=384, output_dim=2)
    print(f"✅ Classifier created: {clf.get_stats()}\n")

    # Create synthetic data
    X_train = np.random.randn(100, 384).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)

    X_test = np.random.randn(20, 384).astype(np.float32)
    y_test = np.random.randint(0, 2, 20)

    # Train
    print("Training...")
    for epoch in range(5):
        stats = clf.train_step(X_train, y_train)
        print(f"  Epoch {epoch+1}: loss={stats['loss']:.4f}, accuracy={stats['accuracy']:.2%}")

    # Evaluate
    loss = clf.compute_loss(X_test, y_test)
    accuracy = np.mean(clf.predict(X_test) == y_test)
    print(f"\nTest: loss={loss:.4f}, accuracy={accuracy:.2%}")

    # Get weights for distribution
    weights = clf.get_weights()
    print(f"\nModel parameters: {len(weights)} total")
    print(f"  Weights (W): {clf.input_dim * clf.output_dim}")
    print(f"  Bias (b): {clf.output_dim}")
