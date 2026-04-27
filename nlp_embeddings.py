"""
NLP Embeddings integration with MiniLM
Converts text to embeddings for federated learning
"""

import numpy as np
from typing import List, Tuple

# Try to import sentence-transformers, fallback to random embeddings if not available
try:
    from sentence_transformers import SentenceTransformer
    MINILM_AVAILABLE = True
except ImportError:
    MINILM_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Using random embeddings for demo.")
    print("   Install with: pip install sentence-transformers")


class MiniLMEmbedder:
    """Handles text-to-embedding conversion using MiniLM"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder with pre-trained MiniLM model
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.embedding_dim = 384
        
        if MINILM_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded {model_name} ({self.embedding_dim} dimensions)")
        else:
            self.model = None
            print(f"⚠️  Using random embeddings (install sentence-transformers for real embeddings)")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to 384-dimensional embedding
        
        Args:
            text: Input text string
            
        Returns:
            384-dim numpy array
        """
        if self.model:
            # Use real MiniLM embeddings
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        else:
            # Demo: random embedding
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            (N, 384) numpy array
        """
        if self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        else:
            # Demo: random embeddings
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)


# Sample spam/ham data for testing
SAMPLE_SPAM_TEXTS = [
    "Click here to win FREE money now!!!",
    "Limited time offer - buy now before it's gone",
    "Congratulations! You've won the lottery",
    "Urgent action required - verify your account",
    "Get rich quick - no work needed",
]

SAMPLE_HAM_TEXTS = [
    "Hi, how are you doing today?",
    "The meeting is scheduled for 3pm tomorrow",
    "Thanks for your help with the project",
    "Can you send me the report when you have time",
    "See you at the coffee shop later",
]


def create_nlp_training_data(num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic spam/ham training data
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (embeddings, labels)
        - embeddings: (num_samples, 384) array
        - labels: (num_samples,) binary array (0=ham, 1=spam)
    """
    embedder = MiniLMEmbedder()
    
    # Split samples between spam and ham
    spam_count = num_samples // 2
    ham_count = num_samples - spam_count
    
    # Create embeddings
    embeddings_list = []
    labels_list = []
    
    # Spam samples
    for i in range(spam_count):
        # Randomly sample from spam pool or generate variations
        text = np.random.choice(SAMPLE_SPAM_TEXTS)
        embedding = embedder.embed_text(text)
        # Add noise for variation
        embedding += np.random.randn(384) * 0.01
        embeddings_list.append(embedding)
        labels_list.append(1)  # Spam label
    
    # Ham samples
    for i in range(ham_count):
        # Randomly sample from ham pool or generate variations
        text = np.random.choice(SAMPLE_HAM_TEXTS)
        embedding = embedder.embed_text(text)
        # Add noise for variation
        embedding += np.random.randn(384) * 0.01
        embeddings_list.append(embedding)
        labels_list.append(0)  # Ham label
    
    # Convert to numpy arrays
    X = np.array(embeddings_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


if __name__ == "__main__":
    # Test embeddings
    print("Testing MiniLM embeddings...\n")
    
    embedder = MiniLMEmbedder()
    
    # Test single text
    text = "This is a spam message"
    embedding = embedder.embed_text(text)
    print(f"Text: '{text}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dimensions): {embedding[:10]}\n")
    
    # Test batch
    texts = ["Hello world", "This is spam", "Good morning"]
    embeddings = embedder.embed_texts(texts)
    print(f"Batch embedding shape: {embeddings.shape}\n")
    
    # Create training data
    X, y = create_nlp_training_data(num_samples=50)
    print(f"Created training data:")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Spam samples: {np.sum(y)}")
    print(f"  Ham samples: {np.sum(y == 0)}")
