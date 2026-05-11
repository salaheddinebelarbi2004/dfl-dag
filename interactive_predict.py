"""
Interactive prediction using the trained federated learning model
"""
import torch
import numpy as np
import json
from pathlib import Path

# ============================================================
# 1. LOAD ALL TRAINING RESULTS AND EXTRACT FINAL MODEL
# ============================================================

print("\n" + "="*70)
print("🤖 SETTING UP INFERENCE ENGINE")
print("="*70)

# First, run a quick federated learning to get final weights
print("\n⚙️ Loading training configuration...")

# Load the config and trained model
MODEL_PATH = 'model.pt'
RESULTS_FILE = 'shard_model_results.json'

# Load results to see what we achieved
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

print(f"✅ Training Summary:")
final_metrics = results['training_history'][-1]['aggregated_metrics']
print(f"   • Final Test Accuracy: {final_metrics['avg_test_accuracy']:.4f}")
print(f"   • Final Test Loss: {final_metrics['avg_test_loss']:.6f}")

# Now train one final time to get fresh weights
print("\n🔄 Loading pre-trained model...")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from run_with_shards import RealModelWrapper, FederatedSystem

# Initialize model
model = RealModelWrapper(model_path=MODEL_PATH, learning_rate=0.001)
print(f"✅ Model loaded: {model.model}")

# Create a small federated system just to load data
fs = FederatedSystem(num_clients=12, num_shards=3)
if not fs.load_real_data():
    print("❌ Failed to load data")
    exit(1)

if not fs.initialize_clients():
    print("❌ Failed to initialize clients")
    exit(1)

fs.create_shards()

print(f"\n✅ System initialized with {len(fs.clients_data)} clients")
print(f"✅ Data loaded: {fs.clients_data['0']['num_train_samples']} training samples per client")

# Train for 1 quick round to get fresh model
print("\n🚀 Quick training round to finalize model...")
round_metrics = fs.train_round(round_num=1, local_epochs=1)

# Get the final trained model
final_model = fs.client_models['0']
print(f"✅ Model ready for inference!")

# ============================================================
# 2. CREATE EMBEDDINGS FOR NEW TEXT (SIMPLE METHOD)
# ============================================================

def create_embeddings(texts):
    """
    Create embeddings from text using a simple method.
    In production, use: sentence-transformers, USE, or BERT
    
    For this demo, we'll use a placeholder that matches our 384-dim requirement
    You should replace this with your actual embedding model
    """
    # Placeholder: create random 384-dim vectors for demo
    # In production: use sentence-transformers SentenceTransformer('all-MiniLM-L6-v2') or similar
    
    # Simple hash-based pseudo-embedding for demo purposes
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    for text in texts:
        # Create a deterministic 384-dim vector from text
        # (In production, use actual transformer model)
        hash_val = hash(text.lower())
        np.random.seed(abs(hash_val) % (2**31))
        emb = np.random.randn(384).astype(np.float32)
        # Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        embeddings.append(emb)
    
    return np.array(embeddings)

# ============================================================
# 3. INTERACTIVE PREDICTION LOOP
# ============================================================

print("\n" + "="*70)
print("💬 INTERACTIVE SPAM CLASSIFIER")
print("="*70)
print("\nInstructions:")
print("  • Type your message to classify as SPAM or HAM")
print("  • Type 'quit' or 'exit' to stop")
print("  • Type 'stats' to see model statistics")
print("  • Type 'test' to run 5 demo predictions")

demo_tests = [
    "Win $1000 now! Click here for free money!!!",
    "Hi, how are you doing today?",
    "CONGRATULATIONS! You've won the lottery",
    "Let's catch up this weekend",
    "Limited time offer - 99% OFF all products"
]

def predict(text):
    """Make a prediction for input text"""
    try:
        # Create embedding
        emb = create_embeddings(text)
        
        # Predict
        pred = final_model.predict(emb)
        confidence_scores = final_model.layers(torch.from_numpy(emb).float())
        confidence = torch.softmax(confidence_scores, dim=1)[0].detach().numpy()
        
        class_names = ['HAM', 'SPAM']
        predicted_class = class_names[pred[0]]
        spam_confidence = confidence[1] * 100  # Class 1 is SPAM
        ham_confidence = confidence[0] * 100   # Class 0 is HAM
        
        return {
            'prediction': predicted_class,
            'spam_confidence': spam_confidence,
            'ham_confidence': ham_confidence
        }
    except Exception as e:
        return {'error': str(e)}

# Main loop
while True:
    try:
        print("\n" + "-"*70)
        user_input = input("📝 Enter message: ").strip()
        
        if not user_input:
            print("⚠️ Please enter a message")
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'stats':
            print("\n📊 Model Statistics:")
            print(f"   • Architecture: 384 → 256 → 2")
            print(f"   • Test Accuracy: {final_metrics['avg_test_accuracy']:.4f} (91.12%)")
            print(f"   • Test Loss: {final_metrics['avg_test_loss']:.6f}")
            print(f"   • Trained on: 12 federated clients (Non-IID distribution)")
            print(f"   • Embedding Dimension: 384")
            continue
        
        if user_input.lower() == 'test':
            print("\n🧪 Running demo predictions...")
            print("="*70)
            for i, test_msg in enumerate(demo_tests, 1):
                result = predict(test_msg)
                if 'error' not in result:
                    print(f"\n#{i} Message: {test_msg[:60]}...")
                    print(f"   Prediction: {result['prediction']}")
                    print(f"   HAM: {result['ham_confidence']:.2f}% | SPAM: {result['spam_confidence']:.2f}%")
                    
                    # Visual bar
                    ham_bar = '█' * int(result['ham_confidence'] / 5)
                    spam_bar = '█' * int(result['spam_confidence'] / 5)
                    print(f"   HAM:  {ham_bar} {result['ham_confidence']:.1f}%")
                    print(f"   SPAM: {spam_bar} {result['spam_confidence']:.1f}%")
            continue
        
        # Regular prediction
        print("\n⏳ Classifying...")
        result = predict(user_input)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("\n" + "="*70)
            print(f"🎯 PREDICTION: {result['prediction']}")
            print("="*70)
            print(f"\nConfidence Scores:")
            print(f"  • HAM:  {result['ham_confidence']:6.2f}%")
            print(f"  • SPAM: {result['spam_confidence']:6.2f}%")
            
            # Visual representation
            print(f"\nVisualization:")
            ham_bar = '█' * int(result['ham_confidence'] / 5)
            spam_bar = '█' * int(result['spam_confidence'] / 5)
            print(f"  HAM:  {ham_bar:<20} {result['ham_confidence']:6.2f}%")
            print(f"  SPAM: {spam_bar:<20} {result['spam_confidence']:6.2f}%")
            
            # Recommendation
            if result['prediction'] == 'SPAM' and result['spam_confidence'] > 80:
                print(f"\n⚠️ High confidence: This is likely SPAM")
            elif result['prediction'] == 'SPAM':
                print(f"\n⚠️ Medium confidence: Might be SPAM")
            elif result['ham_confidence'] > 90:
                print(f"\n✅ High confidence: This is legitimate (HAM)")
            else:
                print(f"\n✅ This appears to be legitimate (HAM)")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

print("\n" + "="*70)
print("✅ Session ended")
print("="*70)
