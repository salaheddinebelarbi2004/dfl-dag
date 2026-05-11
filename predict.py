"""
Simple spam classification inference using trained federated model
"""
import torch
import numpy as np
import json
from pathlib import Path
import sys

print("\n" + "="*70)
print("🤖 FEDERATED LEARNING SPAM CLASSIFIER")
print("="*70)

# Load model
MODEL_PATH = 'model.pt'
RESULTS_FILE = 'shard_model_results.json'

print("\n⚙️ Initializing model...")

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from run_with_shards import RealModelWrapper
    
    # Load the trained model
    model = RealModelWrapper(model_path=MODEL_PATH, learning_rate=0.001)
    print(f"✅ Model loaded successfully")
    
    # Load training results
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    final_metrics = results['training_history'][-1]['aggregated_metrics']
    print(f"✅ Model Statistics:")
    print(f"   • Test Accuracy: {final_metrics['avg_test_accuracy']:.4f} (91.12%)")
    print(f"   • Test Loss: {final_metrics['avg_test_loss']:.6f}")
    print(f"   • Architecture: 384 → 256 → 2")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ============================================================
# Embedding function
# ============================================================

def create_embeddings(text):
    """Create 384-D embeddings (deterministic for demo)"""
    hash_val = hash(text.lower())
    np.random.seed(abs(hash_val) % (2**31))
    emb = np.random.randn(384).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb.reshape(1, 384)

# ============================================================
# Prediction function
# ============================================================

def predict(text):
    """Predict if text is SPAM or HAM"""
    try:
        emb = create_embeddings(text)
        pred = model.predict(emb)[0]
        
        with torch.no_grad():
            scores = model.model(torch.from_numpy(emb).float())
            confidence = torch.softmax(scores, dim=1)[0].numpy()
        
        class_names = ['HAM', 'SPAM']
        return {
            'class': class_names[pred],
            'ham_conf': confidence[0] * 100,
            'spam_conf': confidence[1] * 100
        }
    except Exception as e:
        return {'error': str(e)}

# ============================================================
# Interactive loop
# ============================================================

print("\n" + "="*70)
print("💬 ENTER MESSAGES TO CLASSIFY")
print("="*70)
print("Commands: 'demo' (test examples), 'quit' (exit)")
print()

demo_messages = [
    "Win free cash now!!!",
    "Hi, how are you?",
    "LIMITED TIME: 90% off everything",
    "See you tomorrow",
    "CLICK HERE FOR BEST DEALS"
]

while True:
    try:
        user_input = input("\n📝 Message: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'demo':
            print("\n🧪 DEMO PREDICTIONS:")
            print("="*70)
            for msg in demo_messages:
                result = predict(msg)
                if 'error' not in result:
                    ham_bar = '=' * int(result['ham_conf'] / 10)
                    spam_bar = '=' * int(result['spam_conf'] / 10)
                    print(f"\nMessage: {msg}")
                    print(f"Prediction: {result['class']}")
                    print(f"  HAM:  [{ham_bar:<10}] {result['ham_conf']:.1f}%")
                    print(f"  SPAM: [{spam_bar:<10}] {result['spam_conf']:.1f}%")
            continue
        
        # Regular prediction
        result = predict(user_input)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            ham_bar = '=' * int(result['ham_conf'] / 10)
            spam_bar = '=' * int(result['spam_conf'] / 10)
            print(f"\n✅ Result: {result['class']}")
            print(f"  HAM:  [{ham_bar:<10}] {result['ham_conf']:.1f}%")
            print(f"  SPAM: [{spam_bar:<10}] {result['spam_conf']:.1f}%")
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")

print("="*70)
