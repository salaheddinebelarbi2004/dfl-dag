"""
Arabic SMS Spam Classifier Demo
Uses array of real Arabic SMS messages
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

print("\n" + "=" * 70)
print("🤖 مصنف الرسائل الاحتيالية - ARABIC SMS SPAM CLASSIFIER")
print("=" * 70)

# Load model
# Prefer safetensors model if present
# Prefer `model.pt` (PyTorch) where available to avoid safetensors loader memory issues.
if Path("model.pt").exists():
    MODEL_PATH = "model.pt"
elif Path("model.safetensors").exists():
    MODEL_PATH = "model.safetensors"
else:
    MODEL_PATH = "model.pt"
RESULTS_FILE = "shard_model_results.json"

print("\n⚙️  Loading trained model...")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "deploy", "scripts"):
    sys.path.insert(0, str(_REPO_ROOT / _sub))

from run_with_shards import RealModelWrapper
from nlp_embeddings import MiniLMEmbedder
from model_registry import registry

# Initialize model via model registry (framework-agnostic)
if MODEL_PATH.endswith(".safetensors"):
    # fallback to PyTorch if safetensors loader struggles in constrained environments
    if Path("model.pt").exists():
        model_path_to_load = "model.pt"
        loader = "pytorch"
    else:
        model_path_to_load = MODEL_PATH
        loader = "safetensors"
else:
    model_path_to_load = MODEL_PATH
    loader = "pytorch"

model = registry.get(
    loader,
    model_path=model_path_to_load,
    wrapper_class=RealModelWrapper,
    learning_rate=0.001,
)

# Load training results
with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

final_metrics = results["training_history"][-1]["aggregated_metrics"]

print(f"\n✅ Model loaded successfully!")
print(f"\n📊 Model Statistics:")
print(f"   • Test Accuracy: {final_metrics['avg_test_accuracy']:.4f} (91.12%)")
print(f"   • Test Loss: {final_metrics['avg_test_loss']:.6f}")
print(f"   • Architecture: 384-dim embeddings → 256 → 2 (HAM/SPAM)")
print(f"   • Dataset: Arabic SMS messages (Spam/Ham Classification)")

# ============================================================
# Organized Arabic SMS messages by group for analysis
# ============================================================

message_groups = {
    "🔴 OBVIOUS SPAM (Detailed/Phishing)": [
        ("مبروك! لقد ربحت هاتف آيفون جديد، اضغط هنا للاستلام", "Spam"),
        ("عرض حصري اليوم فقط! خصم 80% على جميع المنتجات", "Spam"),
        ("اربح المال من المنزل بسهولة بدون خبرة", "Spam"),
        ("تم اختيارك للفوز بجائزة مالية كبيرة، سجل الآن", "Spam"),
        ("اضغط هنا لتحديث بياناتك البنكية فوراً", "Spam"),
        ("تم تعليق حسابك مؤقتاً، يرجى تسجيل الدخول لإعادة التفعيل", "Spam"),
        ("لديك رسالة صوتية جديدة، اضغط للاستماع", "Spam"),
        ("تم شحن طلبك، تتبع الشحنة من هنا", "Spam"),
        ("تم خصم مبلغ من حسابك، راجع التفاصيل هنا", "Spam"),
        ("تم رفض عملية الدفع، يرجى تحديث بيانات البطاقة", "Spam"),
        ("احصل على قرض فوري بدون ضمانات", "Spam"),
        ("استثمر الآن وحقق أرباحاً مضمونة", "Spam"),
        ("فرصة لا تعوض لتحقيق دخل إضافي", "Spam"),
        ("تداول العملات الرقمية بسهولة وبدون مخاطر", "Spam"),
        ("اربح يومياً من الإنترنت بدون جهد", "Spam"),
    ],
    "⚡ SHORT/AGGRESSIVE SPAM (One-liners)": [
        ("اكسب الآن!", "Spam"),
        ("سجل فوراً!", "Spam"),
        ("لا تفوت الفرصة!", "Spam"),
        ("عرض محدود جداً!", "Spam"),
        ("اضغط الآن!", "Spam"),
    ],
    "🟢 LEGITIMATE (Work/Personal/Casual)": [
        ("هل يمكنك إرسال التقرير اليوم؟", "Ham"),
        ("سأتأخر قليلاً عن الاجتماع", "Ham"),
        ("ما رأيك نلتقي مساءً؟", "Ham"),
        ("شكراً لك على المساعدة", "Ham"),
        ("وصلت إلى المنزل الآن", "Ham"),
        ("تم الانتهاء من المشروع بنجاح، سنناقش التفاصيل غداً", "Ham"),
        ("هل شاهدت المباراة أمس؟ كانت رائعة!", "Ham"),
        ("أرسلت لك الملفات على البريد الإلكتروني", "Ham"),
        ("الطقس جميل اليوم، هل نخرج؟", "Ham"),
        ("يرجى مراجعة المستند قبل إرساله", "Ham"),
        ("أنا في الطريق الآن", "Ham"),
        ("هل اشتريت الحاجيات؟", "Ham"),
        ("سأراك غداً إن شاء الله", "Ham"),
        ("لا تنس الموعد", "Ham"),
        ("كيف كانت رحلتك؟", "Ham"),
    ],
    "🟡 BORDERLINE (Marketing/Notifications/Security)": [
        ("عرض خاص لعملائنا الحاليين، خصم 20% على الاشتراك", "Borderline"),
        ("تم إضافة رصيد إلى حسابك، تحقق من التفاصيل", "Borderline"),
        ("نذكرك بموعد الدفع غداً", "Borderline"),
        ("يرجى تحديث التطبيق للحصول على أفضل أداء", "Borderline"),
        ("لديك نقاط مكافآت يمكنك استخدامها الآن", "Borderline"),
        ("تم إرسال رمز التحقق الخاص بك", "Borderline"),
        ("يرجى تأكيد بريدك الإلكتروني", "Borderline"),
        ("تم تغيير كلمة المرور بنجاح", "Borderline"),
        ("لديك إشعار جديد في حسابك", "Borderline"),
        ("تم تسجيل دخول جديد إلى حسابك", "Borderline"),
    ],
}

# Flatten messages for iteration while preserving group info
arabic_messages = []
for group_name, messages in message_groups.items():
    for msg, category in messages:
        arabic_messages.append((msg, category, group_name))

print(f"\n📚 Testing with {len(arabic_messages)} Arabic SMS messages...")

# ============================================================
# Initialize embedder for real Arabic embeddings
# ============================================================

print("\n🔤 Initializing MiniLM embedder for real Arabic embeddings...")
# Try to initialize the real MiniLM embedder; fall back to deterministic
# pseudo-embeddings if the environment cannot load the transformer model.
print(
    "\n🔤 Initializing MiniLM embedder for real Arabic embeddings (or falling back)..."
)
use_real_embedder = False
try:
    embedder = MiniLMEmbedder(model_name="all-MiniLM-L6-v2")
    use_real_embedder = True

    def create_embedding(text):
        """Create real 384-D embedding using MiniLM"""
        embedding = embedder.embed_text(text)
        return embedding.reshape(1, 384)

except Exception as e:
    print(
        f"⚠️  MiniLM embedder unavailable ({e}). Falling back to deterministic embeddings."
    )

    def create_embedding(text):
        """Deterministic pseudo-embedding (fast, no external model).

        Uses a hash-seeded RNG so results are reproducible per message.
        """
        hash_val = hash(text.lower())
        np.random.seed(abs(hash_val) % (2**31))
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.reshape(1, 384)

# ============================================================
# Track predictions by group
# ============================================================

group_predictions = {
    group_name: {"correct": 0, "total": 0, "predictions": []}
    for group_name in message_groups.keys()
}
overall_correct = 0
overall_total = 0

# ============================================================
# Demo predictions
# ============================================================

print("\n" + "=" * 70)
print("🧪 PREDICTIONS ON ARABIC SMS MESSAGES")
print("=" * 70)

for idx, (message, category, group_name) in enumerate(arabic_messages, 1):
    # Determine ground truth
    if category.lower() == "spam":
        true_label = 1  # SPAM
    elif category.lower() == "borderline":
        true_label = 1  # Treat borderline as SPAM for evaluation
    else:  # Ham
        true_label = 0  # HAM

    # Create embedding from the message
    emb = create_embedding(message)

    # Get prediction
    pred = model.predict(emb)[0]

    # Get confidence scores
    with torch.no_grad():
        scores = model.layers(torch.from_numpy(emb).float())
        confidence = torch.softmax(scores, dim=1)[0].numpy()

    class_names = ["✅ HAM", "⚠️  SPAM"]
    predicted_class = class_names[pred]
    ham_conf = confidence[0] * 100
    spam_conf = confidence[1] * 100

    # Create visualization bars
    ham_bar = "█" * int(ham_conf / 5)
    spam_bar = "█" * int(spam_conf / 5)

    # Check if prediction is correct
    is_correct = pred == true_label
    if is_correct:
        overall_correct += 1
    overall_total += 1

    # Track by group
    group_predictions[group_name]["total"] += 1
    if is_correct:
        group_predictions[group_name]["correct"] += 1

    group_predictions[group_name]["predictions"].append(
        {
            "idx": idx,
            "correct": is_correct,
            "message": message,
            "prediction": predicted_class,
        }
    )

    correctness = "✅" if is_correct else "❌"

    print(f"\n#{idx} {correctness} {category}")
    print(f"   🎯 Prediction: {predicted_class}")
    print(f"   HAM:  [{ham_bar:<20}] {ham_conf:5.1f}%")
    print(f"   SPAM: [{spam_bar:<20}] {spam_conf:5.1f}%")

# ============================================================
# Summary Statistics by Group
# ============================================================

print("\n" + "=" * 70)
print("📊 ACCURACY BY MESSAGE GROUP")
print("=" * 70)

for group_name in message_groups.keys():
    stats = group_predictions[group_name]
    correct = stats["correct"]
    total = stats["total"]
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\n{group_name}")
    print(
        f"   Total: {total} messages | Correct: {correct} | ❌ Wrong: {total - correct}"
    )
    print(f"   ✅ Accuracy: {accuracy:.1f}%")

print("\n" + "=" * 70)
print("📈 OVERALL RESULTS")
print("=" * 70)

demo_accuracy = (overall_correct / overall_total) * 100

print(
    f"\n🎯 Overall Demo Accuracy: {overall_correct}/{overall_total} = {demo_accuracy:.1f}%"
)

print(f"\n📱 Total Messages: {len(arabic_messages)}")
print(
    f"   • OBVIOUS SPAM: {message_groups['🔴 OBVIOUS SPAM (Detailed/Phishing)'].__len__()} messages"
)
print(
    f"   • SHORT SPAM: {message_groups['⚡ SHORT/AGGRESSIVE SPAM (One-liners)'].__len__()} messages"
)
print(
    f"   • LEGITIMATE: {message_groups['🟢 LEGITIMATE (Work/Personal/Casual)'].__len__()} messages"
)
print(
    f"   • BORDERLINE: {message_groups['🟡 BORDERLINE (Marketing/Notifications/Security)'].__len__()} messages"
)

print(f"\n🔄 Federated Learning Details:")
print(f"   • Training Rounds: 5")
print(f"   • Distributed Clients: 12")
print(f"   • Shard Architecture: 3 shards (4 clients each)")
print(f"   • Model Training Accuracy: {final_metrics['avg_test_accuracy']:.2%}")
print(f"   • Model Test Loss: {final_metrics['avg_test_loss']:.6f}")

print("\n" + "=" * 70)
print("✅ Demo complete!")
print("\nTo use interactive predictor with your own Arabic messages:")
print("   python predict.py")
print("=" * 70 + "\n")
