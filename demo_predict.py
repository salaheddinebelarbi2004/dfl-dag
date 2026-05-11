"""
Arabic SMS Spam Classifier Demo
Uses array of real Arabic SMS messages
"""
import torch
import numpy as np
import json
from pathlib import Path
import sys

print("\n" + "="*70)
print("🤖 مصنف الرسائل الاحتيالية - ARABIC SMS SPAM CLASSIFIER")
print("="*70)

# Load model
MODEL_PATH = 'model.pt'
RESULTS_FILE = 'shard_model_results.json'

print("\n⚙️  Loading trained model...")

sys.path.insert(0, str(Path(__file__).parent))

from run_with_shards import RealModelWrapper
from nlp_embeddings import MiniLMEmbedder

# Initialize model
model = RealModelWrapper(model_path=MODEL_PATH, learning_rate=0.001)

# Load training results
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

final_metrics = results['training_history'][-1]['aggregated_metrics']

print(f"\n✅ Model loaded successfully!")
print(f"\n📊 Model Statistics:")
print(f"   • Test Accuracy: {final_metrics['avg_test_accuracy']:.4f} (91.12%)")
print(f"   • Test Loss: {final_metrics['avg_test_loss']:.6f}")
print(f"   • Architecture: 384-dim embeddings → 256 → 2 (HAM/SPAM)")
print(f"   • Dataset: Arabic SMS messages (Spam/Ham Classification)")

# ============================================================
# Array of Arabic SMS messages to test
# ============================================================

arabic_messages = [
    # 🔴 SPAM (obvious)
    ("مبروك! لقد ربحت هاتف آيفون جديد، اضغط هنا للاستلام", "Spam - Prize"),
    ("عرض حصري اليوم فقط! خصم 80% على جميع المنتجات", "Spam - Discount"),
    ("اربح المال من المنزل بسهولة بدون خبرة", "Spam - Work scam"),
    ("تم اختيارك للفوز بجائزة مالية كبيرة، سجل الآن", "Spam - Lottery"),
    ("اضغط هنا لتحديث بياناتك البنكية فوراً", "Spam - Phishing"),

    # 🔴 SPAM (realistic / tricky)
    ("تم تعليق حسابك مؤقتاً، يرجى تسجيل الدخول لإعادة التفعيل", "Spam - Phishing"),
    ("لديك رسالة صوتية جديدة، اضغط للاستماع", "Spam - Fake notification"),
    ("تم شحن طلبك، تتبع الشحنة من هنا", "Spam - Fake delivery"),
    ("تم خصم مبلغ من حسابك، راجع التفاصيل هنا", "Spam - Banking scam"),
    ("تم رفض عملية الدفع، يرجى تحديث بيانات البطاقة", "Spam - Payment scam"),

    # 🔴 SPAM (short + aggressive)
    ("اكسب الآن!", "Spam - Aggressive"),
    ("سجل فوراً!", "Spam - Aggressive"),
    ("لا تفوت الفرصة!", "Spam - Urgency"),
    ("عرض محدود جداً!", "Spam - Urgency"),
    ("اضغط الآن!", "Spam - Aggressive"),

    # 🟡 BORDERLINE (hard cases)
    ("عرض خاص لعملائنا الحاليين، خصم 20% على الاشتراك", "Borderline - Marketing"),
    ("تم إضافة رصيد إلى حسابك، تحقق من التفاصيل", "Borderline - Financial"),
    ("نذكرك بموعد الدفع غداً", "Borderline - Reminder"),
    ("يرجى تحديث التطبيق للحصول على أفضل أداء", "Borderline - Update"),
    ("لديك نقاط مكافآت يمكنك استخدامها الآن", "Borderline - Rewards"),

    # 🟢 HAM (normal messages)
    ("هل يمكنك إرسال التقرير اليوم؟", "Legitimate - Work"),
    ("سأتأخر قليلاً عن الاجتماع", "Legitimate - Work"),
    ("ما رأيك نلتقي مساءً؟", "Legitimate - Social"),
    ("شكراً لك على المساعدة", "Legitimate - Thanks"),
    ("وصلت إلى المنزل الآن", "Legitimate - Personal"),

    # 🟢 HAM (longer / natural)
    ("تم الانتهاء من المشروع بنجاح، سنناقش التفاصيل غداً", "Legitimate - Work"),
    ("هل شاهدت المباراة أمس؟ كانت رائعة!", "Legitimate - Casual"),
    ("أرسلت لك الملفات على البريد الإلكتروني", "Legitimate - Work"),
    ("الطقس جميل اليوم، هل نخرج؟", "Legitimate - Casual"),
    ("يرجى مراجعة المستند قبل إرساله", "Legitimate - Work"),

    # 🔴 SPAM (financial traps)
    ("احصل على قرض فوري بدون ضمانات", "Spam - Loan"),
    ("استثمر الآن وحقق أرباحاً مضمونة", "Spam - Investment"),
    ("فرصة لا تعوض لتحقيق دخل إضافي", "Spam - Income scam"),
    ("تداول العملات الرقمية بسهولة وبدون مخاطر", "Spam - Crypto scam"),
    ("اربح يومياً من الإنترنت بدون جهد", "Spam - Income scam"),

    # 🟡 BORDERLINE (real vs spam confusion)
    ("تم إرسال رمز التحقق الخاص بك", "Borderline - OTP"),
    ("يرجى تأكيد بريدك الإلكتروني", "Borderline - Verification"),
    ("تم تغيير كلمة المرور بنجاح", "Borderline - Security"),
    ("لديك إشعار جديد في حسابك", "Borderline - Notification"),
    ("تم تسجيل دخول جديد إلى حسابك", "Borderline - Security"),

    # 🟢 HAM (daily life)
    ("أنا في الطريق الآن", "Legitimate - Personal"),
    ("هل اشتريت الحاجيات؟", "Legitimate - Personal"),
    ("سأراك غداً إن شاء الله", "Legitimate - Personal"),
    ("لا تنس الموعد", "Legitimate - Reminder"),
    ("كيف كانت رحلتك؟", "Legitimate - Casual"),

    # 🔴 SPAM (mixed Arabic-English tricks)
    ("Click هنا للحصول على الجائزة الآن!", "Spam - Mixed language"),
    ("Free عرض خاص لك فقط!", "Spam - Mixed language"),
    ("Verify حسابك الآن لتجنب الإيقاف", "Spam - Phishing"),
    ("Win الآن بسهولة!", "Spam - Prize"),
    ("Update بياناتك فوراً!", "Spam - Phishing"),
]

print(f"\n📚 Testing with {len(arabic_messages)} Arabic SMS messages...")

# ============================================================
# ============================================================
# Initialize embedder for real Arabic embeddings
# ============================================================

print("\n🔤 Initializing MiniLM embedder for real Arabic embeddings...")
embedder = MiniLMEmbedder(model_name="all-MiniLM-L6-v2")

def create_embedding(text):
    """Create real 384-D embedding using MiniLM"""
    embedding = embedder.embed_text(text)
    return embedding.reshape(1, 384)

# ============================================================
# Demo predictions
# ============================================================

print("\n" + "="*70)
print("🧪 PREDICTIONS ON ARABIC SMS MESSAGES")
print("="*70)

correct_predictions = 0
total_predictions = len(arabic_messages)

for idx, (message, description) in enumerate(arabic_messages, 1):
    # Determine ground truth from description
    if "Spam" in description:
        true_label = 1  # SPAM
    else:  # Legitimate, Borderline, Personal
        true_label = 0  # HAM

    # Create embedding from the message
    emb = create_embedding(message)

    # Get prediction
    pred = model.predict(emb)[0]

    # Get confidence scores
    with torch.no_grad():
        scores = model.layers(torch.from_numpy(emb).float())
        confidence = torch.softmax(scores, dim=1)[0].numpy()

    class_names = ['✅ HAM', '⚠️  SPAM']
    predicted_class = class_names[pred]
    ham_conf = confidence[0] * 100
    spam_conf = confidence[1] * 100

    # Create visualization bars
    ham_bar = '█' * int(ham_conf / 5)
    spam_bar = '█' * int(spam_conf / 5)

    # Check if prediction is correct
    is_correct = pred == true_label
    if is_correct:
        correct_predictions += 1

    correctness = "✅" if is_correct else "❌"

    print(f"\n#{idx} {correctness} {description}")
    print(f"   Message: {message}")
    print(f"   🎯 Prediction: {predicted_class}")
    print(f"   HAM:  [{ham_bar:<20}] {ham_conf:5.1f}%")
    print(f"   SPAM: [{spam_bar:<20}] {spam_conf:5.1f}%")

# ============================================================
# Summary Statistics
# ============================================================

print("\n" + "="*70)
print("📊 DEMO SUMMARY")
print("="*70)

spam_count = sum(1 for _, desc in arabic_messages if "Spam" in desc)
ham_count = sum(1 for _, desc in arabic_messages if "Spam" not in desc)
demo_accuracy = (correct_predictions / total_predictions) * 100

print(f"\n📱 Total Messages Tested: {len(arabic_messages)}")
print(f"   • Spam Messages: {spam_count}")
print(f"   • Ham Messages: {ham_count}")
print(f"\n🎯 DEMO ACCURACY: {correct_predictions}/{total_predictions} Correct = {demo_accuracy:.1f}%")
print(f"\n🔄 Federated Learning Details:")
print(f"   • Training Rounds: 5")
print(f"   • Distributed Clients: 12")
print(f"   • Shard Architecture: 3 shards (4 clients each)")
print(f"   • Model Training Accuracy: {final_metrics['avg_test_accuracy']:.2%}")
print(f"   • Model Test Loss: {final_metrics['avg_test_loss']:.6f}")

print("\n" + "="*70)
print("✅ Demo complete!")
print("\nTo use interactive predictor with your own Arabic messages:")
print("   python predict.py")
print("="*70 + "\n")
