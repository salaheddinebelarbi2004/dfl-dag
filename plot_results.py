"""
Plot federated learning results from shard_model_results.json
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Load results
with open('shard_model_results.json', 'r') as f:
    results = json.load(f)

training_history = results['training_history']
num_rounds = len(training_history)

# ============================================================
# 1. GLOBAL TEST ACCURACY & LOSS CONVERGENCE
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

rounds = [h['round'] for h in training_history]
test_accs = [h['aggregated_metrics']['avg_test_accuracy'] for h in training_history]
test_losses = [h['aggregated_metrics']['avg_test_loss'] for h in training_history]

ax1.plot(rounds, test_accs, 'o-', linewidth=2, markersize=8, color='#2ecc71')
ax1.fill_between(rounds, test_accs, alpha=0.3, color='#2ecc71')
ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Global Test Accuracy Over Rounds', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.8, 0.95])
for i, (r, acc) in enumerate(zip(rounds, test_accs)):
    ax1.text(r, acc + 0.003, f'{acc:.4f}', ha='center', fontsize=9)

ax2.plot(rounds, test_losses, 'o-', linewidth=2, markersize=8, color='#e74c3c')
ax2.fill_between(rounds, test_losses, alpha=0.3, color='#e74c3c')
ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
ax2.set_title('Global Test Loss Over Rounds', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
for i, (r, loss) in enumerate(zip(rounds, test_losses)):
    ax2.text(r, loss + 0.003, f'{loss:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('01_convergence.png', dpi=100, bbox_inches='tight')
print("✅ Saved: 01_convergence.png")
plt.close()

# ============================================================
# 2. TRAIN VS TEST ACCURACY (OVERFITTING ANALYSIS)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

train_accs = []
test_accs_plot = []

for h in training_history:
    train_acc_round = np.mean([m['train_accuracy'] for m in h['client_metrics'].values()])
    test_acc_round = np.mean([m['test_accuracy'] for m in h['client_metrics'].values()])
    train_accs.append(train_acc_round)
    test_accs_plot.append(test_acc_round)

ax.plot(rounds, train_accs, 'o-', linewidth=2, markersize=8, label='Training Accuracy', color='#3498db')
ax.plot(rounds, test_accs_plot, 's-', linewidth=2, markersize=8, label='Test Accuracy', color='#2ecc71')
ax.fill_between(rounds, train_accs, test_accs_plot, alpha=0.2, color='#95a5a6', label='Generalization Gap')

ax.set_xlabel('Round', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_title('Training vs Test Accuracy', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.75, 1.0])

for i, (r, tr, te) in enumerate(zip(rounds, train_accs, test_accs_plot)):
    gap = tr - te
    ax.text(r, (tr + te) / 2, f'{gap:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('02_train_vs_test.png', dpi=100, bbox_inches='tight')
print("✅ Saved: 02_train_vs_test.png")
plt.close()

# ============================================================
# 3. PER-CLIENT TEST ACCURACY (FINAL ROUND)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

last_round = training_history[-1]
client_ids = sorted(last_round['client_metrics'].keys(), key=lambda x: int(x))
client_test_accs = [last_round['client_metrics'][cid]['test_accuracy'] for cid in client_ids]

colors = ['#2ecc71' if acc >= 0.9 else '#f39c12' if acc >= 0.85 else '#e74c3c' for acc in client_test_accs]
bars = ax.bar(client_ids, client_test_accs, color=colors, edgecolor='black', linewidth=1)

ax.set_xlabel('Client ID', fontsize=11, fontweight='bold')
ax.set_ylabel('Test Accuracy (Round 5)', fontsize=11, fontweight='bold')
ax.set_title('Per-Client Test Accuracy - Final Round', fontsize=12, fontweight='bold')
ax.set_ylim([0.75, 1.05])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, client_test_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{acc:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('03_per_client_accuracy.png', dpi=100, bbox_inches='tight')
print("✅ Saved: 03_per_client_accuracy.png")
plt.close()

# ============================================================
# 4. PER-SHARD PERFORMANCE COMPARISON
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Shard test loss over rounds
for shard_id in range(3):
    shard_losses = []
    for h in training_history:
        loss = h['shard_metrics'][str(shard_id)]['avg_test_loss']
        shard_losses.append(loss)
    ax1.plot(rounds, shard_losses, 'o-', linewidth=2, markersize=7, label=f'Shard {shard_id}')

ax1.set_xlabel('Round', fontsize=11, fontweight='bold')
ax1.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
ax1.set_title('Shard-Level Test Loss', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Final round shard chief comparison
last_round = training_history[-1]
shard_chiefs = [last_round['shard_metrics'][str(i)]['chief'] for i in range(3)]
ax2.bar(range(3), range(3), color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.6, edgecolor='black', linewidth=1)
ax2.set_xticks(range(3))
ax2.set_xticklabels([f'Shard {i}' for i in range(3)], fontsize=10, fontweight='bold')
ax2.set_ylabel('Shard ID', fontsize=11, fontweight='bold')
ax2.set_title('Shard Chiefs', fontsize=12, fontweight='bold')
ax2.set_ylim([-0.5, 2.5])

for i, chief in enumerate(shard_chiefs):
    ax2.text(i, i + 0.1, f'C{chief}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('04_shard_performance.png', dpi=100, bbox_inches='tight')
print("✅ Saved: 04_shard_performance.png")
plt.close()

# ============================================================
# 5. CLIENT ACCURACY DISTRIBUTION (HEATMAP OVER ROUNDS)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

# Build matrix: clients x rounds
client_ids = sorted([int(cid) for cid in training_history[0]['client_metrics'].keys()])
heatmap_data = []

for cid in client_ids:
    client_accs = [training_history[r]['client_metrics'][str(cid)]['test_accuracy'] for r in range(num_rounds)]
    heatmap_data.append(client_accs)

im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.75, vmax=1.0)

ax.set_xticks(range(num_rounds))
ax.set_xticklabels([f'R{r+1}' for r in range(num_rounds)], fontsize=10)
ax.set_yticks(range(len(client_ids)))
ax.set_yticklabels([f'C{cid}' for cid in client_ids], fontsize=9)

ax.set_xlabel('Round', fontsize=11, fontweight='bold')
ax.set_ylabel('Client', fontsize=11, fontweight='bold')
ax.set_title('Per-Client Test Accuracy Heatmap', fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Accuracy', fontsize=10, fontweight='bold')

# Add text annotations
for i, cid in enumerate(client_ids):
    for j in range(num_rounds):
        acc = heatmap_data[i][j]
        text_color = 'white' if acc < 0.9 else 'black'
        ax.text(j, i, f'{acc:.2f}', ha='center', va='center', color=text_color, fontsize=8)

plt.tight_layout()
plt.savefig('05_accuracy_heatmap.png', dpi=100, bbox_inches='tight')
print("✅ Saved: 05_accuracy_heatmap.png")
plt.close()

# ============================================================
# 6. IMPROVEMENT METRICS
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

first_accs = [training_history[0]['client_metrics'][str(cid)]['test_accuracy'] for cid in client_ids]
last_accs = [training_history[-1]['client_metrics'][str(cid)]['test_accuracy'] for cid in client_ids]
improvements = np.array(last_accs) - np.array(first_accs)

colors_imp = ['#2ecc71' if imp >= 0.02 else '#f39c12' if imp >= 0 else '#e74c3c' for imp in improvements]
bars = ax.bar(client_ids, improvements, color=colors_imp, edgecolor='black', linewidth=1)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Client ID', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy Improvement', fontsize=11, fontweight='bold')
ax.set_title('Per-Client Accuracy Improvement (Round 5 - Round 1)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    y_pos = height + 0.002 if height > 0 else height - 0.003
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{imp:+.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig('06_accuracy_improvement.png', dpi=100, bbox_inches='tight')
print("✅ Saved: 06_accuracy_improvement.png")
plt.close()

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("📊 FEDERATED LEARNING RESULTS SUMMARY")
print("="*70)

print(f"\n🎯 Global Performance:")
print(f"   Round 1 → Round 5:")
print(f"   • Test Accuracy: {test_accs[0]:.4f} → {test_accs[-1]:.4f} (+{test_accs[-1]-test_accs[0]:.4f})")
print(f"   • Test Loss:     {test_losses[0]:.6f} → {test_losses[-1]:.6f} ({test_losses[-1]-test_losses[0]:.6f})")

print(f"\n📈 Convergence Stats:")
print(f"   • Accuracy Gain: {(test_accs[-1]-test_accs[0])/(test_accs[0])*100:.2f}%")
print(f"   • Loss Reduction: {abs(test_losses[-1]-test_losses[0])/test_losses[0]*100:.2f}%")
print(f"   • Avg Generalization Gap: {np.mean(np.array(train_accs) - np.array(test_accs_plot)):.4f}")

print(f"\n👥 Per-Client Analysis (Final Round):")
print(f"   • Best Client:  {client_ids[np.argmax(last_accs)]} ({max(last_accs):.4f})")
print(f"   • Worst Client: {client_ids[np.argmin(last_accs)]} ({min(last_accs):.4f})")
print(f"   • Avg Accuracy: {np.mean(last_accs):.4f}")
print(f"   • Std Dev:      {np.std(last_accs):.4f}")

print(f"\n🏛️ Shard Summary:")
for i in range(3):
    chief = last_round['shard_metrics'][str(i)]['chief']
    print(f"   • Shard {i}: Chief=Client {chief}")

print(f"\n✅ All plots saved to current directory!")
print("="*70)
