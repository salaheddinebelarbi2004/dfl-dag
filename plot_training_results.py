"""
Comprehensive visualization of federated learning training results
Creates multiple plots showing model performance, convergence, and per-client metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load training results
RESULTS_FILE = 'shard_model_results.json'

with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

training_history = results['training_history']
system_config = results['system_config']

# Extract data for plotting
num_rounds = len(training_history)
rounds = list(range(1, num_rounds + 1))

# Aggregated metrics per round
avg_test_loss = [r['aggregated_metrics']['avg_test_loss'] for r in training_history]
avg_test_accuracy = [r['aggregated_metrics']['avg_test_accuracy'] for r in training_history]

# Per-client metrics for final round
final_round = training_history[-1]
client_metrics = final_round['client_metrics']
clients = sorted(client_metrics.keys(), key=lambda x: int(x))

final_test_acc_per_client = [client_metrics[c]['test_accuracy'] for c in clients]
final_test_loss_per_client = [client_metrics[c]['test_loss'] for c in clients]
final_train_acc_per_client = [client_metrics[c]['train_accuracy'] for c in clients]

# Per-shard metrics for all rounds
shard_metrics_history = {
    'shard_0': {'losses': []},
    'shard_1': {'losses': []},
    'shard_2': {'losses': []},
}

for round_data in training_history:
    for shard_id, shard_data in round_data['shard_metrics'].items():
        if f'shard_{shard_id}' not in shard_metrics_history:
            shard_metrics_history[f'shard_{shard_id}'] = {'losses': []}
        shard_metrics_history[f'shard_{shard_id}']['losses'].append(shard_data['avg_test_loss'])

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# ============================================================
# PLOT 1: Test Accuracy Over Rounds (Line Plot)
# ============================================================
ax1 = plt.subplot(2, 3, 1)
ax1.plot(rounds, avg_test_accuracy, 'b-o', linewidth=2.5, markersize=8, label='Test Accuracy')
ax1.fill_between(rounds, avg_test_accuracy, alpha=0.3)
ax1.set_xlabel('Training Round', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Global Test Accuracy Over Rounds', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.79, 0.88])
ax1.legend(fontsize=10)

# Add value labels
for x, y in zip(rounds, avg_test_accuracy):
    ax1.text(x, y + 0.002, f'{y:.3f}', ha='center', fontsize=9, fontweight='bold')

# ============================================================
# PLOT 2: Test Loss Over Rounds (Line Plot)
# ============================================================
ax2 = plt.subplot(2, 3, 2)
ax2.plot(rounds, avg_test_loss, 'r-s', linewidth=2.5, markersize=8, label='Test Loss')
ax2.fill_between(rounds, avg_test_loss, alpha=0.3, color='red')
ax2.set_xlabel('Training Round', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Global Test Loss Over Rounds', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Add value labels
for x, y in zip(rounds, avg_test_loss):
    ax2.text(x, y + 0.005, f'{y:.3f}', ha='center', fontsize=9, fontweight='bold')

# ============================================================
# PLOT 3: Per-Client Final Round Accuracy (Bar Chart)
# ============================================================
ax3 = plt.subplot(2, 3, 3)
colors = ['#2ecc71' if acc > 0.86 else '#f39c12' if acc > 0.83 else '#e74c3c' 
          for acc in final_test_acc_per_client]
bars = ax3.bar(clients, final_test_acc_per_client, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axhline(y=np.mean(final_test_acc_per_client), color='blue', linestyle='--', linewidth=2, label='Mean')
ax3.set_xlabel('Client ID', fontsize=11, fontweight='bold')
ax3.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
ax3.set_title('Per-Client Test Accuracy (Round 5)', fontsize=12, fontweight='bold')
ax3.set_ylim([0.76, 0.92])
ax3.grid(True, axis='y', alpha=0.3)
ax3.legend(fontsize=10)

# Add value labels on bars
for bar, val in zip(bars, final_test_acc_per_client):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============================================================
# PLOT 4: Per-Client Final Round Loss (Bar Chart)
# ============================================================
ax4 = plt.subplot(2, 3, 4)
colors_loss = ['#e74c3c' if loss > 0.35 else '#f39c12' if loss > 0.30 else '#2ecc71' 
               for loss in final_test_loss_per_client]
bars2 = ax4.bar(clients, final_test_loss_per_client, color=colors_loss, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.axhline(y=np.mean(final_test_loss_per_client), color='blue', linestyle='--', linewidth=2, label='Mean')
ax4.set_xlabel('Client ID', fontsize=11, fontweight='bold')
ax4.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
ax4.set_title('Per-Client Test Loss (Round 5)', fontsize=12, fontweight='bold')
ax4.grid(True, axis='y', alpha=0.3)
ax4.legend(fontsize=10)

# Add value labels on bars
for bar, val in zip(bars2, final_test_loss_per_client):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============================================================
# PLOT 5: Per-Shard Loss Convergence (Multi-line)
# ============================================================
ax5 = plt.subplot(2, 3, 5)
colors_shards = ['#3498db', '#e74c3c', '#2ecc71']
for shard_id, color in zip(sorted(shard_metrics_history.keys()), colors_shards):
    losses = shard_metrics_history[shard_id]['losses']
    ax5.plot(rounds, losses, 'o-', linewidth=2.5, markersize=7, 
             label=shard_id.replace('_', ' ').title(), color=color)

ax5.set_xlabel('Training Round', fontsize=11, fontweight='bold')
ax5.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
ax5.set_title('Per-Shard Test Loss Over Rounds', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# ============================================================
# PLOT 6: Train vs Test Accuracy (Final Round)
# ============================================================
ax6 = plt.subplot(2, 3, 6)
x_pos = np.arange(len(clients))
width = 0.35

bars_train = ax6.bar(x_pos - width/2, final_train_acc_per_client, width, 
                     label='Train', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
bars_test = ax6.bar(x_pos + width/2, final_test_acc_per_client, width, 
                    label='Test', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)

ax6.set_xlabel('Client ID', fontsize=11, fontweight='bold')
ax6.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax6.set_title('Train vs Test Accuracy (Round 5)', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(clients, fontsize=9)
ax6.set_ylim([0.82, 0.92])
ax6.grid(True, axis='y', alpha=0.3)
ax6.legend(fontsize=10)

plt.tight_layout()
plt.savefig('training_results_comprehensive.png', dpi=300, bbox_inches='tight')
print("✅ Saved: training_results_comprehensive.png")
plt.close()

# ============================================================
# ADDITIONAL PLOT: Convergence Summary (2x1)
# ============================================================
fig2, (ax_conv_acc, ax_conv_loss) = plt.subplots(1, 2, figsize=(14, 5))

# Convergence - Accuracy
ax_conv_acc.plot(rounds, avg_test_accuracy, 'g-o', linewidth=3, markersize=10, label='Test Accuracy')
ax_conv_acc.fill_between(rounds, avg_test_accuracy, alpha=0.2, color='green')
ax_conv_acc.set_xlabel('Training Round', fontsize=12, fontweight='bold')
ax_conv_acc.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax_conv_acc.set_title('Test Accuracy Convergence', fontsize=13, fontweight='bold')
ax_conv_acc.grid(True, alpha=0.4)
initial_acc = avg_test_accuracy[0]
final_acc = avg_test_accuracy[-1]
improvement_pct = ((final_acc - initial_acc) / initial_acc) * 100
ax_conv_acc.text(0.5, 0.95, f'Improvement: +{improvement_pct:.2f}%\n({initial_acc:.4f} → {final_acc:.4f})',
                 transform=ax_conv_acc.transAxes, fontsize=11, fontweight='bold',
                 verticalalignment='top', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Convergence - Loss
ax_conv_loss.plot(rounds, avg_test_loss, 'r-s', linewidth=3, markersize=10, label='Test Loss')
ax_conv_loss.fill_between(rounds, avg_test_loss, alpha=0.2, color='red')
ax_conv_loss.set_xlabel('Training Round', fontsize=12, fontweight='bold')
ax_conv_loss.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax_conv_loss.set_title('Test Loss Convergence', fontsize=13, fontweight='bold')
ax_conv_loss.grid(True, alpha=0.4)
initial_loss = avg_test_loss[0]
final_loss = avg_test_loss[-1]
improvement_loss_pct = ((initial_loss - final_loss) / initial_loss) * 100
ax_conv_loss.text(0.5, 0.95, f'Improvement: -{improvement_loss_pct:.2f}%\n({initial_loss:.4f} → {final_loss:.4f})',
                  transform=ax_conv_loss.transAxes, fontsize=11, fontweight='bold',
                  verticalalignment='top', horizontalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('training_convergence.png', dpi=300, bbox_inches='tight')
print("✅ Saved: training_convergence.png")
plt.close()

# ============================================================
# ADDITIONAL PLOT: Per-Client Performance Matrix (Heatmap)
# ============================================================
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

# Extract all test accuracies across all rounds for all clients
client_round_accuracies = {}
for client_id in clients:
    client_round_accuracies[client_id] = []
    for round_data in training_history:
        if client_id in round_data['client_metrics']:
            acc = round_data['client_metrics'][client_id]['test_accuracy']
            client_round_accuracies[client_id].append(acc)

# Create matrix
accuracy_matrix = np.array([client_round_accuracies[c] for c in clients])

# Heatmap 1: Accuracy
im1 = axes[0].imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0.76, vmax=0.92)
axes[0].set_xlabel('Round', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Client ID', fontsize=11, fontweight='bold')
axes[0].set_title('Per-Client Test Accuracy Across Rounds', fontsize=12, fontweight='bold')
axes[0].set_xticks(range(num_rounds))
axes[0].set_xticklabels(rounds)
axes[0].set_yticks(range(len(clients)))
axes[0].set_yticklabels(clients)

# Add text annotations
for i in range(len(clients)):
    for j in range(num_rounds):
        text = axes[0].text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8, fontweight='bold')

plt.colorbar(im1, ax=axes[0], label='Accuracy')

# Loss matrix across rounds
loss_matrix = np.zeros((len(clients), num_rounds))
for i, client_id in enumerate(clients):
    for j, round_data in enumerate(training_history):
        if client_id in round_data['client_metrics']:
            loss = round_data['client_metrics'][client_id]['test_loss']
            loss_matrix[i, j] = loss

# Heatmap 2: Loss
im2 = axes[1].imshow(loss_matrix, cmap='YlOrRd', aspect='auto', vmin=0.27, vmax=0.45)
axes[1].set_xlabel('Round', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Client ID', fontsize=11, fontweight='bold')
axes[1].set_title('Per-Client Test Loss Across Rounds', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(num_rounds))
axes[1].set_xticklabels(rounds)
axes[1].set_yticks(range(len(clients)))
axes[1].set_yticklabels(clients)

# Add text annotations
for i in range(len(clients)):
    for j in range(num_rounds):
        text = axes[1].text(j, i, f'{loss_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8, fontweight='bold')

plt.colorbar(im2, ax=axes[1], label='Loss')

plt.tight_layout()
plt.savefig('per_client_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: per_client_heatmap.png")
plt.close()

# ============================================================
# PRINT SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("📊 TRAINING RESULTS SUMMARY")
print("="*70)

print(f"\n🎯 Final Performance (Round {num_rounds}):")
print(f"   • Test Accuracy:     {avg_test_accuracy[-1]:.4f} ({avg_test_accuracy[-1]*100:.2f}%)")
print(f"   • Test Loss:         {avg_test_loss[-1]:.6f}")

print(f"\n📈 Improvement over {num_rounds} rounds:")
print(f"   • Accuracy:  {avg_test_accuracy[0]:.4f} → {avg_test_accuracy[-1]:.4f} "
      f"(+{(avg_test_accuracy[-1] - avg_test_accuracy[0]):.4f}, "
      f"+{((avg_test_accuracy[-1] - avg_test_accuracy[0]) / avg_test_accuracy[0] * 100):.2f}%)")
print(f"   • Loss:      {avg_test_loss[0]:.6f} → {avg_test_loss[-1]:.6f} "
      f"(-{(avg_test_loss[0] - avg_test_loss[-1]):.6f}, "
      f"-{((avg_test_loss[0] - avg_test_loss[-1]) / avg_test_loss[0] * 100):.2f}%)")

print(f"\n👥 Per-Client Statistics (Round {num_rounds}):")
print(f"   • Accuracy: min={min(final_test_acc_per_client):.4f}, "
      f"max={max(final_test_acc_per_client):.4f}, "
      f"mean={np.mean(final_test_acc_per_client):.4f}, "
      f"std={np.std(final_test_acc_per_client):.4f}")
print(f"   • Loss:     min={min(final_test_loss_per_client):.6f}, "
      f"max={max(final_test_loss_per_client):.6f}, "
      f"mean={np.mean(final_test_loss_per_client):.6f}, "
      f"std={np.std(final_test_loss_per_client):.6f}")

print(f"\n🏗️  System Architecture:")
print(f"   • Clients: {system_config['num_clients']}")
print(f"   • Shards:  {system_config['num_shards']}")
print(f"   • Total samples: {system_config['total_samples']}")
print(f"   • Embedding dim: {system_config['embedding_dim']}")

print(f"\n📁 Generated Plots:")
print(f"   ✓ training_results_comprehensive.png (6-panel overview)")
print(f"   ✓ training_convergence.png (accuracy & loss convergence)")
print(f"   ✓ per_client_heatmap.png (per-client performance matrix)")

print("\n" + "="*70)
