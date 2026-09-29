import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = r"C:\Users\Lenovo\Desktop\p2p-network-main - Copie\experiments\figures\syso.png"

fig, ax = plt.subplots(figsize=(12,6), dpi=120)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axis('off')

def box(x,y,w,h,text,fc='#f0f0f0'):
    p = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.02", 
                       linewidth=1, edgecolor='#333333', facecolor=fc)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10, wrap=True)

# Title
ax.text(0.5, 0.95, 'System Workflow — Compression contribution vs Baseline', ha='center', fontsize=14, weight='bold')

# Boxes
box(0.03, 0.58, 0.20, 0.29, 'Clients\n(local compute)\n- compute update \n- residual buffer (client-side)\n- clip gradients', fc='#d9eaf7')
box(0.26, 0.58, 0.20, 0.29, 'Compression (Option A)\n- compensate (residual)\n- compress (top-k / quantize)\n- measure serialized bytes', fc='#ffd9b3')
box(0.49, 0.58, 0.20, 0.29, 'Shard Chief\n- decompress payload\n- verify (ZKP over original)\n- robust aggregation', fc='#d9f7df')
box(0.72, 0.58, 0.25, 0.29, 'Aggregator / Metrics\n- collect communication_stats\n- produce system metrics\n- persist JSON artifacts', fc='#f2e6ff')

# Arrows
def arrow(x1,y1,x2,y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))

arrow(0.23, 0.725, 0.26, 0.725)
arrow(0.46, 0.725, 0.49, 0.725)
arrow(0.69, 0.725, 0.72, 0.725)

# Annotations: baseline vs contribution
box(0.05, 0.12, 0.42, 0.30, 'Baseline (before Option A)\n- Clients compute updates and transmit dense gradients\n- Shard chief verifies and aggregates (no compression)\n- Metrics: model accuracy, verification counts', fc='#efefef')
box(0.53, 0.12, 0.42, 0.30, 'Compression contribution (Option A)\n- Adds compression layer between client and shard chief\n- Adds residual/error-feedback at client (not transmitted)\n- Commits to compressed payload; measures actual wire bytes\n- Decompress before verify to preserve verifier semantics', fc='#fff4d6')

# Legend bullets
ax.text(0.05, 0.05, 'Key:', fontsize=10, weight='bold')
ax.text(0.05, 0.02, '\u2022 Orange = New compression component\n\u2022 Blue = Client-side changes\n\u2022 Green = Shard-side changes\n\u2022 Purple = Metrics/aggregation', fontsize=9)

plt.tight_layout()
plt.savefig(out, bbox_inches='tight')
print('Saved', out)
