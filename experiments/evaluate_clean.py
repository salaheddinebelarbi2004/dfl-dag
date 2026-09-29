import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, precision_score, recall_score
from evaluate_malicious_model import load_real_federated_dataset, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM
from client import NeuralNetworkSimulator
from lightweight_classifier import LightweightClassifier

OUTPUT_DIR = Path('results/malicious_baseline')
OUTPUT_DIR.mkdir(exist_ok=True)

THRESH = 0.45  # F1-optimized threshold


def load_model():
    model_path = OUTPUT_DIR / 'global_model_malicious.npy'
    weights = np.load(model_path)
    expected_size = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    if len(weights) == expected_size:
        model = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model.set_params(weights)
        return model, 'nn'
    else:
        out_dim = 2
        if (len(weights) - out_dim) % out_dim != 0:
            input_dim = 384
        else:
            input_dim = int((len(weights) - out_dim) / out_dim)
        clf = LightweightClassifier(input_dim=input_dim, output_dim=out_dim)
        clf.set_weights(weights)
        return clf, 'clf'


def plot_confusion_matrix(cm, outpath, title='Confusion Matrix'):
    """Plot a clean, high-quality confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Ham (0)', 'Spam (1)'], fontsize=11)
    ax.set_yticklabels(['Ham (0)', 'Spam (1)'], fontsize=11)
    
    # Values in cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black',
                         fontsize=12, fontweight='bold')
    
    fig.colorbar(im, ax=ax, label='Count')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    model, mtype = load_model()
    X_train, X_test, y_train, y_test = load_real_federated_dataset(Path('real_federated_data'))

    # Get predictions at F1-optimized threshold
    if mtype == 'nn':
        probs = model.forward(X_test)[0][:, 1]
    else:
        probs = model.forward(X_test)[1][:, 1]
    
    preds = (probs >= THRESH).astype(int)

    # Compute metrics
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds)

    # Compute ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    # Save metrics JSON
    metrics = {
        'threshold': float(THRESH),
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'roc_auc': float(roc_auc)
    }
    with open(OUTPUT_DIR / 'metrics.json', 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # Save confusion matrix as JSON
    cm_dict = {
        'confusion_matrix': cm.tolist(),
        'labels': ['Ham', 'Spam']
    }
    with open(OUTPUT_DIR / 'confusion_matrix.json', 'w') as fh:
        json.dump(cm_dict, fh, indent=2)

    # Plot confusion matrix
    plot_confusion_matrix(cm, OUTPUT_DIR / 'confusion_matrix.png', 
                         title=f'Global Confusion Matrix (Threshold={THRESH})')

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve - Global Model', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('\n' + '='*60)
    print('EVALUATION COMPLETE')
    print('='*60)
    print(f'Threshold:  {THRESH}')
    print(f'Accuracy:   {acc:.4f}')
    print(f'Precision:  {prec:.4f}')
    print(f'Recall:     {rec:.4f}')
    print(f'F1 Score:   {f1:.4f}')
    print(f'ROC-AUC:    {roc_auc:.4f}')
    print(f'\nConfusion Matrix:')
    print(f'  [[TN={cm[0,0]:<5} FP={cm[0,1]:<5}]')
    print(f'   [FN={cm[1,0]:<5} TP={cm[1,1]:<5}]]')
    print('='*60)
    print(f'Results saved to: {OUTPUT_DIR}')
    print('  - metrics.json')
    print('  - confusion_matrix.json')
    print('  - confusion_matrix.png')
    print('  - roc_curve.png')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
