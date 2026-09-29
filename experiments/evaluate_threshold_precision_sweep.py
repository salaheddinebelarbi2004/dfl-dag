import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_score, accuracy_score, log_loss
from evaluate_malicious_model import load_real_federated_dataset, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM
from client import NeuralNetworkSimulator
from lightweight_classifier import LightweightClassifier

OUTPUT_DIR = Path('results/malicious_baseline')

def load_model_and_probs():
    model_path = OUTPUT_DIR / 'global_model_malicious.npy'
    weights = np.load(model_path)
    expected_size = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    if len(weights) == expected_size:
        model = NeuralNetworkSimulator(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model.set_params(weights)
        use_nn = True
    else:
        out_dim = 2
        if (len(weights) - out_dim) % out_dim != 0:
            input_dim = 384
        else:
            input_dim = int((len(weights) - out_dim) / out_dim)
        clf = LightweightClassifier(input_dim=input_dim, output_dim=out_dim)
        clf.set_weights(weights)
        use_nn = False

    X_train, X_test, y_train, y_test = load_real_federated_dataset(Path('real_federated_data'))
    if use_nn:
        probs = model.forward(X_test)[0][:, 1]
    else:
        probs = clf.forward(X_test)[1][:, 1]
    return probs, y_test


def main():
    probs, y_test = load_model_and_probs()
    best = {'threshold':0.5,'precision':0.0,'accuracy':0.0,'loss':None}
    from sklearn.metrics import precision_score, accuracy_score
    for t in np.linspace(0.1,0.9,81):
        preds = (probs >= t).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        acc = accuracy_score(y_test, preds)
        try:
            loss = log_loss(y_test, np.vstack([1-probs, probs]).T)
        except Exception:
            loss = None
        # choose threshold that maximizes precision, tie-breaker by accuracy
        if prec > best['precision'] or (prec == best['precision'] and acc > best['accuracy']):
            best.update({'threshold':float(t),'precision':float(prec),'accuracy':float(acc),'loss':loss})
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / 'evaluation_threshold_precision_sweep.json','w') as f:
        json.dump(best, f, indent=2)
    print('Saved best threshold results (precision) to', OUTPUT_DIR / 'evaluation_threshold_precision_sweep.json')
    print(best)

if __name__ == '__main__':
    main()
