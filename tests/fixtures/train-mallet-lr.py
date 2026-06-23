#!/usr/bin/env python3
"""
Train logistic regression on Mallet noise correlation features.

Reads mallet-features.jsonl, trains LR with L2 regularisation,
evaluates with 5-fold CV, and writes weights to ../../lib/mallet-weights.json
for use in detector.js.

Usage:
  python3 tests/fixtures/train-mallet-lr.py
  python3 tests/fixtures/train-mallet-lr.py --C 0.01   # stronger regularisation
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

FEATURES_FILE = Path(__file__).parent / 'mallet-features.jsonl'
WEIGHTS_FILE  = Path(__file__).parent.parent.parent / 'lib' / 'mallet-weights.json'


def load_features(path):
    X, y, paths = [], [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            X.append(row['features'])
            y.append(row['label'])
            paths.append(row['path'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=0.1,
                        help='LR inverse regularisation strength (smaller = more regularised)')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    print(f'Loading {FEATURES_FILE}...')
    X, y, paths = load_features(FEATURES_FILE)
    print(f'  {X.shape[0]} samples, {X.shape[1]} features')
    print(f'  AI: {y.sum()}  Real: {(y==0).sum()}')

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(C=args.C, max_iter=1000, class_weight='balanced')),
    ])

    print(f'\nCross-validation ({args.folds}-fold, C={args.C})...')
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    scores = cross_validate(pipe, X, y, cv=cv,
                            scoring=['accuracy', 'roc_auc', 'f1'],
                            return_train_score=False)

    print(f'  Accuracy:  {scores["test_accuracy"].mean():.3f} ± {scores["test_accuracy"].std():.3f}')
    print(f'  ROC-AUC:   {scores["test_roc_auc"].mean():.3f} ± {scores["test_roc_auc"].std():.3f}')
    print(f'  F1:        {scores["test_f1"].mean():.3f} ± {scores["test_f1"].std():.3f}')

    # Train final model on all data
    print('\nTraining final model on full dataset...')
    pipe.fit(X, y)

    scaler = pipe.named_steps['scaler']
    lr     = pipe.named_steps['lr']

    # Bake scaler into the weights so detector.js only needs a dot product + sigmoid
    # Effective weight: w_eff[i] = coef[i] / scale[i]
    # Effective bias:   b_eff = intercept - sum(coef[i] * mean[i] / scale[i])
    coef      = lr.coef_[0]
    intercept = lr.intercept_[0]
    mean_     = scaler.mean_
    scale_    = scaler.scale_

    w_eff = coef / scale_
    b_eff = float(intercept - np.sum(coef * mean_ / scale_))

    payload = {
        'T':         int(X.shape[1] // 3 * 2 / (int(X.shape[1] / 3 * 2 / 1) ) ),
        'n_features': int(X.shape[1]),
        'C':          args.C,
        'weights':    w_eff.tolist(),
        'bias':       b_eff,
        'cv_auc':     float(scores['test_roc_auc'].mean()),
        'cv_acc':     float(scores['test_accuracy'].mean()),
        'n_train':    int(X.shape[0]),
        'n_ai':       int(y.sum()),
        'n_real':     int((y==0).sum()),
    }

    # Compute T from n_features: 3 * T*(T-1)/2 = n_features → T*(T-1) = 2*n_features/3
    n = X.shape[1] // 3
    T = int((1 + (1 + 8*n)**0.5) / 2)
    payload['T'] = T

    WEIGHTS_FILE.write_text(json.dumps(payload, indent=2))
    print(f'\nWeights written to {WEIGHTS_FILE}')
    print(f'  T={T}, {X.shape[1]} features, CV AUC={payload["cv_auc"]:.3f}')
    print(f'\nTo use in detector.js:')
    print(f'  import weights from "./mallet-weights.json" assert {{ type: "json" }};')
    print(f'  // or load inline as a JS object literal')


if __name__ == '__main__':
    main()
