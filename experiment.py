"""
Mobile Price Classification — experiment script.
This is the file the AI agent modifies to find the best model.

Usage: uv run experiment.py
"""

import os
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb

from prepare import load_data, evaluate, TIME_BUDGET

# ---------------------------------------------------------------------------
# AGENT MODIFIES THIS SECTION
# ---------------------------------------------------------------------------

MODEL_NAME = "LGBMClassifier"


def build_model():
    scaler = StandardScaler()
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        device='gpu',
        n_jobs=-1,
        verbose=-1,
    )
    return scaler, clf


def train(scaler, model, X_train, y_train, X_val, y_val):
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_train_s, y_train), (X_val_s, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    return scaler, model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    X_train, X_val, y_train, y_val = load_data()

    scaler, model = build_model()
    scaler, model = train(scaler, model, X_train, y_train, X_val, y_val)

    class ModelWrapper:
        def __init__(self, s, m): self.s, self.m = s, m
        def predict(self, X): return self.m.predict(self.s.transform(X))

    wrapped = ModelWrapper(scaler, model)
    t1 = time.time()

    results = evaluate(wrapped, X_val, y_val)
    total_time = t1 - t0

    print("---")
    print(f"accuracy:        {results['accuracy']:.6f}")
    print(f"f1_macro:        {results['f1_macro']:.6f}")
    print(f"training_seconds:{total_time:.1f}")
    print(f"model:           {MODEL_NAME}")

    # ---- Evaluation artifacts ----
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        commit_hash = "local_run"

    out_dir = os.path.join("evaluation_output", commit_hash)
    os.makedirs(out_dir, exist_ok=True)

    preds = wrapped.predict(X_val)

    # 1. Confusion matrix
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. Training history
    try:
        er = model.evals_result_
        sets = list(er.keys())
        metrics = list(er[sets[0]].keys())
        epochs = len(er[sets[0]][metrics[0]])
        x_axis = range(epochs)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(x_axis, er[sets[0]][metrics[0]], label='Train')
        ax[0].plot(x_axis, er[sets[1]][metrics[0]], label='Val')
        ax[0].legend(); ax[0].set_ylabel(metrics[0]); ax[0].set_title(f'{metrics[0]}')
        if len(metrics) > 1:
            ax[1].plot(x_axis, er[sets[0]][metrics[1]], label='Train')
            ax[1].plot(x_axis, er[sets[1]][metrics[1]], label='Val')
            ax[1].legend(); ax[1].set_ylabel(metrics[1]); ax[1].set_title(f'{metrics[1]}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'history.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: history plot failed: {e}")

    # 3. Classification report
    rep = classification_report(y_val, preds, output_dict=True)
    df_rep = pd.DataFrame(rep).transpose()
    df_rep.to_csv(os.path.join(out_dir, 'classification_report.csv'))
    df_rep.to_excel(os.path.join(out_dir, 'classification_report.xlsx'))
