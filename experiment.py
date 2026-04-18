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
from sklearn.linear_model import LogisticRegression

from prepare import load_data, evaluate, TIME_BUDGET

# ---------------------------------------------------------------------------
# AGENT MODIFIES THIS SECTION
# ---------------------------------------------------------------------------

MODEL_NAME = "LogisticRegression"

def build_model():
    scaler = StandardScaler()

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    return scaler, clf

def train(scaler, model, X_train, y_train, X_val, y_val):
    X_train_s = scaler.fit_transform(X_train)
    model.fit(X_train_s, y_train)
    return scaler, model

class ModelWrapper:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    X_train, X_val, y_train, y_val = load_data()

    scaler, model = build_model()
    scaler, model = train(scaler, model, X_train, y_train, X_val, y_val)

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
    plt.title(f'Confusion Matrix — {MODEL_NAME}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. History plot (Placeholder because linear models do not expose evals_result easily)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, 'Logistic Regression\nNo training history available', 
            ha='center', va='center', fontsize=14)
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, 'history.png'))
    plt.close()

    # 3. Classification report
    rep = classification_report(y_val, preds, output_dict=True)
    df_rep = pd.DataFrame(rep).transpose()
    df_rep.to_csv(os.path.join(out_dir, 'classification_report.csv'))
    df_rep.to_excel(os.path.join(out_dir, 'classification_report.xlsx'))
