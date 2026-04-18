import os
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

from prepare import load_data, evaluate, TIME_BUDGET

# ---------------------------------------------------------------------------
# AGENT MODIFIES THIS SECTION
# Everything below is fair game: model, hyperparameters, preprocessing, etc.
# ---------------------------------------------------------------------------

def build_model():
    """
    Build and return the model pipeline.
    Must implement .fit(X, y) and .predict(X).
    """
    scaler = StandardScaler()
    # Use XGBoost with cuda device
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric=["mlogloss", "merror"],
        early_stopping_rounds=50,
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    return scaler, clf

# ---------------------------------------------------------------------------
# Training loop — agent can modify this too
# ---------------------------------------------------------------------------

def train(scaler, model, X_train, y_train, X_val, y_val):
    """Train the model with eval_set for history tracking."""
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
        verbose=False
    )
    return scaler, model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Load data
    X_train, X_val, y_train, y_val = load_data()

    # Build + train
    scaler, model = build_model()
    scaler, model = train(scaler, model, X_train, y_train, X_val, y_val)
    
    # We'll create a simple wrapper so evaluate() (which expects .predict()) works.
    class ModelWrapper:
        def __init__(self, scaler, model):
            self.scaler = scaler
            self.model = model
        def predict(self, X):
            return self.model.predict(self.scaler.transform(X))
            
    wrapped_model = ModelWrapper(scaler, model)

    t1 = time.time()

    # Evaluate
    results = evaluate(wrapped_model, X_val, y_val)
    total_time = t1 - t0

    # Print summary — agent reads this
    print("---")
    print(f"accuracy:        {results['accuracy']:.6f}")
    print(f"f1_macro:        {results['f1_macro']:.6f}")
    print(f"training_seconds:{total_time:.1f}")
    print(f"model:           XGBClassifier")

    # ----- EVALUATION ARTIFACTS -----
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except Exception:
        commit_hash = "local_run"
        
    out_dir = os.path.join("evaluation_output", commit_hash)
    os.makedirs(out_dir, exist_ok=True)
    
    X_val_scaled = scaler.transform(X_val)
    preds = model.predict(X_val_scaled)
    
    # 1. Confusion matrix
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. History plots (loss & acc)
    try:
        evals_result = model.evals_result()
        if evals_result:
            epochs = len(evals_result['validation_0']['mlogloss'])
            x_axis = range(0, epochs)
            
            fig, ax = plt.subplots(1, 2, figsize=(12,5))
            
            # Plot Log Loss
            ax[0].plot(x_axis, evals_result['validation_0']['mlogloss'], label='Train')
            ax[0].plot(x_axis, evals_result['validation_1']['mlogloss'], label='Val')
            ax[0].legend()
            ax[0].set_ylabel('Log Loss')
            ax[0].set_title('XGBoost Log Loss')
            
            # Plot Accuracy (1 - merror)
            ax[1].plot(x_axis, [1-e for e in evals_result['validation_0']['merror']], label='Train Acc')
            ax[1].plot(x_axis, [1-e for e in evals_result['validation_1']['merror']], label='Val Acc')
            ax[1].legend()
            ax[1].set_ylabel('Accuracy')
            ax[1].set_title('XGBoost Accuracy')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'history.png'))
            plt.close()
    except Exception as e:
        print(f"Warning: Could not plot history: {e}")
    
    # 3. Classification report
    report_dict = classification_report(y_val, preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(out_dir, 'classification_report.csv'))
    report_df.to_excel(os.path.join(out_dir, 'classification_report.xlsx'))
