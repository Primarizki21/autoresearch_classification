"""
Mobile Price Classification — experiment script.
This is the file the AI agent modifies to find the best model.

Usage: python experiment.py
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ))
    ])
    return model

# ---------------------------------------------------------------------------
# Training loop — agent can modify this too
# ---------------------------------------------------------------------------

def train(model, X_train, y_train):
    """Train the model. Agent can add cross-validation, ensembling, etc."""
    model.fit(X_train, y_train)
    return model

# ---------------------------------------------------------------------------
# Main — do not modify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Load data
    X_train, X_val, y_train, y_val = load_data()

    # Build + train
    model = build_model()
    model = train(model, X_train, y_train)

    t1 = time.time()

    # Evaluate
    results = evaluate(model, X_val, y_val)

    total_time = t1 - t0

    # Print summary — agent reads this
    print("---")
    print(f"accuracy:        {results['accuracy']:.6f}")
    print(f"f1_macro:        {results['f1_macro']:.6f}")
    print(f"training_seconds:{total_time:.1f}")
    print(f"model:           {type(model.named_steps.get('clf', model)).__name__}")
