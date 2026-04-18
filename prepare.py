"""
Fixed data preparation for mobile price classification autoresearch.
Loads train.csv and test.csv, splits into train/val.

DO NOT MODIFY — this is the fixed evaluation harness.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
VAL_SIZE = 0.2
TARGET_COL = "price_range"
N_CLASSES = 4
TIME_BUDGET = 300  # 5 minutes per experiment

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load and split data into train/val. Returns X_train, X_val, y_train, y_val."""
    df = pd.read_csv(TRAIN_PATH)
    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_val, y_train, y_val


def load_test():
    """Load test data. Returns test_ids, X_test."""
    df = pd.read_csv(TEST_PATH)
    test_ids = df["id"].values
    X = df.drop(columns=["id"]).values.astype(np.float32)
    return test_ids, X


def get_feature_names():
    df = pd.read_csv(TRAIN_PATH)
    return [c for c in df.columns if c != TARGET_COL]

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(model, X_val, y_val):
    """
    Fixed evaluation function.
    Returns dict with accuracy and macro F1.
    Model must implement .predict(X) returning class labels.
    """
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


# ---------------------------------------------------------------------------
# Main — sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Checking data...")
    X_train, X_val, y_train, y_val = load_data()
    print(f"  Train size : {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"  Val size   : {len(X_val)} samples")
    print(f"  Classes    : {np.unique(y_train).tolist()}")
    print(f"  Class dist : {np.bincount(y_train).tolist()}")
    print()
    print("Features:", get_feature_names())
    print()
    print("Done! Ready to run experiment.py")
