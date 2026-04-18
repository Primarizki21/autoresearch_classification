# mobile-autoresearch

Autonomous ML research agent for Mobile Price Classification.
Inspired by @karpathy's autoresearch — same concept, different domain.

## Dataset

- **Task**: Classify mobile phones into 4 price ranges (0, 1, 2, 3)
- **Features**: 20 numeric features (battery_power, ram, px_height, etc.)
- **Train**: 2000 samples | **Val**: 400 samples (stratified split, fixed)
- **Files**: `train.csv`, `test.csv`

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr18`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Read these files**: `prepare.py`, `experiment.py`, `program.md`
4. **Verify data**: check that `train.csv` and `test.csv` exist in the folder.
5. **Initialize results.tsv**: create with just the header row.
6. **Confirm and go**.

## Experimentation

Run experiments with:
```
uv run experiment.py > run.log 2>&1
```

**What you CAN modify** (only `experiment.py`):
- Model type: RandomForest, XGBoost, LightGBM, SVM, MLP, ensembles, stacking, etc.
- Hyperparameters: n_estimators, max_depth, learning_rate, etc.
- Preprocessing: StandardScaler, MinMaxScaler, feature engineering, PCA, etc.
- Training strategy: cross-validation, early stopping, etc.
- Available libraries: scikit-learn, xgboost, lightgbm, numpy, pandas. You are allowed to access the terminal and install new dependencies via `uv add <package_name>` if your model requires a package that is not in the current virtual environment.

**What you CANNOT modify**:
- `prepare.py` — fixed data split and evaluation
- The `evaluate()` function — this is the ground truth metric
- The data files

**Primary metric: f1_macro** (higher = better). Use accuracy as secondary.

**Simplicity criterion**: A tiny f1 improvement that adds 50 lines of complex code is not worth it. Simpler and equally good = keep.

**First run**: Always run the baseline (`experiment.py` as-is) first.

## Evaluation Outputs

For every experiment, the `experiment.py` script run must output evaluation artifacts to a subfolder like `evaluation_output/<run_id_or_commit>` (so I can see the history of training over different loops).
The evaluation artifacts MUST consist of:
1. Confusion matrix (saved as `.png`)
2. History training-val loss and accuracy plots (saved as `.png`)
3. Classification report (saved as `.csv` and `.xlsx`)

## Output format

```
---
accuracy:        0.876250
f1_macro:        0.875318
training_seconds:2.3
model:           RandomForestClassifier
```

Extract metric:
```
grep "^f1_macro:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	f1_macro	accuracy	status	description
```

- commit: 7-char git hash
- f1_macro: e.g. 0.875318 — use 0.000000 for crashes
- accuracy: e.g. 0.876250
- status: `keep`, `discard`, or `crash`
- description: short text

Example:
```
commit	f1_macro	accuracy	status	description
a1b2c3d	0.875318	0.876250	keep	baseline RandomForest 100 trees
b2c3d4e	0.912000	0.913000	keep	XGBoost n_estimators=200
c3d4e5f	0.860000	0.862000	discard	SVM rbf kernel
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit)
2. Modify `experiment.py` with a new idea
3. If necessary, access the terminal and run `uv add <package_name>` to install missing dependencies
4. `git commit`
5. Run: `uv run experiment.py > run.log 2>&1`
6. Read results: `grep "^f1_macro:\|^accuracy:" run.log`
7. If grep is empty → crashed. Run `tail -n 30 run.log` for stack trace.
8. Log to `results.tsv`
9. If f1_macro improved → keep the commit (advance branch)
10. If f1_macro equal or worse → `git reset --hard HEAD~1`

**NEVER STOP**: loop until the human interrupts you. No asking "should I continue?". You are autonomous.

**Ideas to explore** (in rough order of promise):
- XGBoost / LightGBM with tuned hyperparameters
- Feature engineering (e.g. pixel area = px_height * px_width, screen_ratio, etc.)
- Ensemble / stacking (RF + XGB + LGB)
- Optuna hyperparameter search within the time budget
- Neural network (MLP via sklearn or PyTorch)
- Feature selection (drop low-importance features)
- Calibration, label smoothing

## Hardware
- Use GPU if necessary (e.g., PyTorch, XGBoost `tree_method='hist'`, `device='cuda'`, etc.) so it runs on my RTX 5050 GPU laptop.
- Each experiment should finish in under 5 minutes
- If a run exceeds 10 minutes, kill it and treat as failure
