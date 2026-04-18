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
5. **Reconstruct experiment memory if needed**: Check if `experiment_memory.md` exists and is non-empty.
   - **If it exists and has content**: read it carefully — it contains the full history of what has been tried. Do NOT repeat experiments already listed there.
   - **If it is empty or missing**: reconstruct it from git history before doing anything else:
     1. Run `git log --oneline` to list all previous commits
     2. For each commit (oldest to newest), run `git show <commit> -- experiment.py` to see what was changed
     3. Check if `evaluation_output/<commit>/experiment_card.txt` exists and read it if available
     4. From that information, reconstruct and append an entry to `experiment_memory.md` following the memory format below
     5. Once all commits are reconstructed, proceed normally
   - **If there are no previous commits at all**: `experiment_memory.md` stays empty, proceed with baseline run.
6. **Initialize results.tsv**: create with just the header row (if it doesn't exist yet).
7. **Confirm and go**.

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

For every experiment, the `experiment.py` script run must output evaluation artifacts to a subfolder `evaluation_output/<commit_hash>/` (7-char git hash as folder name, so history is preserved per commit).

The evaluation artifacts MUST consist of:
1. Confusion matrix (saved as `confusion_matrix.png`)
2. Training/val loss and accuracy plots (saved as `training_history.png`) — if model supports it, otherwise skip gracefully
3. Classification report (saved as `classification_report.csv` and `classification_report.xlsx`)
4. **Experiment card** (saved as `experiment_card.txt`) — see format below

### Experiment Card format (`experiment_card.txt`)

This file is the most important artifact. It must be human-readable and contain:

```
=== EXPERIMENT CARD ===
commit:         a1b2c3d
date:           2025-04-18 22:31
branch:         autoresearch/apr18

--- Model ---
type:           XGBoostClassifier
base_params:    default (no tuning)  |  tuned  |  optuna-searched
key_params:
  n_estimators: 200
  max_depth:    6
  learning_rate:0.1
  subsample:    0.8

--- Preprocessing ---
scaler:         StandardScaler  |  MinMaxScaler  |  None
feature_engineering:
  - added px_area = px_height * px_width
  - dropped: sc_w (low importance)
  - None

--- Training Strategy ---
method:         fit()  |  cross-validation (5-fold)  |  early stopping
tuning:         None  |  Optuna (50 trials)  |  GridSearch

--- Results ---
f1_macro:       0.912000
accuracy:       0.913000
status:         keep  |  discard  |  crash

--- Why tried ---
Short explanation of the hypothesis behind this experiment.
Example: "RAM is the dominant feature — trying tree booster with depth limit
to prevent overfitting on other weak features."

--- What worked / didn't ---
Short note on the outcome. Example: "Good improvement. Depth=6 seems sweet spot,
depth=8 overfit slightly. Feature px_area helped ~0.003 f1."
```

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

## Experiment Memory (`experiment_memory.md`)

After every experiment (keep OR discard), append a new entry to `experiment_memory.md`.
This file is the **collective memory** of all experiments — it must never be deleted or committed to git (keep it untracked).

**Purpose**: Any future agent session that starts fresh must read this file first to avoid re-trying things that were already explored.

### Format — append one block per experiment:

```markdown
### [commit: a1b2c3d] RandomForest baseline — KEEP (f1: 0.8823)
- Model: RandomForestClassifier, n_estimators=100, default params
- Preprocessing: StandardScaler
- Feature engineering: none
- Training: fit()
- Result: f1_macro=0.8823, accuracy=0.8825
- Notes: Good baseline. RAM is clearly the dominant feature (importance ~0.4).

### [commit: b2c3d4e] XGBoost tuned — KEEP (f1: 0.9120)
- Model: XGBoostClassifier, n_estimators=200, max_depth=6, lr=0.1
- Preprocessing: StandardScaler
- Feature engineering: added px_area = px_height * px_width
- Training: fit() with early stopping on val
- Result: f1_macro=0.9120, accuracy=0.9130
- Notes: Big jump. px_area feature helped. Depth > 6 starts overfitting.

### [commit: c3d4e5f] SVM rbf — DISCARD (f1: 0.8600)
- Model: SVC kernel=rbf, C=1.0
- Preprocessing: StandardScaler
- Feature engineering: none
- Training: fit()
- Result: f1_macro=0.8600, accuracy=0.8625
- Notes: Worse than RF. SVM struggles with feature scale variance here.
```

**Rules for experiment_memory.md**:
- Never delete entries, only append
- Always include why something was tried AND what the outcome was
- If a run crashed, still log it with "CRASH" status and the error type
- Do NOT commit this file to git (add to .gitignore)
- Before proposing a new experiment idea, scan this file to confirm it hasn't been tried

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit)
2. Read `experiment_memory.md` to see what's been tried — pick something NEW
3. Modify `experiment.py` with a new idea
4. If necessary, run `uv add <package_name>` to install missing dependencies
5. `git commit`
6. Run: `uv run experiment.py > run.log 2>&1`
7. Read results: `grep "^f1_macro:\|^accuracy:" run.log`
8. If grep is empty → crashed. Run `tail -n 30 run.log` for stack trace.
9. Save evaluation artifacts to `evaluation_output/<commit>/`
10. Log to `results.tsv`
11. Append to `experiment_memory.md`
12. If f1_macro improved → keep the commit (advance branch)
13. If f1_macro equal or worse → `git reset --hard HEAD~1` (but keep the memory entry!)

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
