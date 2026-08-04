"""Package 4 (v2) training module — Super Customer score, early features only.

Extracted from `super_customer_score_early_features.ipynb` so the trained
model can be imported by both the notebook (unchanged presentation/analysis)
and `scripts/export_models.py` (model persistence for the API), instead of
being trapped inside notebook-only cells.

Uses the hyperparameters already found by that notebook's `GridSearchCV`
(depth=4, iterations=200, learning_rate=0.01, best CV F1 ≈ 0.769) directly —
does not re-run the search, matching what was confirmed with the user.

Note: this module has top-level data-loading and training code (runs on
import, same pattern every analysis script in this project uses) — that's
fine for the notebook and `scripts/export_models.py`, both one-off local
runs, but it must never be imported by the live backend. `predict_super_customer_score`
itself is re-exported here from `funnel_scoring.py` (side-effect-free) for
backward compatibility with existing imports of this module.
"""

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from funnel_scoring import predict_super_customer_score  # noqa: F401 (re-exported)

DATA_DIR = Path(__file__).parent / "data"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_funnel_data.csv"

TARGET_COL = "referred"
# Excluded: cumulative_profit (this package's original leakage exclusion) and
# upsell/ltv_months (late-relationship outcomes, per this version's early-funnel scope).
EXCLUDED_COLS = ["cumulative_profit", "upsell", "ltv_months"]
RANDOM_STATE = 0

# Best hyperparameters already found via GridSearchCV in
# super_customer_score_early_features.ipynb (depth/learning_rate/iterations
# over a grid, scored on 5-fold stratified CV F1) — reused directly here.
BEST_PARAMS = {"depth": 4, "iterations": 200, "learning_rate": 0.01}


df = pd.read_csv(CLEANED_CSV_PATH)

X = df.drop(columns=[TARGET_COL, *EXCLUDED_COLS])
y = (df[TARGET_COL] == "Yes").astype(int)

# `referred` was the only non-numeric column in the cleaned dataset and it's the
# target, so every remaining feature is already numeric — no categorical columns
# to pass to CatBoost's native handling.
cat_features = X.select_dtypes(include="object").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

best_model = CatBoostClassifier(
    cat_features=cat_features or None,
    auto_class_weights="Balanced",
    random_state=RANDOM_STATE,
    verbose=False,
    **BEST_PARAMS,
)
best_model.fit(X_train, y_train)
