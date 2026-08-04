# %% [markdown]
# # Package 2 – Customer Lifetime Prediction
#
# First stage of Package 2: prepare the data correctly, prevent data leakage, train
# three baseline regression models, and run an initial evaluation. No cross-validation,
# feature importance, SHAP, hyperparameter tuning, or business interpretation here —
# those come in a later notebook.
#
# Target: `ltv_months` (customer lifetime in months).

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).parent / "data"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_funnel_data.csv"

TARGET_COL = "ltv_months"
LEAKAGE_COLS = ["cumulative_profit", "referred", "upsell"]

RANDOM_STATE = 0
TEST_SIZE = 0.20

# %% [markdown]
# ## Load Dataset

# %%
df = pd.read_csv(CLEANED_CSV_PATH)

print(f"Shape: {df.shape}")
df.head()

# %%
print("Columns:", df.columns.tolist())
df.dtypes

# %% [markdown]
# ## Prevent Data Leakage
#
# Before building `X`/`y`, four columns must be excluded from the feature set:
#
# - **`ltv_months`** — this is the prediction target; it cannot also be an input.
# - **`cumulative_profit`** — only known once the full customer lifecycle has played
#   out, so it contains information from *after* the outcome we're predicting.
# - **`referred`** — typically only becomes known after a customer has been active
#   for a while, so it isn't available at prediction time either.
# - **`upsell`** — a purchase event that can happen at any point during the
#   relationship, with no guarantee it resolves before `ltv_months` is known; the
#   same late-relationship-information risk already fixed for Package 3
#   (`upsell_classification_v2.py`, which drops `ltv_months` for the mirror-image
#   reason) and Package 4 (`super_customer_score_early_features.ipynb`).
#
# Including any of the last three would leak future information into the model and
# produce unrealistically optimistic performance that would not hold up in production.

# %% [markdown]
# ## Create Features and Target

# %%
X = df.drop(columns=[TARGET_COL, *LEAKAGE_COLS])
y = df[TARGET_COL]

print(f"Number of features: {X.shape[1]}")
print("Feature names:", X.columns.tolist())

# %%
y.describe()

# %% [markdown]
# ## Train/Test Split
#
# `random_state=0` makes the split reproducible: every run of this script assigns the
# exact same rows to the training and test sets, so results are directly comparable
# across runs.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# %% [markdown]
# ## Feature Scaling
#
# No `StandardScaler`/`MinMaxScaler` is applied. XGBoost, LightGBM, and CatBoost are
# all decision-tree-based: they split on raw feature values (`feature <= threshold`),
# so the scale or distribution of a feature doesn't affect how they choose splits.
# Scaling would be required for distance- or gradient-based models like linear
# regression, SVMs, or k-NN — not for tree ensembles.

# %% [markdown]
# ## Train Regression Models


# %%
def build_models(random_state: int) -> dict[str, RegressorMixin]:
    """Instantiate the three regressors with reasonable defaults (no tuning)."""
    return {
        "XGBoost": XGBRegressor(random_state=random_state),
        "LightGBM": LGBMRegressor(random_state=random_state, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=random_state, verbose=False),
    }


# %%
models = build_models(RANDOM_STATE)

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Trained {name}")

# %% [markdown]
# ## Initial Model Evaluation


# %%
def evaluate_model(model: RegressorMixin, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict:
    """Return RMSE, MAE, and R² for a fitted model on held-out data."""
    y_pred = model.predict(X_eval)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_eval, y_pred)),
        "MAE": mean_absolute_error(y_eval, y_pred),
        "R2": r2_score(y_eval, y_pred),
    }


# %%
results = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}

comparison = (
    pd.DataFrame(results)
    .T.rename_axis("Model")
    .reset_index()
    .sort_values("RMSE", ascending=True)
    .reset_index(drop=True)
)
comparison

# %%
best_model_name = comparison.iloc[0]["Model"]
print(f"Best-performing model (lowest RMSE): {best_model_name}")
