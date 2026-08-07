# %% [markdown]
# # Package 6 – Budget Optimization Challenge
#
# Northbound spends ₪50,000/month, split equally across campaigns. This
# package trains a model to predict `cumulative_profit`, then uses it to test
# whether a different allocation of the same ₪50,000 would produce more
# profit than an equal split.
#
# **Limitation, stated up front:** the dataset has no campaign identifier or
# advertising channel column. No campaign-level data is invented here —
# instead, budget scenarios are simulated by varying `ad_budget` for real,
# sampled customer profiles and reading off the trained model's predicted
# profit. Every other feature (leads, follow-ups, etc.) is held at each
# sampled profile's original value, so the simulation reflects the model's
# learned association between spend and profit *holding the rest of the
# funnel fixed* — not a full re-simulation of how more spend would also
# change lead volume (a relationship Package 1 found exists, but modeling it
# is out of scope here).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from xgboost import XGBRegressor

from budget_simulation import (
    calculate_feature_importance,
    generate_business_report,
    simulate_budget_scenarios,
)

sns.set_theme(style="whitegrid")

DATA_DIR = Path(__file__).parent / "data"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_funnel_data.csv"

TARGET_COL = "cumulative_profit"
LEAKAGE_COLS = ["ltv_months", "upsell", "referred"]

RANDOM_STATE = 0
TEST_SIZE = 0.20
N_SPLITS = 5
MONTHLY_BUDGET = 50_000

# %% [markdown]
# ## Load Data & Prevent Data Leakage
#
# `ltv_months`, `upsell`, and `referred` are all outcomes that only resolve
# after long-term customer behavior is known — using them to predict profit
# would leak the future into the model. `closed`, `not_closed`, `purchased`,
# `calls_to_closed`, and `calls_to_not_closed` describe the conversion stage
# itself and are kept as features, per this package's brief (the same
# treatment Package 2 gave these columns for LTV prediction).


# %%
def load_data(path: Path = CLEANED_CSV_PATH) -> pd.DataFrame:
    """Read the cleaned dataset. Read-only — never writes back to `path`."""
    return pd.read_csv(path)


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for cumulative_profit prediction, excluding target + leakage."""
    X = data.drop(columns=[TARGET_COL, *LEAKAGE_COLS])
    y = data[TARGET_COL]
    return X, y


# %%
df = load_data()
X, y = prepare_features(df)

print(f"Number of features: {X.shape[1]}")
print("Feature names:", X.columns.tolist())

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# %% [markdown]
# ## Train Regression Models


# %%
def train_models(X_tr: pd.DataFrame, y_tr: pd.Series, random_state: int = RANDOM_STATE) -> dict:
    """Instantiate and fit three regressors with default parameters."""
    models = {
        "XGBoost": XGBRegressor(random_state=random_state),
        "LightGBM": LGBMRegressor(random_state=random_state, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=random_state, verbose=False),
    }
    for model in models.values():
        model.fit(X_tr, y_tr)
    return models


# %%
models = train_models(X_train, y_train)
for name in models:
    print(f"Trained {name}")

# %% [markdown]
# ## Model Comparison (RMSE / MAE / R²)


# %%
def compare_models(models: dict, X_eval: pd.DataFrame, y_eval: pd.Series) -> pd.DataFrame:
    """RMSE/MAE/R² per model on a given split, sorted by RMSE ascending."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_eval)
        rows.append(
            {
                "Model": name,
                "RMSE": mean_squared_error(y_eval, y_pred) ** 0.5,
                "MAE": mean_absolute_error(y_eval, y_pred),
                "R2": r2_score(y_eval, y_pred),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# %%
holdout_comparison = compare_models(models, X_test, y_test)
holdout_comparison

# %%
best_model_name = holdout_comparison.iloc[0]["Model"]
best_model = models[best_model_name]
print(f"Best model by holdout RMSE: {best_model_name}")

# %% [markdown]
# ## 5-Fold Cross-Validation
#
# Run on `X_train`/`y_train` only — `X_test` stays untouched.


# %%
def cross_validate_models(models: dict, X_cv: pd.DataFrame, y_cv: pd.Series, cv) -> pd.DataFrame:
    """5-fold CV per model; Mean/Std RMSE and Mean/Std R², sorted by Mean RMSE."""
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_cv,
            y_cv,
            cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        )
        rows.append(
            {
                "Model": name,
                "Mean RMSE": -scores["test_rmse"].mean(),
                "Std RMSE": scores["test_rmse"].std(),
                "Mean R2": scores["test_r2"].mean(),
                "Std R2": scores["test_r2"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean RMSE").reset_index(drop=True)


# %%
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_comparison = cross_validate_models(models, X_train, y_train, cv)
cv_comparison

# %% [markdown]
# ## Feature Importance


# %%
importances = calculate_feature_importance(best_model, X_train.columns.tolist())
top_10_features = importances.head(10)
top_10_features

# %%
plt.figure(figsize=(8, 6))
plot_order = top_10_features.iloc[::-1]
sns.barplot(x=plot_order.to_numpy(), y=plot_order.index, orient="h")
plt.title(f"Top 10 feature importances — {best_model_name}")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Budget Scenario Simulation
#
# `N` "campaign slots" are needed to split a fixed monthly budget, but the
# dataset has no campaign identifier. `N = round(monthly_budget / mean(ad_budget))`
# (≈10) is used as a data-grounded stand-in: it's roughly how many campaigns
# Northbound's current average spend implies are already running concurrently.
# `N` real customer profiles are sampled from `X_test` to represent those
# slots — no synthetic campaign data is created.


# %%
n_campaigns = round(MONTHLY_BUDGET / df["ad_budget"].mean())
print(f"N campaign slots: {n_campaigns} (NIS {MONTHLY_BUDGET:,} / NIS {df['ad_budget'].mean():,.0f} avg ad_budget)")

profile_sample = X_test.sample(n=n_campaigns, random_state=RANDOM_STATE)
scenario_comparison = simulate_budget_scenarios(best_model, profile_sample)
scenario_comparison

# %% [markdown]
# ## Business Report


# %%
report = generate_business_report(scenario_comparison, importances)
print("ANSWERS:")
for key, value in report["answers"].items():
    print(f"  {key}: {value}")
print()
print("RECOMMENDATION:")
print(report["recommendation"])
