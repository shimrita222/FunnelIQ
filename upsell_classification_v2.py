# %% [markdown]
# # Package 3 (v2) – Upsell Classification, without ltv_months
#
# Predict which customers are most likely to buy additional services.
#
# Target: `upsell` (1 = bought additional services, 0 = did not).
#
# **Why this v2 exists:** `upsell_classification.py` (v1) kept `ltv_months`
# as a feature, per that file's original brief. A later audit found
# `ltv_months` was v1's #1 feature at ~26% importance (with `purchased`
# right behind at ~22%) — but `ltv_months` is only fully known once a
# customer relationship has *ended*, the same category of late-relationship
# information Package 4 identified and removed in its own v1→v2 fix. This
# file mirrors v1 exactly except for that one change, so the two can be
# compared apples-to-apples.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")

DATA_DIR = Path(__file__).parent / "data"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_funnel_data.csv"

TARGET_COL = "upsell"
LEAKAGE_COLS = ["cumulative_profit", "referred", "ltv_months"]

RANDOM_STATE = 0
TEST_SIZE = 0.20
N_SPLITS = 5

# Business-rule thresholds: flag a customer for outreach if they've already
# made an initial purchase and their acquisition cost is below-median. v1's
# rule used `ltv_months > median`, but that column is no longer a feature in
# v2 (see leakage note below) — using it here would reintroduce the exact
# leakage this file exists to remove. `purchased` is a legitimately
# early-available substitute with similar intent: favor already-engaged,
# cheaply-acquired customers.
BUSINESS_RULE_CAC_THRESHOLD = 1000

# %% [markdown]
# ## Load Dataset

# %%
df = pd.read_csv(CLEANED_CSV_PATH)

print(f"Shape: {df.shape}")
df.head()

# %% [markdown]
# ## Prevent Data Leakage
#
# `cumulative_profit` and `referred` are excluded for the same reason as in
# v1: both are outcomes that only exist once the full customer relationship
# has played out. `ltv_months` is *also* excluded here — unlike v1, which
# kept it — because a customer's final lifetime is, by definition, only
# fully known once their relationship with Northbound has ended, so it
# wouldn't be available at the moment an upsell prediction is actually
# needed. The rest of the funnel columns are kept as features.


# %%
def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for upsell classification, excluding the target and leakage columns.

    No categorical encoding is needed: `referred` is the only non-numeric column
    in the cleaned dataset, and it's already excluded above as leakage.
    """
    X = data.drop(columns=[TARGET_COL, *LEAKAGE_COLS])
    y = data[TARGET_COL]
    return X, y


# %% [markdown]
# ## Prepare Data

# %%
X, y = prepare_data(df)

print(f"Number of features: {X.shape[1]}")
print("Feature names:", X.columns.tolist())
print("Class balance (y):")
print(y.value_counts(normalize=True))

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
print("Train class balance:")
print(y_train.value_counts(normalize=True))

# %% [markdown]
# ## Train Classification Models
#
# `upsell` is imbalanced (~58/42), so each model is configured with its own
# imbalance-handling mechanism rather than left to optimize for raw accuracy,
# which an imbalanced target can trivially inflate by always predicting the
# majority class.


# %%
def train_models(
    X_tr: pd.DataFrame, y_tr: pd.Series, random_state: int = RANDOM_STATE
) -> dict[str, ClassifierMixin]:
    """Instantiate and fit three classifiers, each with class-imbalance handling."""
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    scale_pos_weight = neg / pos

    models = {
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=random_state
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced", random_state=random_state, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            auto_class_weights="Balanced", random_state=random_state, verbose=False
        ),
    }
    for model in models.values():
        model.fit(X_tr, y_tr)
    return models


# %%
models = train_models(X_train, y_train)
for name in models:
    print(f"Trained {name}")

# %% [markdown]
# ## 5-Fold Stratified Cross-Validation
#
# Stratified so every fold preserves the ~58/42 class ratio. Run on `X_train`/
# `y_train` only — `X_test` stays untouched until final evaluation. Accuracy is
# reported alongside Precision/Recall/F1, but the model is *not* selected on
# accuracy alone, since that metric alone is misleading on an imbalanced target.


# %%
def cross_validate_models(
    classifiers: dict[str, ClassifierMixin], X_cv: pd.DataFrame, y_cv: pd.Series, cv
) -> pd.DataFrame:
    """Stratified k-fold CV per model; mean Accuracy/Precision/Recall/F1, sorted by F1."""
    rows = []
    for name, model in classifiers.items():
        scores = cross_validate(
            model,
            X_cv,
            y_cv,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1": scores["test_f1"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


# %%
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_comparison = cross_validate_models(models, X_train, y_train, cv)
cv_comparison

# %%
best_model_name = cv_comparison.iloc[0]["Model"]
best_model = models[best_model_name]
print(f"Best model by 5-fold CV F1: {best_model_name}")

# %% [markdown]
# ## Final Model Evaluation
#
# The best model by CV F1 is selected for business use — F1 balances catching
# real upsell candidates (recall) against not wasting outreach on customers who
# won't buy (precision), which matters more here than raw accuracy on an
# imbalanced target. `X_test`/`y_test` are touched for the first time now.


# %%
def evaluate_models(
    classifiers: dict[str, ClassifierMixin], X_eval: pd.DataFrame, y_eval: pd.Series
) -> pd.DataFrame:
    """Accuracy/Precision/Recall/F1 per model on a given (already-held-out) split."""
    rows = []
    for name, model in classifiers.items():
        y_pred = model.predict(X_eval)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_eval, y_pred),
                "Precision": precision_score(y_eval, y_pred),
                "Recall": recall_score(y_eval, y_pred),
                "F1": f1_score(y_eval, y_pred),
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


# %%
holdout_comparison = evaluate_models(models, X_test, y_test)
holdout_comparison

# %%
y_test_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"True negatives:  {tn}")
print(f"False positives: {fp}  (predicted upsell, but customer did not buy)")
print(f"False negatives: {fn}  (customer would upsell, but model missed them)")
print(f"True positives:  {tp}")

# %%
ConfusionMatrixDisplay(cm, display_labels=["No upsell", "Upsell"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_model_name} (test set)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **False positives** are customers flagged as likely upsells who don't buy —
# the cost of a false positive is wasted outreach effort/budget on that customer.
# **False negatives** are actual upsell customers the model missed — the cost is
# lost upsell revenue from a customer who was never targeted. Which error matters
# more depends on Northbound's cost of an outreach touch vs. the value of a
# missed upsell; the counts above quantify the current trade-off for the
# selected model.

# %% [markdown]
# ## Feature Importance


# %%
def get_feature_importance(model: ClassifierMixin, feature_names: list[str]) -> pd.Series:
    """Read-only accessor: importances off an already-fitted model, sorted descending."""
    if hasattr(model, "get_feature_importance"):  # CatBoost
        importances = model.get_feature_importance()
    else:  # XGBoost / LightGBM
        importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# %%
feature_names = X_train.columns.tolist()
importances = get_feature_importance(best_model, feature_names)
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

# %%
top_share = top_10_features.iloc[0] / importances.sum()
print(f"Top feature ({top_10_features.index[0]}) share of total importance: {top_share:.1%}")

# %% [markdown]
# If the top feature holds a large majority of total importance, upsell is
# effectively driven by one dominant signal; if importance is spread more evenly
# across the top 10, it's a combination of factors rather than a single lever —
# check the printed share above against the bar chart to judge which applies here.

# %% [markdown]
# ## Business Rule Comparison
#
# A simple, explainable baseline: flag a customer for upsell outreach if
# they've already made an initial purchase and their acquisition cost is
# below the dataset median — no model required. (v1's rule used
# `ltv_months > median`; that column is no longer a feature here, so
# `purchased` substitutes as a genuinely early-available signal in the same
# spirit — see the leakage note above.)


# %%
def apply_business_rule(data: pd.DataFrame) -> pd.Series:
    """Flag customers for outreach: already purchased and below-median CAC."""
    return (
        (data["purchased"] == 1)
        & (data["customer_acquisition_cost"] < BUSINESS_RULE_CAC_THRESHOLD)
    ).astype(int)


def evaluate_rule(rule_pred: pd.Series, y_eval: pd.Series) -> dict:
    """Accuracy/Precision/Recall/F1 for a rule-based prediction, same metrics as the models."""
    return {
        "Model": "Business rule",
        "Accuracy": accuracy_score(y_eval, rule_pred),
        "Precision": precision_score(y_eval, rule_pred),
        "Recall": recall_score(y_eval, rule_pred),
        "F1": f1_score(y_eval, rule_pred),
    }


# %%
rule_pred_test = apply_business_rule(X_test)
rule_vs_model = pd.concat(
    [
        holdout_comparison[holdout_comparison["Model"] == best_model_name],
        pd.DataFrame([evaluate_rule(rule_pred_test, y_test)]),
    ],
    ignore_index=True,
)
rule_vs_model

# %% [markdown]
# **Where does the ML model outperform the rule, and where does the rule win?**
# Compare each metric row above: a higher-Recall, lower-Precision rule usually
# means it casts a wider net (catches more real upsells but also flags more
# customers who won't buy) than the tuned model, or vice versa — read the actual
# Precision/Recall/F1 gap above to judge whether the added complexity of the ML
# model earns its keep over this two-threshold rule for Northbound's use case.
