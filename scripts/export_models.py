"""Persist the best-performing model from each analytical package to models/.

Training only ever happened in memory during a script/notebook run and
disappeared afterward. This script imports the already-verified training
modules (each import re-runs that module's own load -> train -> evaluate
pipeline, exactly as every analysis notebook in this project already relies
on) and dumps each package's selected best model to a `.joblib` file, so the
FastAPI backend can load them once at startup instead of retraining.

Also writes models/metadata.json: real evaluation metrics per model (holdout
or cross-validation, whichever that package's training script already
computes) plus, for the budget model, per-feature "driver directions" - a
lightweight two-point partial-dependence check (predict with each top
feature held at its 10th vs 90th percentile, everything else unchanged)
computed once here from the already-fitted model, not per API request, and
not a live correlation against raw data (which can contradict what a
tree-based model actually learned). This is what the API's confidence score
and business-impact text read from instead of any hardcoded constant.

Usage: uv run python -m scripts.export_models
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score

from budget_simulation import calculate_feature_importance

MODELS_DIR = Path(__file__).parent.parent / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_VERSION = "1.0"


def export(name: str, model, features: list[str], target: str) -> None:
    """Save one model + its feature order + target name to models/<name>.joblib."""
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump({"model": model, "features": features, "target": target}, path)
    print(f"Exported {name}: {len(features)} features, target={target!r} -> {path}")


def build_metadata(model_name: str, target: str, **metrics) -> dict:
    return {
        "model_name": model_name,
        "target": target,
        "training_date": datetime.now(UTC).isoformat(),
        "version": MODEL_VERSION,
        **metrics,
    }


def compute_driver_directions(model, X_train, feature_names: list[str], top_n: int = 5) -> dict:
    """Two-point partial dependence for the top-N important features.

    For each feature, predicts with every row's value replaced by that
    feature's 10th vs. 90th percentile (all other features unchanged) and
    takes the mean prediction difference - the model's own learned
    direction, not a raw correlation that could disagree with it.
    """
    importance = calculate_feature_importance(model, feature_names)
    top_features = importance.head(top_n).index.tolist()
    sample = X_train.sample(n=min(300, len(X_train)), random_state=0)

    effects = {}
    for feature in top_features:
        low = sample[feature].quantile(0.10)
        high = sample[feature].quantile(0.90)
        low_profit = model.predict(sample.assign(**{feature: low})).mean()
        high_profit = model.predict(sample.assign(**{feature: high})).mean()
        effects[feature] = float(high_profit - low_profit)

    ranked = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)
    n = len(ranked)
    directions = {}
    for i, (feature, effect) in enumerate(ranked):
        strength = "strong" if i < n / 3 else "medium" if i < 2 * n / 3 else "weak"
        directions[feature] = {
            "direction": "positive" if effect > 0 else "negative",
            "effect": effect,
            "strength": strength,
        }
    return directions


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    metadata: dict[str, dict] = {}

    from customer_lifetime_regression import TARGET_COL as ltv_target
    from customer_lifetime_regression import X_train as ltv_X_train
    from customer_lifetime_regression import best_model_name as ltv_best_name
    from customer_lifetime_regression import comparison as ltv_comparison
    from customer_lifetime_regression import models as ltv_models

    export("customer_lifetime_model", ltv_models[ltv_best_name], ltv_X_train.columns.tolist(), ltv_target)
    ltv_row = ltv_comparison[ltv_comparison["Model"] == ltv_best_name].iloc[0]
    metadata["customer_lifetime_model"] = build_metadata(
        ltv_best_name, ltv_target, holdout_rmse=float(ltv_row["RMSE"]), holdout_r2=float(ltv_row["R2"])
    )

    from upsell_classification_v2 import TARGET_COL as upsell_target
    from upsell_classification_v2 import X_train as upsell_X_train
    from upsell_classification_v2 import best_model_name as upsell_best_name
    from upsell_classification_v2 import cv_comparison as upsell_cv_comparison
    from upsell_classification_v2 import models as upsell_models

    export("upsell_model", upsell_models[upsell_best_name], upsell_X_train.columns.tolist(), upsell_target)
    upsell_row = upsell_cv_comparison[upsell_cv_comparison["Model"] == upsell_best_name].iloc[0]
    metadata["upsell_model"] = build_metadata(
        upsell_best_name, upsell_target, cv_f1=float(upsell_row["F1"]), cv_accuracy=float(upsell_row["Accuracy"])
    )

    from super_customer_score_v2 import TARGET_COL as super_target
    from super_customer_score_v2 import X_test as super_X_test
    from super_customer_score_v2 import X_train as super_X_train
    from super_customer_score_v2 import best_model as super_best_model
    from super_customer_score_v2 import y_test as super_y_test

    export("super_customer_model", super_best_model, super_X_train.columns.tolist(), super_target)
    super_pred = super_best_model.predict(super_X_test)
    metadata["super_customer_model"] = build_metadata(
        "CatBoost",
        super_target,
        holdout_accuracy=float(accuracy_score(super_y_test, super_pred)),
        holdout_f1=float(f1_score(super_y_test, super_pred)),
    )

    from budget_optimization import TARGET_COL as profit_target
    from budget_optimization import X_train as profit_X_train
    from budget_optimization import best_model_name as profit_best_name
    from budget_optimization import cv_comparison as profit_cv_comparison
    from budget_optimization import models as profit_models

    profit_best_model = profit_models[profit_best_name]
    export("budget_optimizer_model", profit_best_model, profit_X_train.columns.tolist(), profit_target)
    profit_row = profit_cv_comparison[profit_cv_comparison["Model"] == profit_best_name].iloc[0]
    metadata["budget_optimizer_model"] = build_metadata(
        profit_best_name,
        profit_target,
        cv_r2=float(profit_row["Mean R2"]),
        cv_rmse=float(profit_row["Mean RMSE"]),
        driver_directions=compute_driver_directions(profit_best_model, profit_X_train, profit_X_train.columns.tolist()),
    )

    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
