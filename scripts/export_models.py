"""Persist the best-performing model from each analytical package to models/.

Training only ever happened in memory during a script/notebook run and
disappeared afterward. This script imports the already-verified training
modules (each import re-runs that module's own load -> train -> evaluate
pipeline, exactly as every analysis notebook in this project already relies
on) and dumps each package's selected best model to a `.joblib` file, so the
FastAPI backend can load them once at startup instead of retraining.

Usage: uv run python -m scripts.export_models
"""

from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"


def export(name: str, model, features: list[str], target: str) -> None:
    """Save one model + its feature order + target name to models/<name>.joblib."""
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump({"model": model, "features": features, "target": target}, path)
    print(f"Exported {name}: {len(features)} features, target={target!r} -> {path}")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    from customer_lifetime_regression import TARGET_COL as ltv_target
    from customer_lifetime_regression import X_train as ltv_X_train
    from customer_lifetime_regression import best_model_name as ltv_best_name
    from customer_lifetime_regression import models as ltv_models

    export(
        "customer_lifetime_model",
        ltv_models[ltv_best_name],
        ltv_X_train.columns.tolist(),
        ltv_target,
    )

    from upsell_classification_v2 import TARGET_COL as upsell_target
    from upsell_classification_v2 import X_train as upsell_X_train
    from upsell_classification_v2 import best_model_name as upsell_best_name
    from upsell_classification_v2 import models as upsell_models

    export(
        "upsell_model",
        upsell_models[upsell_best_name],
        upsell_X_train.columns.tolist(),
        upsell_target,
    )

    from super_customer_score_v2 import TARGET_COL as super_target
    from super_customer_score_v2 import X_train as super_X_train
    from super_customer_score_v2 import best_model as super_best_model

    export(
        "super_customer_model",
        super_best_model,
        super_X_train.columns.tolist(),
        super_target,
    )

    from budget_optimization import TARGET_COL as profit_target
    from budget_optimization import X_train as profit_X_train
    from budget_optimization import best_model_name as profit_best_name
    from budget_optimization import models as profit_models

    export(
        "budget_optimizer_model",
        profit_models[profit_best_name],
        profit_X_train.columns.tolist(),
        profit_target,
    )


if __name__ == "__main__":
    main()
