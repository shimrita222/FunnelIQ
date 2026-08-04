"""Prediction endpoints backed by the models exported to models/*.joblib.

Every endpoint here is gated behind the same `get_current_user` auth already
protecting `/customers`/`/statistics` — FunnelIQ is an internal tool, and
these predictions are no exception.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.auth import get_current_user
from backend.supabase_client import get_supabase_admin_client
from budget_simulation import simulate_budget_scenarios
from followup_metrics import (
    analyze_sales_calls,
    calculate_dropout_rates,
    calculate_followup_statistics,
    generate_business_summary,
)
from funnel_scoring import predict_super_customer_score
from supabase import Client

router = APIRouter()

MODELS_DIR = Path(__file__).parent.parent / "models"


def _load(name: str) -> dict:
    """Load one exported model bundle: {"model", "features", "target"}."""
    return joblib.load(MODELS_DIR / f"{name}.joblib")


_lifetime_bundle = _load("customer_lifetime_model")
_upsell_bundle = _load("upsell_model")
_super_customer_bundle = _load("super_customer_model")
_budget_bundle = _load("budget_optimizer_model")


class CustomerFeatures(BaseModel):
    """The 15 features shared by every prediction endpoint (same exclusions
    applied across Packages 2/3v2/4v2/6: cumulative_profit, referred, and
    whichever of ltv_months/upsell isn't that model's own target)."""

    ad_budget: float
    num_leads: float
    leads_answered: float
    leads_not_answered: float
    followup_1: float
    followup_2: float
    followup_3: float
    followup_4: float
    followup_5: float
    not_closed: float
    closed: float
    calls_to_closed: float
    calls_to_not_closed: float
    customer_acquisition_cost: float
    purchased: int


def _to_frame(features: CustomerFeatures, expected_columns: list[str]) -> pd.DataFrame:
    """Build a one-row DataFrame reindexed to a model's exact training column order."""
    return pd.DataFrame([features.model_dump()])[expected_columns]


@router.post("/predict/lifetime")
def predict_lifetime(features: CustomerFeatures, _user=Depends(get_current_user)):
    row = _to_frame(features, _lifetime_bundle["features"])
    prediction = _lifetime_bundle["model"].predict(row)[0]
    return {"predicted_ltv_months": float(prediction)}


@router.post("/predict/upsell")
def predict_upsell(features: CustomerFeatures, _user=Depends(get_current_user)):
    row = _to_frame(features, _upsell_bundle["features"])
    model = _upsell_bundle["model"]
    probability = float(model.predict_proba(row)[0, 1])
    return {"upsell_probability": probability, "predicted_upsell": int(model.predict(row)[0])}


@router.post("/predict/super-customer")
def predict_super_customer(features: CustomerFeatures, _user=Depends(get_current_user)):
    row = _to_frame(features, _super_customer_bundle["features"])
    result = predict_super_customer_score(row, _super_customer_bundle["model"]).iloc[0]
    return {"score": float(result["Score"]), "tier": result["Tier"]}


class BudgetOptimizationRequest(BaseModel):
    monthly_budget: float = 50_000


def _fetch_all(supabase: Client, columns: str, page_size: int = 1000) -> list[dict]:
    """Same pagination pattern as backend/main.py's `_fetch_all`."""
    rows: list[dict] = []
    offset = 0
    while True:
        page = (
            supabase.table("customers")
            .select(columns)
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


@router.post("/budget-optimization")
def budget_optimization(
    request: BudgetOptimizationRequest = BudgetOptimizationRequest(),
    supabase: Client = Depends(get_supabase_admin_client),
    _user=Depends(get_current_user),
):
    feature_cols = _budget_bundle["features"]
    rows = _fetch_all(supabase, ",".join(feature_cols))
    profiles = pd.DataFrame(rows)[feature_cols]
    profiles["purchased"] = profiles["purchased"].astype(int)

    n_campaigns = round(request.monthly_budget / profiles["ad_budget"].mean())
    profile_sample = profiles.sample(n=n_campaigns, random_state=0)

    scenarios = simulate_budget_scenarios(
        _budget_bundle["model"], profile_sample, monthly_budget=request.monthly_budget
    )
    return {"n_campaigns": n_campaigns, "scenarios": scenarios.to_dict(orient="records")}


@router.get("/followup-analysis")
def followup_analysis(
    supabase: Client = Depends(get_supabase_admin_client),
    _user=Depends(get_current_user),
):
    columns = "leads_answered,followup_1,followup_2,followup_3,followup_4,followup_5,closed,calls_to_closed,calls_to_not_closed"
    rows = _fetch_all(supabase, columns)
    df = pd.DataFrame(rows)

    stats = calculate_followup_statistics(df)
    dropout_stats = calculate_dropout_rates(stats)
    call_stats = analyze_sales_calls(df)
    summary = generate_business_summary(dropout_stats, call_stats)

    return {
        # The baseline row has no previous stage, so its Drop Rate/Count are
        # NaN (see followup_metrics.calculate_dropout_rates) — not valid JSON.
        "dropout_by_stage": dropout_stats.astype(object).where(dropout_stats.notna(), None).to_dict(orient="records"),
        "call_stats": call_stats,
        "conclusion": summary["conclusion"],
        "recommendation": summary["recommendation"],
    }
