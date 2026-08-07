"""Pure budget-scenario simulation shared by budget_optimization.py and the API.

Deliberately has no data-loading or training side effects — safe to import
in the FastAPI backend at startup without retraining a model or needing
data/cleaned_funnel_data.csv to exist (it doesn't, on Railway; data/ is
gitignored and never deployed).
"""

import pandas as pd
from sklearn.base import RegressorMixin


def calculate_feature_importance(model: RegressorMixin, feature_names: list[str]) -> pd.Series:
    """Read-only accessor: importances off an already-fitted model, sorted descending."""
    if hasattr(model, "get_feature_importance"):  # CatBoost
        importances = model.get_feature_importance()
    else:  # XGBoost / LightGBM
        importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def generate_business_report(comparison_df: pd.DataFrame, importance: pd.Series) -> dict:
    """Answer the brief's business questions from the actual computed results."""
    best_scenario = comparison_df.loc[comparison_df["Predicted Profit"].idxmax()]
    equal_row = comparison_df.iloc[0]
    increase_row = comparison_df.iloc[1]
    decrease_row = comparison_df.iloc[2]

    equal_is_optimal = best_scenario["Scenario"] == equal_row["Scenario"]
    more_budget_helps = increase_row["Predicted Profit"] > equal_row["Predicted Profit"]
    less_budget_hurts = decrease_row["Predicted Profit"] < equal_row["Predicted Profit"]
    top_feature = importance.index[0]

    answers = {
        "is_equal_allocation_optimal": equal_is_optimal,
        "best_scenario": best_scenario["Scenario"],
        "does_more_budget_always_help": more_budget_helps and less_budget_hurts,
        "top_profit_driver": top_feature,
    }

    recommendation = (
        f"Best-performing scenario: '{best_scenario['Scenario']}' "
        f"(predicted profit NIS {best_scenario['Predicted Profit']:,.0f}, "
        f"{best_scenario['Profit Difference']:+,.0f} vs. the current equal-allocation "
        f"strategy). Equal allocation is "
        + ("" if equal_is_optimal else "NOT ")
        + "the optimal strategy among those tested. "
        + (
            "More total budget does increase predicted profit, and less budget "
            "reduces it, consistent with a straightforward spend-response "
            "relationship in this range."
            if answers["does_more_budget_always_help"]
            else "Predicted profit does not scale simply with total budget size — "
            "check the scenario table for where the relationship breaks down."
        )
        + f" The strongest driver of predicted profit is '{top_feature}'."
    )

    return {"answers": answers, "recommendation": recommendation}


def simulate_budget_scenarios(
    model,
    profile_sample: pd.DataFrame,
    monthly_budget: float = 50_000,
) -> pd.DataFrame:
    """Predict total profit under 4 budget-allocation scenarios for a fixed set of profiles.

    Only `ad_budget` is varied per scenario; every other feature is held at
    each profile's original value (the model was not trained to simulate
    how more spend would also change lead volume — out of scope here).
    """
    n = len(profile_sample)

    def predict_total_profit(budgets: pd.Series) -> float:
        scenario_profiles = profile_sample.copy()
        scenario_profiles["ad_budget"] = budgets.to_numpy()
        return model.predict(scenario_profiles).sum()

    # Scenario 1: current strategy — equal allocation of the base budget.
    equal_budget = monthly_budget / n
    equal_budgets = pd.Series([equal_budget] * n, index=profile_sample.index)
    scenario_1_profit = predict_total_profit(equal_budgets)

    # Scenario 2 / 3: equal allocation of a +/-20% total budget.
    increased_budget = monthly_budget * 1.2 / n
    decreased_budget = monthly_budget * 0.8 / n
    scenario_2_profit = predict_total_profit(pd.Series([increased_budget] * n, index=profile_sample.index))
    scenario_3_profit = predict_total_profit(pd.Series([decreased_budget] * n, index=profile_sample.index))

    # Scenario 4: rank profiles by their Scenario-1 predicted profit, then
    # allocate the base budget in linearly decreasing shares (rank 1 gets the
    # largest share, rank N the smallest) — robust to zero/negative predicted
    # profits, unlike a profit-proportional split.
    scenario_1_per_profile = model.predict(
        profile_sample.assign(ad_budget=equal_budget)
    )
    ranks = pd.Series(scenario_1_per_profile, index=profile_sample.index).rank(
        ascending=False, method="first"
    )
    linear_weights = (n - ranks + 1) / (n * (n + 1) / 2)
    weighted_budgets = linear_weights * monthly_budget
    scenario_4_profit = predict_total_profit(weighted_budgets)

    scenarios = pd.DataFrame(
        [
            {"Scenario": "1. Equal Allocation (Current)", "Total Budget": monthly_budget, "Predicted Profit": scenario_1_profit},
            {"Scenario": "2. Increase Budget +20%", "Total Budget": monthly_budget * 1.2, "Predicted Profit": scenario_2_profit},
            {"Scenario": "3. Decrease Budget -20%", "Total Budget": monthly_budget * 0.8, "Predicted Profit": scenario_3_profit},
            {"Scenario": "4. Profit-Weighted Allocation", "Total Budget": monthly_budget, "Predicted Profit": scenario_4_profit},
        ]
    )
    scenarios["Average Budget"] = scenarios["Total Budget"] / n
    scenarios["Profit Difference"] = scenarios["Predicted Profit"] - scenario_1_profit
    scenarios["ROI"] = (scenarios["Predicted Profit"] - scenarios["Total Budget"]) / scenarios["Total Budget"]
    return scenarios[["Scenario", "Average Budget", "Predicted Profit", "Profit Difference", "ROI"]]


def simulate_by_campaign_count(
    model,
    profile_pool: pd.DataFrame,
    monthly_budget: float = 50_000,
    campaign_counts: tuple[int, ...] = (1, 5, 10, 25, 50),
    random_state: int = 0,
) -> pd.DataFrame:
    """Compare spreading the same total budget across different campaign counts.

    For each candidate count N, samples N real profiles from `profile_pool`,
    splits `monthly_budget` equally across them (`ad_budget = monthly_budget / N`
    per profile, every other feature held at its original value), and sums the
    model's predicted profit. Counts larger than the pool size are skipped.
    """
    rows = []
    for n in campaign_counts:
        if n > len(profile_pool):
            continue
        sample = profile_pool.sample(n=n, random_state=random_state).copy()
        sample["ad_budget"] = monthly_budget / n
        predicted_profit = model.predict(sample).sum()
        rows.append({"n_campaigns": n, "predicted_profit": predicted_profit})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Business-intelligence layer: recommendation confidence, driver categorization
# and business-impact text, narrative analysis, and risks/opportunities.
# All of this reads only from already-computed scenario/sweep tables and the
# model's exported metadata (models/metadata.json) - no live Supabase calls,
# no retraining, no correlation computed against raw data at request time.
# ---------------------------------------------------------------------------

LEAD_VOLUME_FEATURES = {"num_leads", "leads_answered", "leads_not_answered"}
FUNNEL_EXECUTION_FEATURES = {
    "followup_1", "followup_2", "followup_3", "followup_4", "followup_5",
    "calls_to_closed", "calls_to_not_closed", "closed", "not_closed", "purchased",
}
COST_FEATURES = {"ad_budget", "customer_acquisition_cost"}

CATEGORY_ACTION = {
    "lead volume": "growing qualified lead volume",
    "funnel execution": "improving follow-up execution and conversion efficiency",
    "acquisition cost & spend": "reducing acquisition cost",
}

STRENGTH_ADVERB = {"strong": "strongly", "medium": "moderately", "weak": "weakly"}


def categorize_budget_driver(feature_name: str) -> str:
    """Map a raw feature name to a plain-language business category."""
    if feature_name in LEAD_VOLUME_FEATURES:
        return "lead volume"
    if feature_name in FUNNEL_EXECUTION_FEATURES:
        return "funnel execution"
    if feature_name in COST_FEATURES:
        return "acquisition cost & spend"
    return "other"


def describe_driver_business_impact(feature_name: str, direction: str, strength: str) -> str:
    """Turn a metadata driver_directions entry into a plain business sentence.

    Pure string formatting - direction/strength already came from the
    model's own two-point partial dependence (see scripts/export_models.py),
    not recomputed here.
    """
    display = feature_name.replace("_", " ")
    adverb = STRENGTH_ADVERB.get(strength, "")
    verb = "Lower" if direction == "negative" else "Higher"
    return f"{verb} {display} is {adverb} associated with higher predicted profit.".replace("  ", " ")


def calculate_recommendation_confidence(scenarios: pd.DataFrame, model_cv_r2: float) -> dict:
    """Confidence in the recommendation - not just the model.

    Combines two already-available signals, weighted equally, no arbitrary
    heuristics: (1) the budget model's own cross-validation R2 (from
    models/metadata.json, not recomputed here - that would mean retraining),
    and (2) how clearly the best scenario leads the next-best alternative
    (a scenario table where all options are near-identical warrants lower
    confidence in recommending one over the others, regardless of model fit).
    """
    sorted_profits = scenarios["Predicted Profit"].sort_values(ascending=False)
    best, second_best = sorted_profits.iloc[0], sorted_profits.iloc[1]
    separation = min(1.0, (best - second_best) / abs(best)) if best else 0.0
    score = round((model_cv_r2 + separation) / 2, 2)
    confidence = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
    explanation = (
        f"The underlying model explains ~{model_cv_r2 * 100:.0f}% of profit variance in "
        f"cross-validation (R² ≈ {model_cv_r2:.2f}), and the recommended scenario's "
        f"predicted profit leads the next-best alternative tested by {separation * 100:.0f}%."
    )
    return {"confidence": confidence, "confidence_score": score, "explanation": explanation}


def generate_budget_business_analysis(
    scenarios: pd.DataFrame,
    importance: pd.Series,
    by_campaign_count: pd.DataFrame,
    driver_directions: dict,
) -> dict:
    """3-part narrative: Business Insight / Key Observation / Business Impact."""
    top_feature = importance.index[0]
    top_category = categorize_budget_driver(top_feature)
    n_features = len(importance)
    ad_budget_rank = list(importance.index).index("ad_budget") + 1 if "ad_budget" in importance.index else None

    if ad_budget_rank is not None and top_feature != "ad_budget" and ad_budget_rank > n_features / 2:
        business_insight = (
            f"Historical data indicates that {top_category} has more influence on predicted profit "
            f"than the size of the ad budget itself (ad budget ranks {ad_budget_rank} of {n_features} "
            "features the model considered)."
        )
    else:
        business_insight = (
            f"Historical data indicates that {top_category} is the strongest measurable influence "
            "on predicted profit among the features the model considered."
        )

    bcc = by_campaign_count.copy()
    bcc["profit_per_campaign"] = bcc["predicted_profit"] / bcc["n_campaigns"]
    bcc_sorted = bcc.sort_values("n_campaigns")
    if len(bcc_sorted) >= 2 and bcc_sorted.iloc[-1]["profit_per_campaign"] < bcc_sorted.iloc[0]["profit_per_campaign"]:
        campaign_trend = (
            "profit per campaign declines as more campaigns are added in the range tested, "
            "a sign of diminishing returns at higher campaign counts"
        )
    else:
        campaign_trend = (
            "profit per campaign holds up as more campaigns are added in the range tested, "
            "with no strong sign of diminishing returns"
        )
    if top_feature in driver_directions:
        d = driver_directions[top_feature]
        direction_phrase = "decreases" if d["direction"] == "negative" else "increases"
        key_observation = (
            f"The model's predicted profit generally {direction_phrase} as {top_feature.replace('_', ' ')} "
            f"rises, holding other factors fixed; separately, {campaign_trend}."
        )
    else:
        key_observation = campaign_trend.capitalize() + "."

    business_impact = (
        f"Focusing on {CATEGORY_ACTION.get(top_category, top_category)} is likely to have more effect "
        "on expected profit than adjusting the ad budget size alone."
    )

    return {
        "business_insight": business_insight,
        "key_observation": key_observation,
        "business_impact": business_impact,
    }


def generate_budget_risks_and_opportunities(
    scenarios: pd.DataFrame,
    importance: pd.Series,
    by_campaign_count: pd.DataFrame,
) -> dict:
    """Risks/opportunities split into data-driven (verified against real
    computed signals) and general (labeled guidance, never presented as a
    model finding)."""
    risks_data_driven: list[str] = []
    opportunities_data_driven: list[str] = []

    bcc = by_campaign_count.copy()
    bcc["profit_per_campaign"] = bcc["predicted_profit"] / bcc["n_campaigns"]
    bcc_sorted = bcc.sort_values("n_campaigns")
    if len(bcc_sorted) >= 2 and bcc_sorted.iloc[-1]["profit_per_campaign"] < bcc_sorted.iloc[0]["profit_per_campaign"]:
        first, last = bcc_sorted.iloc[0], bcc_sorted.iloc[-1]
        risks_data_driven.append(
            f"Diminishing returns at higher campaign counts: predicted profit per campaign falls from "
            f"NIS {first['profit_per_campaign']:,.0f} at {int(first['n_campaigns'])} campaigns to "
            f"NIS {last['profit_per_campaign']:,.0f} at {int(last['n_campaigns'])} campaigns."
        )

    n_features = len(importance)
    ad_budget_rank = list(importance.index).index("ad_budget") + 1 if "ad_budget" in importance.index else None
    if ad_budget_rank is not None and ad_budget_rank > n_features / 2:
        risks_data_driven.append(
            f"Ad budget ranks {ad_budget_rank} of {n_features} features in predictive importance - "
            "assuming higher spend alone will drive higher profit is not well supported by the model."
        )
        opportunities_data_driven.append(
            "Since ad spend has limited predictive leverage here, reallocating effort toward the "
            "higher-ranked operational drivers below is likely to be more effective than increasing budget."
        )

    top_feature = importance.index[0]
    top_category = categorize_budget_driver(top_feature)
    category_opportunity = {
        "lead volume": "Increasing qualified lead volume appears to be a high-leverage opportunity for profit growth.",
        "funnel execution": "Improving follow-up execution and conversion efficiency appears to be a high-leverage opportunity for profit growth.",
        "acquisition cost & spend": "Reducing acquisition cost appears to be a high-leverage opportunity for profit growth.",
    }
    driver_opportunity = category_opportunity.get(
        top_category, f"Optimizing {top_category} appears to be a high-leverage opportunity for profit growth."
    )
    if driver_opportunity not in opportunities_data_driven:
        opportunities_data_driven.append(driver_opportunity)

    general_risks = [
        (
            "This simulation reflects the historical data range - results may not hold for budgets or "
            "market conditions well outside it."
        ),
    ]
    general_opportunities = [
        "Validate any real allocation change with a controlled pilot before a full rollout.",
    ]

    return {
        "risks": {"data_driven": risks_data_driven, "general": general_risks},
        "opportunities": {"data_driven": opportunities_data_driven, "general": general_opportunities},
    }
