"""Pure budget-scenario simulation shared by budget_optimization.py and the API.

Deliberately has no data-loading or training side effects — safe to import
in the FastAPI backend at startup without retraining a model or needing
data/cleaned_funnel_data.csv to exist (it doesn't, on Railway; data/ is
gitignored and never deployed).
"""

import pandas as pd


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
