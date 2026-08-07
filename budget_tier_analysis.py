"""Pure budget-tier conversion analysis shared by exploration_and_cleaning.py and the API.

Deliberately has no data-loading side effects — safe to import in the FastAPI
backend at startup without needing data/cleaned_funnel_data.csv to exist (it
doesn't, on Railway; data/ is gitignored and never deployed).
"""

import pandas as pd

TIER_ORDER = ["Low (<=1500)", "Gap (1500-2000)", "Mid (2000-5000)", "High (>5000)"]


def assign_budget_tier(ad_budget: float) -> str:
    """Bucket ad_budget into the brief's Low/Mid/High tiers.

    Note: the brief defines Low as <=1500 and Mid as 2000-5000, leaving a gap
    between 1500 and 2000 that belongs to neither tier. Rows in that gap are
    labeled "Gap (1500-2000)" and reported separately rather than silently
    forced into an adjacent tier.
    """
    if ad_budget <= 1500:
        return "Low (<=1500)"
    if ad_budget < 2000:
        return "Gap (1500-2000)"
    if ad_budget <= 5000:
        return "Mid (2000-5000)"
    return "High (>5000)"


def calculate_conversion_by_tier(data: pd.DataFrame) -> pd.DataFrame:
    """Average conversion rate (closed / num_leads) grouped by budget tier.

    `data` needs `ad_budget`, `closed`, and `num_leads` columns. Returns one
    row per tier (in `TIER_ORDER`) with `avg_conversion_rate` and
    `n_observations`.
    """
    conversion_rate = data["closed"] / data["num_leads"]
    budget_tier = data["ad_budget"].apply(assign_budget_tier)
    grouped = (
        pd.DataFrame({"conversion_rate": conversion_rate, "budget_tier": budget_tier})
        .groupby("budget_tier")["conversion_rate"]
        .agg(avg_conversion_rate="mean", n_observations="count")
        .reindex(TIER_ORDER)
    )
    # A tier with zero matching rows is a real "0", not missing/unknown data -
    # reindex() leaves it NaN, which the API layer would otherwise render
    # identically to a genuinely unknown value.
    grouped["n_observations"] = grouped["n_observations"].fillna(0).astype(int)
    return grouped.reset_index().rename(columns={"budget_tier": "Budget Tier"})
