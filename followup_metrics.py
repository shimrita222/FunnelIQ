"""Pure follow-up funnel metrics shared by follow_up_analysis.py and the API.

Deliberately has no data-loading or training side effects — safe to import
in the FastAPI backend at startup without needing data/cleaned_funnel_data.csv
to exist (it doesn't, on Railway; data/ is gitignored and never deployed).
"""

import pandas as pd

FOLLOWUP_COLS = ["followup_1", "followup_2", "followup_3", "followup_4", "followup_5"]
STAGE_LABELS = ["Leads Answered", "Follow-up 1", "Follow-up 2", "Follow-up 3", "Follow-up 4", "Follow-up 5"]


def calculate_followup_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Build the follow-up funnel table: total remaining leads at each stage.

    Returns a DataFrame with one row per stage (`Leads Answered` baseline plus
    `Follow-up 1`..`Follow-up 5`) and a `Remaining Leads` column (column sums).
    """
    remaining = [data["leads_answered"].sum(), *(data[col].sum() for col in FOLLOWUP_COLS)]
    return pd.DataFrame({"Stage": STAGE_LABELS, "Remaining Leads": remaining})


def calculate_dropout_rates(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add Drop Count and Drop Rate (%) columns to a follow-up funnel table."""
    result = stats_df.copy()
    previous = result["Remaining Leads"].shift(1)
    result["Drop From Previous Stage"] = previous - result["Remaining Leads"]
    result["Drop Rate (%)"] = (result["Drop From Previous Stage"] / previous) * 100
    return result


def analyze_sales_calls(data: pd.DataFrame) -> dict:
    """Compare average calls before closing vs. before giving up on a lead."""
    avg_calls_to_closed = data["calls_to_closed"].mean()
    avg_calls_to_not_closed = data["calls_to_not_closed"].mean()
    return {
        "avg_calls_to_closed": avg_calls_to_closed,
        "avg_calls_to_not_closed": avg_calls_to_not_closed,
        "difference": avg_calls_to_closed - avg_calls_to_not_closed,
    }


def generate_business_summary(stats_df: pd.DataFrame, call_stats: dict) -> dict:
    """Synthesize an evidence-grounded answer + recommendations.

    Built entirely from the two data-supported angles available in this
    dataset (retention curve, calls-to-outcome comparison) — explicitly does
    not claim to attribute any specific close to a specific follow-up stage,
    since that information doesn't exist in the data.
    """
    late_stage = stats_df.set_index("Stage").loc[["Follow-up 3", "Follow-up 4", "Follow-up 5"]]
    stage_4_drop = late_stage.loc["Follow-up 4", "Drop Rate (%)"]
    stage_5_drop = late_stage.loc["Follow-up 5", "Drop Rate (%)"]
    calls_favor_closing = call_stats["avg_calls_to_closed"] > call_stats["avg_calls_to_not_closed"]

    conclusion = (
        f"Follow-up 4 and Follow-up 5 still show a {stage_4_drop:.1f}% and "
        f"{stage_5_drop:.1f}% drop-off respectively — leads are still being lost "
        f"(and by extension, still being actively worked) well past the 3rd call. "
        f"Separately, closed deals require {call_stats['avg_calls_to_closed']:.2f} calls "
        f"on average vs. {call_stats['avg_calls_to_not_closed']:.2f} for deals that were "
        f"abandoned — "
        + (
            "closed deals take MORE calls on average, meaning persistence past the "
            "early calls is associated with success, not wasted effort."
            if calls_favor_closing
            else "closed deals take FEWER calls on average, meaning deals that need "
            "many calls are less likely to close, supporting the case for cutting "
            "off effort earlier."
        )
        + " Note: the data cannot attribute a specific close to a specific "
        "follow-up stage, so this conclusion rests on the retention curve and "
        "the calls-to-outcome comparison, not on a direct causal count."
    )

    recommendation = (
        "Continue the full 5-stage follow-up sequence rather than stopping after "
        "the 3rd call, since later stages still meaningfully affect outcomes; "
        "consider using the calls-to-closed distribution to set a per-lead "
        "call budget instead of a blanket stage cutoff."
        if calls_favor_closing
        else "There is support for tightening the follow-up policy after the 3rd "
        "call, since additional calls do not appear associated with a higher "
        "chance of closing; consider reallocating that effort to fresher leads."
    )

    return {"conclusion": conclusion, "recommendation": recommendation}
