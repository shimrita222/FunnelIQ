# %% [markdown]
# # Package 5 – Follow-up Analysis
#
# Business question: is the sales manager right that follow-up calls after the
# 3rd call are a waste of time?
#
# **Important limitation, stated up front:** this dataset has no column that
# attributes a specific closed deal to a specific follow-up stage (there is no
# `closed_after_followup_4`, etc.). So this analysis cannot say "N deals closed
# after the 4th call." What it *can* do — and does — is (a) trace the
# stage-by-stage lead-retention/dropout curve, and (b) compare `calls_to_closed`
# vs. `calls_to_not_closed` as an aggregate proxy for how much contact effort
# precedes a close vs. a give-up. Conclusions are built only from those two
# angles, not from an invented metric.
#
# Read-only analysis: `data/cleaned_funnel_data.csv` is never modified.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_DIR = Path(__file__).parent / "data"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_funnel_data.csv"

FOLLOWUP_COLS = ["followup_1", "followup_2", "followup_3", "followup_4", "followup_5"]
STAGE_LABELS = ["Leads Answered", "Follow-up 1", "Follow-up 2", "Follow-up 3", "Follow-up 4", "Follow-up 5"]

# %% [markdown]
# ## Load Data


# %%
def load_data(path: Path = CLEANED_CSV_PATH) -> pd.DataFrame:
    """Read the cleaned dataset. Read-only — never writes back to `path`."""
    return pd.read_csv(path)


# %%
df = load_data()
print(f"Shape: {df.shape}")

# %% [markdown]
# ## Follow-up Funnel (Step 1)
#
# Each row in the cleaned dataset is a cohort-level record (consistent with how
# Packages 1-4 already treat `num_leads`/`leads_answered`/`followup_*` as
# counts, not per-lead flags), so the funnel is built from **column sums**
# across the whole dataset — the total number of leads still active at each
# stage. `leads_answered` is used as the stage-0 baseline (verified: no row has
# `followup_1 > leads_answered`, confirming the funnel narrows monotonically
# from there), so it's included as its own row to make Follow-up 1's "drop
# from previous stage" traceable.


# %%
def calculate_followup_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Build the follow-up funnel table: total remaining leads at each stage.

    Returns a DataFrame with one row per stage (`Leads Answered` baseline plus
    `Follow-up 1`..`Follow-up 5`) and a `Remaining Leads` column (column sums).
    """
    remaining = [data["leads_answered"].sum(), *(data[col].sum() for col in FOLLOWUP_COLS)]
    return pd.DataFrame({"Stage": STAGE_LABELS, "Remaining Leads": remaining})


# %%
funnel_stats = calculate_followup_statistics(df)
funnel_stats

# %% [markdown]
# ## Dropout Rate (Step 2)
#
# `Drop Count = previous stage - current stage`,
# `Drop Rate (%) = Drop Count / previous stage * 100`. The baseline row has no
# previous stage, so its drop is left undefined (`NaN`).


# %%
def calculate_dropout_rates(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add Drop Count and Drop Rate (%) columns to a follow-up funnel table."""
    result = stats_df.copy()
    previous = result["Remaining Leads"].shift(1)
    result["Drop From Previous Stage"] = previous - result["Remaining Leads"]
    result["Drop Rate (%)"] = (result["Drop From Previous Stage"] / previous) * 100
    return result


# %%
dropout_stats = calculate_dropout_rates(funnel_stats)
dropout_stats

# %% [markdown]
# ## Visualizations (Step 3)


# %%
def create_visualizations(stats_df: pd.DataFrame) -> dict[str, plt.Figure]:
    """Build the three follow-up funnel charts; returns them for reuse (e.g. by
    a future dashboard) rather than only displaying them inline.
    """
    figures: dict[str, plt.Figure] = {}

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=stats_df, x="Stage", y="Remaining Leads", marker="o", ax=ax1)
    ax1.set_title("Lead Retention Across Follow-up Stages")
    ax1.tick_params(axis="x", rotation=30)
    fig1.tight_layout()
    figures["retention_line"] = fig1

    drop_only = stats_df.dropna(subset=["Drop Rate (%)"]) if "Drop Rate (%)" in stats_df else stats_df
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=drop_only, x="Stage", y="Drop Rate (%)", ax=ax2)
    ax2.set_title("Lead Drop-off Rate by Follow-up Stage")
    ax2.tick_params(axis="x", rotation=30)
    fig2.tight_layout()
    figures["dropoff_bar"] = fig2

    # No native funnel-chart type in matplotlib/seaborn — a horizontal bar
    # chart ordered by stage is the documented fallback for this shape.
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    funnel_order = stats_df.iloc[::-1]
    sns.barplot(data=funnel_order, y="Stage", x="Remaining Leads", orient="h", ax=ax3)
    ax3.set_title("Follow-up Funnel: Leads Remaining by Stage")
    fig3.tight_layout()
    figures["funnel"] = fig3

    return figures


# %%
figures = create_visualizations(dropout_stats)
for fig in figures.values():
    plt.show()

# %% [markdown]
# ## Late Follow-up Analysis (Step 4)
#
# Compares how many leads are still active after Follow-ups 3, 4, and 5
# against the total number of `closed` deals. Since deals can't be attributed
# to a specific stage, this only shows whether late-stage survivors are still
# a *plausible* pool for meaningful closes, not proof of causation.

# %%
total_closed = df["closed"].sum()
late_stage_remaining = dropout_stats.set_index("Stage").loc[
    ["Follow-up 3", "Follow-up 4", "Follow-up 5"], "Remaining Leads"
]

print(f"Total closed deals: {total_closed}")
for stage, remaining in late_stage_remaining.items():
    share = total_closed / remaining * 100
    print(f"{stage}: {remaining} leads remaining -> closed deals are {share:.1f}% of that pool")

# %% [markdown]
# ## Sales Call Analysis (Step 5)
#
# `calls_to_closed` / `calls_to_not_closed` are this dataset's actual measure
# of contact effort per outcome (unlike the follow-up-stage columns, these
# *are* already scoped to closed vs. not-closed leads), so they're the
# appropriate proxy for "how much follow-up effort precedes each outcome."


# %%
def analyze_sales_calls(data: pd.DataFrame) -> dict:
    """Compare average calls before closing vs. before giving up on a lead."""
    avg_calls_to_closed = data["calls_to_closed"].mean()
    avg_calls_to_not_closed = data["calls_to_not_closed"].mean()
    return {
        "avg_calls_to_closed": avg_calls_to_closed,
        "avg_calls_to_not_closed": avg_calls_to_not_closed,
        "difference": avg_calls_to_closed - avg_calls_to_not_closed,
    }


# %%
call_stats = analyze_sales_calls(df)
print(f"Average calls to close a deal:      {call_stats['avg_calls_to_closed']:.2f}")
print(f"Average calls before giving up:     {call_stats['avg_calls_to_not_closed']:.2f}")
print(f"Difference (closed - not closed):   {call_stats['difference']:+.2f}")

# %% [markdown]
# ## Business Summary (Steps 6-7)


# %%
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


# %%
summary = generate_business_summary(dropout_stats, call_stats)
print("CONCLUSION:")
print(summary["conclusion"])
print()
print("RECOMMENDATION:")
print(summary["recommendation"])
