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

from followup_metrics import (
    analyze_sales_calls,
    calculate_dropout_rates,
    calculate_followup_statistics,
    generate_business_summary,
)

sns.set_theme(style="whitegrid")

DATA_DIR = Path(__file__).parent / "data"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_funnel_data.csv"

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
funnel_stats = calculate_followup_statistics(df)
funnel_stats

# %% [markdown]
# ## Dropout Rate (Step 2)
#
# `Drop Count = previous stage - current stage`,
# `Drop Rate (%) = Drop Count / previous stage * 100`. The baseline row has no
# previous stage, so its drop is left undefined (`NaN`).


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
call_stats = analyze_sales_calls(df)
print(f"Average calls to close a deal:      {call_stats['avg_calls_to_closed']:.2f}")
print(f"Average calls before giving up:     {call_stats['avg_calls_to_not_closed']:.2f}")
print(f"Difference (closed - not closed):   {call_stats['difference']:+.2f}")

# %% [markdown]
# ## Business Summary (Steps 6-7)


# %%
summary = generate_business_summary(dropout_stats, call_stats)
print("CONCLUSION:")
print(summary["conclusion"])
print()
print("RECOMMENDATION:")
print(summary["recommendation"])
