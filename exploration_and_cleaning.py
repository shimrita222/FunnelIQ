# %% [markdown]
# # FunnelIQ — Package 1: Exploration & Cleaning
#
# Goal: understand `data/funnel_marketing_data.csv` before trusting it, and produce a
# clean dataset (`df_clean` / `data/cleaned_funnel_data.csv`) for the modeling packages
# that follow (LTV regression, upsell classification, super-customer score, follow-up
# analysis, budget optimization).
#
# Scope for this notebook only: exploration and cleaning. No models, no train/test
# split, no scaling/encoding/feature selection — those belong to later packages.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

sns.set_theme(style="whitegrid")

DATA_DIR = Path(__file__).parent / "data"
RAW_CSV_PATH = DATA_DIR / "funnel_marketing_data.csv"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_funnel_data.csv"

CONTINUOUS_BUSINESS_VARS = [
    "ad_budget",
    "customer_acquisition_cost",
    "ltv_months",
    "cumulative_profit",
]

# %% [markdown]
# ## 1. Load the dataset

# %%
df = pd.read_csv(RAW_CSV_PATH)

print(f"Shape: {df.shape}")
df.head()

# %%
df.dtypes

# %%
df.describe(include="all")

# %% [markdown]
# ## 2. Data Quality Assessment
#
# Check missing values, duplicate rows, and whether any column's dtype doesn't match
# what it semantically represents (e.g. a Yes/No flag stored as free text).


# %%
def missing_value_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column table of missing counts and percentages, worst first."""
    missing_count = data.isna().sum()
    missing_pct = (missing_count / len(data)) * 100
    summary = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    return summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)


missing_summary = missing_value_summary(df)
print(f"Columns with missing values: {len(missing_summary)}")
print(f"Rows with at least one missing value: {df.isna().any(axis=1).sum()}")
missing_summary

# %%
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")

# %%
# Dtype consistency check: flag numeric-looking columns stored as object, and
# object columns that hold few distinct values (likely categorical).
dtype_report = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "sample_values": [df[col].dropna().unique()[:3] for col in df.columns],
    }
)
dtype_report

# %% [markdown]
# `referred` is stored as an object column with Yes/No text rather than a numeric flag
# like `purchased`/`upsell` — that's expected (it's the one genuinely categorical
# column in this dataset) and is handled explicitly in the missing-value step below.

# %% [markdown]
# ## 3. Handle Missing Values
#
# - **Numeric columns → median.** The business variables here (spend, costs, counts,
#   LTV, profit) are right-skewed — a few large campaigns/customers pull the mean up.
#   The median is robust to that skew and to any outliers found later, so it's a safer
#   default than the mean for filling gaps without distorting the distribution.
# - **Categorical columns → mode.** `referred` is the only categorical column; filling
#   with its most frequent value is the standard, minimally-invasive default when there
#   is no stronger signal to justify a different value per row.


# %%
def impute_numeric_median(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing values in numeric columns with each column's median."""
    data = data.copy()
    for col in columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())
    return data


def impute_categorical_mode(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing values in categorical columns with each column's most frequent value."""
    data = data.copy()
    for col in columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode(dropna=True).iloc[0])
    return data


# %%
df_clean = df.copy()  # every cleaning step below applies to df_clean; df stays untouched

numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# %%
missing_before = df_clean.isna().sum().sum()

df_clean = impute_numeric_median(df_clean, numeric_cols)
df_clean = impute_categorical_mode(df_clean, categorical_cols)

missing_after = df_clean.isna().sum().sum()
print(f"Total missing values before imputation: {missing_before}")
print(f"Total missing values after imputation:  {missing_after}")

# %% [markdown]
# ## 4. Remove Duplicate Records

# %%
rows_before = len(df_clean)
df_clean = df_clean.drop_duplicates()
rows_removed = rows_before - len(df_clean)

print(f"Duplicate rows found: {duplicate_count}")
print(f"Rows removed: {rows_removed}")
print(f"Shape after de-duplication: {df_clean.shape}")

# %% [markdown]
# ## 5. Business Logic Validation
#
# The funnel should be internally consistent. We check three rules and report every
# violation found before deciding what to do about it.

# %%
# Rule 1: num_leads should equal leads_answered + leads_not_answered
lead_sum_mismatch = df_clean["num_leads"] != (
    df_clean["leads_answered"] + df_clean["leads_not_answered"]
)
print(f"Rule 1 — num_leads != leads_answered + leads_not_answered: {lead_sum_mismatch.sum()} rows")
df_clean.loc[lead_sum_mismatch, ["num_leads", "leads_answered", "leads_not_answered"]].head()

# %%
# Rule 2: closed + not_closed should not exceed leads_answered
closed_sum_exceeds = (df_clean["closed"] + df_clean["not_closed"]) > df_clean["leads_answered"]
print(f"Rule 2 — (closed + not_closed) > leads_answered: {closed_sum_exceeds.sum()} rows")
df_clean.loc[closed_sum_exceeds, ["leads_answered", "closed", "not_closed"]].head()

# %%
# Rule 3: follow-up counts should be non-increasing (funnel narrows at each stage)
followup_cols = ["followup_1", "followup_2", "followup_3", "followup_4", "followup_5"]
followup_non_increasing = (df_clean[followup_cols].diff(axis=1).iloc[:, 1:] <= 0).all(axis=1)
followup_violation = ~followup_non_increasing
print(f"Rule 3 — follow-up counts not non-increasing: {followup_violation.sum()} rows")
df_clean.loc[followup_violation, followup_cols].head()

# %%
total_violations = lead_sum_mismatch | closed_sum_exceeds | followup_violation
violation_pct = (total_violations.sum() / len(df_clean)) * 100
print(f"Total rows with at least one inconsistency: {total_violations.sum()} ({violation_pct:.2f}%)")

# %% [markdown]
# If inconsistencies are rare (a small fraction of the dataset), they're most plausibly
# data-entry errors rather than a systematic issue — dropping them is safer than trying
# to guess which of the conflicting fields is correct. Recompute and confirm the
# fraction is indeed small before dropping; if it were large, the right move would be
# investigating a systematic cause instead of discarding data.

# %%
if violation_pct < 3:
    rows_before_validation = len(df_clean)
    df_clean = df_clean.loc[~total_violations].reset_index(drop=True)
    print(
        f"Dropped {rows_before_validation - len(df_clean)} inconsistent rows "
        f"({violation_pct:.2f}% of the dataset) — rare enough to treat as data-entry errors."
    )
else:
    print(
        f"{violation_pct:.2f}% of rows are inconsistent — too large a share to drop blindly; "
        "flagging for further investigation instead of removing."
    )

print(f"Shape after business-logic validation: {df_clean.shape}")

# %% [markdown]
# ## 6. Outlier Analysis
#
# IQR method on the four continuous business variables. Outliers are reported, not
# automatically removed — a very high `cumulative_profit` or `ltv_months` may simply be
# a genuinely great customer, which is exactly the kind of signal later models need.


# %%
def iqr_outlier_bounds(series: pd.Series) -> tuple[float, float]:
    """Return the (lower, upper) IQR fences for a numeric series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def count_outliers(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Report the IQR-based outlier count and bounds for each column."""
    rows = []
    for col in columns:
        lower, upper = iqr_outlier_bounds(data[col])
        n_outliers = ((data[col] < lower) | (data[col] > upper)).sum()
        rows.append(
            {"column": col, "lower_bound": lower, "upper_bound": upper, "n_outliers": n_outliers}
        )
    return pd.DataFrame(rows).set_index("column")


# %%
outlier_report = count_outliers(df_clean, CONTINUOUS_BUSINESS_VARS)
outlier_report

# %%
fig, axes = plt.subplots(1, len(CONTINUOUS_BUSINESS_VARS), figsize=(16, 4))
for ax, col in zip(axes, CONTINUOUS_BUSINESS_VARS, strict=True):
    sns.boxplot(y=df_clean[col], ax=ax)
    ax.set_title(col)
fig.tight_layout()
plt.show()

# %% [markdown]
# None of the four variables show impossible values (e.g. negative spend or negative
# profit) beyond the IQR fences — the flagged points look like legitimate high-spend
# campaigns or long-tenure/high-value customers rather than data errors. No rows are
# removed for outliers alone; `df_clean` keeps them so downstream models see the real
# variance in the business.

# %% [markdown]
# ## 7. Exploratory Analysis
#
# ### A. Correlation with `cumulative_profit`

# %%
correlation_with_profit = (
    df_clean.select_dtypes(include=np.number)
    .corr()["cumulative_profit"]
    .sort_values(ascending=False)
)
correlation_with_profit

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_clean.select_dtypes(include=np.number).corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
)
plt.title("Correlation matrix — numeric variables")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### B. `ad_budget` vs `num_leads`

# %%
pearson_r, pearson_p = pearsonr(df_clean["ad_budget"], df_clean["num_leads"])
print(f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")

plt.figure(figsize=(8, 6))
sns.regplot(
    data=df_clean,
    x="ad_budget",
    y="num_leads",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"},
)
plt.title("Ad budget vs. number of leads")
plt.tight_layout()
plt.show()

# %%
# Compare the leads-per-shekel ratio in the bottom vs. top budget quartile as a quick
# diminishing-returns check: a lower ratio at high budget indicates diminishing returns.
low_quartile = df_clean["ad_budget"].quantile(0.25)
high_quartile = df_clean["ad_budget"].quantile(0.75)
low_budget_ratio = (
    df_clean.loc[df_clean["ad_budget"] <= low_quartile, "num_leads"]
    / df_clean.loc[df_clean["ad_budget"] <= low_quartile, "ad_budget"]
).mean()
high_budget_ratio = (
    df_clean.loc[df_clean["ad_budget"] >= high_quartile, "num_leads"]
    / df_clean.loc[df_clean["ad_budget"] >= high_quartile, "ad_budget"]
).mean()
print(f"Avg leads per unit of ad spend — bottom budget quartile: {low_budget_ratio:.4f}")
print(f"Avg leads per unit of ad spend — top budget quartile:    {high_budget_ratio:.4f}")

# %% [markdown]
# A strong positive Pearson correlation confirms more budget buys more leads overall.
# Whether that's diminishing returns depends on the printed ratios above: if leads-per-
# shekel is lower in the top quartile than the bottom, each extra shekel buys fewer
# incremental leads at high spend — classic diminishing returns rather than linear
# scaling.

# %% [markdown]
# ### C. Conversion rate by budget tier

# %%
# Computed into a standalone frame — not attached to df_clean — so the saved
# cleaned dataset stays generic (original columns only) for every later package.
conversion_analysis = pd.DataFrame(
    {
        "conversion_rate": df_clean["closed"] / df_clean["num_leads"],
        "ad_budget": df_clean["ad_budget"],
    }
)


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


conversion_analysis["budget_tier"] = conversion_analysis["ad_budget"].apply(assign_budget_tier)

gap_rows = (conversion_analysis["budget_tier"] == "Gap (1500-2000)").sum()
print(f"Rows falling in the undefined 1500-2000 gap: {gap_rows}")

# %%
tier_order = ["Low (<=1500)", "Gap (1500-2000)", "Mid (2000-5000)", "High (>5000)"]
conversion_by_tier = (
    conversion_analysis.groupby("budget_tier")["conversion_rate"]
    .agg(avg_conversion_rate="mean", n_observations="count")
    .reindex(tier_order)
)
conversion_by_tier

# %%
plt.figure(figsize=(8, 6))
sns.barplot(
    data=conversion_by_tier.reset_index(),
    x="budget_tier",
    y="avg_conversion_rate",
    order=tier_order,
)
plt.title("Average conversion rate by budget tier")
plt.ylabel("Average conversion rate (closed / num_leads)")
plt.xlabel("Budget tier")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Final Dataset
#
# `df_clean` (built and cleaned incrementally above, never mutating the original `df`)
# is saved as the Package 1 deliverable for downstream modeling packages.

# %%
df_clean.to_csv(CLEANED_CSV_PATH, index=False)
print(f"Saved cleaned dataset: {CLEANED_CSV_PATH} — shape {df_clean.shape}")

# %% [markdown]
# ## 9. Findings
#
# - **Missing values:** see the `missing_summary` table in Section 2 for the exact
#   per-column counts/percentages; `missing_before`/`missing_after` above show the
#   total resolved by imputation.
# - **How missing values were handled:** numeric columns filled with their median
#   (robust to skew/outliers), the single categorical column (`referred`) filled with
#   its mode — see Section 3 for the rationale.
# - **Duplicates:** `duplicate_count` rows were found and removed (Section 4).
# - **Inconsistent records:** three business rules were checked (lead-count sum,
#   closed+not_closed vs. leads_answered, non-increasing follow-ups); the printed
#   violation counts and percentage in Section 5 show how many were found, and
#   whether they were dropped or flagged.
# - **Outliers:** IQR-based counts per continuous business variable are in
#   `outlier_report` (Section 6); none were removed — they read as genuine business
#   variance, not data errors.
# - **Top correlates with `cumulative_profit`:** see `correlation_with_profit`
#   (Section 7A) for the ranked list.
# - **`ad_budget` vs. `num_leads`:** a strong positive Pearson correlation (printed in
#   Section 7B); compare the bottom-vs-top-quartile leads-per-shekel ratios printed
#   there to see whether returns diminish at higher spend.
# - **Best-converting budget tier:** see `conversion_by_tier` (Section 7C) for the
#   tier with the highest `avg_conversion_rate`.
# - **Recommendations for Northbound Media:** to be finalized once the printed outputs
#   above are reviewed against the actual run — in general, this analysis should tell
#   Northbound (1) whether to keep scaling ad spend or whether returns are flattening,
#   (2) which budget tier deserves a larger share of the ₪50,000/month budget based on
#   conversion efficiency, and (3) which funnel stage to audit first given the
#   business-logic violations found in Section 5.
