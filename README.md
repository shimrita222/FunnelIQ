# FunnelIQ

FunnelIQ turns Northbound Media's raw marketing-funnel data into a deployed,
login-gated intelligence tool: predicting customer lifetime, upsell probability,
and a 0–100 "super customer" score, plus data-driven recommendations on
follow-up policy and ad budget allocation.

Live URL: https://funnelmarketingdata-production.up.railway.app

## Architecture

| Layer      | Tech                                   | Folder       |
|------------|-----------------------------------------|--------------|
| Backend    | Python, FastAPI                         | `backend/`   |
| Frontend   | HTML/JS dashboard + Supabase Auth login | `frontend/`  |
| Data       | Supabase Postgres                       | `sql/`, `data/` |
| Analysis   | pandas / seaborn exploration & cleaning, gradient-boosting regression & classification, funnel dropout analysis, budget-scenario simulation | `exploration_and_cleaning.py`, `customer_lifetime_regression.py`, `customer_lifetime_cv_feature_importance.ipynb`, `customer_lifetime_business_analysis.ipynb`, `upsell_classification.py`, `upsell_classification_analysis.ipynb`, `upsell_classification_v2.py`, `upsell_classification_v2_analysis.ipynb`, `super_customer_score.ipynb`, `super_customer_score_early_features.ipynb`, `follow_up_analysis.py`, `follow_up_policy_analysis.ipynb`, `budget_optimization.py`, `budget_optimization_analysis.ipynb` |
| Models     | XGBoost / LightGBM / CatBoost           | `models/`    |
| Docs       | Findings notes, project brief           | `docs/`      |
| Tests      | pytest                                  | `tests/`     |

- **GitHub**: source of truth + CI (`.github/workflows/ci.yml` runs lint and tests on every push).
- **Supabase**: Postgres holds the dataset (`sql/schema.sql`, loaded via a repeatable script); Supabase Auth gates the app behind a login screen (`frontend/login.html` + `backend/auth.py`); Row Level Security restricts data reads to authenticated users.
- **Railway**: hosts the FastAPI service on a public URL, redeploying on every push to `main`.

## Local setup

```bash
# install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync --all-groups

# configure environment
cp .env.example .env
# fill in SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY

# run the API locally
uv run uvicorn backend.main:app --reload
# -> http://127.0.0.1:8000/health

# lint and test
uv run ruff check .
uv run pytest
```

## Dataset

`data/funnel_marketing_data.csv` — one row per customer/campaign record from
Northbound Media's funnel (ad spend, leads, follow-up outcomes, closed deals,
lifetime value, upsells, referrals). See `docs/` for the full project brief.

The CSV is the source file, but the running app never reads it directly — it's
loaded once into Supabase Postgres and served from there.

## Analysis (Package 1 — Exploration & Cleaning)

`exploration_and_cleaning.py` is a `# %%` cell-marker script (runs cell-by-cell in
VS Code / Jupyter, or end-to-end as a plain script) that turns the raw CSV into a
clean dataset for the modeling packages that follow. No modeling, train/test
splitting, scaling, or encoding happens here — that's out of scope for this package.

```bash
uv run python exploration_and_cleaning.py
```

Steps performed, in order: load & inspect the raw data, assess data quality (missing
values, duplicates, dtype consistency), impute missing values (median for numeric
columns, mode for the one categorical column), drop duplicate rows, validate business
logic (lead-count sums, closed/not_closed vs. leads_answered, non-increasing
follow-up counts), analyze outliers in the continuous business variables (IQR method,
reported but not auto-removed), and run the required exploratory analysis
(correlation with `cumulative_profit`, `ad_budget` vs. `num_leads`, and conversion
rate by budget tier — the conversion-rate/budget-tier analysis is computed in a
local, throwaway frame and never attached to the saved dataset). The result is
saved to `data/cleaned_funnel_data.csv` **with the original 19 columns only** — no
Package-1-only derived columns — so every later package works from the same
generic, reusable dataset. The original `data/funnel_marketing_data.csv` is never
modified.

**Findings from the current run:** 33 missing values across 2 columns (imputed), 10
exact duplicate rows (removed), 0 business-logic violations, no outliers requiring
removal (they read as genuine business variance, not data errors). `ad_budget` and
`num_leads` are strongly correlated (Pearson r ≈ 0.98), but the leads-per-shekel ratio
drops sharply from the bottom to the top budget quartile — a clear diminishing-returns
signal Northbound should factor into budget-allocation decisions (see Package 6).

## Analysis (Package 2, part 1 — Customer Lifetime Prediction)

`customer_lifetime_regression.py` is the first stage of Package 2: predicting a
customer's lifetime in months (`ltv_months`). Same `# %%` cell-marker style as the
Package 1 script.

```bash
uv run python customer_lifetime_regression.py
```

Loads `data/cleaned_funnel_data.csv`, drops the four columns that would leak future
information (`ltv_months` itself, `cumulative_profit`, `referred`, and `upsell` —
the last added after a project-wide leakage audit, see the Data Leakage Decision
below), splits 80/20 (`random_state=0` for reproducibility), and trains three
baseline regressors (XGBoost, LightGBM, CatBoost) with default parameters — no
scaling (unnecessary for tree-based models), no cross-validation, no tuning; those
come in the next notebook. Evaluates each model on the held-out test set (RMSE,
MAE, R²) and reports the best performer by RMSE. Current run: all three models
land around R² ≈ 0.93–0.94, with CatBoost narrowly ahead (RMSE ≈ 3.09 months).

## Analysis (Package 2, part 2 — Cross-Validation & Feature Importance)

`customer_lifetime_cv_feature_importance.ipynb` is a real Jupyter notebook (unlike
the two `# %%`-style scripts above, kept as an actual `.ipynb` for portfolio
presentation). It imports the prepared train/test split and the already-fitted
models directly from `customer_lifetime_regression.py` — nothing is reloaded,
re-cleaned, or retrained from scratch.

Adds: 5-fold cross-validation on `X_train`/`y_train` only (so the Part-1 holdout set
stays untouched), reporting Mean/Std RMSE and R² per model; per-model feature
importance (top-10 tables, one horizontal bar chart per model, and a cross-model
comparison table); and a model-stability discussion. Current run: CatBoost leads the
CV comparison (Mean RMSE ≈ 2.95, Mean R² ≈ 0.94) with low fold-to-fold variance,
close to its Part-1 holdout numbers — a good sign of generalization rather than
overfitting. `calls_to_closed` and `closed` dominate XGBoost's and CatBoost's
importance rankings; LightGBM's default split-count importance surfaces a different
top feature (`leads_not_answered`), a reminder that the three models' importance
scores aren't on a directly comparable scale.

## Analysis (Package 2, part 3 — Business Analysis & Final Report)

`customer_lifetime_business_analysis.ipynb` does **no additional model training**.
It imports the already-fitted models, `X_test`/`y_test`, and the holdout
`comparison` table directly from `customer_lifetime_regression.py`, and reproduces
(rather than re-runs) the 5-fold CV table from
`customer_lifetime_cv_feature_importance.ipynb` as a labeled reference — re-running
`cross_validate` would itself mean re-fitting models, which this notebook is
explicitly not meant to do.

Covers: final model selection (CatBoost, based on CV), final held-out test
performance, an actual-vs-predicted scatter plot and a residual-distribution plot for
the selected model, a feature-importance business interpretation, the data-leakage
rationale written for a non-technical audience, and evidence-based business
recommendations — see the "Customer Lifetime Prediction" summary below, copied
directly from that notebook's final report section.

## Customer Lifetime Prediction

### Objective

Predict how many months a new customer will stay (`ltv_months`).

### Models

- XGBoost
- LightGBM
- CatBoost

### Evaluation

5-fold cross-validation (on the training split only):

| Model    | Mean RMSE | Std RMSE | Mean R² | Std R² |
|----------|-----------|----------|---------|--------|
| CatBoost | 2.95      | 0.21     | 0.943   | 0.009  |
| LightGBM | 2.99      | 0.20     | 0.942   | 0.009  |
| XGBoost  | 3.24      | 0.17     | 0.932   | 0.008  |

Final held-out test performance (CatBoost, selected model):

| Model    | RMSE | MAE  | R²   |
|----------|------|------|-------|
| CatBoost | 3.09 | 2.30 | 0.938 |

### Key Findings

- **Strongest predictor:** `calls_to_closed` — the number of calls before a deal
  closes is the single most important feature across models, and it correlates
  *negatively* with lifetime (r ≈ −0.66): customers who take longer to close tend
  to stay for **shorter**, not longer.
- **Model agreement:** XGBoost and CatBoost agree closely (both gain-based
  importance); LightGBM's default split-count importance surfaces funnel-volume
  features instead — a measurement-scale difference, not evidence of a different
  underlying relationship.
- **Business implication:** a fast close is an early positive signal for customer
  lifetime, not just sales efficiency — accounts that need many calls to close are
  the group most likely to leave sooner, and are the ones retention efforts should
  prioritize.

### Data Leakage Decision

`cumulative_profit`, `referred`, and `upsell` were excluded from the feature
set. `cumulative_profit` only exists after the customer relationship has
played out, so using it would leak the future into the prediction.
`referred` typically only occurs after a customer has already been active
for a while, so it also isn't available at prediction time. `upsell` was
added later, after a project-wide leakage audit found it kept as a feature
here despite being a purchase event that can happen at any point during the
relationship with no guarantee it resolves before `ltv_months` is known —
the same category of issue already fixed in Package 3 v2 and Package 4 v2.
Its actual impact turned out negligible (~0.4% importance before removal),
so this was a same-file fix rather than a parallel v2. All three would have
made the model look better in testing while being unusable in production.

## Analysis (Package 3 — Upsell Classification)

`upsell_classification.py` predicts which customers are most likely to buy
additional services (`upsell`), as reusable functions (`prepare_data`,
`train_models`, `cross_validate_models`, `evaluate_models`,
`get_feature_importance`, `apply_business_rule`/`evaluate_rule`) so a later
notebook — and eventually the API — can import from it directly.

```bash
uv run python upsell_classification.py
```

Drops `cumulative_profit`/`referred` as leakage (same reasoning as Package 2);
`ltv_months` and the rest of the funnel columns stay as features per this
package's brief. Stratified 80/20 split, three classifiers each with their own
class-imbalance handling (`scale_pos_weight` / `class_weight="balanced"` /
`auto_class_weights="Balanced"`), 5-fold stratified CV scored on
Accuracy/Precision/Recall/F1 (selecting by F1, not accuracy, given the ~58/42
imbalance), a final holdout evaluation with a confusion matrix, feature
importance, and a comparison against a simple `ltv_months`/`customer_acquisition_cost`
threshold business rule.

`upsell_classification_analysis.ipynb` adds no additional training — it imports
the already-fitted models and result tables directly from the script, and only
recreates the business-rule baseline in-notebook (using thresholds computed from
`X_train` only, not the whole dataset, for a fair train/test comparison). Adds
styled comparison tables, grouped bar charts, an annotated confusion-matrix
heatmap, and a business-recommendations write-up.

Current run: CatBoost wins by CV and holdout F1 (≈0.78 / ≈0.77) and clearly
outperforms the business-rule baseline (F1 0.77 vs. 0.39), mainly via much
higher recall (0.87 vs. 0.27) at similar precision. Feature importance is split
between `ltv_months` (~26%) and `purchased` (~22%), not one dominant feature.

## Upsell Classification

### Objective

Predict which customers are most likely to buy additional services (`upsell`).

### Models

- XGBoost (`scale_pos_weight`)
- LightGBM (`class_weight="balanced"`)
- CatBoost (`auto_class_weights="Balanced"`)

### Evaluation

5-fold stratified cross-validation (on the training split only):

| Model    | Accuracy | Precision | Recall | F1    |
|----------|----------|-----------|--------|-------|
| CatBoost | 0.797    | 0.711     | 0.872  | 0.783 |
| LightGBM | 0.781    | 0.695     | 0.853  | 0.766 |
| XGBoost  | 0.761    | 0.686     | 0.796  | 0.737 |

Final held-out test performance (CatBoost, selected model):

| Model    | Accuracy | Precision | Recall | F1    |
|----------|----------|-----------|--------|-------|
| CatBoost | 0.778    | 0.684     | 0.874  | 0.768 |

Confusion matrix (test set, CatBoost): 287 true negatives, 118 false positives,
37 false negatives, 256 true positives.

### Key Findings

- **Model selected by F1, not accuracy** — the ~58/42 class imbalance makes raw
  accuracy misleading; F1 balances catching real upsells against wasted outreach.
- **Feature importance is top-heavy but not single-feature:** `ltv_months`
  (~26%) and `purchased` (~22%) together carry about half the model's decisions.
- **The ML model substantially beats a simple business rule** (`ltv_months` and
  `customer_acquisition_cost` thresholds, computed from the train set): F1 0.77
  vs. 0.39, driven almost entirely by much higher recall (0.87 vs. 0.27).

### Data Leakage Decision

`cumulative_profit` and `referred` were excluded, for the same reason as the
LTV regression task: both are outcomes that only exist after the customer
relationship has played out, so they wouldn't be available at the moment an
upsell prediction is actually needed.

## Analysis (Package 3, v2 — Upsell Classification without ltv_months)

A project-wide audit found that `upsell_classification.py` (v1, above) kept
`ltv_months` as a feature, even though it's only fully known once a customer
relationship has ended — the same category of late-relationship information
Package 4 identified and removed in its own v1→v2 fix. `ltv_months` was v1's
#1 feature at ~26% importance. `upsell_classification_v2.py` removes it
(`LEAKAGE_COLS = ["cumulative_profit", "referred", "ltv_months"]`) and is
otherwise an exact structural mirror of v1, for a fair comparison; v1 is left
untouched as the historical record of what was built to the original brief.

```bash
uv run python upsell_classification_v2.py
```

`upsell_classification_v2_analysis.ipynb` adds no recomputation and includes
a v1-vs-v2 comparison section, reproducing v1's verified numbers as a
reference table (not recomputed). One additional change was required: v1's
business-rule baseline used `ltv_months > median`, which is no longer a
feature in v2 — it's replaced with `purchased == 1` (a genuinely
early-available signal in the same spirit), alongside the unchanged
`customer_acquisition_cost` threshold.

## Upsell Classification (v2 — without ltv_months)

### Objective

A project-wide audit found that `upsell_classification.py` (v1) kept
`ltv_months` as a feature, even though it's only fully known once a
customer relationship has ended — the same category of late-relationship
information Package 4 identified and removed in its own v1→v2 fix.
`ltv_months` was v1's #1 feature at ~26% importance. This v2 removes it
and is otherwise an exact structural mirror of v1, for a fair comparison.

### Evaluation

| Version | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| v1 (with ltv_months) | 0.778 | 0.684 | 0.874 | 0.768 |
| v2 (without ltv_months) | 0.769 | 0.677 | 0.860 | 0.758 |

CatBoost remains the best model in both versions.

### Key Findings

- The performance cost of removing `ltv_months` is small (~1 F1 point) —
  far smaller than Package 4's ~4-point v1→v2 drop.
- `purchased` (already a legitimate feature) absorbs most of the lost
  signal, rising from ~22% to ~35% importance.
- The business-rule baseline had to change too: v1's `ltv_months`-based
  rule isn't usable in v2, so a `purchased`-based rule substitutes — and
  performs even more weakly than v1's rule did, reinforcing that the ML
  model is the right tool here regardless of feature-set version.

### Recommendation

Use v2 for any production upsell-scoring use case. v1's slightly higher
numbers were partly resting on a feature that wouldn't be reliably
available at real prediction time; v2's small performance cost buys a
model that's actually deployable.

## Analysis (Package 4 — Super Customer Score)

Two notebooks, both predicting `referred` (a customer who refers others) as a
0–100 "super customer" score via a hyperparameter-tuned `CatBoostClassifier`
(`GridSearchCV` over depth/learning_rate/iterations, `auto_class_weights="Balanced"`,
5-fold stratified CV scored on F1).

**`super_customer_score.ipynb` (v1)** follows the package brief literally,
keeping every remaining column (including `upsell`, `ltv_months`, `purchased`)
as a feature, dropping only `cumulative_profit`. Test performance: Accuracy
0.824, Precision 0.803, Recall 0.723, F1 0.761, **ROC-AUC 0.867**. Its own
feature-importance analysis surfaced an important caveat: `upsell` (~36%),
`ltv_months` (~30%), and `purchased` (~20%) together carry ~86% of the model's
decision weight — the same category of late-relationship information Packages 2
and 3 excluded as leakage for their own targets. That makes v1 much better at
describing an already-mature customer's referral likelihood than at scoring a
genuinely new customer on day one.

**`super_customer_score_early_features.ipynb` (v2)** restricts the feature set
to what's actually known at or near acquisition time (drops `upsell`,
`ltv_months`, and `cumulative_profit`, keeps `purchased` and the rest of the
funnel columns). Test performance: Accuracy 0.755, Precision 0.647, Recall
0.812, F1 0.720, ROC-AUC 0.817 — a real but modest cost versus v1, with Recall
actually *improving*. Feature importance is still concentrated in two features
(`purchased` ~46%, `calls_to_closed` ~40%), but this time both are genuinely
available early. v2's predicted probabilities are also far less extreme than
v1's (only 14 of 698 test customers land in the "High" tier vs. 225 in v1), so
v1's fixed 40/75 tier thresholds don't transfer directly — v2's should be
recalibrated against its own score distribution before use.

Both notebooks implement a reusable `predict_super_customer_score(customer_df,
model)` function (probability × 100, rounded, with a Low/Medium/High tier),
ready to be served by a future FunnelIQ API endpoint — v2's is the version
actually usable at the true start of the funnel.

## Analysis (Package 5 — Follow-up Policy Analysis)

`follow_up_analysis.py` is a reusable module (`load_data`,
`calculate_followup_statistics`, `calculate_dropout_rates`,
`create_visualizations`, `analyze_sales_calls`, `generate_business_summary`) —
not a predictive model, an EDA/funnel investigation answering one business
question: is the sales manager right that follow-up calls after the 3rd call
are a waste of time? The dataset has no column linking a specific closed deal
to a specific follow-up stage, so the analysis relies only on (a) the
stage-by-stage lead-retention/dropout curve and (b) `calls_to_closed` vs.
`calls_to_not_closed` as an aggregate proxy for contact effort — it does not
invent a per-stage close count.

```bash
uv run python follow_up_analysis.py
```

`follow_up_policy_analysis.ipynb` adds no recomputation — it imports every
result directly from the script and adds richer charts and business narrative.

See the "Follow-up Analysis" summary below, copied directly from that
notebook's final report section.

## Follow-up Analysis

### Objective

Northbound's sales manager claims follow-up calls after the 3rd call are a
waste of time. This analysis checks that claim against the funnel data —
the dataset has no column linking a specific closed deal to a specific
follow-up stage, so the check relies on the stage-by-stage lead-retention
curve and the calls-to-closed vs. calls-to-not-closed comparison, not on an
invented per-stage close count.

### Methodology

Dropout rates were calculated as `(previous stage - current stage) /
previous stage`, using `leads_answered` as the stage-0 baseline and summing
`followup_1`–`followup_5` across the dataset (each row is a cohort-level
record, consistent with how earlier packages treat these columns). Call
effort was compared via the mean of `calls_to_closed` vs.
`calls_to_not_closed`.

### Key Findings

- Drop-off is not smoothly declining: Follow-up 4 has the **lowest**
  drop-off of any stage (10.4%), while Follow-up 5 has the **highest**
  (29.2%) — a clear anomaly worth investigating on its own.
- Leads remaining at Follow-up 3/4/5 (46,323 / 41,517 / 29,384) stay far
  larger than the total closed-deal count (10,557) throughout, so the
  funnel is still actively working well past the 3rd call.
- Closed deals average fewer calls (3.52) than abandoned deals (3.93) — a
  real but modest signal that dragging a deal out is mildly associated
  with a lower chance of closing.

### Business Insight

Additional follow-ups do create measurable value past the 3rd call —
Follow-up 4 is the strongest-retaining stage in the entire funnel. The one
genuine weak point is Follow-up 5 specifically, not "anything after call 3."

### Recommendation

Keep the full 5-call sequence; do not cut off after the 3rd call. Instead,
investigate and address the Follow-up 5 drop-off spike directly (script,
timing, or lead-fatigue review), and use rising call count past ~4 calls as
a soft prioritization signal rather than a hard policy cutoff.

## Analysis (Package 6 — Budget Optimization Challenge)

`budget_optimization.py` is a reusable module (`load_data`,
`prepare_features`, `train_models`, `compare_models`, `cross_validate_models`,
`calculate_feature_importance`, `simulate_budget_scenarios`,
`generate_business_report`) that tests whether reallocating Northbound's
₪50,000/month ad budget could increase expected profit. The dataset has no
campaign identifier or channel column, so no campaign-level data is invented
— budget scenarios are simulated by varying `ad_budget` for real, sampled
customer profiles, holding every other feature fixed.

```bash
uv run python budget_optimization.py
```

Drops `ltv_months`, `upsell`, and `referred` as leakage (outcomes that only
resolve after long-term customer behavior is known), leaving 15 features to
predict `cumulative_profit`. Trains XGBoost/LightGBM/CatBoost with default
parameters, evaluates with an 80/20 split plus 5-fold CV, and simulates 4
budget scenarios across `N = round(₪50,000 / mean(ad_budget)) = 11` sampled
"campaign slot" profiles.

`budget_optimization_analysis.ipynb` adds no recomputation — it imports every
result directly from the script and adds charts and business narrative.

See the "Budget Optimization Challenge" summary below, copied directly from
that notebook's final report section.

## Budget Optimization Challenge

### Objective

Northbound spends ₪50,000/month, split equally across campaigns. This
package tests whether a different allocation of the same ₪50,000 would
produce more expected profit than the current equal split.

### Modeling Approach

- **Target:** `cumulative_profit`.
- **Leakage prevention:** `ltv_months`, `upsell`, and `referred` are dropped
  from the feature set — all are outcomes that only resolve after long-term
  customer behavior is known. 15 features remain.
- **Models:** XGBoost, LightGBM, and CatBoost regressors (default
  parameters), evaluated with an 80/20 split and 5-fold cross-validation on
  the training set only.

### Evaluation

LightGBM performed best (holdout RMSE ≈ 5,668, R² ≈ 0.75; 5-fold CV Mean
RMSE ≈ 5,356, Mean R² ≈ 0.76), narrowly ahead of CatBoost and clearly ahead
of XGBoost (R² ≈ 0.61).

### Simulation

The dataset has no campaign identifier or channel column, so no
campaign-level data was invented. Instead, `N = 11` real customer profiles
were sampled from the test set to stand in for concurrent "campaign slots"
(₪50,000 ÷ the dataset's average `ad_budget` ≈ 11), and four scenarios were
simulated by varying only each profile's `ad_budget` (all other features
held fixed): (1) equal allocation of the current ₪50,000, (2) equal
allocation of a +20% budget, (3) equal allocation of a −20% budget, and (4)
a profit-weighted allocation favoring profiles the model ranked as more
profitable at equal spend.

### Key Findings

- `ad_budget` ranks **last** of all 15 features in importance — once lead
  volume and follow-up execution are known, raw ad spend adds almost no
  further predictive signal for profit.
- A 20% budget increase produced **zero** additional predicted profit; a
  20% decrease produced only a small (~1%) predicted decline. Increasing
  budget does not straightforwardly increase profit in this model.
- The profit-weighted reallocation (Scenario 4) **underperformed** simple
  equal allocation — reshuffling a low-leverage variable didn't help.
- The strongest profit drivers are `num_leads`, `leads_answered`, and
  `leads_not_answered`, followed by follow-up-stage engagement and
  `customer_acquisition_cost`.

### Recommendation

Do not increase ad spend expecting more profit on its own. Redirect focus
toward increasing lead volume and improving follow-up execution — the
actual drivers this model identifies — rather than the size of the ad
budget itself. Treat any real budget change as a hypothesis to validate
with a controlled pilot, not a conclusion to roll out directly from this
simulation.

## Database (Supabase)

- `sql/schema.sql` — defines the `customers` table and enables Row Level
  Security with a `select` policy restricted to authenticated users.
- `scripts/load_csv_to_supabase.py` — repeatable loader: reads the CSV,
  clears the table, and re-inserts a fresh copy via the service-role client.
  Re-run any time the CSV changes: `uv run python -m scripts.load_csv_to_supabase`.

## API

- `GET /health` — health check
- `GET /login`, `GET /dashboard` — the frontend pages (see Authentication below)
- `GET /customers?limit=50&offset=0` — paginated customer records from Supabase (auth required)
- `GET /statistics` — aggregate metrics (conversion rate, referral rate, upsell rate, avg profit/LTV) computed from Supabase (auth required)

## Authentication

FunnelIQ is an internal tool: `/customers` and `/statistics` require a valid
Supabase session and return `401` without one.

- **Accounts are admin-provisioned, not self-signup.** Create a teammate's
  login with `uv run python -m scripts.create_user <email> <password>`.
- **`frontend/login.html`** — email/password sign-in via `supabase-js`, using
  the public anon key (`frontend/config.js` — safe to commit, it's designed
  to be public and is scoped by RLS). On success, redirects to `/dashboard`.
- **`frontend/dashboard.html`** — reads the Supabase session client-side; no
  session redirects back to `/login`. With a session, it calls `/customers`
  and `/statistics` with `Authorization: Bearer <access_token>`. "Sign out"
  clears the session and redirects to `/login`.
- **Backend enforcement** (`backend/auth.py`): every protected endpoint
  depends on `get_current_user`, which validates the bearer token against
  Supabase Auth — so the API itself is gated, not just the page.
- **Key split**: the anon key is the only Supabase credential in `frontend/`;
  the service-role key (`backend/supabase_client.py`) never leaves the server.

## Status

FastAPI backend live on Railway, reading from a real Supabase Postgres
database (`customers` table, RLS enabled) behind Supabase Auth. Package 1
(exploration & cleaning) is done — see `exploration_and_cleaning.py` and
`data/cleaned_funnel_data.csv`. **Package 2 is fully done**: baseline regression
(`customer_lifetime_regression.py`), cross-validation and feature importance
(`customer_lifetime_cv_feature_importance.ipynb`), and the business analysis/final
report (`customer_lifetime_business_analysis.ipynb`). **Package 3 is done**:
upsell classification (`upsell_classification.py`) plus its analysis notebook
(`upsell_classification_analysis.ipynb`), plus a **v2** without `ltv_months`
(`upsell_classification_v2.py` / `upsell_classification_v2_analysis.ipynb`) —
found via a project-wide leakage audit; use v2 for production. **Package 4 is done**: the super
customer score, in two versions (`super_customer_score.ipynb` and the
early-features-only `super_customer_score_early_features.ipynb`). **Package 5
is done**: the follow-up policy analysis (`follow_up_analysis.py` plus
`follow_up_policy_analysis.ipynb`). **Package 6 is done**: the budget
optimization challenge (`budget_optimization.py` plus
`budget_optimization_analysis.ipynb`). **All six analytical packages from the
project brief are now complete.** None of the Package 2–4/6 models are wired
into the API/`models/` yet, and the dashboard UI is still in progress.
