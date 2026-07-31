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
| Analysis   | pandas / seaborn exploration & cleaning, gradient-boosting regression & classification | `exploration_and_cleaning.py`, `customer_lifetime_regression.py`, `customer_lifetime_cv_feature_importance.ipynb`, `customer_lifetime_business_analysis.ipynb`, `upsell_classification.py`, `upsell_classification_analysis.ipynb`, `super_customer_score.ipynb`, `super_customer_score_early_features.ipynb` |
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

Loads `data/cleaned_funnel_data.csv`, drops the three columns that would leak future
information (`ltv_months` itself, `cumulative_profit`, and `referred`), splits
80/20 (`random_state=0` for reproducibility), and trains three baseline regressors
(XGBoost, LightGBM, CatBoost) with default parameters — no scaling (unnecessary for
tree-based models), no cross-validation, no tuning; those come in the next notebook.
Evaluates each model on the held-out test set (RMSE, MAE, R²) and reports the
best performer by RMSE. Current run: all three models land around R² ≈ 0.93–0.94,
with CatBoost narrowly ahead (RMSE ≈ 3.07 months).

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
CV comparison (Mean RMSE ≈ 2.93, Mean R² ≈ 0.94) with low fold-to-fold variance,
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
| CatBoost | 2.93      | 0.21     | 0.944   | 0.009  |
| LightGBM | 2.97      | 0.20     | 0.942   | 0.009  |
| XGBoost  | 3.19      | 0.16     | 0.933   | 0.008  |

Final held-out test performance (CatBoost, selected model):

| Model    | RMSE | MAE  | R²   |
|----------|------|------|-------|
| CatBoost | 3.07 | 2.27 | 0.939 |

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

`cumulative_profit` and `referred` were excluded from the feature set.
`cumulative_profit` only exists after the customer relationship has played out, so
using it would leak the future into the prediction. `referred` typically only
occurs after a customer has already been active for a while, so it also isn't
available at prediction time. Both would have made the model look better in
testing while being unusable in production.

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
(`upsell_classification_analysis.ipynb`). **Package 4 is done**: the super
customer score, in two versions (`super_customer_score.ipynb` and the
early-features-only `super_customer_score_early_features.ipynb`). None of these
models are wired into the API/`models/` yet. Packages 5–6 and the rest of the
dashboard UI are in progress.
