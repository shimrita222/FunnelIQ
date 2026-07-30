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
| Analysis   | pandas / seaborn exploration & cleaning, gradient-boosting regression | `exploration_and_cleaning.py`, `customer_lifetime_regression.py`, `customer_lifetime_cv_feature_importance.ipynb` |
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
`data/cleaned_funnel_data.csv`. Package 2 is done — baseline regression
(`customer_lifetime_regression.py`) plus cross-validation and feature importance
(`customer_lifetime_cv_feature_importance.ipynb`). Packages 3–6 and the rest of
the dashboard UI are in progress.
