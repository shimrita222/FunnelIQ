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
database (`customers` table, RLS enabled) behind Supabase Auth. Data
exploration, the six analytical work packages, and the rest of the dashboard
UI are in progress.
