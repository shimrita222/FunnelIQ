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
- **Supabase**: Postgres holds the dataset (`sql/schema.sql`, loaded via a repeatable script); Supabase Auth gates the app behind a login screen; Row Level Security restricts data reads to authenticated users.
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
- `GET /customers?limit=50&offset=0` — paginated customer records from Supabase
- `GET /statistics` — aggregate metrics (conversion rate, referral rate, upsell rate, avg profit/LTV) computed from Supabase

## Status

FastAPI backend live on Railway, reading from a real Supabase Postgres
database (`customers` table, RLS enabled). Data exploration, the six
analytical work packages, Supabase Auth, and the frontend dashboard are in
progress.
