from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import predictions
from backend.auth import get_current_user
from backend.supabase_client import get_supabase_admin_client
from supabase import Client

app = FastAPI(title="FunnelIQ")
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.include_router(predictions.router)


@app.get("/")
def root():
    return {"message": "FunnelIQ backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/login")
def login_page():
    return FileResponse("frontend/login.html")


@app.get("/dashboard")
def dashboard_page():
    return FileResponse("frontend/dashboard.html")


@app.get("/customers")
def get_customers(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    supabase: Client = Depends(get_supabase_admin_client),
    _user=Depends(get_current_user),
):
    response = (
        supabase.table("customers")
        .select("*")
        .range(offset, offset + limit - 1)
        .execute()
    )
    return {"limit": limit, "offset": offset, "count": len(response.data), "customers": response.data}


def _fetch_all(supabase: Client, columns: str, page_size: int = 1000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        page = (
            supabase.table("customers")
            .select(columns)
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


@app.get("/statistics")
def get_statistics(
    supabase: Client = Depends(get_supabase_admin_client),
    _user=Depends(get_current_user),
):
    rows = _fetch_all(supabase, "num_leads,closed,referred,upsell,cumulative_profit,ltv_months")
    total = len(rows)

    def avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "total_customers": total,
        "conversion_rate": (sum(r["closed"] for r in rows) / sum(r["num_leads"] for r in rows)) if total else None,
        "referral_rate": (sum(1 for r in rows if r["referred"]) / total) if total else None,
        "upsell_rate": (sum(1 for r in rows if r["upsell"]) / total) if total else None,
        "avg_cumulative_profit": avg([r["cumulative_profit"] for r in rows if r["cumulative_profit"] is not None]),
        "avg_ltv_months": avg([r["ltv_months"] for r in rows if r["ltv_months"] is not None]),
    }
