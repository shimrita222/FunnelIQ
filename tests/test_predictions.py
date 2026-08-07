import pytest
from fastapi.testclient import TestClient

from backend.auth import get_current_user
from backend.main import app
from backend.supabase_client import get_supabase_admin_client

SAMPLE_FEATURES = {
    "ad_budget": 5000.0,
    "num_leads": 100.0,
    "leads_answered": 80.0,
    "leads_not_answered": 20.0,
    "followup_1": 70.0,
    "followup_2": 60.0,
    "followup_3": 50.0,
    "followup_4": 40.0,
    "followup_5": 30.0,
    "not_closed": 15.0,
    "closed": 25.0,
    "calls_to_closed": 3.5,
    "calls_to_not_closed": 4.0,
    "customer_acquisition_cost": 200.0,
    "purchased": 1,
}

FAKE_CUSTOMER_ROWS = [
    {**SAMPLE_FEATURES, "ad_budget": 4000.0 + i * 100, "purchased": True} for i in range(15)
]


class FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_args, **_kwargs):
        return self

    def range(self, start, end):
        self._rows = self._rows[start : end + 1]
        return self

    def execute(self):
        return type("Response", (), {"data": self._rows})()


class FakeSupabaseClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return FakeQuery(self._rows)


def override_supabase():
    return FakeSupabaseClient(FAKE_CUSTOMER_ROWS)


client = TestClient(app)


@pytest.fixture(autouse=True)
def _overrides():
    # Shared `app` across test modules — scope overrides to this module's
    # tests only, then restore whatever test_api.py had configured.
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_supabase_admin_client] = override_supabase
    app.dependency_overrides[get_current_user] = lambda: {"id": "test-user"}
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


def test_predict_lifetime_requires_auth():
    app.dependency_overrides.pop(get_current_user, None)
    response = client.post("/predict/lifetime", json=SAMPLE_FEATURES)
    assert response.status_code == 401


def test_predict_lifetime():
    response = client.post("/predict/lifetime", json=SAMPLE_FEATURES)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predicted_ltv_months"], float)


def test_predict_upsell():
    response = client.post("/predict/upsell", json=SAMPLE_FEATURES)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["upsell_probability"] <= 1.0
    assert body["predicted_upsell"] in (0, 1)


def test_predict_super_customer():
    response = client.post("/predict/super-customer", json=SAMPLE_FEATURES)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 100.0
    assert body["tier"] in ("Low", "Medium", "High")


def test_budget_optimization():
    response = client.post("/budget-optimization", json={"monthly_budget": 50_000})
    assert response.status_code == 200
    body = response.json()
    assert body["n_campaigns"] > 0
    assert len(body["scenarios"]) == 4
    assert {s["Scenario"] for s in body["scenarios"]} == {
        "1. Equal Allocation (Current)",
        "2. Increase Budget +20%",
        "3. Decrease Budget -20%",
        "4. Profit-Weighted Allocation",
    }
    # FAKE_CUSTOMER_ROWS only has 15 rows, so campaign counts above the pool
    # size (25, 50) are skipped — only 1/5/10 remain.
    assert {s["n_campaigns"] for s in body["by_campaign_count"]} == {1, 5, 10}

    exec_summary = body["executive_summary"]
    assert exec_summary["recommended_strategy"]
    assert isinstance(exec_summary["expected_profit"], float)
    assert 0.0 <= exec_summary["confidence_score"] <= 1.0

    assert set(body["analysis"].keys()) == {"business_insight", "key_observation", "business_impact"}
    assert all(body["analysis"].values())

    assert body["recommendation"]
    assert len(body["why"]) > 0

    assert body["confidence"] in {"High", "Medium", "Low"}
    assert 0.0 <= body["confidence_score"] <= 1.0
    assert body["confidence_explanation"]

    assert len(body["top_drivers"]) > 0
    for driver in body["top_drivers"]:
        assert driver["feature"]
        assert isinstance(driver["importance"], float)
        assert driver["business_impact"]

    for group in (body["risks"], body["opportunities"]):
        assert "data_driven" in group
        assert "general" in group
        assert len(group["general"]) > 0


def test_followup_analysis():
    response = client.get("/followup-analysis")
    assert response.status_code == 200
    body = response.json()
    assert len(body["dropout_by_stage"]) == 6
    assert "avg_calls_to_closed" in body["call_stats"]
    assert body["conclusion"]
    assert body["recommendation"]


def test_conversion_by_budget_tier():
    response = client.get("/conversion-by-budget-tier")
    assert response.status_code == 200
    body = response.json()
    assert len(body["tiers"]) == 4
    assert {row["Budget Tier"] for row in body["tiers"]} == {
        "Low (<=1500)",
        "Gap (1500-2000)",
        "Mid (2000-5000)",
        "High (>5000)",
    }
