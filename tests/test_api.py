from fastapi.testclient import TestClient

from backend.auth import get_current_user
from backend.main import app
from backend.supabase_client import get_supabase_admin_client

FAKE_ROWS = [
    {
        "num_leads": 10,
        "closed": 2,
        "referred": True,
        "upsell": True,
        "cumulative_profit": 1000.0,
        "ltv_months": 12.0,
        "customer_acquisition_cost": 500.0,
    },
    {
        "num_leads": 20,
        "closed": 4,
        "referred": False,
        "upsell": False,
        "cumulative_profit": None,
        "ltv_months": None,
        "customer_acquisition_cost": 300.0,
    },
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
    return FakeSupabaseClient(FAKE_ROWS)


app.dependency_overrides[get_supabase_admin_client] = override_supabase
app.dependency_overrides[get_current_user] = lambda: {"id": "test-user"}
client = TestClient(app)


def test_get_customers():
    response = client.get("/customers")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["customers"] == FAKE_ROWS


def test_get_statistics():
    response = client.get("/statistics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_customers"] == 2
    assert body["conversion_rate"] == 6 / 30
    assert body["referral_rate"] == 0.5
    assert body["upsell_rate"] == 0.5
    assert body["avg_cumulative_profit"] == 1000.0
    assert body["avg_ltv_months"] == 12.0
    assert body["avg_customer_acquisition_cost"] == 400.0
