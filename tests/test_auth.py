from fastapi.testclient import TestClient

from backend.auth import get_current_user
from backend.main import app

client = TestClient(app)


def test_customers_requires_auth():
    app.dependency_overrides.pop(get_current_user, None)
    try:
        response = client.get("/customers")
        assert response.status_code == 401
    finally:
        app.dependency_overrides[get_current_user] = lambda: {"id": "test-user"}


def test_statistics_requires_auth():
    app.dependency_overrides.pop(get_current_user, None)
    try:
        response = client.get("/statistics")
        assert response.status_code == 401
    finally:
        app.dependency_overrides[get_current_user] = lambda: {"id": "test-user"}


def test_login_page_served():
    response = client.get("/login")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_dashboard_page_served():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
