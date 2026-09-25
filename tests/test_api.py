import os

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-tests")
os.environ.setdefault("APP_API_KEY", "test-secret")

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "agentic-cover-letter-generator",
        "version": "1.0",
    }


def test_generate_rejects_wrong_api_key():
    response = client.post(
        "/generate",
        json={},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401