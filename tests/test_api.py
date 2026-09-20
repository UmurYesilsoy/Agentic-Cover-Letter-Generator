import os

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-tests")
os.environ.setdefault("APP_API_KEY", "test-secret")

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}