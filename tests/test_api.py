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


AUTH = {"X-API-Key": "test-secret"}


def test_generate_rejects_too_short_job_ad():
    response = client.post("/generate", json={"job_ad": "too short"}, headers=AUTH)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("job_ad" in str(error.get("loc")) for error in detail)
    assert not any("input" in error or "ctx" in error for error in detail)


def test_generate_rejects_too_short_cv():
    response = client.post("/generate", json={"cv": "too short"}, headers=AUTH)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("cv" in str(error.get("loc")) for error in detail)


def test_generate_rejects_too_short_past_letter():
    response = client.post(
        "/generate",
        json={"past_letters": ["way too short to be a real letter"]},
        headers=AUTH,
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("past_letters" in str(error.get("loc")) for error in detail)


def test_generate_rejects_both_job_ad_and_job_ad_url():
    long_enough_job_ad = "word " * 200  # passes the length check by itself
    response = client.post(
        "/generate",
        json={"job_ad": long_enough_job_ad, "job_ad_url": "https://example.com/job"},
        headers=AUTH,
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("not both" in str(error.get("msg", "")) for error in detail)