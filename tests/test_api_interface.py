"""Tests for Stage 5: FastAPI endpoints, Web UI, Swagger docs, and session management."""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app


@pytest.fixture
def api_client(tmp_path: Path) -> TestClient:
    """Create a FastAPI TestClient wired to a temporary database."""
    db_file = tmp_path / "test_api.db"
    test_app = create_app(db_path=db_file)
    return TestClient(test_app)


def test_health_endpoint_reports_qwen_primary(api_client: TestClient) -> None:
    """Verify /api/health returns status ok and qwen/qwen3.8-27b as primary model."""
    res = api_client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["primary_model"] == "qwen/qwen3.8-27b"


def test_web_ui_html_served(api_client: TestClient) -> None:
    """Verify root endpoint serves the glassmorphism Web Chat UI."""
    res = api_client.get("/")
    assert res.status_code == 200
    assert "Empório da Música" in res.text


def test_chat_session_reset_and_failed_intents_endpoint(api_client: TestClient) -> None:
    """Verify low-confidence message via API logs failed intent, stores history, and resets cleanly."""
    chat_res = api_client.post(
        "/api/chat",
        json={"message": "???", "session_id": "api_sess_1"},
    )
    assert chat_res.status_code == 200
    payload = chat_res.json()
    assert payload["logged_failed_intent"] is True
    assert payload["session_id"] == "api_sess_1"

    hist_res = api_client.get("/api/session/api_sess_1/history")
    assert hist_res.status_code == 200
    assert len(hist_res.json()["history"]) == 2

    fail_res = api_client.get("/api/failed-intents")
    assert fail_res.status_code == 200
    assert fail_res.json()["count"] >= 1

    reset_res = api_client.post("/api/session/reset", json={"session_id": "api_sess_1"})
    assert reset_res.status_code == 200

    hist_after = api_client.get("/api/session/api_sess_1/history")
    assert len(hist_after.json()["history"]) == 0
