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
    """Verify root endpoint serves the glassmorphism Web Chat UI with Product Carousel and Cart."""
    res = api_client.get("/")
    assert res.status_code == 200
    assert "Empório da Música" in res.text
    assert "buildProductCarousel" in res.text
    assert "Adicionar ao Carrinho" in res.text


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


def test_web_ui_includes_3d_flip_card_and_description_button(api_client: TestClient) -> None:
    """Verify that the embedded Web UI includes the 3D flip card, 'Descrição' button, and category flip back."""
    res = api_client.get("/")
    assert res.status_code == 200
    html = res.text
    # 3D Flip Card classes
    assert "card-inner" in html
    assert "card-front" in html
    assert "card-back" in html
    assert "is-flipped" in html
    assert "rotateY(180deg)" in html

    # Flip interaction buttons and functions
    assert "btn-desc-flip" in html
    assert "cat-tag-clickable" in html
    assert "flipCardToBack" in html
    assert "flipCardToFront" in html
    assert "Descrição" in html


def test_chat_endpoint_resilience_recovery(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that /api/chat recovers gracefully and returns HTTP 200 without raising HTTP 500 on unexpected errors."""
    from unittest.mock import MagicMock
    from src.api import app as api_module

    # Force process_message to raise an unexpected runtime error (e.g. LLM provider rate limit)
    def broken_process_message(*args, **kwargs):
        raise RuntimeError("Simulated external LLM provider downtime")

    monkeypatch.setattr(
        "src.services.chat_service.ChatService.process_message",
        broken_process_message,
    )

    res = api_client.post("/api/chat", json={"message": "Olá", "session_id": "test_resilience"})
    assert res.status_code == 200
    data = res.json()
    assert "oscilação" in data["reply"] or "demanda" in data["reply"] or "WhatsApp" in data["reply"]
    assert data["model_used"] == "api_error_recovery"
