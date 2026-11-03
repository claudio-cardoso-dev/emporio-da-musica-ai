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


def test_cep_lookup_endpoint(api_client: TestClient) -> None:
    """Verify GET /api/cep/{cep} resolves address data via ViaCEP service."""
    res = api_client.get("/api/cep/79202333")
    assert res.status_code == 200
    data = res.json()
    assert data["found"] is True
    assert data["city"] == "Campo Grande"
    assert data["state"] == "MS"


def test_customer_register_and_address_endpoints(api_client: TestClient) -> None:
    """Verify POST /api/customer/register and POST /api/customer/address endpoints."""
    # 1. Register new customer
    payload_reg = {
        "name": "Mariana Souza Ramos",
        "phone": "(67) 99876-5432",
        "email": "mariana.ramos@exemplo.com",
        "cep": "79202-333",
        "address": "Rua 14 de Maio, 100",
        "city": "Campo Grande",
    }
    res_reg = api_client.post("/api/customer/register", json=payload_reg)
    assert res_reg.status_code == 200
    data_reg = res_reg.json()
    assert data_reg["success"] is True
    assert data_reg["name"] == "Mariana Souza Ramos"

    # 2. Update address
    payload_addr = {
        "customer_identifier": "mariana.ramos@exemplo.com",
        "address": "Avenida Afonso Pena, 200",
        "cep": "79002-000",
        "city": "Campo Grande",
    }
    res_addr = api_client.post("/api/customer/address", json=payload_addr)
    assert res_addr.status_code == 200
    data_addr = res_addr.json()
    assert data_addr["success"] is True
    assert data_addr["cep"] == "79002-000"


def test_web_ui_includes_customer_modal_and_cep_lookup(api_client: TestClient) -> None:
    """Verify Web UI contains the inline chat customer registration and Busca CEP elements, with static header button removed."""
    res = api_client.get("/")
    assert res.status_code == 200
    html = res.text
    # Ensure static header button 'Cadastro / CEP' is removed
    assert 'title="Cadastrar cliente ou atualizar endereço"' not in html
    # Ensure inline chat registration and address forms are present
    assert "chatRegisterForm" in html
    assert "chatAddressForm" in html
    assert "triggerCepSearch" in html


def test_web_ui_cart_button_removed_and_chat_cart_integrated(api_client: TestClient) -> None:
    """Verify top header cart button is removed and inline chat cart inspection logic is present."""
    res = api_client.get("/")
    assert res.status_code == 200
    html = res.text
    # Top header cart button removed
    assert "btn-cart" not in html
    # Inline chat cart inspection and management
    assert "showCartInChat" in html
    assert "isCartQuery" in html
    assert "esvaziar carrinho" in html


def test_pix_qrcode_endpoint_and_chat_simulation(api_client: TestClient) -> None:
    """Verify that simulated PIX QR Code SVG endpoint works and UI contains QR box rendering."""
    # 1. Test SVG QR Code endpoint
    res = api_client.get("/api/pix-qrcode?amount=348.81&name=Claudio")
    assert res.status_code == 200
    assert "image/svg+xml" in res.headers["content-type"]
    svg_text = res.text
    assert "<svg" in svg_text
    assert "PAGAMENTO INSTANTÂNEO PIX" in svg_text
    assert "R$ 348.81" in svg_text

    # 2. Test Web UI includes QR code container styling
    res_ui = api_client.get("/")
    assert res_ui.status_code == 200
    assert "qr-code-box" in res_ui.text

