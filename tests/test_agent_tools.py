"""Tests for Stage 4: Agent tools, LLM model fallback order, multilingual detection, and guardrails."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.agents.assistant_agent import AssistantAgent, clean_thinking_tags
from src.agents.prompts import detect_user_language
from src.agents.tools import ToolExecutor
from src.db.ingestion import ingest_all_csvs
from src.services.chat_service import ChatService


@pytest.fixture
def agent_db(tmp_path: Path) -> Path:
    """Create and seed a temporary database for agent tool and guardrail tests."""
    db_file = tmp_path / "test_agent.db"
    ingest_all_csvs(db_path=db_file, force_reload=True)
    return db_file


def test_model_fallback_chain_order_qwen_primary(agent_db: Path) -> None:
    """Verify that qwen/qwen3-32b is the primary LLM model followed by Llama fallbacks."""
    agent = AssistantAgent(db_path=agent_db)
    assert agent.model_chain[0] == "qwen/qwen3-32b"
    assert agent.model_chain[1] == "llama-3.3-70b-versatile"
    assert agent.model_chain[2] == "llama-3.1-8b-instant"


def test_clean_thinking_tags() -> None:
    """Verify stripping <think>...</think> blocks from Qwen3 outputs."""
    raw = "<think>\nChecking stock for Takamine GD20...\n</think>\nO Takamine GD20 custa R$ 2.199,00!"
    assert clean_thinking_tags(raw) == "O Takamine GD20 custa R$ 2.199,00!"


def test_multilingual_language_detection() -> None:
    """Verify language detection across Portuguese, English, and Spanish user prompts."""
    assert detect_user_language("Quais opções de violões disponíveis até R$ 1000?") == "pt"
    assert detect_user_language("What acoustic guitars do you have in stock and how much is shipping?") == "en"
    assert detect_user_language("¿Hola, cuánto cuesta la guitarra y cuál es el horario de la tienda?") == "es"


def test_low_confidence_and_gibberish_guardrail_logs_failed_intent(agent_db: Path) -> None:
    """
    Verify that gibberish, ultra-short, or vague context-free messages trigger the
    low-confidence disambiguation menu and persist to the failed_intents table.
    """
    chat_service = ChatService(db_path=agent_db)

    # 1. Gibberish input
    res_gibberish = chat_service.process_message("xkcdqwrtyplmn", session_id="sess_low_1")
    assert res_gibberish["logged_failed_intent"] is True
    assert res_gibberish["model_used"] == "guardrail_low_confidence"
    assert "1️⃣" in res_gibberish["reply"]

    # 2. Vague single-word utterance with no prior history
    res_vague = chat_service.process_message("aquilo", session_id="sess_low_2")
    assert res_vague["logged_failed_intent"] is True
    assert "outras palavras" in res_vague["reply"]

    # 3. Verify records were saved in failed_intents repository
    failures = chat_service.agent.failed_intent_repo.get_all_failures()
    assert len(failures) == 2


def test_tool_executor_catalog_and_unavailable_alternatives(agent_db: Path) -> None:
    """Verify ToolExecutor returns available alternatives when querying out-of-stock or discontinued items."""
    executor = ToolExecutor(db_path=agent_db)

    # Out of stock product: Giannini GF-3D (ID 96)
    out_of_stock_res = executor.execute(
        "get_instrument_details_and_pricing",
        {"product_name": "Giannini GF-3D"},
    )
    assert out_of_stock_res["found"] is True
    assert out_of_stock_res["product"]["product_id"] == 96
    assert out_of_stock_res["product"]["in_stock"] is False
    assert len(out_of_stock_res["recommended_alternatives_in_stock"]) == 3

    # Discontinued product: Shelby SN-7C 7 Cordas (ID 113)
    disc_res = executor.execute(
        "get_instrument_details_and_pricing",
        {"product_name": "Shelby SN-7C"},
    )
    assert disc_res["found"] is True
    assert disc_res["product"]["status"] == "discontinued"
    assert len(disc_res["recommended_alternatives_in_stock"]) == 3


def test_tool_executor_ambiguous_customer_disambiguation(agent_db: Path) -> None:
    """
    Verify that searching orders by a shared customer first name (e.g. 'Bruno' or 'Camila')
    returns an ambiguity flag and masked options so the agent can ask the user to disambiguate.
    """
    executor = ToolExecutor(db_path=agent_db)
    res = executor.execute("check_order_status", {"customer_identifier": "Bruno"})
    assert res["found"] is False
    assert res.get("ambiguous_customers") is True
    assert len(res["matching_customers"]) >= 2


def test_llm_fallback_mechanism_on_primary_failure(agent_db: Path) -> None:
    """Verify that if primary model (qwen/qwen3-32b) raises an error, fallback model is used."""
    agent = AssistantAgent(db_path=agent_db)

    mock_client = MagicMock()
    call_models = []

    def side_effect(**kwargs):
        model = kwargs["model"]
        call_models.append(model)
        if model == "qwen/qwen3-32b":
            raise RuntimeError("Simulated rate limit on primary model")
        mock_resp = MagicMock()
        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = "Olá! Aqui é da Empório da Música, como posso ajudar?"
        mock_resp.choices = [MagicMock(message=mock_msg)]
        return mock_resp

    mock_client.chat.completions.create.side_effect = side_effect
    agent.client = mock_client

    result = agent.run("Olá, bom dia!", session_id="test_fallback")
    assert call_models == ["qwen/qwen3-32b", "llama-3.3-70b-versatile"]
    assert result["model_used"] == "llama-3.3-70b-versatile"
    assert "Empório da Música" in result["reply"]
