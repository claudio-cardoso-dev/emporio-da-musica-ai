"""Tests for Stage 4: Agent tools, LLM model fallback order, multilingual detection, and guardrails."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.agents.assistant_agent import (
    AssistantAgent,
    clean_thinking_tags,
    extract_claimed_customer_name,
    make_llm_tool_payload,
)
from src.agents.prompts import SYSTEM_PROMPT, detect_user_language
from src.agents.tools import TOOL_SCHEMAS, ToolExecutor, is_name_match
from src.db.ingestion import ingest_all_csvs
from src.services.chat_service import ChatService


@pytest.fixture
def agent_db(tmp_path: Path) -> Path:
    """Create and seed a temporary database for agent tool and guardrail tests."""
    db_file = tmp_path / "test_agent.db"
    ingest_all_csvs(db_path=db_file, force_reload=True)
    return db_file


def test_model_fallback_chain_order_qwen_primary(agent_db: Path) -> None:
    """Verify that Qwen (qwen/qwen3.8-27b) is the primary LLM model followed by fallbacks."""
    agent = AssistantAgent(db_path=agent_db)
    assert agent.model_chain[0] == "qwen/qwen3.8-27b"
    assert agent.model_chain[1] == "openai/gpt-oss-120b"
    assert agent.model_chain[2] == "openai/gpt-oss-20b"


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


def test_tool_executor_name_only_requires_primary_key_confirmation(agent_db: Path) -> None:
    """
    Verify that searching orders by customer name alone (even a full name like
    'Camila Duarte Nascimento' or first name 'Bruno') blocks order retrieval and requires
    phone or email confirmation as a primary key.
    """
    executor = ToolExecutor(db_path=agent_db)

    # 1. Shared first name ('Bruno') -> blocked, requires phone/email
    res_bruno = executor.execute("check_order_status", {"customer_identifier": "Bruno"})
    assert res_bruno["found"] is False
    assert res_bruno.get("requires_primary_key_confirmation") is True

    # 2. Unique full name ('Camila Duarte Nascimento') -> still blocked without phone/email
    res_camila = executor.execute(
        "check_order_status", {"customer_identifier": "Camila Duarte Nascimento"}
    )
    assert res_camila["found"] is False
    assert res_camila.get("requires_primary_key_confirmation") is True

    # 3. Verified with email primary key -> succeeds
    res_verified = executor.execute(
        "check_order_status", {"customer_identifier": "camila.duarte@inlook.com"}
    )
    assert res_verified["found"] is True
    assert res_verified["orders"][0]["order_id"] == 16


def test_llm_fallback_mechanism_on_primary_failure(agent_db: Path) -> None:
    """Verify that if primary model (qwen/qwen3.8-27b) raises an error, fallback model is used."""
    agent = AssistantAgent(db_path=agent_db)

    mock_client = MagicMock()
    call_models = []

    def side_effect(**kwargs):
        model = kwargs["model"]
        call_models.append(model)
        if model == "qwen/qwen3.8-27b":
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
    assert call_models == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    assert result["model_used"] == "openai/gpt-oss-120b"
    assert "Empório da Música" in result["reply"]


def test_tool_schemas_do_not_expose_valid_data_or_id_ranges() -> None:
    """
    Security & Privacy check: Ensure tool schemas and prompts never leak valid customer
    tracking codes (like BRJL5544332BR) or internal database ID ranges (like '1 to 20').
    """
    schemas_str = str(TOOL_SCHEMAS)
    assert "BRJL5544332BR" not in schemas_str
    assert "1 to 20" not in schemas_str
    assert "NEVER reveal internal database ID ranges" in SYSTEM_PROMPT


def test_identity_mismatch_prevents_data_leak_and_returns_no_third_party_info(agent_db: Path) -> None:
    """
    LGPD / Privacy check: When a customer gives their name (e.g. Camila Duarte Nascimento)
    and provides another user's phone (e.g. Bruno's 67985432109), the tool must:
    1. Mark verification_failed = True, found = False.
    2. NOT return 'customer' dictionary.
    3. NOT return 'orders' dictionary.
    4. NEVER mention Bruno's name or email anywhere in the response payload.
    """
    executor = ToolExecutor(db_path=agent_db)

    # Camila provides Bruno's phone number
    res_mismatch = executor.execute(
        "check_order_status",
        {
            "customer_identifier": "67985432109",
            "customer_name": "Camila Duarte Nascimento",
        },
    )
    assert res_mismatch["found"] is False
    assert res_mismatch.get("verification_failed") is True
    assert "customer" not in res_mismatch
    assert "orders" not in res_mismatch
    assert "Bruno" not in json.dumps(res_mismatch)

    # Camila provides unknown phone number
    res_unknown = executor.execute(
        "check_order_status",
        {
            "customer_identifier": "67999999999",
            "customer_name": "Camila Duarte Nascimento",
        },
    )
    assert res_unknown["found"] is False
    assert res_unknown.get("verification_failed") is True
    assert "customer" not in res_unknown
    assert "orders" not in res_unknown

    # Camila provides her own correct phone number
    res_valid = executor.execute(
        "check_order_status",
        {
            "customer_identifier": "(67) 99567-8901",
            "customer_name": "Camila Duarte Nascimento",
        },
    )
    assert res_valid["found"] is True
    assert res_valid["customer"]["name"] == "Camila Duarte Nascimento"
    assert len(res_valid["orders"]) >= 1


def test_extract_claimed_customer_name_patterns() -> None:
    """Verify claimed customer name extraction across multiple natural conversational patterns."""
    # 1. Full self-introduction
    msgs1 = [
        {"role": "user", "content": "olá, meu nome é Camila Duarte Nascimento, gostaria de ver meu rastreio"}
    ]
    assert extract_claimed_customer_name(msgs1) == "Camila Duarte Nascimento"

    # 2. 'sou a...' pattern
    msgs2 = [{"role": "user", "content": "sou a Camila Duarte"}]
    assert extract_claimed_customer_name(msgs2) == "Camila Duarte"

    # 3. 'me chamo...' pattern
    msgs3 = [{"role": "user", "content": "me chamo João Silva"}]
    assert extract_claimed_customer_name(msgs3) == "João Silva"

    # 4. English 'my name is...'
    msgs4 = [{"role": "user", "content": "hello, my name is John Doe, need tracking"}]
    assert extract_claimed_customer_name(msgs4) == "John Doe"

    # 5. Assistant greeting fallback
    msgs5 = [
        {"role": "assistant", "content": "Olá, Camila! Tudo bem? Me informe seu telefone."}
    ]
    assert extract_claimed_customer_name(msgs5) == "Camila"


def test_make_llm_tool_payload_compacts_catalog_output() -> None:
    """Verify make_llm_tool_payload strips heavy descriptions and limits sample to protect Groq TPM limit."""
    mock_products = [
        {
            "product_id": i,
            "name": f"Instrument {i}",
            "description": "Very long description " * 20,
            "category_name": "Ukuleles",
            "effective_price_brl": 199.9,
            "pix_price_brl": 189.9,
            "stock_quantity": 5,
            "in_stock": True,
            "active_promotion": None,
            "strings_type": "nylon",
        }
        for i in range(1, 16)
    ]
    tool_out = {
        "category": "Ukuleles",
        "products": mock_products,
    }
    llm_payload = make_llm_tool_payload("search_instruments", tool_out)

    assert llm_payload["total_matches"] == 15
    assert len(llm_payload["sample_products"]) == 10
    # Description should be omitted from LLM payload
    for sample in llm_payload["sample_products"]:
        assert "description" not in sample
        assert "price" in sample
        assert "pix_price" in sample
        assert "stock" in sample
