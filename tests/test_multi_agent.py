"""Tests for Multi-Agent Architecture: Router, Specialist Agents, and SRP Isolation."""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from src.agents.assistant_agent import AssistantAgent
from src.agents.router import IntentRouter
from src.agents.specialists import (
    CatalogAgent,
    CheckoutAgent,
    CustomerAgent,
    SupportAgent,
)
from src.api.app import create_app
from src.db.ingestion import ingest_all_csvs


@pytest.fixture
def multi_agent_db(tmp_path: Path) -> Path:
    """Create a temporary database with ingested seed data."""
    db_file = tmp_path / "test_multi_agent.db"
    ingest_all_csvs(db_path=db_file, force_reload=True)
    return db_file


def test_intent_router_classifies_stages() -> None:
    """Verify IntentRouter correctly classifies stages across Catalog, Checkout, Customer, Support."""
    # 1. Catalog stage
    assert IntentRouter.classify_stage("Quais opções de violões disponíveis até 1000 reais?") == "catalog"
    assert IntentRouter.classify_stage("Vocês têm ukulele soprano em estoque?") == "catalog"
    assert IntentRouter.classify_stage("Qual o preço do Takamine GD20?") == "catalog"

    # 2. Checkout stage
    assert IntentRouter.classify_stage("quero fechar o pedido") == "checkout"
    assert IntentRouter.classify_stage("Quero finalizar a compra agora") == "checkout"
    assert IntentRouter.classify_stage("Como faço para pagar?") == "checkout"
    assert IntentRouter.classify_stage("Checkout do meu carrinho") == "checkout"
    assert IntentRouter.classify_stage("ver carrinho") == "checkout"
    assert IntentRouter.classify_stage("quais itens no carrinho?") == "checkout"

    # 3. Customer registration & address stage
    assert IntentRouter.classify_stage("Quero fazer meu cadastro na loja") == "customer"
    assert IntentRouter.classify_stage("Gostaria de atualizar meu endereço de entrega") == "customer"
    assert IntentRouter.classify_stage("Meu CEP é 79202-333") == "customer"
    assert IntentRouter.classify_stage("Novo endereço para entrega") == "customer"

    # 4. Support & policies stage
    assert IntentRouter.classify_stage("Qual o status do meu pedido?") == "support"
    assert IntentRouter.classify_stage("Gostaria de ver meu rastreio") == "support"
    assert IntentRouter.classify_stage("Qual a política de troca e devolução?") == "support"
    assert IntentRouter.classify_stage("Qual o horário de funcionamento da loja física?") == "support"
    assert IntentRouter.classify_stage("Onde fica a loja?") == "support"


def test_agent_transitions_state_machine_flows() -> None:
    """Verify seamless transitions between agents following the state machine."""
    # 1. Transition from Support (store address with CEP) to Catalog (guitar inquiry)
    # This was the exact bug reported where store address with CEP tricked router into Customer agent
    history_support_store_address = [
        {"role": "user", "content": "Onde fica a loja e qual o horário de funcionamento?"},
        {
            "role": "assistant",
            "content": (
                "Endereço da loja: Rua 14 de Maio, 3200 — Centro, Campo Grande – MS, CEP 79202-333.\n"
                "Horário: Segunda a sexta: 08:30–18:30. WhatsApp: (67) 3321-4500."
            ),
            "metadata": {"active_agent": "support"},
        },
    ]
    # The customer now asks about guitars under 1000:
    next_stage = IntentRouter.classify_stage(
        user_message="Quais opções de violões disponíveis custando até R$1000?",
        chat_history=history_support_store_address,
    )
    assert next_stage == "catalog", f"Expected catalog, got {next_stage}"

    # 2. Transition from Catalog to Checkout
    history_catalog = [
        {"role": "user", "content": "Quais opções de ukuleles?"},
        {
            "role": "assistant",
            "content": "Aqui estão os ukuleles disponíveis no catálogo!",
            "metadata": {"active_agent": "catalog"},
        },
    ]
    stage_to_checkout = IntentRouter.classify_stage(
        user_message="Quero fechar o pedido",
        chat_history=history_catalog,
    )
    assert stage_to_checkout == "checkout"

    # 3. Transition from Checkout to Customer (address / CEP)
    history_checkout = [
        {"role": "user", "content": "Quero fechar o pedido"},
        {
            "role": "assistant",
            "content": "Perfeito! Para onde enviamos seu pedido?",
            "metadata": {"active_agent": "checkout"},
        },
    ]
    stage_to_customer = IntentRouter.classify_stage(
        user_message="Meu CEP é 79202-333",
        chat_history=history_checkout,
    )
    assert stage_to_customer == "customer"

    # 4. Transition from Customer to Support (policies query)
    history_customer = [
        {"role": "user", "content": "Meu CEP é 79202-333"},
        {
            "role": "assistant",
            "content": "Localizei seu endereço na Rua 14 de Maio. Qual o número?",
            "metadata": {"active_agent": "customer"},
        },
    ]
    stage_to_support = IntentRouter.classify_stage(
        user_message="Qual a política de troca e devolução se eu não me adaptar?",
        chat_history=history_customer,
    )
    assert stage_to_support == "support"

    # 5. Slot Filling continuation in Support (phone authentication)
    history_support_auth = [
        {"role": "user", "content": "Gostaria de ver meu rastreio"},
        {
            "role": "assistant",
            "content": "Para consultar seu rastreio com segurança, por favor confirme seu telefone ou e-mail cadastrado.",
            "metadata": {"active_agent": "support"},
        },
    ]
    stage_continuation = IntentRouter.classify_stage(
        user_message="67985432109",
        chat_history=history_support_auth,
    )
    assert stage_continuation == "support"


def test_specialist_agents_tool_scoping(multi_agent_db: Path) -> None:
    """Verify each Specialist Agent only has access to its single responsibility tools."""
    agent = AssistantAgent(db_path=multi_agent_db)

    # Catalog Agent: search & product details
    catalog_tools = [t["function"]["name"] for t in agent.catalog_agent.scoped_tool_schemas]
    assert "search_instruments" in catalog_tools
    assert "get_instrument_details_and_pricing" in catalog_tools
    assert "check_order_status" not in catalog_tools
    assert "lookup_cep" not in catalog_tools

    # Checkout Agent: payment simulation & policies (NEVER catalog search)
    checkout_tools = [t["function"]["name"] for t in agent.checkout_agent.scoped_tool_schemas]
    assert "simulate_payment_and_shipping" in checkout_tools
    assert "consult_store_policies" in checkout_tools
    assert "search_instruments" not in checkout_tools
    assert "get_instrument_details_and_pricing" not in checkout_tools

    # Customer Agent: CEP and registration
    customer_tools = [t["function"]["name"] for t in agent.customer_agent.scoped_tool_schemas]
    assert "lookup_cep" in customer_tools
    assert "register_customer" in customer_tools
    assert "update_customer_address" in customer_tools
    assert "search_instruments" not in customer_tools

    # Support Agent: order status & policies
    support_tools = [t["function"]["name"] for t in agent.support_agent.scoped_tool_schemas]
    assert "check_order_status" in support_tools
    assert "consult_store_policies" in support_tools
    assert "search_instruments" not in support_tools


def test_multi_agent_orchestrator_dispatch(multi_agent_db: Path) -> None:
    """Verify AssistantAgent dispatches turns to correct specialist and populates active_agent."""
    orchestrator = AssistantAgent(db_path=multi_agent_db)

    # 1. Checkout dispatch
    res_checkout = orchestrator.run("quero fechar o pedido", session_id="test_ma_1")
    assert res_checkout["active_agent"] == "checkout"
    assert res_checkout["products"] == []

    # 2. Customer dispatch
    res_cust = orchestrator.run("Meu CEP é 79202-333, quero cadastrar meu endereço", session_id="test_ma_2")
    assert res_cust["active_agent"] == "customer"
    assert res_cust["products"] == []

    # 3. Support dispatch
    res_supp = orchestrator.run("Qual o horário de funcionamento da loja física?", session_id="test_ma_3")
    assert res_supp["active_agent"] == "support"
    assert res_supp["products"] == []


def test_api_chat_response_includes_active_agent(tmp_path: Path) -> None:
    """Verify /api/chat response contains active_agent field."""
    db_file = tmp_path / "test_api_ma.db"
    test_app = create_app(db_path=db_file)
    client = TestClient(test_app)

    res = client.post("/api/chat", json={"message": "Quais ukuleles vocês têm?", "session_id": "test_ma_api"})
    assert res.status_code == 200
    data = res.json()
    assert "active_agent" in data
    assert data["active_agent"] in ("catalog", "checkout", "customer", "support", "orchestrator")


def test_unregistered_customer_form_flow_and_address_step(multi_agent_db: Path) -> None:
    """
    Verify complete customer onboarding flow:
    1. Unregistered customer data lookup fails -> Assistant asks if user is a new customer.
    2. User says 'Sim, sou novo cliente' -> Assistant returns action='SHOW_REGISTER_FORM'.
    3. User submits [DADOS_DO_CADASTRO] -> Assistant returns action='SHOW_ADDRESS_FORM'.
    4. User submits [DADOS_DO_ENDERECO] -> Customer and address are registered in database.
    """
    agent = AssistantAgent(db_path=multi_agent_db)

    # 1. Unregistered lookup in tool
    res_tool = agent.tool_executor.check_order_status(customer_identifier="11945256750")
    assert res_tool["found"] is False
    assert res_tool.get("customer_not_found") is True
    assert "Sim, sou novo cliente" in res_tool.get("options", [])

    # 2. User confirms they are a new customer -> Bot speaks about Termo de Aceite (LGPD)
    history_after_question = [
        {"role": "user", "content": "11945256750"},
        {
            "role": "assistant",
            "content": "Não localizei um cadastro associado a este contato. Você é um novo cliente aqui na Empório da Música? 🎶",
            "metadata": {"active_agent": "support", "options": ["Sim, sou novo cliente", "Não, vou tentar outro dado"]},
        },
    ]
    res_confirm = agent.run("Sim, sou novo cliente", chat_history=history_after_question)
    assert "termo" in res_confirm["reply"].lower()
    assert "Aceito os termos" in res_confirm["options"]
    assert res_confirm["active_agent"] == "customer"

    # 3. User accepts terms -> Opens registration form
    history_after_terms = history_after_question + [
        {"role": "user", "content": "Sim, sou novo cliente"},
        {"role": "assistant", "content": res_confirm["reply"], "metadata": {"options": ["Aceito os termos", "Não aceito"]}},
    ]
    res_accept = agent.run("Aceito os termos", chat_history=history_after_terms)
    assert res_accept["action"] == "SHOW_REGISTER_FORM"
    assert res_accept["active_agent"] == "customer"

    # 4. User submits registration data -> Opens address form
    form_msg = '[DADOS_DO_CADASTRO]: {"name": "Carlos Eduardo", "phone": "11945256750", "email": "carlos@teste.com"}'
    history_after_form1 = history_after_terms + [
        {"role": "user", "content": "Aceito os termos"},
        {"role": "assistant", "content": res_accept["reply"], "metadata": {"action": "SHOW_REGISTER_FORM"}},
    ]
    res_step2 = agent.run(form_msg, chat_history=history_after_form1)
    assert res_step2["action"] == "SHOW_ADDRESS_FORM"
    assert "Carlos Eduardo" in res_step2["reply"]
    assert res_step2["active_agent"] == "customer"

    # 5. User submits address data -> Registration completed
    addr_msg = '[DADOS_DO_ENDERECO]: {"cep": "79202-333", "street": "Rua 14 de Maio", "number": "3200", "neighborhood": "Centro", "city": "Campo Grande"}'
    history_after_step2 = history_after_form1 + [
        {"role": "user", "content": form_msg},
        {"role": "assistant", "content": res_step2["reply"], "metadata": {"action": "SHOW_ADDRESS_FORM"}},
    ]
    res_step3 = agent.run(addr_msg, chat_history=history_after_step2)
    assert res_step3["action"] is None
    assert "Carlos Eduardo" in res_step3["reply"]
    assert "Rua 14 de Maio" in res_step3["reply"]
    assert res_step3["active_agent"] == "customer"

    # Verify customer is now saved in SQLite database
    saved_custs = agent.tool_executor.customer_repo.find_customers("11945256750")
    assert len(saved_custs) == 1
    assert saved_custs[0]["name"] == "Carlos Eduardo"
    assert "Rua 14 de Maio" in saved_custs[0]["address"]


def test_checkout_unregistered_email_triggers_register_form(multi_agent_db: Path) -> None:
    """Verify that in checkout, providing an unregistered email asks if user is a new customer, then presents terms and triggers SHOW_REGISTER_FORM."""
    agent = AssistantAgent(db_path=multi_agent_db)

    history_checkout = [
        {"role": "user", "content": "fechar pedido"},
        {
            "role": "assistant",
            "content": "Oi! Vamos fechar seu pedido rapidinho. Para concluir a compra, preciso de alguns dados: 1. Telefone/WhatsApp ou e-mail...",
            "metadata": {"active_agent": "checkout"},
        },
    ]

    res = agent.run("kardoso@teste.com", chat_history=history_checkout)
    assert "novo cliente" in res["reply"].lower()
    assert "Sim, sou novo cliente" in res.get("options", [])

    history_confirm = history_checkout + [
        {"role": "user", "content": "kardoso@teste.com"},
        {"role": "assistant", "content": res["reply"], "metadata": {"active_agent": "checkout"}},
    ]
    res_confirm = agent.run("Sim, sou novo cliente", chat_history=history_confirm)
    assert "termo" in res_confirm["reply"].lower()
    assert "Aceito os termos" in res_confirm.get("options", [])

    history_accept = history_confirm + [
        {"role": "user", "content": "Sim, sou novo cliente"},
        {"role": "assistant", "content": res_confirm["reply"], "metadata": {"options": ["Aceito os termos", "Não aceito"]}},
    ]
    res_accept = agent.run("Aceito os termos", chat_history=history_accept)
    assert res_accept["action"] == "SHOW_REGISTER_FORM"
    assert res_accept.get("prefill", {}).get("email") == "kardoso@teste.com"


def test_explicit_registration_and_address_update_triggers(multi_agent_db: Path) -> None:
    """Verify that explicit requests to register or update address open forms directly in chat."""
    agent = AssistantAgent(db_path=multi_agent_db)

    # 1. Explicit registration
    res_reg = agent.run("Quero me cadastrar")
    assert res_reg["action"] == "SHOW_REGISTER_FORM"
    assert res_reg["active_agent"] == "customer"

    # 2. Explicit address update
    res_addr = agent.run("Atualizar endereço")
    assert res_addr["action"] == "SHOW_ADDRESS_FORM"
    assert res_addr["active_agent"] == "customer"


def test_checkout_registered_customer_address_confirmation(multi_agent_db: Path) -> None:
    """Verify that in checkout, providing a registered email displays their address and asks to keep or update."""
    agent = AssistantAgent(db_path=multi_agent_db)

    history_checkout = [
        {"role": "user", "content": "fechar pedido"},
        {
            "role": "assistant",
            "content": "Oi! Vamos fechar seu pedido rapidinho. Por favor informe seu e-mail ou telefone.",
            "metadata": {"active_agent": "checkout"},
        },
    ]

    # Lucas Mendes da Silva is in the seed DB
    res = agent.run("lucas.mendes@jmail.com", chat_history=history_checkout)
    assert "Lucas" in res["reply"]
    assert "Manter este endereço" in res["options"]
    assert "Atualizar endereço" in res["options"]
    assert res["action"] is None

    # Customer confirms keeping address
    history_keep = history_checkout + [
        {"role": "user", "content": "lucas.mendes@jmail.com"},
        {"role": "assistant", "content": res["reply"], "metadata": {"active_agent": "checkout"}},
    ]
    res_keep = agent.run("Manter este endereço", chat_history=history_keep)
    assert "Endereço de entrega confirmado" in res_keep["reply"] or "confirmado" in res_keep["reply"]
    assert "PIX" in res_keep["reply"]


def test_catalog_ukuleles_excludes_zero_stock_and_deactivated(multi_agent_db: Path) -> None:
    """Verify that querying ukulele options displays the carousel and strictly excludes zero-stock/deactivated items."""
    agent = AssistantAgent(db_path=multi_agent_db)

    res = agent.run("Quais opções de ukuleles disponíveis?")
    assert res["active_agent"] == "catalog"
    prods = res.get("products", [])
    assert len(prods) > 0, "Product carousel cards must be present"

    # Ensure every single product in carousel is active and has stock > 0
    prod_ids = set()
    for p in prods:
        prod_ids.add(p["product_id"])
        assert p["status"] == "active", f"Found deactivated product {p['name']}"
        assert p["stock_quantity"] > 0, f"Found zero-stock product {p['name']}"

    # Product 130 (Kala KA-ABP-CTG Barítono Cedar) has status='coming_soon' and must not appear
    assert 130 not in prod_ids, "Deactivated 'coming_soon' product 130 must not appear in available list"


def test_checkout_cart_simulation_calculates_totals_and_installments_without_asking_user(multi_agent_db: Path) -> None:
    """
    Verify that when user simulates cart (e.g. 'Selecionei no carrinho: 1x Tagima Memphis AC-39..., 1x Rozini RC-104...'),
    the CheckoutAgent calculates the subtotal (R$ 889,80), PIX discount (R$ 845,31), installments up to 8x,
    free motoboy shipping, and NEVER asks the user to calculate the total in R$.
    """
    agent = AssistantAgent(db_path=multi_agent_db)

    user_msg = (
        "Selecionei no carrinho: 1x Tagima Memphis AC-39 Nylon Natural, 1x Rozini RC-104 Nylon Estudante. "
        "Como ficam as opções de pagamento (PIX e parcelamento sem juros) e o frete para Campo Grande?"
    )

    res = agent.run(user_msg)
    assert res["active_agent"] == "checkout"
    reply = res["reply"]

    # Verify calculation of total and PIX
    assert "889,80" in reply
    assert "845,31" in reply
    assert "PIX" in reply

    # Verify installments
    assert "8x" in reply
    assert "111,22" in reply

    # Verify shipping
    assert "Frete Grátis" in reply or "gratuito" in reply.lower()
    assert "Campo Grande" in reply

    # Strict check: NEVER ask user to provide total or calculate values
    assert "preciso apenas do valor total" not in reply.lower()
    assert "você pode me passar esse valor" not in reply.lower()
    assert "soma dos dois" not in reply.lower()


def test_complete_customer_journey_flow_end_to_end(multi_agent_db: Path) -> None:
    """
    Test the full conversational flow:
    1. Greeting / Identification request.
    2. Phone not found in DB -> Ask if new customer or retry.
    3. User says 'Sou novo cliente' -> Termo de aceite (LGPD).
    4. User accepts terms -> SHOW_REGISTER_FORM.
    5. User submits contact form -> SHOW_ADDRESS_FORM.
    6. User submits address form -> Confirms onboarding, asks 'Como posso ajudar?'.
    7. User closes cart ('Quero fechar meu pedido...') -> Resumo do pedido + Pergunta Cartão ou PIX.
    8. User chooses 'PIX' -> Presents registered address and asks if correct.
    9. User confirms address ('Sim, endereço correto') -> Final summary with 5% PIX discount & free motoboy shipping!
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_customer_journey_full_flow"

    # Step 1 & 2: User provides unregistered phone
    res1 = agent.run("67998877665", session_id=sess_id)
    assert "Não localizamos" in res1["reply"] or "novo cliente" in res1["reply"].lower()
    assert "Sim, sou novo cliente" in res1["options"]

    # Step 3: User says new customer -> Termo de Aceite
    res2 = agent.run("Sim, sou novo cliente", session_id=sess_id)
    assert "Termo de Consentimento" in res2["reply"] or "LGPD" in res2["reply"]
    assert "Aceito os termos" in res2["options"]

    # Step 4: User accepts terms -> SHOW_REGISTER_FORM
    res3 = agent.run("Aceito os termos", session_id=sess_id)
    assert res3["action"] == "SHOW_REGISTER_FORM"
    assert res3["prefill"]["phone"] == "67998877665"

    # Step 5: User submits registration -> SHOW_ADDRESS_FORM
    contact_payload = '[DADOS_DO_CADASTRO]: {"name": "Fernanda Lima", "phone": "67998877665", "email": "fernanda@exemplo.com"}'
    res4 = agent.run(contact_payload, session_id=sess_id)
    assert res4["action"] == "SHOW_ADDRESS_FORM"
    assert "Fernanda Lima" in res4["reply"]

    # Step 6: User submits address -> Confirmation + asks how to help
    address_payload = '[DADOS_DO_ENDERECO]: {"cep": "79202-333", "street": "Rua 14 de Maio", "number": "100", "neighborhood": "Centro", "city": "Campo Grande"}'
    res5 = agent.run(address_payload, session_id=sess_id)
    assert res5["action"] is None
    assert "Fernanda Lima" in res5["reply"]
    assert "Como posso te ajudar agora?" in res5["reply"]

    # Step 7: User intends to checkout
    cart_msg = "Quero fechar meu pedido com 1x Tagima Memphis AC-39 Nylon Natural, 1x Rozini RC-104 Nylon Estudante"
    res6 = agent.run(cart_msg, session_id=sess_id)
    assert "Resumo do seu Pedido" in res6["reply"]
    assert "889,80" in res6["reply"]
    assert "Qual forma de pagamento você prefere" in res6["reply"]
    assert "Pagar no PIX (5% off)" in res6["options"][0]

    # Step 8: User chooses PIX -> Presents registered address
    res7 = agent.run("Pagar no PIX", session_id=sess_id)
    assert "Rua 14 de Maio" in res7["reply"]
    assert "endereço de entrega está correto" in res7["reply"].lower()
    assert "Sim, endereço correto" in res7["options"]

    # Step 9: User confirms address -> Final summary for confirmation
    res8 = agent.run("Sim, endereço correto", session_id=sess_id)
    assert "Pedido Pronto para Finalização" in res8["reply"]
    assert "845,31" in res8["reply"]  # 889.80 - 5% = 845.31
    assert "Grátis" in res8["reply"]
    assert "Fernanda Lima" in res8["reply"]
    assert "Sim, confirmar pedido" in res8["options"]

    # Step 10: User confirms order -> Concluded
    res9 = agent.run("Sim, confirmar pedido", session_id=sess_id)
    assert "Pedido Confirmado com Sucesso" in res9["reply"]
    assert "Fernanda Lima" in res9["reply"]
    assert "🔄 Novo Atendimento" in res9["options"]


def test_identification_cannot_be_bypassed_with_greetings_or_catalog_queries(multi_agent_db: Path) -> None:
    """Verify that in the web chat, greeting or sending catalog queries cannot bypass the identification stage."""
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "web_test_enforce_identity"

    # 1. User says "oi" -> Must prompt for phone/e-mail, NOT greet and ask "como posso ajudar"
    res1 = agent.run("oi", session_id=sess_id)
    assert "telefone" in res1["reply"].lower() or "e-mail" in res1["reply"].lower()
    assert "como posso te ajudar hoje" not in res1["reply"].lower()

    # 2. User tries to ask for violões without identifying -> Must still enforce identification
    res2 = agent.run("Quais opções de violões disponíveis custando até R$1000?", session_id=sess_id)
    assert "telefone" in res2["reply"].lower() or "e-mail" in res2["reply"].lower()
    assert res2.get("products") == []

    # 3. User provides unknown phone -> NOW bot asks if new customer or retry
    res3 = agent.run("11999998888", session_id=sess_id)
    assert "novo cliente" in res3["reply"].lower()
    assert "Sim, sou novo cliente" in res3.get("options", [])
    assert "Tentar novamente" in res3.get("options", [])


def test_catalog_query_price_limit_returns_products_and_displays_carousel(multi_agent_db: Path) -> None:
    """Verify that querying products with a price ceiling (e.g. 'quero ver produtos até 300 reais') triggers search and populates carousel."""
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_catalog_ceiling_300"

    # Pre-identify customer
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # Ask for products under R$ 300
    res = agent.run("quero ver produtos até 300 reais", session_id=sess_id)
    assert res["active_agent"] == "catalog"
    products = res.get("products", [])
    assert len(products) == 4, f"Expected 4 instruments under R$300, got {len(products)}"
    for p in products:
        assert p["effective_price_brl"] <= 300.0
        assert p["stock_quantity"] > 0
        assert p["status"] == "active"


def test_pix_checkout_generates_qrcode_and_whatsapp_followup(multi_agent_db: Path) -> None:
    """
    Verify that checkout with PIX:
    1. Simulates QR Code generation (/api/pix-qrcode) and Copy-and-Paste code instead of sending CNPJ.
    2. Does NOT ask the customer to send payment receipt.
    3. Informs that the store will contact via WhatsApp once payment is cleared to provide order status.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_pix_qr_followup"

    # Step 1: Customer identification
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # Step 2: User closes cart with item
    cart_msg = "Quero fechar meu pedido com 1x Tagima Memphis AC-39 Nylon Natural"
    res_close = agent.run(cart_msg, session_id=sess_id)
    assert "Pagar no PIX (5% off)" in res_close["options"][0]

    # Step 4: Choose PIX
    res_pix = agent.run("Pagar no PIX", session_id=sess_id)
    assert "Sim, endereço correto" in res_pix["options"]

    # Step 5: Confirm address -> Pre-confirmation Summary
    res_summary = agent.run("Sim, endereço correto", session_id=sess_id)
    assert "12.345.678/0001-90" not in res_summary["reply"]
    assert "QR Code" in res_summary["reply"]
    assert "Sim, confirmar pedido" in res_summary["options"]

    # Step 6: User clicks 'Sim, confirmar pedido' -> Final Order Confirmation
    res_confirm = agent.run("Sim, confirmar pedido", session_id=sess_id)
    reply = res_confirm["reply"]

    # 1. QR Code and Copia e Cola present
    assert "/api/pix-qrcode" in reply
    assert "Código PIX Copia e Cola" in reply
    assert "00020126580014" in reply

    # 2. No CNPJ sent
    assert "12.345.678/0001-90" not in reply

    # 3. No request for receipt
    assert "enviar o comprovante" not in reply
    assert "envie o comprovante" not in reply

    # 4. WhatsApp follow-up upon clearance
    assert "compensado" in reply
    assert "WhatsApp" in reply
    assert "status" in reply.lower()


def test_registered_customer_checkout_never_asks_cep_again(multi_agent_db: Path) -> None:
    """Verify that a registered customer who proceeds to checkout is NEVER asked for CEP again."""
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_no_cep_reprompt"

    # 1. Customer identified (Lucas Mendes da Silva, CEP already registered)
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # 2. Closes order with item
    cart_msg = "Quero fechar meu pedido com 2x Shelby SN-100 Folk Aço Preto"
    res = agent.run(cart_msg, session_id=sess_id)

    # 3. Must present cart summary, NOT ask for CEP
    reply = res["reply"]
    assert "CEP de entrega" not in reply
    assert "preciso de duas informações rápidas" not in reply
    assert "Resumo do seu Pedido" in reply
    assert "1.098,00" in reply
    assert "Pagar no PIX (5% off)" in res["options"][0]

    # 4. Choose payment
    res_pay = agent.run("Pagar no PIX", session_id=sess_id)
    assert "endereço de entrega está correto" in res_pay["reply"].lower()
    assert "Sim, endereço correto" in res_pay["options"]


def test_catalog_reply_anti_redundancy_excludes_tables_when_carousel_present(multi_agent_db: Path) -> None:
    """Verify that catalog responses with carousel products do NOT include redundant markdown tables in chat text."""
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_anti_redundancy"

    # Identify customer
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # Ask for ukuleles
    res = agent.run("Quais opções de ukuleles disponíveis?", session_id=sess_id)
    assert res["active_agent"] == "catalog"
    assert len(res["products"]) > 0

    # Ensure text does NOT contain Markdown table dividers
    reply = res["reply"]
    assert "| ---" not in reply
    assert "| Instrumento" not in reply
    assert "| Modelo" not in reply






