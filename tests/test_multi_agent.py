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

    # Step 7: User intends to checkout -> Directly presents cart summary AND address confirmation
    cart_msg = "Quero fechar meu pedido com 1x Tagima Memphis AC-39 Nylon Natural, 1x Rozini RC-104 Nylon Estudante"
    res6 = agent.run(cart_msg, session_id=sess_id)
    assert "Resumo do seu Pedido" in res6["reply"]
    assert "889,80" in res6["reply"]
    assert "Rua 14 de Maio" in res6["reply"]
    assert "endereço de entrega está correto" in res6["reply"].lower()
    assert "Sim, endereço correto" in res6["options"]

    # Step 8: User confirms address -> Final summary presents payment options directly
    res7 = agent.run("Sim, endereço correto", session_id=sess_id)
    assert "Pedido Pronto para Finalização" in res7["reply"]
    assert "845,31" in res7["reply"]  # 889.80 - 5% = 845.31
    assert "Grátis" in res7["reply"]
    assert "Fernanda Lima" in res7["reply"]
    assert any("PIX" in opt for opt in res7["options"])
    assert any("Parcelar" in opt for opt in res7["options"])
    assert "Sim, confirmar pedido" not in res7["options"]
    assert "Não, alterar endereço" not in res7["options"]

    # Step 9: User chooses payment method -> Final Order Review
    res8 = agent.run("⚡ Pagar no PIX (5% off)", session_id=sess_id)
    assert "Resumo Final do Pedido" in res8["reply"]
    assert "845,31" in res8["reply"]
    assert "Sim, confirmar pedido" in res8["options"]
    assert "Não, alterar pagamento" in res8["options"]

    # Step 10: User confirms order -> Concluded with success, asks CSAT 1 to 5
    res9 = agent.run("Sim, confirmar pedido", session_id=sess_id)
    assert "Pedido Confirmado com Sucesso" in res9["reply"]
    assert "Fernanda Lima" in res9["reply"]
    assert any("⭐ 1" in opt for opt in res9["options"])
    assert any("⭐ 5" in opt for opt in res9["options"])

    # Step 11: User provides rating -> Closed with thanks
    res10 = agent.run("5", session_id=sess_id)
    assert "Agradecemos muito pela sua avaliação" in res10["reply"]
    assert res10["action"] == "CLOSE_CHAT"
    assert "🔄 Novo Atendimento" in res10["options"]


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

    # Step 2: User closes cart with item -> Directly asks for address confirmation
    cart_msg = "Quero fechar meu pedido com 1x Tagima Memphis AC-39 Nylon Natural"
    res_close = agent.run(cart_msg, session_id=sess_id)
    assert "Sim, endereço correto" in res_close["options"]

    # Step 3: Confirm address -> Summary with payment options
    res_summary = agent.run("Sim, endereço correto", session_id=sess_id)
    assert "12.345.678/0001-90" not in res_summary["reply"]
    assert "QR Code" in res_summary["reply"]
    assert any("PIX" in opt for opt in res_summary["options"])
    assert "Sim, confirmar pedido" not in res_summary["options"]

    # Step 4: User clicks payment option -> Final Order Review
    res_review = agent.run("⚡ Pagar no PIX (5% off)", session_id=sess_id)
    assert "Resumo Final do Pedido" in res_review["reply"]
    assert "Sim, confirmar pedido" in res_review["options"]
    assert "Não, alterar pagamento" in res_review["options"]

    # Step 5: User confirms order -> Final Order Confirmation with PIX QR Code & WhatsApp follow-up
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

    # 3. Must present cart summary, NOT ask for CEP, and directly prompt for address confirmation
    reply = res["reply"]
    assert "CEP de entrega" not in reply
    assert "preciso de duas informações rápidas" not in reply
    assert "Resumo do seu Pedido" in reply
    assert "1.098,00" in reply
    assert "endereço de entrega está correto" in reply.lower()
    assert "Sim, endereço correto" in res["options"]


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


def test_all_store_payment_options_from_policies(multi_agent_db: Path) -> None:
    """
    Verify Section 3 payment options are presented during checkout:
    PIX (5% off), Cartão de Crédito sem juros, Boleto Bancário (3 dias úteis), Cartão de Débito,
    and simulation completes safely without requesting customer card credentials.
    """
    agent = AssistantAgent(db_path=multi_agent_db)

    # Pre-identified customer
    sess_id = "test_payment_options_sec3"
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # 1. Close cart with items
    res_cart = agent.run("Quero fechar meu pedido com 2x Tagima Memphis AC-39 Nylon Natural", session_id=sess_id)
    reply = res_cart["reply"]

    # Verify all Section 3 methods are displayed
    assert "À vista no PIX" in reply
    assert "Cartão de Crédito" in reply
    assert "Boleto Bancário" in reply
    assert "Cartão de Débito" in reply
    assert "3 dias úteis" in reply

    # Flow jumps straight to address confirmation
    assert "Sim, endereço correto" in res_cart["options"]
    assert "Não, alterar endereço" in res_cart["options"]

    # When address is confirmed, all Section 3 methods are displayed as interactive buttons
    res_summary = agent.run("Sim, endereço correto", session_id=sess_id)
    options = res_summary["options"]
    assert any("PIX" in opt for opt in options)
    assert any("Cartão" in opt for opt in options) or any("Parcelar" in opt for opt in options)
    assert any("Boleto Bancário" in opt for opt in options)
    assert any("Cartão de Débito" in opt for opt in options)

    # 2. Test choosing Boleto Bancário
    sess_boleto = "test_pay_boleto"
    agent.run("lucas.mendes@jmail.com", session_id=sess_boleto)
    agent.run("Quero fechar meu pedido com 2x Tagima Memphis AC-39 Nylon Natural", session_id=sess_boleto)
    res_sel_bol = agent.run("Boleto Bancário", session_id=sess_boleto)
    assert "Boleto Bancário" in res_sel_bol["reply"]
    assert "endereço de entrega está correto" in res_sel_bol["reply"].lower()

    res_conf_addr = agent.run("Sim, endereço correto", session_id=sess_boleto)
    assert "Boleto Bancário" in res_conf_addr["reply"]
    assert "Prazo de Compensação" in res_conf_addr["reply"]

    res_final_bol = agent.run("Sim, confirmar pedido", session_id=sess_boleto)
    assert "Boleto Bancário Gerado com Sucesso" in res_final_bol["reply"]
    assert "Linha Digitável" in res_final_bol["reply"]
    assert "3 dias úteis" in res_final_bol["reply"]

    # 3. Test choosing Cartão de Crédito (simulated safely without asking for card details)
    sess_card = "test_pay_credit_card"
    agent.run("lucas.mendes@jmail.com", session_id=sess_card)
    agent.run("Quero fechar meu pedido com 2x Tagima Memphis AC-39 Nylon Natural", session_id=sess_card)
    res_sel_card = agent.run("Cartão de Crédito", session_id=sess_card)
    assert "Cartão de Crédito" in res_sel_card["reply"]

    res_conf_addr_card = agent.run("Sim, endereço correto", session_id=sess_card)
    assert "Cartão de Crédito" in res_conf_addr_card["reply"]
    assert "não é necessário digitar os dados do seu cartão" in res_conf_addr_card["reply"].lower()

    res_final_card = agent.run("Sim, confirmar pedido", session_id=sess_card)
    assert "Cartão de Crédito Aprovado" in res_final_card["reply"]
    # Ensure card details (number, CVV, password) were NEVER requested
    assert "número do cartão" not in res_final_card["reply"].lower()
    assert "cvv" not in res_final_card["reply"].lower()


def test_checkout_summary_presents_payment_and_installment_options(multi_agent_db: Path) -> None:
    """
    Verify that in final checkout summary:
    1. Generic 'Sim, confirmar pedido' and 'Não, alterar endereço' are NOT displayed.
    2. Interactive options display à vista payment buttons and installment button.
    3. Clicking installment button displays released installment options breakdown and buttons.
    4. Selecting a released installment confirms order with exact installment details.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_installments_interactive_flow"

    # Identify customer
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # Add items to cart (2x Tagima Memphis AC-39 = R$ 859,80)
    agent.run("Quero fechar meu pedido com 2x Tagima Memphis AC-39 Nylon Natural", session_id=sess_id)

    # Address confirm step
    agent.run("Pagar no Cartão de Crédito", session_id=sess_id)
    res_summary = agent.run("Sim, endereço correto", session_id=sess_id)

    # 1. Verify summary options are payment options
    opts = res_summary["options"]
    assert "Sim, confirmar pedido" not in opts
    assert "Não, alterar endereço" not in opts
    assert any("PIX" in opt for opt in opts)
    assert any("1x" in opt for opt in opts)
    assert any("Parcelar" in opt for opt in opts)
    assert any("Boleto" in opt for opt in opts)
    assert any("Débito" in opt for opt in opts)

    # 2. Click on Parcelar -> returns table and released installment buttons
    parcelar_btn = next(opt for opt in opts if "Parcelar" in opt)
    res_installments = agent.run(parcelar_btn, session_id=sess_id)
    inst_reply = res_installments["reply"]
    inst_opts = res_installments["options"]

    assert "Opções de Parcelamento" in inst_reply
    assert "| Parcelas |" in inst_reply
    # Total R$ 859,80 allows up to 8x sem juros (8x de R$ 107,47 >= R$ 100)
    assert "8x" in inst_reply
    assert any("1x" in opt for opt in inst_opts)
    assert any("3x" in opt for opt in inst_opts)
    assert any("8x" in opt for opt in inst_opts)

    # 3. Choose 3x installment -> Presents Final Order Review with chosen 3x installment
    opt_3x = next(opt for opt in inst_opts if "3x" in opt)
    res_review = agent.run(opt_3x, session_id=sess_id)
    review_reply = res_review["reply"]

    assert "Resumo Final do Pedido" in review_reply
    assert "3x" in review_reply
    assert "sem juros" in review_reply
    assert "Lucas" in review_reply
    assert "Sim, confirmar pedido" in res_review["options"]
    assert "Não, alterar pagamento" in res_review["options"]

    # 4. User confirms order -> Confirms order with 3x sem juros
    res_confirmed = agent.run("Sim, confirmar pedido", session_id=sess_id)
    conf_reply = res_confirmed["reply"]

    assert "Pedido Confirmado com Sucesso" in conf_reply
    assert "3x" in conf_reply
    assert "sem juros" in conf_reply
    assert "Lucas" in conf_reply

    # 5. Verify "Não, alterar pagamento" flow
    sess_change = "test_change_payment_flow"
    agent.run("lucas.mendes@jmail.com", session_id=sess_change)
    agent.run("Quero fechar meu pedido com 2x Tagima Memphis AC-39 Nylon Natural", session_id=sess_change)
    agent.run("Sim, endereço correto", session_id=sess_change)
    # Customer chooses PIX
    res_pix_rev = agent.run("⚡ Pagar no PIX (5% off)", session_id=sess_change)
    assert "Não, alterar pagamento" in res_pix_rev["options"]
    # Customer clicks "Não, alterar pagamento"
    res_back = agent.run("Não, alterar pagamento", session_id=sess_change)
    assert any("PIX" in opt for opt in res_back["options"])
    assert any("Cartão" in opt or "Parcelar" in opt for opt in res_back["options"])


def test_natural_language_free_text_checkout_flow_pix(multi_agent_db: Path) -> None:
    """
    Verify complete natural language conversational flow without clicking any UI buttons:
    1. 'quero fechar o pedido' with synchronized cart payload.
    2. 'está correto' / 'ta certo' for address confirmation.
    3. 'vou de pix' for payment choice.
    4. 'pode confirmar' for final order placement.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_nl_pix_free_text"

    # 1. Customer Identification via free text
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # 2. Free text checkout intent with frontend-synced cart
    cart_payload = [
        {"product_id": 1, "name": "Tagima Memphis AC-39 Nylon Natural", "price_brl": 429.90, "effective_price_brl": 429.90, "qty": 2}
    ]
    res1 = agent.run("quero fechar o pedido", session_id=sess_id, cart=cart_payload)
    assert "Resumo do seu Pedido" in res1["reply"]
    assert "859,80" in res1["reply"]
    assert "endereço de entrega está correto" in res1["reply"].lower()

    # 3. Free text address confirmation ("está correto")
    res2 = agent.run("está correto", session_id=sess_id)
    assert "Como você deseja realizar o pagamento" in res2["reply"]
    assert any("PIX" in opt for opt in res2["options"])

    # 4. Free text payment selection ("vou de pix")
    res3 = agent.run("vou de pix", session_id=sess_id)
    assert "Resumo Final do Pedido" in res3["reply"]
    assert "À vista no PIX" in res3["reply"]
    assert "816,81" in res3["reply"]  # 859.80 - 5% = 816.81

    # 5. Free text final order confirmation ("pode confirmar")
    res4 = agent.run("pode confirmar", session_id=sess_id)
    assert "Pedido Confirmado com Sucesso" in res4["reply"]
    assert "/api/pix-qrcode" in res4["reply"]
    assert "Código PIX Copia e Cola" in res4["reply"]
    assert "Lucas" in res4["reply"]


def test_natural_language_free_text_checkout_flow_installments(multi_agent_db: Path) -> None:
    """
    Verify complete natural language conversational flow for credit card installments without buttons:
    1. 'vamos fechar a compra' with synchronized cart items.
    2. 'ta certo pode mandar nesse' for address confirmation.
    3. 'quero parcelar no cartao' triggers installment options table.
    4. 'em 8x' selects 8 installments and presents final review.
    5. 'tudo certo pode finalizar' confirms order.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_nl_installments_free_text"

    # 1. Customer Identification
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # 2. Free text checkout with cart payload
    cart_payload = [
        {"product_id": 1, "name": "Tagima Memphis AC-39 Nylon Natural", "price_brl": 429.90, "effective_price_brl": 429.90, "qty": 2}
    ]
    res1 = agent.run("vamos fechar a compra", session_id=sess_id, cart=cart_payload)
    assert "Resumo do seu Pedido" in res1["reply"]
    assert "endereço de entrega está correto" in res1["reply"].lower()

    # 3. Free text address confirmation ("ta certo")
    res2 = agent.run("ta certo pode mandar nesse", session_id=sess_id)
    assert "Como você deseja realizar o pagamento" in res2["reply"]

    # 4. Free text request for installments ("quero parcelar no cartao")
    res3 = agent.run("quero parcelar no cartão", session_id=sess_id)
    assert "Opções de Parcelamento no Cartão de Crédito sem Juros" in res3["reply"]
    assert "8x" in res3["reply"]

    # 5. Free text installment selection ("em 8x")
    res4 = agent.run("em 8x", session_id=sess_id)
    assert "Resumo Final do Pedido" in res4["reply"]
    assert "8x de R$ 107,47 sem juros" in res4["reply"]
    assert "Podemos confirmar e finalizar o seu pedido?" in res4["reply"]

    # 6. Free text final order confirmation ("tudo certo pode finalizar")
    res5 = agent.run("tudo certo pode finalizar", session_id=sess_id)
    assert "Pedido Confirmado com Sucesso" in res5["reply"]
    assert "8x de R$ 107,47 sem juros" in res5["reply"]
    assert "Lucas" in res5["reply"]


def test_add_to_cart_via_text_without_carousel(multi_agent_db: Path) -> None:
    """
    Verify that typing 'quero 2 Yamaha C40 Nylon Natural' directly adds items to cart
    instead of displaying the product search carousel.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_text_add_cart_no_carousel"

    # Identify user
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # User expresses direct buy/add-to-cart intent via chat text
    res = agent.run("quero 2 Yamaha C40 Nylon Natural", session_id=sess_id)

    assert res["action"] == "ADD_TO_CART"
    assert res["product"]["name"] == "Yamaha C40 Nylon Natural"
    assert res["qty"] == 2
    assert len(res["products"]) == 0  # No carousel returned!
    assert "2x Yamaha C40 Nylon Natural" in res["reply"]
    assert "adicionado ao seu carrinho" in res["reply"]
    assert "carrossel" not in res["reply"].lower()


def test_add_to_cart_and_checkout_flow_all_in_text(multi_agent_db: Path) -> None:
    """
    Test full cycle without touching any buttons:
    1. Identify user via email.
    2. Add product via chat text ("quero 1 Shelby SN-100").
    3. Type "quero fechar o pedido".
    4. Confirm address via text ("sim, endereco correto").
    5. Choose payment method via text ("pix").
    6. Confirm final order via text ("sim, confirmar pedido").
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_text_cycle_full"

    # 1. Identify user
    agent.run("lucas.mendes@jmail.com", session_id=sess_id)

    # 2. Add product via chat text
    res_add = agent.run("quero 1 Shelby SN-100", session_id=sess_id)
    assert res_add["action"] == "ADD_TO_CART"
    assert res_add["qty"] == 1
    assert len(res_add["products"]) == 0

    # 3. Checkout via chat text
    res_chk = agent.run("quero fechar o pedido", session_id=sess_id)
    assert "Resumo do seu Pedido" in res_chk["reply"]
    assert "Shelby SN-100" in res_chk["reply"]
    assert "endereço de entrega está correto" in res_chk["reply"].lower()

    # 4. Address confirmation via text
    res_addr = agent.run("sim, endereco correto", session_id=sess_id)
    assert "Como você deseja realizar o pagamento" in res_addr["reply"]

    # 5. Payment choice via text
    res_pay = agent.run("pagar no pix", session_id=sess_id)
    assert "Resumo Final do Pedido" in res_pay["reply"]
    assert "À vista no PIX" in res_pay["reply"]

    # 6. Final confirmation via text
    res_fin = agent.run("sim, confirmar pedido", session_id=sess_id)
    assert "Pedido Confirmado com Sucesso" in res_fin["reply"]
    assert res_fin["action"] == "CLEAR_CART"


def test_formatted_phone_numbers_identification_and_verification(multi_agent_db: Path) -> None:
    """
    Verify that formatted phone numbers with hyphens, parentheses, and spaces (e.g. '6799123-0123')
    are correctly recognized as phone identification inputs and NOT flagged as low confidence or gibberish.
    """
    agent = AssistantAgent(db_path=multi_agent_db)

    # 1. Registered customer phone with hyphen '6799123-0123' (Fernanda Almeida)
    res1 = agent.run("6799123-0123", session_id="test_fmt_phone_1")
    assert res1["model_used"] == "direct_customer_verification"
    assert res1["active_agent"] == "customer"
    assert "Fernanda Almeida" in res1["reply"]
    assert "guardrail_low_confidence" not in res1["model_used"]

    # 2. Registered customer phone with parentheses and space '(67) 99123-0123'
    res2 = agent.run("(67) 99123-0123", session_id="test_fmt_phone_2")
    assert res2["model_used"] == "direct_customer_verification"
    assert "Fernanda Almeida" in res2["reply"]

    # 3. Registered customer phone without DDD '99123-0123'
    res3 = agent.run("99123-0123", session_id="test_fmt_phone_3")
    assert res3["model_used"] == "direct_customer_verification"
    assert "Fernanda Almeida" in res3["reply"]

    # 4. Unregistered customer phone with hyphen '6799988-7766'
    res4 = agent.run("6799988-7766", session_id="test_fmt_phone_4")
    assert res4["model_used"] == "direct_customer_verification"
    assert "Não localizamos" in res4["reply"] or "novo cliente" in res4["reply"].lower()
    assert "Sim, sou novo cliente" in res4["options"]
    assert "guardrail_low_confidence" not in res4["model_used"]


def test_authenticated_customer_with_no_orders_tracking_inquiry(multi_agent_db: Path) -> None:
    """
    Verify that an authenticated customer without any registered orders (e.g., Fernanda Almeida)
    who asks 'quero saber sobre meu rastreio' is informed clearly that there are no orders/tracking linked to their account.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_fernanda_tracking_suite"

    # Step 1: Customer identification
    res1 = agent.run("6799123-0123", session_id=sess_id)
    assert "Fernanda Almeida" in res1["reply"]

    # Step 2: Customer asks for tracking
    res2 = agent.run("quero saber sobre meu rastreio", session_id=sess_id)
    assert res2["active_agent"] == "support"
    assert res2["model_used"] == "direct_order_tracking"
    assert "não há nenhum pedido ou código de rastreio vinculado" in res2["reply"].lower()
    assert "Fernanda Almeida" in res2["reply"]


def test_authenticated_customer_with_active_order_tracking_inquiry(multi_agent_db: Path) -> None:
    """
    Verify that an authenticated customer with active orders (e.g., Ana Carolina Ferreira)
    who asks 'quero saber sobre meu rastreio' receives complete order details, status, and tracking code.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_ana_tracking_suite"

    # Step 1: Customer identification
    res1 = agent.run("(67) 99654-7821", session_id=sess_id)
    assert "Ana Carolina" in res1["reply"]

    # Step 2: Customer asks for tracking
    res2 = agent.run("quero saber sobre meu rastreio", session_id=sess_id)
    assert res2["active_agent"] == "support"
    assert res2["model_used"] == "direct_order_tracking"
    assert "BRJL5544332BR" in res2["reply"]
    assert "Pedido #8" in res2["reply"]
    assert "Enviado / Em trânsito" in res2["reply"]


def test_authenticated_customer_ver_meus_pedidos_no_orders(multi_agent_db: Path) -> None:
    """
    Verify that Fernanda Almeida (no registered orders) who inputs 'quero ver meus pedidos'
    is routed strictly to SupportAgent, receives a clear message that no orders are linked,
    and NO product carousel or instruments are returned.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_fernanda_meus_pedidos"

    # Step 1: Identification via phone with hyphen
    res1 = agent.run("6799123-0123", session_id=sess_id)
    assert "Fernanda Almeida" in res1["reply"]

    # Step 2: Customer asks 'quero ver meus pedidos'
    res2 = agent.run("quero ver meus pedidos", session_id=sess_id)
    assert res2["active_agent"] == "support"
    assert res2["model_used"] == "direct_order_tracking"
    assert "não há nenhum pedido ou código de rastreio vinculado" in res2["reply"].lower()
    assert "Fernanda Almeida" in res2["reply"]
    assert len(res2["products"]) == 0


def test_authenticated_customer_ver_meus_pedidos_with_orders(multi_agent_db: Path) -> None:
    """
    Verify that Ana Carolina Ferreira (has order #8) who inputs 'quero ver meus pedidos'
    receives the complete breakdown of her orders without any instrument carousel.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_ana_meus_pedidos"

    # Step 1: Identification
    res1 = agent.run("(67) 99654-7821", session_id=sess_id)
    assert "Ana Carolina" in res1["reply"]

    # Step 2: Customer asks 'quero ver meus pedidos'
    res2 = agent.run("quero ver meus pedidos", session_id=sess_id)
    assert res2["active_agent"] == "support"
    assert res2["model_used"] == "direct_order_tracking"
    assert "Pedido #8" in res2["reply"]
    assert "BRJL5544332BR" in res2["reply"]
    assert len(res2["products"]) == 0


def test_intent_router_classifies_order_and_history_queries() -> None:
    """
    Verify that IntentRouter correctly routes all variations of order and history inquiries to support.
    """
    order_queries = [
        "quero ver meus pedidos",
        "ver meus pedidos",
        "meus pedidos",
        "minhas compras",
        "pedidos anteriores",
        "historico de pedidos",
        "onde estao meus pedidos",
        "status do pedido",
        "quero saber sobre meu rastreio",
        "consultar meus pedidos",
        "quando chega meu pedido",
    ]
    for q in order_queries:
        stage = IntentRouter.classify_stage(q)
        assert stage == "support", f"Query '{q}' was classified as '{stage}', expected 'support'"


def test_catalog_query_price_limit_2000_returns_all_products_not_truncated_to_15(multi_agent_db: Path) -> None:
    """
    Verify that querying products up to R$ 2000 returns all matching products (37 items)
    and is not arbitrarily truncated to 15, reaching instruments close to R$ 2000.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_catalog_ceiling_2000"

    # Pre-identify customer
    agent.run("6799123-0123", session_id=sess_id)

    # Ask for products up to R$ 2000
    res = agent.run("quero ver produtos até 2000 reais", session_id=sess_id)
    assert res["active_agent"] == "catalog"
    products = res.get("products", [])
    assert len(products) >= 30, f"Expected at least 30 instruments under R$2000, got {len(products)}"
    for p in products:
        assert p["effective_price_brl"] <= 2000.0
        assert p["stock_quantity"] > 0
        assert p["status"] == "active"

    max_price = max(p["effective_price_brl"] for p in products)
    assert max_price > 1800.0, f"Expected max price close to 2000, got {max_price}"


def test_catalog_pandeiro_query_returns_no_products_and_no_carousel(multi_agent_db: Path) -> None:
    """
    Verify that querying for instruments not sold by the store (e.g. pandeiro):
    1. Routes correctly to catalog specialist (or explains non-availability) without trapping in customer onboarding.
    2. Does NOT return unrelated products or populate the carousel.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_catalog_pandeiro"

    # Pre-identify customer using seed customer phone
    agent.run("6799123-0123", session_id=sess_id)

    # First turn: tem pandeiro?
    res1 = agent.run("tem pandeiro?", session_id=sess_id)
    assert res1["active_agent"] == "catalog"
    assert len(res1.get("products", [])) == 0

    # Second turn: me mande as opções de pandeiro
    res2 = agent.run("me mande as opções de pandeiro", session_id=sess_id)
    assert res2["active_agent"] == "catalog"
    assert len(res2.get("products", [])) == 0


def test_satisfaction_survey_and_attendance_closure_flow(multi_agent_db: Path) -> None:
    """
    Verify complete post-checkout satisfaction survey (CSAT 1-5) and attendance closure:
    1. After order confirmation, system asks for CSAT rating from 1 to 5.
    2. Interactive options provide 1-5 star rating buttons.
    3. User provides rating response.
    4. System registers score, thanks user, and concludes/closes attendance.
    5. Follow-up messages in closed state politely inform attendance was concluded.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_csat_survey_closure"

    # Step 1: Identify customer
    agent.run("6799123-0123", session_id=sess_id)

    # Step 2: Add item to cart
    agent.run("adicionar 1 Yamaha F310 ao carrinho", session_id=sess_id)

    # Step 3: Checkout intent
    agent.run("fechar meu pedido", session_id=sess_id)

    # Step 4: Confirm address
    agent.run("Sim, endereço correto", session_id=sess_id)

    # Step 5: Choose PIX
    agent.run("⚡ Pagar no PIX (5% off)", session_id=sess_id)

    # Step 6: Confirm final order
    res_confirm = agent.run("Sim, confirmar pedido", session_id=sess_id)
    assert "Pedido Confirmado com Sucesso" in res_confirm["reply"]
    assert "Pesquisa de Satisfação" in res_confirm["reply"]
    assert "1 a 5" in res_confirm["reply"]
    assert any("⭐ 5" in opt for opt in res_confirm["options"])
    assert any("⭐ 1" in opt for opt in res_confirm["options"])

    # Step 7: User provides CSAT rating
    res_csat = agent.run("⭐ 5 (Excelente)", session_id=sess_id)
    assert res_csat["model_used"] == "direct_csat_closure"
    assert "Agradecemos muito pela sua avaliação" in res_csat["reply"]
    assert "encerrado com sucesso" in res_csat["reply"]
    assert res_csat["action"] == "CLOSE_CHAT"
    assert "🔄 Novo Atendimento" in res_csat["options"]

    # Step 8: Next casual message in closed state
    res_closed = agent.run("obrigado e tchau", session_id=sess_id)
    assert res_closed["model_used"] == "direct_attendance_closed"
    assert "encerrado com sucesso" in res_closed["reply"]


def test_support_return_and_regret_policy_not_intercepted_by_order_tracking(multi_agent_db: Path) -> None:
    """
    Ensure inquiries regarding return policies, right of regret, exchanges, or defects
    are properly routed and answered with store policies (Section 4 & CDC)
    instead of being intercepted by order tracking just because they mention 'minha compra' or 'meu pedido'.
    """
    agent = AssistantAgent(db_path=multi_agent_db)
    sess_id = "test_support_return_policy_session"

    # Step 1: Identify customer Lucas Mendes da Silva (has delivered Order #4)
    r1 = agent.run("67 99812-3456", session_id=sess_id)
    assert "Lucas Mendes da Silva" in r1["reply"]

    # Step 2: Customer asks about right of regret and return
    r2 = agent.run("Me arrependi da minha compra, posso devolver meu pedido?", session_id=sess_id)
    assert r2["active_agent"] == "support"
    assert r2["model_used"] == "direct_store_policy"
    # Verify policy contents are explained
    assert "7 (sete) dias" in r2["reply"] or "7 dias" in r2["reply"]
    assert "Código de Defesa do Consumidor" in r2["reply"] or "CDC" in r2["reply"]
    assert "Frete de Devolução" in r2["reply"] or "frete de devolução" in r2["reply"]
    assert "reembolso" in r2["reply"].lower()
    assert "embalagem original" in r2["reply"].lower()
    # Identifies his delivered Order #4
    assert "Pedido #4" in r2["reply"]
    assert "Giannini" in r2["reply"]
    # Ensure it's not the generic tracking table
    assert "Status do seu Pedido — Empório da Música" not in r2["reply"]

    # Step 3: Customer asks specifically about order tracking
    r3 = agent.run("quero ver meus pedidos", session_id=sess_id)
    assert r3["active_agent"] == "support"
    assert r3["model_used"] == "direct_order_tracking"
    assert "Status do seu Pedido — Empório da Música" in r3["reply"]
    assert "Pedido #4" in r3["reply"]
















