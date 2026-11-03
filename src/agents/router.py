"""Intent router and stage classifier to dispatch customer queries to specialized agents,
following the Conversational State Machine & Agent Transition architecture.
"""

import re
from typing import Dict, List, Optional
from src.repositories.product_repository import normalize_text


# 1. CATALOG / COMPRAR PATTERNS (SHOPPING / RECOMMENDATIONS)
CATALOG_PATTERNS = [
    # Instrument categories
    r"\bviol[aã]o\b", r"\bviol[oõ]es\b", r"\bguitarra[s]?\b", r"\bbaixo[s]?\b", r"\bcontrabaixo[s]?\b",
    r"\bbateria[s]?\b", r"\bpercuss[aã]o\b", r"\bteclado[s]?\b", r"\bpiano[s]?\b", r"\bukulele[s]?\b",
    r"\bsopro\b", r"\bsax(?:ofone)?\b", r"\btrompete[s]?\b", r"\bflauta[s]?\b", r"\bclarinete[s]?\b",
    r"\bviolino[s]?\b", r"\bvioloncelo[s]?\b",
    # Instrument brands & popular models
    r"\byamaha\b", r"\btakamine\b", r"\bgiannini\b", r"\bfender\b", r"\bibanez\b",
    r"\bshelby\b", r"\bkalani\b", r"\bkala\b", r"\bohana\b", r"\btagima\b", r"\broland\b", r"\bcasio\b",
    r"\bgd20\b", r"\bf310\b", r"\bc40\b", r"\bgf-?3d\b",
    # Catalog queries, pricing & recommendations
    r"\bop[çc][õo]es\s+(?:de\s+)?",
    r"\bmodelos?\s+(?:de\s+)?",
    r"\bcat[áa]logo\b",
    r"\binstrumentos?\b",
    r"\brecomenda(?:[çc][ãa]o|r)?\b",
    r"\bquanto\s+custa\b",
    r"\bqual\s+(?:[ée]\s+o\s+)?pre[çc]o\b",
    r"\bpre[çc]o[s]?\b",
    r"\bvalor(?:es)?\b",
    r"\bcustando\b",
    r"\bfaixa\s+de\s+pre[çc]o\b",
    r"\bat[ée]\s+r?\$?\s*\d+",
    r"\bmenos\s+de\s+r?\$?\s*\d+",
    r"\bmais\s+(?:barato|caro)\b",
    r"\bpromo[çc][ãa]o\b", r"\bpromo[çc][õo]es\b", r"\bem\s+promo[çc][ãa]o\b",
    r"\bestoque\b", r"\bdispon[íi]ve(?:l|is)\b",
    r"\btem\s+(?:em\s+estoque|dispon[íi]vel)\b",
    r"\bvoc[eê]s?\s+t[eê]m\b",
    r"\bquero\s+ver\b(?!\s+(?:meus?\s+)?(?:pedidos?|compras?|rastreio|carrinho|cadastro|endere[çc]o|dados|termos))",
    r"\bquero\s+olhar\b(?!\s+(?:meus?\s+)?(?:pedidos?|compras?|rastreio|carrinho|cadastro|endere[çc]o|dados|termos))",
    r"\bver\s+ukuleles\b",
    r"\bviol[õo]es\s+at[ée]\s+r?\$?\s*1000\b",
    r"\bprodutos\s+em\s+promo[çc][ãa]o\b",
    r"\bpre[çc]o\s+takamine\b",
    r"\bcordas?\s+de\s+(?:a[çc]o|nylon)\b",
    r"\bac[úu]stico|eletroac[úu]stico\b",
    r"\bdetalhes\s+do\b",
    r"\bespecifica[çc][õo]es\b",
]

# 2. CHECKOUT PATTERNS (CHECKOUT / CART)
CHECKOUT_PATTERNS = [
    r"\bcarrinho\b",
    r"\bver\s+(?:o\s+)?carrinho\b",
    r"\bmeu\s+carrinho\b",
    r"\bitens\s+(?:no|do)\s+carrinho\b",
    r"\bo\s+que\s+tem\s+no\s+carrinho\b",
    r"\bmostrar\s+carrinho\b",
    r"\besvaziar\s+carrinho\b",
    r"\blimpar\s+carrinho\b",
    r"\bfechar\s+(?:o\s+|meu\s+|o\s+meu\s+)?pedido\b",
    r"\bfinalizar\s+(?:a\s+)?compra\b",
    r"\bfinalizar\s+(?:o\s+|meu\s+)?pedido\b",
    r"\bconcluir\s+(?:a\s+)?compra\b",
    r"\bconcluir\s+(?:o\s+|meu\s+)?pedido\b",
    r"\bfechar\s+(?:a\s+)?compra\b",
    r"\bfechar\s+(?:o\s+)?carrinho\b",
    r"\bcheckout\b",
    r"\bquero\s+(?:fechar|finalizar|comprar|pagar)\b",
    r"\bvamos\s+(?:fechar|finalizar|pagar)\b",
    r"\bir\s+para\s+(?:o\s+)?pagamento\b",
    r"\bselecionei\s+no\s+carrinho\b",
    r"\bcomo\s+fa[çc]o\s+para\s+pagar\b",
    r"\bformas?\s+de\s+pagamento\b",
    r"\bop[çc][õo]es\s+de\s+pagamento\b",
    r"\bparcelamento\b", r"\bparcelar\b", r"\bquantas\s+vezes\b",
    r"\bsimular\s+pagamento\b",
    r"\bcalcular\s+frete\b", r"\bquanto\s+fica\s+o\s+frete\b",
    r"\bdesconto\s+(?:no\s+)?pix\b",
    r"\bquero\s+\d+\s+x\b",
    r"\bquero\s+\d+\s+(?:unidade|unidades)\b",
    r"\b(?:pagar\s+no\s+|no\s+)?pix\b",
    r"\b(?:pagar\s+no\s+|no\s+)?cart[aã]o(?:\s+de\s+cr[ée]dito)?\b",
    r"\b(?:pagar\s+no\s+|no\s+)?boleto(?:\s+banc[áa]rio)?\b",
    r"\b(?:pagar\s+no\s+|no\s+)?cart[aã]o\s+de\s+d[ée]bito\b",
    r"\b(?:pagar\s+em\s+|no\s+)?d[ée]bito\b",
    r"\bpagamento\s+combinado\b",
    r"\bcombinar\s+pagamento\b",
    r"\bendere[çc]o\s+correto\b",
    r"\bsim,?\s*endere[çc]o\s+correto\b",
    r"\bn[ãa]o,?\s*alterar\s+endere[çc]o\b",
    r"\bmanter\s+(?:este\s+)?endere[çc]o\b",
    r"\bconfirmar\s+pedido\b",
    r"\bsim,?\s*confirmar\s+pedido\b",
    r"\bpode\s+confirmar\b",
]

# 3. CUSTOMER / ONBOARDING / CEP PATTERNS (VALIDATION / REGISTRATION)
CUSTOMER_PATTERNS = [
    r"\baceito(?:\s+os\s+termos)?\b",
    r"\bconcordo\b",
    r"\btermo(?:s)?(?:\s+de\s+aceite)?\b",
    r"\btentar\s+novamente\b",
    r"\btentar\s+outro\s+dado\b",
    r"\b(?:fazer|novo|meu|quero(?:\s+me)?)\s+cadastro\b",
    r"\bme\s+cadastr(?:ar|e)\b",
    r"\bcadastr(?:ar|e)\b",
    r"\b(?:sim,?\s*)?(?:sou\s+)?novo\s+cliente\b",
    r"\bsou\s+novo(?:\s+aqui)?\b",
    r"\batualizar\s+(?:meu\s+)?endere[çc]o\b",
    r"\balterar\s+(?:meu\s+)?endere[çc]o\b",
    r"\bmudar\s+(?:meu\s+)?endere[çc]o\b",
    r"\btrocar\s+(?:meu\s+)?endere[çc]o\b",
    r"\bnovo\s+endere[çc]o\b",
    r"\bmeu\s+cep\s+[ée]\b",
    r"\bcep\s+\d{5}[-\s]?\d{3}\b",
    r"^\s*\d{5}[-\s]?\d{3}\s*$",  # Standalone CEP
    r"\[dados_do_cadastro\]",
    r"\[dados_do_formulario\]",
    r"\[dados_do_endereco\]",
]

# 4. SUPPORT / ORDER TRACKING / POLICIES PATTERNS (ORDER TRACKING / POLICIES)
SUPPORT_PATTERNS = [
    # Order Tracking & Status
    r"\brastre(?:io|ar|amento)\b",
    r"\bmeu[s]?\s+rastreio[s]?\b",
    r"\bc[óo]digo\s+de\s+rastre(?:io|amento)\b",
    r"\bstatus\s+(?:do[s]?\s+|de\s+)?pedido[s]?\b",
    r"\bstatus\s+(?:da[s]?\s+)?entrega[s]?\b",
    r"\bmeu[s]?\s+pedido[s]?\b",
    r"\bver\s+(?:meu[s]?\s+)?pedido[s]?\b",
    r"\bconsultar\s+(?:meu[s]?\s+)?pedido[s]?\b",
    r"\bacompanhar\s+(?:meu[s]?\s+)?pedido[s]?\b",
    r"\bacompanhamento\s+(?:do[s]?\s+|de\s+)?pedido[s]?\b",
    r"\bonde\s+est[áa](?:o)?\s+(?:o[s]?\s+|meu[s]?\s+)?pedido[s]?\b",
    r"\bcad[eê]\s+(?:o[s]?\s+|meu[s]?\s+)?pedido[s]?\b",
    r"\bquando\s+chega(?:\s+meu\s+pedido)?\b",
    r"\bprevis[ãa]o\s+de\s+entrega\b",
    r"\bmeu[s]?\s+pedido[s]?\s+j[áa]\s+saiu\b",
    r"\bminha[s]?\s+compra[s]?\b",
    r"\bcompras?\s+anteriore[s]?\b",
    r"\bpedidos?\s+anteriore[s]?\b",
    r"\bhist[óo]rico\s+(?:de\s+)?(?:pedidos|compras)\b",
    r"\bpedido\s+#?\d+\b",
    r"^\s*pedidos\s*$",
    r"^\s*meus\s+pedidos\s*$",
    r"^\s*rastreio\s*$",
    r"^\s*meu\s+rastreio\s*$",
    # Store Policies, Returns & Customer Service
    r"\btroca(?:r)?\b",
    r"\bdevolu[çc][ãa]o\b", r"\bdevolver\b",
    r"\bgarantia\b",
    r"\bdefeito\b",
    r"\barrependimento\b",
    r"\bcancelar\s+pedido\b",
    r"\bhor[áa]rio\s+(?:de\s+funcionamento|da\s+loja)?\b",
    r"\bendere[çc]o\s+(?:f[íi]sico|da\s+loja)\b",
    r"\bonde\s+fica\s+(?:a\s+loja)?\b",
    r"\bcnpj\b",
    r"\bpol[íi]tica\b",
    r"\bwhatsapp\s+da\s+loja\b",
    r"\btelefone\s+da\s+loja\b",
]


class IntentRouter:
    """
    Classifies user turn intent and manages transitions between specialist agents,
    following state transition rules and intent routing logic.
    """

    @staticmethod
    def detect_explicit_intent(norm_msg: str) -> Optional[str]:
        """Detect explicit intent from user message keywords."""
        # 1. Checkout intent check (takes precedence when user explicitly wants to pay/close order)
        for pat in CHECKOUT_PATTERNS:
            if re.search(pat, norm_msg):
                return "checkout"

        # 2. Customer registration / CEP check
        for pat in CUSTOMER_PATTERNS:
            if re.search(pat, norm_msg):
                return "customer"

        # 3. Support / Order tracking / Store policies check
        for pat in SUPPORT_PATTERNS:
            if re.search(pat, norm_msg):
                return "support"

        # 4. Catalog / Product search / Instrument pricing check
        for pat in CATALOG_PATTERNS:
            if re.search(pat, norm_msg):
                return "catalog"

        return None

    @staticmethod
    def get_last_active_agent(chat_history: Optional[List[Dict[str, str]]]) -> Optional[str]:
        """Extract the active agent from the most recent assistant message in chat history."""
        if not chat_history:
            return None
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                metadata = msg.get("metadata") or {}
                if isinstance(metadata, dict) and metadata.get("active_agent"):
                    return str(metadata["active_agent"])
        return None

    @classmethod
    def classify_stage(
        cls,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Classifies current conversation stage and manages transitions between agents:
        - 'checkout': Closing order, cart checkout, payment/shipping options
        - 'customer': Customer onboarding, registration, CEP and address
        - 'support': Order tracking, returns, warranty, store address/hours
        - 'catalog': Searching or comparing musical instruments (default)
        """
        norm_msg = normalize_text(user_message).strip()

        # Step 1: Detect explicit user intent from message
        explicit_intent = cls.detect_explicit_intent(norm_msg)
        if explicit_intent:
            return explicit_intent

        # Step 2: Contextual follow-up / Slot Filling (State Continuation)
        if chat_history:
            last_assistant_msg = ""
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = normalize_text(msg.get("content") or "")
                    break

            last_agent = cls.get_last_active_agent(chat_history)

            # A0. New Customer Confirmation:
            # If assistant asked whether the user is a new customer ("novo cliente" in last_assistant_msg):
            if "novo cliente" in last_assistant_msg or "novo por aqui" in last_assistant_msg or "abrir o formulario de cadastro" in last_assistant_msg:
                if norm_msg in ("sim", "sou", "sou novo", "sou novo cliente", "quero", "claro", "fazer cadastro", "cadastrar") or "sim" in norm_msg.split():
                    return "customer"

            # A. Address Update Request:
            if any(term in norm_msg for term in ["atualizar endereco", "mudar endereco", "alterar endereco", "trocar endereco", "novo endereco"]):
                return "customer"

            # B. Customer Registration Request:
            if any(term in norm_msg for term in ["fazer cadastro", "me cadastrar", "novo cadastro", "cadastrar", "quero me cadastrar"]):
                return "customer"

            # C. Support Authentication Continuation:
            # If assistant asked for phone/email to authenticate an order tracking request
            if ("rastreio" in last_assistant_msg or "pedido" in last_assistant_msg) and (
                "telefone" in last_assistant_msg or "e-mail" in last_assistant_msg or "confirmar" in last_assistant_msg
            ) and last_agent == "support":
                digits = re.sub(r"\D", "", norm_msg)
                if len(digits) >= 8 or "@" in norm_msg or len(norm_msg.split()) <= 4:
                    return "support"

            # D. Checkout Closing Continuation:
            # If assistant was waiting for confirmation, user email/phone, or closing data in checkout
            if last_agent == "checkout" or (
                ("fechar" in last_assistant_msg or "carrinho" in last_assistant_msg or "pagamento" in last_assistant_msg)
                and ("confirmar" in last_assistant_msg or "qual" in last_assistant_msg or "dados" in last_assistant_msg)
            ):
                return "checkout"

            # E. Customer Registration Continuation:
            if last_agent == "customer":
                return "customer"

            # F. Support Continuation:
            if last_agent == "support" and len(norm_msg.split()) <= 5:
                return "support"

        # Step 3: Default to Catalog Specialist for instrument exploration & general assistance
        return "catalog"
