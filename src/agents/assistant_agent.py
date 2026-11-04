"""Multi-Agent Orchestrator and Specialist Dispatcher for Empório da Música."""

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
from groq import Groq

from src.agents.prompts import (
    LOW_CONFIDENCE_TEMPLATES,
    SYSTEM_PROMPT,
    detect_user_language,
)
from src.agents.router import IntentRouter, is_human_handoff_intent
from src.agents.specialists import (
    CatalogAgent,
    CheckoutAgent,
    CustomerAgent,
    SupportAgent,
    clean_thinking_tags,
    make_llm_tool_payload,
)
from src.agents.tools import TOOL_SCHEMAS, ToolExecutor, is_name_match
from src.config.settings import settings
from src.repositories.failed_intent_repository import FailedIntentRepository
from src.repositories.product_repository import normalize_text
from src.services.pix_service import PixService


def is_checkout_intent(text: str) -> bool:
    """Detect if user intends to finalize purchase, checkout, or close their order."""
    t = (text or "").lower().strip()
    checkout_patterns = [
        r"\bfechar\s+(?:o\s+)?pedido\b",
        r"\bfinalizar\s+(?:a\s+)?compra\b",
        r"\bfinalizar\s+(?:o\s+)?pedido\b",
        r"\bconcluir\s+(?:a\s+)?compra\b",
        r"\bconcluir\s+(?:o\s+)?pedido\b",
        r"\bcheckout\b",
        r"\bquero\s+(?:pagar|fechar)\b",
        r"\bir\s+para\s+o\s+pagamento\b",
        r"\bselecionei\s+no\s+carrinho\b",
    ]
    return any(re.search(p, t) for p in checkout_patterns)


def extract_claimed_customer_name(messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract any customer name claimed by the user across conversation turns.
    Scans recent messages for introductions, greetings, and standalone names.
    """
    prepositions = {"de", "da", "do", "das", "dos", "e"}

    # 1. Search user messages for self-identification patterns
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        match = re.search(
            r"(?:meu nome [ée]|me chamo|sou(?:\s+[oa])?|aqui [ée]|my name is|i am|mi nombre es|soy)\s+([A-Za-zÀ-ÖØ-öø-ÿ\s]+?)(?:[,\.\!\?]|\s+gostaria|\s+quero|\s+e\s+|\s+meu|\s+minha|\s+por favor|$)",
            content,
            re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip()
            tokens = [t for t in candidate.split() if len(t) > 1 and t.lower() not in prepositions]
            if tokens:
                return candidate.strip()

        # Standalone name check: 2 to 5 words, letters only, capitalized
        words = content.split()
        if 2 <= len(words) <= 5:
            clean_words = [re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", w) for w in words]
            if all(len(w) >= 2 and w[0].isupper() for w in clean_words):
                norm = normalize_text(content)
                common_terms = {
                    "violao", "guitarra", "ukulele", "baixo", "bateria", "teclado",
                    "bom dia", "boa tarde", "boa noite", "cartao", "pix", "frete",
                    "loja", "emporio",
                }
                if not any(term in norm for term in common_terms):
                    return " ".join(clean_words)

    # 2. Check assistant greetings addressing the user by name (e.g. "Olá, Camila! Tudo bem?")
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        match = re.search(r"(?:[Oo]l[áa]|[Oo]i),?\s+([A-ZÀ-Ö][a-zà-öø-ÿ]+)[!\,\.]", content)
        if match:
            greeted_name = match.group(1).strip()
            if greeted_name.lower() not in {"amigo", "amiga", "cliente", "tudo", "pessoal"}:
                return greeted_name

    return None


class AssistantAgent:
    """
    Multi-Agent Orchestrator for Empório da Música.
    Evaluates input clarity/confidence, determines stage via IntentRouter,
    and dispatches to specialized agents with single responsibility:
      - CatalogAgent: Instrument search, specs, stock, and recommendations (returns Product Carousel).
      - CheckoutAgent: Order closure, payment calculations (PIX discount, installments), shipping simulation.
      - CustomerAgent: Customer onboarding, address registration, and ViaCEP auto-enrichment.
      - SupportAgent: Order tracking, return policies, warranty, and store hours.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        pdf_path: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.pdf_path = pdf_path
        self.api_key = api_key or settings.groq_api_key
        self._client = Groq(api_key=self.api_key) if self.api_key else None
        self.tool_executor = ToolExecutor(db_path=db_path, pdf_path=pdf_path)
        self.tools = self.tool_executor
        self.failed_intent_repo = FailedIntentRepository(db_path=db_path)

        # Ordered model chain: Qwen3 primary -> Llama fallbacks
        self.model_chain = [
            settings.primary_model,
            settings.fallback_model_1,
            settings.fallback_model_2,
        ]

        # Initialize domain specialist agents
        self.catalog_agent = CatalogAgent(
            tool_executor=self.tool_executor,
            client=self._client,
            model_chain=self.model_chain,
        )
        self.checkout_agent = CheckoutAgent(
            tool_executor=self.tool_executor,
            client=self._client,
            model_chain=self.model_chain,
        )
        self.customer_agent = CustomerAgent(
            tool_executor=self.tool_executor,
            client=self._client,
            model_chain=self.model_chain,
        )
        self.support_agent = SupportAgent(
            tool_executor=self.tool_executor,
            client=self._client,
            model_chain=self.model_chain,
        )

        self.specialists: Dict[str, Any] = {
            "catalog": self.catalog_agent,
            "checkout": self.checkout_agent,
            "customer": self.customer_agent,
            "support": self.support_agent,
        }
        self.sessions: Dict[str, Any] = {}

    @property
    def client(self) -> Optional[Groq]:
        return self._client

    @client.setter
    def client(self, new_client: Optional[Groq]) -> None:
        self._client = new_client
        if hasattr(self, "specialists"):
            for s in self.specialists.values():
                s.client = new_client

    def evaluate_clarity_and_confidence(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate whether the user message is intelligible, ambiguous, or clear.
        Low-confidence or gibberish inputs are intercepted at low temperature and logged
        to `failed_intents` to continuously improve NLU accuracy.
        """
        lang = detect_user_language(user_message)
        stripped = (user_message or "").strip()

        # Single digit response (e.g. CSAT rating 1-5, interactive menu options 1, 2, 3)
        alnum_chars = re.sub(r"[^a-zA-Z0-9]", "", stripped)
        if stripped in ("1", "2", "3", "4", "5") or (len(alnum_chars) == 1 and alnum_chars.isdigit()):
            return {
                "is_clear": True,
                "confidence": 1.0,
                "language": lang,
            }

        # Installment / quantity shorthand (e.g. "12x", "10x", "5x", "1x")
        if re.search(r"^\s*\d+\s*x\s*$", stripped, re.IGNORECASE):
            return {
                "is_clear": True,
                "confidence": 1.0,
                "language": lang,
            }

        # 1. Empty or ultra-short non-word noise
        if len(alnum_chars) < 2:
            return {
                "is_clear": False,
                "reason": "too_short_or_empty",
                "confidence": 0.10,
                "language": lang,
                "strategy": "ask_rephrase_with_options",
            }

        # 2. Gibberish / keyboard smash heuristic (no vowels or random symbol clusters)
        norm = normalize_text(stripped)
        words = re.findall(r"[a-z]+", norm)
        vowels = set("aeiou")
        digits = re.sub(r"\D", "", stripped)
        if words:
            vowel_words = [w for w in words if any(ch in vowels for ch in w)]
            total_vowels_first = sum(1 for ch in words[0] if ch in vowels)
            if len(vowel_words) == 0 or (
                len(words) == 1 and len(words[0]) > 6 and total_vowels_first <= 1
            ):
                return {
                    "is_clear": False,
                    "reason": "gibberish_or_unintelligible",
                    "confidence": 0.15,
                    "language": lang,
                    "strategy": "ask_rephrase_with_options",
                }
        elif not digits:
            return {
                "is_clear": False,
                "reason": "no_recognizable_words",
                "confidence": 0.15,
                "language": lang,
                "strategy": "ask_rephrase_with_options",
            }

        # 3. Context-free vague utterances when there is no prior conversation history
        vague_phrases = {
            "isso", "aquilo", "aquele", "negocio", "coisa", "quero", "preciso",
            "ajuda", "duvida", "nao sei", "hm", "hmm", "talvez", "qualquer um",
            "that thing", "something", "whatever", "idk", "eso", "aquello",
        }
        if not chat_history and norm in vague_phrases:
            return {
                "is_clear": False,
                "reason": "vague_without_context",
                "confidence": 0.35,
                "language": lang,
                "strategy": "disambiguation_menu",
            }

        return {
            "is_clear": True,
            "reason": "clear",
            "confidence": 0.92,
            "language": lang,
            "strategy": "proceed_to_agent",
        }

    def _call_llm_with_fallbacks(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> tuple[Any, str]:
        """Direct fallback caller used by orchestrator and legacy tests."""
        return self.catalog_agent._call_llm_with_fallbacks(
            messages=messages, tools=tools, temperature=temperature
        )

    def clear_session_state(self, session_id: str) -> None:
        """Clear cached session flow state when resetting conversation."""
        if hasattr(self, "sessions") and session_id in self.sessions:
            del self.sessions[session_id]

    def _get_session(
        self,
        session_id: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Retrieve or initialize in-memory session flow state with chat_history recovery."""
        if not hasattr(self, "sessions"):
            self.sessions = {}
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "stage": "IDENTIFY",
                "customer": None,
                "identifier_tried": None,
                "terms_accepted": False,
                "pending_customer": None,
                "payment_method": None,
                "cart_items": [],
                "cart_total": 0.0,
                "has_promo": False,
            }
        sess = self.sessions[session_id]

        # Context recovery from recent messages if customer was identified earlier
        if not sess["customer"] and chat_history:
            for msg in reversed(chat_history):
                c = msg.get("content", "")
                e_m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", c)
                if e_m:
                    matched = self.tool_executor.customer_repo.find_customers(e_m.group(0))
                    if matched:
                        sess["customer"] = matched[0]
                        sess["stage"] = "SHOPPING"
                        break
                d_m = re.sub(r"\D", "", c)
                if 8 <= len(d_m) <= 12 and len(c.strip().split()) <= 4:
                    matched = self.tool_executor.customer_repo.find_customers(d_m)
                    if matched:
                        sess["customer"] = matched[0]
                        sess["stage"] = "SHOPPING"
                        break
                if "Cadastro e endereço de entrega" in c:
                    name_m = re.search(r"\*\*([A-Za-zÀ-ÖØ-öø-ÿ\s]+?)\*\*", c)
                    if name_m:
                        matched = self.tool_executor.customer_repo.find_customers(name_m.group(1))
                        if matched:
                            sess["customer"] = matched[0]
                            sess["stage"] = "SHOPPING"
                            break

        return sess

    def _build_final_checkout_summary(
        self,
        sess: Dict[str, Any],
        detected_lang: str,
    ) -> Dict[str, Any]:
        """Build and return the comprehensive final order breakdown with payment options and shipping."""
        found_items = sess.get("cart_items") or []
        total = float(sess.get("cart_total") or 0.0)
        has_promo = bool(sess.get("has_promo", False))
        cust = sess.get("customer") or {}

        cust_name = cust.get("name", "Cliente")
        cust_phone = cust.get("phone", "")
        city = cust.get("city", "Campo Grande")
        address = cust.get("address", "Endereço cadastrado")
        cep = cust.get("cep", "")

        def _fmt(val: float) -> str:
            return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # Shipping rules
        is_campo_grande = "campo grande" in city.lower()
        if is_campo_grande:
            if total >= 300.0 or total == 0.0:
                shipping_cost = 0.0
                shipping_text = (
                    "• 🚚 **Custo de Frete:** **Grátis** para Campo Grande/MS via motoboy próprio da loja\n"
                    "• ⏱️ **Prazo de Entrega:** 1 a 3 dias úteis (com agendamento telefônico prévio)"
                )
            else:
                shipping_cost = 25.0
                shipping_text = (
                    "• 🚚 **Custo de Frete:** R$ 25,00 para Campo Grande/MS via motoboy próprio da loja\n"
                    "• ⏱️ **Prazo de Entrega:** 1 a 3 dias úteis (com agendamento telefônico prévio)"
                )
        else:
            shipping_cost = 45.0
            shipping_text = (
                f"• 🚚 **Custo de Frete:** R$ 45,00 para {city} via transportadora parceira\n"
                "• ⏱️ **Prazo de Entrega:** 3 a 7 dias úteis (com código de rastreamento)"
            )

        total_regular = total + shipping_cost
        discount_val = (total * 0.05) if not has_promo else 0.0
        total_pix = (total - discount_val) + shipping_cost

        # Calculate max installments and valid installment options according to store policy:
        # 1x-3x: min R$50 | 4x-6x: min R$80 | 7x-12x: min R$100
        max_inst = 1
        valid_insts = []
        for n in range(1, 13):
            val = round(total_regular / n, 2)
            if n == 1:
                ok = True
            elif 2 <= n <= 3:
                ok = val >= 50.0
            elif 4 <= n <= 6:
                ok = val >= 80.0
            else:
                ok = val >= 100.0
            if ok:
                max_inst = n
                valid_insts.append((n, val))

        inst_val = total_regular / max_inst

        sess["max_inst"] = max_inst
        sess["valid_insts"] = valid_insts
        sess["total_final"] = total_regular
        sess["total_pix"] = total_pix
        sess["shipping_cost"] = shipping_cost

        pay_lines = [
            f"• **Subtotal dos Produtos:** {_fmt(total)}",
        ]
        if shipping_cost > 0:
            pay_lines.append(f"• **Custo de Frete:** {_fmt(shipping_cost)}")
        else:
            pay_lines.append("• **Frete:** Grátis")
        pay_lines.append(f"• **Valor Final a Pagar:** **{_fmt(total_regular)}**\n")

        pay_lines.append("💳 **Opções de Pagamento:**")
        if discount_val > 0:
            pay_lines.append(f"• ⚡ **À vista no PIX (5% de desconto):** **{_fmt(total_pix)}** (economia de {_fmt(discount_val)})")
        else:
            pay_lines.append(f"• ⚡ **À vista no PIX:** **{_fmt(total_pix)}**")
        pay_lines.append(f"• 💳 **Cartão de Crédito à vista (1x):** **{_fmt(total_regular)}**")
        if max_inst > 1:
            pay_lines.append(f"• 💳 **Cartão de Crédito Parcelado:** em até **{max_inst}x de {_fmt(inst_val)} sem juros** (ou 1x de {_fmt(total_regular)})")
        pay_lines.append(f"• 📄 **Boleto Bancário à vista:** **{_fmt(total_regular)}** (Prazo de Compensação: até 3 dias úteis)")
        pay_lines.append(f"• 💳 **Cartão de Débito à vista:** **{_fmt(total_regular)}**")
        if total_regular >= 2000.0:
            half_pix = total * 0.5
            half_card = total_regular - half_pix
            pay_lines.append(f"• 🔀 **Pagamento Combinado:** Entrada no PIX ({_fmt(half_pix)}) + saldo no Cartão ({_fmt(half_card)} sem juros)")
        pay_lines.append("\n🔒 *(Para fins de simulação e segurança, não é necessário digitar os dados do seu cartão).*")
        pay_lines.append("📱 *(O QR Code PIX ou dados de pagamento serão gerados logo após a escolha da opção).*")

        lines = [
            "🎉 **Pedido Pronto para Finalização — Resumo do Pedido** 🎶✨\n",
        ]
        if found_items:
            lines.append("| Item | Quantidade | Valor Unitário | Subtotal |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for p, qty in found_items:
                u_p = float(p.get("effective_price_brl") or p.get("price_brl", 0.0))
                lines.append(f"| {p['name']} | {qty} | {_fmt(u_p)} | {_fmt(u_p * qty)} |")
            tot_q = sum(q for _, q in found_items)
            lines.append(f"| **Subtotal dos Produtos** | **{tot_q} item(ns)** | | **{_fmt(total)}** |\n")

        lines.append("📍 **Endereço de entrega confirmado e prazo de envio:**")
        lines.append(f"• **Destinatário:** {cust_name}")
        lines.append(f"• **Endereço:** {address} - {city}" + (f" (CEP: {cep})" if cep else ""))
        lines.append(shipping_text)
        lines.append("\n💳 **Valores e Pagamento:**")
        lines.extend(pay_lines)
        lines.append(
            "\n---\n"
            f"**{cust_name}**, por favor confira todos os valores, o frete e o prazo de entrega acima.\n"
            "**Como você deseja realizar o pagamento para finalizar o seu pedido?** Escolha uma opção à vista ou clique para parcelar no cartão:"
        )

        sess["stage"] = "CHECKOUT_PAYMENT_SELECT"

        options = [
            "⚡ Pagar no PIX (5% off)" if discount_val > 0 else "⚡ Pagar no PIX",
            "💳 Pagar à vista no Cartão (1x)",
        ]
        if max_inst > 1:
            options.append(f"💳 Parcelar em até {max_inst}x sem juros")
        else:
            options.append("💳 Parcelar no Cartão")
        options.extend([
            "📄 Boleto Bancário",
            "💳 Cartão de Débito",
        ])
        if total_regular >= 2000.0:
            options.append("🔀 Pagamento Combinado")

        return {
            "reply": "\n".join(lines),
            "language": detected_lang,
            "confidence": 1.0,
            "model_used": "direct_checkout_final_summary",
            "tools_called": [],
            "products": [],
            "active_agent": "checkout",
            "action": None,
            "options": options,
            "logged_failed_intent": False,
        }

    def _build_final_order_review(
        self,
        sess: Dict[str, Any],
        detected_lang: str,
        method: str,
        installments: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build and return the comprehensive final order review with the customer's chosen payment method
        and conditions, requesting final confirmation before placing the order."""
        found_items = sess.get("cart_items") or []
        total = float(sess.get("cart_total") or 0.0)
        has_promo = bool(sess.get("has_promo", False))
        cust = sess.get("customer") or {}

        cust_name = cust.get("name", "Cliente")
        cust_phone = cust.get("phone", "")
        city = cust.get("city", "Campo Grande")
        address = cust.get("address", "Endereço cadastrado")
        cep = cust.get("cep", "")

        def _fmt(val: float) -> str:
            return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # Shipping calculation
        is_campo_grande = "campo grande" in city.lower()
        if is_campo_grande:
            if total >= 300.0 or total == 0.0:
                shipping_cost = 0.0
                shipping_text = (
                    "• 🚚 **Custo de Frete:** **Grátis** para Campo Grande/MS via motoboy próprio da loja\n"
                    "• ⏱️ **Prazo de Entrega:** 1 a 3 dias úteis (com agendamento telefônico prévio)"
                )
            else:
                shipping_cost = 25.0
                shipping_text = (
                    "• 🚚 **Custo de Frete:** R$ 25,00 para Campo Grande/MS via motoboy próprio da loja\n"
                    "• ⏱️ **Prazo de Entrega:** 1 a 3 dias úteis (com agendamento telefônico prévio)"
                )
        else:
            shipping_cost = 45.0
            shipping_text = (
                f"• 🚚 **Custo de Frete:** R$ 45,00 para {city} via transportadora parceira\n"
                "• ⏱️ **Prazo de Entrega:** 3 a 7 dias úteis (com código de rastreamento)"
            )

        total_regular = total + shipping_cost
        discount_val = (total * 0.05) if not has_promo else 0.0
        total_pix = (total - discount_val) + shipping_cost

        sess["payment_method"] = method
        if installments is not None:
            safe_inst = min(max(1, int(installments)), 12)
            if sess.get("max_inst"):
                safe_inst = min(safe_inst, int(sess["max_inst"]))
            sess["installments"] = safe_inst
        elif method == "credit_card" and not sess.get("installments"):
            sess["installments"] = 1

        sess["stage"] = "CHECKOUT_FINAL_CONFIRM"

        lines = [
            "📋 **Resumo Final do Pedido — Confirmação** 🎸✨\n",
        ]

        if found_items:
            lines.append("| Item | Quantidade | Valor Unitário | Subtotal |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for p, qty in found_items:
                u_p = float(p.get("effective_price_brl") or p.get("price_brl", 0.0))
                lines.append(f"| {p['name']} | {qty} | {_fmt(u_p)} | {_fmt(u_p * qty)} |")
            tot_q = sum(q for _, q in found_items)
            lines.append(f"| **Subtotal dos Produtos** | **{tot_q} item(ns)** | | **{_fmt(total)}** |\n")

        lines.append("📍 **Endereço de entrega confirmado e prazo de envio:**")
        lines.append(f"• **Destinatário:** {cust_name}")
        lines.append(f"• **Endereço:** {address} - {city}" + (f" (CEP: {cep})" if cep else ""))
        lines.append(shipping_text)

        lines.append("\n💳 **Valores e Forma de Pagamento Escolhida:**")
        lines.append(f"• **Subtotal dos Produtos:** {_fmt(total)}")
        if shipping_cost > 0:
            lines.append(f"• **Custo de Frete:** {_fmt(shipping_cost)}")
        else:
            lines.append("• **Frete:** Grátis")

        if method == "pix":
            lines.append("• **Método Escolhido:** ⚡ À vista no PIX (5% de desconto)")
            if discount_val > 0:
                lines.append(f"• **Desconto PIX:** -{_fmt(discount_val)}")
            lines.append(f"• **Valor Final a Pagar:** **{_fmt(total_pix)}**\n")
            lines.append("📱 *(O QR Code PIX dinâmico e a chave Copia e Cola serão exibidos na confirmação final).*")
        elif method == "credit_card":
            inst_n = sess.get("installments", 1)
            inst_val = total_regular / inst_n
            lines.append("• **Método Escolhido:** 💳 Cartão de Crédito sem juros")
            if inst_n > 1:
                lines.append(f"• **Condição de Pagamento:** **{inst_n}x de {_fmt(inst_val)} sem juros** (Total: {_fmt(total_regular)})")
            else:
                lines.append(f"• **Condição de Pagamento:** **1x de {_fmt(total_regular)} à vista**")
            lines.append(f"• **Valor Final a Pagar:** **{_fmt(total_regular)}**\n")
            lines.append("🔒 *(Para fins de simulação e segurança, não é necessário digitar os dados confidenciais do seu cartão).*")
        elif method == "boleto":
            lines.append("• **Método Escolhido:** 📄 Boleto Bancário à vista")
            lines.append(f"• **Valor Final a Pagar:** **{_fmt(total_regular)}**")
            lines.append("• **Prazo de Compensação:** Até 3 dias úteis\n")
            lines.append("🔢 *(A linha digitável do boleto será gerada na confirmação final).*")
        elif method == "debit_card":
            lines.append("• **Método Escolhido:** 💳 Cartão de Débito à vista")
            lines.append(f"• **Valor Final a Pagar:** **{_fmt(total_regular)}**\n")
            lines.append("🔒 *(Para fins de simulação e segurança, não é necessário digitar os dados confidenciais do seu cartão).*")
        elif method == "combined":
            half_pix = total * 0.5
            half_card = total_regular - half_pix
            lines.append("• **Método Escolhido:** 🔀 Pagamento Combinado")
            lines.append(f"• **Entrada no PIX (50%):** {_fmt(half_pix)}")
            lines.append(f"• **Saldo no Cartão de Crédito (50%):** {_fmt(half_card)} sem juros")
            lines.append(f"• **Valor Final a Pagar:** **{_fmt(total_regular)}**\n")
            lines.append("📱 *(O QR Code para a entrada no PIX será exibido na confirmação final).*")
        else:
            lines.append(f"• **Valor Final a Pagar:** **{_fmt(total_regular)}**")

        lines.append(
            "\n---\n"
            f"**{cust_name}**, por favor confira todos os valores, o frete, o endereço e a forma de pagamento acima.\n"
            "**Podemos confirmar e finalizar o seu pedido?**"
        )

        return {
            "reply": "\n".join(lines),
            "language": detected_lang,
            "confidence": 1.0,
            "model_used": "direct_checkout_order_review",
            "tools_called": [],
            "products": [],
            "active_agent": "checkout",
            "action": None,
            "options": ["Sim, confirmar pedido", "Não, alterar pagamento"],
            "logged_failed_intent": False,
        }

    def _detect_add_to_cart_intent(self, user_message: str) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Detects if user intends to purchase or add a specific musical instrument to their cart via chat text,
        e.g.: 'quero 2 Yamaha C40 Nylon Natural', 'adicionar 1 Yamaha F310 ao carrinho', 'quero comprar 2 Yamaha C40'.
        Returns the resolved product dict and quantity, or None if it's general browsing or checkout closing.
        """
        norm = normalize_text(user_message).lower()
        if "selecionei no carrinho" in norm:
            return None

        if any(k in norm for k in [
            "fechar carrinho", "fechar pedido", "fechar o pedido", "fechar meu pedido",
            "finalizar compra", "finalizar o pedido", "finalizar pedido", "concluir compra",
            "concluir pedido", "concluir o pedido", "vamos fechar",
            "vamos pagar", "quero pagar", "quero fechar", "fechar a compra",
            "pagar agora", "checkout"
        ]) or is_checkout_intent(user_message):
            return None

        is_browsing = any(q in norm for q in [
            "ver", "olhar", "mostrar", "quais", "qual", "opcoes", "recomendar", "recomenda",
            "catalogo", "modelos", "quanto custa", "tem disponivel", "tem em estoque", "existe",
            "comparar", "diferenca"
        ])
        explicit_cart = any(k in norm for k in [
            "carrinho", "adicionar", "adiciona", "colocar", "coloca", "coloque",
            "bota", "bote", "incluir", "inclui", "por", "poe"
        ])
        if is_browsing and not explicit_cart:
            return None

        is_intent = explicit_cart or any(w in norm for w in [
            "quero", "comprar", "levar", "vou levar", "pode ser", "adiciona", "adicionar", "coloca"
        ]) or bool(re.search(r"^\s*\d+\s*x\s+", user_message))
        if not is_intent:
            return None

        qty = 1
        qty_m = re.search(
            r"(?:quero|adicionar|adiciona|coloca|coloque|bota|bote|comprar|levar|^)\s*(\d+)\s*(?:x|unidades?|itens?)?\b",
            user_message,
            re.IGNORECASE,
        )
        if qty_m and 1 <= int(qty_m.group(1)) <= 100:
            qty = int(qty_m.group(1))
        else:
            standalone_m = re.search(r"\b(\d+)\s*(?:x|unidades?|itens?)?\b", user_message)
            if standalone_m and 1 <= int(standalone_m.group(1)) <= 100:
                qty = int(standalone_m.group(1))

        all_products = self.tool_executor.product_repo.search_products(limit=200)
        best_p = None
        best_score = 0
        norm_clean = re.sub(r"[^a-z0-9]", "", norm)
        for p in all_products:
            p_norm = normalize_text(p["name"]).lower()
            p_norm_clean = re.sub(r"[^a-z0-9]", "", p_norm)
            if p_norm in norm or (len(p_norm_clean) >= 4 and p_norm_clean in norm_clean):
                score = 1000 + len(p_norm)
            else:
                tokens = [
                    t for t in p_norm.split()
                    if len(t) >= 2 and t not in [
                        "nylon", "natural", "aco", "folk", "eletrico", "preto", "sunburst",
                        "soprano", "tenor", "concert", "violao", "guitarra", "baixo",
                        "bateria", "ukulele", "de", "do", "da", "com"
                    ]
                ]
                matching = []
                for t in tokens:
                    t_clean = re.sub(r"[^a-z0-9]", "", t)
                    if t in norm or (len(t_clean) >= 2 and t_clean in norm_clean):
                        matching.append(t)
                if len(matching) >= 2:
                    score = 500 + sum(len(t) for t in matching)
                elif len(matching) == 1 and any(ch.isdigit() for ch in matching[0]):
                    score = 300 + len(matching[0])
                else:
                    score = 0
            if score > best_score:
                best_score = score
                best_p = p

        if best_p and best_score >= 100:
            return best_p, qty
        return None

    def run(
        self,
        user_message: str,
        session_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
        cart: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates multi-agent routing and execution following the conversational state machine:
        1. Identification: Welcomes user and requests phone/email.
        2. Database verification: If customer not found, asks if new customer or retry.
        3. Onboarding: If new customer, displays LGPD terms -> registration form -> address form -> how to help.
        4. Shopping: Continuous catalog recommendations and adding items to cart.
        5. Checkout: Summary -> PIX vs. Card -> Delivery address confirmation/update -> Final breakdown.
        """
        clarity = self.evaluate_clarity_and_confidence(
            user_message=user_message,
            chat_history=chat_history,
        )

        detected_lang = clarity["language"]

        # 1. Low-Confidence or Gibberish Guardrail
        if not clarity["is_clear"]:
            self.failed_intent_repo.log_failure(
                session_id=session_id,
                user_message=user_message,
                detected_language=detected_lang,
                reason=clarity["reason"],
                confidence=clarity["confidence"],
                resolution_strategy=clarity.get("strategy", "ask_rephrase"),
            )
            template = LOW_CONFIDENCE_TEMPLATES.get(
                detected_lang, LOW_CONFIDENCE_TEMPLATES["pt"]
            )
            return {
                "reply": template,
                "language": detected_lang,
                "confidence": clarity["confidence"],
                "model_used": "guardrail_low_confidence",
                "tools_called": [],
                "products": [],
                "active_agent": "orchestrator",
                "logged_failed_intent": True,
            }

        # Retrieve active session state
        history = chat_history or []
        sess = self._get_session(session_id=session_id, chat_history=history)

        # Synchronize cart payload if provided by frontend or caller
        if cart:
            cart_items = []
            total = 0.0
            has_promo = False
            for item in cart:
                pid = item.get("product_id")
                prod = self.tool_executor.product_repo.get_by_id(pid) if pid else None
                if not prod:
                    prod = {
                        "product_id": pid or 0,
                        "name": item.get("name", "Instrumento"),
                        "price_brl": float(item.get("price_brl", 0.0)),
                        "effective_price_brl": float(item.get("effective_price_brl") or item.get("price_brl", 0.0)),
                    }
                qty = int(item.get("qty", 1))
                u_p = float(prod.get("effective_price_brl") or prod.get("price_brl", 0.0))
                total += u_p * qty
                if prod.get("active_promotion"):
                    has_promo = True
                cart_items.append((prod, qty))
            if cart_items:
                sess["cart_items"] = cart_items
                sess["cart_total"] = total
                sess["has_promo"] = has_promo

        all_msgs = history + [{"role": "user", "content": user_message}]
        claimed_name = extract_claimed_customer_name(all_msgs)
        norm_user_msg = normalize_text(user_message).strip()

        last_assistant_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_assistant_msg = normalize_text(msg.get("content") or "")
                break

        # 1.5. Human Handoff Intent (Attendant / Operator / Human Support)
        if is_human_handoff_intent(norm_user_msg):
            cust = sess.get("customer") or {}
            cust_name = cust.get("name") or claimed_name or ""
            greeting = f"Olá, **{cust_name}**!" if cust_name else "Olá!"

            reply = (
                f"{greeting} Para falar diretamente com um de nossos **atendentes, consultores musicais ou equipe de suporte**, "
                "por favor entre em contato pelo nosso **WhatsApp oficial**: 📲🎸\n\n"
                "• 📱 **WhatsApp:** **(67) 3321-4500**\n"
                "• 🔗 **Link Direto:** [Clique aqui para conversar no WhatsApp](https://wa.me/556733214500)\n"
                "• ⏰ **Horário de Atendimento:** Segunda a Sexta das 08:30 às 18:30 e Sábados das 08:30 às 13:00\n"
                "• ⚡ **Tempo Médio de Resposta:** Menos de 5 minutos durante o horário comercial!\n\n"
                "Se preferir continuar por aqui, estou 100% à disposição para te ajudar a escolher instrumentos, simular pagamentos, consultar frete ou rastrear seus pedidos!"
            )
            return {
                "reply": reply,
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_human_handoff_whatsapp",
                "tools_called": [],
                "products": [],
                "active_agent": "support",
                "action": None,
                "options": [
                    "📲 Abrir WhatsApp: (67) 3321-4500",
                    "🎸 Ver Catálogo",
                    "📦 Meus Pedidos",
                    "📍 Informações da Loja",
                ],
                "logged_failed_intent": False,
            }

        # 2. Direct Interface Form Submissions
        # 2.1. Registration Form Submission (Step 1)
        if user_message.startswith("[DADOS_DO_CADASTRO]:") or user_message.startswith("[DADOS_DO_FORMULARIO]:"):
            raw_json = re.sub(r"^\[(?:DADOS_DO_CADASTRO|DADOS_DO_FORMULARIO)\]:\s*", "", user_message)
            try:
                form_data = json.loads(raw_json)
            except Exception:
                form_data = {}

            name = form_data.get("name", claimed_name or "Cliente")
            phone = form_data.get("phone") or form_data.get("telefone") or ""
            email = form_data.get("email", "")

            sess["pending_customer"] = {"name": name, "phone": phone, "email": email}
            sess["stage"] = "REGISTER_ADDRESS"

            reply_text = (
                f"Perfeito, **{name}**! Já registrei seus dados de contato. 🎶\n\n"
                "Agora, por favor, preencha o formulário abaixo com o seu **endereço de entrega** "
                "(você pode digitar o CEP para busca automática da rua e bairro):"
            )
            return {
                "reply": reply_text,
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_form_handler",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": "SHOW_ADDRESS_FORM",
                "options": [],
                "logged_failed_intent": False,
            }

        # 2.2. Address Form Submission (Step 2)
        if user_message.startswith("[DADOS_DO_ENDERECO]:"):
            raw_json = re.sub(r"^\[DADOS_DO_ENDERECO\]:\s*", "", user_message)
            try:
                addr_data = json.loads(raw_json)
            except Exception:
                addr_data = {}

            pending = sess.get("pending_customer") or {}
            cust_name = addr_data.get("name") or pending.get("name") or claimed_name or "Cliente"
            cust_phone = addr_data.get("phone") or pending.get("phone")
            cust_email = addr_data.get("email") or pending.get("email")

            cep = addr_data.get("cep")
            street = addr_data.get("street") or addr_data.get("address") or ""
            number = addr_data.get("number", "")
            complement = addr_data.get("complement", "")
            neighborhood = addr_data.get("neighborhood", "")
            city = addr_data.get("city", "Campo Grande")

            addr_parts = [p for p in [street, number, complement, neighborhood] if p]
            full_addr = ", ".join(addr_parts) if addr_parts else street

            reg_res = self.tool_executor.register_customer(
                name=cust_name,
                phone=cust_phone,
                email=cust_email,
                cep=cep,
                address=full_addr,
                city=city,
            )

            matched = self.tool_executor.customer_repo.find_customers(cust_email or cust_phone or cust_name)
            if matched:
                sess["customer"] = matched[0]
            else:
                sess["customer"] = {
                    "name": cust_name, "phone": cust_phone, "email": cust_email,
                    "address": full_addr, "city": city, "cep": cep
                }

            # If this address update occurred during checkout
            if sess.get("stage") == "CHECKOUT_ADDRESS_UPDATE" or sess.get("payment_method"):
                return self._build_final_checkout_summary(sess=sess, detected_lang=detected_lang)

            # Standard onboarding completion
            sess["stage"] = "SHOPPING"
            reply_text = (
                f"🎉 Cadastro e endereço de entrega realizados com sucesso, **{cust_name}**!\n\n"
                f"📍 **Endereço cadastrado:** {full_addr} - {city} (CEP: {cep or '—'})\n\n"
                "Seu cadastro está ativo no nosso sistema! Como posso te ajudar agora? "
                "Gostaria de ver nossas opções de violões, ukuleles ou produtos em promoção? 🎸✨"
            )
            return {
                "reply": reply_text,
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_form_handler",
                "tools_called": [{"tool": "register_customer", "arguments": reg_res}],
                "products": [],
                "active_agent": "customer",
                "action": None,
                "options": ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção"],
                "logged_failed_intent": False,
            }

        # 3. Explicit Customer Identification Check (Email or Phone)
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", user_message)
        phone_match = re.search(r"(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?(?:9\s*)?\d{4}[-\s]?\d{4}", user_message)
        digits_only = re.sub(r"\D", "", phone_match.group(0)) if phone_match else re.sub(r"\D", "", user_message)
        has_phone_digits = 8 <= len(digits_only) <= 13
        in_checkout_stages = sess.get("stage") in (
            "CHECKOUT_ADDRESS_CONFIRM", "CHECKOUT_PAYMENT_SELECT",
            "CHECKOUT_INSTALLMENT_SELECTION", "CHECKOUT_FINAL_CONFIRM"
        )
        is_phone = has_phone_digits and (
            (sess.get("customer") is None and not in_checkout_stages)
            or any(k in norm_user_msg for k in ["telefone", "zap", "whatsapp", "celular", "meu contato", "sou o", "meu zap"])
            or len(user_message.strip().split()) <= 3
        )
        is_email = bool(email_match)

        last_agent = IntentRouter.get_last_active_agent(history)
        if is_email or is_phone:
            identifier = email_match.group(0) if is_email else digits_only
            matched_customers = self.tool_executor.customer_repo.find_customers(identifier)

            if not matched_customers:
                sess["identifier_tried"] = identifier
                sess["stage"] = "AWAITING_NEW_OR_RETRY"
                prefill = {"email": identifier} if is_email else {"phone": user_message.strip()}
                return {
                    "reply": (
                        f"Não localizamos nenhum cadastro vinculado a **{identifier}**. 😊\n\n"
                        "Você é um **novo cliente** na Empório da Música ou deseja **tentar novamente** com outro número/e-mail?"
                    ),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_customer_verification",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "checkout" if last_agent == "checkout" else "customer",
                    "action": None,
                    "options": ["Sim, sou novo cliente", "Tentar novamente"],
                    "prefill": prefill,
                    "logged_failed_intent": False,
                }
            else:
                cust = matched_customers[0]
                sess["customer"] = cust
                sess["stage"] = "SHOPPING"
                cust_name = cust.get("name", "Cliente")
                cust_addr = cust.get("address") or ""
                cust_city = cust.get("city") or "Campo Grande"
                cust_cep = cust.get("cep") or ""
                addr_display = f"{cust_addr} - {cust_city}" + (f" (CEP: {cust_cep})" if cust_cep else "")

                if last_agent == "checkout":
                    return {
                        "reply": (
                            f"Perfeito, **{cust_name}**! Localizamos seu cadastro com sucesso. 🎸\n\n"
                            f"📍 **Endereço de entrega cadastrado:**\n{addr_display}\n\n"
                            "Você deseja **manter este endereço** para a entrega ou prefere **atualizar o endereço**?"
                        ),
                        "language": detected_lang,
                        "confidence": 1.0,
                        "model_used": "direct_customer_verification",
                        "tools_called": [],
                        "products": [],
                        "active_agent": "checkout",
                        "action": None,
                        "options": ["Manter este endereço", "Atualizar endereço"],
                        "logged_failed_intent": False,
                    }

                return {
                    "reply": (
                        f"Olá, **{cust_name}**! Que alegria ter você de volta à **Empório da Música**! 🎸✨\n\n"
                        "Identificamos seu cadastro com sucesso. Como posso te ajudar hoje?"
                    ),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_customer_verification",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "customer",
                    "action": None,
                    "options": ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção"],
                    "logged_failed_intent": False,
                }

        # 4. User wishes to retry identification ("Tentar novamente")
        if norm_user_msg in ("tentar novamente", "tentar outro dado", "nao, vou tentar outro dado", "nao, tentar outro dado", "outro numero", "outro email", "tentar de novo"):
            sess["stage"] = "IDENTIFY"
            return {
                "reply": "Sem problemas! Por favor, digite o número de telefone/WhatsApp ou e-mail cadastrado:",
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_customer_verification",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": None,
                "options": [],
                "logged_failed_intent": False,
            }

        # 5. New Customer Confirmation -> Speaks about Termo de Aceite (LGPD)
        is_new_cust = (
            norm_user_msg in ("sim, sou novo cliente", "sou novo cliente", "sim sou novo cliente", "sou novo", "novo cliente", "sim", "sou novo por aqui")
            or any(k in norm_user_msg for k in ["novo cliente", "sou novo", "sou novo por aqui"])
            or re.search(r"\b(?:sim,?\s*)?(?:sou\s+)?novo\s+cliente\b", norm_user_msg)
        )
        if (
            (is_new_cust and ("novo cliente" in last_assistant_msg or sess.get("stage") == "AWAITING_NEW_OR_RETRY"))
            or norm_user_msg in ("sim, sou novo cliente", "sou novo cliente", "sou novo", "novo cliente")
            or (sess.get("customer") is None and is_new_cust and norm_user_msg not in ("sim", "quero me cadastrar", "fazer cadastro"))
        ):
            sess["stage"] = "TERMS_CONSENT"
            return {
                "reply": (
                    "Que ótimo! Seja muito bem-vindo(a) à nossa família musical! 🎸✨\n\n"
                    "Antes de preenchermos seus dados, precisamos do seu consentimento com os nossos **Termos de Uso e Política de Privacidade (LGPD)**:\n\n"
                    "📄 **Termo de Consentimento:**\n"
                    "• Seus dados cadastrais (nome, telefone, e-mail e endereço) serão utilizados de forma segura e exclusiva para emissão de nota fiscal, faturamento e entrega dos seus pedidos.\n"
                    "• Não comercializamos nem compartilhamos suas informações com terceiros.\n\n"
                    "Você aceita os termos para prosseguirmos com seu cadastro?"
                ),
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_terms_consent",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": None,
                "options": ["Aceito os termos", "Não aceito"],
                "logged_failed_intent": False,
            }

        # 6. Terms Acceptance or Rejection
        if norm_user_msg in ("aceito os termos", "aceito", "concordo", "sim, aceito", "aceitar termos", "sim aceito", "eu aceito") or (
            sess.get("stage") == "TERMS_CONSENT" and (
                norm_user_msg in ("sim", "aceito", "concordo", "ok", "claro", "pode ser", "tudo bem")
                or any(k in norm_user_msg for k in ["aceito", "concordo", "aceitar"])
            )
        ):
            sess["terms_accepted"] = True
            sess["stage"] = "REGISTER_CONTACT"
            prefill_data = {}
            ident = sess.get("identifier_tried")
            if not ident and history:
                for msg in reversed(history):
                    if msg.get("role") == "user":
                        u_t = msg.get("content", "")
                        if "@" in u_t:
                            ident = u_t
                            break
                        d_t = re.sub(r"\D", "", u_t)
                        if 8 <= len(d_t) <= 12:
                            ident = u_t
                            break
            if ident:
                if "@" in ident:
                    prefill_data["email"] = ident
                else:
                    prefill_data["phone"] = ident

            return {
                "reply": (
                    "Excelente! Muito obrigado pela confiança. 🎶\n\n"
                    "Por favor, preencha seus dados de contato no formulário abaixo:"
                ),
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_form_trigger",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": "SHOW_REGISTER_FORM",
                "options": [],
                "prefill": prefill_data or None,
                "logged_failed_intent": False,
            }

        if norm_user_msg in ("nao aceito", "nao concordo", "recuso") or (
            sess.get("stage") == "TERMS_CONSENT" and norm_user_msg in ("nao", "nao aceito")
        ):
            return {
                "reply": (
                    "Compreendo perfeitamente! O aceite dos termos é necessário para efetuarmos vendas, faturamento e entregas com nota fiscal. "
                    "Se mudar de ideia, basta clicar em **Aceito os termos** a qualquer momento!\n\n"
                    "Como posso te ajudar enquanto isso?"
                ),
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_terms_consent",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": None,
                "options": ["Aceito os termos", "🎸 Ver Catálogo", "📍 Informações da Loja"],
                "logged_failed_intent": False,
            }

        # 7. Explicit Registration / Address Update Triggers
        if (
            any(term in norm_user_msg for term in ["fazer cadastro", "me cadastrar", "novo cadastro", "quero me cadastrar", "abrir cadastro", "formulario de cadastro"])
            or norm_user_msg == "cadastrar"
        ):
            sess["stage"] = "REGISTER_CONTACT"
            return {
                "reply": (
                    "Com certeza! Seja muito bem-vindo(a) à **Empório da Música**! 🎸🎶\n\n"
                    "Preencha seus dados de contato no formulário abaixo para iniciarmos seu cadastro:"
                ),
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_form_trigger",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": "SHOW_REGISTER_FORM",
                "options": [],
                "logged_failed_intent": False,
            }

        if any(term in norm_user_msg for term in ["atualizar endereco", "mudar endereco", "alterar endereco", "trocar endereco", "novo endereco"]):
            sess["stage"] = "CHECKOUT_ADDRESS_UPDATE" if last_agent == "checkout" else "REGISTER_ADDRESS"
            return {
                "reply": (
                    "Claro! Vamos atualizar o seu endereço de entrega. 📍\n\n"
                    "Preencha o formulário abaixo com os novos dados (digite seu CEP para busca automática da rua e bairro):"
                ),
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_form_trigger",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": "SHOW_ADDRESS_FORM",
                "options": [],
                "logged_failed_intent": False,
            }

        # 7.1. Enforcement of Identification / Onboarding Stages (Flow Guard)
        if sess.get("customer") is None:
            # Case A: Awaiting New Customer or Retry
            if sess.get("stage") == "AWAITING_NEW_OR_RETRY":
                ident = sess.get("identifier_tried", "")
                return {
                    "reply": (
                        f"Ainda não localizamos cadastro com **{ident}**. 😊\n\n"
                        "Por favor, confirme se você é um **novo cliente** (para fazermos seu cadastro e prosseguirmos) "
                        "ou se deseja **tentar novamente** com outro número ou e-mail:"
                    ),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_customer_verification",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "customer",
                    "action": None,
                    "options": ["Sim, sou novo cliente", "Tentar novamente"],
                    "logged_failed_intent": False,
                }

            # Case B: Terms Consent Awaiting Acceptance
            if sess.get("stage") == "TERMS_CONSENT":
                return {
                    "reply": (
                        "Para darmos andamento ao seu cadastro e liberarmos seu atendimento, você precisa aceitar os "
                        "nossos **Termos de Uso e Política de Privacidade (LGPD)**.\n\n"
                        "Você aceita os termos para prosseguirmos?"
                    ),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_terms_consent",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "customer",
                    "action": None,
                    "options": ["Aceito os termos", "Não aceito"],
                    "logged_failed_intent": False,
                }

            # Case C: Registration Form Pending
            if sess.get("stage") == "REGISTER_CONTACT":
                return {
                    "reply": "Por favor, preencha seus dados de contato no formulário acima para prosseguirmos com seu cadastro! 🎶",
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_form_trigger",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "customer",
                    "action": "SHOW_REGISTER_FORM",
                    "options": [],
                    "logged_failed_intent": False,
                }

            # Case D: Address Form Pending
            if sess.get("stage") == "REGISTER_ADDRESS":
                return {
                    "reply": "Por favor, preencha seu endereço de entrega no formulário acima para finalizarmos seu cadastro! 📍",
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_form_trigger",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "customer",
                    "action": "SHOW_ADDRESS_FORM",
                    "options": [],
                    "logged_failed_intent": False,
                }

            # Case E: User is in IDENTIFY stage (must not bypass identification to catalog)
            is_greeting = bool(
                norm_user_msg in ("oi", "ola", "ola!", "oi!", "opa", "e ai", "e ae", "opa!", "bom dia", "boa tarde", "boa noite", "tudo bem", "tudo bom", "hello", "hi", "hey")
                or re.match(r"^(?:oi+|ol[aá]+|opa|e\s*a[íi]|bom\s*dia|boa\s*tarde|boa\s*noite)\b", norm_user_msg)
            )
            # In web session, ANY attempt to browse catalog or chat without identifying is intercepted
            if is_greeting or session_id.startswith("web_"):
                detected_stage = IntentRouter.classify_stage(user_message=user_message, chat_history=history)
                if detected_stage != "support" or is_greeting:
                    return {
                        "reply": (
                            "Olá! Seja muito bem-vindo(a) à **Empório da Música**! 🎸🎶\n\n"
                            "Para começarmos o seu atendimento e identificarmos seu cadastro, por favor informe o seu **número de telefone/WhatsApp ou e-mail**."
                        ),
                        "language": detected_lang,
                        "confidence": 1.0,
                        "model_used": "direct_customer_verification",
                        "tools_called": [],
                        "products": [],
                        "active_agent": "customer",
                        "action": None,
                        "options": [],
                        "logged_failed_intent": False,
                    }

        # 7.5. Intent to Add Product to Cart via Chat Text (e.g. "quero 2 Yamaha C40 Nylon Natural", "adicionar 1 Yamaha F310 ao carrinho")
        add_cart_match = self._detect_add_to_cart_intent(user_message)
        if add_cart_match:
            p, req_qty = add_cart_match
            avail_stock = int(p.get("stock_quantity", 0))
            if avail_stock <= 0 or p.get("status") != "active":
                alts = self.tool_executor.product_repo.get_alternatives_for_unavailable(p)
                alt_names = [f"• **{alt['name']}** (R$ {alt['effective_price_brl']:,.2f})".replace(",", "X").replace(".", ",").replace("X", ".") for alt in alts[:3]]
                alt_text = "\n".join(alt_names) if alt_names else "Consulte nosso catálogo para outros modelos semelhantes."
                return {
                    "reply": (
                        f"Puxa, o instrumento **{p['name']}** está temporariamente esgotado no momento. 🎸\n\n"
                        f"Mas temos ótimas alternativas disponíveis em estoque agora mesmo:\n{alt_text}\n\n"
                        "Gostaria de conhecer algum desses modelos?"
                    ),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_out_of_stock",
                    "tools_called": [],
                    "products": alts[:3],
                    "active_agent": "catalog",
                    "action": None,
                    "options": [alt["name"] for alt in alts[:3]],
                    "logged_failed_intent": False,
                }

            final_qty = min(avail_stock, req_qty)
            cart_items = sess.get("cart_items") or []
            existing_idx = None
            for idx, (cp, cqty) in enumerate(cart_items):
                if cp["product_id"] == p["product_id"]:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                old_p, old_qty = cart_items[existing_idx]
                new_qty = min(avail_stock, old_qty + final_qty)
                cart_items[existing_idx] = (old_p, new_qty)
            else:
                cart_items.append((p, final_qty))

            sess["cart_items"] = cart_items
            tot_val = sum(float(cp.get("effective_price_brl") or cp.get("price_brl", 0.0)) * cqty for cp, cqty in cart_items)
            tot_items = sum(cqty for _, cqty in cart_items)
            sess["cart_total"] = tot_val
            sess["has_promo"] = any(bool(cp.get("active_promotion")) for cp, _ in cart_items)

            def _fmt_m(val: float) -> str:
                return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            item_word = "1 item" if tot_items == 1 else f"{tot_items} itens"
            stock_warning = f" *(Ajustado para o máximo disponível em estoque: {avail_stock})*" if req_qty > avail_stock else ""

            reply = (
                f"🛒 **{final_qty}x {p['name']}** adicionado ao seu carrinho!{stock_warning}\n\n"
                f"Seu carrinho agora possui **{item_word}** (Total: **{_fmt_m(tot_val)}**).\n\n"
                "Deseja adicionar mais algum instrumento ou ver mais opções?"
            )

            return {
                "reply": reply,
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_add_to_cart",
                "tools_called": [],
                "products": [],
                "active_agent": "checkout",
                "action": "ADD_TO_CART",
                "product": p,
                "qty": final_qty,
                "options": ["🎸 Ver mais produtos", "🛒 Ver Carrinho", "⚡ Fechar Pedido"],
                "logged_failed_intent": False,
            }

        # 7.6. Intent to View Current Cart
        if any(k in norm_user_msg for k in ["ver carrinho", "meu carrinho", "itens no carrinho", "o que tem no carrinho", "abrir carrinho", "olhar carrinho"]) and not any(k in norm_user_msg for k in ["fechar", "finalizar"]):
            cart_items = sess.get("cart_items") or []
            if cart_items:
                def _fmt_cart(val: float) -> str:
                    return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                tot_items = sum(cqty for _, cqty in cart_items)
                tot_word = "1 item" if tot_items == 1 else f"{tot_items} itens"
                lines = [
                    f"### 🛒 Itens no seu Carrinho ({tot_word})\n",
                    "| Item | Quantidade | Valor Unitário | Subtotal |",
                    "| :--- | :---: | :---: | :---: |",
                ]
                tot_reg = 0.0
                for cp, cqty in cart_items:
                    u_p = float(cp.get("effective_price_brl") or cp.get("price_brl", 0.0))
                    sub = u_p * cqty
                    tot_reg += sub
                    lines.append(f"| {cp['name']} | {cqty} | {_fmt_cart(u_p)} | {_fmt_cart(sub)} |")
                lines.append(f"| **Total do Pedido** | **{tot_word}** | | **{_fmt_cart(tot_reg)}** |\n")
                lines.append("Deseja fechar o seu pedido agora ou adicionar mais produtos?")
                return {
                    "reply": "\n".join(lines),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_show_cart",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "checkout",
                    "action": None,
                    "options": ["⚡ Fechar Pedido", "🎸 Continuar Comprando"],
                    "logged_failed_intent": False,
                }

        # 7.7. Order Tracking / History Inquiry (Direct dispatch to Support Specialist)
        norm_clean_msg = normalize_text(user_message).lower()
        is_order_tracking_query = any(k in norm_clean_msg for k in [
            "rastreio", "rastrear", "rastreamento", "status do pedido", "status de pedido",
            "status dos pedidos", "status da entrega", "status das entregas",
            "onde esta meu pedido", "onde esta o pedido", "onde estao meus pedidos",
            "cade meu pedido", "cade o pedido", "cade meus pedidos", "minha entrega",
            "acompanhar pedido", "acompanhamento do pedido", "acompanhar meus pedidos",
            "consultar pedido", "consultar pedidos", "consultar rastreio",
            "consultar meus pedidos", "ver meu pedido", "ver meus pedidos", "meus pedidos",
            "meu pedido", "pedidos anteriores", "compras anteriores", "minhas compras",
            "minha compra", "historico de pedidos", "historico de compras",
            "meu pedido ja saiu", "meus pedidos ja sairam", "quando chega meu pedido",
            "quando chega", "previsao de entrega"
        ]) or norm_clean_msg in ("pedidos", "meus pedidos", "compras", "rastreio", "meu rastreio")

        if is_order_tracking_query and not any(k in norm_clean_msg for k in ["fechar", "finalizar", "concluir", "pagar"]):
            return self.support_agent.execute(
                user_message=user_message,
                session_id=session_id,
                chat_history=history,
                claimed_customer_name=claimed_name,
                session_context=sess,
            )

        # 8. Checkout Intent: Intention to close order / checkout
        is_checkout_intent_flag = is_checkout_intent(user_message) or any(
            k in norm_user_msg for k in [
                "fechar carrinho", "fechar pedido", "fechar o pedido", "fechar meu pedido",
                "finalizar compra", "finalizar o pedido", "finalizar pedido", "concluir compra",
                "concluir pedido", "concluir o pedido", "vamos fechar",
                "vamos pagar", "quero pagar", "quero fechar", "fechar a compra",
                "pagar agora", "fechar", "checkout"
            ]
        )
        if is_checkout_intent_flag:
            found_items, total, has_promo = self.checkout_agent._parse_cart_items_and_total(
                user_message=user_message,
                chat_history=history,
            )
            if not found_items and sess.get("cart_items"):
                found_items = sess["cart_items"]
                total = float(sess.get("cart_total", 0.0))
                has_promo = bool(sess.get("has_promo", False))

            if not found_items and total == 0.0:
                return {
                    "reply": (
                        "🛒 **Seu carrinho ainda não possui instrumentos adicionados.**\n\n"
                        "Que tal escolhermos um instrumento no catálogo primeiro? "
                        "Gostaria de ver nossas opções de violões, ukuleles ou instrumentos em promoção? 🎸✨"
                    ),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_empty_cart",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "checkout",
                    "action": None,
                    "options": ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção"],
                    "logged_failed_intent": False,
                }

            if found_items or total > 0.0:
                sess["cart_items"] = found_items
                sess["cart_total"] = total
                sess["has_promo"] = has_promo
                sess["stage"] = "CHECKOUT_ADDRESS_CONFIRM"

                def _fmt_money(val: float) -> str:
                    return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                lines = [
                    "### 🛒 Resumo do seu Pedido\n",
                    "| Item | Quantidade | Valor Unitário | Subtotal |",
                    "| :--- | :---: | :---: | :---: |",
                ]
                for p, qty in found_items:
                    u_p = float(p.get("effective_price_brl") or p.get("price_brl", 0.0))
                    lines.append(f"| {p['name']} | {qty} | {_fmt_money(u_p)} | {_fmt_money(u_p * qty)} |")
                tot_items = sum(q for _, q in found_items)
                lines.append(f"| **Subtotal dos Produtos** | **{tot_items} item(ns)** | | **{_fmt_money(total)}** |\n")

                pix_val = total * 0.95 if not has_promo else total
                lines.append("Valores e condições de pagamento disponíveis para este pedido:")
                if not has_promo:
                    lines.append(f"• ⚡ **À vista no PIX:** com 5% de desconto (**{_fmt_money(pix_val)}**)")
                else:
                    lines.append(f"• ⚡ **À vista no PIX:** **{_fmt_money(pix_val)}** (desconto PIX não cumulativo com promoção)")

                max_inst = 1
                for n in range(12, 0, -1):
                    val = total / n
                    if n >= 7 and val >= 100.0:
                        max_inst = n
                        break
                    if 4 <= n <= 6 and val >= 80.0:
                        max_inst = n
                        break
                    if 2 <= n <= 3 and val >= 50.0:
                        max_inst = n
                        break
                    if n == 1:
                        max_inst = 1
                        break
                inst_val = total / max_inst
                if max_inst > 1:
                    lines.append(f"• 💳 **Cartão de Crédito:** em até **{max_inst}x de {_fmt_money(inst_val)} sem juros** (ou 1x de {_fmt_money(total)})")
                else:
                    lines.append(f"• 💳 **Cartão de Crédito:** 1x de {_fmt_money(total)}")

                lines.append(f"• 📄 **Boleto Bancário:** à vista no valor de **{_fmt_money(total)}** (compensação em até 3 dias úteis)")
                lines.append(f"• 💳 **Cartão de Débito:** à vista no valor de **{_fmt_money(total)}** (todas as bandeiras aceitas)")

                if total >= 2000.0:
                    lines.append("• 🔀 **Pagamento Combinado:** entrada no PIX + saldo parcelado no Cartão de Crédito (permitido para compras acima de R$ 2.000,00)")

                cust = sess.get("customer")
                if not cust and history:
                    for msg in reversed(history):
                        c = msg.get("content", "")
                        if "@" in c:
                            em = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", c)
                            if em:
                                m = self.tool_executor.customer_repo.find_customers(em.group(0))
                                if m:
                                    cust = m[0]
                                    sess["customer"] = cust
                                    break
                        d = re.sub(r"\D", "", c)
                        if 8 <= len(d) <= 12:
                            m = self.tool_executor.customer_repo.find_customers(d)
                            if m:
                                cust = m[0]
                                sess["customer"] = cust
                                break

                cust_city = cust.get("city", "Campo Grande") if cust else "Campo Grande"
                is_cg = "campo grande" in cust_city.lower()
                if is_cg:
                    if total >= 300.0 or total == 0.0:
                        lines.append("\n• 🚚 **Frete:** **Frete Grátis** para Campo Grande/MS via motoboy próprio da loja (1 a 3 dias úteis).")
                    else:
                        lines.append("\n• 🚚 **Frete:** R$ 25,00 para Campo Grande/MS via motoboy próprio da loja (1 a 3 dias úteis).")
                else:
                    lines.append(f"\n• 🚚 **Frete:** R$ 45,00 para {cust_city} via transportadora parceira (3 a 7 dias úteis).")

                if cust:
                    addr_disp = f"{cust.get('address', '')} - {cust.get('city', 'Campo Grande')}" + (f" (CEP: {cust.get('cep')})" if cust.get('cep') else "")
                    lines.append(f"\n📍 **Endereço de entrega cadastrado:**")
                    lines.append(f"**{addr_disp}**\n")
                    lines.append("O endereço de entrega está correto para o envio?")

                    return {
                        "reply": "\n".join(lines),
                        "language": detected_lang,
                        "confidence": 1.0,
                        "model_used": "direct_checkout_address_confirm",
                        "tools_called": [],
                        "products": [],
                        "active_agent": "checkout",
                        "action": None,
                        "options": ["Sim, endereço correto", "Não, alterar endereço"],
                        "logged_failed_intent": False,
                    }
                else:
                    lines.append("\nPara confirmarmos o envio e emitirmos a nota fiscal, por favor informe seu **e-mail** ou **telefone/WhatsApp** cadastrado:")
                    return {
                        "reply": "\n".join(lines),
                        "language": detected_lang,
                        "confidence": 1.0,
                        "model_used": "direct_checkout_address_confirm",
                        "tools_called": [],
                        "products": [],
                        "active_agent": "checkout",
                        "action": None,
                        "options": ["Fazer meu cadastro"],
                        "logged_failed_intent": False,
                    }

        # 9. Address Confirmation in Checkout
        clean_last_asst = re.sub(r"[*_]", "", last_assistant_msg).strip().lower()
        in_addr_confirm = (
            sess.get("stage") == "CHECKOUT_ADDRESS_CONFIRM"
            or "o endereco de entrega esta correto" in clean_last_asst
            or "deseja manter este endereco" in clean_last_asst
            or "endereco de entrega cadastrado" in clean_last_asst
        )

        is_addr_negative = in_addr_confirm and (
            norm_user_msg in (
                "nao, alterar endereco", "alterar endereco", "mudar endereco", "trocar endereco",
                "nao", "endereco errado", "atualizar endereco", "mudar o endereco", "trocar o endereco",
                "outro endereco", "novo endereco", "mudou", "errado", "incorreto", "negativo", "nao quero"
            )
            or any(w in norm_user_msg for w in [
                "alterar endereco", "mudar endereco", "trocar endereco", "atualizar endereco",
                "outro endereco", "novo endereco", "endereco errado", "mudar o endereco"
            ])
        )

        is_addr_affirmative = in_addr_confirm and not is_addr_negative and (
            norm_user_msg in (
                "sim, endereco correto", "endereco correto", "esta correto", "sim", "correto",
                "pode ser esse", "manter este endereco", "manter endereco", "manter", "tudo certo",
                "ta certo", "esse mesmo", "este mesmo", "pode mandar", "ok", "beleza", "confirmo",
                "exato", "perfeito", "claro", "isso", "positivo", "sim por favor", "sim pode ser",
                "pode ser", "sim ta certo", "sim esta correto", "sim esta certo", "ta correto",
                "manda nesse", "esse", "este", "certo"
            )
            or any(w in norm_user_msg for w in [
                "correto", "certo", "manter", "pode ser esse", "esse mesmo", "manda nesse",
                "tudo certo", "ta certo", "confirmo", "perfeito", "endereco correto"
            ])
        )

        # Check if user made a payment choice while waiting for address confirmation
        is_pix_choice = any(k in norm_user_msg for k in ["pagar no pix", "no pix", "pix", "chave pix", "a vista no pix", "opcao pix", "vou de pix"])
        is_boleto_choice = any(k in norm_user_msg for k in ["boleto", "boleto bancario", "pagar no boleto", "no boleto", "opcao boleto", "vou de boleto"])
        is_debit_choice = any(k in norm_user_msg for k in ["cartao de debito", "debito", "no debito", "pagar no debito", "pagar em debito"])
        is_combined_choice = any(k in norm_user_msg for k in ["combinado", "pagamento combinado", "pix e cartao", "pix + cartao", "combinar"])
        is_credit_choice = (
            not is_debit_choice
            and not is_combined_choice
            and any(k in norm_user_msg for k in ["cartao de credito", "credito", "cartao", "pagar no cartao", "pagar no cartao de credito", "parcelado", "parcelar", "vezes", "12x", "8x", "6x", "3x", "sem juros"])
        )
        has_payment_choice = is_pix_choice or is_credit_choice or is_boleto_choice or is_debit_choice or is_combined_choice

        if in_addr_confirm and has_payment_choice and not is_addr_affirmative and not is_addr_negative:
            cur_total = float(sess.get("cart_total") or 0.0)
            max_inst = 1
            for n in range(12, 0, -1):
                val = cur_total / n
                if n >= 7 and val >= 100.0:
                    max_inst = n
                    break
                if 4 <= n <= 6 and val >= 80.0:
                    max_inst = n
                    break
                if 2 <= n <= 3 and val >= 50.0:
                    max_inst = n
                    break
                if n == 1:
                    max_inst = 1
                    break

            if is_pix_choice:
                chosen_method = "pix"
                method_label = "À vista no PIX (5% de desconto)" if not sess.get("has_promo") else "À vista no PIX"
            elif is_boleto_choice:
                chosen_method = "boleto"
                method_label = "Boleto Bancário à vista (compensação em até 3 dias úteis)"
            elif is_debit_choice:
                chosen_method = "debit_card"
                method_label = "Cartão de Débito à vista"
            elif is_combined_choice:
                chosen_method = "combined"
                method_label = "Pagamento Combinado (Entrada no PIX + Cartão de Crédito)"
            else:
                chosen_method = "credit_card"
                method_label = f"Cartão de Crédito em até {max_inst}x sem juros"

            sess["payment_method"] = chosen_method
            sess["stage"] = "CHECKOUT_ADDRESS_CONFIRM"

            cust = sess.get("customer")
            if not cust and history:
                for msg in reversed(history):
                    c = msg.get("content", "")
                    if "@" in c:
                        em = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", c)
                        if em:
                            m = self.tool_executor.customer_repo.find_customers(em.group(0))
                            if m:
                                cust = m[0]
                                sess["customer"] = cust
                                break
                    d = re.sub(r"\D", "", c)
                    if 8 <= len(d) <= 12:
                        m = self.tool_executor.customer_repo.find_customers(d)
                        if m:
                            cust = m[0]
                            sess["customer"] = cust
                            break

            if cust:
                addr_disp = f"{cust.get('address', '')} - {cust.get('city', 'Campo Grande')}" + (f" (CEP: {cust.get('cep')})" if cust.get('cep') else "")
                return {
                    "reply": (
                        f"Ótima escolha! Forma de pagamento selecionada: **{method_label}**. 💳✨\n\n"
                        f"📍 **Endereço de entrega cadastrado:**\n"
                        f"**{addr_disp}**\n\n"
                        "O endereço de entrega está correto para o envio?"
                    ),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_checkout_address_confirm",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "checkout",
                    "action": None,
                    "options": ["Sim, endereço correto", "Não, alterar endereço"],
                    "logged_failed_intent": False,
                }

        if is_addr_negative:
            sess["stage"] = "CHECKOUT_ADDRESS_UPDATE"
            return {
                "reply": "Sem problemas! Preencha o formulário abaixo com o seu novo endereço de entrega:",
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_form_trigger",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": "SHOW_ADDRESS_FORM",
                "options": [],
                "logged_failed_intent": False,
            }

        if is_addr_affirmative:
            return self._build_final_checkout_summary(sess=sess, detected_lang=detected_lang)

        # 10. Checkout Flow: Payment Choice, Installment Selection, Final Confirmation
        in_checkout_confirm_stage = (
            in_addr_confirm
            or sess.get("stage") in (
                "CHECKOUT_PAYMENT_SELECT", "CHECKOUT_INSTALLMENT_SELECTION", "CHECKOUT_FINAL_CONFIRM"
            )
            or "como voce deseja realizar o pagamento" in clean_last_asst
            or "podemos confirmar e finalizar o seu pedido" in clean_last_asst
            or "opcoes de parcelamento no cartao" in clean_last_asst
            or "resumo final do pedido" in clean_last_asst
        )

        # 10.1. Address Update from Final Summary Review
        is_change_addr = in_checkout_confirm_stage and any(k in norm_user_msg for k in [
            "nao, alterar endereco", "alterar endereco", "mudar endereco", "trocar endereco",
            "endereco errado", "atualizar endereco", "mudar o endereco", "trocar o endereco", "novo endereco"
        ])
        if is_change_addr:
            sess["stage"] = "CHECKOUT_ADDRESS_UPDATE"
            return {
                "reply": "Sem problemas! Preencha o formulário abaixo com o novo endereço de entrega:",
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_form_trigger",
                "tools_called": [],
                "products": [],
                "active_agent": "customer",
                "action": "SHOW_ADDRESS_FORM",
                "options": [],
                "logged_failed_intent": False,
            }

        # 10.2. Change Payment Method ("Não, alterar pagamento")
        is_change_payment = in_checkout_confirm_stage and (
            any(k in norm_user_msg for k in [
                "nao, alterar pagamento", "alterar pagamento", "mudar pagamento",
                "trocar pagamento", "outra forma de pagamento", "mudar forma de pagamento",
                "trocar forma de pagamento", "escolher outro pagamento", "mudar a forma",
                "trocar a forma", "quero mudar o pagamento", "quero trocar o pagamento"
            ])
            or (sess.get("stage") == "CHECKOUT_FINAL_CONFIRM" and norm_user_msg in ("nao", "nao quero", "alterar", "mudar"))
        )
        if is_change_payment:
            sess["stage"] = "CHECKOUT_PAYMENT_SELECT"
            return self._build_final_checkout_summary(sess=sess, detected_lang=detected_lang)

        # 10.3. Final Order Confirmation in CHECKOUT_FINAL_CONFIRM or with selected payment method
        is_order_confirmation = (
            (sess.get("stage") == "CHECKOUT_FINAL_CONFIRM" or (sess.get("stage") == "CHECKOUT_PAYMENT_SELECT" and sess.get("payment_method")))
            and not is_change_payment
            and not is_change_addr
            and (
                norm_user_msg in (
                    "sim, confirmar pedido", "confirmar pedido", "confirmar", "sim", "pode confirmar",
                    "finalizar pedido", "finalizar", "concluir pedido", "concluir", "fechar",
                    "sim confirmar pedido", "pode fechar", "pode concluir", "sim pode confirmar",
                    "pode finalizar", "tudo certo", "tudo certo pode finalizar", "fechado", "beleza",
                    "manda ver", "com certeza", "ok", "pode ser", "confirmo", "perfeito", "isso",
                    "positivo", "confirmado", "aprovar", "pode gerar", "sim, confirmar", "sim por favor",
                    "sim tudo certo", "com certeza pode confirmar", "pode mandar", "mandar bala"
                )
                or any(k in norm_user_msg for k in [
                    "confirmar pedido", "pode confirmar", "finalizar pedido", "concluir pedido",
                    "pode fechar", "pode finalizar", "tudo certo pode", "sim confirmo"
                ])
            )
        )

        # 10.4. Installment Pattern & Method Detection
        is_single_digit_inst = (
            sess.get("stage") == "CHECKOUT_INSTALLMENT_SELECTION"
            and norm_user_msg.isdigit()
            and 1 <= int(norm_user_msg) <= 12
        )
        is_pix_select = in_checkout_confirm_stage and any(k in norm_user_msg for k in [
            "pagar no pix", "no pix", "pix", "chave pix", "a vista no pix", "vou de pix",
            "quero pix", "manda o pix", "prefiro pix", "opcao pix"
        ])
        is_boleto_select = in_checkout_confirm_stage and any(k in norm_user_msg for k in [
            "boleto", "boleto bancario", "pagar no boleto", "no boleto", "opcao boleto",
            "vou de boleto", "gera boleto", "quero boleto"
        ])
        is_debit_select = in_checkout_confirm_stage and any(k in norm_user_msg for k in [
            "cartao de debito", "debito", "no debito", "pagar no debito", "opcao debito"
        ])
        is_card_vista_select = in_checkout_confirm_stage and any(k in norm_user_msg for k in [
            "pagar a vista no cartao", "cartao a vista", "1x no cartao", "a vista no cartao",
            "1x", "1 vez", "uma vez", "uma parcela"
        ])
        is_combined_select = in_checkout_confirm_stage and any(k in norm_user_msg for k in [
            "pagamento combinado", "combinado", "pix e cartao", "pix + cartao", "metade no pix"
        ])

        # 10.5. Request installment options (Parcelar no Cartão / Opções de parcelas)
        is_request_installments = (
            in_checkout_confirm_stage
            and not is_order_confirmation
            and not is_pix_select
            and not is_boleto_select
            and not is_debit_select
            and not is_card_vista_select
            and not is_combined_select
            and any(k in norm_user_msg for k in [
                "parcelar", "parcelamento", "ver parcelas", "opcoes de parcelamento",
                "quero parcelar", "tabela de parcelas", "quantas vezes", "dividir no cartao",
                "cartao de credito", "cartao", "no cartao", "credito", "pagar no cartao",
                "pagar com cartao", "dividir em parcelas", "parcelas"
            ])
            and not re.search(r"\b\d+\s*x\s+de\b", norm_user_msg)
            and not re.search(r"^\s*\d+\s*x\s*$", norm_user_msg)
            and not re.search(r"^\s*\d+\s*(?:vezes|parcelas)\s*$", norm_user_msg)
            and not re.search(r"\bem\s+\d+\s*x\b", norm_user_msg)
            and not re.search(r"\bem\s+\d+\s*(?:vezes|parcelas)\b", norm_user_msg)
            and not is_single_digit_inst
        )

        inst_match = None
        if not is_request_installments:
            inst_match = re.search(r"\b(\d+)\s*x\b", norm_user_msg) or re.search(r"\b(\d+)\s*(?:vezes|parcelas)\b", norm_user_msg)
        if is_request_installments:
            sess["stage"] = "CHECKOUT_INSTALLMENT_SELECTION"
            cust = sess.get("customer") or {}
            cust_name = cust.get("name", "Cliente")
            total = float(sess.get("cart_total") or 0.0)
            city = cust.get("city", "Campo Grande")
            is_campo_grande = "campo grande" in city.lower()
            shipping_cost = 0.0 if (is_campo_grande and (total >= 300.0 or total == 0.0)) else (25.0 if is_campo_grande else 45.0)
            total_final = total + shipping_cost

            def _fmt(val: float) -> str:
                return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            valid_insts = []
            max_inst = 1
            for n in range(1, 13):
                val = round(total_final / n, 2)
                if n == 1:
                    ok = True
                elif 2 <= n <= 3:
                    ok = val >= 50.0
                elif 4 <= n <= 6:
                    ok = val >= 80.0
                else:
                    ok = val >= 100.0
                if ok:
                    max_inst = n
                    valid_insts.append((n, val))

            sess["max_inst"] = max_inst
            sess["valid_insts"] = valid_insts

            inst_lines = [
                "💳 **Opções de Parcelamento no Cartão de Crédito sem Juros** 🎸✨\n",
                f"Para o valor total do seu pedido (**{_fmt(total_final)}**), confira abaixo as opções liberadas pelas políticas da loja:\n",
                "| Parcelas | Valor de cada Parcela | Total sem Juros |",
                "| :---: | :---: | :---: |",
            ]
            inst_options = []
            for n, val in valid_insts:
                inst_lines.append(f"| {n}x | {_fmt(val)} | {_fmt(total_final)} |")
                inst_options.append(f"💳 {n}x de {_fmt(val)}")

            inst_lines.append("\n📌 *Políticas da loja: parcela mínima de R$ 50,00 para até 3x; R$ 80,00 de 4x a 6x; e R$ 100,00 de 7x a 12x sem juros.*")
            inst_lines.append("🔒 *Para fins de simulação e segurança, não é necessário digitar os dados confidenciais do seu cartão.*\n")
            inst_lines.append(f"**{cust_name}**, por favor selecione abaixo em quantas parcelas deseja finalizar o seu pedido:")

            has_promo = bool(sess.get("has_promo", False))
            pix_discount = (total * 0.05) if not has_promo else 0.0
            inst_options.append("⚡ Pagar no PIX (5% off)" if pix_discount > 0 else "⚡ Pagar no PIX")

            return {
                "reply": "\n".join(inst_lines),
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_checkout_installments",
                "tools_called": [],
                "products": [],
                "active_agent": "checkout",
                "action": None,
                "options": inst_options,
                "logged_failed_intent": False,
            }

        # 10.6. Payment Method / Installment Selection -> Displays Final Order Review
        if not is_order_confirmation and in_checkout_confirm_stage:
            if is_pix_select:
                return self._build_final_order_review(sess=sess, detected_lang=detected_lang, method="pix")
            elif inst_match or is_single_digit_inst:
                chosen_n = int(inst_match.group(1)) if inst_match else int(norm_user_msg)

                # Compute current order total and max allowed installments deterministically
                total = float(sess.get("cart_total") or 0.0)
                cust = sess.get("customer") or {}
                city = cust.get("city", "Campo Grande")
                is_campo_grande = "campo grande" in city.lower()
                shipping_cost = 0.0 if (is_campo_grande and (total >= 300.0 or total == 0.0)) else (25.0 if is_campo_grande else 45.0)
                total_final = total + shipping_cost

                valid_insts = []
                max_inst = 1
                for n in range(1, 13):
                    val = round(total_final / n, 2)
                    if n == 1:
                        ok = True
                    elif 2 <= n <= 3:
                        ok = val >= 50.0
                    elif 4 <= n <= 6:
                        ok = val >= 80.0
                    else:
                        ok = val >= 100.0
                    if ok:
                        max_inst = n
                        valid_insts.append((n, val))

                sess["max_inst"] = max_inst
                sess["valid_insts"] = valid_insts

                # Validate if chosen_n is within allowed limits
                if chosen_n > 12 or chosen_n > max_inst or chosen_n < 1:
                    def _fmt(val: float) -> str:
                        return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                    cust_name = cust.get("name", "Cliente")
                    greeting = f"Olá, **{cust_name}**!" if cust_name else "Olá!"
                    max_val = round(total_final / max_inst, 2)

                    if chosen_n > 12:
                        reason = (
                            f"O parcelamento em **{chosen_n}x ultrapassa o limite máximo** permitido pela loja. "
                            "Na **Empório da Música**, o parcelamento com cartão de crédito é em até **12x sem juros** "
                            "(conforme a Seção 3 das Políticas da Loja)."
                        )
                    else:
                        reason = (
                            f"O parcelamento em **{chosen_n}x não está disponível** para este valor. "
                            "Pelas regras da loja (Seção 3), a parcela mínima deve ser de R$ 50,00 (até 3x), R$ 80,00 (4x a 6x) e R$ 100,00 (7x a 12x)."
                        )

                    lines = [
                        "⚠️ **Condição de Parcelamento Indisponível** 🎸💳\n",
                        f"{greeting} {reason}\n",
                        f"Para o valor total do seu pedido (**{_fmt(total_final)}**), o parcelamento máximo liberado é de até **{max_inst}x de {_fmt(max_val)} sem juros**.\n",
                        "| Parcelas | Valor de cada Parcela | Total sem Juros |",
                        "| :---: | :---: | :---: |",
                    ]
                    inst_options = []
                    for n, val in valid_insts:
                        lines.append(f"| {n}x | {_fmt(val)} | {_fmt(total_final)} |")
                        inst_options.append(f"💳 {n}x de {_fmt(val)}")

                    has_promo = bool(sess.get("has_promo", False))
                    pix_discount = (total * 0.05) if not has_promo else 0.0
                    inst_options.append("⚡ Pagar no PIX (5% off)" if pix_discount > 0 else "⚡ Pagar no PIX")

                    lines.append(f"\nPor favor, escolha uma das opções válidas acima (de **1x até {max_inst}x sem juros**) ou opte pelo pagamento à vista no PIX com 5% de desconto:")

                    sess["stage"] = "CHECKOUT_INSTALLMENT_SELECTION"
                    return {
                        "reply": "\n".join(lines),
                        "language": detected_lang,
                        "confidence": 1.0,
                        "model_used": "direct_checkout_installments",
                        "tools_called": [],
                        "products": [],
                        "active_agent": "checkout",
                        "action": None,
                        "options": inst_options,
                        "logged_failed_intent": False,
                    }

                return self._build_final_order_review(sess=sess, detected_lang=detected_lang, method="credit_card", installments=chosen_n)
            elif is_card_vista_select:
                return self._build_final_order_review(sess=sess, detected_lang=detected_lang, method="credit_card", installments=1)
            elif is_boleto_select:
                return self._build_final_order_review(sess=sess, detected_lang=detected_lang, method="boleto")
            elif is_debit_select:
                return self._build_final_order_review(sess=sess, detected_lang=detected_lang, method="debit_card")
            elif is_combined_select:
                return self._build_final_order_review(sess=sess, detected_lang=detected_lang, method="combined")

        # 10.7. Final Order Confirmation after Customer Review
        should_finalize = (
            in_checkout_confirm_stage and is_order_confirmation
        )

        if should_finalize:
            cust = sess.get("customer") or {}
            cust_name = cust.get("name", "Cliente")
            cust_phone = cust.get("phone", "")
            city = cust.get("city", "Campo Grande")
            is_campo_grande = "campo grande" in city.lower()

            def _fmt(val: float) -> str:
                return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            total = float(sess.get("cart_total") or 0.0)
            shipping_cost = 0.0 if (is_campo_grande and (total >= 300.0 or total == 0.0)) else (25.0 if is_campo_grande else 45.0)
            total_final = total + shipping_cost

            # Retrieve payment method from session
            method = sess.get("payment_method") or "pix"

            sess["payment_method"] = method

            final_lines = [
                "🎉 **Pedido Confirmado com Sucesso!** 🎸✨\n",
                f"Muito obrigado pela sua compra, **{cust_name}**! Seu pedido foi registrado no sistema.",
            ]

            if method == "pix":
                has_promo = bool(sess.get("has_promo", False))
                discount_val = (total * 0.05) if not has_promo else 0.0
                total_final = (total - discount_val) + shipping_cost
                txid = f"EMP{abs(hash(session_id)) % 1000000:06d}"
                copia_e_cola = PixService.generate_pix_copia_e_cola(amount=total_final, txid=txid)
                amt_str = f"{total_final:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                final_lines.append("\n📱 **QR Code PIX para Pagamento:**")
                final_lines.append(f"![QR Code PIX](/api/pix-qrcode?amount={amt_str}&name={cust_name}&txid={txid})\n")
                final_lines.append("📋 **Código PIX Copia e Cola:**")
                final_lines.append(f"`{copia_e_cola}`\n")
                final_lines.append(
                    f"📲 **Acompanhamento do Pedido:**\n"
                    f"Entraremos em contato pelo seu WhatsApp (**{cust_phone or 'cadastrado'}**) assim que o pagamento for compensado para passar o status do seu pedido!"
                )
            elif method == "boleto":
                linha_digitavel = f"34191.79001 01043.510047 91020.150008 4 {int(total_final*100):014d}"
                final_lines.append("\n📄 **Boleto Bancário Gerado com Sucesso!**\n")
                final_lines.append("🔢 **Linha Digitável do Boleto:**")
                final_lines.append(f"`{linha_digitavel}`\n")
                final_lines.append("⏱️ **Prazo de Compensação:** Até 3 dias úteis.")
                final_lines.append(
                    f"\n📲 **Acompanhamento do Pedido:**\n"
                    f"Assim que o pagamento for compensado pelo banco, entraremos em contato pelo seu WhatsApp (**{cust_phone or 'cadastrado'}**) para darmos início ao envio!"
                )
            elif method == "debit_card":
                final_lines.append(f"\n💳 **Pagamento no Cartão de Débito Aprovado com Sucesso!** ({_fmt(total_final)} à vista)")
                final_lines.append("🔒 *Simulação autenticada com segurança sem necessidade de dados confidenciais do cartão.*")
                final_lines.append(
                    f"\n📲 **Acompanhamento do Pedido:**\n"
                    f"Entraremos em contato pelo seu WhatsApp (**{cust_phone or 'cadastrado'}**) com as etapas de entrega do seu pedido!"
                )
            elif method == "combined":
                half_pix = total * 0.5
                half_card = total_final - half_pix
                txid = f"EMP{abs(hash(session_id)) % 1000000:06d}"
                copia_e_cola = PixService.generate_pix_copia_e_cola(amount=half_pix, txid=txid)
                amt_str = f"{half_pix:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                final_lines.append("\n🔀 **Pagamento Combinado Registrado com Sucesso:**")
                final_lines.append(f"• **Entrada no PIX (50%):** R$ {amt_str}")
                final_lines.append(f"• **Saldo no Cartão de Crédito (50%):** {_fmt(half_card)} sem juros (simulação aprovada)\n")
                final_lines.append("📱 **QR Code PIX para a Entrada:**")
                final_lines.append(f"![QR Code PIX](/api/pix-qrcode?amount={amt_str}&name={cust_name}&txid={txid})\n")
                final_lines.append("📋 **Código PIX Copia e Cola:**")
                final_lines.append(f"`{copia_e_cola}`\n")
                final_lines.append(
                    f"📲 **Acompanhamento do Pedido:**\n"
                    f"Entraremos em contato pelo seu WhatsApp (**{cust_phone or 'cadastrado'}**) assim que a entrada for compensada para despachar seu pedido!"
                )
            else:
                # Credit Card
                chosen_n = sess.get("installments")
                if not chosen_n:
                    max_inst = 1
                    for n in range(12, 0, -1):
                        val = total_final / n
                        if n >= 7 and val >= 100.0:
                            max_inst = n
                            break
                        if 4 <= n <= 6 and val >= 80.0:
                            max_inst = n
                            break
                        if 2 <= n <= 3 and val >= 50.0:
                            max_inst = n
                            break
                        if n == 1:
                            max_inst = 1
                            break
                    chosen_n = max_inst

                chosen_inst_val = total_final / chosen_n
                inst_desc = f"{chosen_n}x de {_fmt(chosen_inst_val)} sem juros" if chosen_n > 1 else f"1x de {_fmt(total_final)} à vista"
                final_lines.append(f"\n💳 **Pagamento no Cartão de Crédito Aprovado em {inst_desc}!** (Total: {_fmt(total_final)})")
                final_lines.append("🔒 *Por segurança e privacidade, a simulação foi autenticada sem necessidade de digitar os dados confidenciais do seu cartão.*")
                final_lines.append(
                    f"\n📲 **Acompanhamento do Pedido:**\n"
                    f"Entraremos em contato pelo seu WhatsApp (**{cust_phone or 'cadastrado'}**) com a confirmação e código de rastreamento do envio!"
                )

            if is_campo_grande:
                final_lines.append(f"\n🚚 A entrega será realizada pelo nosso motoboy próprio da loja com agendamento telefônico prévio.")
            else:
                final_lines.append(f"\n🚚 Seu pedido será despachado via transportadora parceira. Enviaremos o código de rastreamento no seu WhatsApp/Telefone ({cust_phone or 'cadastrado'}).")

            final_lines.append("\nAgradecemos por comprar na **Empório da Música**! Sua música começa aqui! 🎶✨")
            final_lines.append(
                "\n---\n"
                "⭐ **Pesquisa de Satisfação:**\n"
                f"**{cust_name}**, como você avalia o nosso atendimento hoje? (Por favor, selecione ou digite uma nota de **1 a 5**, onde 1 é insatisfeito e 5 é excelente):"
            )

            sess["stage"] = "SATISFACTION_SURVEY"
            return {
                "reply": "\n".join(final_lines),
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_checkout_confirmed",
                "tools_called": [],
                "products": [],
                "active_agent": "checkout",
                "action": "CLEAR_CART",
                "options": [
                    "⭐ 1 (Ruim)",
                    "⭐ 2 (Regular)",
                    "⭐ 3 (Bom)",
                    "⭐ 4 (Muito bom)",
                    "⭐ 5 (Excelente)",
                ],
                "logged_failed_intent": False,
            }

        # 10.8. Customer Satisfaction Survey Response (CSAT 1-5) & Attendance Conclusion
        if sess.get("stage") == "SATISFACTION_SURVEY":
            cust = sess.get("customer") or {}
            cust_name = cust.get("name", "Cliente")

            if any(w in norm_user_msg for w in ["novo atendimento", "reiniciar", "outro atendimento", "novo pedido"]):
                sess["stage"] = "SHOPPING"
                sess["cart_items"] = []
                sess["cart_total"] = 0.0
                sess["payment_method"] = None
                return {
                    "reply": f"Com certeza, **{cust_name}**! Vamos iniciar um novo atendimento. Como posso te ajudar hoje? 🎸🎶",
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_new_attendance",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "catalog",
                    "action": "CLEAR_CART",
                    "options": ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção"],
                    "logged_failed_intent": False,
                }

            # Extract rating 1-5 or descriptive sentiment
            rating = None
            m_digit = re.search(r"\b([1-5])\b", norm_user_msg)
            if m_digit:
                rating = int(m_digit.group(1))
            elif any(w in norm_user_msg for w in ["excelente", "otim", "maravilh", "perfeit", "adorei", "show", "nota dez", "dez"]):
                rating = 5
            elif any(w in norm_user_msg for w in ["muito bom", "muito boa"]):
                rating = 4
            elif any(w in norm_user_msg for w in ["bom", "boa"]):
                rating = 3
            elif any(w in norm_user_msg for w in ["regular", "medio", "mais ou menos"]):
                rating = 2
            elif any(w in norm_user_msg for w in ["ruim", "pessim", "horrivel"]):
                rating = 1

            sess["csat_rating"] = rating or 5
            sess["stage"] = "ATTENDANCE_CLOSED"

            star_display = f" ({rating} ⭐)" if rating else ""
            reply_text = (
                f"🙏 **Agradecemos muito pela sua avaliação{star_display}, {cust_name}!** 🎸✨\n\n"
                "Sua avaliação foi registrada com sucesso e é fundamental para continuarmos aprimorando nosso atendimento e oferecendo os melhores instrumentos musicais.\n\n"
                "O seu atendimento foi **concluído e encerrado com sucesso**. Agradecemos pela preferência e confiança na **Empório da Música** (*Sua música começa aqui!*).\n\n"
                "Tenha um excelente dia e aproveite muito seu novo instrumento musical! Até logo! 👋🎶"
            )

            return {
                "reply": reply_text,
                "language": detected_lang,
                "confidence": 1.0,
                "model_used": "direct_csat_closure",
                "tools_called": [],
                "products": [],
                "active_agent": "checkout",
                "action": "CLOSE_CHAT",
                "options": ["🔄 Novo Atendimento", "📍 Informações da Loja"],
                "logged_failed_intent": False,
            }

        if sess.get("stage") == "ATTENDANCE_CLOSED":
            cust = sess.get("customer") or {}
            cust_name = cust.get("name", "Cliente")

            if any(w in norm_user_msg for w in ["novo", "reiniciar", "comecar", "outro", "comprar", "violao", "guitarra", "ukulele", "catalogo", "ver", "ola", "oi", "quero"]):
                sess["stage"] = "SHOPPING"
                sess["cart_items"] = []
                sess["cart_total"] = 0.0
                sess["payment_method"] = None
                # Allow proceeding to shopping or normal catalog routing below
            else:
                return {
                    "reply": (
                        f"O seu atendimento anterior foi encerrado com sucesso, **{cust_name}**! 😊\n\n"
                        "Se desejar iniciar uma nova consulta, explorar novos instrumentos ou tirar dúvidas, basta clicar em **Novo Atendimento** abaixo! 🎸"
                    ),
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_attendance_closed",
                    "tools_called": [],
                    "products": [],
                    "active_agent": "checkout",
                    "action": None,
                    "options": ["🔄 Novo Atendimento", "🎸 Ver Catálogo", "📍 Informações da Loja"],
                    "logged_failed_intent": False,
                }

        # 11. Route to dedicated Specialist Agent
        stage = IntentRouter.classify_stage(user_message=user_message, chat_history=history)
        specialist = self.specialists.get(stage, self.catalog_agent)

        # 12. Execute the specialist agent
        result = specialist.execute(
            user_message=user_message,
            session_id=session_id,
            chat_history=history,
            claimed_customer_name=claimed_name,
            session_context=sess,
        )
        if result and "reply" in result:
            from src.agents.specialists import clean_carousel_mentions
            result["reply"] = clean_carousel_mentions(result["reply"])

        return result
