"""Specialist agents with single responsibility (Catalog, Checkout, Customer, Support)."""

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
from groq import Groq

from src.agents.prompts import detect_user_language
from src.agents.tools import TOOL_SCHEMAS, ToolExecutor, is_name_match
from src.config.settings import settings
from src.repositories.product_repository import normalize_text


# Map all tool schemas by function name for clean selective scoping
TOOL_SCHEMAS_BY_NAME: Dict[str, Dict[str, Any]] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}


def clean_thinking_tags(text: str) -> str:
    """Remove internal <think>...</think> reasoning traces from model outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def make_llm_tool_payload(fn_name: str, tool_output: Any) -> Any:
    """
    Produce a lightweight, token-efficient JSON payload for LLM function response.
    Protects TPM limits while preserving essential attributes.
    """
    if not isinstance(tool_output, dict):
        return tool_output

    if fn_name == "search_instruments" and "products" in tool_output:
        raw_list = tool_output.get("products") or []
        sample = []
        for p in raw_list[:10]:
            item = {
                "id": p.get("product_id"),
                "name": p.get("name"),
                "category": p.get("category_name"),
                "price": p.get("effective_price_brl"),
                "pix_price": p.get("pix_price_brl"),
                "stock": p.get("stock_quantity"),
                "in_stock": p.get("in_stock"),
            }
            if p.get("active_promotion"):
                item["promo_pct"] = p["active_promotion"].get("discount_percent")
            if p.get("strings_type"):
                item["strings"] = p["strings_type"]
            sample.append(item)
        return {
            "category": tool_output.get("category"),
            "total_matches": len(raw_list),
            "sample_products": sample,
        }

    if fn_name == "get_instrument_details_and_pricing" and "product" in tool_output:
        p = tool_output.get("product") or {}
        alts = tool_output.get("recommended_alternatives_in_stock") or []
        return {
            "found": tool_output.get("found", True),
            "product": {
                "id": p.get("product_id"),
                "name": p.get("name"),
                "price": p.get("effective_price_brl"),
                "pix_price": p.get("pix_price_brl"),
                "in_stock": p.get("in_stock"),
                "stock_quantity": p.get("stock_quantity"),
                "status": p.get("status"),
                "promotion": p.get("active_promotion"),
            },
            "payment_simulation": tool_output.get("payment_simulation"),
            "shipping_simulation": tool_output.get("shipping_simulation"),
            "alternatives": [
                {
                    "id": a.get("product_id"),
                    "name": a.get("name"),
                    "price": a.get("price_brl"),
                    "stock": a.get("stock_quantity"),
                }
                for a in alts[:3]
            ],
        }

    return tool_output


def clean_carousel_mentions(text: str) -> str:
    """
    Ensure the word 'carrossel' or 'carousel' is never presented to the customer.
    Substitutes with 'confira abaixo a lista de opções disponíveis...' or equivalent natural phrasing.
    """
    if not text:
        return text

    # 1. "Confira abaixo no carrossel as opções disponíveis..." -> "Confira abaixo a lista de opções disponíveis..."
    text = re.sub(
        r"(?i)\bconfira\s+abaixo\s+no\s+carrossel\s+as\s+op[çc][õo]es\s+dispon[íi]veis\b",
        "Confira abaixo a lista de opções disponíveis",
        text,
    )
    # 2. "Confira no carrossel (abaixo )?as opções disponíveis..." -> "Confira abaixo a lista de opções disponíveis..."
    text = re.sub(
        r"(?i)\bconfira\s+no\s+carrossel\s+(?:abaixo\s+)?as\s+op[çc][õo]es\s+dispon[íi]veis\b",
        "Confira abaixo a lista de opções disponíveis",
        text,
    )
    # 3. "Dá/Dê uma olhada no carrossel (abaixo)?" -> "Confira abaixo a lista de opções disponíveis"
    text = re.sub(
        r"(?i)\bd[áâe]\s+uma\s+olhada\s+no\s+carrossel(?:\s+abaixo)?\b",
        "Confira abaixo a lista de opções disponíveis",
        text,
    )
    # 4. "Explore/Veja os cards/cartões (abaixo )?(no carrossel)?" -> "Confira abaixo a lista de opções disponíveis"
    text = re.sub(
        r"(?i)\b(?:explore|veja|olhe)\s+(?:os\s+)?(?:cards|cart[õo]es)(?:\s+abaixo)?(?:\s+no\s+carrossel)?\b",
        "Confira abaixo a lista de opções disponíveis",
        text,
    )
    # 5. "no carrossel abaixo" -> "abaixo na lista de opções disponíveis"
    text = re.sub(
        r"(?i)\bno\s+carrossel\s+abaixo\b",
        "abaixo na lista de opções disponíveis",
        text,
    )
    # 6. "no carrossel" -> "na lista de opções disponíveis"
    text = re.sub(
        r"(?i)\bno\s+carrossel\b",
        "na lista de opções disponíveis",
        text,
    )
    # 7. "do carrossel" -> "da lista de opções disponíveis"
    text = re.sub(
        r"(?i)\bdo\s+carrossel\b",
        "da lista de opções disponíveis",
        text,
    )
    # 8. "o carrossel abaixo" -> "a lista de opções disponíveis abaixo"
    text = re.sub(
        r"(?i)\bo\s+carrossel\s+abaixo\b",
        "a lista de opções disponíveis abaixo",
        text,
    )
    # 9. "o carrossel" -> "a lista de opções disponíveis"
    text = re.sub(
        r"(?i)\bo\s+carrossel\b",
        "a lista de opções disponíveis",
        text,
    )
    # 10. Standalone "carrossel" / "carrosséis" / "carousel"
    text = re.sub(
        r"(?i)\bcarross[eé]is?\b",
        "lista de opções disponíveis",
        text,
    )
    text = re.sub(
        r"(?i)\bcarousels?\b",
        "lista de opções disponíveis",
        text,
    )
    return text


def sanitize_catalog_reply(text: str, products: List[Dict[str, Any]]) -> str:
    """
    Remove redundant markdown tables and repetitive product itemization from the text
    when the interactive Product Carousel cards are already presented to the user.
    """
    if not products or not text:
        return clean_carousel_mentions(text or "")

    lines = text.split("\n")
    cleaned_lines = []
    in_table = False

    prod_names = [p["name"].lower() for p in products if isinstance(p, dict) and "name" in p]

    for line in lines:
        stripped = line.strip()
        # 1. Detect and strip Markdown table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            continue
        elif in_table and stripped.startswith("|"):
            continue
        else:
            in_table = False

        # 2. Skip bullet points repeating individual product items with prices/specs
        is_bullet = stripped.startswith(("- ", "• ", "* ")) or bool(re.match(r"^\d+\.\s+", stripped))
        has_prod_match = any(p_name in stripped.lower() for p_name in prod_names)
        has_price_match = bool(re.search(r"r\$\s*\d+", stripped, re.IGNORECASE))
        if is_bullet and (has_prod_match or (has_price_match and any(term in stripped.lower() for term in ["pix", "estoque", "unidades", "tampos", "cordas"]))):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    # Remove 3 or more consecutive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # If the text became too short after removing the table/list
    if len(cleaned) < 25:
        count = len(products)
        item_word = "instrumento encontrado" if count == 1 else f"{count} instrumentos encontrados"
        cleaned = (
            f"Encontrei **{item_word}** disponível(is) em estoque para você! 🎸✨\n\n"
            "Confira abaixo a lista de opções disponíveis para ver fotos, detalhes e adicionar ao seu carrinho com 5% de desconto no PIX. "
            "Se quiser que eu filtre por categoria ou faixa de preço, é só me dizer! 🎶"
        )

    return clean_carousel_mentions(cleaned)


class BaseSpecialistAgent:
    """Base class providing LLM fallback execution loop and tool calling capabilities."""

    name: str = "base"
    allowed_tool_names: List[str] = []

    def __init__(
        self,
        tool_executor: ToolExecutor,
        client: Optional[Groq] = None,
        model_chain: Optional[List[str]] = None,
    ) -> None:
        self.tool_executor = tool_executor
        self.client = client
        self.model_chain = model_chain or [
            settings.primary_model,
            settings.fallback_model_1,
            settings.fallback_model_2,
        ]

    @property
    def scoped_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return only the tool schemas allowed for this specialist agent."""
        return [
            TOOL_SCHEMAS_BY_NAME[name]
            for name in self.allowed_tool_names
            if name in TOOL_SCHEMAS_BY_NAME
        ]

    def get_system_prompt(self, user_language: str) -> str:
        """Subclasses override this to return their focused, single-responsibility system prompt."""
        raise NotImplementedError

    def _call_llm_with_fallbacks(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> tuple[Any, str]:
        """Invoke Groq Chat Completions with primary Qwen3 and Llama fallbacks."""
        if not self.client:
            raise RuntimeError("Groq API client is not initialized (missing GROQ_API_KEY).")

        temp = temperature if temperature is not None else settings.llm_temperature
        last_error: Optional[Exception] = None

        for model_name in self.model_chain:
            for attempt in range(2):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": 1100,
                    }
                    if tools:
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = "auto"

                    response = self.client.chat.completions.create(**kwargs)
                    return response, model_name
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc).lower()
                    if ("429" in err_str or "413" in err_str or "rate_limit" in err_str) and attempt == 0:
                        time.sleep(2.5)
                        continue
                    break

        raise RuntimeError(f"All LLM models in fallback chain failed. Last error: {last_error}")

    def execute(
        self,
        user_message: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        claimed_customer_name: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the agent loop for this specialist."""
        detected_lang = detect_user_language(user_message)
        system_content = self.get_system_prompt(detected_lang)

        # Inject rich session state and customer/cart context so LLM never loses track
        if session_context:
            cust = session_context.get("customer") or {}
            cust_name = cust.get("name") or claimed_customer_name or "Cliente"
            cust_phone = cust.get("phone", "")
            cust_email = cust.get("email", "")
            cust_addr = cust.get("address", "")
            cust_city = cust.get("city", "")
            cust_cep = cust.get("cep", "")

            cart_items = session_context.get("cart_items") or []
            cart_total = float(session_context.get("cart_total") or 0.0)
            stage = session_context.get("stage", "SHOPPING")

            cart_desc = ", ".join(f"{qty}x {p['name']}" for p, qty in cart_items) if cart_items else "Nenhum item adicionado"

            context_block = f"""

### CURRENT CONVERSATION & SESSION CONTEXT:
- Customer Identified: {cust_name} (Phone: {cust_phone or 'Not set'}, Email: {cust_email or 'Not set'}, City: {cust_city or 'Not set'}, Address: {cust_addr or 'Not set'}, CEP: {cust_cep or 'Not set'})
- Active Cart: {cart_desc} (Total: R$ {cart_total:,.2f})
- Current Flow Stage: {stage}
- CRITICAL CONTINUITY DIRECTIVE: Maintain full conversational continuity. You know exactly what products are in the customer's cart and what stage they are in. If the customer indicates an intent to continue shopping, confirm address, choose payment, or finalize their purchase, guide them smoothly without asking them to repeat details or re-enter items already in context.
"""
            system_content += context_block

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]

        # Inject recent chat history (last 8 turns)
        history = chat_history or []
        for msg in history[-8:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                c = msg["content"]
                if len(c) > 600:
                    c = c[:600] + "..."
                messages.append({"role": msg["role"], "content": c})

        messages.append({"role": "user", "content": user_message})

        tools_called_log: List[Dict[str, Any]] = []
        collected_products: Dict[int, Dict[str, Any]] = {}
        model_used = self.model_chain[0]
        tools_to_offer = self.scoped_tool_schemas or None

        try:
            for _ in range(4):
                response, model_used = self._call_llm_with_fallbacks(
                    messages=messages,
                    tools=tools_to_offer,
                    temperature=settings.llm_temperature,
                )
                choice_msg = response.choices[0].message
                tool_calls = getattr(choice_msg, "tool_calls", None)

                if not tool_calls:
                    final_text = clean_thinking_tags(choice_msg.content or "")

                    # Catalog Guardrail: If catalog agent generated text without tool calls
                    # but the user asked about instruments or categories, execute search_instruments
                    # to prevent hallucination (e.g. fake brands) and guarantee carousel display!
                    if self.name == "catalog" and not collected_products:
                        norm_u = normalize_text(user_message)
                        cat_found = None
                        for c_name, aliases in [
                            ("Ukuleles", ["ukulele", "ukuleles"]),
                            ("Violões", ["violao", "violoes", "violão", "violões", "acoustic guitar"]),
                            ("Guitarras", ["guitarra", "guitarras", "stratocaster", "telecaster", "les paul", "electric guitar"]),
                            ("Baixos", ["baixo", "baixos", "contrabaixo", "bass"]),
                            ("Baterias e Percussão", ["bateria", "baterias", "percussao", "drums"]),
                            ("Teclados e Pianos", ["teclado", "teclados", "piano", "pianos", "sintetizador", "keyboard"]),
                            ("Instrumentos de Sopro (Madeiras)", ["sopro", "sax", "flauta", "clarinete"]),
                            ("Instrumentos de Sopro (Metais)", ["trompete", "trombone", "trompa"]),
                            ("Cordas Orquestrais", ["violino", "viola", "violoncelo", "orquestral"]),
                        ]:
                            if any(a in norm_u for a in aliases):
                                cat_found = c_name
                                break

                        p_match = re.search(r"(?:ate|maximo|menos de|custando ate)\s*(?:r\$)?\s*(\d+(?:\.\d+)?)", norm_u)
                        max_p = float(p_match.group(1)) if p_match else None
                        only_promo = "promocao" in norm_u or "promocoes" in norm_u

                        brand_found = None
                        for b in ["yamaha", "takamine", "tagima", "giannini", "fender", "ibanez", "shelby", "kalani", "kala", "ohana", "roland", "casio", "cort", "rozini"]:
                            if b in norm_u:
                                brand_found = b
                                break

                        catalog_keywords = [
                            "instrumento", "instrumentos", "produto", "produtos", "opcao", "opcoes",
                            "opção", "opções", "catalogo", "catálogo", "modelo", "modelos", "preco",
                            "preço", "valor", "valores", "reais", "recomenda", "recomendacao",
                            "barato", "baratos", "disponivel", "disponiveis", "estoque",
                            "ver catalogo", "ver produtos", "ver instrumentos", "quais", "tem"
                        ]

                        is_order_inquiry = any(k in norm_u for k in [
                            "pedido", "pedidos", "rastreio", "rastreamento", "compras anteriores",
                            "minhas compras", "minha compra", "historico", "garantia", "troca", "devolucao"
                        ])

                        if not is_order_inquiry and (cat_found or brand_found or max_p is not None or only_promo or any(w in norm_u for w in catalog_keywords)):
                            forced_args = {
                                "category_query": cat_found,
                                "query": brand_found,
                                "max_price": max_p,
                                "only_available": True,
                                "only_on_promotion": only_promo,
                                "limit": 60,
                            }
                            tool_output = self.tool_executor.execute("search_instruments", forced_args)
                            tools_called_log.append({"tool": "search_instruments", "arguments": forced_args})

                            if isinstance(tool_output, dict) and "products" in tool_output:
                                for prod in tool_output["products"]:
                                    if prod.get("status") == "active" and int(prod.get("stock_quantity", 0)) > 0:
                                        collected_products[int(prod["product_id"])] = prod

                            # Provide real catalog items to LLM to write authentic summary
                            llm_payload = make_llm_tool_payload("search_instruments", tool_output)
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_forced_catalog_01",
                                            "type": "function",
                                            "function": {
                                                "name": "search_instruments",
                                                "arguments": json.dumps(forced_args),
                                            },
                                        }
                                    ],
                                }
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": "call_forced_catalog_01",
                                    "name": "search_instruments",
                                    "content": json.dumps(llm_payload, ensure_ascii=False),
                                }
                            )
                            response, model_used = self._call_llm_with_fallbacks(
                                messages=messages,
                                tools=None,
                                temperature=settings.llm_temperature,
                            )
                            choice_msg = response.choices[0].message
                            final_text = clean_thinking_tags(choice_msg.content or "")

                    options_to_offer = []
                    if "novo cliente" in final_text.lower():
                        options_to_offer = ["Sim, sou novo cliente", "Não, vou tentar outro dado"]
                    prods = [p for p in collected_products.values() if p.get("status") == "active" and int(p.get("stock_quantity", 0)) > 0] if self.name == "catalog" else []
                    if self.name == "catalog" and prods:
                        final_text = sanitize_catalog_reply(final_text, prods)
                    else:
                        final_text = clean_carousel_mentions(final_text)
                    return {
                        "reply": final_text,
                        "language": detected_lang,
                        "confidence": 0.95,
                        "model_used": model_used,
                        "tools_called": tools_called_log,
                        "products": prods,
                        "active_agent": self.name,
                        "action": None,
                        "options": options_to_offer,
                        "logged_failed_intent": False,
                    }

                messages.append(
                    {
                        "role": "assistant",
                        "content": choice_msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    fn_name = tc.function.name
                    raw_args = tc.function.arguments
                    try:
                        fn_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        fn_args = {}

                    # Security cross-verification injection for order status tool
                    if fn_name == "check_order_status":
                        if claimed_customer_name and not fn_args.get("customer_name"):
                            fn_args["customer_name"] = claimed_customer_name
                        digits_m = re.sub(r"\D", "", user_message)
                        if len(digits_m) >= 8 and not fn_args.get("customer_identifier"):
                            fn_args["customer_identifier"] = digits_m

                    tool_output = self.tool_executor.execute(fn_name, fn_args)
                    tools_called_log.append({"tool": fn_name, "arguments": fn_args})

                    # Collect structured product cards ONLY if this is the Catalog agent
                    if self.name == "catalog" and isinstance(tool_output, dict):
                        if "products" in tool_output and isinstance(tool_output["products"], list):
                            for prod in tool_output["products"]:
                                if isinstance(prod, dict) and "product_id" in prod:
                                    if prod.get("status") == "active" and int(prod.get("stock_quantity", 0)) > 0:
                                        collected_products[int(prod["product_id"])] = prod
                        if "product" in tool_output and isinstance(tool_output["product"], dict):
                            prod = tool_output["product"]
                            if "product_id" in prod:
                                if prod.get("status") == "active" and int(prod.get("stock_quantity", 0)) > 0:
                                    collected_products[int(prod["product_id"])] = prod
                        if "recommended_alternatives_in_stock" in tool_output and isinstance(
                            tool_output["recommended_alternatives_in_stock"], list
                        ):
                            for alt in tool_output["recommended_alternatives_in_stock"]:
                                if isinstance(alt, dict) and "product_id" in alt:
                                    full_alt = self.tool_executor.product_repo.get_by_id(int(alt["product_id"]))
                                    if full_alt and full_alt.get("status") == "active" and int(full_alt.get("stock_quantity", 0)) > 0:
                                        collected_products[int(full_alt["product_id"])] = full_alt

                    llm_payload = make_llm_tool_payload(fn_name, tool_output)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": fn_name,
                            "content": json.dumps(llm_payload, ensure_ascii=False),
                        }
                    )

            # Final synthesis call without tools
            response, model_used = self._call_llm_with_fallbacks(
                messages=messages,
                tools=None,
                temperature=settings.llm_temperature,
            )
            final_text = clean_thinking_tags(response.choices[0].message.content or "")
            options_to_offer = []
            if "novo cliente" in final_text.lower():
                options_to_offer = ["Sim, sou novo cliente", "Não, vou tentar outro dado"]
            prods = [p for p in collected_products.values() if p.get("status") == "active" and int(p.get("stock_quantity", 0)) > 0] if self.name == "catalog" else []
            if self.name == "catalog" and prods:
                final_text = sanitize_catalog_reply(final_text, prods)
            else:
                final_text = clean_carousel_mentions(final_text)
            return {
                "reply": final_text,
                "language": detected_lang,
                "confidence": 0.95,
                "model_used": model_used,
                "tools_called": tools_called_log,
                "products": prods,
                "active_agent": self.name,
                "action": None,
                "options": options_to_offer,
                "logged_failed_intent": False,
            }
        except Exception as exc:
            if self.name == "catalog" and not collected_products:
                norm_u = normalize_text(user_message)
                cat_found = None
                for c_name, aliases in [
                    ("Ukuleles", ["ukulele", "ukuleles"]),
                    ("Violões", ["violao", "violoes", "violão", "violões", "acoustic guitar"]),
                    ("Guitarras", ["guitarra", "guitarras", "stratocaster", "telecaster", "les paul", "electric guitar"]),
                    ("Baixos", ["baixo", "baixos", "contrabaixo", "bass"]),
                    ("Baterias e Percussão", ["bateria", "baterias", "percussao", "drums"]),
                    ("Teclados e Pianos", ["teclado", "teclados", "piano", "pianos", "sintetizador", "keyboard"]),
                    ("Instrumentos de Sopro (Madeiras)", ["sopro", "sax", "flauta", "clarinete"]),
                    ("Instrumentos de Sopro (Metais)", ["trompete", "trombone", "trompa"]),
                    ("Cordas Orquestrais", ["violino", "viola", "violoncelo", "orquestral"]),
                ]:
                    if any(a in norm_u for a in aliases):
                        cat_found = c_name
                        break
                p_match = re.search(r"(?:ate|maximo|menos de|custando ate)\s*(?:r\$)?\s*(\d+(?:\.\d+)?)", norm_u)
                max_p = float(p_match.group(1)) if p_match else None
                only_promo = "promocao" in norm_u or "promocoes" in norm_u
                brand_found = None
                for b in ["yamaha", "takamine", "tagima", "giannini", "fender", "ibanez", "shelby", "kalani", "kala", "ohana", "roland", "casio", "cort", "rozini"]:
                    if b in norm_u:
                        brand_found = b
                        break

                catalog_keywords = [
                    "instrumento", "instrumentos", "produto", "produtos", "opcao", "opcoes",
                    "opção", "opções", "catalogo", "catálogo", "modelo", "modelos", "preco",
                    "preço", "valor", "valores", "reais", "recomenda", "recomendacao",
                    "barato", "baratos", "disponivel", "disponiveis", "estoque", "ver", "quais", "tem"
                ]

                if cat_found or brand_found or max_p is not None or only_promo or any(w in norm_u for w in catalog_keywords):
                    forced_args = {
                        "category_query": cat_found,
                        "query": brand_found,
                        "max_price": max_p,
                        "only_available": True,
                        "only_on_promotion": only_promo,
                    }
                    try:
                        tool_output = self.tool_executor.execute("search_instruments", forced_args)
                        if isinstance(tool_output, dict) and "products" in tool_output:
                            for prod in tool_output["products"]:
                                if prod.get("status") == "active" and int(prod.get("stock_quantity", 0)) > 0:
                                    collected_products[int(prod["product_id"])] = prod
                    except Exception:
                        pass

            prods = [p for p in collected_products.values() if p.get("status") == "active" and int(p.get("stock_quantity", 0)) > 0] if self.name == "catalog" else []
            if prods:
                fallback_reply = (
                    "Aqui estão as opções encontradas no nosso catálogo! "
                    "Confira abaixo a lista de opções disponíveis para ver os detalhes, valores e adicionar direto ao carrinho. 🎸"
                )
            else:
                fallback_reply = (
                    "Tivemos uma pequena oscilação temporária de conexão com o provedor de IA. "
                    "Por favor, tente novamente em alguns segundos ou contate nosso WhatsApp (67) 3321-4500! 🎵"
                )
            return {
                "reply": fallback_reply,
                "language": detected_lang,
                "confidence": 0.6,
                "model_used": "resilience_fallback",
                "tools_called": tools_called_log,
                "products": prods,
                "active_agent": self.name,
                "action": None,
                "options": [],
                "logged_failed_intent": True,
            }


class CatalogAgent(BaseSpecialistAgent):
    """Specialist Agent responsible exclusively for Musical Instruments Catalog & Recommendations."""

    name = "catalog"
    allowed_tool_names = ["search_instruments", "get_instrument_details_and_pricing"]

    def get_system_prompt(self, user_language: str) -> str:
        return f"""You are the official Virtual Assistant of **Empório da Música** ("Sua música começa aqui"), founded in 2008 in Campo Grande/MS, specialized in Musical Instruments Catalog & Recommendations.

### YOUR EXCLUSIVE RESPONSIBILITIES:
1. Help customers discover, compare, and understand musical instruments (Violões, Guitarras, Baixos, Baterias e Percussão, Teclados e Pianos, Sopro, Cordas Orquestrais, Ukuleles).
2. **MANDATORY TOOL USAGE (ZERO HALLUCINATION & IMMEDIATE DISPLAY):**
   Whenever the customer asks for instruments, options, models, recommendations, or a price range (e.g. "opções de violões até R$1000", "ver ukuleles", "quais opções de ukuleles disponíveis?", "guitarras"), you **MUST IMMEDIATELY call `search_instruments`** (e.g., `category_query='Ukuleles'`, `only_available=True`) or `get_instrument_details_and_pricing`.
   - **NEVER** respond describing products without calling the tool!
   - **NEVER INVENT OR HALLUCINATE INSTRUMENTS (e.g., never invent brands or models like Cordoba not returned by the tool)!**
   - **NEVER DISPLAY DEACTIVATED (status != 'active') OR OUT-OF-STOCK (stock_quantity == 0) PRODUCTS WHEN CUSTOMER ASKS FOR OPTIONS/AVAILABLE INSTRUMENTS. Always set `only_available=True`.**
3. Highlight key specs (nylon vs steel strings, acoustic vs electroacoustic, wood types, brand, price, PIX discount 5% off, and stock availability).
4. **Out-of-Stock / Discontinued / Coming Soon Handling:**
   - When customer specifically asks for an individual model by name that is out of stock (e.g. Takamine GD20): explain it is temporarily out of stock and present the available alternatives returned by the tool.
   - For general category browsing, NEVER show or recommend discontinued, coming soon, or zero-stock instruments.
5. **Exclusive Scope:** The store sells EXCLUSIVELY musical instruments. We DO NOT sell accessories (loose strings/encordoamentos, picks/palhetas, cables, cases, pedals, amplifiers). Politely explain this if asked.
6. **UI Product Display Integration (CRITICAL ANTI-REDUNDANCY & TERMINOLOGY):**
   - The web interface automatically renders every single product returned by catalog tools as an interactive 3D visual card below your message, showing the instrument's photo, title, price, PIX discount, stock count, quantity selector, specs button, and "Adicionar ao Carrinho".
   - **STRICTLY PROHIBITED IN TEXT:** DO NOT generate Markdown tables (`| ... |`) listing the products. DO NOT generate repetitive item-by-item bullet lists repeating product names, prices, or specs in the text. Products MUST be shown EXCLUSIVELY in the visual cards below to avoid duplicate/polluted information.
   - **MANDATORY TERMINOLOGY & TEXT FORMAT (CRITICAL):**
     * NEVER mention the word "carrossel" (or "carousel") to the customer!
     * ALWAYS invite the customer using: **"confira abaixo a lista de opções disponíveis..."** (e.g.: *"Confira abaixo a lista de opções disponíveis para ver detalhes, fotos e adicionar diretamente ao carrinho!"*).
     * Keep your text response concise, warm, and musician-friendly.

### UNIFIED BRAND PERSONA (MANDATORY):
- ALWAYS represent Empório da Música as a friendly, expert musician assistant.
- NEVER use robotic department labels or send customers away to external links.
- Respond strictly in {user_language}. Be warm, knowledgeable, and helpful like a fellow musician.
"""


class CheckoutAgent(BaseSpecialistAgent):
    """Specialist Agent responsible exclusively for Cart, Checkout, Payments, and Shipping Policies."""

    name = "checkout"
    allowed_tool_names = ["simulate_payment_and_shipping", "consult_store_policies"]

    def get_system_prompt(self, user_language: str) -> str:
        return f"""You are the official Virtual Assistant of **Empório da Música** ("Sua música começa aqui"), specialized in Cart, Checkout, Payments, and Order Finalization.

### YOUR EXCLUSIVE RESPONSIBILITIES:
1. Guide the customer smoothly through finalizing their purchase, closing their order, and calculating payments.
2. When the customer mentions products in the cart or items they want to buy, you **MUST ALWAYS** calculate or confirm the payment options and shipping using `simulate_payment_and_shipping`.
3. **NEVER ask the customer to calculate or send the total price in R$ of products that are in our store.** You are the store assistant and you already know the product prices.
4. Calculate payment options and shipping costs/deadlines using `simulate_payment_and_shipping`:
   - **PIX:** 5% cash discount (STRICT RULE: NOT cumulative with products that already have an active promotion).
   - **Cartão de Crédito:** Up to 12x interest-free with strict installment minimums:
     * 1x to 3x: minimum installment R$ 50.00 (1x has no minimum).
     * 4x to 6x: minimum installment R$ 80.00.
     * 7x to 12x: minimum installment R$ 100.00.
   - **Boleto Bancário:** Cash payment, bank clearing in up to 3 business days.
   - **Cartão de Débito:** Cash payment, all card flags accepted.
   - **Combined Payment (PIX + Card):** Allowed exclusively for orders above R$ 2,000.00.
   - **SECURITY / SIMULATION RULE (MANDATORY):** For payment simulation, NEVER request confidential credit or debit card numbers, CVVs, expiry dates, or passwords. Simply confirm the selected installment or payment method and simulate the approval.
   - **Shipping Rules:**
     * Campo Grande/MS and metro area: Free shipping on orders above R$ 300.00. Delivery in 1 to 3 business days via store motoboy (with prior telephone confirmation).
     * Other cities: Handled via partner carrier; quote depends on destination and whether it is a large instrument.
5. **Customer Information for Order Closing:**
   - Present the breakdown with subtotal, PIX price (5% off), installment table, and shipping.
   - **CRITICAL REGISTERED CUSTOMER RULE:** If the customer is ALREADY REGISTERED or identified in context, you **MUST NEVER ask for their CEP, address, phone or email again**. Their delivery address, city, and CEP are already registered. Always use their registered data and ask if they wish to keep or update their delivery address.
   - **CRITICAL CART TRUTH RULE:** NEVER hallucinate, invent, or assume any product model that the customer did not explicitly select in the cart or chat history.
   - Only ask for identification (Telefone/WhatsApp or E-mail) if the customer is completely anonymous/unidentified.

### UNIFIED BRAND PERSONA (MANDATORY):
- ALWAYS represent Empório da Música as a friendly, expert store assistant.
- NEVER use robotic department labels like "Eu sou o agente de checkout".
- Respond strictly in {user_language}. Be clear, transparent, and encouraging.
"""

    def _parse_cart_items_and_total(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[List[tuple[Dict[str, Any], int]], float, bool]:
        """Parses products, quantities, and order total from message or recent history."""
        all_prods = self.tool_executor.product_repo.search_products(limit=250)
        norm_msg = normalize_text(user_message).lower()

        found_items: List[tuple[Dict[str, Any], int]] = []
        for p in all_prods:
            p_norm = normalize_text(p["name"]).lower()
            if p_norm in norm_msg:
                qty_m = re.search(r"(\d+)\s*x\s*" + re.escape(p["name"]), user_message, re.IGNORECASE)
                if not qty_m:
                    qty_m = re.search(r"(\d+)\s*x\s*" + re.escape(p_norm), norm_msg, re.IGNORECASE)
                qty = int(qty_m.group(1)) if qty_m else 1
                found_items.append((p, qty))

        if not found_items and chat_history:
            for msg in reversed(chat_history[-10:]):
                c = msg.get("content", "")
                norm_c = normalize_text(c).lower()
                for p in all_prods:
                    p_norm = normalize_text(p["name"]).lower()
                    if p_norm in norm_c:
                        qty_m = re.search(r"(\d+)\s*x\s*" + re.escape(p["name"]), c, re.IGNORECASE)
                        if not qty_m:
                            qty_m = re.search(r"(\d+)\s*x\s*" + re.escape(p_norm), norm_c, re.IGNORECASE)
                        qty = int(qty_m.group(1)) if qty_m else 1
                        if not any(item[0]["product_id"] == p["product_id"] for item in found_items):
                            found_items.append((p, qty))

        total_m = re.search(
            r"(?:total(?: do pedido| do carrinho)?|valor total)\s*:?\s*r?\$?\s*(\d+(?:[,\.]\d{2})?)",
            user_message,
            re.IGNORECASE,
        )
        if total_m:
            try:
                total_val = float(total_m.group(1).replace(",", "."))
            except Exception:
                total_val = 0.0
        elif found_items:
            total_val = sum(float(p["effective_price_brl"]) * qty for p, qty in found_items)
        else:
            total_val = 0.0

        has_promo = any(bool(p.get("active_promotion")) for p, _ in found_items)
        return found_items, total_val, has_promo

    def _format_checkout_breakdown(
        self,
        found_items: List[tuple[Dict[str, Any], int]],
        total: float,
        sim_res: Dict[str, Any],
    ) -> str:
        pay = sim_res.get("payment", {})
        ship = sim_res.get("shipping", {})

        def _fmt_money(val: float) -> str:
            return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        lines = [
            "Perfeito! Vamos fechar seu pedido com o cálculo detalhado de pagamento e entrega: 🎶✨\n"
        ]

        if found_items:
            lines.append("| Item | Quantidade | Valor Unitário | Subtotal |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for p, qty in found_items:
                u_p = float(p["effective_price_brl"])
                sub = u_p * qty
                lines.append(f"| {p['name']} | {qty} | {_fmt_money(u_p)} | {_fmt_money(sub)} |")
            total_items = sum(q for _, q in found_items)
            lines.append(f"| **Total do Pedido** | **{total_items} item(ns)** | | **{_fmt_money(total)}** |\n")
        else:
            lines.append(f"**Total do Pedido:** **{_fmt_money(total)}**\n")

        lines.append("### 💳 Opções de Pagamento:")
        pix_price = float(pay.get("pix_price_brl", total))
        if pay.get("pix_5_percent_applied"):
            eco = total - pix_price
            lines.append(f"• **À vista no PIX (5% de desconto):** **{_fmt_money(pix_price)}** (economia de {_fmt_money(eco)})")
        else:
            lines.append(f"• **À vista no PIX:** **{_fmt_money(pix_price)}** (desconto PIX não cumulativo com produtos em promoção)")

        max_inst = pay.get("max_installments", 1)
        valid_opts = pay.get("valid_installment_options", [])
        if valid_opts:
            lines.append(f"• **Cartão de Crédito sem juros:** em até **{max_inst}x**:")
            for inst in valid_opts:
                v = float(inst["installment_value_brl"])
                lines.append(f"  - {inst['installments']}x de {_fmt_money(v)}")
        else:
            lines.append(f"• **Cartão de Crédito:** 1x de {_fmt_money(total)}")

        if total >= 2000.0:
            lines.append("• **Pagamento Combinado (PIX + Cartão):** Elegível para este pedido (valor total acima de R$ 2.000,00)!")

        lines.append("\n### 🚚 Frete e Prazo de Entrega:")
        dest = ship.get("destination", "Campo Grande")
        if ship.get("free_shipping"):
            lines.append(f"• **Frete Grátis** para {dest} (pedidos a partir de R$ 300,00).")
        else:
            cost = float(ship.get("shipping_cost_brl", 0.0))
            lines.append(f"• **Frete:** {_fmt_money(cost)} para {dest}.")
        lines.append(f"• **Prazo de entrega:** {ship.get('delivery_estimate', '1 a 3 dias úteis')} via {ship.get('carrier', 'motoboy da loja')}.")

        lines.append(
            "\n---\n"
            "Para finalizarmos seu pedido e emitirmos o comprovante, por favor informe seu **e-mail** ou **telefone/WhatsApp** cadastrado "
            "(se ainda não tiver cadastro, faremos rapidinho aqui mesmo!). 🎸"
        )
        return "\n".join(lines)

    def execute(
        self,
        user_message: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        claimed_customer_name: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Custom execute for CheckoutAgent. Automatically resolves products mentioned,
        simulates payment and shipping, and ensures user is NEVER asked to calculate totals.
        """
        found_items, total, has_promo = self._parse_cart_items_and_total(
            user_message=user_message,
            chat_history=chat_history,
        )

        # Fallback to rich session context if message didn't specify items
        if not found_items and session_context and session_context.get("cart_items"):
            found_items = session_context["cart_items"]
            total = float(session_context.get("cart_total") or 0.0)
            has_promo = bool(session_context.get("has_promo", False))

        tools_called_log: List[Dict[str, Any]] = []
        if total > 0.0:
            dest_city = "Campo Grande"
            if "dourados" in user_message.lower():
                dest_city = "Dourados"
            elif "corumba" in user_message.lower():
                dest_city = "Corumbá"
            elif "tres lagoas" in user_message.lower():
                dest_city = "Três Lagoas"
            elif session_context and session_context.get("customer") and session_context["customer"].get("city"):
                dest_city = session_context["customer"]["city"]

            sim_args = {
                "amount_brl": total,
                "has_active_promotion": has_promo,
                "city": dest_city,
            }
            try:
                sim_res = self.tool_executor.execute("simulate_payment_and_shipping", sim_args)
                tools_called_log.append({"tool": "simulate_payment_and_shipping", "arguments": sim_args})
            except Exception:
                sim_res = {}

            if isinstance(sim_res, dict) and "payment" in sim_res:
                deterministic_text = self._format_checkout_breakdown(
                    found_items=found_items,
                    total=total,
                    sim_res=sim_res,
                )
                detected_lang = detect_user_language(user_message)
                return {
                    "reply": deterministic_text,
                    "language": detected_lang,
                    "confidence": 0.98,
                    "model_used": "checkout_pricing_engine",
                    "tools_called": tools_called_log,
                    "products": [],
                    "active_agent": self.name,
                    "action": None,
                    "options": [],
                    "logged_failed_intent": False,
                }

        norm_u = normalize_text(user_message).lower()
        if any(k in norm_u for k in ["carrinho", "meu carrinho", "itens no carrinho", "o que tem no carrinho", "ver carrinho"]):
            detected_lang = detect_user_language(user_message)
            return {
                "reply": (
                    "🛒 **Seu carrinho está vazio no momento.**\n\n"
                    "Nenhum produto foi selecionado ainda. Você pode navegar pelo nosso catálogo para escolher seu instrumento favorito! 🎸✨"
                ),
                "language": detected_lang,
                "confidence": 0.95,
                "model_used": "checkout_cart_engine",
                "tools_called": [],
                "products": [],
                "active_agent": self.name,
                "action": None,
                "options": ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção"],
                "logged_failed_intent": False,
            }

        return super().execute(
            user_message=user_message,
            session_id=session_id,
            chat_history=chat_history,
            claimed_customer_name=claimed_customer_name,
            session_context=session_context,
        )


class CustomerAgent(BaseSpecialistAgent):
    """Specialist Agent responsible exclusively for Customer Identification, Onboarding, and Address Registration."""

    name = "customer"
    allowed_tool_names = ["lookup_cep", "register_customer", "update_customer_address"]

    def get_system_prompt(self, user_language: str) -> str:
        return f"""You are the official Virtual Assistant of **Empório da Música** ("Sua música começa aqui"), specialized in Customer Onboarding, CEP lookup, and Delivery Address registration.

### YOUR EXCLUSIVE RESPONSIBILITIES:
1. Welcome and register new customers who do not yet have an account in the store.
2. When the customer provides a Brazilian CEP (8 digits), you **MUST** call `lookup_cep` to resolve street, neighborhood, city, and state via ViaCEP.
3. Confirm the house/apt number and complement, then call `register_customer` to save their registration in the database.
4. If an existing customer wants to update their delivery address, ask for their authenticated phone/email and call `update_customer_address`.
5. **STRICT PRIVACY & LGPD RULES (ZERO DATA LEAK):**
   - Never reveal or cite any other customer's name, email, or details under ANY circumstance.
   - If verification fails, inform politely that the provided phone or email does not correspond to the full name provided, and ask them to verify.
   - Never use real customer IDs or tracking codes as examples (use only placeholders like `#0000`, `BR000000000BR`, `seuemail@exemplo.com`).

### UNIFIED BRAND PERSONA & SEAMLESS SERVICE (MANDATORY):
- ALWAYS represent Empório da Música as a single, friendly, expert store assistant.
- NEVER introduce yourself with internal department titles like "Eu sou o especialista em cadastro e endereço da Empório da Música".
- NEVER send the customer away to "acesse nosso site oficial" to look up musical instruments. If the user asks about instruments, answer helpfully and guide them to explore our models.
- Respond strictly in {user_language}. Be welcoming, organized, and secure.
"""


class SupportAgent(BaseSpecialistAgent):
    """Specialist Agent responsible exclusively for Post-Sale Support, Order Tracking, and Store Policies."""

    name = "support"
    allowed_tool_names = ["check_order_status", "consult_store_policies"]

    def get_system_prompt(self, user_language: str) -> str:
        return f"""You are the official Virtual Assistant of **Empório da Música** ("Sua música começa aqui"), specialized in Post-Sale Support, Order Tracking, and Store Policies.

### YOUR EXCLUSIVE RESPONSIBILITIES:
1. **Order Tracking & Status:**
   - Look up order and delivery status using `check_order_status`.
   - **MANDATORY PRIMARY KEY AUTHENTICATION:** Before revealing ANY order or tracking info, the customer MUST provide their registered **e-mail or phone number**. Never look up an order based only on a name or raw order ID without primary key validation.
   - Never show data of another person's order (Strict Data Isolation).
2. **Store Policies & Procedures (from Store Manual via `consult_store_policies`):**
   - **Online Right of Regret (Arrependimento):** Up to 7 calendar days after delivery, store pays return shipping, refund in up to 10 business days.
   - **Exchange by Preference:** Up to 7 calendar days, product intact/sealed.
   - **Ineligible for Exchange:** Custom setups, clearance "venda final", and **wind instrument mouthpieces (boquilhas de sopro)** for hygiene reasons.
   - **Defects & Warranty:** Store exchanges defects up to 30 calendar days; after 30 days, intermediate with manufacturer. Legal warranty (CDC) is 90 days. Natural wear, misuse, drops, and climate exposure are NOT covered.
   - **Store Address & Hours:** Rua 14 de Maio, 3200 — Centro, Campo Grande/MS (CEP 79202-333). Mon-Fri 08:30-18:30, Sat 08:30-13:00, Sun closed. WhatsApp: (67) 3321-4500.

### UNIFIED BRAND PERSONA (MANDATORY):
- ALWAYS represent Empório da Música as a single, friendly, and expert store assistant.
- NEVER use robotic department labels like "Eu sou o agente de suporte".
- Respond strictly in {user_language}. Be empathetic, respectful, and reassuring.
"""

    def execute(
        self,
        user_message: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        claimed_customer_name: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute post-sale support or order tracking.
        When customer is already identified in session_context and asks about their order or tracking,
        queries orders deterministically and returns active orders or states clearly if none exist.
        """
        norm_msg = normalize_text(user_message).lower()
        is_tracking_intent = any(k in norm_msg for k in [
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
            "quando chega", "previsao de entrega", "pedidos"
        ]) or norm_msg in ("pedidos", "meus pedidos", "compras", "rastreio", "meu rastreio")

        cust = session_context.get("customer") if session_context else None

        # Check if user message provided an explicit order ID or tracking code
        order_id_m = re.search(r"(?:pedido|ordem|numero|n[ºo°]|#)\s*#?\s*(\d+)", user_message, re.IGNORECASE)
        req_order_id = int(order_id_m.group(1)) if order_id_m else None
        track_m = re.search(r"\b(BR[A-Z0-9]{9}BR)\b", user_message, re.IGNORECASE)
        req_tracking = track_m.group(1).upper() if track_m else None

        if is_tracking_intent or req_order_id or req_tracking:
            detected_lang = detect_user_language(user_message)
            if cust:
                cust_name = cust.get("name") or claimed_customer_name or "Cliente"
                cust_identifier = cust.get("phone") or cust.get("email") or str(cust.get("customer_id"))

                status_res = self.tool_executor.check_order_status(
                    order_id=req_order_id,
                    tracking_code=req_tracking,
                    customer_identifier=cust_identifier,
                    customer_name=cust_name,
                )

                tools_called_log = [{
                    "tool": "check_order_status",
                    "arguments": {
                        "order_id": req_order_id,
                        "tracking_code": req_tracking,
                        "customer_identifier": cust_identifier,
                        "customer_name": cust_name,
                    }
                }]

                if status_res.get("ownership_mismatch"):
                    return {
                        "reply": (
                            f"Olá, **{cust_name}**! Por questões de segurança e privacidade (LGPD), "
                            "o pedido informado não está vinculado ao seu cadastro atual.\n\n"
                            "Caso tenha realizado a compra com outro número ou e-mail, por favor me informe o contato correspondente!"
                        ),
                        "language": detected_lang,
                        "confidence": 1.0,
                        "model_used": "direct_order_tracking",
                        "tools_called": tools_called_log,
                        "products": [],
                        "active_agent": self.name,
                        "action": None,
                        "options": ["📍 Informações da Loja", "🎸 Ver Catálogo"],
                        "logged_failed_intent": False,
                    }

                orders = status_res.get("orders") or []
                if orders:
                    lines = [f"📦 **Status do seu Pedido — Empório da Música** 🎸\n"]
                    for o in orders:
                        oid = o.get("order_id")
                        status = o.get("status", "")
                        status_map = {
                            "delivered": "✅ **Entregue com sucesso**",
                            "shipped": "🚚 **Enviado / Em trânsito**",
                            "confirmed": "📋 **Confirmado (Em separação para envio)**",
                            "processing": "⏳ **Em processamento**",
                            "cancelled": "❌ **Cancelado**",
                        }
                        status_label = status_map.get(status.lower(), f"📌 **{status.capitalize()}**")
                        t_code = o.get("tracking_code")
                        t_display = f"`{t_code}`" if t_code else "Entrega local via motoboy próprio"
                        est_date = o.get("estimated_delivery") or "1 a 3 dias úteis"
                        tot_val = float(o.get("total_brl") or 0.0)
                        items = o.get("items") or []
                        items_desc = ", ".join(f"{it['quantity']}x {it.get('product_name') or 'Item'}" for it in items) if items else "Instrumento(s) musical(is)"

                        def _fmt_money(val: float) -> str:
                            return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                        lines.append(f"• **Pedido #{oid}:**")
                        lines.append(f"  - **Status:** {status_label}")
                        lines.append(f"  - **Itens:** {items_desc}")
                        lines.append(f"  - **Valor Total:** {_fmt_money(tot_val)}")
                        lines.append(f"  - **Código de Rastreamento:** {t_display}")
                        lines.append(f"  - **Previsão de Entrega:** {est_date}")
                        if o.get("notes"):
                            lines.append(f"  - **Observações:** {o['notes']}")
                        lines.append("")

                    lines.append(f"**{cust_name}**, qualquer dúvida sobre a entrega ou código de rastreamento estamos à disposição!")
                    return {
                        "reply": "\n".join(lines).strip(),
                        "language": detected_lang,
                        "confidence": 1.0,
                        "model_used": "direct_order_tracking",
                        "tools_called": tools_called_log,
                        "products": [],
                        "active_agent": self.name,
                        "action": None,
                        "options": ["📍 Informações da Loja", "🎸 Ver Produtos em Promoção"],
                        "logged_failed_intent": False,
                    }
                else:
                    reply = (
                        f"Olá, **{cust_name}**! Consultamos seu cadastro e verificamos que no momento "
                        "**não há nenhum pedido ou código de rastreio vinculado à sua conta**. 📦\n\n"
                        "Se você realizou uma compra recente com outro e-mail ou telefone, ou possui o número específico de outro pedido, "
                        "por favor me informe para eu localizar para você!\n\n"
                        "Como posso te ajudar agora? Você pode explorar nossos instrumentos musicais disponíveis no catálogo:"
                    )
                    return {
                        "reply": reply,
                        "language": detected_lang,
                        "confidence": 1.0,
                        "model_used": "direct_order_tracking",
                        "tools_called": tools_called_log,
                        "products": [],
                        "active_agent": self.name,
                        "action": None,
                        "options": ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção", "📍 Informações da Loja"],
                        "logged_failed_intent": False,
                    }
            else:
                # Customer not authenticated in session
                email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", user_message)
                phone_match = re.search(r"(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?(?:9\s*)?\d{4}[-\s]?\d{4}", user_message)
                digits_only = re.sub(r"\D", "", phone_match.group(0)) if phone_match else re.sub(r"\D", "", user_message)
                has_phone = 8 <= len(digits_only) <= 13
                if email_match or has_phone:
                    pk = email_match.group(0) if email_match else digits_only
                    status_res = self.tool_executor.check_order_status(
                        order_id=req_order_id,
                        tracking_code=req_tracking,
                        customer_identifier=pk,
                        customer_name=claimed_customer_name,
                    )
                    orders = status_res.get("orders") or []
                    if status_res.get("found") and orders:
                        verified_cust = status_res.get("customer") or {}
                        c_name = verified_cust.get("name", "Cliente")
                        lines = [f"📦 **Status do seu Pedido — Empório da Música** 🎸\n"]
                        for o in orders:
                            oid = o.get("order_id")
                            status = o.get("status", "")
                            status_map = {
                                "delivered": "✅ **Entregue com sucesso**",
                                "shipped": "🚚 **Enviado / Em trânsito**",
                                "confirmed": "📋 **Confirmado (Em separação para envio)**",
                                "processing": "⏳ **Em processamento**",
                                "cancelled": "❌ **Cancelado**",
                            }
                            status_label = status_map.get(status.lower(), f"📌 **{status.capitalize()}**")
                            t_code = o.get("tracking_code")
                            t_display = f"`{t_code}`" if t_code else "Entrega local via motoboy próprio"
                            est_date = o.get("estimated_delivery") or "1 a 3 dias úteis"
                            tot_val = float(o.get("total_brl") or 0.0)

                            def _fmt_money(val: float) -> str:
                                return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                            lines.append(f"• **Pedido #{oid}:**")
                            lines.append(f"  - **Status:** {status_label}")
                            lines.append(f"  - **Valor Total:** {_fmt_money(tot_val)}")
                            lines.append(f"  - **Código de Rastreamento:** {t_display}")
                            lines.append(f"  - **Previsão de Entrega:** {est_date}")
                            lines.append("")

                        return {
                            "reply": "\n".join(lines).strip(),
                            "language": detected_lang,
                            "confidence": 1.0,
                            "model_used": "direct_order_tracking",
                            "tools_called": [{"tool": "check_order_status", "arguments": {"customer_identifier": pk}}],
                            "products": [],
                            "active_agent": self.name,
                            "action": None,
                            "options": ["📍 Informações da Loja", "🎸 Ver Produtos em Promoção"],
                            "logged_failed_intent": False,
                        }

                reply = (
                    "Para consultar o status do seu pedido e o código de rastreamento com segurança e privacidade (LGPD), "
                    "por favor informe o seu **número de telefone/WhatsApp ou e-mail cadastrado**:"
                )
                return {
                    "reply": reply,
                    "language": detected_lang,
                    "confidence": 1.0,
                    "model_used": "direct_order_tracking",
                    "tools_called": [],
                    "products": [],
                    "active_agent": self.name,
                    "action": None,
                    "options": ["Fazer meu cadastro", "📍 Informações da Loja"],
                    "logged_failed_intent": False,
                }

        return super().execute(
            user_message=user_message,
            session_id=session_id,
            chat_history=chat_history,
            claimed_customer_name=claimed_customer_name,
            session_context=session_context,
        )

