"""Structured tools for LLM Function Calling covering catalog, pricing, orders, and store policies."""

import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

from src.rag.policy_retriever import PolicyRetriever
from src.repositories.customer_repository import CustomerRepository
from src.repositories.order_repository import OrderRepository
from src.repositories.product_repository import ProductRepository, normalize_text
from src.services.pricing_service import PricingService


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_instruments",
            "description": (
                "Search the Empório da Música catalog for musical instruments by keyword, "
                "category (Violões, Guitarras, Baixos, Baterias e Percussão, Teclados e Pianos, Ukuleles, etc.), "
                "price range (max_price, min_price), strings type ('nylon' or 'steel'), "
                "electronics ('yes' or 'no'), availability, or active promotions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Model name, brand, or keyword (e.g., 'Takamine GD20', 'Yamaha', '7 cordas', 'Stratocaster'). Leave empty if browsing a whole category.",
                    },
                    "category_query": {
                        "type": "string",
                        "description": "Category filter (e.g., 'Violões', 'Guitarras', 'Baixos', 'Baterias', 'Teclados', 'Ukuleles').",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price in BRL (e.g., 1000 for 'até R$1000').",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price in BRL.",
                    },
                    "only_available": {
                        "type": "boolean",
                        "description": "Set to true if the customer specifically asks for available/in-stock options.",
                    },
                    "only_on_promotion": {
                        "type": "boolean",
                        "description": "Set to true if the customer asks for products currently on promotion/sale.",
                    },
                    "strings_type": {
                        "type": "string",
                        "description": "Optional filter for strings: 'nylon' or 'steel' (aço).",
                    },
                    "electronics": {
                        "type": "string",
                        "description": "Optional filter for electroacoustic/pickup: 'yes' (elétrico/eletroacústico) or 'no' (acústico).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_instrument_details_and_pricing",
            "description": (
                "Get complete details, stock status, active/expired promotions, PIX price, installment breakdown, "
                "and available alternatives (if out of stock or discontinued) for a specific instrument by ID or name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Exact product_id if known.",
                    },
                    "product_name": {
                        "type": "string",
                        "description": "Product name or model to look up (e.g., 'Takamine GD20', 'Giannini GF-3D', 'Shelby SN-7C').",
                    },
                    "city": {
                        "type": "string",
                        "description": "Customer city for shipping simulation (defaults to 'Campo Grande').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": (
                "Look up customer orders, delivery status, tracking codes, order items, and cancellation reasons. "
                "STRICT SECURITY & LGPD RULE: Before ANY registration, order, or tracking lookup can be performed, "
                "the customer MUST provide and confirm their registered E-MAIL or PHONE NUMBER (`customer_identifier`) as primary key. "
                "If the customer informed their name (e.g. 'Camila Duarte Nascimento'), pass it in `customer_name`. "
                "The tool will verify that the phone/email strictly belongs to that customer name. "
                "If they do not match, the tool will return a verification failure WITHOUT revealing any customer data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Customer name if informed in this turn or earlier in conversation (e.g., 'Camila Duarte Nascimento'). ALWAYS provide this when a name was mentioned so the tool can cross-verify that the phone/email matches that name.",
                    },
                    "customer_identifier": {
                        "type": "string",
                        "description": "MANDATORY primary key: customer's registered email address or phone number. If the user has not provided their email or phone yet, leave empty or pass what they gave and the tool will request email/phone confirmation.",
                    },
                    "order_id": {
                        "type": "integer",
                        "description": "Optional numeric order ID informed by the customer (will be validated against the owner of customer_identifier).",
                    },
                    "tracking_code": {
                        "type": "string",
                        "description": "Optional shipping tracking code informed by the customer (format: BR + 9 alphanumeric chars + BR, e.g., 'BR000000000BR').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_payment_and_shipping",
            "description": (
                "Deterministically calculate PIX 5% cash discount eligibility, credit card interest-free installments "
                "(1x to 12x respecting minimum installment rules), combined payment eligibility, and shipping cost/deadlines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount_brl": {
                        "type": "number",
                        "description": "Total product or order price in BRL.",
                    },
                    "has_active_promotion": {
                        "type": "boolean",
                        "description": "True if the price already comes from an active promotion.",
                    },
                    "discount_percent": {
                        "type": "number",
                        "description": "Active promotional discount percentage if applicable.",
                    },
                    "city": {
                        "type": "string",
                        "description": "Destination city (e.g., 'Campo Grande', 'Dourados', 'Três Lagoas', 'Corumbá', 'São Paulo').",
                    },
                    "is_large_instrument": {
                        "type": "boolean",
                        "description": "True for acoustic drums, digital pianos, or double basses.",
                    },
                },
                "required": ["amount_brl"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_store_policies",
            "description": (
                "Search the official Empório da Música Store Policies & Procedures Manual (PDF) for rules on: "
                "opening hours, store address/CNPJ/contacts, payment & installment rules, returns/exchanges/regret (7 days), "
                "ineligible exchange items (mouthpieces, custom setup), shipping/tracking/insurance, promotions, "
                "WhatsApp SLA, warranty (90-day legal & manufacturer), and LGPD data privacy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question or topic to search in the store manual.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of policy sections to retrieve (default 3).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def is_name_match(provided_name: str, db_name: str) -> bool:
    """Verify that a user-provided name strictly matches the customer name in the database."""
    if not provided_name or not db_name:
        return False
    norm_provided = normalize_text(provided_name)
    norm_db = normalize_text(db_name)

    if norm_provided == norm_db:
        return True

    prepositions = {"de", "da", "do", "das", "dos", "e"}
    prov_tokens = [t for t in norm_provided.split() if len(t) > 1 and t not in prepositions]
    db_tokens = [t for t in norm_db.split() if len(t) > 1 and t not in prepositions]

    if not prov_tokens or not db_tokens:
        return False

    # First name must match
    if prov_tokens[0] != db_tokens[0]:
        return False

    # If full name was provided (>= 2 names), at least 2 tokens must overlap
    if len(prov_tokens) >= 2:
        overlap = set(prov_tokens).intersection(set(db_tokens))
        return len(overlap) >= 2

    return True


class ToolExecutor:
    """Executes structured tool calls against repositories, pricing engine, and RAG retriever."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        pdf_path: Optional[Path] = None,
    ) -> None:
        self.product_repo = ProductRepository(db_path)
        self.customer_repo = CustomerRepository(db_path)
        self.order_repo = OrderRepository(db_path)
        self.policy_retriever = PolicyRetriever(pdf_path)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool call by name and return JSON-serializable dictionary."""
        dispatch: Dict[str, Callable[..., Dict[str, Any]]] = {
            "search_instruments": self.search_instruments,
            "get_instrument_details_and_pricing": self.get_instrument_details_and_pricing,
            "check_order_status": self.check_order_status,
            "simulate_payment_and_shipping": self.simulate_payment_and_shipping,
            "consult_store_policies": self.consult_store_policies,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            clean_args = {k: v for k, v in (arguments or {}).items() if v is not None}
            return handler(**clean_args)
        except Exception as exc:
            return {"error": f"Tool execution error in {tool_name}: {str(exc)}"}

    def search_instruments(
        self,
        query: Optional[str] = None,
        category_query: Optional[str] = None,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        only_available: bool = False,
        only_on_promotion: bool = False,
        strings_type: Optional[str] = None,
        electronics: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute catalog search and attach alternatives if specific items are unavailable."""
        products = self.product_repo.search_products(
            query=query,
            category_query=category_query,
            max_price=max_price,
            min_price=min_price,
            only_available=only_available,
            only_on_promotion=only_on_promotion,
            strings_type=strings_type,
            electronics=electronics,
            limit=15,
        )

        # If user searched for a specific item that is out of stock, discontinued, or coming_soon,
        # proactively attach available alternatives from the same category.
        unavailable_alternatives: Dict[int, List[Dict[str, Any]]] = {}
        for p in products[:3]:
            if not p["in_stock"]:
                alts = self.product_repo.get_alternatives_for_unavailable(p["product_id"], limit=3)
                unavailable_alternatives[p["product_id"]] = [
                    {
                        "product_id": a["product_id"],
                        "name": a["name"],
                        "effective_price_brl": a["effective_price_brl"],
                        "pix_price_brl": a["pix_price_brl"],
                        "stock_quantity": a["stock_quantity"],
                    }
                    for a in alts
                ]

        return {
            "total_found": len(products),
            "filters_applied": {
                "query": query,
                "category_query": category_query,
                "max_price": max_price,
                "min_price": min_price,
                "only_available": only_available,
                "only_on_promotion": only_on_promotion,
                "strings_type": strings_type,
                "electronics": electronics,
            },
            "products": products,
            "alternatives_for_unavailable": unavailable_alternatives,
            "all_store_categories": [
                c["name"] for c in self.product_repo.get_all_categories()
            ]
            if len(products) == 0
            else [],
        }

    def get_instrument_details_and_pricing(
        self,
        product_id: Optional[int] = None,
        product_name: Optional[str] = None,
        city: str = "Campo Grande",
    ) -> Dict[str, Any]:
        """Get full details, promotion history, payment breakdown, and alternatives for a product."""
        product = None
        if product_id is not None:
            product = self.product_repo.get_by_id(int(product_id))
        if not product and product_name:
            matches = self.product_repo.search_products(query=product_name, limit=5)
            if matches:
                product = matches[0]

        if not product:
            return {
                "found": False,
                "message": "Instrument not found in Empório da Música catalog.",
                "available_categories": self.product_repo.get_all_categories(),
            }

        active_promo = product.get("active_promotion")
        has_active = active_promo is not None
        discount_pct = float(active_promo["discount_percent"]) if has_active else 0.0

        payment_details = PricingService.calculate_payment_options(
            base_price_brl=float(product["price_brl"]),
            has_active_promotion=has_active,
            discount_percent=discount_pct,
        )

        is_large = product["category_id"] in (2, 3, 4)
        shipping_details = PricingService.calculate_shipping_policy(
            order_total_brl=float(product["effective_price_brl"]),
            city=city,
            is_large_instrument=is_large,
        )

        alternatives = []
        if not product["in_stock"]:
            raw_alts = self.product_repo.get_alternatives_for_unavailable(
                product["product_id"], limit=3
            )
            alternatives = [
                {
                    "product_id": a["product_id"],
                    "name": a["name"],
                    "price_brl": a["price_brl"],
                    "effective_price_brl": a["effective_price_brl"],
                    "pix_price_brl": a["pix_price_brl"],
                    "stock_quantity": a["stock_quantity"],
                    "status": a["status"],
                }
                for a in raw_alts
            ]

        return {
            "found": True,
            "product": product,
            "payment_simulation": payment_details,
            "shipping_simulation": shipping_details,
            "recommended_alternatives_in_stock": alternatives,
        }

    def check_order_status(
        self,
        order_id: Optional[int] = None,
        tracking_code: Optional[str] = None,
        customer_identifier: Optional[str] = None,
        customer_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Look up order status strictly enforcing primary key verification (phone or email).
        Cross-verifies customer_name if provided and returns ONLY data belonging to that customer.
        """
        clean_id = (customer_identifier or "").strip()

        # Verify whether a valid primary key (email or phone with >= 8 digits) was provided
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", clean_id)
        digits_only = re.sub(r"\D", "", clean_id)
        has_email = email_match is not None
        has_phone = len(digits_only) >= 8

        if not has_email and not has_phone:
            return {
                "found": False,
                "requires_primary_key_confirmation": True,
                "ambiguous_customers": True,
                "provided_identifier": clean_id or None,
                "requested_order_id": order_id,
                "requested_tracking_code": tracking_code,
                "message": (
                    "SECURITY & LGPD BLOCK: Before looking up ANY registration, order number, or tracking code, "
                    "the customer MUST first confirm their registered E-MAIL or PHONE NUMBER as primary key. "
                    "Do NOT show any order, tracking, or customer data. Politely ask the user to provide their "
                    "registered email address or phone number to authenticate their identity."
                ),
            }

        # Resolve the authenticated customer owner from the primary key (email or phone)
        pk_query = email_match.group(0) if email_match else digits_only
        matched_customers = self.customer_repo.find_customers(pk_query)

        if not matched_customers:
            return {
                "found": False,
                "verification_failed": True,
                "message": (
                    "FALHA DE CONFIRMAÇÃO: O telefone ou e-mail informado não corresponde ao nome informado. "
                    "Por motivos de segurança e privacidade (LGPD), NÃO confirme se o número existe ou de quem é. "
                    "Apenas informe educadamente que o telefone ou e-mail não corresponde ao nome completo informado, "
                    "e solicite que o cliente verifique e informe o telefone ou e-mail correto cadastrado."
                ),
            }

        verified_customer = matched_customers[0]

        # Cross-verify customer_name if provided
        if customer_name and customer_name.strip():
            if not is_name_match(customer_name, verified_customer.get("name", "")):
                return {
                    "found": False,
                    "verification_failed": True,
                    "message": (
                        f"FALHA DE CONFIRMAÇÃO: O telefone ou e-mail informado não corresponde ao nome completo informado ({customer_name.strip()}). "
                        "Por motivos de segurança e privacidade (LGPD), NUNCA confirme se o número/e-mail existe no sistema e NUNCA revele a quem ele pertence. "
                        "Apenas informe educadamente que a confirmação falhou e que o número não corresponde ao nome completo informado, "
                        "solicitando que confira e informe o telefone ou e-mail correto cadastrado."
                    ),
                }

        verified_cust_id = int(verified_customer["customer_id"])
        customer_orders = self.order_repo.get_orders_by_customer(verified_cust_id)

        # If user also specified an order_id, verify ownership strictly
        if order_id is not None:
            owned_order = next(
                (o for o in customer_orders if int(o["order_id"]) == int(order_id)),
                None,
            )
            if not owned_order:
                return {
                    "found": False,
                    "ownership_mismatch": True,
                    "customer": verified_customer,
                    "orders": customer_orders,
                    "message": (
                        f"O pedido #{order_id} não está vinculado ao seu cadastro. "
                        f"Exibindo apenas os pedidos vinculados ao seu cadastro."
                    ),
                }
            return {
                "found": True,
                "customer": verified_customer,
                "orders": [owned_order],
                "message": "Order verified and belongs to the authenticated primary key owner.",
            }

        # If user also specified a tracking_code, verify ownership strictly
        if tracking_code and tracking_code.strip():
            clean_track = tracking_code.strip().upper()
            owned_track_order = next(
                (
                    o
                    for o in customer_orders
                    if (o.get("tracking_code") or "").strip().upper() == clean_track
                ),
                None,
            )
            if not owned_track_order:
                return {
                    "found": False,
                    "ownership_mismatch": True,
                    "customer": verified_customer,
                    "orders": customer_orders,
                    "message": (
                        f"O código de rastreamento informado não está vinculado ao seu cadastro. "
                        f"Exibindo apenas os pedidos vinculados ao seu cadastro."
                    ),
                }
            return {
                "found": True,
                "customer": verified_customer,
                "orders": [owned_track_order],
                "message": "Tracking code verified and belongs to the authenticated primary key owner.",
            }

        return {
            "found": len(customer_orders) > 0,
            "customer": verified_customer,
            "orders": customer_orders,
            "message": (
                "Customer authenticated via primary key. Showing ONLY orders owned by this customer."
                if customer_orders
                else "Customer authenticated via primary key, but has no registered orders."
            ),
        }

    def simulate_payment_and_shipping(
        self,
        amount_brl: float,
        has_active_promotion: bool = False,
        discount_percent: float = 0.0,
        city: str = "Campo Grande",
        is_large_instrument: bool = False,
    ) -> Dict[str, Any]:
        """Run deterministic payment and shipping calculations."""
        return {
            "payment": PricingService.calculate_payment_options(
                base_price_brl=float(amount_brl),
                has_active_promotion=has_active_promotion,
                discount_percent=float(discount_percent),
            ),
            "shipping": PricingService.calculate_shipping_policy(
                order_total_brl=float(amount_brl),
                city=city,
                is_large_instrument=is_large_instrument,
            ),
        }

    def consult_store_policies(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve relevant store policy sections from políticas_da_loja.pdf."""
        sections = self.policy_retriever.search(query=query, top_k=top_k)
        return {
            "query": query,
            "matched_sections": sections,
        }

