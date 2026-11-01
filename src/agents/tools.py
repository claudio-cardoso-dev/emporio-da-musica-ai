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
                "Look up customer orders, delivery status, tracking codes, order items, and cancellation reasons "
                "by order_id, tracking_code, or customer identifier (name, phone, or email)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "Numeric order ID (1 to 20).",
                    },
                    "tracking_code": {
                        "type": "string",
                        "description": "Shipping tracking code (e.g., 'BRJL5544332BR').",
                    },
                    "customer_identifier": {
                        "type": "string",
                        "description": "Customer name, phone number, or email address.",
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
            alternatives = self.product_repo.get_alternatives_for_unavailable(
                product["product_id"], limit=3
            )

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
    ) -> Dict[str, Any]:
        """Look up order status by order ID, tracking code, or customer name/phone/email."""
        if order_id is not None:
            order = self.order_repo.get_by_order_id(int(order_id))
            if order:
                return {"found": True, "orders": [order]}

        if tracking_code:
            order = self.order_repo.get_by_tracking_code(tracking_code)
            if order:
                return {"found": True, "orders": [order]}

        if customer_identifier:
            # Check if customer_identifier looks like 'Pedido 8' or '#8'
            num_match = re.search(r"#?(\d{1,2})\b", customer_identifier)
            if num_match and len(customer_identifier.strip()) <= 10:
                maybe_order = self.order_repo.get_by_order_id(int(num_match.group(1)))
                if maybe_order:
                    return {"found": True, "orders": [maybe_order]}

            customers = self.customer_repo.find_customers(customer_identifier)
            if len(customers) > 1:
                return {
                    "found": False,
                    "ambiguous_customers": True,
                    "message": "Multiple customers matched this name. Ask the user to confirm their email, phone, or full name.",
                    "matching_customers": [
                        {
                            "customer_id": c["customer_id"],
                            "name": c["name"],
                            "city": c["city"],
                            "masked_email": c["email"][:3] + "***@" + c["email"].split("@")[-1],
                        }
                        for c in customers
                    ],
                }
            if len(customers) == 1:
                cust = customers[0]
                orders = self.order_repo.get_orders_by_customer(cust["customer_id"])
                return {
                    "found": len(orders) > 0,
                    "customer": cust,
                    "orders": orders,
                    "message": "Orders retrieved successfully."
                    if orders
                    else "Customer found in database, but has no registered orders.",
                }

        return {
            "found": False,
            "message": "No order or customer found with the provided identifier. Please verify the order number, email, phone, or tracking code.",
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
