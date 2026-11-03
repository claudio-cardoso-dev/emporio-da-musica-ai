"""Repository for querying musical instruments, categories, and promotions."""

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional
from rapidfuzz import fuzz, process

from src.db.database import db_cursor


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and stripping accents for robust matching."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


class ProductRepository:
    """Data access layer for products, categories, and associated promotions."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path

    def _enrich_product(self, row: Any) -> Dict[str, Any]:
        """Enrich raw SQLite product row with parsed specs and promotion metadata."""
        product = dict(row)
        try:
            product["specs"] = json.loads(product.get("specs_json") or "{}")
        except json.JSONDecodeError:
            product["specs"] = {}
        product.pop("specs_json", None)

        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                """
                SELECT promotion_id, discount_percent, description, is_active
                FROM promotions
                WHERE product_id = ?
                ORDER BY is_active DESC, discount_percent DESC
                """,
                (product["product_id"],),
            )
            promos = [dict(r) for r in cursor.fetchall()]

        active_promo = next((p for p in promos if p["is_active"] == 1), None)
        expired_promos = [p for p in promos if p["is_active"] == 0]

        base_price = float(product["price_brl"])
        if active_promo:
            discount = float(active_promo["discount_percent"])
            final_price = round(base_price * (1.0 - discount / 100.0), 2)
            # Store Policy 6.2: PIX 5% discount is NOT cumulative with active promotions
            pix_price = final_price
            pix_discount_applied = False
        else:
            discount = 0.0
            final_price = base_price
            pix_price = round(base_price * 0.95, 2)
            pix_discount_applied = True

        product["active_promotion"] = active_promo
        product["expired_promotions"] = expired_promos
        product["effective_price_brl"] = final_price
        product["pix_price_brl"] = pix_price
        product["pix_extra_discount_eligible"] = pix_discount_applied
        product["in_stock"] = (
            product["status"] == "active" and int(product["stock_quantity"]) > 0
        )
        return product

    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Retrieve all store musical instrument categories."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                "SELECT category_id, name, description FROM categories ORDER BY category_id;"
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_by_id(self, product_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single product by ID with full category and promotion details."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                """
                SELECT p.*, c.name AS category_name
                FROM products p
                JOIN categories c ON p.category_id = c.category_id
                WHERE p.product_id = ?
                """,
                (product_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._enrich_product(row)

    def search_products(
        self,
        query: Optional[str] = None,
        category_query: Optional[str] = None,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        only_available: bool = False,
        only_on_promotion: bool = False,
        strings_type: Optional[str] = None,
        electronics: Optional[str] = None,
        limit: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Search products using structured filters combined with fuzzy/semantic matching.
        """
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                """
                SELECT p.*, c.name AS category_name, c.description AS category_description
                FROM products p
                JOIN categories c ON p.category_id = c.category_id
                ORDER BY p.price_brl ASC
                """
            )
            all_products = [self._enrich_product(r) for r in cursor.fetchall()]

        results = all_products

        # 1. Category filter
        if category_query and category_query.strip():
            norm_cat = normalize_text(category_query)
            cat_aliases = {
                "violao": "violoes",
                "violoes": "violoes",
                "acoustic guitar": "violoes",
                "guitarra": "guitarras",
                "electric guitar": "guitarras",
                "baixo": "baixos",
                "contrabaixo": "baixos",
                "bass": "baixos",
                "bateria": "baterias",
                "percussao": "baterias",
                "drums": "baterias",
                "teclado": "teclados",
                "piano": "teclados",
                "sintetizador": "teclados",
                "keyboard": "teclados",
                "ukulele": "ukuleles",
                "sopro": "sopro",
                "violino": "cordas orquestrais",
                "orquestral": "cordas orquestrais",
            }
            mapped_cat = cat_aliases.get(norm_cat, norm_cat)
            results = [
                p
                for p in results
                if mapped_cat in normalize_text(p["category_name"])
                or mapped_cat in normalize_text(p.get("category_description", ""))
                or norm_cat in normalize_text(p["name"])
            ]

        # 2. Strings type filter (e.g., 'nylon' or 'steel'/'aço')
        if strings_type and strings_type.strip():
            norm_str = normalize_text(strings_type)
            if norm_str in ("aco", "steel"):
                target_str = "steel"
            elif "nylon" in norm_str:
                target_str = "nylon"
            else:
                target_str = norm_str
            results = [
                p
                for p in results
                if normalize_text(str(p.get("specs", {}).get("strings", ""))) == target_str
                or norm_str in normalize_text(p["name"])
            ]

        # 3. Electronics filter ('yes'/'no' or 'eletrico'/'acustico')
        if electronics and electronics.strip():
            norm_elec = normalize_text(electronics)
            if norm_elec in ("yes", "sim", "true", "eletrico", "eletroacustico"):
                results = [
                    p
                    for p in results
                    if str(p.get("specs", {}).get("electronics", "")).lower() == "yes"
                ]
            elif norm_elec in ("no", "nao", "false", "acustico"):
                results = [
                    p
                    for p in results
                    if str(p.get("specs", {}).get("electronics", "")).lower() == "no"
                ]

        # 4. Availability filter
        if only_available:
            results = [p for p in results if p["in_stock"]]

        # 5. Promotion filter
        if only_on_promotion:
            results = [p for p in results if p["active_promotion"] is not None]

        # 6. Price filters (check effective price or base price)
        if max_price is not None:
            results = [
                p
                for p in results
                if p["effective_price_brl"] <= max_price or p["price_brl"] <= max_price
            ]
        if min_price is not None:
            results = [
                p
                for p in results
                if p["effective_price_brl"] >= min_price or p["price_brl"] >= min_price
            ]

        # 7. Free-text query matching (exact token + fuzzy score via rapidfuzz)
        if query and query.strip():
            norm_q = normalize_text(query)
            scored_results = []
            for p in results:
                searchable_blob = normalize_text(
                    f"{p['name']} {p['category_name']} {p['description']} {json.dumps(p.get('specs', {}))}"
                )
                name_norm = normalize_text(p["name"])
                if norm_q in name_norm:
                    score = 100.0
                elif norm_q in searchable_blob:
                    score = 92.0
                else:
                    score_name = fuzz.WRatio(norm_q, name_norm)
                    score_blob = fuzz.partial_ratio(norm_q, searchable_blob)
                    score = max(score_name, score_blob * 0.9)

                if score >= 62.0:
                    p_copy = dict(p)
                    p_copy["match_score"] = round(score, 2)
                    scored_results.append(p_copy)

            scored_results.sort(
                key=lambda x: (-x["match_score"], not x["in_stock"], x["effective_price_brl"])
            )
            return scored_results[:limit]

        return results[:limit]

    def get_alternatives_for_unavailable(
        self, product_id: int, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar in-stock active products in the same category when a requested
        product is out of stock, discontinued, or coming soon (Store Policy 7.3).
        """
        target = self.get_by_id(product_id)
        if not target:
            return []

        cat_id = target["category_id"]
        target_price = float(target["price_brl"])
        target_strings = target.get("specs", {}).get("strings")

        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                """
                SELECT p.*, c.name AS category_name
                FROM products p
                JOIN categories c ON p.category_id = c.category_id
                WHERE p.category_id = ?
                  AND p.product_id != ?
                  AND p.status = 'active'
                  AND p.stock_quantity > 0
                """,
                (cat_id, product_id),
            )
            candidates = [self._enrich_product(r) for r in cursor.fetchall()]

        def similarity_key(cand: Dict[str, Any]) -> tuple:
            same_strings = (
                0
                if target_strings and cand.get("specs", {}).get("strings") == target_strings
                else 1
            )
            price_diff = abs(float(cand["effective_price_brl"]) - target_price)
            return (same_strings, price_diff)

        candidates.sort(key=similarity_key)
        return candidates[:limit]
