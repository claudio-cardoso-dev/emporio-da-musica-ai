"""Repository for querying customer orders and order items."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.db.database import db_cursor
from src.repositories.customer_repository import CustomerRepository


class OrderRepository:
    """Data access layer for orders and associated items."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path
        self.customer_repo = CustomerRepository(db_path)

    def _enrich_order(self, order_row: Any) -> Dict[str, Any]:
        """Attach customer info and order items to an order record."""
        order = dict(order_row)
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                """
                SELECT oi.quantity, oi.product_id, p.name AS product_name, p.price_brl
                FROM order_items oi
                JOIN products p ON oi.product_id = p.product_id
                WHERE oi.order_id = ?
                """,
                (order["order_id"],),
            )
            order["items"] = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                "SELECT customer_id, name, phone, email, city FROM customers WHERE customer_id = ?;",
                (order["customer_id"],),
            )
            cust = cursor.fetchone()
            order["customer"] = dict(cust) if cust else None

        return order

    def get_by_order_id(self, order_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific order by its order_id."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute("SELECT * FROM orders WHERE order_id = ?;", (order_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._enrich_order(row)

    def get_by_tracking_code(self, tracking_code: str) -> Optional[Dict[str, Any]]:
        """Retrieve an order by its shipping tracking code (e.g., BRJL5544332BR)."""
        if not tracking_code:
            return None
        clean_code = tracking_code.strip().upper()
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                "SELECT * FROM orders WHERE UPPER(tracking_code) = ?;",
                (clean_code,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._enrich_order(row)

    def get_orders_by_customer(self, customer_id: int) -> List[Dict[str, Any]]:
        """Retrieve all orders for a given customer_id ordered by most recent."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_date DESC, order_id DESC;",
                (customer_id,),
            )
            rows = cursor.fetchall()
            return [self._enrich_order(r) for r in rows]
