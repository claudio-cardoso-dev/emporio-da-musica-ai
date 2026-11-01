"""Repository for customer lookup and identification."""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from src.db.database import db_cursor
from src.repositories.product_repository import normalize_text


class CustomerRepository:
    """Data access layer for store customers."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path

    def get_by_id(self, customer_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve customer by customer_id."""
        with db_cursor(self.db_path) as cursor:
            cursor.execute(
                "SELECT * FROM customers WHERE customer_id = ?;",
                (customer_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def find_customers(self, identifier: str) -> List[Dict[str, Any]]:
        """
        Find customers by ID, email, phone number (digits), or partial/full name.
        Returns a list to allow disambiguation if multiple customers share a name.
        """
        if not identifier or not identifier.strip():
            return []

        clean_id = identifier.strip()

        # Check exact numeric customer_id
        if clean_id.isdigit() and len(clean_id) <= 4:
            cust = self.get_by_id(int(clean_id))
            return [cust] if cust else []

        with db_cursor(self.db_path) as cursor:
            cursor.execute("SELECT * FROM customers;")
            all_customers = [dict(r) for r in cursor.fetchall()]

        # Check exact email match
        norm_q = normalize_text(clean_id)
        email_matches = [
            c for c in all_customers if normalize_text(c["email"]) == norm_q
        ]
        if email_matches:
            return email_matches

        # Check phone digits match
        digits_only = re.sub(r"\D", "", clean_id)
        if len(digits_only) >= 8:
            phone_matches = [
                c
                for c in all_customers
                if digits_only in re.sub(r"\D", "", c["phone"])
            ]
            if phone_matches:
                return phone_matches

        # Check name match
        name_matches = [
            c for c in all_customers if norm_q in normalize_text(c["name"])
        ]
        return name_matches
