"""ETL pipeline to ingest store operational CSV files into SQLite."""

import csv
import json
from pathlib import Path
from typing import Dict, Optional

from src.config.settings import settings
from src.db.database import db_cursor, init_schema


CSV_FILENAMES = {
    "categories": "desafio_tecnico_ai_eng - categories.csv",
    "customers": "desafio_tecnico_ai_eng - customers.csv",
    "products": "desafio_tecnico_ai_eng - products.csv",
    "promotions": "desafio_tecnico_ai_eng - promotions.csv",
    "orders": "desafio_tecnico_ai_eng - orders.csv",
    "order_items": "desafio_tecnico_ai_eng - order_items.csv",
}


def validate_specs_json(raw_specs: str) -> str:
    """Validate and normalize product specifications JSON string."""
    if not raw_specs or not raw_specs.strip():
        return "{}"
    try:
        parsed = json.loads(raw_specs)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return json.dumps({"raw": raw_specs}, ensure_ascii=False)


def ingest_all_csvs(
    db_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> Dict[str, int]:
    """
    Ingest all operational CSV files into the SQLite database.

    Returns a dictionary mapping table names to the total row count in the DB.
    """
    target_db = db_path or settings.resolved_db_path
    target_data = data_dir or settings.resolved_data_dir

    init_schema(target_db)

    with db_cursor(target_db) as cursor:
        if force_reload:
            cursor.execute("DELETE FROM order_items;")
            cursor.execute("DELETE FROM orders;")
            cursor.execute("DELETE FROM promotions;")
            cursor.execute("DELETE FROM products;")
            cursor.execute("DELETE FROM customers;")
            cursor.execute("DELETE FROM categories;")

        # Check if data is already loaded
        cursor.execute("SELECT COUNT(*) as cnt FROM products;")
        existing_products = cursor.fetchone()["cnt"]
        if existing_products > 0 and not force_reload:
            return get_table_counts(target_db)

        # 1. Categories
        cat_path = target_data / CSV_FILENAMES["categories"]
        with open(cat_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO categories (category_id, name, description)
                    VALUES (?, ?, ?)
                    """,
                    (
                        int(row["category_id"]),
                        row["name"].strip(),
                        row["description"].strip(),
                    ),
                )

        # 2. Customers
        cust_path = target_data / CSV_FILENAMES["customers"]
        with open(cust_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO customers (customer_id, name, phone, email, city)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        int(row["customer_id"]),
                        row["name"].strip(),
                        row["phone"].strip(),
                        row["email"].strip(),
                        row["city"].strip(),
                    ),
                )

        # 3. Products
        prod_path = target_data / CSV_FILENAMES["products"]
        with open(prod_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO products (
                        product_id, price_brl, name, category_id, description,
                        stock_quantity, status, specs_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(row["product_id"]),
                        float(row["price_brl"]),
                        row["name"].strip(),
                        int(row["category_id"]),
                        row["description"].strip(),
                        int(row["stock_quantity"]),
                        row["status"].strip(),
                        validate_specs_json(row.get("specs", "{}")),
                        row["created_at"].strip(),
                    ),
                )

        # 4. Promotions
        promo_path = target_data / CSV_FILENAMES["promotions"]
        with open(promo_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO promotions (
                        promotion_id, product_id, discount_percent, description, is_active
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        int(row["promotion_id"]),
                        int(row["product_id"]),
                        float(row["discount_percent"]),
                        row["description"].strip(),
                        int(row["is_active"]),
                    ),
                )

        # 5. Orders
        orders_path = target_data / CSV_FILENAMES["orders"]
        with open(orders_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO orders (
                        order_id, customer_id, order_date, status, total_brl,
                        payment_method, tracking_code, estimated_delivery, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(row["order_id"]),
                        int(row["customer_id"]),
                        row["order_date"].strip(),
                        row["status"].strip(),
                        float(row["total_brl"]),
                        row["payment_method"].strip() if row.get("payment_method") else None,
                        row["tracking_code"].strip() if row.get("tracking_code") else None,
                        row["estimated_delivery"].strip() if row.get("estimated_delivery") else None,
                        row["notes"].strip() if row.get("notes") else None,
                    ),
                )

        # 6. Order Items
        items_path = target_data / CSV_FILENAMES["order_items"]
        cursor.execute("DELETE FROM order_items;")
        with open(items_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """
                    INSERT INTO order_items (order_id, quantity, product_id)
                    VALUES (?, ?, ?)
                    """,
                    (
                        int(row["order_id"]),
                        int(row["quantity"]),
                        int(row["product_id"]),
                    ),
                )

    return get_table_counts(target_db)


def get_table_counts(db_path: Optional[Path] = None) -> Dict[str, int]:
    """Return row counts for all ingested operational tables."""
    tables = ["categories", "customers", "products", "promotions", "orders", "order_items"]
    counts: Dict[str, int] = {}
    with db_cursor(db_path) as cursor:
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as cnt FROM {table};")
            counts[table] = int(cursor.fetchone()["cnt"])
    return counts
