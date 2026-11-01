"""Tests for Stage 1: Database schema creation and CSV ingestion."""

import json
from pathlib import Path
import pytest

from src.db.database import db_cursor, init_schema
from src.db.ingestion import ingest_all_csvs, get_table_counts


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database and ingest operational CSVs."""
    db_file = tmp_path / "test_emporio.db"
    ingest_all_csvs(db_path=db_file, force_reload=True)
    return db_file


def test_table_counts_match_csv_rows(temp_db: Path) -> None:
    """Verify that all 6 operational CSV files are ingested with exact row counts."""
    counts = get_table_counts(temp_db)
    assert counts["categories"] == 9
    assert counts["customers"] == 50
    assert counts["products"] == 65
    assert counts["promotions"] == 25
    assert counts["orders"] == 20
    assert counts["order_items"] == 22


def test_product_specs_json_validity(temp_db: Path) -> None:
    """Verify that every product row contains a valid JSON object in specs_json."""
    with db_cursor(temp_db) as cursor:
        cursor.execute("SELECT product_id, name, specs_json FROM products;")
        rows = cursor.fetchall()
        assert len(rows) == 65
        for row in rows:
            parsed = json.loads(row["specs_json"])
            assert isinstance(parsed, dict), f"Invalid specs for product {row['product_id']}"


def test_special_product_statuses_and_stock(temp_db: Path) -> None:
    """Verify edge cases in products table: out-of-stock active, discontinued, coming_soon."""
    with db_cursor(temp_db) as cursor:
        # Giannini GF-3D Dreadnought Sunburst (ID 96): active but 0 stock
        cursor.execute("SELECT status, stock_quantity FROM products WHERE product_id = 96;")
        prod_96 = cursor.fetchone()
        assert prod_96["status"] == "active"
        assert prod_96["stock_quantity"] == 0

        # Shelby SN-7C 7 Cordas (ID 113) & Ibanez Bass 2X (ID 136): discontinued
        cursor.execute("SELECT product_id FROM products WHERE status = 'discontinued';")
        discontinued_ids = {r["product_id"] for r in cursor.fetchall()}
        assert discontinued_ids == {113, 136}

        # Kala KA-ABP-CTG Baritono Cedar (ID 130): coming_soon
        cursor.execute("SELECT status FROM products WHERE product_id = 130;")
        prod_130 = cursor.fetchone()
        assert prod_130["status"] == "coming_soon"


def test_active_vs_expired_promotions(temp_db: Path) -> None:
    """Verify that exactly 4 promotions are active and 21 are expired."""
    with db_cursor(temp_db) as cursor:
        cursor.execute("SELECT product_id, discount_percent FROM promotions WHERE is_active = 1;")
        active_promos = {r["product_id"]: r["discount_percent"] for r in cursor.fetchall()}
        assert active_promos == {
            90: 18.0,
            94: 8.0,
            121: 20.0,
            127: 10.0,
        }

        cursor.execute("SELECT COUNT(*) as cnt FROM promotions WHERE is_active = 0;")
        expired_count = cursor.fetchone()["cnt"]
        assert expired_count == 21


def test_idempotent_ingestion(temp_db: Path) -> None:
    """Verify calling ingest_all_csvs twice without force_reload does not duplicate rows."""
    counts_second_run = ingest_all_csvs(db_path=temp_db, force_reload=False)
    assert counts_second_run["products"] == 65
    assert counts_second_run["order_items"] == 22
