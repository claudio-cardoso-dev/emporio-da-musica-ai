"""Tests for Stage 2: Repositories and deterministic pricing, installment, and shipping rules."""

from pathlib import Path
import pytest

from src.db.ingestion import ingest_all_csvs
from src.repositories.customer_repository import CustomerRepository
from src.repositories.order_repository import OrderRepository
from src.repositories.product_repository import ProductRepository
from src.repositories.failed_intent_repository import FailedIntentRepository
from src.services.pricing_service import PricingService


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """Create and seed a temporary database for repository tests."""
    db_file = tmp_path / "test_repo.db"
    ingest_all_csvs(db_path=db_file, force_reload=True)
    return db_file


def test_search_acoustic_guitars_under_1000_available(seeded_db: Path) -> None:
    """
    Scenario: 'Quais opções de violões disponíveis custando até R$1000?'
    Must exclude out-of-stock (ID 96 Giannini GF-3D) and discontinued (ID 113 Shelby SN-7C).
    """
    repo = ProductRepository(seeded_db)
    results = repo.search_products(
        category_query="Violões",
        max_price=1000.0,
        only_available=True,
        limit=25,
    )
    assert len(results) == 12
    returned_ids = {p["product_id"] for p in results}
    assert 96 not in returned_ids, "Out-of-stock Giannini GF-3D must not appear in available list"
    assert 113 not in returned_ids, "Discontinued Shelby SN-7C must not appear in available list"
    for prod in results:
        assert prod["effective_price_brl"] <= 1000.0
        assert prod["in_stock"] is True


def test_takamine_gd20_price_and_expired_promotion(seeded_db: Path) -> None:
    """
    Scenario: 'Quanto custa o Takamine GD20?'
    Takamine GD20 (ID 95) has base price R$ 2199.0, an expired promotion (ID 4, 12% off, is_active=0),
    and therefore is eligible for the 5% PIX discount (R$ 2089.05).
    """
    repo = ProductRepository(seeded_db)
    results = repo.search_products(query="Takamine GD20")
    assert len(results) >= 1
    gd20 = results[0]
    assert gd20["product_id"] == 95
    assert gd20["price_brl"] == 2199.0
    assert gd20["active_promotion"] is None
    assert len(gd20["expired_promotions"]) == 1
    assert gd20["expired_promotions"][0]["discount_percent"] == 12.0
    assert gd20["effective_price_brl"] == 2199.0
    assert gd20["pix_price_brl"] == 2089.05
    assert gd20["pix_extra_discount_eligible"] is True


def test_active_promotion_non_cumulative_pix_rule(seeded_db: Path) -> None:
    """
    Policy 6.2: Promotional items do NOT get an additional 5% PIX discount.
    Product ID 90 (Crafter HT-100 Folk) has base price R$ 2399.0 and active 18% promotion (ID 19).
    """
    repo = ProductRepository(seeded_db)
    crafter = repo.get_by_id(90)
    assert crafter is not None
    assert crafter["active_promotion"] is not None
    assert crafter["active_promotion"]["discount_percent"] == 18.0
    expected_promo_price = round(2399.0 * 0.82, 2)  # 1967.18
    assert crafter["effective_price_brl"] == expected_promo_price
    assert crafter["pix_price_brl"] == expected_promo_price
    assert crafter["pix_extra_discount_eligible"] is False


def test_installment_tiers_and_combined_payment() -> None:
    """
    Policy 3.1:
    - Up to 3x: min R$ 50/installment
    - 4x to 6x: min R$ 80/installment
    - 7x to 12x: min R$ 100/installment
    - Combined payment allowed for > R$ 2000.0
    """
    # Item costing R$ 159.90 (e.g., Shelby SU-21S Ukulele):
    # 3x = 53.30 (>= 50 valid), 4x = 39.98 (< 80 invalid) -> max_installments = 3
    ukulele_calc = PricingService.calculate_payment_options(159.90)
    assert ukulele_calc["max_installments"] == 3
    assert ukulele_calc["combined_payment_allowed"] is False

    # Item costing R$ 479.00 (e.g., Kalani KAL-500C):
    # 5x = 95.80 (>= 80 valid), 6x = 79.83 (< 80 invalid) -> max_installments = 5
    kalani_calc = PricingService.calculate_payment_options(479.00)
    assert kalani_calc["max_installments"] == 5

    # Item costing R$ 2199.00 (Takamine GD20):
    # 12x = 183.25 (>= 100 valid), > 2000 so combined payment is True
    takamine_calc = PricingService.calculate_payment_options(2199.00)
    assert takamine_calc["max_installments"] == 12
    assert takamine_calc["combined_payment_allowed"] is True


def test_shipping_rules_campo_grande_vs_other_cities() -> None:
    """
    Policy 5.1 & 5.2:
    - Campo Grande >= R$ 500 -> Free shipping via motoboy (1-3 business days)
    - Campo Grande < R$ 500 -> R$ 35.00 fixed fee
    - Other cities (e.g., Dourados) -> PAC, SEDEX, Jadlog
    """
    cg_free = PricingService.calculate_shipping_policy(599.90, city="Campo Grande")
    assert cg_free["free_shipping"] is True
    assert cg_free["shipping_cost_brl"] == 0.0

    cg_paid = PricingService.calculate_shipping_policy(349.90, city="Campo Grande")
    assert cg_paid["free_shipping"] is False
    assert cg_paid["shipping_cost_brl"] == 35.0

    other_city = PricingService.calculate_shipping_policy(1200.0, city="Dourados", is_large_instrument=True)
    assert other_city["free_shipping"] is False
    assert len(other_city["modalities"]) == 3


def test_alternatives_for_unavailable_product(seeded_db: Path) -> None:
    """Policy 7.3: When product is out of stock (ID 96), suggest available alternatives in same category."""
    repo = ProductRepository(seeded_db)
    alternatives = repo.get_alternatives_for_unavailable(96, limit=3)
    assert len(alternatives) == 3
    for alt in alternatives:
        assert alt["product_id"] != 96
        assert alt["category_id"] == 5
        assert alt["in_stock"] is True


def test_order_and_customer_lookup(seeded_db: Path) -> None:
    """Verify order lookup by ID, tracking code, and customer search."""
    order_repo = OrderRepository(seeded_db)
    cust_repo = CustomerRepository(seeded_db)

    # Order #8: shipped with tracking BRJL5544332BR
    order_8 = order_repo.get_by_order_id(8)
    assert order_8 is not None
    assert order_8["status"] == "shipped"
    assert order_8["tracking_code"] == "BRJL5544332BR"
    assert len(order_8["items"]) == 1

    # Lookup by tracking code
    by_track = order_repo.get_by_tracking_code("brjl5544332br")
    assert by_track is not None
    assert by_track["order_id"] == 8

    # Lookup customer by phone
    customers = cust_repo.find_customers("99654-7821")
    assert len(customers) == 1
    assert customers[0]["name"] == "Ana Carolina Ferreira"


def test_failed_intent_repository_logging(seeded_db: Path) -> None:
    """Verify logging and retrieval of ambiguous/low-confidence user messages."""
    failed_repo = FailedIntentRepository(seeded_db)
    row_id = failed_repo.log_failure(
        session_id="sess-123",
        user_message="quero ver aquele negocio la",
        detected_language="pt",
        reason="ambiguous_intent",
        confidence=0.32,
        resolution_strategy="disambiguation_options",
    )
    assert row_id > 0
    logs = failed_repo.get_all_failures()
    assert len(logs) == 1
    assert logs[0]["session_id"] == "sess-123"
    assert logs[0]["reason"] == "ambiguous_intent"
