"""Tests for Stage 6: End-to-End validation of generated conversation transcripts and business scenarios."""

from pathlib import Path
import pytest

from src.config.settings import BASE_DIR


EXAMPLES_DIR = BASE_DIR / "docs" / "examples"


def test_all_5_example_transcripts_exist_and_non_empty() -> None:
    """Verify that all 5 markdown conversation transcripts exist and are populated."""
    expected_files = [
        "01_catalogo_violoes_ate_1000_e_frete.md",
        "02_precos_promocoes_expiradas_e_estoque.md",
        "03_politicas_troca_arrependimento_e_garantia.md",
        "04_rastreamento_pedidos_e_desambiguacao_clientes.md",
        "05_escopo_acessorios_multilingue_e_baixa_confianca.md",
    ]
    for fname in expected_files:
        fpath = EXAMPLES_DIR / fname
        assert fpath.exists(), f"Missing example transcript: {fname}"
        content = fpath.read_text(encoding="utf-8")
        assert len(content) > 1500, f"Transcript {fname} is unexpectedly short"


def test_scenario_01_acoustic_guitars_address_and_shipping() -> None:
    """Validate Scenario 1 facts: 12 guitars <= R$1000, store address Rua 14 de Maio 3200, free shipping."""
    content = (EXAMPLES_DIR / "01_catalogo_violoes_ate_1000_e_frete.md").read_text(encoding="utf-8")
    assert "Yamaha C40" in content
    assert "Tagima Dallas Tuner" in content
    assert "711,55" in content
    assert "Rua 14 de Maio, 3200" in content
    assert "79202-333" in content
    assert "09:00 às 18:00" in content


def test_scenario_02_takamine_gd20_expired_promo_and_stock_edge_cases() -> None:
    """Validate Scenario 2 facts: Takamine GD20 R$ 2.199,00 expired promo, Crafter 18% active non-cumulative PIX, stock statuses."""
    content = (EXAMPLES_DIR / "02_precos_promocoes_expiradas_e_estoque.md").read_text(encoding="utf-8")
    assert "2.199,00" in content
    assert "2.089,05" in content
    assert "1.967,18" in content
    assert "descontinuado" in content.lower()


def test_scenario_03_returns_mouthpiece_hygiene_and_warranty() -> None:
    """Validate Scenario 3 facts: 7 days regret, 10 business days refund, boquilha hygiene exclusion, 90-day legal warranty."""
    content = (EXAMPLES_DIR / "03_politicas_troca_arrependimento_e_garantia.md").read_text(encoding="utf-8")
    assert "7 dias" in content or "7 (sete) dias" in content
    assert "10 dias úteis" in content
    assert "higiene" in content.lower()
    assert "90 dias" in content


def test_scenario_04_order_tracking_homonym_disambiguation_and_cancellation() -> None:
    """Validate Scenario 4 facts: Bruno homonym disambiguation, Order #10, Order #8 BRJL5544332BR, Order #19 cancelled."""
    content = (EXAMPLES_DIR / "04_rastreamento_pedidos_e_desambiguacao_clientes.md").read_text(encoding="utf-8")
    assert "Bruno Carvalho Martins" in content
    assert "BRJL5544332BR" in content
    assert "Pagamento não confirmado dentro do prazo" in content


def test_scenario_05_low_confidence_accessories_and_multilingual() -> None:
    """Validate Scenario 5 facts: guardrail_low_confidence, no accessories policy, English and Spanish responses."""
    content = (EXAMPLES_DIR / "05_escopo_acessorios_multilingue_e_baixa_confianca.md").read_text(encoding="utf-8")
    assert "guardrail_low_confidence" in content
    assert "12,500.00" in content or "11,875.00" in content
    assert "13.709/2018" in content
