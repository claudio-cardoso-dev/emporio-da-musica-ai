"""Tests for Stage 3: RAG Policy Retriever with >= 3 test cases per Store Policy Section (32 tests total)."""

import pytest

from src.rag.policy_retriever import PolicyRetriever


@pytest.fixture(scope="module")
def retriever() -> PolicyRetriever:
    """Initialize the PolicyRetriever once for the module."""
    return PolicyRetriever()


def test_all_10_sections_parsed(retriever: PolicyRetriever) -> None:
    """Verify that all 10 numbered sections of políticas_da_loja.pdf are parsed."""
    assert len(retriever.chunks) == 10
    section_numbers = [c["section_number"] for c in retriever.chunks]
    assert section_numbers == list(range(1, 11))


# ============================================================================
# Section 1: Sobre a Empório da Música (3+ tests)
# ============================================================================
def test_sec1_store_address_and_cnpj(retriever: PolicyRetriever) -> None:
    """Test 1.1: Retrieve store address, CEP, and CNPJ."""
    results = retriever.search("Qual o endereço físico, CEP e CNPJ da Empório da Música?", top_k=2)
    sec_nums = [r["section_number"] for r in results]
    assert 1 in sec_nums
    sec1 = retriever.get_section_by_number(1)
    assert "Rua 14 de Maio, 3200" in sec1["content"]
    assert "12.345.678/0001-90" in sec1["content"]


def test_sec1_mission_and_foundation_year(retriever: PolicyRetriever) -> None:
    """Test 1.2: Retrieve store mission and foundation year (2008)."""
    results = retriever.search("Qual é a missão da loja e em que ano ela foi fundada?", top_k=2)
    assert 1 in [r["section_number"] for r in results]
    assert "2008" in results[0]["content"]


def test_sec1_exclusive_instruments_no_accessories(retriever: PolicyRetriever) -> None:
    """Test 1.3: Verify policy on not selling accessories (strings, picks, cables, amps)."""
    results = retriever.search("Vocês vendem encordoamento, palhetas, cabos, pedais ou amplificadores?", top_k=2)
    sec_nums = [r["section_number"] for r in results]
    assert 1 in sec_nums or 7 in sec_nums


# ============================================================================
# Section 2: Horário de Funcionamento (3+ tests)
# ============================================================================
def test_sec2_weekday_opening_hours(retriever: PolicyRetriever) -> None:
    """Test 2.1: Retrieve Monday to Friday opening hours (09:00 to 18:00)."""
    results = retriever.search("Qual o horário de funcionamento de segunda a sexta-feira?", top_k=2)
    assert results[0]["section_number"] == 2
    assert "09:00 às 18:00" in results[0]["content"]


def test_sec2_saturday_and_sunday_hours(retriever: PolicyRetriever) -> None:
    """Test 2.2: Retrieve Saturday hours (09:00 to 13:00) and Sunday/Holiday closure."""
    results = retriever.search("A loja abre aos sábados, domingos e feriados?", top_k=2)
    assert results[0]["section_number"] == 2
    assert "09:00 às 13:00" in results[0]["content"]
    assert "Fechado" in results[0]["content"]


def test_sec2_special_dates_extended_hours(retriever: PolicyRetriever) -> None:
    """Test 2.3: Retrieve rule for extended hours on special dates (Black Friday, Natal)."""
    results = retriever.search("O horário de expediente muda no Natal ou Black Friday?", top_k=2)
    assert 2 in [r["section_number"] for r in results]


# ============================================================================
# Section 3: Formas de Pagamento e Parcelamento (3+ tests)
# ============================================================================
def test_sec3_pix_5_percent_discount_and_boleto(retriever: PolicyRetriever) -> None:
    """Test 3.1: Retrieve PIX 5% cash discount and Boleto 3 business days compensation."""
    results = retriever.search("Tem desconto pagando à vista no PIX? E quanto tempo demora o boleto?", top_k=2)
    assert 3 in [r["section_number"] for r in results]
    sec3 = retriever.get_section_by_number(3)
    assert "5% de desconto" in sec3["content"]
    assert "3 dias úteis" in sec3["content"]


def test_sec3_credit_card_installment_minimums(retriever: PolicyRetriever) -> None:
    """Test 3.2: Retrieve minimum installment rules for 3x, 4x-6x (R$ 80), and 7x-12x (R$ 100)."""
    results = retriever.search("Quais são as regras de parcelamento sem juros e valor mínimo de parcela?", top_k=2)
    assert results[0]["section_number"] == 3
    assert "R$ 80,00" in results[0]["content"]
    assert "R$ 100,00" in results[0]["content"]


def test_sec3_combined_payment_methods_above_2000(retriever: PolicyRetriever) -> None:
    """Test 3.3: Retrieve rule allowing combined payment (PIX + Card) above R$ 2,000.00."""
    results = retriever.search("Posso combinar duas formas de pagamento como dar uma entrada no PIX e parcelar o resto no cartão?", top_k=2)
    assert results[0]["section_number"] == 3
    assert "R$ 2.000,00" in results[0]["content"]


# ============================================================================
# Section 4: Política de Trocas e Devoluções (3+ tests)
# ============================================================================
def test_sec4_online_regret_7_days_and_return_shipping(retriever: PolicyRetriever) -> None:
    """Test 4.1: Retrieve 7-day right of regret, 10 business days refund, and free return shipping."""
    results = retriever.search("Me arrependi da compra online, qual o prazo para devolver e quem paga o frete de devolução?", top_k=2)
    assert results[0]["section_number"] == 4
    assert "7 (sete) dias corridos" in results[0]["content"]
    assert "10 dias úteis" in results[0]["content"]


def test_sec4_manufacturing_defect_30_days_and_preference_exchange(retriever: PolicyRetriever) -> None:
    """Test 4.2: Retrieve 30-day store defect exchange and 7-day preference exchange."""
    results = retriever.search("O instrumento veio com defeito de fabricação ou quero trocar de cor, quantos dias eu tenho?", top_k=2)
    assert results[0]["section_number"] == 4
    assert "30 (trinta) dias" in results[0]["content"]


def test_sec4_ineligible_items_for_exchange(retriever: PolicyRetriever) -> None:
    """Test 4.3: Retrieve non-exchangeable items (boquilhas de sopro por higiene, setup sob encomenda, venda final)."""
    results = retriever.search("Posso trocar uma boquilha de saxofone ou um violão com regulagem especial sob encomenda?", top_k=2)
    assert results[0]["section_number"] == 4
    assert "Boquilhas" in results[0]["content"]


# ============================================================================
# Section 5: Política de Frete, Entregas e Rastreamento (3+ tests)
# ============================================================================
def test_sec5_campo_grande_metro_shipping_and_motoboy(retriever: PolicyRetriever) -> None:
    """Test 5.1: Retrieve Campo Grande free shipping threshold (R$ 500), R$ 35 fee, and 1-3 days motoboy."""
    results = retriever.search("Quanto custa o frete para Campo Grande e qual o prazo de entrega por motoboy?", top_k=2)
    assert results[0]["section_number"] == 5
    assert "R$ 500,00" in results[0]["content"]
    assert "R$ 35,00" in results[0]["content"]


def test_sec5_other_cities_carriers_and_large_instruments(retriever: PolicyRetriever) -> None:
    """Test 5.2: Retrieve PAC, SEDEX, Jadlog delivery times and large instrument special quote."""
    results = retriever.search("Quais transportadoras entregam para outras cidades (PAC, SEDEX, Jadlog) e como funciona para baterias ou pianos grandes?", top_k=2)
    assert results[0]["section_number"] == 5
    assert "Jadlog" in results[0]["content"]


def test_sec5_tracking_code_format_and_damaged_package_insurance(retriever: PolicyRetriever) -> None:
    """Test 5.3: Retrieve tracking code format (BR + 9 chars + BR), insurance, and damaged package refusal."""
    results = retriever.search("Qual o formato do código de rastreamento e o que fazer em caso de avaria no transporte?", top_k=2)
    assert results[0]["section_number"] == 5
    assert "BR4K7M2X9P1BR" in results[0]["content"]


# ============================================================================
# Section 6: Promoções e Descontos (3+ tests)
# ============================================================================
def test_sec6_seasonal_promotional_campaigns(retriever: PolicyRetriever) -> None:
    """Test 6.1: Retrieve seasonal campaigns (Aniversário Agosto, Black Friday, Volta às Aulas, Semana do Músico)."""
    results = retriever.search("Quais são as campanhas de promoção ao longo do ano como Semana do Músico e Aniversário da Loja?", top_k=2)
    assert results[0]["section_number"] == 6
    assert "22 de novembro" in results[0]["content"]


def test_sec6_non_cumulative_promotions_and_pix(retriever: PolicyRetriever) -> None:
    """Test 6.2: Verify non-cumulativity rule (PIX 5% does not apply to promotional prices)."""
    results = retriever.search("Promoções são cumulativas com o desconto de 5% do PIX?", top_k=2)
    assert results[0]["section_number"] == 6
    assert "não são cumulativas" in results[0]["content"]


def test_sec6_no_rain_check_and_price_transparency(retriever: PolicyRetriever) -> None:
    """Test 6.3: Verify no rain check on sold-out promotional items and transparent price communication."""
    results = retriever.search("Se o produto acabar na promoção tenho direito a rain check (reserva de preço)?", top_k=2)
    assert results[0]["section_number"] == 6
    assert "rain check" in results[0]["content"]


# ============================================================================
# Section 7: Atendimento via WhatsApp e Situações Especiais (3+ tests)
# ============================================================================
def test_sec7_whatsapp_number_tone_and_5min_sla(retriever: PolicyRetriever) -> None:
    """Test 7.1: Retrieve WhatsApp channel (67) 3321-4500, tone of voice, and 5-minute response time."""
    results = retriever.search("Qual o número do WhatsApp de atendimento, tom de voz e tempo máximo de resposta?", top_k=2)
    assert results[0]["section_number"] == 7
    assert "(67) 3321-4500" in results[0]["content"]
    assert "5 minutos" in results[0]["content"]


def test_sec7_out_of_stock_discontinued_and_expired_promo_handling(retriever: PolicyRetriever) -> None:
    """Test 7.2: Retrieve procedure for out-of-stock, discontinued products, and expired promotions."""
    results = retriever.search("Como proceder quando um produto está fora de estoque, descontinuado ou com promoção expirada?", top_k=2)
    assert results[0]["section_number"] == 7
    assert "temporariamente indisponível" in results[0]["content"]


def test_sec7_complaint_handling_24h_sla(retriever: PolicyRetriever) -> None:
    """Test 7.3: Retrieve complaint escalation procedure and 24 business hours SLA."""
    results = retriever.search("Quero registrar uma reclamação no atendimento, qual o prazo de retorno do responsável?", top_k=2)
    assert results[0]["section_number"] == 7
    assert "24 horas úteis" in results[0]["content"]


# ============================================================================
# Section 8: Garantia (3+ tests)
# ============================================================================
def test_sec8_legal_warranty_90_days_cdc(retriever: PolicyRetriever) -> None:
    """Test 8.1: Retrieve 90-day legal warranty under CDC from delivery date."""
    results = retriever.search("Qual o prazo da garantia legal pelo Código de Defesa do Consumidor?", top_k=2)
    assert results[0]["section_number"] == 8
    assert "90 (noventa) dias" in results[0]["content"]


def test_sec8_manufacturer_warranty_6m_to_2y(retriever: PolicyRetriever) -> None:
    """Test 8.2: Retrieve manufacturer warranty duration (6 months to 2 years)."""
    results = retriever.search("Quanto tempo dura a garantia do fabricante além da garantia legal?", top_k=2)
    assert results[0]["section_number"] == 8
    assert "6 meses a 2 anos" in results[0]["content"]


def test_sec8_warranty_exclusions_wear_and_misuse(retriever: PolicyRetriever) -> None:
    """Test 8.3: Retrieve what is not covered by warranty (natural wear of frets/strings, misuse, aesthetic damage)."""
    results = retriever.search("O que a garantia não cobre? Desgaste natural de trastes, cordas ou danos estéticos são cobertos?", top_k=2)
    assert results[0]["section_number"] == 8
    assert "Desgaste natural" in results[0]["content"]


# ============================================================================
# Section 9: Privacidade e Proteção de Dados - LGPD (3+ tests)
# ============================================================================
def test_sec9_lgpd_compliance_and_data_usage_purposes(retriever: PolicyRetriever) -> None:
    """Test 9.1: Retrieve LGPD compliance (Law 13.709/2018) and allowed data purposes."""
    results = retriever.search("Como a loja utiliza meus dados pessoais segundo a LGPD?", top_k=2)
    assert results[0]["section_number"] == 9
    assert "13.709/2018" in results[0]["content"]


def test_sec9_no_third_party_marketing_sharing(retriever: PolicyRetriever) -> None:
    """Test 9.2: Verify customer data is never shared with third parties for marketing."""
    results = retriever.search("Meus dados pessoais são compartilhados com terceiros para marketing?", top_k=2)
    assert results[0]["section_number"] == 9
    assert "não são compartilhados com terceiros" in results[0]["content"]


def test_sec9_right_to_data_deletion(retriever: PolicyRetriever) -> None:
    """Test 9.3: Verify how a customer can request deletion of personal data."""
    results = retriever.search("Como posso solicitar a exclusão dos meus dados pessoais do cadastro da loja?", top_k=2)
    assert results[0]["section_number"] == 9
    assert "exclusão de seus dados" in results[0]["content"]


# ============================================================================
# Section 10: Disposições Finais & Multilingual Queries (4+ tests)
# ============================================================================
def test_sec10_internal_document_and_periodic_updates(retriever: PolicyRetriever) -> None:
    """Test 10.1: Verify internal manual status and periodic updates."""
    results = retriever.search("Disposições finais sobre atualizações periódicas deste manual interno", top_k=2)
    assert results[0]["section_number"] == 10
    assert "atualizações periódicas" in results[0]["content"]


def test_sec10_consult_management_on_policy_doubts(retriever: PolicyRetriever) -> None:
    """Test 10.2: Verify rule to consult management (gerência) when in doubt about policies."""
    results = retriever.search("Quem o atendente deve consultar em caso de dúvidas sobre aplicação de políticas?", top_k=2)
    assert results[0]["section_number"] == 10
    assert "consultar a gerência" in results[0]["content"]


def test_sec10_manual_version_and_date(retriever: PolicyRetriever) -> None:
    """Test 10.3: Verify manual version (Versão 2.1 - Junho de 2025)."""
    sec10 = retriever.get_section_by_number(10)
    assert "Versão 2.1" in sec10["content"]
    assert "Junho de 2025" in sec10["content"]


def test_multilingual_english_and_spanish_policy_queries(retriever: PolicyRetriever) -> None:
    """Test 10.4: Verify that English and Spanish policy queries retrieve the right sections."""
    en_res = retriever.search("What are the opening hours on Saturday and Sunday?", top_k=2)
    assert en_res[0]["section_number"] == 2

    es_res = retriever.search("¿Cuál es la política de devolución, cambio y reembolso?", top_k=2)
    assert es_res[0]["section_number"] == 4

