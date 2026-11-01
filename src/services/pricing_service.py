"""Deterministic pricing, installment, and shipping rules engine based on Store Policies."""

from typing import Any, Dict, List

from src.repositories.product_repository import normalize_text


CAMPO_GRANDE_METRO_CITIES = {
    "campo grande",
    "terenos",
    "jaraguari",
    "sidrolandia",
    "rochedo",
}


class PricingService:
    """
    Deterministic calculator enforcing Emporio da Musica financial and shipping rules:
    - Section 3 & 6.2: PIX has 5% discount ONLY when the item does NOT have an active promotion.
    - Section 3.1: Credit card up to 12x interest-free with minimum installment thresholds:
      * 1x to 3x: minimum installment R$ 50.00 (except 1x which can be any value)
      * 4x to 6x: minimum installment R$ 80.00
      * 7x to 12x: minimum installment R$ 100.00
      * Combined payment (e.g., PIX + Credit Card) allowed for orders > R$ 2,000.00.
    - Section 5.1 & 5.2: Shipping rules for Campo Grande Metro vs. other cities.
    """

    @staticmethod
    def calculate_payment_options(
        base_price_brl: float,
        has_active_promotion: bool = False,
        discount_percent: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute exact PIX, debit, boleto, and credit card installment breakdown."""
        if has_active_promotion and discount_percent > 0:
            effective_price = round(base_price_brl * (1.0 - discount_percent / 100.0), 2)
            # Policy 6.2: PIX 5% discount is NOT cumulative with promotional prices
            pix_price = effective_price
            pix_discount_applied = False
        else:
            effective_price = round(base_price_brl, 2)
            pix_price = round(base_price_brl * 0.95, 2)
            pix_discount_applied = True

        valid_installments: List[Dict[str, Any]] = []
        max_installments = 1

        for n in range(1, 13):
            installment_value = round(effective_price / n, 2)
            if n == 1:
                is_valid = True
            elif 2 <= n <= 3:
                is_valid = installment_value >= 50.0
            elif 4 <= n <= 6:
                is_valid = installment_value >= 80.0
            else:  # 7 <= n <= 12
                is_valid = installment_value >= 100.0

            if is_valid:
                max_installments = n
                valid_installments.append(
                    {
                        "installments": n,
                        "installment_value_brl": installment_value,
                        "interest_free": True,
                    }
                )

        max_option = next(
            (opt for opt in valid_installments if opt["installments"] == max_installments),
            {"installments": 1, "installment_value_brl": effective_price},
        )

        return {
            "original_price_brl": round(base_price_brl, 2),
            "has_active_promotion": has_active_promotion,
            "promotion_discount_percent": discount_percent if has_active_promotion else 0.0,
            "effective_price_brl": effective_price,
            "pix_price_brl": pix_price,
            "pix_5_percent_applied": pix_discount_applied,
            "pix_rule_note": (
                "Promoções não são cumulativas: o desconto de 5% no PIX não se aplica a itens já em promoção."
                if has_active_promotion
                else "5% de desconto aplicado para pagamento à vista no PIX."
            ),
            "debit_card_brl": effective_price,
            "boleto_brl": effective_price,
            "boleto_compensation_days": "até 3 dias úteis",
            "max_installments": max_installments,
            "max_installment_value_brl": max_option["installment_value_brl"],
            "valid_installment_options": valid_installments,
            "combined_payment_allowed": effective_price > 2000.0,
        }

    @staticmethod
    def calculate_shipping_policy(
        order_total_brl: float,
        city: str = "Campo Grande",
        is_large_instrument: bool = False,
    ) -> Dict[str, Any]:
        """Compute shipping cost, modalities, and delivery time based on destination city."""
        norm_city = normalize_text(city)
        is_cg_metro = any(metro in norm_city for metro in CAMPO_GRANDE_METRO_CITIES)

        if is_cg_metro:
            shipping_cost = 0.0 if order_total_brl >= 500.0 else 35.0
            return {
                "destination": city,
                "region_type": "Região Metropolitana de Campo Grande/MS",
                "free_shipping": order_total_brl >= 500.0,
                "shipping_cost_brl": shipping_cost,
                "delivery_estimate": "1 a 3 dias úteis",
                "carrier": "Motoboy próprio (contato telefônico prévio antes da entrega)",
                "insurance_included": True,
            }

        return {
            "destination": city,
            "region_type": "Outras Cidades / Nacional",
            "free_shipping": False,
            "shipping_cost_note": (
                "Calculado automaticamente com base no CEP de destino, peso e dimensões."
                + (
                    " Atenção: Instrumentos de grande porte (baterias acústicas, pianos digitais, contrabaixos) podem exigir frete especial com cotação individual."
                    if is_large_instrument
                    else ""
                )
            ),
            "modalities": [
                {
                    "name": "SEDEX (Correios)",
                    "delivery_estimate": "2 a 5 dias úteis",
                    "tracking": True,
                    "insurance": True,
                },
                {
                    "name": "Jadlog (.package)",
                    "delivery_estimate": "3 a 8 dias úteis",
                    "tracking": True,
                    "insurance": True,
                },
                {
                    "name": "PAC (Correios)",
                    "delivery_estimate": "5 a 12 dias úteis",
                    "tracking": True,
                    "insurance": True,
                },
            ],
            "insurance_included": True,
        }
