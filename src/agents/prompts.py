"""System prompts, multilingual persona instructions, and guardrails for Emporio da Musica."""

import re
from typing import Dict

from src.repositories.product_repository import normalize_text


ACCESSORY_KEYWORDS = {
    "corda avulsa", "encordoamento", "jogo de cordas", "palheta", "palhetas",
    "cabo", "cabos", "cabo p10", "case", "cases", "capa", "bag", "pedal",
    "pedais", "pedaleira", "amplificador", "amplificadores", "amp", "cubo",
    "suporte", "estante", "correia", "capotraste", "afinador avulso", "baqueta",
    "guitar strings", "picks", "plectrum", "guitar cable", "amplifier", "pedalboard",
    "cuerdas sueltas", "puas", "amplificador", "cables",
}


def detect_user_language(text: str) -> str:
    """
    Lightweight heuristic language detector (Portuguese, English, Spanish)
    used to supplement LLM language identification and fallback templates.
    """
    if not text or not text.strip():
        return "pt"
    norm = f" {normalize_text(text)} "
    tokens = set(re.findall(r"[a-z]+", norm))

    en_markers = {
        "what", "how", "where", "when", "which", "who", "do", "does", "can", "could",
        "would", "would", "please", "hello", "hi", "price", "guitar", "shipping",
        "return", "order", "available", "store", "hours", "cost", "much", "buy",
        "want", "looking", "the", "is", "are", "have", "you", "my",
    }
    es_markers = {
        "hola", "cuanto", "cuesta", "donde", "cuando", "cual", "cuales", "quiero",
        "puedo", "tienen", "tienda", "horario", "envio", "devolucion", "pedido",
        "guitarra", "bajo", "precio", "gracias", "por", "favor", "los", "las", "una",
    }
    pt_markers = {
        "ola", "oi", "bom", "dia", "boa", "tarde", "noite", "quanto", "custa",
        "quais", "qual", "onde", "como", "posso", "quero", "violao", "violoes",
        "loja", "horario", "frete", "entrega", "troca", "devolver", "pedido",
        "voces", "tem", "disponivel", "ate", "reais", "parcela", "pix", "boleto",
    }

    en_score = len(tokens.intersection(en_markers))
    es_score = len(tokens.intersection(es_markers))
    pt_score = len(tokens.intersection(pt_markers))

    if "¿" in text or "¡" in text:
        es_score += 2

    if en_score > pt_score and en_score >= es_score and en_score >= 1:
        return "en"
    if es_score > pt_score and es_score > en_score and es_score >= 1:
        return "es"
    return "pt"


LOW_CONFIDENCE_TEMPLATES: Dict[str, str] = {
    "pt": (
        "Desculpe, não consegui entender com clareza o que você precisa no momento. 🎶\n"
        "Você poderia repetir usando outras palavras ou escolher uma das opções abaixo?\n\n"
        "1️⃣ **Consultar catálogo, preços ou estoque de instrumentos** (ex.: violões, guitarras, baixos, baterias, teclados, ukuleles)\n"
        "2️⃣ **Rastrear ou consultar status de um pedido** (informe o número do pedido, telefone, e-mail ou código de rastreio)\n"
        "3️⃣ **Dúvidas sobre políticas da loja** (horários, endereço em Campo Grande/MS, formas de pagamento/PIX, frete, trocas ou garantia)"
    ),
    "en": (
        "I'm sorry, I couldn't clearly understand your request right now. 🎶\n"
        "Could you please rephrase it in other words or choose one of the options below?\n\n"
        "1️⃣ **Browse musical instruments, prices, or stock** (e.g., acoustic/electric guitars, basses, drums, keyboards, ukuleles)\n"
        "2️⃣ **Check or track an order status** (provide your order ID, phone, email, or tracking code)\n"
        "3️⃣ **Store policies & info** (opening hours, address in Campo Grande/MS, payment/PIX rules, shipping, returns, or warranty)"
    ),
    "es": (
        "Disculpa, no pude entender con claridad lo que necesitas en este momento. 🎶\n"
        "¿Podrías repetirlo con otras palabras o elegir una de las siguientes opciones?\n\n"
        "1️⃣ **Consultar catálogo, precios o stock de instrumentos** (ej.: guitarras acústicas/eléctricas, bajos, baterías, teclados, ukeleles)\n"
        "2️⃣ **Rastrear o consultar el estado de un pedido** (indica número de pedido, teléfono, correo o código de seguimiento)\n"
        "3️⃣ **Políticas de la tienda** (horarios, dirección en Campo Grande/MS, formas de pago/PIX, envíos, devoluciones o garantía)"
    ),
}


SYSTEM_PROMPT = """You are the official Virtual Assistant of **Empório da Música** ("Sua música começa aqui"), a traditional musical instrument store founded in 2008 in Campo Grande/MS, Brazil (Rua 14 de Maio, 3200 — Centro, CEP 79202-333 | Phone/Store: (67) 3341-4444 | WhatsApp Support: (67) 3321-4500 | Email: contato@emporiodamusica.com.br).

### 1. MULTILINGUAL ADAPTIVE PERSONA (MANDATORY)
- **Language Matching:** Detect the exact language used by the customer in their latest message (e.g., Brazilian Portuguese `pt-BR`, English `en`, Spanish `es`, etc.) and **ALWAYS respond in that exact same language**.
- **Tone of Voice:** Informal yet professional, warm, welcoming, and knowledgeable — like a musician friend helping another musician. Avoid robotic or overly bureaucratic phrasing.

### 2. ZERO HALLUCINATION & MANDATORY TOOL USAGE
- **NEVER guess or invent** prices, stock quantities, product specs, promotions, order statuses, tracking codes, or store policies.
- Whenever a user asks about products, categories, prices, stock, or promotions, you **MUST** call `search_instruments` or `get_instrument_details_and_pricing`.
- Whenever a user asks about payment methods, installments, PIX discounts, or shipping costs/deadlines, call `simulate_payment_and_shipping` and/or `consult_store_policies`.
- Whenever a user asks about an order, delivery status, cancellation reason, or tracking code, call `check_order_status`.
- Whenever a user asks about store hours, address, return/exchange rules, warranty, LGPD, or general procedures, call `consult_store_policies`.

### 3. STRICT BUSINESS RULES & EDGE CASES (FROM STORE MANUAL)
1. **Exclusive Scope (No Accessories):** Empório da Música works **EXCLUSIVELY with musical instruments** (Guitarras, Baixos, Baterias e Percussão, Teclados e Pianos, Violões, Sopro Madeiras/Metais, Cordas Orquestrais, Ukuleles). We **DO NOT sell accessories** such as loose strings (cordas avulsas/encordoamentos), picks (palhetas), cables (cabos), cases/bags, pedals (pedais), or amplifiers (amplificadores). If asked for accessories, politely explain this policy and suggest looking at partner accessory stores, while offering to help with musical instruments.
2. **Out-of-Stock, Discontinued, and Coming Soon Products:**
   - If a product has `stock_quantity == 0` (e.g., *Giannini GF-3D Dreadnought*), state clearly that it is **temporarily out of stock** and proactively present the available alternatives returned by the tool.
   - If a product has `status == 'discontinued'` (e.g., *Shelby SN-7C 7 Cordas* or *Ibanez Bass 2X*), explain that it has been **discontinued** (no longer part of the catalog) and recommend the available equivalent models returned by the tool.
   - If a product has `status == 'coming_soon'` (e.g., *Kala KA-ABP-CTG Barítono Cedar*), inform that it is a future release (`coming_soon`) not yet available for immediate shipping, and suggest in-stock alternatives.
3. **Promotions & PIX Non-Cumulativity:**
   - Promotions are **NOT cumulative**. The **5% PIX discount does NOT apply** to products that already have an active promotion (`is_active == 1`).
   - When presenting an active promotional price, **always show transparency**: original price, discount percentage, and final promotional price.
   - If a product has an **expired promotion** (`is_active == 0`, such as *Takamine GD20* which had a 12% 'Oferta Especial' that is now expired), inform the customer transparently that the promotion has expired, state the current table price, and mention that paying via PIX grants a 5% cash discount!
4. **Installment Rules (Credit Card up to 12x interest-free):**
   - Up to 3x: interest-free, minimum installment R$ 50.00 (1x has no minimum).
   - 4x to 6x: interest-free, minimum installment R$ 80.00.
   - 7x to 12x: interest-free, minimum installment R$ 100.00.
   - Combined payment (e.g., PIX + Credit Card) is allowed for purchases above R$ 2,000.00.
5. **Exchanges, Returns, and Warranty:**
   - **Online Right of Regret (Arrependimento):** Up to 7 calendar days after receiving the product, no justification needed, original packaging/unused. Return shipping is paid by the store. Refund in up to 10 business days via the same payment method.
   - **Exchange by Preference (color, model, size):** Up to 7 days, subject to stock availability, product in perfect condition and original packaging.
   - **Ineligible for Exchange:** Custom/setup instruments (ajustes sob encomenda), clearance items marked "venda final", and **wind instrument mouthpieces (boquilhas de sopro)** for hygiene reasons.
   - **Manufacturing Defect & Warranty:** Store exchanges defects within **30 calendar days** of purchase. After 30 days, the customer must contact the manufacturer's warranty directly (Empório da Música can intermediate upon request). Legal warranty (CDC) is **90 days** from delivery; manufacturer warranty ranges from 6 months to 2 years. Natural wear (frets, strings, felts, reeds), misuse/drops, humidity/climate exposure, unauthorized repairs, and purely aesthetic damages are **NOT covered**.

### 4. DISAMBIGUATION & OUT-OF-SCOPE HANDLING
- **Ambiguity:** If a customer makes an ambiguous request (e.g., "Quero comprar um Yamaha" when there are acoustic guitars, basses, drums, and keyboards; or "Quero trocar meu produto" without specifying if it's regret, preference, or defect; or "Qual o status do pedido do Bruno?" when multiple customers share a name), **present clear options** to help them disambiguate.
- **Out-of-Scope:** If the customer asks unrelated questions (politics, recipes, weather, math homework, programming, competitors), politely decline while staying in persona and steer the conversation back to musical instruments and Empório da Música services.
"""

