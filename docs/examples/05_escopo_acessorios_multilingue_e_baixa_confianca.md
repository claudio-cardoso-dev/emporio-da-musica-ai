# Cenário 5: Guardrails de Escopo (Acessórios), Baixa Confiança/Ambiguidade e Atendimento Multilíngue (EN/ES)

> **Resumo do Cenário:** Valida a recusa educada de acessórios (cordas, palhetas, cabos, pedais e amplificadores), o interceptador de baixa confiança (`FailedIntentRepository`), a proteção contra assuntos fora de escopo e a detecção automática de idioma respondendo em Inglês e Espanhol.

---

### 👤 Turno 1 — Cliente

**Mensagem:**
> xkcdqwrty

### 🎸 Turno 1 — Assistente Empório da Música

- **Modelo utilizado:** `guardrail_low_confidence`
- **Idioma detectado:** `pt`
- **Confiança NLU:** `0.15`
- **Ferramentas acionadas:** `none (direct / guardrail)`


Desculpe, não consegui entender com clareza o que você precisa no momento. 🎶
Você poderia repetir usando outras palavras ou escolher uma das opções abaixo?

1️⃣ **Consultar catálogo, preços ou estoque de instrumentos** (ex.: violões, guitarras, baixos, baterias, teclados, ukuleles)
2️⃣ **Rastrear ou consultar status de um pedido** (informe o número do pedido, telefone, e-mail ou código de rastreio)
3️⃣ **Dúvidas sobre políticas da loja** (horários, endereço em Campo Grande/MS, formas de pagamento/PIX, frete, trocas ou garantia)

---

### 👤 Turno 2 — Cliente

**Mensagem:**
> Oi! Quero comprar um jogo de cordas de aço, 3 palhetas, um cabo P10 e um amplificador Fender para minha guitarra. Vocês têm? E aproveitando, me passa uma receita de bolo de cenoura?

### 🎸 Turno 2 — Assistente Empório da Música

- **Modelo utilizado:** `qwen/qwen3.8-27b`
- **Idioma detectado:** `pt`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `none (direct / guardrail)`


Oi! Tudo bem? 🎸

Aqui no **Empório da Música** trabalhamos **exclusivamente com instrumentos musicais** (violões, guitarras, baixos, baterias, teclados, ukuleles, sopros e cordas orquestrais). Por isso, **não vendemos acessórios** como cordas avulsas, palhetas, cabos ou amplificadores.

Para esses itens, sugiro procurar em lojas especializadas em acessórios musicais da sua região. Mas, se você estiver precisando de uma **guitarra** (elétrica, acústica ou eletroacústica), eu posso te ajudar a encontrar o modelo ideal no nosso catálogo!

Quanto à **receita de bolo de cenoura**, infelizmente não posso te ajudar com isso, pois meu foco é todo voltado para o mundo da música e para os serviços da nossa loja. 🎶

Posso te ajudar a buscar alguma guitarra ou outro instrumento musical?

---

### 👤 Turno 3 — Cliente

**Mensagem:**
> Hello! I am an expat living in Dourados/MS. Do you have the Fender Stratocaster American Pro II in stock? Can I pay using a combination of PIX and credit card, and what are the shipping options to Dourados?

### 🎸 Turno 3 — Assistente Empório da Música

- **Modelo utilizado:** `qwen/qwen3.8-27b`
- **Idioma detectado:** `en`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `get_instrument_details_and_pricing`


Hello! Welcome to Empório da Música! 🎸

Great news — the **Fender Stratocaster American Pro II** is **in stock** (5 units available). Here are the details:

### 🎸 Product
- **Model:** Fender Stratocaster American Pro II (Sunburst)
- **Specs:** Alder body, Maple neck, 3 V-Mod II single-coil pickups
- **Price:** **R$ 12,500.00**
- **PIX price (5% cash discount):** **R$ 11,875.00**

### 💳 Payment Options
- **Combined payment (PIX + Credit Card):** ✅ **Yes, allowed!** Since the purchase is above R$ 2,000.00, you can split the payment between PIX and credit card.
- **Credit Card Installments:** Up to **12x interest-free** (minimum installment R$ 100.00 for 7x–12x).
  - Example: 12x of **R$ 1,041.67**
  - Example: 6x of **R$ 2,083.33**
- **PIX:** 5% discount applied (R$ 11,875.00)

### 🚚 Shipping to Dourados/MS
Shipping is calculated based on your CEP, weight, and dimensions. Here are the available modalities (all with **tracking** and **insurance included**):

| Modality | Delivery Estimate | Tracking | Insurance |
|----------|-------------------|----------|-----------|
| **SEDEX (Correios)** | 2 to 5 business days | ✅ | ✅ |
| **Jadlog** | 3 to 8 business days | ✅ | ✅ |
| **PAC (Correios)** | 5 to 12 business days | ✅ | ✅ |

> 💡 **Tip:** If you'd like, I can help you simulate the exact shipping cost for your specific CEP in Dourados. Just let me know!

Would you like to proceed with the purchase, or do you have any other questions? 🎶

---

### 👤 Turno 4 — Cliente

**Mensagem:**
> ¡Hola! ¿Cuál es la política de privacidad de datos (LGPD) de la tienda y cómo puedo solicitar la eliminación de mis datos personales?

### 🎸 Turno 4 — Assistente Empório da Música

- **Modelo utilizado:** `qwen/qwen3.8-27b`
- **Idioma detectado:** `es`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `consult_store_policies`


¡Hola! Con gusto te ayudo con eso. 🎶

### 🔒 Política de Privacidad y Protección de Datos (LGPD)

La **Empório da Música** está en plena conformidad con la **Ley General de Protección de Datos (LGPD — Ley nº 13.709/2018)**.

**¿Para qué usamos tus datos personales?**
Los datos que recopilamos durante el atendimento (nombre, teléfono, correo electrónico y dirección) se utilizan **exclusivamente** para:
- Procesar y entregar tus pedidos.
- Comunicarte sobre el estado de tus pedidos y el seguimiento (rastreo).
- Enviarte promociones y novedades (solo con tu **consentimiento explícito**).
- Cumplir con obligaciones legales y fiscales.

**¿Compartimos tus datos con terceros?**
No. **No compartimos** los datos de nuestros clientes con terceros con fines de marketing.

### 🗑️ ¿Cómo solicitar la eliminación de mis datos?

Como titular de los datos, puedes solicitar la **exclusión de tus datos personales en cualquier momento**. Para hacerlo, simplemente debes ponerte en contacto con nosotros a través de:

- **WhatsApp:** (67) 3321-4500
- **Correo electrónico:** contato@emporiodamusica.com.br

Basta con enviar una solicitud indicando que deseas la eliminación de tus datos y nuestro equipo se encargará de procesarla.

¿Hay algo más en lo que pueda ayudarte, como buscar algún instrumento musical o consultar un pedido? 🎸

---
