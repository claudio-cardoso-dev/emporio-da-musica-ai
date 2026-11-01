# Cenário 2: Consulta de Preço (Takamine GD20), Promoção Expirada, Cumulatividade PIX e Ruptura de Estoque

> **Resumo do Cenário:** Cobre a sugestão do desafio ('Quanto custa o Takamine GD20?') e todas as situações especiais da Seção 6.2 e 7.3 do manual: promoção expirada (`is_active=0`), promoção ativa (`Crafter HT-100`) não cumulativa com PIX, produto sem estoque (`Giannini GF-3D`), descontinuado (`Shelby SN-7C`) e lançamento (`Kala Barítono Cedar`).

---

### 👤 Turno 1 — Cliente

**Mensagem:**
> Oi! Quanto custa o Takamine GD20? Eu vi que ele tinha uma promoção de 12% de Oferta Especial, ela ainda está valendo?

### 🎸 Turno 1 — Assistente Empório da Música

- **Modelo utilizado:** `qwen/qwen3.8-27b`
- **Idioma detectado:** `pt`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `get_instrument_details_and_pricing`


Oi! Tudo bem? 😊

Boa pergunta! Vou te dar a resposta com transparência:

**Takamine GD20 Dreadnought Natural**
- **Preço atual na loja:** R$ 2.199,00
- **Sobre a promoção de 12% ("Oferta Especial"):** infelizmente ela **já expirou** e não está mais valendo. 😕

**Mas tem uma boa notícia!** 🎸
Como a promoção não está mais ativa, você pode aproveitar o **desconto de 5% no PIX**:
- **Preço no PIX:** R$ 2.089,05 (economia de R$ 109,95)

**Outras formas de pagamento:**
- **Cartão de crédito:** até **12x sem juros** de R$ 183,25
- **Débito/Boleto:** R$ 2.199,00 (boleto compensa em até 3 dias úteis)
- **Pagamento combinado** (PIX + cartão) também é permitido, já que o valor é acima de R$ 2.000,00

**Frete (para Campo Grande/MS):**
- **Gratuito** 🚚
- Entrega em **1 a 3 dias úteis** via motoboy próprio (com contato telefônico prévio)
- Seguro incluso

**Disponibilidade:** está em estoque (5 unidades), então dá pra garantir o seu!

Quer que eu simule o frete para outra cidade ou te ajude com mais alguma coisa? 🎶

---

### 👤 Turno 2 — Cliente

**Mensagem:**
> E o violão Crafter HT-100 Folk está em promoção ativa? Se eu pagar ele no PIX, ganho mais 5% de desconto em cima da promoção?

### 🎸 Turno 2 — Assistente Empório da Música

- **Modelo utilizado:** `qwen/qwen3.8-27b`
- **Idioma detectado:** `pt`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `get_instrument_details_and_pricing`


Ótima escolha! O **Crafter HT-100 Folk** é um violão excelente — tampo sólido em Spruce Engelmann, projeção sonora superior e acabamento impecável. 🎸

**Sobre a promoção:**
- **Sim, ele está em promoção ativa!** 🎉
- **Promoção "Desconto Exclusivo": 18% OFF**
- **Preço original:** R$ 2.399,00
- **Preço promocional:** **R$ 1.967,18** (economia de R$ 431,82)

**Sobre o PIX + 5%:**
Aqui vou ser transparente com você: **não, os descontos não são cumulativos.** 😕
Como o violão já está com a promoção de 18% ativa, o **desconto extra de 5% no PIX não se aplica**. Ou seja, pagando no PIX, no débito ou no boleto, o preço final é o mesmo: **R$ 1.967,18**.

**Formas de pagamento:**
- **PIX / Débito / Boleto:** R$ 1.967,18 (boleto compensa em até 3 dias úteis)
- **Cartão de crédito:** até **12x sem juros** de R$ 163,93

**Frete (para Campo Grande/MS):**
- **Gratuito** 🚚
- Entrega em **1 a 3 dias úteis** via motoboy próprio (com contato telefônico prévio)
- Seguro incluso

**Disponibilidade:** está em estoque (3 unidades), então dá pra garantir o seu!

Quer que eu simule o frete para outra cidade ou te ajude com mais alguma coisa? 🎶

---

### 👤 Turno 3 — Cliente

**Mensagem:**
> Vocês têm o violão Giannini GF-3D Dreadnought Sunburst, o Shelby SN-7C 7 Cordas ou o ukulele Kala KA-ABP-CTG Barítono Cedar disponíveis para pronta entrega hoje?

### 🎸 Turno 3 — Assistente Empório da Música

- **Modelo utilizado:** `openai/gpt-oss-120b`
- **Idioma detectado:** `pt`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `get_instrument_details_and_pricing`, `get_instrument_details_and_pricing`, `get_instrument_details_and_pricing`


Oi, tudo bem? 😊 Vamos conferir a disponibilidade dos três instrumentos que você mencionou:

---

### 1. Giannini GF‑3D Dreadnought Sunburst  
- **Situação:** **temporariamente fora de estoque** (stock = 0).  
- **Motivo:** ainda não chegou a reposição.  
- **Alternativas em estoque (pronta entrega):**  

| Produto | Preço (R$) | PIX (5 % off) | Quantidade |
|---|---|---|---|
| **Shelby SGD‑195E Elétrico Aço Sunburst** | 799,00 | 759,05 | 6 |
| **Tagima Dallas Tuner Aço Natural** | 749,00 | 711,55 | 9 |
| **Tagima Woodstock Dreadnought Natural** | 899,00 | 854,05 | 11 |

Se quiser mais detalhes de algum desses modelos, é só me avisar!

---

### 2. Shelby SN‑7C 7 Cordas (Nylon Natural)  
- **Situação:** **descontinuado** – não faz mais parte do nosso catálogo e não há estoque.  
- **Alternativas em estoque (pronta entrega):**  

| Produto | Preço (R$) | PIX (5 % off) | Quantidade |
|---|---|---|---|
| **Giannini GN‑15 Nylon Cedr Natural** | 689,00 | 654,55 | 6 |
| **Yamaha C70 Nylon Natural** | 849,00 | 806,55 | 8 |
| **Yamaha C40 Nylon Natural** | 599,90 | 569,90 | 12 |

Esses violões de 6 cordas são ótimas opções para quem busca qualidade e conforto.

---

### 3. Kala KA‑ABP‑CTG Barítono Cedar (Ukulele)  
- **Situação:** **coming soon** – o modelo ainda está em fase de lançamento e não pode ser enviado hoje.  
- **Alternativas em estoque (pronta entrega):**  

| Produto | Preço (R$) | PIX (5 % off) | Quantidade |
|---|---|---|---|
| **Kala KA‑SRT‑CTG Tenor Elétrico Cedar** | 1 599,00 | 1 519,05 | 4 |
| **Ohana BK‑35E Barítono Elétrico Natural** | 1 899,00 | 1 804,05 | 1 |
| **Ohana TK‑35G Tenor Sólido Natural** | 1 299,00 | 1 234,05 | 2 |

Esses ukuleles já estão disponíveis e podem ser enviados em até 1‑3 dias úteis (frete gratuito para Campo Grande).

---

#### Próximos passos
- **Quer reservar algum dos itens acima?**  
- **Precisa de simulação de pagamento (PIX, parcelamento, etc.)** para algum modelo específico?  
- **Deseja mais informações técnicas** (tampo, escala, etc.) de alguma alternativa?

Estou aqui para ajudar a escolher o instrumento perfeito para você! 🎶

---
