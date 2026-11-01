# Cenário 4: Rastreamento de Pedidos, Desambiguação de Homônimos e Motivos de Cancelamento

> **Resumo do Cenário:** Demonstra a consulta em tempo real às tabelas `customers`, `orders` e `order_items`, incluindo desambiguação ativa quando há múltiplos clientes com o nome 'Bruno', consulta de pedido enviado com código de rastreio e explicação transparente de pedido cancelado por boleto vencido.

---

### 👤 Turno 1 — Cliente

**Mensagem:**
> Olá, meu nome é Bruno e quero saber o status do meu pedido.

### 🎸 Turno 1 — Assistente Empório da Música

- **Modelo utilizado:** `qwen/qwen3.8-27b`
- **Idioma detectado:** `pt`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `check_order_status`


Olá, Bruno! Tudo bem? 😊

Encontrei alguns clientes com o seu nome, então preciso confirmar qual é o seu para não consultar o pedido errado. Pode me dizer qual destas opções é a sua?

1. **Bruno Carvalho Martins** — Três Lagoas (e-mail: bru***@jmail.com)
2. **Bruno Carvalho** — Dourados (e-mail: bru***@inlook.com)
3. **Bruno Martins** — Três Lagoas (e-mail: bru***@jmail.com)

Se preferir, também pode me passar seu **número de pedido**, **código de rastreio** ou seu **telefone/WhatsApp** que eu localizo na hora! 🎸

---

### 👤 Turno 2 — Cliente

**Mensagem:**
> Sou o Bruno Carvalho Martins de Três Lagoas (bruno.martins@jmail.com). E também queria checar o pedido #8 da minha esposa Ana Carolina Ferreira.

### 🎸 Turno 2 — Assistente Empório da Música

- **Modelo utilizado:** `qwen/qwen3.8-27b`
- **Idioma detectado:** `pt`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `check_order_status`, `check_order_status`, `check_order_status`


Perfeito, Bruno! Encontrei os dois. Segue o resumo: 🎸

**📦 Seu pedido (Bruno Carvalho Martins — Três Lagoas)**
- **Pedido #10** — realizado em 25/02/2026
- **Status:** ✅ Confirmado
- **Item:** Music Man Bass 1X Electric Bass (1x)
- **Total:** R$ 8.732,00 (pagamento via PIX)
- **Rastreio:** ainda não gerado (o pedido está em preparação para envio)

**📦 Pedido da Ana Carolina (Pedido #8)**
- **Pedido #8** — realizado em 15/02/2026
- **Status:** 🚚 Enviado
- **Item:** Kala KA-C Concert Mogno (1x)
- **Total:** R$ 349,90 (pagamento no débito)
- **Código de rastreio:** BRJL5544332BR
- **Previsão de entrega:** 28/02/2026

Se quiser, posso te ajudar a acompanhar o rastreio do pedido da Ana ou tirar qualquer dúvida sobre o seu baixo. É só chamar! 😊

---

### 👤 Turno 3 — Cliente

**Mensagem:**
> Obrigado! Meu primo Pedro Henrique Oliveira disse que o pedido #19 dele deu problema. O que aconteceu com o pedido #19 e como rastreio um código padrão da loja?

### 🎸 Turno 3 — Assistente Empório da Música

- **Modelo utilizado:** `qwen/qwen3.8-27b`
- **Idioma detectado:** `pt`
- **Confiança NLU:** `0.92`
- **Ferramentas acionadas:** `check_order_status`, `consult_store_policies`


Oi, Bruno! Falei com o sistema e vou te passar o que aconteceu com o pedido do Pedro e como funciona o rastreamento. 👇

**📦 Pedido #19 — Pedro Henrique Oliveira (Campo Grande)**
- **Status:** ❌ Cancelado
- **Data do pedido:** 22/02/2026
- **Item:** Taylor 214ce Elétrico Aço Natural (1x)
- **Total:** R$ 7.499,00
- **Forma de pagamento:** Boleto
- **Motivo do cancelamento:** *"Pagamento não confirmado dentro do prazo."*

Ou seja, o boleto não foi pago dentro do prazo e o pedido foi cancelado automaticamente. Se o Pedro ainda quiser o Taylor 214ce, ele pode entrar em contato pelo WhatsApp **(67) 3321-4500** para reabrir a compra e gerar um novo boleto. 🎸

**🔍 Como rastrear um código padrão da loja**
- O código de rastreio é enviado **automaticamente por e-mail e WhatsApp** assim que o pedido é despachado.
- **Formato padrão:** `BR` + 9 caracteres alfanuméricos + `BR` (exemplo: `BR4K7M2X9P1BR`).
- O rastreamento pode ser consultado **diretamente no site dos Correios ou da Jadlog**, dependendo da transportadora usada.
- Todos os envios já incluem **seguro contra extravios e danos**. Em caso de avaria, o cliente deve recusar o recebimento e avisar a loja imediatamente.

⚠️ Só um detalhe: como o pedido #19 foi cancelado, ele **não tem código de rastreio** (nunca foi despachado).

Posso ajudar em mais alguma coisa? 😊

---
