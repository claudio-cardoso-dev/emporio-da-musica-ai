# 🎸 Empório da Música — AI Customer Support & Sales Agent

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Qwen3%20%2B%20Groq%20Fallbacks-F55036)
![Architecture](https://img.shields.io/badge/Architecture-Specialist%20Multi--Agent-6366F1)
![Database](https://img.shields.io/badge/Database-SQLite%20%2B%20Hybrid%20RAG-003B57?logo=sqlite&logoColor=white)
![APIs](https://img.shields.io/badge/Integrations-ViaCEP%20%2B%20PIX%20SVG-059669)
![Pytest](https://img.shields.io/badge/Tests-93%20Passed-10B981?logo=pytest&logoColor=white)

> **Assistente Conversacional de E-commerce com IA & Arquitetura Multiagente Especializada**  
> Desenvolvido para a **Empório da Música** (*"Sua música começa aqui"* — Campo Grande/MS), fundamentado em **Clean Architecture, SOLID, Function Calling Nativo, RAG Semântico Híbrido, Motor Financeiro Determinístico e Interface Conversacional Glassmorphism**.

---

## 📋 Sumário Executivo

A **Empório da Música** é uma loja tradicional de instrumentos musicais fundada em 2008 em Campo Grande/MS que buscava otimizar e escalar seu atendimento ao cliente sem perder a proximidade, o conhecimento musical e a segurança de dados.

Este projeto entrega uma solução pronta para produção que soluciona o desafio de ponta a ponta através de uma **Arquitetura Multiagente Especializada e Híbrida**:

1. **Arquitetura Multiagente Especialista (`IntentRouter`):** Roteamento determinístico e dinâmico que encaminha cada turno do cliente para agentes dedicados com escopo restrito de ferramentas e prompts especializados:
   - **`CustomerAgent`:** Onboarding seguro (LGPD), identificação unívoca por telefone/e-mail, termos de aceite e integração com **ViaCEP** para preenchimento e validação automática de endereço.
   - **`CatalogAgent`:** Busca e recomendação de instrumentos musicais com zero alucinação, filtros de estoque/preço e renderização de cards interativos 3D.
   - **`CheckoutAgent`:** Fechamento de carrinho, cálculo determinístico de parcelas e descontos de 5% no PIX, verificação de frete (motoboy em Campo Grande ou transportadora regional) e simulação de QR Code PIX dinâmico (SVG).
   - **`SupportAgent`:** Políticas de troca/devolução (CDC), consulta de status e rastreamento de pedidos com validação estrita de titularidade e dúvidas institucionais.
2. **Zero Alucinação Numérica e Regras de Negócio:** Consultas estruturadas em banco relacional **SQLite** (alimentado por pipeline ETL idempotente dos CSVs operacionais) acopladas ao `PricingService` para garantir conformidade matemática estrita (bloqueio de desconto duplo de PIX em promoções ativas, limites de parcelamento e regras de frete).
3. **RAG Semântico Híbrido para Políticas da Loja (`políticas_da_loja.pdf`):** Extração hierárquica do PDF de políticas da loja, normalização de layouts e motor de busca combinando **BM25 + Expansão de Conceitos Multilíngues (PT/EN/ES) + Similaridade Fuzzy (`rapidfuzz`)**.
4. **Resiliência de LLM (Multi-Model Fallback Chain):** Inferência de ultra-baixa latência via **Groq API** com **Qwen (`qwen/qwen3.8-27b`) como primário**, garantindo raciocínio ágil em português brasileiro, com *fallback* automático em cascata (`openai/gpt-oss-120b` $\rightarrow$ `openai/gpt-oss-20b`) contra oscilações de conexão ou *rate limits*.
5. **Experiência Conversacional de Alto Padrão (Glassmorphism Web UI):**
   - Cards 3D interativos com rotação frente/verso (*flip card*) exibindo especificações de madeira, tampo e acabamento.
   - Seletores de quantidade (`-` / `+`) e adição direta ao carrinho no chat.
   - Formulários modais e inline integrados com consulta instantânea de CEP.
   - Comunicação humanizada: sem jargões técnicos de frontend (substituindo termos como "carrossel" por *"confira abaixo a lista de opções disponíveis..."*).

---

## 🏗️ Arquitetura do Sistema

```mermaid
flowchart TD
    User(["👤 Cliente (Web Chat UI / Swagger API / CLI)"]) --> App["FastAPI Server / main.py"]
    App --> ChatService["ChatService (Gestão de Sessão & Histórico SQLite)"]
    ChatService --> Router["IntentRouter (Classificador de Estágios e Intenções)"]
    
    Router --> |Onboarding / LGPD / CEP| Customer["CustomerAgent"]
    Router --> |Catálogo / Recomendações| Catalog["CatalogAgent"]
    Router --> |Carrinho / Checkout / PIX| Checkout["CheckoutAgent"]
    Router --> |Políticas / Rastreio / SAC| Support["SupportAgent"]
    
    Customer --> ToolCEP["CepService (ViaCEP API + Cache)"]
    Customer --> RepoCustomer["CustomerRepository"]
    
    Catalog --> ToolSearch["search_instruments"]
    Catalog --> ToolDetails["get_instrument_details_and_pricing"]
    ToolSearch & ToolDetails --> RepoProduct["ProductRepository"]
    
    Checkout --> ToolSim["simulate_payment_and_shipping"]
    Checkout --> PixService["PixService (Geração de QR Code SVG)"]
    ToolSim --> Pricing["PricingService (Regras Determinísticas)"]
    
    Support --> ToolOrder["check_order_status (Validação Estrita de Chave)"]
    Support --> ToolPolicy["consult_store_policies"]
    ToolOrder --> RepoOrder["OrderRepository"]
    ToolPolicy --> PolicyRAG["PolicyRetriever (BM25 + Fuzzy + Expansão Conceitual)"]
    
    RepoProduct & RepoCustomer & RepoOrder --> SQLite[("SQLite Database")]
    PolicyRAG --> PDF[("data/políticas_da_loja.pdf")]
    
    Router -- "Confiança Baixa / Ruído" --> FailedRepo[("failed_intents (Auditoria de Observabilidade)")]
```

---

### 📂 Estrutura de Diretórios (Clean Architecture & Modularidade)

```text
📦 emporio-da-musica-ai
 ┣ 📂 data/                          # Dados operacionais (6 CSVs + PDF de Políticas)
 ┣ 📂 docs/
 ┃ ┣ 📂 examples/                    # 5 Transcrições reais multi-turno (.md)
 ┃ ┗ 📜 generate_examples.py         # Script de geração dos cenários de teste
 ┣ 📂 src/
 ┃ ┣ 📂 agents/                      # Orquestração e Agentes Especialistas
 ┃ ┃ ┣ 📜 assistant_agent.py         # Orquestrador central e gerenciador de estado da sessão
 ┃ ┃ ┣ 📜 router.py                  # IntentRouter para classificação e transição de estágios
 ┃ ┃ ┣ 📜 specialists.py             # Agentes dedicados: Catalog, Checkout, Customer e Support
 ┃ ┃ ┣ 📜 prompts.py                 # Persona humanizada, diretrizes de segurança e regras de negócio
 ┃ ┃ ┗ 📜 tools.py                   # Definição e schemas tipados das ferramentas de Function Calling
 ┃ ┣ 📂 api/                         # Camada de Apresentação
 ┃ ┃ ┣ 📜 app.py                     # API FastAPI + Swagger (/docs) + Web Chat UI Glassmorphism
 ┃ ┃ ┗ 📜 cli.py                     # Terminal interativo com biblioteca Rich
 ┃ ┣ 📂 config/
 ┃ ┃ ┗ 📜 settings.py                # Configurações tipadas via Pydantic Settings (.env)
 ┃ ┣ 📂 db/
 ┃ ┃ ┣ 📜 database.py                # Conexão SQLite, DDL de tabelas e context managers
 ┃ ┃ ┗ 📜 ingestion.py               # Pipeline ETL idempotente com validação de dados
 ┃ ┣ 📂 rag/
 ┃ ┃ ┗ 📜 policy_retriever.py        # Parser hierárquico do PDF e motor de busca BM25 + Fuzzy
 ┃ ┣ 📂 repositories/                # Camada de Acesso a Dados (Pattern Repository)
 ┃ ┃ ┣ 📜 customer_repository.py     # Gestão segura de clientes, homônimos e atualização de endereço
 ┃ ┃ ┣ 📜 failed_intent_repository.py# Observabilidade e auditoria de intenções ambíguas
 ┃ ┃ ┣ 📜 order_repository.py        # Consulta de pedidos, itens e histórico de entrega
 ┃ ┃ ┗ 📜 product_repository.py      # Catálogo de instrumentos com join de promoções ativas/expiradas
 ┃ ┗ 📂 services/
 ┃   ┣ 📜 cep_service.py             # Integração resiliente com ViaCEP e cache offline de CEPs
 ┃   ┣ 📜 chat_service.py            # Persistência de sessões e histórico de mensagens
 ┃   ┣ 📜 pix_service.py             # Geração de payload PIX Copia-e-Cola e QR Code dinâmico em SVG
 ┃   ┗ 📜 pricing_service.py         # Cálculos de desconto PIX (5%), parcelas e frete por localidade
 ┣ 📂 tests/                         # Suíte automatizada com 93 testes (Unitários, RAG, API e E2E)
 ┣ 📜 .env.example                   # Modelo de configuração de ambiente
 ┣ 📜 main.py                        # Ponto de entrada unificado (--mode cli ou --mode api)
 ┣ 📜 pyproject.toml                 # Configuração de build e testes
 ┗ 📜 requirements.txt               # Dependências enxutas e auditadas
```

---

## 🎯 Diferenciais Técnicos e Regras de Negócio Implementadas

### 1. Segurança Cadastral e Conformidade com a LGPD
- **Autenticação por Chave Primária:** Acesso a pedidos e dados cadastrais exige confirmação estrita de telefone ou e-mail registrado.
- **Tratamento de Homônimos:** Em clientes com nomes idênticos (ex.: "Bruno"), o sistema nunca expõe dados privados de terceiros e mascara e-mails (`bru***@jmail.com`).
- **Termo de Consentimento:** Fluxo de novos clientes apresenta termo explícito de aceite antes da coleta de dados.

### 2. Integração ViaCEP com Autocompletação de Endereço
- O serviço [`CepService`](src/services/cep_service.py) consulta a API dos Correios/ViaCEP em tempo real, resolvendo logradouro, bairro, cidade e UF automaticamente.
- Inclui *fallback* local pré-carregado para municípios do Mato Grosso do Sul (ex.: Campo Grande, Dourados, Três Lagoas), garantindo disponibilidade mesmo em caso de lentidão externa.

### 3. Simulação Visual de Pagamento PIX e Status Pró-ativo
- O serviço [`PixService`](src/services/pix_service.py) gera payloads padronizados no padrão EMVCo e renderiza o QR Code diretamente em SVG no corpo da mensagem.
- Ao concluir o pedido, o assistente notifica o cliente de que a equipe da loja entrará em contato via WhatsApp para passar o status do envio assim que o pagamento for compensado bancariamente.

### 4. Transparência de Preços e Promoções Expiradas
- O catálogo avalia rigorosamente a vigência de promoções (`is_active = 1`). Caso uma promoção esteja expirada (como no caso do *Takamine GD20*), o assistente informa com total clareza que o desconto anterior expirou, informa o preço de tabela e oferece a opção de 5% de desconto à vista via PIX.
- Aplicação determinística da regra de não-cumulatividade: itens com promoção já ativa não recebem o desconto extra de PIX.

---

## 🚀 Como Executar Localmente

### Pré-requisitos
* **Python 3.11+** instalado.
* Chave de API da **Groq** (`GROQ_API_KEY`) — obtenha gratuitamente em [console.groq.com](https://console.groq.com).

### 1. Clonar o repositório e instalar dependências
```bash
git clone <URL_DO_REPOSITORIO>
cd emporio-da-musica-ai

# Criar e ativar ambiente virtual (recomendado)
python -m venv .venv

# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Linux/macOS:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente (`.env`)
```bash
cp .env.example .env
```
Edite o `.env` informando sua chave:
```env
GROQ_API_KEY=gsk_sua_chave_groq_aqui
PRIMARY_MODEL=qwen/qwen3.8-27b
FALLBACK_MODEL_1=openai/gpt-oss-120b
FALLBACK_MODEL_2=openai/gpt-oss-20b
LLM_TEMPERATURE=0.1
PORT=8000
DB_PATH=src/db/emporio_musica.db
DATA_DIR=data
CONFIDENCE_THRESHOLD=0.65
```

### 3. Executar a Aplicação

#### Modo API com Interface Web Glassmorphism (Recomendado)
```bash
python main.py --mode api --port 8000
```
- **Web Chat UI:** Abra [`http://localhost:8000`](http://localhost:8000) no seu navegador para utilizar o chat completo com cards 3D interativos e carrinho integrado.
- **Documentação Interativa Swagger:** Acesse [`http://localhost:8000/docs`](http://localhost:8000/docs).

#### Modo Terminal (Rich CLI)
```bash
python main.py --mode cli
```
- Comandos interativos no terminal:
  - `/reset` — Reinicia a sessão da conversa.
  - `/history` — Exibe o histórico de mensagens formatado.
  - `/failed` — Consulta registros de baixa confiança para auditoria.
  - `/exit` — Encerra a execução.

---

## 🧪 Suíte de Testes Automatizados (93 Testes)

A suíte de testes do projeto garante 100% de confiabilidade em todas as camadas da aplicação:
- Ingestão de banco de dados e integridade referencial dos CSVs.
- Regras financeiras e de cálculo de frete (`PricingService`).
- Cobertura exaustiva das 10 seções do PDF de políticas da loja (RAG Semântico).
- Roteamento multiagente e transições de estado (`IntentRouter`).
- Sanitização de terminologia e proibição da palavra carrossel (`clean_carousel_mentions`).
- Endpoints FastAPI, consultas ViaCEP e geração de QR Code PIX.
- Testes E2E de jornada completa do cliente.

Para executar todos os testes:
```bash
pytest -v
```

---

## 💬 Transcrições de Exemplo (`docs/examples/`)

O repositório inclui 5 transcrições de conversas multi-turno cobrindo os cenários principais e casos de borda:

1. 📄 **[`01_catalogo_violoes_ate_1000_e_frete.md`](docs/examples/01_catalogo_violoes_ate_1000_e_frete.md)** — Filtro por preço, comparativo técnico Nylon x Aço, simulação de parcelas e política de frete grátis em Campo Grande.
2. 📄 **[`02_precos_promocoes_expiradas_e_estoque.md`](docs/examples/02_precos_promocoes_expiradas_e_estoque.md)** — Transparência em promoções vencidas (Takamine GD20), não-cumulatividade do PIX e recomendação de alternativas para itens sem estoque.
3. 📄 **[`03_politicas_troca_arrependimento_e_garantia.md`](docs/examples/03_politicas_troca_arrependimento_e_garantia.md)** — Direito de arrependimento (7 dias), recusa justificada de troca para itens de higiene (boquilhas) e distinção entre desgaste natural e garantia legal.
4. 📄 **[`04_rastreamento_pedidos_e_desambiguacao_clientes.md`](docs/examples/04_rastreamento_pedidos_e_desambiguacao_clientes.md)** — Desambiguação segura de clientes homônimos, mascaramento LGPD e consulta de status de envio com código de rastreamento.
5. 📄 **[`05_escopo_acessorios_multilingue_e_baixa_confianca.md`](docs/examples/05_escopo_acessorios_multilingue_e_baixa_confianca.md)** — Interceptação de ruído com log em `failed_intents`, recusa de venda de acessórios e atendimento nativo em inglês e espanhol.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Linguagem & Runtime:** Python 3.11+
- **Framework Web & API:** FastAPI, Uvicorn, Pydantic v2
- **Provedor de IA & Modelos:** Groq API (`qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`)
- **Acesso a Dados & RAG:** SQLite 3, BM25 (`rank-bm25`), RapidFuzz, pypdf
- **Interface & Terminal:** Vanilla JS (Glassmorphism UI, CSS 3D Transforms, SVG), Rich CLI
- **Qualidade & Testes:** Pytest, AnyIO, Starlette TestClient
