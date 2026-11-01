"""Script to execute real multi-turn agent conversations and export markdown transcripts."""

import time
from pathlib import Path
from typing import Dict, List
from src.config.settings import BASE_DIR
from src.services.chat_service import ChatService


SCENARIOS: List[Dict[str, object]] = [
    {
        "filename": "01_catalogo_violoes_ate_1000_e_frete.md",
        "title": "Cenário 1: Catálogo de Violões até R$ 1.000, Comparação Técnica, Parcelamento e Endereço",
        "description": (
            "Cobre as sugestões do desafio ('Quais opções de violões disponíveis custando até R$1000?' e "
            "'Qual o endereço da loja?'), além de filtragem entre cordas de nylon/aço, cálculo de "
            "parcelamento respeitando parcela mínima e frete grátis para Campo Grande/MS."
        ),
        "session_id": "example_01",
        "turns": [
            "Olá! Quais opções de violões disponíveis custando até R$1000?",
            "Qual a diferença entre o Tagima Dallas Tuner Aço e o Yamaha C40 Nylon? Se eu levar o Tagima Dallas Tuner, em quantas vezes sem juros consigo parcelar no cartão e quanto fica no PIX?",
            "Qual o endereço e o horário de funcionamento da loja para eu testar pessoalmente? E se eu pedir para entregar aqui em Campo Grande, quanto custa o frete?",
        ],
    },
    {
        "filename": "02_precos_promocoes_expiradas_e_estoque.md",
        "title": "Cenário 2: Consulta de Preço (Takamine GD20), Promoção Expirada, Cumulatividade PIX e Ruptura de Estoque",
        "description": (
            "Cobre a sugestão do desafio ('Quanto custa o Takamine GD20?') e todas as situações especiais "
            "da Seção 6.2 e 7.3 do manual: promoção expirada (`is_active=0`), promoção ativa (`Crafter HT-100`) "
            "não cumulativa com PIX, produto sem estoque (`Giannini GF-3D`), descontinuado (`Shelby SN-7C`) e lançamento (`Kala Barítono Cedar`)."
        ),
        "session_id": "example_02",
        "turns": [
            "Oi! Quanto custa o Takamine GD20? Eu vi que ele tinha uma promoção de 12% de Oferta Especial, ela ainda está valendo?",
            "E o violão Crafter HT-100 Folk está em promoção ativa? Se eu pagar ele no PIX, ganho mais 5% de desconto em cima da promoção?",
            "Vocês têm o violão Giannini GF-3D Dreadnought Sunburst, o Shelby SN-7C 7 Cordas ou o ukulele Kala KA-ABP-CTG Barítono Cedar disponíveis para pronta entrega hoje?",
        ],
    },
    {
        "filename": "03_politicas_troca_arrependimento_e_garantia.md",
        "title": "Cenário 3: Direito de Arrependimento, Itens Não Elegíveis (Boquilha) e Garantia Legal/Fabricante",
        "description": (
            "Cobre a sugestão do desafio ('Me arrependi da minha compra, posso devolver meu pedido?') e "
            "aprofunda nas regras das Seções 4, 8 e 9: frete de devolução por conta da loja, vedação de troca "
            "de boquilhas por higiene, defeito após 45 dias (intermediação com fabricante e garantia legal de 90 dias) e LGPD."
        ),
        "session_id": "example_03",
        "turns": [
            "Me arrependi da minha compra online que chegou há 4 dias, posso devolver meu pedido? Quem paga o frete de devolução e em quanto tempo recebo o reembolso?",
            "Aproveitando: comprei também uma boquilha de saxofone e um violão que pedi para fazerem regulagem especial sob encomenda. Posso trocar esses dois itens por outros modelos?",
            "Meu amigo comprou uma guitarra há 45 dias na loja e o captador parou de funcionar por defeito de fabricação, e os trastes estão gastos de tanto tocar. Como funciona a garantia?",
        ],
    },
    {
        "filename": "04_rastreamento_pedidos_e_desambiguacao_clientes.md",
        "title": "Cenário 4: Rastreamento de Pedidos, Desambiguação de Homônimos e Motivos de Cancelamento",
        "description": (
            "Demonstra a consulta em tempo real às tabelas `customers`, `orders` e `order_items`, incluindo "
            "desambiguação ativa quando há múltiplos clientes com o nome 'Bruno', consulta de pedido enviado "
            "com código de rastreio e explicação transparente de pedido cancelado por boleto vencido."
        ),
        "session_id": "example_04",
        "turns": [
            "Olá, meu nome é Bruno e quero saber o status do meu pedido.",
            "Sou o Bruno Carvalho Martins de Três Lagoas (bruno.martins@jmail.com). E também queria checar o pedido #8 da minha esposa Ana Carolina Ferreira.",
            "Obrigado! Meu primo Pedro Henrique Oliveira disse que o pedido #19 dele deu problema. O que aconteceu com o pedido #19 e como rastreio um código padrão da loja?",
        ],
    },
    {
        "filename": "05_escopo_acessorios_multilingue_e_baixa_confianca.md",
        "title": "Cenário 5: Guardrails de Escopo (Acessórios), Baixa Confiança/Ambiguidade e Atendimento Multilíngue (EN/ES)",
        "description": (
            "Valida a recusa educada de acessórios (cordas, palhetas, cabos, pedais e amplificadores), "
            "o interceptador de baixa confiança (`FailedIntentRepository`), a proteção contra assuntos fora de escopo "
            "e a detecção automática de idioma respondendo em Inglês e Espanhol."
        ),
        "session_id": "example_05",
        "turns": [
            "xkcdqwrty",
            "Oi! Quero comprar um jogo de cordas de aço, 3 palhetas, um cabo P10 e um amplificador Fender para minha guitarra. Vocês têm? E aproveitando, me passa uma receita de bolo de cenoura?",
            "Hello! I am an expat living in Dourados/MS. Do you have the Fender Stratocaster American Pro II in stock? Can I pay using a combination of PIX and credit card, and what are the shipping options to Dourados?",
            "¡Hola! ¿Cuál es la política de privacidad de datos (LGPD) de la tienda y cómo puedo solicitar la eliminación de mis datos personales?",
        ],
    },
]


def generate_examples() -> None:
    """Execute all 5 scenarios against the live agent and write markdown files to docs/examples/."""
    examples_dir = BASE_DIR / "docs" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    chat_service = ChatService()

    for scenario in SCENARIOS:
        out_path = examples_dir / str(scenario["filename"])
        if out_path.exists() and out_path.stat().st_size > 500:
            print(f"Skipping already generated: {out_path.name}")
            continue

        session_id = str(scenario["session_id"])
        chat_service.clear_session(session_id)

        lines: List[str] = [
            f"# {scenario['title']}\n",
            f"> **Resumo do Cenário:** {scenario['description']}\n",
            "---\n",
        ]

        turns: List[str] = scenario["turns"]  # type: ignore
        for idx, user_msg in enumerate(turns, start=1):
            print(f"[{scenario['filename']}] Executing Turn {idx}...")
            res = chat_service.process_message(user_msg, session_id=session_id)
            time.sleep(4.0)
            tools_str = (
                ", ".join(f"`{t['tool']}`" for t in res.get("tools_called", []))
                or "`none (direct / guardrail)`"
            )
            lines.append(f"### 👤 Turno {idx} — Cliente\n")
            lines.append(f"**Mensagem:**\n> {user_msg}\n")
            lines.append(f"### 🎸 Turno {idx} — Assistente Empório da Música\n")
            lines.append(
                f"- **Modelo utilizado:** `{res.get('model_used')}`\n"
                f"- **Idioma detectado:** `{res.get('language')}`\n"
                f"- **Confiança NLU:** `{res.get('confidence')}`\n"
                f"- **Ferramentas acionadas:** {tools_str}\n\n"
            )
            lines.append(f"{res['reply']}\n\n---\n")

        out_path = examples_dir / str(scenario["filename"])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_examples()

