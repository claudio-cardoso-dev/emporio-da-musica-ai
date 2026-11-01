"""Semantic PDF chunking and hybrid retrieval engine (BM25 + Multilingual Concepts + Fuzzy) for Store Policies."""

import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pypdf import PdfReader
from rapidfuzz import fuzz

from src.config.settings import settings
from src.repositories.product_repository import normalize_text


# Multilingual concept expansion (Portuguese, English, Spanish) mapped to section numbers
SECTION_CONCEPT_BOOSTS: Dict[int, List[str]] = {
    1: [
        "sobre", "emporio da musica", "empresa", "missao", "razao social", "cnpj",
        "endereco", "rua 14 de maio", "campo grande", "centro", "telefone", "contato",
        "email", "fundacao", "2008", "acessorios", "cordas", "palhetas", "cabos",
        "cases", "pedais", "amplificadores", "exclusivamente instrumentos",
        "address", "location", "phone", "company", "about", "mission", "accessories",
        "strings", "picks", "cables", "amplifiers", "direccion", "telefono", "ubicacion",
    ],
    2: [
        "horario", "funcionamento", "expediente", "abre", "fecha", "segunda", "sexta",
        "sabado", "domingo", "feriado", "09:00", "18:00", "13:00", "fechado",
        "datas especiais", "presencial", "atendimento",
        "hours", "opening", "schedule", "open", "closed", "saturday", "sunday", "weekday",
        "horario de atencion", "abierto", "cerrado",
    ],
    3: [
        "pagamento", "formas de pagamento", "pix", "5%", "desconto", "a vista",
        "cartao de debito", "cartao de credito", "boleto", "compensacao", "3 dias uteis",
        "parcelamento", "parcela", "sem juros", "12x", "3x", "6x", "50,00", "80,00",
        "100,00", "combinacao", "2.000", "2000", "bandeiras",
        "payment", "installments", "credit card", "debit", "discount", "pago", "cuotas", "tarjeta",
    ],
    4: [
        "troca", "trocas", "devolucao", "devolucoes", "arrependimento", "direito de arrependimento",
        "7 dias", "sete dias", "compras online", "reembolso", "10 dias uteis", "frete de devolucao",
        "defeito", "defeito de fabricacao", "30 dias", "trinta dias", "preferencia", "cor",
        "modelo", "tamanho", "nao elegiveis", "personalizacao", "setup", "regulagem",
        "venda final", "liquidacao", "boquilha", "boquilhas", "sopro", "higiene", "embalagem original",
        "return", "exchange", "refund", "regret", "defect", "mouthpiece", "hygiene", "custom",
        "devolucion", "cambio", "reembolso", "defecto", "boquilla",
    ],
    5: [
        "frete", "entrega", "entregas", "regiao metropolitana", "frete gratis", "500,00", "500",
        "taxa fixa", "35,00", "35", "motoboy", "1 a 3 dias", "outras cidades", "correios",
        "pac", "sedex", "jadlog", "5 a 12 dias", "2 a 5 dias", "3 a 8 dias", "grande porte",
        "cotacao individual", "seguro", "extravio", "avaria", "recusar recebimento",
        "rastreamento", "codigo de rastreamento", "br", "alfanumericos",
        "shipping", "delivery", "freight", "tracking", "free shipping", "carrier", "insurance",
        "damage", "envio", "flete", "seguimiento", "rastreo",
    ],
    6: [
        "promocao", "promocoes", "descontos", "campanhas", "aniversario da loja", "agosto",
        "black friday", "novembro", "volta as aulas", "fevereiro", "queima de estoque",
        "semana do musico", "dia do musico", "22 de novembro", "cumulatividade", "cumulativa",
        "nao sao cumulativas", "rain check", "reserva de preco", "esgotado", "transparencia",
        "promotion", "sale", "campaign", "cumulative", "raincheck", "promocion", "descuento",
    ],
    7: [
        "whatsapp", "3321-4500", "diretrizes", "tom de voz", "informal", "acolhido",
        "tempo de resposta", "5 minutos", "propaganda enganosa", "escopo", "lojas parceiras",
        "fluxo de atendimento", "saudacao", "fechamento", "situacoes especiais",
        "fora de estoque", "indisponivel", "descontinuado", "sucessores", "promocao expirada",
        "vencida", "reclamacao", "24 horas uteis", "empatia",
        "out of stock", "discontinued", "expired promotion", "complaint", "tone of voice",
        "reclamo", "agotado", "descontinuado",
    ],
    8: [
        "garantia", "garantia legal", "90 dias", "noventa dias", "cdc", "codigo de defesa do consumidor",
        "garantia do fabricante", "6 meses", "2 anos", "certificado de garantia",
        "nao cobre", "desgaste natural", "trastes", "feltros", "palhetas de sopro",
        "mau uso", "queda", "climaticas", "umidade", "terceiros nao autorizados", "danos esteticos",
        "warranty", "guarantee", "manufacturer", "wear and tear", "misuse", "aesthetic",
        "garantia legal", "fabricante", "desgaste",
    ],
    9: [
        "privacidade", "protecao de dados", "lgpd", "13.709", "dados pessoais",
        "consentimento", "marketing", "terceiros", "exclusao", "excluir dados",
        "privacy", "data protection", "gdpr", "delete data", "personal data", "privacidad", "datos",
    ],
    10: [
        "disposicoes finais", "documento interno", "atualizacoes periodicas", "gerencia",
        "consultar a gerencia", "versao 2.1", "junho de 2025",
        "final provisions", "internal document", "management", "gerencia",
    ],
}

STOPWORDS = {
    "a", "o", "as", "os", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "um", "uma", "uns", "umas", "e", "ou", "que", "se",
    "qual", "quais", "como", "quando", "onde", "quem", "meu", "minha", "seu", "sua",
    "the", "is", "are", "in", "on", "at", "to", "for", "of", "and", "or", "what", "how",
    "can", "i", "my", "el", "la", "los", "las", "un", "una", "y", "o", "que", "como",
}


def tokenize(text: str) -> List[str]:
    """Extract normalized alphanumeric tokens excluding common stopwords."""
    norm = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", norm)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class PolicyRetriever:
    """
    Extracts, chunks by semantic sections, and retrieves store policies from
    `políticas_da_loja.pdf` using a hybrid BM25 + Concept Expansion + Fuzzy scorer.
    """

    def __init__(self, pdf_path: Optional[Path] = None) -> None:
        self.pdf_path = pdf_path or (settings.resolved_data_dir / "políticas_da_loja.pdf")
        self.chunks: List[Dict[str, Any]] = []
        self.idf: Dict[str, float] = {}
        self.avg_dl: float = 1.0
        self._load_and_index()

    def _extract_full_text(self) -> str:
        """Read all pages from the store policies PDF and normalize word-per-line layout."""
        reader = PdfReader(str(self.pdf_path))
        pages_text = []
        for page in reader.pages:
            raw = page.extract_text() or ""
            # pypdf extracts word-per-line: collapse all whitespace sequences into a single space
            cleaned = re.sub(r"\s+", " ", raw).strip()
            cleaned = re.sub(r"Empório da Música\s+Manual de Políticas e Procedimentos", "", cleaned)
            cleaned = re.sub(r"Página \d+", "", cleaned)
            pages_text.append(cleaned.strip())
        return " ".join(pages_text)

    def _load_and_index(self) -> None:
        """Parse PDF text into major semantic sections (1 to 10) and build BM25 index."""
        full_text = self._extract_full_text()

        MAJOR_SECTION_ANCHORS = [
            (1, "1. Sobre a Empório da Música"),
            (2, "2. Horário de Funcionamento"),
            (3, "3. Formas de Pagamento"),
            (4, "4. Política de Trocas e Devoluções"),
            (5, "5. Política de Frete e Entregas"),
            (6, "6. Promoções e Descontos"),
            (7, "7. Atendimento via WhatsApp"),
            (8, "8. Garantia"),
            (9, "9. Privacidade e Proteção de Dados"),
            (10, "10. Disposições Finais"),
        ]

        positions = []
        for sec_num, title in MAJOR_SECTION_ANCHORS:
            idx = full_text.find(title)
            if idx != -1:
                positions.append((sec_num, title, idx))

        # Include cover page info (Rua 14 de Maio, 3200, contact, version) in Section 1
        raw_sections: List[Dict[str, Any]] = []
        for i, (sec_num, sec_title, start_pos) in enumerate(positions):
            actual_start = 0 if i == 0 else start_pos
            end_pos = positions[i + 1][2] if i + 1 < len(positions) else len(full_text)
            sec_body = full_text[actual_start:end_pos].strip()

            concepts = SECTION_CONCEPT_BOOSTS.get(sec_num, [])
            tokens = tokenize(f"{sec_title} {sec_body} {' '.join(concepts)}")

            raw_sections.append(
                {
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "content": sec_body,
                    "concepts": [normalize_text(c) for c in concepts],
                    "tokens": tokens,
                }
            )

        self.chunks = raw_sections
        self._compute_bm25_stats()

    def _compute_bm25_stats(self) -> None:
        """Compute IDF and average document length across all policy sections."""
        n_docs = len(self.chunks)
        if n_docs == 0:
            return
        total_len = sum(len(c["tokens"]) for c in self.chunks)
        self.avg_dl = total_len / n_docs if total_len > 0 else 1.0

        df: Dict[str, int] = {}
        for chunk in self.chunks:
            for token in set(chunk["tokens"]):
                df[token] = df.get(token, 0) + 1

        for token, doc_freq in df.items():
            self.idf[token] = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most relevant policy sections for a user query.
        Supports Portuguese, English, and Spanish queries via concept expansion.
        """
        if not query or not query.strip():
            return []

        norm_query = normalize_text(query)
        query_tokens = tokenize(query)

        k1 = 1.5
        b = 0.75
        scored_chunks: List[Dict[str, Any]] = []

        for chunk in self.chunks:
            doc_tokens = chunk["tokens"]
            doc_len = len(doc_tokens)
            tf_map: Dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            # 1. BM25 lexical score
            bm25_score = 0.0
            for qt in query_tokens:
                # Exact or prefix stem match
                matched_tf = tf_map.get(qt, 0)
                if matched_tf == 0 and len(qt) >= 4:
                    prefix = qt[:5]
                    for dt, count in tf_map.items():
                        if dt.startswith(prefix):
                            matched_tf += count * 0.8
                if matched_tf > 0:
                    idf_val = self.idf.get(qt, 1.2)
                    numerator = matched_tf * (k1 + 1.0)
                    denominator = matched_tf + k1 * (1.0 - b + b * (doc_len / self.avg_dl))
                    bm25_score += idf_val * (numerator / denominator)

            # 2. Concept phrase boost
            concept_boost = 0.0
            for concept in chunk["concepts"]:
                if concept in norm_query:
                    concept_boost += 3.5 if " " in concept else 1.8
                elif len(concept) >= 5 and any(
                    fuzz.ratio(concept, qt) >= 85 for qt in query_tokens
                ):
                    concept_boost += 1.4

            # 3. Title similarity boost
            title_norm = normalize_text(chunk["section_title"])
            title_boost = 2.5 if any(qt in title_norm for qt in query_tokens) else 0.0

            total_score = round(bm25_score + concept_boost + title_boost, 3)

            if total_score > 0:
                scored_chunks.append(
                    {
                        "section_number": chunk["section_number"],
                        "section_title": chunk["section_title"],
                        "content": chunk["content"],
                        "relevance_score": total_score,
                    }
                )

        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_chunks[:top_k]

    def get_section_by_number(self, section_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific section directly by its section number (1 to 10)."""
        for chunk in self.chunks:
            if chunk["section_number"] == section_number:
                return {
                    "section_number": chunk["section_number"],
                    "section_title": chunk["section_title"],
                    "content": chunk["content"],
                }
        return None
