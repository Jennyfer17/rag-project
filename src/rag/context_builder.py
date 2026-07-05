"""
Construtor do contexto aumentado para o prompt do LLM.

Combina os documentos recuperados da base vectorial com os dados edafológicos
da base relacional, produzindo um bloco de contexto estruturado em Português.
"""

from .models import RetrievedDocument, SoilProperty
from .retriever import SoilRetriever


class ContextBuilder:
    """
    Monta o contexto que será inserido no prompt do LLM.

    O contexto é dividido em duas secções:
      1. Dados do Solo — propriedades edafológicas do local de estudo
      2. Conhecimento Agrícola — chunks de manuais recuperados por similaridade
    """

    _MIN_RELEVANCE_SCORE = 0.3

    def build(
        self,
        vector_docs: list[RetrievedDocument],
        soil_properties: list[SoilProperty],
    ) -> str:
        """
        Constrói a string de contexto a partir dos documentos e dados de solo.

        Documentos com relevância abaixo de _MIN_RELEVANCE_SCORE são filtrados
        para evitar ruído no contexto.
        """
        parts: list[str] = []

        if soil_properties:
            parts.append(self._format_soil_section(soil_properties))

        filtered_docs = [
            doc for doc in vector_docs
            if doc.relevance_score >= self._MIN_RELEVANCE_SCORE
        ]
        if filtered_docs:
            parts.append(self._format_documents_section(filtered_docs))

        if not parts:
            return "Não foi encontrado contexto relevante para esta pergunta."

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Métodos privados de formatação
    # ------------------------------------------------------------------

    def _format_soil_section(self, properties: list[SoilProperty]) -> str:
        location = properties[0].location if properties else "Local"
        lines = [f"=== ANÁLISE DO SOLO — {location.upper()} ==="]

        for prop in properties:
            label = SoilRetriever._PROPERTY_LABELS.get(
                prop.property_name, prop.property_name
            )
            unit_str = f" {prop.unit}" if prop.unit else ""
            lines.append(f"  • {label}: {prop.value}{unit_str}")

        return "\n".join(lines)

    def _format_documents_section(self, docs: list[RetrievedDocument]) -> str:
        lines = ["=== CONHECIMENTO AGRÍCOLA RELEVANTE ==="]

        for i, doc in enumerate(docs, start=1):
            relevance_pct = round(doc.relevance_score * 100, 1)
            lines.append(f"\n[Documento {i} | Fonte: {doc.source} | Relevância: {relevance_pct}%]")
            lines.append(doc.content.strip())

        return "\n".join(lines)
