"""
Modelos de dados do sistema RAG.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievedDocument:
    """Documento recuperado da base vectorial ChromaDB."""

    content: str
    source: str
    relevance_score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class SoilProperty:
    """Propriedade edafológica recuperada da base de dados relacional."""

    property_name: str
    value: float
    unit: Optional[str]
    location: str


@dataclass
class RAGResponse:
    """Resposta completa do pipeline RAG."""

    question: str
    answer: str
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)
    soil_context: list[SoilProperty] = field(default_factory=list)
    model_name: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "model_name": self.model_name,
            "sources": [
                {
                    "source": doc.source,
                    "relevance_score": round(doc.relevance_score, 4),
                }
                for doc in self.retrieved_documents
            ],
            "soil_properties_used": len(self.soil_context),
        }
