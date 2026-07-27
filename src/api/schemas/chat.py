"""Schemas Pydantic para o endpoint de query directa."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["Quando devo fazer a sacha no milho?"],
    )
    include_soil: bool | None = Field(
        default=None,
        description=(
            "Inclui os dados do solo no contexto da resposta. "
            "Se omitido, o sistema decide automaticamente com base em "
            "palavras-chave presentes na pergunta."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Identificador da conversa, usado para manter o histórico entre "
            "perguntas de seguimento. Se omitido, a pergunta é tratada de "
            "forma independente, sem memória de conversas anteriores."
        ),
    )


class SourceDocument(BaseModel):
    source: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    model_name: str
    sources: list[SourceDocument]
    soil_properties_used: int
