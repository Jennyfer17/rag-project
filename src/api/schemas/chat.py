"""Schemas Pydantic para o endpoint de query directa."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["Quando devo fazer a sacha no milho?"],
    )
    include_soil: bool = Field(
        default=True,
        description="Inclui os dados do solo no contexto da resposta.",
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
