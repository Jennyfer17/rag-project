"""
Router de chat — endpoint JSON directo para queries ao sistema RAG.

POST /query

Usado para testes directos, integração com frontends web
e qualquer cliente que consuma JSON.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_pipeline
from api.schemas.chat import QueryRequest, QueryResponse, SourceDocument
from rag import RAGPipeline

router = APIRouter(prefix="/query", tags=["Chat"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Consulta ao assistente agrícola",
    description=(
        "Envia uma pergunta ao Amigo do Agricultor e recebe uma resposta "
        "fundamentada nos manuais agrícolas e nos dados de solo da associação."
    ),
)
def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    try:
        response = pipeline.query(
            question=body.question,
            include_soil=body.include_soil,
            session_id=body.session_id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        model_name=response.model_name,
        sources=[
            SourceDocument(
                source=doc.source,
                relevance_score=round(doc.relevance_score, 4),
            )
            for doc in response.retrieved_documents
        ],
        soil_properties_used=len(response.soil_context),
    )
