"""
Router de saúde — verifica o estado de todos os componentes do sistema.

GET /health
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_pipeline
from rag import RAGPipeline

router = APIRouter(prefix="/health", tags=["Saúde"])


@router.get(
    "",
    summary="Estado do sistema",
    description=(
        "Verifica a conectividade com o ChromaDB, a base de dados de solo "
        "e o servidor Ollama. Retorna o estado de cada componente."
    ),
)
def health_check(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    return pipeline.health_check()
