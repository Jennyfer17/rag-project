"""
Injecção de dependências para a API FastAPI.

O RAGPipeline é inicializado uma única vez no arranque da aplicação
e partilhado entre todos os pedidos via FastAPI dependency injection.
Isto evita recarregar os modelos de embedding a cada chamada.
"""

from functools import lru_cache

from rag import RAGConfig, RAGPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """
    Devolve a instância singleton do RAGPipeline.

    O decorator @lru_cache garante que o pipeline é criado apenas uma vez,
    independentemente do número de pedidos concorrentes.
    """
    config = RAGConfig()
    return RAGPipeline(config)
