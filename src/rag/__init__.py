"""
Pacote RAG — Amigo do Agricultor.

Exporta as classes e configurações necessárias para construir a API.
"""

from .config import ConversationConfig, ModelConfig, RAGConfig, RetrieverConfig
from .conversation import ConversationMemory
from .domain_guard import DomainGuard
from .models import RAGResponse, RetrievedDocument, SoilProperty
from .pipeline import RAGPipeline
from .query_reformulator import QueryReformulator
from .query_router import SoilQueryRouter

__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "ModelConfig",
    "RetrieverConfig",
    "ConversationConfig",
    "ConversationMemory",
    "DomainGuard",
    "RAGResponse",
    "RetrievedDocument",
    "SoilProperty",
    "QueryReformulator",
    "SoilQueryRouter",
]
