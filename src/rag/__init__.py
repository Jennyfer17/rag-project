"""
Pacote RAG — Amigo do Agricultor.

Exporta as classes e configurações necessárias para construir a API.
"""

from .config import ModelConfig, RAGConfig, RetrieverConfig
from .models import RAGResponse, RetrievedDocument, SoilProperty
from .pipeline import RAGPipeline

__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "ModelConfig",
    "RetrieverConfig",
    "RAGResponse",
    "RetrievedDocument",
    "SoilProperty",
]
