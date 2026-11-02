"""
Pacote de descrição de imagens — Amigo do Agricultor.

Módulo opcional da Camada de Aplicação: converte imagens (ex: fotos de
plantas ou pragas enviadas no WhatsApp) em texto descritivo antes de
entrarem no pipeline RAG.
"""

from .config import ImageDescriptionConfig
from .describer import ImageDescriber
from .glossary import ImageFindings, find_in_description

__all__ = ["ImageDescriber", "ImageDescriptionConfig", "ImageFindings", "find_in_description"]
