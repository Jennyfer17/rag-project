"""
Pacote de transcrição de áudio — Amigo do Agricultor.

Módulo opcional da Camada de Aplicação: converte mensagens de voz
(ex: notas de voz do WhatsApp) em texto antes de entrarem no pipeline RAG.
"""

from .config import TranscriptionConfig
from .transcriber import AudioTranscriber

__all__ = ["AudioTranscriber", "TranscriptionConfig"]
