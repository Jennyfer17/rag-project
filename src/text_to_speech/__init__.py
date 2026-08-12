"""
Pacote de síntese de voz — Amigo do Agricultor.

Módulo opcional da Camada de Aplicação: converte a resposta do RAG em
áudio, para responder por nota de voz quando a pergunta também chegou
por nota de voz.
"""

from .config import TextToSpeechConfig
from .synthesizer import TextToSpeech

__all__ = ["TextToSpeech", "TextToSpeechConfig"]
