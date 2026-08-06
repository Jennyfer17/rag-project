"""
Injecção de dependências para a API FastAPI.

O RAGPipeline é inicializado uma única vez no arranque da aplicação
e partilhado entre todos os pedidos via FastAPI dependency injection.
Isto evita recarregar os modelos de embedding a cada chamada.
"""

from functools import lru_cache

from image_description import ImageDescriber, ImageDescriptionConfig
from rag import RAGConfig, RAGPipeline
from transcription import AudioTranscriber, TranscriptionConfig


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """
    Devolve a instância singleton do RAGPipeline.

    O decorator @lru_cache garante que o pipeline é criado apenas uma vez,
    independentemente do número de pedidos concorrentes.
    """
    config = RAGConfig()
    return RAGPipeline(config)


@lru_cache(maxsize=1)
def get_transcriber() -> AudioTranscriber:
    """
    Devolve a instância singleton do AudioTranscriber.

    Evita recarregar o modelo Whisper a cada mensagem de voz recebida.
    """
    config = TranscriptionConfig()
    return AudioTranscriber(config)


@lru_cache(maxsize=1)
def get_image_describer() -> ImageDescriber:
    """
    Devolve a instância singleton do ImageDescriber.

    Evita recarregar o modelo moondream2 a cada imagem recebida.
    """
    config = ImageDescriptionConfig()
    return ImageDescriber(config)
