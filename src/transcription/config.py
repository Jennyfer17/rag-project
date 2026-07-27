"""
Configuração do módulo de transcrição de áudio.
"""

from dataclasses import dataclass


@dataclass
class TranscriptionConfig:
    """Parâmetros do modelo Whisper usado para transcrever áudio para texto."""

    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "pt"
