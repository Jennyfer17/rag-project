"""
Configuração do módulo de conversão de texto para áudio.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Raiz do projecto: code/
_BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class TextToSpeechConfig:
    """Parâmetros da voz Piper usada para sintetizar as respostas em áudio."""

    voice: str = "pt_PT-tugão-medium"
    voices_dir: Path = field(default_factory=lambda: _BASE_DIR / "data" / "tts_voices")
    # Frequência de amostragem exigida pelo codec Opus (usado nas notas de voz
    # do WhatsApp) — tem de ser uma das suportadas pelo libopus (8000, 12000,
    # 16000, 24000 ou 48000 Hz); a voz Piper sintetiza a 22050 Hz, por isso
    # o áudio é reamostrado para esta frequência antes de codificar.
    opus_sample_rate: int = 48000
