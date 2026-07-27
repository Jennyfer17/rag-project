"""
Transcritor de áudio para texto.

Usado quando o utilizador envia uma mensagem de voz (ex: nota de voz do
WhatsApp) em vez de texto — a Camada de Aplicação (API) invoca este módulo
para obter o texto equivalente antes de o encaminhar ao pipeline RAG.

A transcrição corre localmente via faster-whisper, sem depender de nenhum
serviço externo pago, mantendo a coerência com o resto do sistema (Ollama,
ChromaDB e SQLite também correm localmente).
"""

import io

from faster_whisper import WhisperModel

from .config import TranscriptionConfig


class AudioTranscriber:
    """Transcreve ficheiros de áudio em Português para texto usando Whisper local."""

    def __init__(self, config: TranscriptionConfig | None = None) -> None:
        self._config = config or TranscriptionConfig()
        self._model = WhisperModel(
            self._config.model_size,
            device=self._config.device,
            compute_type=self._config.compute_type,
        )

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcreve os bytes de um ficheiro de áudio para texto.

        Aceita qualquer formato suportado internamente por ffmpeg/pyav
        (ogg/opus, mp3, wav, ...), incluindo as notas de voz do WhatsApp.
        """
        segments, _ = self._model.transcribe(
            io.BytesIO(audio_bytes),
            language=self._config.language,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
