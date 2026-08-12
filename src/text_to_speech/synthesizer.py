"""
Síntese de voz a partir de texto, usando o modelo Piper (local).

Usado para responder por áudio quando a pergunta do utilizador também
chegou por áudio (nota de voz do WhatsApp) — a Camada de Aplicação (API)
invoca este módulo para gerar a nota de voz de resposta, tal como o módulo
de Transcrição converte a nota de voz recebida em texto.

A síntese corre localmente via Piper (ONNX), sem depender de nenhum serviço
externo pago, mantendo a coerência com o resto do sistema. O áudio sai em
OGG/Opus — o formato que o WhatsApp espera para notas de voz — convertido a
partir do WAV nativo do Piper com PyAV (já uma dependência do módulo de
Transcrição, via faster-whisper), sem exigir um ffmpeg instalado no sistema.
"""

import io
import re
import wave
from pathlib import Path
from urllib.parse import quote

import av
import requests
from av.audio.resampler import AudioResampler
from piper.voice import PiperVoice

from .config import TextToSpeechConfig

_VOICE_PATTERN = re.compile(r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$")


class TextToSpeech:
    """Sintetiza texto em Português para uma nota de voz (OGG/Opus) usando o Piper local."""

    def __init__(self, config: TextToSpeechConfig | None = None) -> None:
        self._config = config or TextToSpeechConfig()
        model_path = self._ensure_voice_downloaded()
        self._voice = PiperVoice.load(model_path)

    def _ensure_voice_downloaded(self) -> Path:
        """
        Garante que os ficheiros da voz (.onnx e .onnx.json) existem
        localmente, descarregando-os do repositório rhasspy/piper-voices
        caso contrário.

        Não usamos `piper.download_voices.download_voice()` porque essa
        função não codifica o nome da voz no URL — nomes com caracteres
        acentuados (como "tugão") fazem o pedido HTTP rebentar com
        UnicodeEncodeError antes mesmo de chegar à rede.
        """
        match = _VOICE_PATTERN.match(self._config.voice)
        if not match:
            raise ValueError(
                f"Nome de voz inválido: {self._config.voice!r} "
                "(esperado o formato '<lingua>_<regiao>-<nome>-<qualidade>', ex: 'pt_PT-tugão-medium')"
            )

        lang_family = match.group("lang_family")
        lang_code = f"{lang_family}_{match.group('lang_region')}"
        voice_name = match.group("voice_name")
        voice_quality = match.group("voice_quality")
        voice_code = f"{lang_code}-{voice_name}-{voice_quality}"

        voices_dir = self._config.voices_dir
        voices_dir.mkdir(parents=True, exist_ok=True)
        model_path = voices_dir / f"{voice_code}.onnx"

        for extension in (".onnx", ".onnx.json"):
            dest = voices_dir / f"{voice_code}{extension}"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            url = (
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                f"{lang_family}/{lang_code}/{quote(voice_name)}/{voice_quality}/"
                f"{quote(voice_code)}{extension}?download=true"
            )
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            dest.write_bytes(response.content)

        return model_path

    def synthesize(self, text: str) -> bytes:
        """
        Sintetiza o texto dado numa nota de voz em OGG/Opus, pronta a
        enviar como anexo de áudio do WhatsApp.
        """
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        wav_buffer.seek(0)

        return self._wav_to_opus_ogg(wav_buffer)

    def _wav_to_opus_ogg(self, wav_buffer: io.BytesIO) -> bytes:
        output_buffer = io.BytesIO()

        input_container = av.open(wav_buffer, mode="r")
        output_container = av.open(output_buffer, mode="w", format="ogg")
        try:
            input_stream = input_container.streams.audio[0]
            output_stream = output_container.add_stream("libopus", rate=self._config.opus_sample_rate)
            resampler = AudioResampler(format="s16", layout="mono", rate=self._config.opus_sample_rate)

            for frame in input_container.decode(input_stream):
                for resampled_frame in resampler.resample(frame):
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)

            for packet in output_stream.encode(None):
                output_container.mux(packet)
        finally:
            output_container.close()
            input_container.close()

        return output_buffer.getvalue()
