"""
Router que serve os áudios gerados pelo módulo de Síntese de Voz.

O Twilio não aceita bytes de áudio directamente no pedido REST — precisa de
um URL público a partir do qual descarrega o ficheiro antes de o reenviar
como nota de voz do WhatsApp. Este router guarda esses ficheiros e serve-os.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

router = APIRouter(prefix="/media", tags=["Media"])

# Raiz do projecto: code/
_BASE_DIR = Path(__file__).resolve().parents[3]
_REPLIES_DIR = _BASE_DIR / "data" / "tts_replies"


def save_reply_audio(audio_bytes: bytes) -> str:
    """Guarda um áudio de resposta e devolve o nome do ficheiro, para construir o URL público."""
    _REPLIES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.ogg"
    (_REPLIES_DIR / filename).write_bytes(audio_bytes)
    return filename


@router.get("/{filename}", summary="Descarrega uma nota de voz de resposta gerada pelo sistema")
async def get_media(filename: str) -> FileResponse:
    # Impede path traversal (ex: filename="../../.env") — o nome resolvido
    # tem de ser idêntico ao nome pedido, sem separadores de directoria.
    if Path(filename).name != filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nome de ficheiro inválido.")

    file_path = _REPLIES_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ficheiro não encontrado.")

    return FileResponse(file_path, media_type="audio/ogg")
