"""
Router do webhook WhatsApp — integração com Twilio.

Fluxo assíncrono (evita timeout do Twilio de ~15s):
  1. Twilio envia POST com a mensagem do utilizador (texto, áudio ou imagem).
  2. Respondemos imediatamente com TwiML vazio (acusa recepção).
  3. Se a mensagem for áudio, é transcrita para texto (módulo de Transcrição).
     Se for imagem, é descrita em texto (módulo de Descrição de Imagens).
  4. O RAG processa em background (pode demorar minutos).
  5. Enviamos a resposta via API REST do Twilio quando estiver pronta.

Variáveis de ambiente necessárias:
  TWILIO_ACCOUNT_SID  — Account SID do painel Twilio
  TWILIO_AUTH_TOKEN   — Auth Token do painel Twilio
"""

import hashlib
import hmac
import logging
import os
from xml.etree.ElementTree import Element, tostring

import requests
from fastapi import APIRouter, BackgroundTasks, Depends, Form, Header, HTTPException, Request, status
from fastapi.responses import Response

from api.dependencies import get_image_describer, get_pipeline, get_transcriber
from image_description import ImageDescriber
from rag import RAGPipeline
from transcription import AudioTranscriber

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp / Twilio"])

_WHATSAPP_MAX_LENGTH = 1600

_FALLBACK_MESSAGE = (
    "Desculpe, não consegui processar a sua pergunta de momento. "
    "Por favor tente novamente mais tarde."
)

_PROCESSING_MESSAGE = (
    "A processar a sua pergunta... Receberá a resposta em breve."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_twiml() -> str:
    """TwiML vazio — acusa recepção ao Twilio sem enviar mensagem."""
    return tostring(Element("Response"), encoding="unicode")


def _truncate(text: str) -> str:
    if len(text) <= _WHATSAPP_MAX_LENGTH:
        return text
    return text[: _WHATSAPP_MAX_LENGTH - 3] + "..."


def _validate_twilio_signature(auth_token: str, url: str, params: dict, signature: str) -> bool:
    import base64
    sorted_params = "".join(f"{k}{v}" for k, v in sorted(params.items()))
    digest = hmac.new(
        auth_token.encode("utf-8"),
        (url + sorted_params).encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature)


def _download_media(url: str) -> bytes:
    """Descarrega um ficheiro de média do Twilio (protegido por auth básica)."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    response = requests.get(url, auth=(account_sid, auth_token), timeout=30)
    response.raise_for_status()
    return response.content


def _resolve_question(
    body: str,
    media_url: str | None,
    media_content_type: str | None,
    transcriber: AudioTranscriber,
    image_describer: ImageDescriber,
) -> str:
    """
    Resolve o texto da pergunta do utilizador.

    Se a mensagem tiver um anexo de imagem (ex: foto de uma planta ou praga),
    descarrega-a e obtém uma descrição textual (módulo de Descrição de
    Imagens); se vier acompanhada de legenda (Body), a legenda é combinada
    com a descrição, já que dá contexto valioso sobre o que o agricultor
    quer saber. Caso contrário, se o Body vier vazio e a mensagem tiver um
    anexo de áudio (nota de voz do WhatsApp), descarrega o ficheiro e
    transcreve-o para texto.
    """
    content_type = media_content_type or ""

    if media_url and content_type.startswith("image"):
        image_bytes = _download_media(media_url)
        description = image_describer.describe(image_bytes)
        if body:
            return f"{body}\n\n[Imagem enviada pelo utilizador: {description}]"
        return description

    if body:
        return body

    if media_url and content_type.startswith("audio"):
        audio_bytes = _download_media(media_url)
        return transcriber.transcribe(audio_bytes)

    return ""


def _send_whatsapp_reply(to: str, from_: str, body: str) -> None:
    """Envia mensagem via API REST do Twilio (usado após processamento async)."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        logger.error("TWILIO_ACCOUNT_SID ou TWILIO_AUTH_TOKEN não definidos — não é possível enviar resposta.")
        return

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        client.messages.create(body=body, from_=from_, to=to)
        logger.info("Resposta enviada via Twilio REST para %s", to)
    except Exception as exc:
        logger.error("Erro ao enviar mensagem Twilio: %s", exc)


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _process_and_reply(
    body: str,
    media_url: str | None,
    media_content_type: str | None,
    to: str,
    from_: str,
    pipeline: RAGPipeline,
    transcriber: AudioTranscriber,
    image_describer: ImageDescriber,
) -> None:
    """Corre em background: resolve a pergunta (texto, áudio ou imagem), chama o RAG e responde via Twilio REST."""
    try:
        question = _resolve_question(body, media_url, media_content_type, transcriber, image_describer)
        if not question:
            logger.warning("Não foi possível obter texto da mensagem de %s", to)
            answer = _FALLBACK_MESSAGE
        else:
            # `to` é o número WhatsApp do utilizador (destino da resposta),
            # usado como session_id para manter o contexto da conversa entre
            # perguntas de seguimento.
            rag_response = pipeline.query(question=question, session_id=to)
            answer = _truncate(rag_response.answer)
            logger.info("Resposta RAG gerada para %s (~%d palavras)", to, len(answer.split()))
    except Exception as exc:
        logger.error("Erro no processamento da mensagem: %s", exc)
        answer = _FALLBACK_MESSAGE

    _send_whatsapp_reply(to=to, from_=from_, body=answer)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/whatsapp",
    summary="Webhook WhatsApp (Twilio)",
    description=(
        "Recebe mensagens WhatsApp via Twilio. Responde imediatamente ao Twilio "
        "e processa o RAG em background para evitar timeout."
    ),
)
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    pipeline: RAGPipeline = Depends(get_pipeline),
    transcriber: AudioTranscriber = Depends(get_transcriber),
    image_describer: ImageDescriber = Depends(get_image_describer),
    MessageSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(default=""),
    NumMedia: str = Form(default="0"),
    MediaUrl0: str | None = Form(default=None),
    MediaContentType0: str | None = Form(default=None),
    x_twilio_signature: str | None = Header(default=None, alias="X-Twilio-Signature"),
) -> Response:

    # --- Validação de assinatura (activa se TWILIO_AUTH_TOKEN estiver definido) ---
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if auth_token and x_twilio_signature:
        form_data = dict(await request.form())
        if not _validate_twilio_signature(auth_token, str(request.url), form_data, x_twilio_signature):
            logger.warning("Assinatura Twilio inválida — From=%s", From)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assinatura inválida.")

    logger.info("Mensagem recebida: From=%s Body='%s' NumMedia=%s", From, Body[:100], NumMedia)

    body = Body.strip()
    has_media = NumMedia != "0" and bool(MediaUrl0)

    if not body and not has_media:
        return Response(content=_empty_twiml(), media_type="application/xml")

    # Processa a mensagem em background (transcrevendo o áudio ou descrevendo
    # a imagem, conforme o tipo de anexo) e envia a resposta via Twilio REST
    background_tasks.add_task(
        _process_and_reply,
        body,
        MediaUrl0,
        MediaContentType0,
        From,
        To,
        pipeline,
        transcriber,
        image_describer,
    )

    # Responde imediatamente ao Twilio para evitar timeout
    return Response(content=_empty_twiml(), media_type="application/xml")
