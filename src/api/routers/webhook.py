"""
Router do webhook WhatsApp — integração com Twilio.

Fluxo assíncrono (evita timeout do Twilio de ~15s):
  1. Twilio envia POST com a mensagem do utilizador.
  2. Respondemos imediatamente com TwiML vazio (acusa recepção).
  3. O RAG processa em background (pode demorar minutos).
  4. Enviamos a resposta via API REST do Twilio quando estiver pronta.

Variáveis de ambiente necessárias:
  TWILIO_ACCOUNT_SID  — Account SID do painel Twilio
  TWILIO_AUTH_TOKEN   — Auth Token do painel Twilio
"""

import hashlib
import hmac
import logging
import os
from xml.etree.ElementTree import Element, tostring

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Header, HTTPException, Request, status
from fastapi.responses import Response

from api.dependencies import get_pipeline
from rag import RAGPipeline

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

def _process_and_reply(question: str, to: str, from_: str, pipeline: RAGPipeline) -> None:
    """Corre em background: chama o RAG e envia a resposta via Twilio REST."""
    try:
        rag_response = pipeline.query(question=question, include_soil=True)
        answer = _truncate(rag_response.answer)
        logger.info("Resposta RAG gerada para %s (~%d palavras)", to, len(answer.split()))
    except Exception as exc:
        logger.error("Erro no RAG pipeline: %s", exc)
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
    MessageSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(...),
    NumMedia: str = Form(default="0"),
    x_twilio_signature: str | None = Header(default=None, alias="X-Twilio-Signature"),
) -> Response:

    # --- Validação de assinatura (activa se TWILIO_AUTH_TOKEN estiver definido) ---
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if auth_token and x_twilio_signature:
        form_data = dict(await request.form())
        if not _validate_twilio_signature(auth_token, str(request.url), form_data, x_twilio_signature):
            logger.warning("Assinatura Twilio inválida — From=%s", From)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assinatura inválida.")

    logger.info("Mensagem recebida: From=%s Body='%s'", From, Body[:100])

    question = Body.strip()

    if not question:
        return Response(content=_empty_twiml(), media_type="application/xml")

    # Processa o RAG em background e envia a resposta via Twilio REST
    background_tasks.add_task(_process_and_reply, question, From, To, pipeline)

    # Responde imediatamente ao Twilio para evitar timeout
    return Response(content=_empty_twiml(), media_type="application/xml")
