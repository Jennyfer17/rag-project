"""
Router do webhook WhatsApp — integração com Twilio.

Fluxo assíncrono (evita timeout do Twilio de ~15s):
  1. Twilio envia POST com a mensagem do utilizador (texto, áudio ou imagem).
  2. Respondemos imediatamente com TwiML vazio (acusa recepção).
  3. Se a mensagem for áudio, é transcrita para texto (módulo de Transcrição).
     Se for imagem, é descrita em texto (módulo de Descrição de Imagens).
  4. O RAG processa em background (pode demorar minutos).
  5. Se a pergunta chegou por áudio, a resposta é também sintetizada em
     áudio (módulo de Síntese de Voz) e servida a partir de /media/{ficheiro}
     — o Twilio precisa de um URL público para a anexar à mensagem.
  6. Enviamos a resposta via API REST do Twilio quando estiver pronta.

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

from api.dependencies import get_image_describer, get_pipeline, get_synthesizer, get_transcriber
from api.routers.media import save_reply_audio
from image_description import ImageDescriber
from rag import RAGPipeline
from text_to_speech import TextToSpeech
from transcription import AudioTranscriber

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp / Twilio"])

_WHATSAPP_MAX_LENGTH = 1600

_FALLBACK_MESSAGE = (
    "Desculpe, não consegui processar a sua pergunta de momento. "
    "Por favor tente novamente mais tarde."
)

_EMPTY_MESSAGE = (
    "Não recebi nenhuma pergunta. Escreva a sua pergunta sobre a sua "
    "machamba, ou envie uma nota de voz ou uma foto da planta."
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


def _send_whatsapp_reply(to: str, from_: str, body: str, media_url: str | None = None) -> None:
    """Envia mensagem via API REST do Twilio (usado após processamento async)."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        logger.error("TWILIO_ACCOUNT_SID ou TWILIO_AUTH_TOKEN não definidos — não é possível enviar resposta.")
        return

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        create_kwargs = {"body": body, "from_": from_, "to": to}
        if media_url:
            create_kwargs["media_url"] = [media_url]
        client.messages.create(**create_kwargs)
        logger.info("Resposta enviada via Twilio REST para %s%s", to, " (com áudio)" if media_url else "")
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
    synthesizer: TextToSpeech,
    base_url: str,
    reply_with_audio: bool,
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

    # Se a pergunta chegou por nota de voz, a resposta também deve ir por
    # nota de voz. Isto corre à parte do bloco acima: uma falha na síntese
    # não deve deitar fora uma resposta de texto que já correu bem — nesse
    # caso enviamos só o texto, em vez de cair para a mensagem de erro genérica.
    audio_reply_url = None
    if reply_with_audio:
        try:
            audio_bytes = synthesizer.synthesize(answer)
            filename = save_reply_audio(audio_bytes)
            audio_reply_url = f"{base_url.rstrip('/')}/media/{filename}"
        except Exception as exc:
            logger.error("Erro ao sintetizar a resposta em áudio para %s: %s", to, exc)

    _send_whatsapp_reply(to=to, from_=from_, body=answer, media_url=audio_reply_url)


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
    synthesizer: TextToSpeech = Depends(get_synthesizer),
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
    # Com o token definido, a assinatura é obrigatória: um pedido sem ela não
    # vem do Twilio e não pode ser aceite.
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if auth_token:
        form_data = dict(await request.form())
        if not x_twilio_signature or not _validate_twilio_signature(
            auth_token, str(request.url), form_data, x_twilio_signature
        ):
            logger.warning("Assinatura Twilio ausente ou inválida — From=%s", From)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assinatura inválida.")

    logger.info("Mensagem recebida: From=%s Body='%s' NumMedia=%s", From, Body[:100], NumMedia)

    body = Body.strip()
    has_media = NumMedia != "0" and bool(MediaUrl0)
    is_audio_message = has_media and (MediaContentType0 or "").startswith("audio")

    if not body and not has_media:
        background_tasks.add_task(_send_whatsapp_reply, to=From, from_=To, body=_EMPTY_MESSAGE)
        return Response(content=_empty_twiml(), media_type="application/xml")

    # Captura o URL público deste servidor (o mesmo que o Twilio usou para
    # chamar este webhook) para poder construir o URL do /media/{ficheiro}
    # da resposta em áudio — o Twilio precisa de um URL, não de bytes.
    base_url = str(request.base_url)

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
        synthesizer,
        base_url,
        is_audio_message,
    )

    # Responde imediatamente ao Twilio para evitar timeout
    return Response(content=_empty_twiml(), media_type="application/xml")
