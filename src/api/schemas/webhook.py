"""
Schemas para a integração com o WhatsApp via Twilio.

Twilio envia os dados da mensagem como form-data (application/x-www-form-urlencoded).
A resposta deve ser TwiML (XML) para que o Twilio envie a mensagem de volta ao utilizador.
"""

from pydantic import BaseModel, Field


class TwilioInbound(BaseModel):
    """
    Payload enviado pelo Twilio quando uma mensagem chega ao número WhatsApp.

    Twilio envia form-data — o FastAPI popula este modelo via Form().
    Apenas os campos essenciais são capturados; o Twilio pode enviar mais.
    """

    MessageSid: str = Field(description="ID único da mensagem no Twilio.")
    From: str = Field(description="Número de origem no formato whatsapp:+XXXXXXXXXXX")
    To: str = Field(description="Número de destino (número Twilio sandbox).")
    Body: str = Field(description="Texto da mensagem enviada pelo utilizador.")
    NumMedia: str = Field(default="0", description="Número de ficheiros multimédia.")


class TwilioOutbound(BaseModel):
    """Representa a resposta que será convertida para TwiML."""

    message: str
