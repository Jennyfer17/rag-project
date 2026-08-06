"""
Descrição de imagens usando o modelo de visão moondream2.

Usado quando o utilizador envia uma imagem (ex: foto de uma folha doente
no WhatsApp) em vez de texto — a Camada de Aplicação (API) invoca este
módulo para obter uma descrição textual da imagem antes de a encaminhar
ao pipeline RAG, tal como o módulo de Transcrição faz para áudio.

O modelo corre localmente via transformers (trust_remote_code, como
indicado pela própria vikhyatk/moondream2), sem depender de nenhum
serviço externo pago, mantendo a coerência com o resto do sistema.
"""

import io

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ImageDescriptionConfig


class ImageDescriber:
    """Descreve imagens em texto usando o modelo de visão moondream2 local."""

    def __init__(self, config: ImageDescriptionConfig | None = None) -> None:
        self._config = config or ImageDescriptionConfig()
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.model_name,
            revision=self._config.revision,
            trust_remote_code=True,
        ).to(self._config.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_name,
            revision=self._config.revision,
        )

    def describe(self, image_bytes: bytes) -> str:
        """
        Descreve os bytes de uma imagem (ex: jpg/png recebido via WhatsApp) em texto.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        encoded_image = self._model.encode_image(image)
        answer = self._model.answer_question(encoded_image, self._config.prompt, self._tokenizer)
        return answer.strip()
