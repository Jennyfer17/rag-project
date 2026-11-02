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
import sys

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ImageDescriptionConfig


def _use_float32(model) -> None:
    """
    Converte o modelo de bfloat16 para float32, para correr em CPU.

    Os CPUs sem instruções AVX-512 BF16 ou AMX emulam o bfloat16, o que torna
    os cálculos muito mais lentos. Além dos pesos, é preciso converter as
    imagens: o código remoto do moondream2 fixa o bfloat16 em `prepare_crops`.
    A revisão do modelo está fixada na configuração, por isso esta função
    interna não muda sem uma alteração explícita da revisão.
    """
    model.float()
    module = sys.modules[type(model.model).__module__]
    prepare_crops = module.prepare_crops

    def prepare_crops_float32(image, config, device):
        crops, size = prepare_crops(image, config, device)
        return crops.to(torch.float32), size

    module.prepare_crops = prepare_crops_float32


class ImageDescriber:
    """Descreve imagens em texto usando o modelo de visão moondream2 local."""

    def __init__(self, config: ImageDescriptionConfig | None = None) -> None:
        self._config = config or ImageDescriptionConfig()
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.model_name,
            revision=self._config.revision,
            trust_remote_code=True,
        ).to(self._config.device)
        if self._config.device == "cpu":
            _use_float32(self._model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_name,
            revision=self._config.revision,
        )

    def _prepare_image(self, image_bytes: bytes) -> Image.Image:
        """Corrige a orientação e reduz a foto: imagens maiores geram mais recortes."""
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        side = self._config.max_image_side
        image.thumbnail((side, side))
        return image

    def describe(self, image_bytes: bytes) -> str:
        """
        Descreve os bytes de uma imagem (ex: jpg/png recebido via WhatsApp) em texto.
        """
        image = self._prepare_image(image_bytes)
        encoded_image = self._model.encode_image(image)
        answer = self._model.answer_question(encoded_image, self._config.prompt, self._tokenizer)
        return answer.strip()
