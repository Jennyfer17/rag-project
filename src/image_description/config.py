"""
Configuração do módulo de descrição de imagens.
"""

from dataclasses import dataclass

# Em inglês: o moondream2 responde nessa língua, mesmo com o pedido em português.
_DEFAULT_PROMPT = (
    "Describe the plant and its problem: crop, affected part, colours, spots, "
    "holes, insects. Two short sentences."
)


@dataclass
class ImageDescriptionConfig:
    """Parâmetros do modelo de visão (moondream2) usado para descrever imagens em texto."""

    model_name: str = "vikhyatk/moondream2"
    revision: str = "2025-06-21"
    device: str = "cpu"
    prompt: str = _DEFAULT_PROMPT
    # Lado maior da imagem em píxeis: fotos maiores geram mais recortes para codificar.
    max_image_side: int = 640
