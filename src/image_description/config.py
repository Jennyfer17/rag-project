"""
Configuração do módulo de descrição de imagens.
"""

from dataclasses import dataclass

_DEFAULT_PROMPT = (
    "Descreve esta imagem em português, com foco em aspectos relevantes "
    "para a agricultura: plantas, folhas, frutos, pragas, doenças, solo ou "
    "equipamento visíveis. Sê específico e objectivo."
)


@dataclass
class ImageDescriptionConfig:
    """Parâmetros do modelo de visão (moondream2) usado para descrever imagens em texto."""

    model_name: str = "vikhyatk/moondream2"
    revision: str = "2025-06-21"
    device: str = "cpu"
    prompt: str = _DEFAULT_PROMPT
