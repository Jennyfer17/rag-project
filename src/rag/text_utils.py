"""
Utilitários de texto partilhados pelos módulos de classificação por
palavras-chave (SoilQueryRouter, DomainGuard).
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Remove acentos e converte para minúsculas para comparação robusta."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def compile_keywords(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """
    Cria um padrão que encontra as palavras-chave como palavras inteiras (com
    plural em "s"), para que "ola" não corresponda a "escola" nem "ph" a "alpha".
    O padrão deve ser aplicado a texto já normalizado.
    """
    alternatives = "|".join(re.escape(normalize_text(keyword)) for keyword in keywords)
    return re.compile(rf"\b(?:{alternatives})s?\b")
