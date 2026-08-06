"""
Utilitários de texto partilhados pelos módulos de classificação por
palavras-chave (SoilQueryRouter, DomainGuard).
"""

import unicodedata


def normalize_text(text: str) -> str:
    """Remove acentos e converte para minúsculas para comparação robusta."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
