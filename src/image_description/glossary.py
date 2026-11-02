"""
Glossário inglês-português para as descrições do modelo de visão.

O moondream2 responde em inglês, mesmo com o pedido em português, mas os
manuais estão em português. Em vez de gastar uma chamada extra ao LLM a
traduzir, extraímos os termos agrícolas da descrição e montamos uma pergunta em
português. Tem de ser uma frase: uma simples lista de termos recupera trechos
sobre outras culturas, e a palavra "praga" ou "doença" orienta a pesquisa para
as secções de diagnóstico e controlo.
"""

import re
from dataclasses import dataclass

_CROPS: tuple[tuple[str, str], ...] = (
    (r"cabbages?|kale|collards?|brassicas?", "couve"),
    (r"lettuces?", "alface"),
    (r"onions?", "cebola"),
    (r"tomato(?:es)?", "tomate"),
    (r"corn|maize", "milho"),
    (r"beans?", "feijão"),
    (r"potato(?:es)?", "batata"),
    (r"carrots?", "cenoura"),
    (r"peppers?", "pimento"),
    (r"cucumbers?", "pepino"),
    (r"pumpkins?|squash(?:es)?", "abóbora"),
    (r"cassava", "mandioca"),
    (r"peanuts?|groundnuts?", "amendoim"),
    (r"bananas?", "banana"),
    (r"rice", "arroz"),
    (r"garlic", "alho"),
)

_SIGNS: tuple[tuple[str, str], ...] = (
    (r"lea(?:f|ves)", "folhas"),
    (r"stems?|stalks?", "caule"),
    (r"roots?", "raízes"),
    (r"fruits?", "fruto"),
    # Buracos nas folhas costumam ser de insectos mastigadores, e é assim que os
    # manuais os descrevem; "buracos" sozinho levava às secções de doenças.
    (r"holes?|chewed|eaten", "comidas por lagartas"),
    (r"spots?|lesions?|blotch(?:es)?", "manchas"),
    (r"yellow(?:ing|ish)?", "amarelas"),
    (r"brown(?:ing|ish)?", "castanhas"),
    (r"black(?:ened)?", "negras"),
    (r"purple", "roxas"),
    (r"v[- ]shaped?", "em forma de V"),
    (r"wilt(?:ed|ing)?|droop(?:ing|y)?", "murchas"),
    (r"rot(?:ting|ten)?|decay(?:ing|ed)?", "podridão"),
    (r"molds?|moulds?|fung(?:us|i|al)", "fungo"),
    (r"webs?|webbing", "teias"),
    (r"caterpillars?|larvae?|worms?", "lagartas"),
    (r"aphids?", "afídeos"),
    (r"insects?|bugs?", "insectos"),
    (r"moths?|butterfl(?:y|ies)", "borboletas"),
    (r"eggs?", "ovos"),
)


_PLANT_PARTS = ("folhas", "caule", "raízes", "fruto")


def _find(glossary: tuple[tuple[str, str], ...], description: str) -> list[str]:
    found = (
        term
        for pattern, term in glossary
        if re.search(rf"\b(?:{pattern})\b", description, re.IGNORECASE)
    )
    return list(dict.fromkeys(found))


@dataclass
class ImageFindings:
    """Culturas e sinais em português identificados na descrição de uma imagem."""

    crops: list[str]
    signs: list[str]

    @property
    def is_agricultural(self) -> bool:
        """
        Só uma cultura ou parte de planta conta: sinais como "ovos" ou "castanhas"
        também descrevem outras fotos, que devem ir à guarda de domínio.
        """
        return bool(self.crops) or any(part in self.signs for part in _PLANT_PARTS)

    def seen(self) -> str:
        """Texto curto com o que foi identificado na foto."""
        return ", ".join(self.crops + self.signs)

    def search_query(self) -> str:
        """Pergunta para pesquisar nos manuais pelas secções de pragas e doenças."""
        crop = " e ".join(self.crops) if self.crops else "planta"
        if self.signs:
            return f"Que praga ou doença da {crop} causa {' '.join(self.signs)} e como controlar?"
        return f"Que pragas e doenças atacam a {crop} e como controlar?"


def find_in_description(description: str) -> ImageFindings:
    """Extrai as culturas e os sinais de uma descrição em inglês."""
    return ImageFindings(crops=_find(_CROPS, description), signs=_find(_SIGNS, description))
