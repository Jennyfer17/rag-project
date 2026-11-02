"""
Pesquisa lexical BM25 e fusão de rankings (Reciprocal Rank Fusion).

Os embeddings encontram sinónimos e paráfrases, mas falham em termos exactos
como nomes de pragas e produtos, que é onde o BM25 é forte. Combinar os dois
rankings por posição não exige calibrar as suas pontuações, que têm escalas
diferentes.
"""

import math
import re
from collections import Counter

from .text_utils import normalize_text

_STOPWORDS = frozenset(
    normalize_text(word)
    for word in (
        "a o as os de da do das dos e em no na nos nas um uma uns umas para por com que "
        "se como mas ao aos à às é são eu tu me minha meu minhas meus aqui muito muitas "
        "muitos mais já não devo fazer faço quando qual quais tem têm"
    ).split()
)


def tokenize(text: str) -> list[str]:
    """Divide em termos normalizados, sem acentos nem palavras muito comuns."""
    return [
        term
        for term in re.findall(r"[a-z0-9]+", normalize_text(text))
        if len(term) > 1 and term not in _STOPWORDS
    ]


class BM25Index:
    """Índice BM25 em memória; adequado para colecções pequenas como esta."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        tokenized = [tokenize(document) for document in documents]
        self._term_frequencies = [Counter(terms) for terms in tokenized]
        self._lengths = [len(terms) for terms in tokenized]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        document_frequency = Counter(term for terms in tokenized for term in set(terms))
        total = len(documents)
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * len(self._term_frequencies)
        for term in tokenize(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self._term_frequencies):
                frequency = frequencies.get(term)
                if frequency:
                    length_ratio = self._lengths[index] / self._average_length
                    scores[index] += (
                        idf
                        * frequency
                        * (self._k1 + 1)
                        / (frequency + self._k1 * (1 - self._b + self._b * length_ratio))
                    )
        return scores

    def rank(self, query: str, limit: int) -> list[int]:
        """Índices dos documentos com pontuação positiva, do melhor para o pior."""
        scores = self.scores(query)
        ranked = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: scores[index],
            reverse=True,
        )
        return ranked[:limit]


def reciprocal_rank_fusion(*rankings: list[int], k: int = 60) -> list[int]:
    """Funde rankings: cada documento soma 1/(k + posição) em cada um deles."""
    scores: Counter[int] = Counter()
    for ranking in rankings:
        for position, document in enumerate(ranking, start=1):
            scores[document] += 1 / (k + position)
    return [document for document, _ in scores.most_common()]
