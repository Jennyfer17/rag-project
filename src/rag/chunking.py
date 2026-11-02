"""
Limpeza do texto extraído dos manuais e divisão em chunks.

O modelo de embeddings lê no máximo `max_seq_length` tokens por chunk e ignora
o resto em silêncio, por isso o tamanho é medido em tokens do próprio modelo
quando a função `count_tokens` é fornecida. Sem limpeza, as palavras cortadas
por hífen e os erros de OCR também prejudicam os embeddings e o BM25.
"""

import re
from collections import Counter
from collections.abc import Callable

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?:;])\s+|\n+")
_PHOTO_NUMBERS = re.compile(r"(?:\(\d+(?:-[A-Z])?\)\s*){2,}")
_TOC_LEADER = re.compile(r"\.{4,}\s*\d*")
_CHAPTER_HEADER = re.compile(r"^Capítulo \d+ \|.*$", re.MULTILINE)
# O OCR confunde "O" com "0" e "10" com "1 O"/"lO" (ex: "0 cultivo", "lO m").
_OCR_ZERO_ARTICLE = re.compile(r"(?<![\d,.])\b0\s+(?=[a-záéíóúâêôãõç]{2,})")
_OCR_TEN = re.compile(r"\b[1l]\s?O\b(?=\s*(?:cm|m|dias|a\b))")
# A brochura de pragas marca cada ponto com "3 3" no início da linha.
_OCR_BULLET = re.compile(r"^[ \t]*3\s+3\s+", re.MULTILINE)


def _remove_repeated_headers(text: str, min_repeats: int = 3) -> str:
    """Remove cabeçalhos de página: linhas em maiúsculas que se repetem no documento."""

    def key(line: str) -> str:
        return re.sub(r"[\d\W_]+", " ", line).strip()

    def is_upper_heading(line: str) -> bool:
        letters = [ch for ch in line if ch.isalpha()]
        return len(letters) >= 8 and sum(ch.isupper() for ch in letters) / len(letters) > 0.8

    lines = text.split("\n")
    counts = Counter(key(line) for line in lines if is_upper_heading(line))
    repeated = {heading for heading, count in counts.items() if count >= min_repeats}
    return "\n".join(line for line in lines if not line.strip() or key(line) not in repeated)


def clean_text(text: str) -> str:
    """Normaliza o texto dos PDFs/OCR sem alterar o conteúdo técnico."""
    text = text.replace("\x00", "").replace("\xa0", " ").replace("­", "")
    text = _CHAPTER_HEADER.sub("", text)
    text = _remove_repeated_headers(text)
    text = _OCR_BULLET.sub("- ", text)
    text = re.sub("\x83+", "- ", text)
    text = _PHOTO_NUMBERS.sub(" ", text)
    text = _TOC_LEADER.sub("\n", text)
    # Junta palavras cortadas no fim da linha ("propa-\ngação" -> "propagação").
    text = re.sub(r"(\w)-[ \t]*\n[ \t]*([a-záéíóúâêôãõç])", r"\1\2", text)
    text = _OCR_ZERO_ARTICLE.sub("o ", text)
    text = _OCR_TEN.sub("10", text)
    # Quebras de linha simples dentro de um parágrafo não são separadores
    # reais, excepto antes de um item de lista.
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n|- |• )", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_oversized(sentence: str, max_size: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Divide por palavras uma frase maior do que `max_size` (ex: uma tabela)."""
    parts: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        if current and count_tokens(" ".join([*current, word])) > max_size:
            parts.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_text(
    text: str,
    max_size: int,
    overlap: int,
    count_tokens: Callable[[str], int] = len,
) -> list[str]:
    """
    Agrupa frases em chunks de até `max_size` unidades, com sobreposição.

    A unidade é definida por `count_tokens`: caracteres por omissão, ou tokens
    se for passado o contador do modelo de embeddings.
    """
    pieces: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if count_tokens(sentence) <= max_size:
            pieces.append(sentence)
        else:
            pieces.extend(_split_oversized(sentence, max_size, count_tokens))

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and count_tokens(" ".join([*current, piece])) > max_size:
            chunks.append(" ".join(current))
            # Sobreposição: repete as últimas frases que couberem em `overlap`,
            # deixando sempre espaço para a frase seguinte.
            kept: list[str] = []
            for previous in reversed(current):
                if (
                    count_tokens(" ".join([previous, *kept])) > overlap
                    or count_tokens(" ".join([previous, *kept, piece])) > max_size
                ):
                    break
                kept.insert(0, previous)
            current = kept
        current.append(piece)
    if current:
        chunks.append(" ".join(current))

    return [chunk for chunk in chunks if len(chunk) >= 40]
