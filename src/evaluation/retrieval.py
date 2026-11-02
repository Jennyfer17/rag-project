"""
Avaliação determinística da recuperação, sem LLM.

Cada pergunta tem frases de evidência retiradas dos manuais. Um chunk é
relevante se contiver alguma dessas frases, por isso a avaliação não depende
da forma como o texto foi dividido, corre em segundos e dá sempre o mesmo
resultado. Serve para comparar configurações do recuperador; a RAGAS continua
a ser usada para a validação final das respostas.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from rag.text_utils import normalize_text


@dataclass
class RetrievalExample:
    """Uma pergunta e as frases dos manuais que uma boa recuperação deve encontrar."""

    question: str
    evidence: list[str]


@dataclass
class RetrievalScore:
    """Resultado da recuperação para uma pergunta."""

    question: str
    average_precision: float
    precision: float
    reciprocal_rank: float
    fact_recall: float
    context_chars: int
    missing_evidence: list[str]


def load_retrieval_dataset(path: Path) -> list[RetrievalExample]:
    """Carrega o conjunto de avaliação (objectos com "question" e "evidence")."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RetrievalExample(question=item["question"], evidence=item["evidence"]) for item in data]


def normalize_for_match(text: str) -> str:
    """Reduz o texto a letras e números, para a comparação ignorar a pontuação."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).split())


def score_retrieval(example: RetrievalExample, contexts: list[str]) -> RetrievalScore:
    """Calcula as métricas de uma pergunta a partir dos contextos recuperados, por ordem."""
    normalized_contexts = [normalize_for_match(context) for context in contexts]
    evidence = [normalize_for_match(phrase) for phrase in example.evidence]

    relevant = [any(phrase in context for phrase in evidence) for context in normalized_contexts]
    hits = 0
    precision_sum = 0.0
    first_hit_rank = 0
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank
            first_hit_rank = first_hit_rank or rank

    missing = [
        original
        for original, phrase in zip(example.evidence, evidence)
        if not any(phrase in context for context in normalized_contexts)
    ]
    return RetrievalScore(
        question=example.question,
        average_precision=precision_sum / hits if hits else 0.0,
        precision=hits / len(contexts) if contexts else 0.0,
        reciprocal_rank=1 / first_hit_rank if first_hit_rank else 0.0,
        fact_recall=1 - len(missing) / len(evidence) if evidence else 0.0,
        context_chars=sum(len(context) for context in contexts),
        missing_evidence=missing,
    )


def summarize(scores: list[RetrievalScore]) -> dict[str, float]:
    """Médias do conjunto, mais o número de perguntas sem nenhum chunk relevante."""
    return {
        "average_precision": mean(score.average_precision for score in scores),
        "precision": mean(score.precision for score in scores),
        "reciprocal_rank": mean(score.reciprocal_rank for score in scores),
        "fact_recall": mean(score.fact_recall for score in scores),
        "context_chars": mean(score.context_chars for score in scores),
        "questions_without_hit": sum(score.average_precision == 0 for score in scores),
    }
