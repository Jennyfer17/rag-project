"""
Carregamento do conjunto de perguntas/respostas de referência (golden dataset)
usado para avaliar o sistema RAG.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldenExample:
    """Um par pergunta/resposta de referência para avaliação."""

    question: str
    ground_truth: str


def load_golden_dataset(path: Path) -> list[GoldenExample]:
    """Carrega o conjunto de avaliação de um ficheiro JSON (lista de objectos
    com os campos "question" e "ground_truth")."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldenExample(question=item["question"], ground_truth=item["ground_truth"])
        for item in data
    ]
