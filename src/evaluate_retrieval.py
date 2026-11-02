"""
Avalia só a recuperação, sem LLM: corre em segundos e é determinística.

Uso:
    python src/evaluate_retrieval.py
    python src/evaluate_retrieval.py --show-misses
    python src/evaluate_retrieval.py --collection agriculture_manuals
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.retrieval import load_retrieval_dataset, score_retrieval, summarize
from rag.config import RetrieverConfig
from rag.retriever import VectorRetriever

_DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "retrieval_benchmark.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia a recuperação sem usar o LLM.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--collection", help="Colecção a avaliar; por omissão, a da configuração.")
    parser.add_argument("--show-misses", action="store_true", help="Lista os factos não encontrados.")
    args = parser.parse_args()

    config = RetrieverConfig()
    if args.collection:
        config.collection_name = args.collection
    retriever = VectorRetriever(config)

    scores = []
    for example in load_retrieval_dataset(args.dataset):
        documents = retriever.retrieve(example.question)
        score = score_retrieval(example, [document.content for document in documents])
        scores.append(score)
        if args.show_misses and score.missing_evidence:
            print(f"- {score.question}\n    em falta: {score.missing_evidence}")

    print(f"\nColecção: {config.collection_name} | Perguntas: {len(scores)}")
    for name, value in summarize(scores).items():
        print(f"  {name:24} {value:.3f}")


if __name__ == "__main__":
    main()
