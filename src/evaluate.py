"""
Script de avaliação do sistema RAG com RAGAS.

Uso (a partir da raiz do projecto "code/"):
    python src/evaluate.py
    python src/evaluate.py --dataset data/evaluation/golden_qa.json --output data/evaluation/results.csv

Requisitos:
    - Ollama a correr localmente com o modelo configurado em RAGConfig.model
    - ChromaDB e base de dados do solo já populadas (ver src/ingest.py)
    - pip install -r requirements.txt (inclui ragas e langchain-ollama)
"""

import argparse
import logging
import sys
from pathlib import Path

# Garante UTF-8 no stdout — necessário no Windows para imprimir os
# resultados (contêm caracteres acentuados em Português) sem crashar.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.dataset import load_golden_dataset
from evaluation.evaluator import RAGEvaluator
from rag import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

_BASE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET = _BASE_DIR / "data" / "evaluation" / "golden_qa.json"
_DEFAULT_OUTPUT = _BASE_DIR / "data" / "evaluation" / "results.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o RAGPipeline com RAGAS.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"A carregar dataset de avaliação: {args.dataset}")
    examples = load_golden_dataset(args.dataset)
    print(f"{len(examples)} perguntas carregadas.\n")

    print("A inicializar o RAGPipeline...")
    pipeline = RAGPipeline()

    evaluator = RAGEvaluator(pipeline)
    print("A correr a avaliação (chama o Ollama várias vezes por pergunta — pode demorar)...\n")
    result = evaluator.evaluate(examples)

    df = result.to_pandas()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")

    print("\n=== Resultado por pergunta ===")
    print(df.to_string(index=False))

    print("\n=== Médias ===")
    print(result)

    print(f"\nResultados detalhados guardados em: {args.output}")


if __name__ == "__main__":
    main()
