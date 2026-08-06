"""
Avaliador do sistema RAG usando RAGAS.

Corre o RAGPipeline sobre um conjunto de perguntas de referência, recolhe a
resposta e os documentos recuperados para cada uma, e calcula métricas RAGAS
— usando o próprio LLaMA (via Ollama) como "juiz" e o mesmo modelo de
embeddings do VectorRetriever, para manter a avaliação totalmente local
(sem chaves de API externas).

Métricas calculadas:
  - Faithfulness                      : a resposta é fundamentada apenas no contexto recuperado?
  - ResponseRelevancy                 : a resposta é relevante face à pergunta feita?
  - LLMContextPrecisionWithReference  : os documentos recuperados relevantes aparecem primeiro?
  - LLMContextRecall                  : o contexto recuperado cobre a resposta de referência?
  - FactualCorrectness                : a resposta está factualmente alinhada com a referência?
"""

import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.evaluation import EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from rag import RAGPipeline

from .dataset import GoldenExample

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Avalia o RAGPipeline com RAGAS, usando Ollama e embeddings locais como juízes."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

        model_config = pipeline.config.model
        judge_llm = LangchainLLMWrapper(
            ChatOllama(
                model=model_config.model_name,
                base_url=model_config.ollama_base_url,
                temperature=0.0,
            )
        )
        judge_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=pipeline.config.retriever.embedding_model,
                model_kwargs={"device": pipeline.config.retriever.embedding_device},
            )
        )

        self._metrics = [
            Faithfulness(llm=judge_llm),
            ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
            LLMContextPrecisionWithReference(llm=judge_llm),
            LLMContextRecall(llm=judge_llm),
            FactualCorrectness(llm=judge_llm),
        ]

    def _run_pipeline(self, examples: list[GoldenExample]) -> EvaluationDataset:
        """Corre o RAGPipeline para cada pergunta e monta o dataset para a RAGAS."""
        samples = []
        for i, example in enumerate(examples, start=1):
            logger.info(
                "A processar pergunta %d/%d: %s", i, len(examples), example.question
            )
            # session_id=None: cada pergunta do conjunto de avaliação é
            # tratada de forma independente, sem histórico de conversa.
            response = self._pipeline.query(example.question, session_id=None)
            samples.append(
                SingleTurnSample(
                    user_input=example.question,
                    response=response.answer,
                    retrieved_contexts=[
                        doc.content for doc in response.retrieved_documents
                    ],
                    reference=example.ground_truth,
                )
            )
        return EvaluationDataset(samples=samples)

    def evaluate(self, examples: list[GoldenExample]) -> EvaluationResult:
        """Corre a avaliação completa e devolve o resultado da RAGAS."""
        dataset = self._run_pipeline(examples)
        # Neste ambiente cada chamada ao LLM (juiz) demora vários minutos
        # (hardware local sem aceleração dedicada) — o timeout por omissão
        # da ragas (180s) corta as métricas a meio. max_workers=1 evita
        # disparar chamadas concorrentes contra o único servidor Ollama
        # local, que só as serializaria e multiplicaria a espera de cada uma.
        run_config = RunConfig(timeout=1200, max_workers=1)
        return evaluate(dataset=dataset, metrics=self._metrics, run_config=run_config)
