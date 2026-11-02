"""
Pipeline RAG principal — orquestra recuperação, construção de contexto e geração.

A classe RAGPipeline é o ponto de entrada único do sistema e será consumida
directamente pela API FastAPI.

Fluxo de uma query:
  1. ConversationMemory → recupera o histórico da sessão (se aplicável)
  2. QueryReformulator  → reescreve perguntas de seguimento como perguntas autónomas
  3. DomainGuard        → recusa perguntas fora do domínio agrícola
  4. SoilQueryRouter    → decide se a pergunta necessita dos dados do solo
  5. VectorRetriever    → recupera chunks similares do ChromaDB
  6. SoilRetriever      → carrega propriedades do solo da BD relacional (se aplicável)
  7. ContextBuilder     → monta o bloco de contexto aumentado
  8. LLaMAGenerator     → gera a resposta final com o LLaMA

Nota de desempenho: os passos 5 e 6 são independentes entre si (não há
dependência de dados um do outro) e correm em paralelo numa thread pool
quando ambos são necessários, para reduzir a latência total da consulta.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from .answer_cache import AnswerCache
from .config import RAGConfig
from .context_builder import ContextBuilder
from .conversation import ConversationMemory
from .domain_guard import OUT_OF_DOMAIN_MESSAGE, DomainGuard
from .generator import LLaMAGenerator
from .models import RAGResponse
from .query_reformulator import QueryReformulator, looks_self_contained
from .query_router import SoilQueryRouter
from .retriever import SoilRetriever, VectorRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orquestra o ciclo completo Retrieval-Augmented Generation.

    Uso básico:
        pipeline = RAGPipeline()                    # configuração padrão
        response = pipeline.query("Como fazer sacha no milho?")
        print(response.answer)

    Uso com configuração personalizada:
        config = RAGConfig(model=ModelConfig(temperature=0.3))
        pipeline = RAGPipeline(config)
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        self._config = config or RAGConfig()
        self._vector_retriever = VectorRetriever(self._config.retriever)
        self._soil_retriever = SoilRetriever(self._config.retriever)
        self._context_builder = ContextBuilder()
        self._generator = LLaMAGenerator(self._config.model)
        self._query_router = SoilQueryRouter()
        self._query_reformulator = QueryReformulator(self._config.model)
        self._domain_guard = DomainGuard(self._config.model)
        self._conversation_memory = ConversationMemory(
            self._config.conversation.max_turns,
            self._config.conversation.max_answer_chars,
        )
        self._answer_cache = AnswerCache(
            self._config.answer_cache.max_size,
            self._config.answer_cache.ttl_seconds,
        )
        self._retrieval_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="rag-retrieval"
        )

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        include_soil: bool | None = None,
        session_id: str | None = None,
        retrieval_query: str | None = None,
    ) -> RAGResponse:
        """
        Processa uma pergunta e retorna a resposta aumentada pelo contexto.

        Parâmetros:
            question     : pergunta do agricultor em linguagem natural
            include_soil : se True/False, força a inclusão/exclusão dos dados
                           do solo. Se None (omitido), o SoilQueryRouter decide
                           automaticamente com base em palavras-chave da pergunta.
            session_id   : identificador da conversa (ex: número de telefone no
                           WhatsApp). Se fornecido, o histórico das trocas
                           anteriores é incluído no prompt e usado para
                           reformular perguntas de seguimento antes da
                           recuperação, e esta troca é adicionada ao
                           histórico da sessão. Se None, a pergunta é
                           tratada de forma independente, sem memória.
            retrieval_query : texto a usar na recuperação em vez da pergunta (ex:
                           termos em português extraídos da descrição de uma
                           imagem). Se None, é derivado da pergunta.

        Retorna:
            RAGResponse com a resposta, documentos recuperados e dados de solo.
        """
        history = (
            self._conversation_memory.get_history(session_id) if session_id else []
        )

        # Uma resposta guardada só serve se a pergunta não depender da conversa.
        cacheable = retrieval_query is None and (not history or looks_self_contained(question))
        requested_soil = include_soil
        if cacheable:
            cached = self._answer_cache.get(question, requested_soil)
            if cached is not None:
                logger.info("Resposta servida da cache.")
                if session_id:
                    self._conversation_memory.add_turn(session_id, question, cached.answer)
                return replace(cached, question=question)

        # Perguntas de seguimento (ex: "e para o feijão?") não contêm contexto
        # suficiente por si só — reformula-se como pergunta autónoma antes de
        # a usar para recuperação e classificação. A pergunta original é
        # mantida para a geração da resposta e para o histórico da conversa.
        if retrieval_query is None:
            retrieval_query = self._query_reformulator.reformulate(question, history)

        if not self._domain_guard.is_in_domain(retrieval_query):
            answer = OUT_OF_DOMAIN_MESSAGE
            if session_id:
                self._conversation_memory.add_turn(session_id, question, answer)
            return RAGResponse(
                question=question,
                answer=answer,
                retrieved_documents=[],
                soil_context=[],
                model_name=self._generator.model_name,
            )

        if include_soil is None:
            include_soil = self._query_router.should_use_soil_data(retrieval_query)

        if include_soil:
            # Recuperação vectorial (ChromaDB) e recuperação do solo (SQLite)
            # são independentes — correm em paralelo para reduzir a latência.
            docs_future = self._retrieval_executor.submit(
                self._vector_retriever.retrieve, retrieval_query
            )
            soil_future = self._retrieval_executor.submit(self._soil_retriever.retrieve)
            vector_docs = docs_future.result()
            soil_properties = soil_future.result()
        else:
            vector_docs = self._vector_retriever.retrieve(retrieval_query)
            soil_properties = []

        context = self._context_builder.build(vector_docs, soil_properties)

        answer = self._generator.generate(
            system_prompt=self._config.system_prompt,
            context=context,
            question=question,
            history=history,
        )

        if session_id:
            self._conversation_memory.add_turn(session_id, question, answer)

        response = RAGResponse(
            question=question,
            answer=answer,
            retrieved_documents=vector_docs,
            soil_context=soil_properties,
            model_name=self._generator.model_name,
        )
        # Só guarda respostas geradas sem histórico, que valem para qualquer sessão.
        if cacheable and not history:
            self._answer_cache.put(question, requested_soil, response)
        return response

    def warm_up(self) -> None:
        """Carrega o LLM em memória; uma falha aqui não impede o arranque."""
        try:
            self._generator.warm_up()
        except Exception as exc:
            logger.warning("Não foi possível pré-carregar o modelo: %s", exc)

    def health_check(self) -> dict:
        """
        Verifica se todos os componentes estão operacionais.

        Retorna um dicionário com o estado de cada componente,
        útil para o endpoint /health da API.
        """
        status: dict = {
            "chroma_db": "ok",
            "soil_db": "ok",
            "ollama": "ok",
            "model": self._config.model.model_name,
        }

        try:
            count = self._vector_retriever.collection_count
            status["chroma_documents"] = count
        except Exception as exc:
            status["chroma_db"] = f"erro: {exc}"

        try:
            props = self._soil_retriever.retrieve()
            status["soil_properties"] = len(props)
        except Exception as exc:
            status["soil_db"] = f"erro: {exc}"

        try:
            available = self._generator.list_available_models()
            model = self._config.model.model_name
            status["model_available"] = any(model in m for m in available)
        except Exception as exc:
            status["ollama"] = f"erro: {exc}"

        return status

    # ------------------------------------------------------------------
    # Propriedades de acesso à configuração
    # ------------------------------------------------------------------

    @property
    def config(self) -> RAGConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self._config.model.model_name
