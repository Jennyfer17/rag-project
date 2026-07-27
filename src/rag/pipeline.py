"""
Pipeline RAG principal — orquestra recuperação, construção de contexto e geração.

A classe RAGPipeline é o ponto de entrada único do sistema e será consumida
directamente pela API FastAPI.

Fluxo de uma query:
  1. ConversationMemory → recupera o histórico da sessão (se aplicável)
  2. QueryReformulator  → reescreve perguntas de seguimento como perguntas autónomas
  3. SoilQueryRouter    → decide se a pergunta necessita dos dados do solo
  4. VectorRetriever    → recupera chunks similares do ChromaDB
  5. SoilRetriever      → carrega propriedades do solo da BD relacional (se aplicável)
  6. ContextBuilder     → monta o bloco de contexto aumentado
  7. LLaMAGenerator     → gera a resposta final com o LLaMA
"""

from .config import RAGConfig
from .context_builder import ContextBuilder
from .conversation import ConversationMemory
from .generator import LLaMAGenerator
from .models import RAGResponse
from .query_reformulator import QueryReformulator
from .query_router import SoilQueryRouter
from .retriever import SoilRetriever, VectorRetriever


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
        self._conversation_memory = ConversationMemory(self._config.conversation.max_turns)

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        include_soil: bool | None = None,
        session_id: str | None = None,
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

        Retorna:
            RAGResponse com a resposta, documentos recuperados e dados de solo.
        """
        history = (
            self._conversation_memory.get_history(session_id) if session_id else []
        )

        # Perguntas de seguimento (ex: "e para o feijão?") não contêm contexto
        # suficiente por si só — reformula-se como pergunta autónoma antes de
        # a usar para recuperação e classificação. A pergunta original é
        # mantida para a geração da resposta e para o histórico da conversa.
        retrieval_query = self._query_reformulator.reformulate(question, history)

        if include_soil is None:
            include_soil = self._query_router.should_use_soil_data(retrieval_query)

        vector_docs = self._vector_retriever.retrieve(retrieval_query)
        soil_properties = self._soil_retriever.retrieve() if include_soil else []

        context = self._context_builder.build(vector_docs, soil_properties)

        answer = self._generator.generate(
            system_prompt=self._config.system_prompt,
            context=context,
            question=question,
            history=history,
        )

        if session_id:
            self._conversation_memory.add_turn(session_id, question, answer)

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_documents=vector_docs,
            soil_context=soil_properties,
            model_name=self._generator.model_name,
        )

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
