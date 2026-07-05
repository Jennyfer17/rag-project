"""
Pipeline RAG principal — orquestra recuperação, construção de contexto e geração.

A classe RAGPipeline é o ponto de entrada único do sistema e será consumida
directamente pela API FastAPI.

Fluxo de uma query:
  1. VectorRetriever  → recupera chunks similares do ChromaDB
  2. SoilRetriever    → carrega propriedades do solo da BD relacional
  3. ContextBuilder   → monta o bloco de contexto aumentado
  4. LLaMAGenerator   → gera a resposta final com o LLaMA
"""

from .config import RAGConfig
from .context_builder import ContextBuilder
from .generator import LLaMAGenerator
from .models import RAGResponse
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

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def query(self, question: str, include_soil: bool = True) -> RAGResponse:
        """
        Processa uma pergunta e retorna a resposta aumentada pelo contexto.

        Parâmetros:
            question     : pergunta do agricultor em linguagem natural
            include_soil : se True, inclui os dados do solo no contexto

        Retorna:
            RAGResponse com a resposta, documentos recuperados e dados de solo.
        """
        vector_docs = self._vector_retriever.retrieve(question)
        soil_properties = self._soil_retriever.retrieve() if include_soil else []

        context = self._context_builder.build(vector_docs, soil_properties)

        answer = self._generator.generate(
            system_prompt=self._config.system_prompt,
            context=context,
            question=question,
        )

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
