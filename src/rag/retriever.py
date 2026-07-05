"""
Recuperadores de informação para o sistema RAG.

Classes:
    BaseRetriever     — interface abstracta comum a todos os recuperadores
    VectorRetriever   — recupera chunks da base vectorial ChromaDB
    SoilRetriever     — recupera propriedades do solo da base relacional SQLite
"""

from abc import ABC, abstractmethod

import chromadb
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from .config import RetrieverConfig
from .models import RetrievedDocument, SoilProperty


class BaseRetriever(ABC):
    """Interface abstracta para todos os recuperadores de informação."""

    @abstractmethod
    def retrieve(self, query: str) -> list:
        """Recupera informação relevante para a query fornecida."""


class VectorRetriever(BaseRetriever):
    """
    Recupera chunks de documentos agrícolas da base vectorial ChromaDB
    usando similaridade semântica via sentence-transformers.
    """

    def __init__(self, config: RetrieverConfig) -> None:
        self._config = config
        self._embedding_model = SentenceTransformer(
            config.embedding_model,
            device=config.embedding_device,
        )
        self._client = chromadb.PersistentClient(path=str(config.chroma_path))
        self._collection = self._client.get_collection(config.collection_name)

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """
        Codifica a query e recupera os n_results chunks mais similares.

        Retorna:
            Lista de RetrievedDocument ordenada por relevância decrescente.
        """
        query_embedding = self._embedding_model.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self._config.n_results,
            include=["documents", "metadatas", "distances"],
        )

        documents: list[RetrievedDocument] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Distância coseno ChromaDB ∈ [0, 2]; converter para similaridade ∈ [0, 1]
            similarity = 1.0 - (dist / 2.0)
            documents.append(
                RetrievedDocument(
                    content=doc,
                    source=meta.get("source", "desconhecido"),
                    relevance_score=similarity,
                    metadata=meta,
                )
            )

        return documents

    @property
    def collection_count(self) -> int:
        """Número total de documentos indexados na colecção."""
        return self._collection.count()


class SoilRetriever(BaseRetriever):
    """
    Recupera as propriedades edafológicas da base de dados relacional SQLite.
    Os dados de solo são sempre retornados na íntegra para o local configurado,
    independentemente da query — funcionam como contexto fixo de localização.
    """

    # Tradução dos nomes técnicos das propriedades para Português
    _PROPERTY_LABELS: dict[str, str] = {
        "ph": "pH do Solo",
        "nitrogen_total": "Azoto Total",
        "phosphorous_extractable": "Fósforo Extractável",
        "potassium_extractable": "Potássio Extractável",
        "calcium_extractable": "Cálcio Extractável",
        "magnesium_extractable": "Magnésio Extractável",
        "iron_extractable": "Ferro Extractável",
        "zinc_extractable": "Zinco Extractável",
        "sulphur_extractable": "Enxofre Extractável",
        "aluminium_extractable": "Alumínio Extractável",
        "carbon_total": "Carbono Total",
        "carbon_organic": "Carbono Orgânico",
        "bulk_density": "Densidade Aparente",
        "stone_content": "Teor de Pedras",
        "silt_content": "Teor de Silte",
        "clay_content": "Teor de Argila",
        "sand_content": "Teor de Areia",
        "cation_exchange_capacity": "Capacidade de Troca Catiónica (CTC)",
    }

    def __init__(self, config: RetrieverConfig) -> None:
        self._config = config
        self._engine = create_engine(config.soil_db_url)

    def retrieve(self, query: str = "") -> list[SoilProperty]:
        """
        Retorna todas as propriedades do solo para o local configurado.

        O argumento query é ignorado — o solo é sempre carregado por localização.
        """
        stmt = text(
            "SELECT location, property, value, unity "
            "FROM soil_data WHERE location = :loc"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt, {"loc": self._config.soil_location}).fetchall()

        return [
            SoilProperty(
                location=row[0],
                property_name=row[1],
                value=row[2],
                unit=row[3],
            )
            for row in rows
        ]

    def get_property_label(self, property_name: str) -> str:
        """Devolve o rótulo em Português para uma propriedade do solo."""
        return self._PROPERTY_LABELS.get(property_name, property_name)
