"""
Recuperadores de informação para o sistema RAG.

Classes:
    BaseRetriever     — interface abstracta comum a todos os recuperadores
    VectorRetriever   — recupera chunks da base vectorial ChromaDB
    SoilRetriever     — recupera propriedades do solo da base relacional SQLite
"""

from abc import ABC, abstractmethod

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from .bm25 import BM25Index, reciprocal_rank_fusion
from .config import RetrieverConfig
from .models import RetrievedDocument, SoilProperty


class BaseRetriever(ABC):
    """Interface abstracta para todos os recuperadores de informação."""

    @abstractmethod
    def retrieve(self, query: str) -> list:
        """Recupera informação relevante para a query fornecida."""


class VectorRetriever(BaseRetriever):
    """
    Recupera chunks dos manuais agrícolas por pesquisa híbrida: similaridade
    semântica (embeddings guardados no ChromaDB) e BM25, fundidas por RRF.

    A colecção é pequena, por isso os embeddings e o índice BM25 são mantidos
    em memória e a comparação é exacta. Os chunks com similaridade abaixo de
    `min_relevance` são descartados, e os primeiros resultados podem incluir o
    chunk seguinte do mesmo manual: as secções costumam ter os sintomas num
    chunk e o tratamento no seguinte.
    """

    def __init__(self, config: RetrieverConfig) -> None:
        self._config = config
        self._embedding_model = SentenceTransformer(
            config.embedding_model,
            device=config.embedding_device,
        )
        self._client = chromadb.PersistentClient(path=str(config.chroma_path))
        self._collection = self._client.get_collection(config.collection_name)

        stored = self._collection.get(include=["documents", "metadatas", "embeddings"])
        self._documents: list[str] = stored["documents"]
        self._metadatas: list[dict] = stored["metadatas"]
        embeddings = np.asarray(stored["embeddings"], dtype=np.float32).reshape(
            len(self._documents), -1
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._embeddings = embeddings / np.clip(norms, 1e-12, None)
        self._bm25 = BM25Index(self._documents)
        self._positions = {
            (meta.get("source"), meta.get("chunk_index")): position
            for position, meta in enumerate(self._metadatas)
        }

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        query_embedding = self._embedding_model.encode(query, normalize_embeddings=True)
        similarities = self._embeddings @ np.asarray(query_embedding, dtype=np.float32)

        limit = self._config.candidates
        dense_ranking = np.argsort(-similarities)[:limit].tolist()
        lexical_ranking = self._bm25.rank(query, limit)
        fused = reciprocal_rank_fusion(dense_ranking, lexical_ranking)

        protected = set(lexical_ranking[: self._config.lexical_protected])
        selected = [
            position
            for position in fused
            if similarities[position] >= self._config.min_relevance or position in protected
        ][: self._config.n_results]

        return self._build_documents(selected, similarities)

    def _build_documents(
        self, selected: list[int], similarities: np.ndarray
    ) -> list[RetrievedDocument]:
        """Monta os documentos, juntando o chunk seguinte aos primeiros resultados."""
        used = set(selected)
        documents: list[RetrievedDocument] = []
        for rank, position in enumerate(selected):
            meta = self._metadatas[position]
            content = self._documents[position]
            if rank < self._config.expand_top:
                neighbour = self._positions.get(
                    (meta.get("source"), (meta.get("chunk_index") or 0) + 1)
                )
                if neighbour is not None and neighbour not in used:
                    used.add(neighbour)
                    content = f"{content} {self._documents[neighbour]}"
            documents.append(
                RetrievedDocument(
                    content=content,
                    source=meta.get("source", "desconhecido"),
                    relevance_score=float(similarities[position]),
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
