"""
Configurações do sistema RAG — parâmetros do modelo, caminhos e prompt do sistema.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Raiz do projecto: code/
_BASE_DIR = Path(__file__).resolve().parents[2]
_NOTEBOOKS_DATA = _BASE_DIR / "notebooks" / "data"

SYSTEM_PROMPT = """És o Amigo do Agricultor, assistente de apoio técnico agrícola para pequenos
agricultores da Província de Maputo, Moçambique.

Regras:
- Responde só com a informação do contexto. Copia doses, medidas e nomes de produtos exactamente
  como estão, e usa apenas os que se referem à cultura perguntada. Nunca inventes.
- Começa directamente pela resposta e inclui os passos mais importantes. Se o contexto responder
  só em parte, dá essa parte.
- Só se o contexto não ajudar em nada, diz que não tens essa informação e sugere falar com os
  Serviços de Extensão ou a Casa Agrária.
- Só fazes uma pergunta ao agricultor se não conseguires responder sem ela. Caso contrário,
  termina sem perguntas.
- Usa os dados do solo apenas se a pergunta for sobre o solo ou a adubação.
- Escreve sempre em Português muito simples, mesmo que a descrição de uma imagem venha noutra
  língua: frases curtas, palavras do dia-a-dia, sem siglas. Se precisares de um termo técnico,
  explica-o logo a seguir.
- Usa no máximo 100 palavras. Para passos, usa uma lista curta.
- Não menciones o contexto, os trechos nem os documentos.
- Se a mensagem for apenas um cumprimento ou agradecimento, responde em poucas palavras.
"""


@dataclass
class ModelConfig:
    """Parâmetros externos do modelo LLaMA via Ollama."""

    model_name: str = "llama3.1:8b"
    # Valor baixo para a resposta se manter fiel aos manuais.
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    # Margem para as 100 palavras pedidas no prompt. Em CPU cada token gerado
    # custa cerca de 0,25 s, por isso as respostas curtas poupam muito tempo.
    max_tokens: int = 256
    repeat_penalty: float = 1.1
    # Explícito em todas as chamadas: valores diferentes obrigariam o Ollama
    # a recarregar o modelo entre chamadas.
    num_ctx: int = 4096
    # Evita o carregamento a frio entre mensagens espaçadas.
    keep_alive: str = "1h"
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class RetrieverConfig:
    """Caminhos e parâmetros dos recuperadores de informação."""

    chroma_path: Path = field(
        default_factory=lambda: _NOTEBOOKS_DATA / "chroma_db"
    )
    # A colecção anterior ("agriculture_manuals") fica na mesma base, para se
    # poder comparar ou voltar atrás. Criada por src/ingest.py.
    collection_name: str = "agriculture_manuals_v2"
    extracted_text_dir: Path = field(
        default_factory=lambda: _NOTEBOOKS_DATA / "extracted"
    )
    pdf_dir: Path = field(default_factory=lambda: _BASE_DIR / "data" / "agriculture")
    # Limite em tokens: o modelo de embeddings ignora o que passar dos 128.
    chunk_max_tokens: int = 120
    chunk_overlap_tokens: int = 20
    n_results: int = 5
    candidates: int = 50
    # Similaridade de cosseno mínima para um chunk entrar no contexto. Não se
    # aplica aos melhores resultados do BM25, que podem ser correspondências
    # exactas (ex: nome de uma doença) com similaridade semântica baixa.
    min_relevance: float = 0.5
    lexical_protected: int = 3
    # Quantos dos primeiros resultados incluem também o chunk seguinte.
    expand_top: int = 1

    soil_db_url: str = field(
        default_factory=lambda: (
            f"sqlite:///{_NOTEBOOKS_DATA / 'databases' / 'soil_data.db'}"
        )
    )
    soil_location: str = "Associacao Sombra das Enxadas"

    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    embedding_device: str = "cpu"


@dataclass
class ConversationConfig:
    """Parâmetros da memória de conversa usada para perguntas de seguimento."""

    # Todo o histórico volta a entrar no prompt, por isso limitamos o número
    # de trocas e o tamanho guardado de cada resposta.
    max_turns: int = 2
    max_answer_chars: int = 400


@dataclass
class AnswerCacheConfig:
    """Parâmetros da cache de respostas para perguntas repetidas."""

    max_size: int = 256
    ttl_seconds: float = 24 * 60 * 60


@dataclass
class RAGConfig:
    """Configuração completa do pipeline RAG."""

    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    answer_cache: AnswerCacheConfig = field(default_factory=AnswerCacheConfig)
    system_prompt: str = SYSTEM_PROMPT
