"""
Configurações do sistema RAG — parâmetros do modelo, caminhos e prompt do sistema.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Raiz do projecto: code/
_BASE_DIR = Path(__file__).resolve().parents[2]
_NOTEBOOKS_DATA = _BASE_DIR / "notebooks" / "data"

SYSTEM_PROMPT = """És o Amigo do Agricultor, um assistente virtual especializado em apoio
técnico agrícola para pequenos agricultores da Província de Maputo, Moçambique.

O teu papel é fornecer orientação técnica clara, precisa e contextualizada sobre:
- Preparação do solo e práticas de cultivo
- Sementeira, operações culturais (sacha, amontoa, rega, adubação, controlo fitossanitário)
- Colheita e tratamento pós-colheita
- Gestão de pragas e doenças das culturas
- Recomendações adaptadas às condições edafoclimáticas da Província de Maputo

Instruções:
- Usa sempre a informação do contexto fornecido para fundamentar as tuas respostas.
- Se a informação contextual for insuficiente, indica-o claramente e fornece orientação
  geral baseada nas boas práticas agrárias.
- Responde sempre em Português, de forma simples e acessível a agricultores
  com baixa literacia digital.
- Não inventes factos nem valores não presentes no contexto.
"""


@dataclass
class ModelConfig:
    """Parâmetros externos do modelo LLaMA via Ollama."""

    model_name: str = "llama3.1:8b"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 1024
    repeat_penalty: float = 1.1
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class RetrieverConfig:
    """Caminhos e parâmetros dos recuperadores de informação."""

    chroma_path: Path = field(
        default_factory=lambda: _NOTEBOOKS_DATA / "chroma_db"
    )
    collection_name: str = "agriculture_manuals"
    n_results: int = 5

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

    max_turns: int = 5


@dataclass
class RAGConfig:
    """Configuração completa do pipeline RAG."""

    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    system_prompt: str = SYSTEM_PROMPT
