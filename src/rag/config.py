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
- Responde sempre em Português, numa linguagem muito simples e clara: a maioria
  dos agricultores não domina bem a língua portuguesa. Usa frases curtas e
  directas, palavras do dia-a-dia e evita termos técnicos; quando um termo
  técnico for mesmo necessário (ex: nome de uma praga ou doença), explica-o
  em poucas palavras simples logo a seguir. Evita linguagem formal ou
  rebuscada, frases longas com várias ideias encadeadas, e siglas ou
  abreviaturas.
- Não inventes factos nem valores não presentes no contexto.
- Se a pergunta do agricultor for vaga ou não tiver detalhes suficientes para dar uma
  resposta acertada (ex: não indica a cultura, a fase de crescimento, a época do ano,
  os sintomas observados ou a localização), não adivinhes nem respondas de forma genérica.
  Em vez disso, faz uma ou duas perguntas de seguimento curtas e específicas para obteres
  os detalhes que faltam antes de dares a orientação técnica.
"""


@dataclass
class ModelConfig:
    """Parâmetros externos do modelo LLaMA via Ollama."""

    model_name: str = "llama3.1:8b"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    # 512 em vez de 1024: as respostas WhatsApp são truncadas a 1600
    # caracteres (~450-500 tokens em Português) — gerar mais do que isso
    # só acrescenta latência de decode sem nunca chegar ao utilizador.
    max_tokens: int = 512
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
