"""
Pacote de avaliação do sistema RAG usando RAGAS.

IMPORTANTE — compatibilidade com a RAGAS:
Todas as versões publicadas da ragas (até 0.4.x) importam incondicionalmente
`langchain_community.chat_models.vertexai.ChatVertexAI`, um submódulo que já
não existe na `langchain-community` actual (o pacote está a ser descontinuado
e a integração VertexAI mudou para `langchain-google-vertexai`, ver
https://github.com/langchain-ai/langchain-community/issues/674). Fixar uma
`langchain-community` antiga o suficiente para ter esse submódulo arrasta
dependências (ex: `zstandard` antigo) sem wheels pré-compiladas para esta
versão do Python, exigindo um compilador C++ que não está disponível aqui.

Como este projecto nunca usa o VertexAI (só Ollama local), registamos um stub
inofensivo em sys.modules antes de qualquer import da ragas — evita a fixação
de versões incompatíveis sem alterar o comportamento real do sistema.
"""

import sys
import types


def _disable_nest_asyncio() -> None:
    """
    A ragas aplica `nest_asyncio.apply()` incondicionalmente ao importar
    `ragas.executor`, para permitir correr dentro de um event loop já activo
    (ex: Jupyter). O nosso script de avaliação corre como processo top-level,
    sem event loop prévio, e esse patch acaba por quebrar a detecção de
    biblioteca assíncrona do `sniffio`/`httpcore` usada pelo cliente HTTP do
    Ollama — as chamadas reais ao Ollama dentro da RAGAS falham com
    "Timeout should be used inside a task" ao fechar a ligação. Como nunca
    corremos dentro de outro event loop, desactivamos o patch (chamando
    `nest_asyncio.apply()` antes de `ragas.executor` o fazer, tornando-o
    inofensivo) em vez de o deixar quebrar as chamadas HTTP.
    """
    import nest_asyncio

    nest_asyncio.apply = lambda *args, **kwargs: None


def _patch_vertexai_shim() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    class _UnavailableVertexAI:
        """Stub — o VertexAI nunca é usado neste projecto (só Ollama local)."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "VertexAI não está disponível neste projecto — usa-se Ollama local."
            )

    stub.ChatVertexAI = _UnavailableVertexAI
    sys.modules[module_name] = stub


_disable_nest_asyncio()
_patch_vertexai_shim()
