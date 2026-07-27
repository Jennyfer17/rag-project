"""
Reformulador de queries — reescreve perguntas de seguimento como perguntas
autónomas, usando o histórico da conversa.

Necessário porque o VectorRetriever recupera documentos com base apenas no
texto da pergunta actual. Perguntas elípticas como "e para o feijão?" não
contêm informação suficiente para uma boa recuperação semântica — este
módulo usa o LLM para reescrevê-las como perguntas completas antes de irem
para o VectorRetriever.
"""

import logging

import ollama

from .config import ModelConfig

logger = logging.getLogger(__name__)

_REFORMULATION_PROMPT = """Dada a história da conversa e uma pergunta de seguimento, reformula a \
pergunta de seguimento como uma pergunta autónoma e completa, em Português, \
que possa ser entendida sem precisar da história da conversa.

Regras:
- Não respondas à pergunta, apenas reformula-a.
- Se a pergunta já for autónoma e não depender da história, devolve-a exactamente como está.
- Mantém o significado original — não acrescentes informação nova.
- Devolve apenas a pergunta reformulada, sem explicações nem aspas."""


class QueryReformulator:
    """
    Reescreve perguntas de seguimento como perguntas autónomas usando o LLM,
    para melhorar a recuperação vectorial em conversas com múltiplas trocas.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.ollama_base_url)

    def reformulate(self, question: str, history: list[dict]) -> str:
        """
        Reformula `question` como pergunta autónoma com base em `history`.

        Se `history` estiver vazio, devolve a pergunta original sem chamar o
        modelo — não há contexto de conversa para resolver. Se a chamada ao
        modelo falhar por qualquer razão, também recorre à pergunta original
        em vez de interromper o fluxo do pipeline.
        """
        if not history:
            return question

        history_text = "\n".join(
            f"{'Agricultor' if turn['role'] == 'user' else 'Assistente'}: {turn['content']}"
            for turn in history
        )
        user_message = (
            f"Histórico da conversa:\n{history_text}\n\n"
            f"Pergunta de seguimento: {question}\n\n"
            f"Pergunta reformulada:"
        )

        try:
            response = self._client.chat(
                model=self._config.model_name,
                messages=[
                    {"role": "system", "content": _REFORMULATION_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": 0.0, "num_predict": 128},
            )
        except Exception as exc:  # reformulação é best-effort — nunca deve interromper o pipeline
            logger.warning("Falha ao reformular a query, a usar pergunta original: %s", exc)
            return question

        reformulated = response.message.content.strip().strip('"')
        return reformulated or question
