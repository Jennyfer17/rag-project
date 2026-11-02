"""
Memória de conversa — mantém o histórico de perguntas e respostas por sessão
para que o sistema consiga responder correctamente a perguntas de seguimento.
"""

import threading
from collections import deque


class ConversationMemory:
    """
    Guarda o histórico recente de trocas (pergunta/resposta) por sessão,
    em memória do processo.

    Cada sessão (ex: número de telefone no WhatsApp, ou um session_id
    fornecido pelo cliente da API) mantém uma janela deslizante das últimas
    `max_turns` trocas — as mais antigas são descartadas automaticamente.
    As respostas são guardadas até `max_answer_chars` caracteres: chega para
    entender perguntas de seguimento, sem encher o prompt das seguintes.
    """

    def __init__(self, max_turns: int = 2, max_answer_chars: int = 400) -> None:
        self._max_turns = max_turns
        self._max_answer_chars = max_answer_chars
        self._sessions: dict[str, deque[dict]] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> list[dict]:
        """Devolve o histórico da sessão como mensagens {role, content} para o LLM."""
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Regista uma troca pergunta/resposta na sessão."""
        with self._lock:
            history = self._sessions.setdefault(
                session_id, deque(maxlen=self._max_turns * 2)
            )
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer[: self._max_answer_chars]})

    def clear(self, session_id: str) -> None:
        """Remove o histórico de uma sessão."""
        with self._lock:
            self._sessions.pop(session_id, None)
