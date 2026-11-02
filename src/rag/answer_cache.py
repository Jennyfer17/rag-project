"""
Cache de respostas para perguntas repetidas.

Gerar uma resposta demora minutos em CPU, e as perguntas frequentes repetem-se.
A chave é o texto normalizado da pergunta, sem correspondência aproximada: duas
perguntas muito parecidas sobre culturas diferentes não podem partilhar resposta.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from .models import RAGResponse
from .text_utils import normalize_text


class AnswerCache:
    """Cache LRU com tempo de vida, segura para uso entre threads."""

    def __init__(
        self,
        max_size: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[tuple[str, bool | None], tuple[float, RAGResponse]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(question: str, include_soil: bool | None) -> tuple[str, bool | None]:
        return " ".join(normalize_text(question).split()), include_soil

    def get(self, question: str, include_soil: bool | None) -> RAGResponse | None:
        key = self._key(question, include_soil)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, response = entry
            if self._clock() - stored_at > self._ttl_seconds:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return response

    def put(self, question: str, include_soil: bool | None, response: RAGResponse) -> None:
        if self._max_size <= 0:
            return
        key = self._key(question, include_soil)
        with self._lock:
            self._entries[key] = (self._clock(), response)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)
