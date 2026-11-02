"""
Gerador de respostas usando o modelo LLaMA via Ollama.

O LLaMAGenerator encapsula toda a comunicação com o servidor Ollama,
expondo os hiperparâmetros definidos em ModelConfig.
"""

import re

import ollama
from ollama import ResponseError

from .config import ModelConfig


def _trim_to_last_sentence(text: str) -> str:
    """Remove a frase incompleta de uma resposta cortada pelo limite de tokens."""
    matches = list(re.finditer(r"[.!?:](?=\s|$)", text))
    return text[: matches[-1].end()].strip() if matches else text.strip()


class LLaMAGenerator:
    """
    Interface com o modelo LLaMA servido localmente pelo Ollama.

    Parâmetros externos do modelo configuráveis via ModelConfig:
      - temperature    : grau de aleatoriedade/criatividade (0 = determinístico)
      - top_p          : nucleus sampling — limita candidatos à massa de probabilidade p
      - top_k          : limita candidatos aos k tokens mais prováveis
      - max_tokens     : comprimento máximo da resposta gerada (num_predict)
      - repeat_penalty : penaliza repetição de tokens recentes (> 1 = mais penalização)
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.ollama_base_url)
        self._options = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "num_predict": config.max_tokens,
            "repeat_penalty": config.repeat_penalty,
            "num_ctx": config.num_ctx,
        }

    def warm_up(self) -> None:
        """Carrega o modelo em memória, para a primeira resposta não pagar esse custo."""
        self._client.generate(model=self._config.model_name, keep_alive=self._config.keep_alive)

    def generate(
        self,
        system_prompt: str,
        context: str,
        question: str,
        history: list[dict] | None = None,
    ) -> str:
        """
        Gera uma resposta fundamentada no contexto recuperado.

        O contexto e a pergunta são combinados numa única mensagem de utilizador
        para que o modelo os processe conjuntamente. Se for fornecido um
        histórico (trocas anteriores da mesma conversa), este é inserido antes
        da mensagem actual para que o modelo consiga responder a perguntas de
        seguimento (ex: "e para o feijão?").

        Lança:
            RuntimeError se o modelo não estiver disponível no servidor Ollama.
        """
        user_message = (
            f"Contexto disponível:\n{context}\n\n"
            f"Pergunta do agricultor: {question}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat(
                model=self._config.model_name,
                messages=messages,
                options=self._options,
                keep_alive=self._config.keep_alive,
            )
        except ResponseError as exc:
            raise RuntimeError(
                f"Erro ao contactar o modelo '{self._config.model_name}' "
                f"em {self._config.ollama_base_url}: {exc}"
            ) from exc

        answer = response.message.content
        if response.done_reason == "length":
            answer = _trim_to_last_sentence(answer)
        return answer

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def list_available_models(self) -> list[str]:
        """Devolve os nomes dos modelos disponíveis no servidor Ollama local."""
        response = self._client.list()
        return [m.model for m in response.models]
