"""
Gerador de respostas usando o modelo LLaMA via Ollama.

O LLaMAGenerator encapsula toda a comunicação com o servidor Ollama,
expondo os hiperparâmetros definidos em ModelConfig.
"""

import ollama
from ollama import ResponseError

from .config import ModelConfig


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
        }

    def generate(
        self,
        system_prompt: str,
        context: str,
        question: str,
    ) -> str:
        """
        Gera uma resposta fundamentada no contexto recuperado.

        O contexto e a pergunta são combinados numa única mensagem de utilizador
        para que o modelo os processe conjuntamente.

        Lança:
            RuntimeError se o modelo não estiver disponível no servidor Ollama.
        """
        user_message = (
            f"Contexto disponível:\n{context}\n\n"
            f"Pergunta do agricultor: {question}"
        )

        try:
            response = self._client.chat(
                model=self._config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options=self._options,
            )
        except ResponseError as exc:
            raise RuntimeError(
                f"Erro ao contactar o modelo '{self._config.model_name}' "
                f"em {self._config.ollama_base_url}: {exc}"
            ) from exc

        return response.message.content

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def list_available_models(self) -> list[str]:
        """Devolve os nomes dos modelos disponíveis no servidor Ollama local."""
        response = self._client.list()
        return [m.model for m in response.models]
