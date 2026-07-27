"""
Guarda de domínio — restringe as respostas do sistema a perguntas
relacionadas com agricultura, recusando educadamente perguntas fora desse
âmbito antes de gastar recursos com recuperação e geração.
"""

import logging

import ollama

from .config import ModelConfig

logger = logging.getLogger(__name__)

OUT_OF_DOMAIN_MESSAGE = (
    "Sou o Amigo do Agricultor e estou habilitado apenas a responder a "
    "perguntas sobre agricultura (preparação do solo, sementeira, adubação, "
    "pragas e doenças, colheita, entre outros temas agrícolas). Por favor, "
    "faça uma pergunta relacionada com agricultura."
)

_DOMAIN_CHECK_PROMPT = """Classifica se a pergunta do utilizador está dentro do domínio de \
agricultura, incluindo: preparação do solo, sementeira, operações culturais \
(sacha, amontoa, rega, adubação), colheita, pragas e doenças das culturas, \
propriedades do solo, e recomendações agrícolas em geral.

Cumprimentos, saudações e agradecimentos simples (ex: "olá", "bom dia", \
"obrigado") também devem ser considerados dentro do domínio, pois fazem \
parte da interacção normal com o assistente.

Responde apenas com uma palavra:
- SIM, se a pergunta está dentro do domínio agrícola (ou é um cumprimento/agradecimento)
- NAO, se está fora desse domínio (ex: política, desporto, entretenimento, saúde humana, tecnologia não agrícola, etc.)"""


class DomainGuard:
    """
    Verifica se uma pergunta pertence ao domínio agrícola, usando o LLM
    como classificador binário.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.ollama_base_url)

    def is_in_domain(self, question: str) -> bool:
        """
        Retorna True se a pergunta pertence ao domínio agrícola (ou é um
        cumprimento/agradecimento), False caso contrário.

        Em caso de falha na chamada ao modelo, assume True (fail-open) — a
        recusa de uma pergunta não deve depender de uma classificação extra
        que pode falhar independentemente do resto do pipeline.
        """
        try:
            response = self._client.chat(
                model=self._config.model_name,
                messages=[
                    {"role": "system", "content": _DOMAIN_CHECK_PROMPT},
                    {"role": "user", "content": f"Pergunta: {question}"},
                ],
                options={"temperature": 0.0, "num_predict": 5},
            )
        except Exception as exc:
            logger.warning(
                "Falha ao classificar o domínio da pergunta, a assumir domínio válido: %s", exc
            )
            return True

        return response.message.content.strip().upper().startswith("SIM")
