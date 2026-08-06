"""
Router de consultas — decide se uma pergunta necessita dos dados do solo.

Usa correspondência de palavras-chave sobre a pergunta do utilizador para
determinar se o contexto edafológico (base de dados relacional do solo)
deve ser incluído na resposta, evitando consultas desnecessárias à BD
quando a pergunta não tem relação com propriedades do solo.
"""

from .text_utils import normalize_text


class SoilQueryRouter:
    """
    Classifica perguntas de agricultores quanto à necessidade de dados do solo.

    A decisão é feita por correspondência de palavras-chave (insensível a
    maiúsculas/minúsculas e a acentos) sobre o texto da pergunta. É uma
    heurística determinística — sem custo de inferência de modelo — usada
    como valor por omissão quando o chamador não indica explicitamente se
    quer incluir os dados do solo.
    """

    _DEFAULT_KEYWORDS: tuple[str, ...] = (
        "solo", "solos", "terra", "terreno",
        "ph", "acidez", "acido", "alcalino", "calagem",
        "adubo", "adubacao", "adubar", "fertilizante", "fertilizacao", "fertilidade",
        "nutriente", "nutrientes", "carencia", "deficiencia",
        "azoto", "nitrogenio", "fosforo", "potassio", "calcio", "magnesio",
        "ferro", "zinco", "enxofre", "aluminio",
        "materia organica", "carbono organico", "composto", "estrume",
        "textura", "argila", "areia", "silte", "pedregosidade",
        "densidade aparente", "capacidade de troca cationica", "ctc",
        "analise do solo", "analise de solo", "propriedades do solo",
    )

    def __init__(self, keywords: tuple[str, ...] | None = None) -> None:
        self._keywords = tuple(
            normalize_text(kw) for kw in (keywords or self._DEFAULT_KEYWORDS)
        )

    def should_use_soil_data(self, question: str) -> bool:
        """
        Retorna True se a pergunta contiver alguma palavra-chave relacionada
        com o solo, indicando que os dados edafológicos devem ser incluídos
        no contexto da resposta.
        """
        normalized_question = normalize_text(question)
        return any(keyword in normalized_question for keyword in self._keywords)
