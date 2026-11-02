"""
Testes das optimizações de recuperação, latência e imagens.

Não usam modelos reais, por isso correm em milissegundos.
"""

import pytest

from evaluation.retrieval import RetrievalExample, score_retrieval
from image_description import find_in_description
from rag.answer_cache import AnswerCache
from rag.bm25 import BM25Index, reciprocal_rank_fusion
from rag.chunking import chunk_text, clean_text
from rag.context_builder import ContextBuilder
from rag.conversation import ConversationMemory
from rag.domain_guard import DomainGuard
from rag.generator import _trim_to_last_sentence
from rag.models import RAGResponse, RetrievedDocument
from rag.query_router import SoilQueryRouter


class TestLimpezaDeTexto:
    def test_junta_palavras_cortadas_por_hifen(self):
        assert "propagação" in clean_text("material de propa-\ngação vegetativa")

    def test_mantem_palavras_compostas_na_mesma_linha(self):
        assert "feijão-nhemba" in clean_text("usar feijão-nhemba na rotação")

    def test_corrige_erros_de_ocr(self):
        cleaned = clean_text("0 cultivo da alface usa canteiros de 1 O m e 0,5 kg")
        assert cleaned.startswith("o cultivo")
        assert "10 m" in cleaned
        assert "0,5 kg" in cleaned

    def test_remove_cabecalhos_repetidos_mas_mantem_texto(self):
        header = "PRAGAS E DOENÇAS DA COUVE E REPOLHO 12"
        text = "\n".join([header, "Primeira frase.", header.replace("12", "13"), "Segunda frase.", header])
        cleaned = clean_text(text)
        assert "PRAGAS E DOENÇAS" not in cleaned
        assert "Primeira frase." in cleaned and "Segunda frase." in cleaned


class TestChunking:
    def test_respeita_o_limite(self):
        text = " ".join(f"Esta é a frase número {i} do manual." for i in range(100))
        chunks = chunk_text(text, max_size=200, overlap=40)
        assert len(chunks) > 1
        assert all(len(chunk) <= 200 for chunk in chunks)

    def test_divide_frases_maiores_que_o_limite(self):
        chunks = chunk_text("palavra " * 200, max_size=100, overlap=10)
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_sobreposicao_repete_a_ultima_frase(self):
        text = "Primeira frase bastante longa. Segunda frase bastante longa. Terceira frase bastante longa."
        chunks = chunk_text(text, max_size=62, overlap=35)
        assert "Segunda frase" in chunks[0] and "Segunda frase" in chunks[1]

    def test_usa_o_contador_de_tokens_fornecido(self):
        text = "um dois três quatro cinco seis sete oito nove dez onze doze treze catorze quinze."
        chunks = chunk_text(text, max_size=5, overlap=0, count_tokens=lambda t: len(t.split()))
        assert all(len(chunk.split()) <= 5 for chunk in chunks)


class TestPesquisaLexical:
    def test_bm25_ignora_acentos_e_prioriza_termo_exacto(self):
        index = BM25Index(["A traça da couve come as folhas.", "A rega do milho faz-se de manhã."])
        assert index.rank("traca", limit=2) == [0]

    def test_bm25_sem_correspondencia_devolve_vazio(self):
        assert BM25Index(["Rega do milho."]).rank("computador", limit=5) == []

    def test_rrf_favorece_documentos_em_ambos_os_rankings(self):
        assert reciprocal_rank_fusion([1, 2, 3], [3, 4, 1])[0] in (1, 3)
        assert set(reciprocal_rank_fusion([1], [2])) == {1, 2}


class TestRouterDoSolo:
    @pytest.mark.parametrize(
        "question",
        [
            "Quero melhorar a terra da minha machamba com composto. Como faço o composto?",
            "O terreno fica encharcado. Como faço os canteiros?",
            "Costumo pôr capim para guardar a humidade da terra.",
            "A alpha da planta",
        ],
    )
    def test_nao_activa_em_perguntas_genericas(self, question):
        assert not SoilQueryRouter().should_use_soil_data(question)

    @pytest.mark.parametrize(
        "question",
        ["Qual é o pH do meu solo?", "Que adubo devo usar?", "Falta potássio?", "Que nutrientes tem?"],
    )
    def test_activa_em_perguntas_sobre_o_solo(self, question):
        assert SoilQueryRouter().should_use_soil_data(question)


class TestGuardaDeDominio:
    class _Client:
        calls = 0

        def chat(self, **kwargs):
            type(self).calls += 1
            from types import SimpleNamespace

            return SimpleNamespace(message=SimpleNamespace(content="NAO"))

    @pytest.fixture
    def guard(self, monkeypatch):
        import rag.domain_guard as module
        from rag import ModelConfig

        self._Client.calls = 0
        monkeypatch.setattr(module.ollama, "Client", lambda host: self._Client())
        return DomainGuard(ModelConfig())

    def test_substring_nao_passa_por_palavra_agricola(self, guard):
        assert not guard.is_in_domain("Qual é a melhor escola de Maputo?")
        assert self._Client.calls == 1

    @pytest.mark.parametrize(
        "question",
        ["As folhas da minha couve têm manchas", "A cebola tem insectos", "couve repolho folhas buracos"],
    )
    def test_perguntas_agricolas_nao_chamam_o_llm(self, guard, question):
        assert guard.is_in_domain(question)
        assert self._Client.calls == 0


class TestCache:
    def _response(self, answer="resposta") -> RAGResponse:
        return RAGResponse(question="q", answer=answer)

    def test_ignora_maiusculas_acentos_e_espacos(self):
        cache = AnswerCache(10, 100)
        cache.put("Como semeio a  Couve?", None, self._response())
        assert cache.get("como semeio a couve?", None) is not None

    def test_perguntas_diferentes_nao_partilham_resposta(self):
        cache = AnswerCache(10, 100)
        cache.put("Como semeio a couve?", None, self._response())
        assert cache.get("Como semeio a alface?", None) is None

    def test_expira(self):
        now = [0.0]
        cache = AnswerCache(10, 100, clock=lambda: now[0])
        cache.put("pergunta", None, self._response())
        now[0] = 101.0
        assert cache.get("pergunta", None) is None

    def test_remove_a_entrada_menos_usada(self):
        cache = AnswerCache(2, 100)
        cache.put("a", None, self._response())
        cache.put("b", None, self._response())
        cache.get("a", None)
        cache.put("c", None, self._response())
        assert cache.get("b", None) is None
        assert cache.get("a", None) is not None


class TestPipelineComCache:
    def test_pergunta_repetida_nao_chama_o_llm(self, client, generator):
        first = client.post("/query", json={"question": "Como faço o composto?"})
        second = client.post("/query", json={"question": "como faço o COMPOSTO?"})

        assert len(generator.calls) == 1
        assert second.json()["answer"] == first.json()["answer"]
        assert second.json()["question"] == "como faço o COMPOSTO?"

    def test_resposta_da_cache_entra_no_historico(self, client, generator):
        client.post("/query", json={"question": "Como faço o composto para a horta?"})
        client.post("/query", json={"question": "Como faço o composto para a horta?", "session_id": "s1"})
        client.post("/query", json={"question": "E quando?", "session_id": "s1"})

        assert generator.calls[-1]["history"][0]["content"] == "Como faço o composto para a horta?"

    def test_seguimento_nao_usa_a_cache(self, client, generator):
        client.post("/query", json={"question": "E quanto devo aplicar de adubo?"})
        client.post("/query", json={"question": "Como controlo a traça da couve?", "session_id": "s1"})
        client.post("/query", json={"question": "E quanto devo aplicar de adubo?", "session_id": "s1"})

        assert len(generator.calls) == 3


class TestWebhook:
    def test_confirma_a_recepcao_sem_contar_como_resposta(self, send_whatsapp, twilio):
        response = send_whatsapp(body="Como semeio a couve?")

        assert "<Message>A processar" in response.text
        assert len(twilio.sent) == 1

    def test_mensagem_vazia_nao_recebe_confirmacao(self, send_whatsapp):
        assert send_whatsapp(body="").text == "<Response />"

    def test_descricao_em_ingles_pesquisa_em_portugues(
        self, send_whatsapp, twilio, image_describer, pipeline, monkeypatch
    ):
        image_describer.text = "A cabbage plant with holes in its leaves."
        twilio.media["https://api.twilio.com/media/i"] = b"jpeg"
        queries: list[str] = []
        original = pipeline._vector_retriever.retrieve

        def spy(query: str):
            queries.append(query)
            return original(query)

        monkeypatch.setattr(pipeline._vector_retriever, "retrieve", spy)

        send_whatsapp(media_url="https://api.twilio.com/media/i", media_type="image/jpeg")

        assert queries == ["Que praga ou doença da couve causa folhas comidas por lagartas e como controlar?"]
        answer = twilio.sent[0]["body"]
        assert "couve" in answer and "cabbage" not in answer

    def test_legenda_e_mantida_com_termos_em_portugues(self, send_whatsapp, twilio, image_describer, generator):
        image_describer.text = "A cabbage plant with holes."
        twilio.media["https://api.twilio.com/media/i"] = b"jpeg"

        send_whatsapp(body="O que tem?", media_url="https://api.twilio.com/media/i", media_type="image/jpeg")

        question = generator.calls[-1]["question"]
        assert question.startswith("O que tem?") and "couve" in question and "cabbage" not in question


class TestGlossario:
    def test_traduz_culturas_e_sintomas(self):
        findings = find_in_description("A lettuce leaf with yellow spots and small aphids.")
        assert findings.crops == ["alface"]
        for expected in ("folhas", "amarelas", "manchas", "afídeos"):
            assert expected in findings.signs

    def test_nao_corresponde_a_partes_de_palavras(self):
        findings = find_in_description("A person in the corner of a sporting event")
        assert findings.crops == [] and findings.signs == []

    def test_foto_sem_planta_nao_e_agricola(self):
        assert not find_in_description("A person smiling at the camera").is_agricultural
        assert not find_in_description("A plate with two eggs on a brown table").is_agricultural

    def test_pesquisa_em_frase_com_e_sem_sinais(self):
        assert "couve causa folhas manchas" in find_in_description("cabbage leaves with spots").search_query()
        assert find_in_description("a field of maize").search_query().startswith("Que pragas e doenças atacam a milho")


class TestContexto:
    def test_nao_expoe_fonte_nem_relevancia(self):
        document = RetrievedDocument(content="Texto.", source="manual.pdf", relevance_score=0.81)
        context = ContextBuilder().build([document], [])
        assert "Texto." in context
        assert "manual.pdf" not in context and "Relevância" not in context

    def test_sem_documentos_avisa(self):
        assert "Não foi encontrado" in ContextBuilder().build([], [])


class TestMemoria:
    def test_encurta_respostas_guardadas(self):
        memory = ConversationMemory(max_turns=2, max_answer_chars=10)
        memory.add_turn("s", "pergunta longa que fica inteira", "x" * 50)
        history = memory.get_history("s")
        assert history[0]["content"] == "pergunta longa que fica inteira"
        assert history[1]["content"] == "x" * 10


class TestGerador:
    def test_remove_frase_incompleta(self):
        assert _trim_to_last_sentence("Regue todos os dias. Depois aplique") == "Regue todos os dias."

    def test_mantem_texto_sem_pontuacao(self):
        assert _trim_to_last_sentence("Regue todos os dias") == "Regue todos os dias"


class TestAvaliacaoDeterministica:
    EXAMPLE = RetrievalExample(question="q", evidence=["Canteiros elevados", "1,2 m"])

    def test_ignora_acentos_e_pontuacao(self):
        score = score_retrieval(self.EXAMPLE, ["Fazer CANTEIROS  elevados, largura de 1,2 m."])
        assert score.fact_recall == 1.0
        assert score.missing_evidence == []

    def test_precisao_media_valoriza_o_topo(self):
        top = score_retrieval(self.EXAMPLE, ["canteiros elevados", "outra coisa"])
        bottom = score_retrieval(self.EXAMPLE, ["outra coisa", "canteiros elevados"])
        assert top.average_precision == 1.0
        assert bottom.average_precision == 0.5
        assert bottom.reciprocal_rank == 0.5

    def test_sem_relevantes_da_zero(self):
        score = score_retrieval(self.EXAMPLE, ["outra coisa"])
        assert score.average_precision == 0
        assert score.fact_recall == 0
