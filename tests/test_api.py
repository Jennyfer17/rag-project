"""
Testes funcionais da API — Amigo do Agricultor.

Correr (a partir da raiz do projecto):
    pytest tests -v

Os testes de tempo de resposta com os modelos reais (Ollama, ChromaDB,
embeddings) estão desligados por omissão; para os correr:
    AMIGO_LIVE_TESTS=1 pytest tests -v -m live
"""

import base64
import hashlib
import hmac
import time

import anyio
import httpx
import pytest

from conftest import FakeSynthesizer, live_tests_enabled
from rag.domain_guard import OUT_OF_DOMAIN_MESSAGE

AUDIO_URL = "https://api.twilio.com/media/audio-1"
IMAGE_URL = "https://api.twilio.com/media/image-1"

# Termos que não devem aparecer numa mensagem dirigida ao agricultor.
_TECHNICAL_TERMS = ("traceback", "exception", "error", "erro interno", "none", "null", "http", "status_code")


def _assert_friendly(message: str) -> None:
    assert message.strip(), "a mensagem de erro não pode vir vazia"
    lowered = message.lower()
    for term in _TECHNICAL_TERMS:
        assert term not in lowered, f"mensagem com detalhe técnico ({term!r}): {message}"


# ---------------------------------------------------------------------------
# 1. Processa mensagens de texto
# ---------------------------------------------------------------------------

class TestMensagensDeTexto:
    def test_query_devolve_resposta_com_fontes(self, client):
        response = client.post("/query", json={"question": "Como faço o composto para a machamba?"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["question"] == "Como faço o composto para a machamba?"
        assert payload["answer"] == "Resposta agrícola para: Como faço o composto para a machamba?"
        assert payload["model_name"] == "fake-llama"
        assert payload["sources"] == [{"source": "manual.txt", "relevance_score": 0.9}]

    def test_whatsapp_texto_recebe_resposta_em_texto(self, send_whatsapp, twilio):
        response = send_whatsapp(body="Quando devo regar a alface?")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/xml")
        assert len(twilio.sent) == 1
        reply = twilio.sent[0]
        assert reply["to"] == "whatsapp:+258840000001"
        assert reply["body"] == "Resposta agrícola para: Quando devo regar a alface?"
        assert reply["media_url"] is None

    def test_whatsapp_trunca_respostas_longas(self, send_whatsapp, twilio, generator, monkeypatch):
        monkeypatch.setattr(generator, "generate", lambda **kwargs: "milho " * 1000)

        send_whatsapp(body="Fala-me de tudo sobre o milho")

        assert len(twilio.sent[0]["body"]) <= 1600


# ---------------------------------------------------------------------------
# 2. Processa mensagens em áudio
# ---------------------------------------------------------------------------

class TestMensagensDeAudio:
    def test_nota_de_voz_e_transcrita_e_respondida(self, send_whatsapp, twilio, transcriber, generator):
        twilio.media[AUDIO_URL] = b"ogg-audio-bytes"

        response = send_whatsapp(media_url=AUDIO_URL, media_type="audio/ogg")

        assert response.status_code == 200
        assert transcriber.received == [b"ogg-audio-bytes"]
        assert generator.calls[-1]["question"] == transcriber.text
        assert twilio.sent[0]["body"] == f"Resposta agrícola para: {transcriber.text}"


# ---------------------------------------------------------------------------
# 3. Manda mensagem de voz quando a entrada é áudio
# ---------------------------------------------------------------------------

class TestRespostaEmVoz:
    def test_entrada_em_audio_recebe_nota_de_voz(self, client, send_whatsapp, twilio, synthesizer):
        twilio.media[AUDIO_URL] = b"ogg-audio-bytes"

        send_whatsapp(media_url=AUDIO_URL, media_type="audio/ogg")

        reply = twilio.sent[0]
        assert synthesizer.received == [reply["body"]]
        assert reply["media_url"] is not None
        assert reply["media_url"].startswith("http://testserver/media/")

        # O Twilio descarrega o áudio a partir do URL público indicado.
        media_path = reply["media_url"].removeprefix("http://testserver")
        audio = client.get(media_path)
        assert audio.status_code == 200
        assert audio.headers["content-type"] == "audio/ogg"
        assert audio.content == FakeSynthesizer.AUDIO

    def test_entrada_em_texto_nao_recebe_nota_de_voz(self, send_whatsapp, twilio, synthesizer):
        send_whatsapp(body="Como semeio a couve?")

        assert synthesizer.received == []
        assert twilio.sent[0]["media_url"] is None

    def test_falha_na_sintese_ainda_envia_texto(self, send_whatsapp, twilio, synthesizer, monkeypatch):
        def broken(text: str) -> bytes:
            raise RuntimeError("Piper indisponível")

        monkeypatch.setattr(synthesizer, "synthesize", broken)
        twilio.media[AUDIO_URL] = b"ogg-audio-bytes"

        send_whatsapp(media_url=AUDIO_URL, media_type="audio/ogg")

        reply = twilio.sent[0]
        assert reply["body"].startswith("Resposta agrícola para:")
        assert reply["media_url"] is None


# ---------------------------------------------------------------------------
# 4. Recebe imagens
# ---------------------------------------------------------------------------

class TestImagens:
    def test_imagem_sem_legenda_usa_descricao_como_pergunta(
        self, send_whatsapp, twilio, image_describer, generator, synthesizer
    ):
        twilio.media[IMAGE_URL] = b"jpeg-bytes"

        send_whatsapp(media_url=IMAGE_URL, media_type="image/jpeg")

        assert image_describer.received == [b"jpeg-bytes"]
        assert generator.calls[-1]["question"] == image_describer.text
        assert twilio.sent[0]["body"] == f"Resposta agrícola para: {image_describer.text}"
        assert twilio.sent[0]["media_url"] is None
        assert synthesizer.received == []

    def test_imagem_com_legenda_combina_legenda_e_descricao(self, send_whatsapp, twilio, image_describer, generator):
        twilio.media[IMAGE_URL] = b"jpeg-bytes"

        send_whatsapp(body="O que tem a minha couve?", media_url=IMAGE_URL, media_type="image/jpeg")

        question = generator.calls[-1]["question"]
        assert question.startswith("O que tem a minha couve?")
        assert image_describer.text in question


# ---------------------------------------------------------------------------
# 5. Restringe ao domínio agrícola
# ---------------------------------------------------------------------------

class TestDominioAgricola:
    @pytest.mark.parametrize(
        "question",
        [
            "Quem ganhou o campeonato de futebol este ano?",
            "Qual é o melhor smartphone para comprar?",
            "Qual é a capital da Austrália?",  # ambígua → decidida pelo classificador (LLM)
        ],
    )
    def test_pergunta_fora_do_dominio_e_recusada(self, client, generator, question):
        response = client.post("/query", json={"question": question})

        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"] == OUT_OF_DOMAIN_MESSAGE
        assert payload["sources"] == []
        assert generator.calls == [], "não deve gastar uma chamada ao LLM com perguntas fora do domínio"

    def test_cumprimento_e_aceite(self, client, generator):
        response = client.post("/query", json={"question": "Olá, bom dia!"})

        assert response.json()["answer"] != OUT_OF_DOMAIN_MESSAGE
        assert len(generator.calls) == 1

    def test_whatsapp_fora_do_dominio_recebe_recusa_educada(self, send_whatsapp, twilio):
        send_whatsapp(body="Quem vai ganhar as eleicoes?")

        assert twilio.sent[0]["body"] == OUT_OF_DOMAIN_MESSAGE


# ---------------------------------------------------------------------------
# 6. Mantém contexto
# ---------------------------------------------------------------------------

class TestContexto:
    def test_pergunta_de_seguimento_recebe_historico_da_sessao(self, client, generator):
        client.post("/query", json={"question": "Como controlo a traça da couve?", "session_id": "s1"})
        client.post("/query", json={"question": "E quanto devo aplicar?", "session_id": "s1"})

        history = generator.calls[1]["history"]
        assert history == [
            {"role": "user", "content": "Como controlo a traça da couve?"},
            {"role": "assistant", "content": "Resposta agrícola para: Como controlo a traça da couve?"},
        ]

    def test_seguimento_ambiguo_nao_e_recusado_gracas_ao_contexto(self, client):
        # "E quanto devo aplicar?" sozinha não tem nada de agrícola; com o
        # contexto da sessão é reformulada e reconhecida como agrícola.
        isolated = client.post("/query", json={"question": "E quanto devo aplicar?"})
        assert isolated.json()["answer"] == OUT_OF_DOMAIN_MESSAGE

        client.post("/query", json={"question": "Como controlo a traça da couve?", "session_id": "s1"})
        response = client.post("/query", json={"question": "E quanto devo aplicar?", "session_id": "s1"})

        assert response.json()["answer"] != OUT_OF_DOMAIN_MESSAGE

    def test_sem_session_id_nao_ha_memoria(self, client, generator):
        client.post("/query", json={"question": "Como controlo a traça da couve?"})
        client.post("/query", json={"question": "Como faço o composto?"})

        assert generator.calls[1]["history"] == []

    def test_whatsapp_mantem_contexto_pelo_numero(self, send_whatsapp, generator):
        send_whatsapp(body="Como semeio a alface?")
        send_whatsapp(body="E quando devo regar?")

        history = generator.calls[1]["history"]
        assert history[0] == {"role": "user", "content": "Como semeio a alface?"}

    def test_historico_limitado_as_ultimas_trocas(self, client, generator, pipeline):
        max_turns = pipeline.config.conversation.max_turns
        for i in range(max_turns + 2):
            client.post("/query", json={"question": f"Pergunta {i} sobre a machamba", "session_id": "s1"})

        last_history = generator.calls[-1]["history"]
        assert len(last_history) == max_turns * 2
        assert "Pergunta 0 sobre a machamba" not in [turn["content"] for turn in last_history]


# ---------------------------------------------------------------------------
# 7. Isolamento entre utilizadores
# ---------------------------------------------------------------------------

class TestIsolamento:
    def test_sessoes_diferentes_nao_partilham_historico(self, client, generator):
        client.post("/query", json={"question": "Como controlo a traça da couve?", "session_id": "ana"})
        client.post("/query", json={"question": "Como faço o composto?", "session_id": "joao"})
        client.post("/query", json={"question": "E quanto devo aplicar?", "session_id": "ana"})

        assert generator.calls[1]["history"] == []
        ana_history = [turn["content"] for turn in generator.calls[2]["history"]]
        assert "Como controlo a traça da couve?" in ana_history
        assert all("composto" not in content for content in ana_history)

    def test_whatsapp_numeros_diferentes_nao_partilham_historico(self, send_whatsapp, twilio, generator):
        send_whatsapp(body="Como semeio a alface?", from_="whatsapp:+258840000001")
        send_whatsapp(body="Como semeio a cebola?", from_="whatsapp:+258840000002")

        assert generator.calls[1]["history"] == []
        assert [msg["to"] for msg in twilio.sent] == ["whatsapp:+258840000001", "whatsapp:+258840000002"]
        assert "alface" in twilio.sent[0]["body"]
        assert "cebola" in twilio.sent[1]["body"]


# ---------------------------------------------------------------------------
# 8. Atendimento concorrente
# ---------------------------------------------------------------------------

class TestConcorrencia:
    def test_pedidos_simultaneos_sao_atendidos_em_paralelo(self, client, generator):
        n_users = 8
        generator.delay = 0.5
        results: dict[int, httpx.Response] = {}

        async def ask(i: int, http: httpx.AsyncClient) -> None:
            results[i] = await http.post(
                "/query",
                json={"question": f"Como cuido da machamba número {i}?", "session_id": f"user-{i}"},
            )

        async def run_all() -> None:
            transport = httpx.ASGITransport(app=client.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30) as http:
                async with anyio.create_task_group() as tg:
                    for i in range(n_users):
                        tg.start_soon(ask, i, http)

        start = time.perf_counter()
        anyio.run(run_all)
        elapsed = time.perf_counter() - start

        assert len(results) == n_users
        for i, response in results.items():
            assert response.status_code == 200
            # Cada utilizador recebe a resposta à sua própria pergunta.
            assert response.json()["answer"] == f"Resposta agrícola para: Como cuido da machamba número {i}?"
        assert generator.max_in_flight > 1, "os pedidos foram processados um de cada vez"
        assert elapsed < n_users * generator.delay

    def test_webhook_responde_ao_twilio_antes_de_processar(self, send_whatsapp, generator):
        # O Twilio desiste ao fim de ~15 s; o processamento corre em background.
        generator.delay = 0.2
        start = time.perf_counter()
        response = send_whatsapp(body="Como semeio a couve?")
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert "A processar" in response.text
        assert elapsed < 15


# ---------------------------------------------------------------------------
# 9. Tempo de resposta menor que 60 s
# ---------------------------------------------------------------------------

class TestTempoDeResposta:
    def test_query_responde_em_menos_de_60s(self, client):
        start = time.perf_counter()
        response = client.post("/query", json={"question": "Como faço o viveiro da couve?"})
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 60

    @pytest.mark.live
    @pytest.mark.skipif(not live_tests_enabled(), reason="defina AMIGO_LIVE_TESTS=1 para usar os modelos reais")
    @pytest.mark.parametrize(
        "question",
        [
            "Como devo fazer os canteiros do viveiro da couve num terreno que fica encharcado?",
            "A minha couve tem lagartas verdes pequenas a comer as folhas. O que devo fazer?",
            "Como faço o composto para melhorar a terra da machamba?",
        ],
    )
    def test_pipeline_real_responde_em_menos_de_60s(self, question):
        from fastapi.testclient import TestClient

        from api.main import app

        with TestClient(app) as live_client:  # corre o lifespan e carrega o pipeline real
            start = time.perf_counter()
            response = live_client.post("/query", json={"question": question}, timeout=120)
            elapsed = time.perf_counter() - start

        assert response.status_code == 200
        assert response.json()["answer"].strip()
        assert elapsed < 60, f"resposta demorou {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# 10. Segurança
# ---------------------------------------------------------------------------

def _twilio_signature(token: str, url: str, params: dict) -> str:
    payload = url + "".join(f"{k}{v}" for k, v in sorted(params.items()))
    digest = hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


class TestSeguranca:
    TOKEN = "test-auth-token"
    URL = "http://testserver/webhook/whatsapp"

    def _form(self, body: str = "Como semeio a couve?") -> dict:
        return {
            "MessageSid": "SM1",
            "From": "whatsapp:+258840000001",
            "To": "whatsapp:+14155238886",
            "Body": body,
            "NumMedia": "0",
        }

    def test_assinatura_twilio_valida_e_aceite(self, client, twilio, monkeypatch):
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", self.TOKEN)
        form = self._form()
        signature = _twilio_signature(self.TOKEN, self.URL, form)

        response = client.post("/webhook/whatsapp", data=form, headers={"X-Twilio-Signature": signature})

        assert response.status_code == 200
        assert len(twilio.sent) == 1

    def test_assinatura_twilio_invalida_e_rejeitada(self, client, twilio, monkeypatch):
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", self.TOKEN)
        form = self._form()
        forged = _twilio_signature("token-errado", self.URL, form)

        response = client.post("/webhook/whatsapp", data=form, headers={"X-Twilio-Signature": forged})

        assert response.status_code == 403
        assert twilio.sent == []

    def test_mensagem_adulterada_e_rejeitada(self, client, twilio, monkeypatch):
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", self.TOKEN)
        signature = _twilio_signature(self.TOKEN, self.URL, self._form())

        tampered = self._form(body="Mensagem alterada por terceiros")
        response = client.post("/webhook/whatsapp", data=tampered, headers={"X-Twilio-Signature": signature})

        assert response.status_code == 403
        assert twilio.sent == []

    def test_pedido_sem_assinatura_e_rejeitado_quando_token_configurado(self, client, twilio, monkeypatch):
        # Com o token configurado, qualquer pessoa que conheça o URL do
        # webhook não deve conseguir fazer-se passar pelo Twilio.
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", self.TOKEN)

        response = client.post("/webhook/whatsapp", data=self._form())

        assert response.status_code == 403
        assert twilio.sent == []

    @pytest.mark.parametrize("filename", ["..%2F..%2F.env", "..%5C..%5C.env", "%2E%2E%2Frequirements.txt"])
    def test_media_bloqueia_path_traversal(self, client, filename):
        response = client.get(f"/media/{filename}")

        assert response.status_code in (400, 404)
        assert "TWILIO" not in response.text

    def test_media_inexistente_devolve_404(self, client):
        assert client.get("/media/nao-existe.ogg").status_code == 404

    def test_pergunta_demasiado_longa_e_rejeitada(self, client, generator):
        response = client.post("/query", json={"question": "milho " * 500})

        assert response.status_code == 422
        assert generator.calls == []


# ---------------------------------------------------------------------------
# 11. Mensagem vazia ou anexo não suportado → mensagem de erro amigável
# ---------------------------------------------------------------------------

class TestErrosAmigaveis:
    @pytest.mark.parametrize("body", ["", "   "])
    def test_whatsapp_mensagem_vazia_recebe_mensagem_amigavel(self, send_whatsapp, twilio, body):
        response = send_whatsapp(body=body)

        assert response.status_code == 200
        assert len(twilio.sent) == 1, "o agricultor não recebeu nenhuma resposta"
        _assert_friendly(twilio.sent[0]["body"])

    @pytest.mark.parametrize(
        "media_type", ["application/pdf", "video/mp4", "text/vcard", "application/vnd.ms-excel"]
    )
    def test_whatsapp_anexo_nao_suportado_recebe_mensagem_amigavel(
        self, send_whatsapp, twilio, generator, media_type
    ):
        url = "https://api.twilio.com/media/unsupported"
        twilio.media[url] = b"some-bytes"

        response = send_whatsapp(media_url=url, media_type=media_type)

        assert response.status_code == 200
        assert generator.calls == []
        assert len(twilio.sent) == 1
        _assert_friendly(twilio.sent[0]["body"])

    def test_falha_no_download_do_anexo_recebe_mensagem_amigavel(self, send_whatsapp, twilio):
        # URL não registado → o download falha (KeyError no duplo do Twilio).
        send_whatsapp(media_url="https://api.twilio.com/media/perdido", media_type="audio/ogg")

        assert len(twilio.sent) == 1
        _assert_friendly(twilio.sent[0]["body"])

    def test_falha_do_modelo_recebe_mensagem_amigavel(self, send_whatsapp, twilio, generator, monkeypatch):
        def broken(**kwargs):
            raise RuntimeError("Ollama não está a correr")

        monkeypatch.setattr(generator, "generate", broken)

        send_whatsapp(body="Como semeio a couve?")

        _assert_friendly(twilio.sent[0]["body"])

    @pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "  "}])
    def test_query_vazia_e_rejeitada(self, client, generator, payload):
        response = client.post("/query", json=payload)

        assert response.status_code == 422
        assert generator.calls == []


if __name__ == "__main__":
    # Permite correr este ficheiro directamente (ex: "Run Python File" no VS Code).
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
