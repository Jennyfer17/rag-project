"""
Fixtures partilhadas pelos testes funcionais da API.

Os testes exercitam a API real (FastAPI + routers + RAGPipeline, com a
memória de conversa e a guarda de domínio verdadeiras), mas substituem os
componentes pesados ou externos por duplos leves e determinísticos:

  - LLaMA / Ollama (geração, reformulação, classificação de domínio)
  - ChromaDB + embeddings e base de dados do solo
  - Whisper (transcrição), moondream2 (descrição de imagens), Piper (voz)
  - Twilio (download de média e envio de mensagens)

Assim os testes correm em segundos, sem rede nem GPU, e nunca enviam
mensagens reais pelo WhatsApp.
"""

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import rag.domain_guard as domain_guard_module  # noqa: E402
import rag.pipeline as pipeline_module  # noqa: E402
from api import dependencies  # noqa: E402
from api.main import app  # noqa: E402
from api.routers import media as media_router  # noqa: E402
from api.routers import webhook as webhook_router  # noqa: E402
from rag import RAGConfig, RAGPipeline  # noqa: E402
from rag.models import RetrievedDocument  # noqa: E402

TWILIO_NUMBER = "whatsapp:+14155238886"


# ---------------------------------------------------------------------------
# Duplos dos componentes do RAG
# ---------------------------------------------------------------------------

class FakeVectorRetriever:
    def __init__(self, config) -> None:
        pass

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        return [RetrievedDocument(content=f"Excerto sobre: {query}", source="manual.txt", relevance_score=0.9)]

    @property
    def collection_count(self) -> int:
        return 1


class FakeSoilRetriever:
    def __init__(self, config) -> None:
        pass

    def retrieve(self, query: str = "") -> list:
        return []


class FakeGenerator:
    """
    Gera uma resposta determinística e regista cada chamada (pergunta e
    histórico recebido), para os testes poderem verificar o contexto.
    Um atraso opcional permite simular a latência do LLM.
    """

    def __init__(self, config) -> None:
        self.calls: list[dict] = []
        self.delay = 0.0
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def generate(self, system_prompt: str, context: str, question: str, history: list[dict]) -> str:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.calls.append({"question": question, "history": list(history)})
        try:
            if self.delay:
                time.sleep(self.delay)
            return f"Resposta agrícola para: {question}"
        finally:
            with self._lock:
                self._in_flight -= 1

    @property
    def model_name(self) -> str:
        return "fake-llama"

    def list_available_models(self) -> list[str]:
        return ["fake-llama"]


class FakeReformulator:
    """Junta a pergunta anterior do utilizador às perguntas de seguimento."""

    def __init__(self, config) -> None:
        pass

    def reformulate(self, question: str, history: list[dict]) -> str:
        previous = [turn["content"] for turn in history if turn["role"] == "user"]
        return f"{previous[-1]} {question}" if previous else question


class FakeOllamaClient:
    """
    Substitui o LLM usado pela DomainGuard real nos casos ambíguos (que não
    batem com nenhuma palavra-chave): responde SIM se a pergunta menciona
    uma destas culturas/pragas, NAO caso contrário.
    """

    AGRICULTURAL_HINTS = ("couve", "traça", "viveiro", "cebola", "repolho")

    def __init__(self, *args, **kwargs) -> None:
        pass

    def chat(self, messages: list[dict], **kwargs):
        question = messages[-1]["content"].lower()
        verdict = "SIM" if any(hint in question for hint in self.AGRICULTURAL_HINTS) else "NAO"
        return SimpleNamespace(message=SimpleNamespace(content=verdict))


# ---------------------------------------------------------------------------
# Duplos dos módulos de média
# ---------------------------------------------------------------------------

class FakeTranscriber:
    def __init__(self) -> None:
        self.received: list[bytes] = []
        self.text = "Como controlo a lagarta na couve?"

    def transcribe(self, audio_bytes: bytes) -> str:
        self.received.append(audio_bytes)
        return self.text


class FakeImageDescriber:
    def __init__(self) -> None:
        self.received: list[bytes] = []
        self.text = "Folhas de couve com manchas amarelas em forma de V nas bordas."

    def describe(self, image_bytes: bytes) -> str:
        self.received.append(image_bytes)
        return self.text


class FakeSynthesizer:
    AUDIO = b"OggS-fake-voice-reply"

    def __init__(self) -> None:
        self.received: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.received.append(text)
        return self.AUDIO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(monkeypatch) -> RAGPipeline:
    monkeypatch.setattr(pipeline_module, "VectorRetriever", FakeVectorRetriever)
    monkeypatch.setattr(pipeline_module, "SoilRetriever", FakeSoilRetriever)
    monkeypatch.setattr(pipeline_module, "LLaMAGenerator", FakeGenerator)
    monkeypatch.setattr(pipeline_module, "QueryReformulator", FakeReformulator)
    monkeypatch.setattr(domain_guard_module.ollama, "Client", FakeOllamaClient)
    return RAGPipeline(RAGConfig())


@pytest.fixture
def generator(pipeline) -> FakeGenerator:
    return pipeline._generator


@pytest.fixture
def transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def image_describer() -> FakeImageDescriber:
    return FakeImageDescriber()


@pytest.fixture
def synthesizer() -> FakeSynthesizer:
    return FakeSynthesizer()


@pytest.fixture
def twilio(monkeypatch, tmp_path):
    """
    Simula o Twilio: regista as mensagens enviadas ao utilizador e serve os
    ficheiros de média registados em `twilio.media`. Por omissão a validação
    de assinatura está desligada (sem TWILIO_AUTH_TOKEN), como em
    desenvolvimento local; os testes de segurança ligam-na explicitamente.
    """
    state = SimpleNamespace(sent=[], media={}, downloaded=[])

    def fake_send(to: str, from_: str, body: str, media_url: str | None = None) -> None:
        state.sent.append({"to": to, "from_": from_, "body": body, "media_url": media_url})

    def fake_download(url: str) -> bytes:
        state.downloaded.append(url)
        return state.media[url]

    monkeypatch.setattr(webhook_router, "_send_whatsapp_reply", fake_send)
    monkeypatch.setattr(webhook_router, "_download_media", fake_download)
    monkeypatch.setattr(media_router, "_REPLIES_DIR", tmp_path / "tts_replies")
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    return state


@pytest.fixture
def client(pipeline, transcriber, image_describer, synthesizer, twilio):
    # Sem `with TestClient(...)` o lifespan não corre, por isso o RAGPipeline
    # real (e os seus modelos) nunca é carregado.
    app.dependency_overrides[dependencies.get_pipeline] = lambda: pipeline
    app.dependency_overrides[dependencies.get_transcriber] = lambda: transcriber
    app.dependency_overrides[dependencies.get_image_describer] = lambda: image_describer
    app.dependency_overrides[dependencies.get_synthesizer] = lambda: synthesizer
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def send_whatsapp(client):
    """Envia uma mensagem ao webhook como o Twilio faria (form-data)."""

    def _send(
        body: str = "",
        from_: str = "whatsapp:+258840000001",
        media_url: str | None = None,
        media_type: str | None = None,
        headers: dict | None = None,
    ):
        data = {
            "MessageSid": f"SM{time.time_ns()}",
            "From": from_,
            "To": TWILIO_NUMBER,
            "Body": body,
            "NumMedia": "1" if media_url else "0",
        }
        if media_url:
            data["MediaUrl0"] = media_url
            data["MediaContentType0"] = media_type
        return client.post("/webhook/whatsapp", data=data, headers=headers or {})

    return _send


def live_tests_enabled() -> bool:
    return os.getenv("AMIGO_LIVE_TESTS") == "1"
