"""
Ponto de entrada da API — Amigo do Agricultor.

Arranque:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Para expor localmente ao Twilio (ngrok):
    ngrok http 8000
    → copiar o URL https://xxxx.ngrok.io/webhook/whatsapp para o painel Twilio
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Garante que src/ está no path (necessário ao correr com uvicorn a partir da raiz)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.dependencies import get_pipeline
from api.routers import chat, health, webhook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ciclo de vida da aplicação
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa o RAGPipeline uma única vez no arranque do servidor."""
    logger.info("A inicializar o RAGPipeline...")
    try:
        pipeline = get_pipeline()
        health_status = pipeline.health_check()
        logger.info("RAGPipeline pronto: %s", health_status)
    except Exception as exc:
        logger.error("Falha ao inicializar o RAGPipeline: %s", exc)
    yield
    logger.info("Servidor a encerrar.")


# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Amigo do Agricultor — API",
    description=(
        "API do assistente virtual agrícola para pequenos agricultores "
        "da Província de Maputo, Moçambique. "
        "Alimentado por RAG com LLaMA, ChromaDB e dados de solo locais."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(webhook.router)


# ---------------------------------------------------------------------------
# Rota raiz
# ---------------------------------------------------------------------------

@app.get("/", tags=["Geral"], summary="Informação da API")
def root() -> JSONResponse:
    return JSONResponse({
        "name": "Amigo do Agricultor",
        "version": "1.0.0",
        "description": "Assistente virtual de apoio técnico agrícola — Moçambique",
        "docs": "/docs",
        "endpoints": {
            "health":           "GET  /health",
            "query":            "POST /query",
            "whatsapp_webhook": "POST /webhook/whatsapp",
        },
    })
