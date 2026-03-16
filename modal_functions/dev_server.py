"""
Servidor local de desenvolvimento — ProECG

FastAPI simples que roda o mesmo pipeline sem precisar do Modal.
Usado para testar localmente antes de deployar.

Uso:
    cd modal_functions
    pip install fastapi uvicorn
    uvicorn dev_server:app --reload --port 8000

Endpoints:
    POST /analyze  — mesmo contrato do Modal endpoint
    GET  /health   — health check
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Garantir que o diretório pipeline está no path
sys.path.insert(0, str(Path(__file__).resolve().parent))

app = FastAPI(
    title="ProECG Dev Server",
    description="Servidor local para testar o pipeline de análise de ECG",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: Quando o ECG-Digitiser estiver integrado, mudar para False
USE_PLACEHOLDER = True


class AnalyzeRequest(BaseModel):
    image_url: str
    token: str | None = None


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Analisa ECG a partir da URL da imagem.

    Mesmo contrato do endpoint Modal (docs/ARQUITETURA.md).
    """
    from pipeline.orchestrator import analyze as run_pipeline
    return run_pipeline(request.image_url, use_placeholder=USE_PLACEHOLDER)


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok", "service": "proecg-ecg-analyzer-dev"}


@app.get("/")
def root() -> dict:
    """Info."""
    return {
        "service": "ProECG Dev Server",
        "docs": "/docs",
        "endpoints": {
            "POST /analyze": "Analisar ECG (image_url)",
            "GET /health": "Health check",
        },
    }
