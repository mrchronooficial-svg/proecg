"""
Modal function principal — ProECG

Ponto de entrada serverless que o Next.js chama via HTTP POST.
Deploy: modal deploy analyze.py
Dev:    modal serve analyze.py
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Imagem Docker com todas as dependências do pipeline
# ---------------------------------------------------------------------------

ecg_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs")
    .pip_install(
        "torch>=2.0,<3",
        "torchvision>=0.17,<1",
        "numpy>=1.24,<2",
        "scipy>=1.10,<2",
        "neurokit2>=0.2,<1",
        "opencv-python-headless>=4.8,<5",
        "requests>=2.28,<3",
        "Pillow>=9.0,<11",
        "scikit-image>=0.21,<1",
        "yacs>=0.1.8,<1",
        "torch-tps>=1.0,<2",
        "fastapi[standard]",
    )
    .run_commands(
        "git lfs install",
        "git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git /root/open_ecg_digitizer",
        "cd /root/open_ecg_digitizer && git lfs pull",
    )
    .add_local_dir("pipeline", remote_path="/root/pipeline")
    .add_local_dir("models", remote_path="/root/models")
)

app = modal.App("proecg-ecg-analyzer", image=ecg_image)

# ---------------------------------------------------------------------------
# Volume para os pesos do modelo (persistente entre deploys)
# ---------------------------------------------------------------------------

model_volume = modal.Volume.from_name("proecg-models", create_if_missing=True)

# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

USE_PLACEHOLDER = False


@app.function(
    gpu="T4",  # nnU-Net segmentação roda significativamente mais rápido com GPU
    timeout=120,
    memory=2048,
    volumes={"/root/models": model_volume},
    secrets=[modal.Secret.from_name("proecg-secrets")],
)
def analyze_ecg(image_url: str) -> dict:
    """Analisa ECG a partir da URL da imagem.

    Chamada internamente pelo web endpoint.
    """
    import sys
    sys.path.insert(0, "/root")

    from pipeline.orchestrator import analyze
    return analyze(image_url, use_placeholder=USE_PLACEHOLDER)


# ---------------------------------------------------------------------------
# Web endpoint — POST /analyze
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",  # nnU-Net segmentação roda significativamente mais rápido com GPU
    timeout=120,
    memory=2048,
    volumes={"/root/models": model_volume},
    secrets=[modal.Secret.from_name("proecg-secrets")],
)
@modal.web_endpoint(method="POST", docs=True)
def analyze_endpoint(request: dict) -> dict:
    """Endpoint HTTP POST que o Next.js chama.

    Request body:
    {
        "image_url": "https://r2.proecg.com/ecgs/abc123.jpg",
        "token": "optional-auth-token"
    }

    Response: contrato JSON definido em docs/ARQUITETURA.md
    """
    import os
    import sys
    sys.path.insert(0, "/root")

    # Validar token (se configurado)
    expected_token = os.environ.get("MODAL_TOKEN")
    if expected_token:
        provided_token = request.get("token", "")
        if provided_token != expected_token:
            return {
                "success": False,
                "error": "Token inválido",
                "measurements": {},
                "findings": [],
                "diagnoses": [],
                "report_text": "",
                "processing_time_ms": 0,
            }

    # Validar image_url
    image_url = request.get("image_url")
    if not image_url:
        return {
            "success": False,
            "error": "image_url é obrigatório",
            "measurements": {},
            "findings": [],
            "diagnoses": [],
            "report_text": "",
            "processing_time_ms": 0,
        }

    from pipeline.orchestrator import analyze
    return analyze(image_url, use_placeholder=USE_PLACEHOLDER)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.function()
@modal.web_endpoint(method="GET", docs=True)
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "proecg-ecg-analyzer"}
