"""
Modal function principal — ProECG

Ponto de entrada serverless que o Next.js chama via HTTP POST.
Deploy: modal deploy analyze.py
Dev:    modal serve analyze.py

Pipeline completo:
  1. ECGDigitizer  (foto → 12 sinais µV)
  2. measure_ecg   (sinais → medições)
  3. apply_rules   (medições → diagnósticos por regras)
  4. classify_ecg  (sinais → diagnósticos por CNN)
  5. generate_report (medições + regras + CNN → laudo)
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Imagem Docker com todas as dependências do pipeline
# ---------------------------------------------------------------------------

ecg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0,<3",
        "torchvision>=0.17,<1",
        "numpy>=1.24,<2",
        "scipy>=1.10,<2",
        "neurokit2>=0.2,<1",
        "opencv-python-headless>=4.8,<5",
        "requests>=2.28,<3",
        "Pillow>=9.0,<11",
        "easyocr>=1.7,<2",
        "fastapi[standard]",
        "anthropic>=0.40.0",
    )
    .add_local_dir("pipeline", remote_path="/root/pipeline")
    .add_local_dir("models", remote_path="/root/models")
    .add_local_dir("../training/models", remote_path="/root/training/models")
)

app = modal.App("proecg-ecg-analyzer", image=ecg_image)

# ---------------------------------------------------------------------------
# Classe com modelos carregados na inicialização (evita cold-load por chamada)
# ---------------------------------------------------------------------------


@app.cls(
    gpu="T4",
    timeout=60,
    memory=2048,
    secrets=[
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("proecg-secrets"),
    ],
    scaledown_window=300,
)
class ECGAnalyzer:
    """Classe Modal que carrega modelos uma vez e reutiliza entre chamadas."""

    @modal.enter()
    def load_models(self):
        """Carrega modelos na inicialização do container (não a cada chamada)."""
        import sys
        import torch

        sys.path.insert(0, "/root")

        # Pré-carregar modelos em GPU para evitar latência por request
        # 1. CNN classificadora
        try:
            from pipeline.classify import _load_model
            self.cnn_model = _load_model()
            self.cnn_available = True
        except Exception as e:
            print(f"[WARN] CNN nao carregou: {e} — laudo sem CNN")
            self.cnn_model = None
            self.cnn_available = False

        # 2. Dotter UNet (digitalizador)
        try:
            from pathlib import Path
            dotter_path = Path("/root/models/digitizer/dotter_best.pt")
            if dotter_path.exists():
                self.dotter_weights = torch.load(
                    str(dotter_path), map_location="cuda", weights_only=True,
                )
                self.dotter_available = True
            else:
                self.dotter_weights = None
                self.dotter_available = False
                print("[WARN] dotter_best.pt nao encontrado — usando mock")
        except Exception as e:
            print(f"[WARN] Dotter nao carregou: {e} — usando mock")
            self.dotter_weights = None
            self.dotter_available = False

        print("[OK] Modelos carregados. CNN=%s, Dotter=%s"
              % (self.cnn_available, self.dotter_available))

    @modal.method()
    def analyze(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        corners: dict | None = None,
        engine: str = "motor1",
    ) -> dict:
        """Analisa ECG a partir de URL ou base64.

        Args:
            image_url: URL da imagem no Cloudflare R2.
            image_base64: Imagem codificada em base64 (alternativa à URL).
            corners: 4 cantos do papel ECG para correção de perspectiva.
            engine: "motor1" (pipeline numérico) ou "motor2" (Claude Vision).

        Returns:
            dict JSON com laudo completo.
        """
        import base64
        import io
        import signal
        import sys
        import tempfile
        import time

        import cv2
        import numpy as np
        from PIL import Image

        sys.path.insert(0, "/root")

        start_time = time.perf_counter()

        # --- Timeout de 30s para o pipeline ---
        class PipelineTimeout(Exception):
            pass

        def _timeout_handler(signum, frame):
            raise PipelineTimeout("Pipeline excedeu 30s")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(30)

        try:
            # --- 0. Obter imagem ---
            if image_url:
                import requests
                resp = requests.get(image_url, timeout=15)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            elif image_base64:
                img_bytes = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                return _error_response("image_url ou image_base64 obrigatorio", engine, 0)

            # --- 0b. Correção de perspectiva (se corners fornecidos) ---
            if corners is not None:
                from pipeline.orchestrator import _apply_perspective_crop
                image = _apply_perspective_crop(image, corners)

            # --- Roteamento por engine ---
            if engine == "motor2":
                from pipeline.motor2 import motor2_analyze
                result = motor2_analyze(image)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                signal.alarm(0)
                from pipeline.orchestrator import _sanitize_for_json
                return _sanitize_for_json({**result, "processing_time_ms": elapsed_ms})

            # --- Motor 1: pipeline numérico completo ---
            from pipeline.orchestrator import (
                _digitizer_signals_to_array,
                _extract_red_flags,
                _fix_orientation,
                _sanitize_for_json,
                _strip_scores,
                TARGET_FS,
            )
            from pipeline.digitize.ecg_digitizer import ECGDigitizer, LEAD_ORDER
            from pipeline.measure import measure_ecg
            from pipeline.rules import apply_clinical_rules
            from pipeline.report import generate_report

            # Salvar imagem em temp para o digitizer
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_bgr = _fix_orientation(img_bgr)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, img_bgr)
                oriented_path = tmp.name

            # --- 1. Digitalizar ---
            use_mock = not self.dotter_available
            digitizer = ECGDigitizer(use_mock=use_mock)
            dig_result = digitizer.run(oriented_path)

            signals = dig_result["signals"]
            dig_sr = dig_result["sampling_rate"]
            px_per_mm = dig_result["px_per_mm"]
            grid_shape = dig_result["grid_shape"]
            quality = dig_result["quality_flags"]

            signal_12lead = _digitizer_signals_to_array(signals, dig_sr)

            leads_active = sum(1 for i in range(12) if np.any(signal_12lead[i] != 0))

            # --- 2. Medições ---
            measurements = measure_ecg(signal_12lead, fs=TARGET_FS)
            measurements["quality_flags"] = quality

            # --- 3. Regras clínicas ---
            rule_findings = apply_clinical_rules(measurements)

            # --- 4. CNN (graceful degradation) ---
            cnn_findings = []
            if self.cnn_available:
                try:
                    from pipeline.classify import classify_ecg
                    cnn_findings = classify_ecg(signal_12lead)
                except Exception as e:
                    print(f"[WARN] CNN falhou em runtime: {e} — laudo sem CNN")

            # --- 5. Laudo ---
            report = generate_report(measurements, rule_findings, cnn_findings)
            red_flags = _extract_red_flags(report["diagnoses"])

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            signal.alarm(0)

            response = {
                "success": True,
                "engine": "motor1",
                "report_text": report["report_text"],
                "measurements": {
                    "heart_rate": measurements.get("heart_rate"),
                    "heart_rate_unit": "bpm",
                    "axis": measurements.get("axis"),
                    "axis_unit": "graus",
                    "pr_interval": measurements.get("pr_interval"),
                    "pr_unit": "ms",
                    "qrs_duration": measurements.get("qrs_duration"),
                    "qrs_unit": "ms",
                    "qt_interval": measurements.get("qt_interval"),
                    "qt_unit": "ms",
                    "qtc_bazett": measurements.get("qtc_bazett"),
                    "qtc_unit": "ms",
                    "rhythm": measurements.get("rhythm"),
                    "p_wave_present": measurements.get("p_wave_present"),
                    "st_segment": measurements.get("st_segment", {}),
                },
                "findings": _strip_scores(report["findings"]),
                "diagnoses": report["diagnoses"],
                "red_flags": _strip_scores(red_flags),
                "metadata": {
                    "layout": quality.get("layout"),
                    "px_per_mm": px_per_mm,
                    "grid_shape": grid_shape,
                    "grid_points_detected": quality.get("grid_points_detected"),
                    "leads_active": leads_active,
                    "undistortion": quality.get("undistortion"),
                    "digitizer_sampling_rate": dig_sr,
                    "output_sampling_rate": TARGET_FS,
                    "cnn_available": self.cnn_available,
                    "dotter_available": self.dotter_available,
                },
                "processing_time_ms": elapsed_ms,
            }

            return _sanitize_for_json(response)

        except PipelineTimeout:
            signal.alarm(0)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return _error_response(
                "Pipeline excedeu timeout de 30 segundos", engine, elapsed_ms,
            )
        except Exception as e:
            signal.alarm(0)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return _error_response(
                f"{type(e).__name__}: {str(e)}", engine, elapsed_ms,
            )


# ---------------------------------------------------------------------------
# Helper para respostas de erro
# ---------------------------------------------------------------------------

def _error_response(error: str, engine: str, elapsed_ms: int) -> dict:
    return {
        "success": False,
        "engine": engine,
        "error": error,
        "report_text": "",
        "measurements": {},
        "findings": [],
        "diagnoses": [],
        "red_flags": [],
        "metadata": {},
        "processing_time_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Web endpoint — POST /analyze
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=60,
    memory=2048,
    secrets=[
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("proecg-secrets"),
    ],
)
@modal.fastapi_endpoint(method="POST", docs=True)
def analyze_endpoint(request: dict) -> dict:
    """Endpoint HTTP POST que o Next.js chama.

    Request body (opção 1 — URL):
    {
        "image_url": "https://r2.proecg.com/ecgs/abc123.jpg",
        "token": "optional-auth-token",
        "engine": "motor1"
    }

    Request body (opção 2 — base64 direto do celular):
    {
        "image_base64": "/9j/4AAQ...",
        "token": "optional-auth-token",
        "engine": "motor1"
    }

    Response: JSON com laudo completo (contrato definido no orchestrator.py)
    """
    import os

    # Validar token (se configurado)
    expected_token = os.environ.get("MODAL_TOKEN")
    if expected_token:
        provided_token = request.get("token", "")
        if provided_token != expected_token:
            return _error_response("Token invalido", "", 0)

    # Validar que tem pelo menos uma fonte de imagem
    image_url = request.get("image_url")
    image_base64 = request.get("image_base64")

    if not image_url and not image_base64:
        return _error_response("image_url ou image_base64 obrigatorio", "", 0)

    corners = request.get("corners")
    engine = request.get("engine", "motor1")

    # Chamar a classe com modelos pré-carregados
    analyzer = ECGAnalyzer()
    return analyzer.analyze.remote(
        image_url=image_url,
        image_base64=image_base64,
        corners=corners,
        engine=engine,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.function()
@modal.fastapi_endpoint(method="GET", docs=True)
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "proecg-ecg-analyzer", "version": "1.0"}
