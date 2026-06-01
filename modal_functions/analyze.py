"""
Modal function principal -- ProECG

Pipeline (Motor 1):
  1. pipeline_completo_v1.run_pipeline(img_path, temp_dir)
       -> ecg_13_leads_mv.npy shape (13, 4096) @ 400Hz
  2. measure_ecg(signal[:12], fs=400)
       -> medições (FC, eixo, PR, QRS, QT, QTc, st_segment, etc.)
  3. apply_clinical_rules(measurements)
       -> rule findings
  4. classify_ecg(signal) -- CNN v5b 24 classes com thresholds Youden
       -> cnn findings (com is_red_flag)
  5. generate_frontend_report(measurements, rule_findings, cnn_findings)
       -> JSON estruturado (severity, diagnoses, red_flags, warnings)

Motor 2 (Claude Vision) continua disponível via engine="motor2".

Deploy:  modal deploy analyze.py
Dev:     modal serve analyze.py
Health:  GET /health  →  {status: "ok"}
Análise: POST /analyze body={image_url|image_base64, token, corners?, engine?}
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Imagem Docker
# ---------------------------------------------------------------------------

ecg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        # Núcleo
        "torch>=2.0,<3",
        "torchvision>=0.17,<1",
        "numpy>=1.24,<2",
        "scipy>=1.10,<2",
        "Pillow>=9.0,<11",
        "opencv-python-headless>=4.8,<5",
        "requests>=2.28,<3",
        # Pipeline numérico
        "neurokit2>=0.2,<1",
        "scikit-image>=0.21,<1",  # vendor SignalExtractor
        "PyYAML>=6.0,<7",         # vendor LeadIdentifier (layouts)
        "matplotlib>=3.7,<4",     # pipeline_completo_v1 plots
        "easyocr>=1.7,<2",        # OCR de calibração (vel/ganho)
        # Web / Claude
        "fastapi[standard]",
        "anthropic>=0.40.0",
    )
    # Código do pipeline numérico
    .add_local_dir("pipeline", remote_path="/root/pipeline")
    # Pesos dos modelos (Dotter, Leader, CNN v5b)
    .add_local_dir("models", remote_path="/root/models")
    # Stenhede UNet + LeadIdentifier UNet + layouts YAML (vendor)
    .add_local_dir(
        "vendor",
        remote_path="/root/vendor",
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/.github/**",
            "**/*.md",
            "**/CHANGELOG*",
            "**/assets/**",
        ],
    )
    # Módulo Python compartilhado com o training (sem pesos)
    .add_local_dir("../training/models", remote_path="/root/training/models")
)

app = modal.App("proecg-ecg-analyzer", image=ecg_image)


# ---------------------------------------------------------------------------
# Classe com modelos carregados na inicialização do container
# ---------------------------------------------------------------------------

@app.cls(
    gpu="T4",
    timeout=300,           # 5 min — pipeline completo na 1ª chamada pode passar de 60s
    memory=4096,           # 4 GB (Dotter + Stenhede + LeadID + CNN + EasyOCR)
    secrets=[
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("proecg-secrets"),
    ],
    scaledown_window=300,  # container fica 5 min vivo após última chamada
)
class ECGAnalyzer:
    """Modelo carregado uma vez por container; reutilizado entre chamadas."""

    @modal.enter()
    def load_models(self):
        """Pré-aquece o que dá pra pré-aquecer (CNN v5b é pequeno e barato).

        Dotter / Stenhede / LeadIdentifier / EasyOCR ficam lazy (carregam
        sob demanda) para não estourar memória no cold start.
        """
        import sys
        sys.path.insert(0, "/root")

        # CNN v5b (27 MB)
        try:
            from pipeline.classify import _load_model
            _load_model()
            self.cnn_available = True
            print("[OK] CNN v5b pré-carregada.")
        except Exception as e:
            print(f"[WARN] CNN v5b não carregou: {e} - laudo sem CNN")
            self.cnn_available = False

    @modal.method()
    def analyze(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        corners: dict | None = None,
        engine: str = "motor1",
    ) -> dict:
        """Analisa um ECG via URL no R2 ou imagem base64.

        Args:
            image_url:  URL pública da foto.
            image_base64: alternativa — base64 da foto.
            corners:    4 cantos do papel ECG (correção de perspectiva manual).
            engine:     "motor1" (pipeline numérico) | "motor2" (Claude Vision).
        """
        import base64
        import io
        import shutil
        import sys
        import tempfile
        import time
        import traceback
        from pathlib import Path

        import numpy as np
        from PIL import Image

        sys.path.insert(0, "/root")

        start_time = time.perf_counter()
        temp_dir: Path | None = None

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
                return _error_response(
                    "image_url ou image_base64 obrigatorio", engine, 0,
                )

            # --- 0b. Correção de perspectiva manual (se 4 cantos vieram) ---
            if corners is not None:
                from pipeline.orchestrator import _apply_perspective_crop
                image = _apply_perspective_crop(image, corners)

            # --- Motor 2: Claude Vision (atalho — não roda digitalização) ---
            if engine == "motor2":
                from pipeline.motor2 import motor2_analyze
                from pipeline.orchestrator import _sanitize_for_json
                result = motor2_analyze(image)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                return _sanitize_for_json(
                    {**result, "processing_time_ms": elapsed_ms},
                )

            # --- 1. Salva imagem em arquivo temporário (run_pipeline espera path) ---
            temp_dir = Path(tempfile.mkdtemp(prefix="proecg_"))
            img_path = temp_dir / "input.jpg"
            image.save(str(img_path), "JPEG", quality=95)
            out_dir = temp_dir / "output"

            # --- 2. Roda pipeline_completo_v1 (digitalização end-to-end) ---
            from pipeline.pipeline_completo_v1 import run_pipeline
            rc = run_pipeline(img_path, out_dir)
            if rc != 0:
                raise RuntimeError(f"pipeline_completo_v1 retornou exit={rc}")

            # --- 3. Carrega sinal digitalizado (13, 4096) @ 400Hz ---
            npy_13 = out_dir / "ecg_13_leads_mv.npy"
            npy_12 = out_dir / "ecg_12_leads_mv.npy"

            if npy_13.exists():
                ecg_signal = np.load(npy_13).astype(np.float32)
                has_rhythm = True
            elif npy_12.exists():
                arr12 = np.load(npy_12).astype(np.float32)  # (12, 4096)
                ecg_signal = np.zeros(
                    (13, arr12.shape[1]), dtype=np.float32,
                )
                ecg_signal[:12] = arr12
                ecg_signal[12] = arr12[1]  # duplica lead II como canal 13
                has_rhythm = False
            else:
                raise FileNotFoundError(
                    "pipeline_completo_v1 não gerou nenhum ecg_*_leads_mv.npy",
                )

            fs = 400  # pipeline_completo_v1 reamostra para 400Hz

            # --- 4. Medições (só os 12 leads, sem rhythm strip) ---
            from pipeline.measure import measure_ecg
            measurements = measure_ecg(ecg_signal[:12], fs=fs)

            # --- 5. Regras clínicas ---
            from pipeline.rules import apply_clinical_rules
            rule_findings = apply_clinical_rules(measurements)

            # --- 6. CNN v5b (recebe os 13 canais nativos) ---
            cnn_findings: list[dict] = []
            if self.cnn_available:
                try:
                    from pipeline.classify import classify_ecg
                    cnn_findings = classify_ecg(ecg_signal)
                except Exception as e:
                    print(f"[WARN] CNN runtime falhou: {e}")

            # --- 7. Laudo estruturado pro frontend ---
            from pipeline.report import generate_frontend_report
            from pipeline.orchestrator import _sanitize_for_json, _strip_scores

            front = generate_frontend_report(
                measurements, rule_findings, cnn_findings,
            )

            # --- 8. Monta resposta no contrato esperado por packages/api/src/lib/modal.ts ---
            iv = measurements.get("intervals", {}) or {}
            hr = measurements.get("heart_rate", {}) or {}
            ax = measurements.get("axis", {}) or {}
            pw = measurements.get("p_wave", {}) or {}

            leads_active = int(
                sum(1 for i in range(13) if np.any(ecg_signal[i] != 0))
            )

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            response = {
                "success": True,
                "engine": "motor1",
                "report_text": front.get("text_report", ""),
                "measurements": {
                    "heart_rate": hr.get("mean_bpm"),
                    "heart_rate_unit": "bpm",
                    "axis": ax.get("degrees"),
                    "axis_unit": "graus",
                    "pr_interval": iv.get("pr_ms"),
                    "pr_unit": "ms",
                    "qrs_duration": iv.get("qrs_ms"),
                    "qrs_unit": "ms",
                    "qt_interval": iv.get("qt_ms"),
                    "qt_unit": "ms",
                    "qtc_bazett": iv.get("qtc_ms"),
                    "qtc_unit": "ms",
                    "rhythm": _derive_rhythm_label(measurements),
                    "p_wave_present": pw.get("present"),
                    "st_segment": measurements.get("st_segment", {}),
                },
                "findings": _strip_scores(front.get("_findings_raw", [])),
                "diagnoses": front.get("diagnoses", []),
                # red_flags em formato de objeto (frontend espera .description)
                "red_flags": [
                    {
                        "code": "v5b_red_flag",
                        "description": rf,
                        "source": "system",
                        "leads_affected": [],
                    }
                    for rf in front.get("red_flags", [])
                ],
                "severity": front.get("severity", "normal"),
                "warnings": front.get("warnings", []),
                "metadata": {
                    "layout": measurements.get(
                        "quality_flags", {},
                    ).get("layout") if isinstance(
                        measurements.get("quality_flags"), dict,
                    ) else None,
                    "leads_active": leads_active,
                    "has_rhythm_strip": has_rhythm,
                    "cnn_available": self.cnn_available,
                    "digitizer_sampling_rate": fs,
                    "output_sampling_rate": fs,
                    "pipeline_version": "pipeline_completo_v1+v5b",
                },
                "processing_time_ms": elapsed_ms,
            }

            return _sanitize_for_json(response)

        except Exception as e:
            traceback.print_exc()
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return _error_response(
                f"{type(e).__name__}: {str(e)}", engine, elapsed_ms,
            )
        finally:
            # Limpa temp dir (todos os PNGs/npys intermediários são descartados)
            if temp_dir is not None and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_rhythm_label(measurements: dict) -> str | None:
    """Mesma heurística usada em measure.to_legacy_format."""
    hr = measurements.get("heart_rate", {}) or {}
    pw = measurements.get("p_wave", {}) or {}
    regular = hr.get("regular")
    p_present = bool(pw.get("present"))
    if p_present and regular:
        return "sinus"
    if p_present and regular is False:
        return "sinus"
    if (not p_present) and regular:
        return "regular_sem_p"
    if (not p_present) and (regular is False):
        return "irregular_sem_p"
    return "indeterminado"


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
# Web endpoint -- POST /analyze
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=300,
    memory=4096,
    secrets=[
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("proecg-secrets"),
    ],
)
@modal.fastapi_endpoint(method="POST", docs=True)
def analyze_endpoint(request: dict) -> dict:
    """Endpoint POST chamado pelo Next.js (packages/api/src/lib/modal.ts).

    Request body:
        {
            "image_url":   "https://r2.proecg.com/ecgs/abc123.jpg",   # OU
            "image_base64": "/9j/4AAQ...",
            "token":  "<MODAL_TOKEN>",
            "corners": { ... },                                       # opcional
            "engine": "motor1" | "motor2"                             # opcional
        }

    Response: JSON conforme ModalRawResponse em modal.ts.
    """
    import os

    # Auth opcional
    expected_token = os.environ.get("MODAL_TOKEN")
    if expected_token:
        provided_token = request.get("token", "")
        if provided_token != expected_token:
            return _error_response("Token invalido", "", 0)

    image_url = request.get("image_url")
    image_base64 = request.get("image_base64")
    if not image_url and not image_base64:
        return _error_response(
            "image_url ou image_base64 obrigatorio", "", 0,
        )

    corners = request.get("corners")
    engine = request.get("engine", "motor1")

    analyzer = ECGAnalyzer()
    return analyzer.analyze.remote(
        image_url=image_url,
        image_base64=image_base64,
        corners=corners,
        engine=engine,
    )


# ---------------------------------------------------------------------------
# Health check -- GET /health
# ---------------------------------------------------------------------------

@app.function()
@modal.fastapi_endpoint(method="GET", docs=True)
def health() -> dict:
    """Health check — usado pra confirmar que o app está deploado."""
    return {
        "status": "ok",
        "service": "proecg-ecg-analyzer",
        "version": "2.0",
        "pipeline": "pipeline_completo_v1 + classify v5b",
    }
