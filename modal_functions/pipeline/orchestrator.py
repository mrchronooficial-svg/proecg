"""
Orquestrador do pipeline de análise de ECG — ProECG

Ponto de entrada que conecta todas as etapas em ordem:
  1. Baixar imagem do R2 (via URL)
  2. Digitalizar (Open-ECG-Digitizer) → sinal 12 derivações
  3. Medir intervalos → measurements
  4. Aplicar regras clínicas → rule_findings
  5. Classificar CNN → cnn_findings
  6. Montar laudo → report
  7. Retornar JSON no formato definido em docs/ARQUITETURA.md

Contrato JSON de resposta:
{
  "success": true,
  "measurements": { ... },
  "findings": [ ... ],
  "diagnoses": [ ... ],
  "report_text": "...",
  "processing_time_ms": 1850
}
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

from .digitize import digitize_ecg
from .measure import measure_ecg
from .rules import apply_clinical_rules
from .classify import classify_ecg
from .report import generate_report


# ---------------------------------------------------------------------------
# Etapa 1: Baixar imagem
# ---------------------------------------------------------------------------

def _download_image(image_url: str, timeout: int = 30) -> Image.Image:
    """Baixa imagem do R2 via URL e retorna como PIL Image."""
    response = requests.get(image_url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


# ---------------------------------------------------------------------------
# Etapa 2: Digitalizar (Open-ECG-Digitizer)
# ---------------------------------------------------------------------------

def _digitize_ecg(image: Image.Image, use_placeholder: bool = False) -> np.ndarray:
    """Converte foto de ECG em papel → sinal digital de 12 derivações.

    Usa Open-ECG-Digitizer (U-Net segmentation) com fallback clássico.
    Retorna array numpy (12, N) com o sinal a 500 Hz.

    Args:
        image: PIL Image do ECG.
        use_placeholder: se True, retorna sinal sintético para dev/teste.
    """
    if use_placeholder:
        return _digitize_ecg_placeholder()

    return digitize_ecg(image)


def _digitize_ecg_placeholder() -> np.ndarray:
    """Gera sinal ECG sintético de 12 derivações para dev/teste.

    Simula um ECG sinusal normal a 500 Hz por 10 segundos.
    NÃO usar em produção — apenas para testar o pipeline.
    """
    fs = 500
    duration = 10  # segundos
    n_samples = fs * duration
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Frequência cardíaca simulada: ~75 bpm (1.25 Hz)
    hr_hz = 1.25

    signal = np.zeros((12, n_samples), dtype=np.float64)

    for lead_idx in range(12):
        # Componente QRS (pico estreito)
        qrs = np.zeros(n_samples)
        for beat in range(int(duration * hr_hz)):
            beat_center = int((beat / hr_hz) * fs)
            if beat_center < n_samples:
                # Gaussian pulse para simular QRS
                width = int(0.04 * fs)  # 40ms
                start = max(0, beat_center - width)
                end = min(n_samples, beat_center + width)
                amplitude = 1.0 + 0.3 * (lead_idx % 3)  # variar por derivação
                for i in range(start, end):
                    qrs[i] = amplitude * np.exp(-0.5 * ((i - beat_center) / (width / 3)) ** 2)

        # Componente T (onda mais larga após QRS)
        t_wave = np.zeros(n_samples)
        for beat in range(int(duration * hr_hz)):
            beat_center = int((beat / hr_hz) * fs) + int(0.25 * fs)  # 250ms após R
            if beat_center < n_samples:
                width = int(0.08 * fs)  # 80ms
                start = max(0, beat_center - width)
                end = min(n_samples, beat_center + width)
                amplitude = 0.3
                for i in range(start, end):
                    t_wave[i] = amplitude * np.exp(-0.5 * ((i - beat_center) / (width / 2)) ** 2)

        # Ruído basal
        noise = 0.02 * np.random.randn(n_samples)

        signal[lead_idx] = qrs + t_wave + noise

    return signal


# ---------------------------------------------------------------------------
# Etapa 3-6: Pipeline completo
# ---------------------------------------------------------------------------

def _build_measurements_response(measurements: dict) -> dict:
    """Formata measurements para o contrato JSON de resposta."""
    return {
        "heart_rate": measurements.get("heart_rate"),
        "heart_rate_unit": measurements.get("heart_rate_unit", "bpm"),
        "axis": measurements.get("axis"),
        "axis_unit": measurements.get("axis_unit", "°"),
        "pr_interval": measurements.get("pr_interval"),
        "pr_unit": measurements.get("pr_unit", "ms"),
        "qrs_duration": measurements.get("qrs_duration"),
        "qrs_unit": measurements.get("qrs_unit", "ms"),
        "qt_interval": measurements.get("qt_interval"),
        "qt_unit": measurements.get("qt_unit", "ms"),
        "qtc_bazett": measurements.get("qtc_bazett"),
        "qtc_unit": measurements.get("qtc_unit", "ms"),
        "rhythm": measurements.get("rhythm"),
    }


def _strip_scores(findings: list[dict]) -> list[dict]:
    """Remove o campo 'score' dos achados (não mostrar confiança ao médico)."""
    return [
        {k: v for k, v in f.items() if k != "score"}
        for f in findings
    ]


def _sanitize_for_json(obj: Any) -> Any:
    """Converte tipos numpy para tipos nativos Python (JSON-safe).

    numpy.float64, numpy.int64, numpy.bool_ etc não são serializáveis
    por json.dumps/FastAPI. NaN e Inf viram None.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Função principal — ponto de entrada
# ---------------------------------------------------------------------------

def _apply_perspective_crop(image: Image.Image, corners: dict) -> Image.Image:
    """Aplica correção de perspectiva e crop baseado nos 4 cantos do usuário.

    Args:
        image: PIL Image original
        corners: dict com keys "top_left", "top_right", "bottom_right", "bottom_left"
                 Cada valor é [x, y] em pixels da imagem original

    Returns:
        PIL Image corrigida e cropada (retangular)
    """
    import cv2
    img_np = np.array(image)

    src_pts = np.float32([
        corners["top_left"],
        corners["top_right"],
        corners["bottom_right"],
        corners["bottom_left"],
    ])

    width_top = np.linalg.norm(src_pts[1] - src_pts[0])
    width_bottom = np.linalg.norm(src_pts[2] - src_pts[3])
    dst_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(src_pts[3] - src_pts[0])
    height_right = np.linalg.norm(src_pts[2] - src_pts[1])
    dst_height = int(max(height_left, height_right))

    if dst_width < 100 or dst_height < 100:
        logger.warning("Perspective crop: dimensões muito pequenas (%dx%d), ignorando",
                       dst_width, dst_height)
        return image

    dst_pts = np.float32([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, M, (dst_width, dst_height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

    logger.info("Perspective crop: %dx%d → %dx%d", img_np.shape[1], img_np.shape[0],
                dst_width, dst_height)
    return Image.fromarray(warped)


def analyze(
    image_url: str,
    corners: dict | None = None,
    use_placeholder: bool = False,
    engine: str = "motor2",
) -> dict[str, Any]:
    """Pipeline completo de análise de ECG.

    Args:
        image_url: URL da imagem no Cloudflare R2.
        corners: dict com 4 cantos do papel ECG (opcional).
        use_placeholder: se True, usa sinal sintético (Motor 1 only).
        engine: "motor2" (Claude Vision, default) ou "motor1" (pipeline numérico).

    Returns:
        dict no formato do contrato JSON.
    """
    start_time = time.perf_counter()

    try:
        # 1. Baixar imagem
        image = _download_image(image_url)

        # 1b. Aplicar crop de perspectiva se cantos fornecidos
        if corners is not None:
            image = _apply_perspective_crop(image, corners)

        # 2. Escolher motor
        if engine == "motor2":
            from .motor2 import motor2_analyze
            result = motor2_analyze(image)
        else:
            # Motor 1 — pipeline numérico
            signal_12lead = _digitize_ecg(image, use_placeholder=use_placeholder)
            measurements = measure_ecg(signal_12lead, fs=500)
            rule_findings = apply_clinical_rules(measurements)
            cnn_findings = classify_ecg(signal_12lead)
            report = generate_report(measurements, rule_findings, cnn_findings)
            result = {
                "success": True,
                "engine": "motor1",
                "measurements": _build_measurements_response(measurements),
                "findings": _strip_scores(report["findings"]),
                "diagnoses": report["diagnoses"],
                "report_text": report["report_text"],
            }

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        response = _sanitize_for_json({
            **result,
            "processing_time_ms": elapsed_ms,
        })

        logger.info("Pipeline (%s) concluído em %dms", engine, elapsed_ms)
        return response

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        response = {
            "success": False,
            "error": f"Erro ao processar ECG: {type(e).__name__}: {str(e)}",
            "engine": engine,
            "measurements": {},
            "findings": [],
            "diagnoses": [],
            "report_text": "",
            "processing_time_ms": elapsed_ms,
        }
        logger.error("Pipeline (%s) erro em %dms: %s", engine, elapsed_ms, e, exc_info=True)
        return response
