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
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

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


# ---------------------------------------------------------------------------
# Função principal — ponto de entrada
# ---------------------------------------------------------------------------

def analyze(image_url: str, use_placeholder: bool = False) -> dict[str, Any]:
    """Pipeline completo de análise de ECG.

    Args:
        image_url: URL da imagem no Cloudflare R2.
        use_placeholder: se True, usa sinal sintético em vez do
            Open-ECG-Digitizer (para dev/teste).

    Returns:
        dict no formato do contrato JSON definido em docs/ARQUITETURA.md:
        {
            "success": bool,
            "measurements": { ... },
            "findings": [ ... ],
            "diagnoses": [ ... ],
            "report_text": str,
            "processing_time_ms": int,
            "error": str (apenas se success=False)
        }
    """
    start_time = time.perf_counter()

    try:
        # 1. Baixar imagem
        image = _download_image(image_url)

        # 2. Digitalizar (foto → sinal 12 derivações)
        signal_12lead = _digitize_ecg(image, use_placeholder=use_placeholder)

        # 3. Medir intervalos
        measurements = measure_ecg(signal_12lead, fs=500)

        # 4. Aplicar regras clínicas
        rule_findings = apply_clinical_rules(measurements)

        # 5. Classificar CNN
        cnn_findings = classify_ecg(signal_12lead)

        # 6. Montar laudo
        report = generate_report(measurements, rule_findings, cnn_findings)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "success": True,
            "measurements": _build_measurements_response(measurements),
            "findings": _strip_scores(report["findings"]),
            "diagnoses": report["diagnoses"],
            "report_text": report["report_text"],
            "processing_time_ms": elapsed_ms,
        }

    except NotImplementedError as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "success": False,
            "error": str(e),
            "measurements": {},
            "findings": [],
            "diagnoses": [],
            "report_text": "",
            "processing_time_ms": elapsed_ms,
        }

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "success": False,
            "error": f"Erro ao processar ECG: {type(e).__name__}: {str(e)}",
            "measurements": {},
            "findings": [],
            "diagnoses": [],
            "report_text": "",
            "processing_time_ms": elapsed_ms,
        }
