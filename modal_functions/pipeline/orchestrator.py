"""
Orquestrador do pipeline de análise de ECG — ProECG

Ponto de entrada que conecta todas as etapas em ordem:
  1. Baixar imagem do R2 (via URL)
  2. Digitalizar (ECG-Digitiser) → sinal 12 derivações
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
# Etapa 2: Digitalizar (ECG-Digitiser)
# ---------------------------------------------------------------------------

def _digitize_ecg(image: Image.Image) -> np.ndarray:
    """Converte foto de ECG em papel → sinal digital de 12 derivações.

    Usa ECG-Digitiser (pré-treinado).
    Retorna array numpy (12, N) com o sinal.

    NOTA: Esta função é um wrapper que será conectado ao ECG-Digitiser
    quando o modelo estiver integrado. Por enquanto, define a interface.
    """
    # TODO: Integrar ECG-Digitiser quando disponível
    # from ecg_digitiser import digitize
    # signal = digitize(image)
    # return signal  # (12, N) numpy array

    # Placeholder: o ECG-Digitiser será integrado aqui.
    # A interface esperada é:
    #   - Entrada: PIL Image (foto do ECG de papel)
    #   - Saída: np.ndarray shape (12, N), onde N é o número de amostras
    #   - Frequência de amostragem: 500 Hz
    #   - Ordem das derivações: DI, DII, DIII, aVR, aVL, aVF, V1-V6
    raise NotImplementedError(
        "ECG-Digitiser ainda não integrado. "
        "Conectar o modelo pré-treinado em _digitize_ecg()."
    )


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

def analyze(image_url: str) -> dict[str, Any]:
    """Pipeline completo de análise de ECG.

    Args:
        image_url: URL da imagem no Cloudflare R2.

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
        signal_12lead = _digitize_ecg(image)

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
