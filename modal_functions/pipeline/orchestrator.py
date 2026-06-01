"""
Orquestrador do pipeline de analise de ECG -- ProECG

Dois modos de operacao:

  analyze_from_file(path)  -- pipeline local: foto no disco -> laudo
  analyze(image_url)       -- pipeline web:  URL no R2 -> laudo

Pipeline local (Motor 1 numerico):
  1. ECGDigitizer  (foto -> 12 sinais uV)
  2. measure_ecg   (sinais -> medicoes)
  3. apply_rules   (medicoes -> diagnosticos por regras)
  4. classify_ecg  (sinais -> diagnosticos por CNN)
  5. generate_report (medicoes + regras + CNN -> laudo)

Contrato JSON de resposta:
{
  "success": true,
  "engine": "motor1",
  "report_text": "...",
  "measurements": { ... },
  "findings": [ ... ],
  "diagnoses": [ ... ],
  "red_flags": [ ... ],
  "metadata": { "layout": "6x2+1", "px_per_mm": 15.5, ... },
  "processing_time_ms": 3200
}
"""

from __future__ import annotations

import io
import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.signal import resample

logger = logging.getLogger(__name__)

from .digitize.ecg_digitizer import ECGDigitizer, LEAD_ORDER
from .measure import measure_ecg
from .rules import apply_clinical_rules
from .classify import (
    classify_ecg,
    EXPECTED_LENGTH,
    RED_FLAG_CODES as CNN_RED_FLAG_CODES,
)
from .report import generate_frontend_report, generate_report, RED_FLAG_CODES

# Frequencia alvo para medicoes (CNN v5b roda em 400Hz e auto-resamplea internamente)
TARGET_FS = 500


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: Any) -> Any:
    """Converte tipos numpy para tipos nativos Python (JSON-safe)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


_INTERNAL_FIELDS = ("score", "threshold", "cnn_score")


def _strip_scores(findings: list[dict]) -> list[dict]:
    """Remove campos internos (score/threshold/cnn_score) — nunca mostrar
    confiança nem limiar ao médico."""
    return [
        {k: v for k, v in f.items() if k not in _INTERNAL_FIELDS}
        for f in findings
    ]


def _extract_red_flags(diagnoses: list[dict]) -> list[dict]:
    """Extrai diagnosticos que sao red-flags (urgencia maxima).

    Reconhece:
      - is_red_flag=True (vem do classify v5b)
      - Códigos do CNN v5b: STE, 3AVB, 2AVB, WPW, LQT, MI
      - Códigos das regras: report.RED_FLAG_CODES
      - Prefixos legados: sca_, tv_, bav_total, hipercalemia_grave
    """
    red = []
    for d in diagnoses:
        if d.get("is_red_flag"):
            red.append(d)
            continue
        code = d.get("code", "")
        if (
            code in RED_FLAG_CODES
            or code in CNN_RED_FLAG_CODES
            or any(
                code.startswith(p)
                for p in ("sca_", "tv_", "bav_total", "hipercalemia_grave")
            )
        ):
            red.append(d)
    return red


def _fix_orientation(img: np.ndarray) -> np.ndarray:
    """Corrige foto portrait -> landscape se necessario."""
    h, w = img.shape[:2]
    if h > w * 1.2:
        logger.info("Rotacionando 90 CCW (portrait %dx%d -> landscape)", w, h)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _digitizer_signals_to_array(
    signals: dict[str, np.ndarray],
    sampling_rate: int,
) -> np.ndarray:
    """Converte dict de sinais do ECGDigitizer -> array (12, 5000) a 500Hz em mV."""
    target_len = int(10.0 * TARGET_FS)  # 5000
    result = np.zeros((12, target_len), dtype=np.float64)

    for i, lead_name in enumerate(LEAD_ORDER):
        sig = signals.get(lead_name)
        if sig is None:
            continue

        # uV -> mV
        sig_mv = sig / 1000.0

        # Remover NaN
        valid = ~np.isnan(sig_mv)
        if valid.sum() < 2:
            continue

        sig_clean = np.nan_to_num(sig_mv, nan=0.0)

        # Resample se sampling_rate != TARGET_FS
        if sampling_rate != TARGET_FS and len(sig_clean) > 2:
            n_target = int(len(sig_clean) * TARGET_FS / sampling_rate)
            if n_target > 1:
                sig_clean = resample(sig_clean, n_target)

        # Encaixar no array de saida
        n = min(len(sig_clean), target_len)
        result[i, :n] = sig_clean[:n]
        if n < target_len and n > 0:
            result[i, n:] = sig_clean[n - 1]

    return result


# ---------------------------------------------------------------------------
# Pipeline local: arquivo -> laudo
# ---------------------------------------------------------------------------

def analyze_from_file(
    image_path: str,
    use_mock: bool = True,
) -> dict[str, Any]:
    """Pipeline completo: foto no disco -> laudo.

    Args:
        image_path: caminho da imagem (JPEG/PNG).
        use_mock: se True, usa Dotter/Leader mock (sem UNet treinado).

    Returns:
        dict JSON-safe com laudo completo.
    """
    start_time = time.perf_counter()

    try:
        # --- 0. Carregar e orientar ---
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Nao foi possivel carregar: {image_path}")
        img = _fix_orientation(img)

        # Salvar temp para o digitizer (que espera path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            oriented_path = tmp.name

        # --- 1. Digitalizar ---
        logger.info("Etapa 1/5: Digitalizacao")
        digitizer = ECGDigitizer(use_mock=use_mock)
        dig_result = digitizer.run(oriented_path)

        signals = dig_result["signals"]
        dig_sr = dig_result["sampling_rate"]
        px_per_mm = dig_result["px_per_mm"]
        grid_shape = dig_result["grid_shape"]
        quality = dig_result["quality_flags"]

        # Converter para array (12, 5000) a 500Hz em mV
        signal_12lead = _digitizer_signals_to_array(signals, dig_sr)

        leads_active = sum(1 for i in range(12) if np.any(signal_12lead[i] != 0))
        logger.info("Digitalizacao: %d/12 leads ativos, layout=%s",
                     leads_active, quality.get("layout", "?"))

        # --- 2. Medicoes ---
        logger.info("Etapa 2/5: Medicoes")
        measurements = measure_ecg(signal_12lead, fs=TARGET_FS)
        # Injetar quality_flags para o report
        measurements["quality_flags"] = quality

        # --- 3. Regras clinicas ---
        logger.info("Etapa 3/5: Regras clinicas")
        rule_findings = apply_clinical_rules(measurements)

        # --- 4. CNN ---
        logger.info("Etapa 4/5: CNN classificacao")
        try:
            cnn_findings = classify_ecg(signal_12lead)
        except Exception as e:
            logger.warning("CNN falhou: %s -- continuando sem CNN", e)
            cnn_findings = []

        # --- 5. Laudo (versão frontend com red flags + severidade) ---
        logger.info("Etapa 5/5: Geracao do laudo")
        front = generate_frontend_report(measurements, rule_findings, cnn_findings)
        report = {
            "report_text": front["text_report"],
            "findings": front.get("_findings_raw", []),
            "diagnoses": front["diagnoses"],
        }
        red_flags_text = front.get("red_flags", [])

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

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
            "red_flags": red_flags_text,
            "severity": front.get("severity", "normal"),
            "warnings": front.get("warnings", []),
            "metadata": {
                "layout": quality.get("layout"),
                "px_per_mm": px_per_mm,
                "grid_shape": grid_shape,
                "grid_points_detected": quality.get("grid_points_detected"),
                "leads_active": leads_active,
                "undistortion": quality.get("undistortion"),
                "digitizer_sampling_rate": dig_sr,
                "output_sampling_rate": TARGET_FS,
            },
            "processing_time_ms": elapsed_ms,
        }

        logger.info("Pipeline concluido em %dms", elapsed_ms)
        return _sanitize_for_json(response)

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error("Pipeline erro em %dms: %s", elapsed_ms, e, exc_info=True)
        return {
            "success": False,
            "engine": "motor1",
            "error": f"{type(e).__name__}: {str(e)}",
            "report_text": "",
            "measurements": {},
            "findings": [],
            "diagnoses": [],
            "red_flags": [],
            "metadata": {},
            "processing_time_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Pipeline web: URL -> laudo (mantido para compatibilidade)
# ---------------------------------------------------------------------------

def _download_image(image_url: str, timeout: int = 30) -> Image.Image:
    """Baixa imagem do R2 via URL."""
    response = requests.get(image_url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _apply_perspective_crop(image: Image.Image, corners: dict) -> Image.Image:
    """Aplica correcao de perspectiva baseada nos 4 cantos do usuario."""
    img_np = np.array(image)
    src_pts = np.float32([
        corners["top_left"], corners["top_right"],
        corners["bottom_right"], corners["bottom_left"],
    ])
    width_top = np.linalg.norm(src_pts[1] - src_pts[0])
    width_bottom = np.linalg.norm(src_pts[2] - src_pts[3])
    dst_width = int(max(width_top, width_bottom))
    height_left = np.linalg.norm(src_pts[3] - src_pts[0])
    height_right = np.linalg.norm(src_pts[2] - src_pts[1])
    dst_height = int(max(height_left, height_right))

    if dst_width < 100 or dst_height < 100:
        return image

    dst_pts = np.float32([
        [0, 0], [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1], [0, dst_height - 1],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, M, (dst_width, dst_height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def analyze(
    image_url: str,
    corners: dict | None = None,
    use_placeholder: bool = False,
    engine: str = "motor2",
) -> dict[str, Any]:
    """Pipeline web: URL -> laudo.

    Args:
        image_url: URL da imagem no Cloudflare R2.
        corners: 4 cantos do papel ECG (opcional).
        use_placeholder: sinal sintetico para dev/teste.
        engine: "motor2" (Claude Vision) ou "motor1" (pipeline numerico).
    """
    start_time = time.perf_counter()

    try:
        image = _download_image(image_url)

        if corners is not None:
            image = _apply_perspective_crop(image, corners)

        if engine == "motor2":
            from .motor2 import motor2_analyze
            result = motor2_analyze(image)
        else:
            # Motor 1 -- salvar temp e usar analyze_from_file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                result = analyze_from_file(tmp.name, use_mock=True)
                # Ja tem processing_time_ms do analyze_from_file; sobreescrever
                # com o tempo total incluindo download
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                result["processing_time_ms"] = elapsed_ms
                return result

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return _sanitize_for_json({**result, "processing_time_ms": elapsed_ms})

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "success": False, "error": f"{type(e).__name__}: {str(e)}",
            "engine": engine, "report_text": "", "measurements": {},
            "findings": [], "diagnoses": [], "red_flags": [],
            "metadata": {}, "processing_time_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Teste com foto real
# ---------------------------------------------------------------------------

# Para testar este modulo diretamente, execute da raiz do projeto:
#   python -c "
#   from modal_functions.pipeline.orchestrator import analyze_from_file
#   result = analyze_from_file('caminho/da/imagem.jpg')
#   print(result['report_text'])
#   "
