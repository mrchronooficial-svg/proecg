"""
Módulo 4.5 — Calibrador
=======================

Plugado no pipeline DEPOIS do Gridder/Undistortion e ANTES da extração de sinal.

Calcula a escala física (pixels ↔ mm ↔ mV/segundos) da imagem normalizada
usando o grid do papel ECG. Cada par de pontos adjacentes na matriz do
Gridder representa `GRID_MAJOR_MM` (5 mm — linha maior do papel ECG).

Padrão brasileiro (FIXO):
    velocidade do papel = 25 mm/s
    ganho               = 10 mm/mV
    grid menor / maior  = 1 mm / 5 mm

A detecção do pulso de calibração de 1 mV está DESATIVADA — o código foi
mantido comentado abaixo (`_detect_calibration_pulse` etc.) pra reativação
futura caso queiramos detectar ganhos não-padrão (5 ou 20 mm/mV) e validar
a escala vertical com o pulso.

Fórmulas de conversão (output do calibrador):
    amplitude_uV  = (y_baseline - y_pixel) * uv_per_pixel
    tempo_s       = x_pixel / (px_per_mm * paper_speed_mm_s)
    sampling_rate = px_per_mm * paper_speed_mm_s            (amostras/s)
    uv_per_pixel  = 1000 / (px_per_mm * gain_mm_per_mV)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .constants import (
    GAIN_DEFAULT,
    GRID_MAJOR_MM,
    PAPER_SPEED_DEFAULT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Calibração pelo grid
# ---------------------------------------------------------------------------

def _measure_axis_spacing(
    grid_matrix: np.ndarray, axis: str
) -> tuple[Optional[float], int]:
    """Mede espaçamento mediano entre pontos adjacentes ao longo de um eixo.

    Args:
        grid_matrix: (R, C, 2) — grid[r, c] = (x, y), NaN = ausente.
        axis: "h" (horizontal: pares (r,c)→(r,c+1)) ou "v" (vertical: (r,c)→(r+1,c)).

    Returns:
        (mediana_em_pixels, n_pares_validos). None se não houver pares válidos.
    """
    n_rows, n_cols, _ = grid_matrix.shape
    deltas: list[float] = []

    if axis == "h":
        for r in range(n_rows):
            for c in range(n_cols - 1):
                p0 = grid_matrix[r, c]
                p1 = grid_matrix[r, c + 1]
                if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
                    continue
                d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                deltas.append(d)
    elif axis == "v":
        for c in range(n_cols):
            for r in range(n_rows - 1):
                p0 = grid_matrix[r, c]
                p1 = grid_matrix[r + 1, c]
                if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
                    continue
                d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                deltas.append(d)
    else:
        raise ValueError(f"axis deve ser 'h' ou 'v', recebido: {axis!r}")

    if len(deltas) < 3:
        return None, len(deltas)

    arr = np.asarray(deltas)
    # Filtrar outliers: ±35% da mediana inicial
    med0 = float(np.median(arr))
    keep = arr[(arr > med0 * 0.65) & (arr < med0 * 1.35)]
    if len(keep) < 3:
        return med0, len(deltas)
    return float(np.median(keep)), int(len(keep))


# ---------------------------------------------------------------------------
# 2. Função pública
# ---------------------------------------------------------------------------

def calibrate(
    grid_matrix: np.ndarray,
    normalized_image: Optional[np.ndarray] = None,  # mantido pra compat. de assinatura
    paper_speed_mm_s: float = float(PAPER_SPEED_DEFAULT),
    gain_mm_per_mV: float = float(GAIN_DEFAULT),
    n_lead_bands: int = 3,  # idem
) -> dict:
    """Calcula calibração espacial e elétrica da imagem normalizada.

    Ganho e velocidade do papel são FIXOS no padrão brasileiro
    (10 mm/mV, 25 mm/s). O `px_per_mm` é derivado do espaçamento entre
    pontos do grid em ambos os eixos.

    Args:
        grid_matrix: (R, C, 2) com pontos do Gridder. NaN para pontos ausentes.
                     Adjacência representa GRID_MAJOR_MM (5 mm).
        normalized_image: aceito mas ignorado (detecção de pulso desativada).
        paper_speed_mm_s: velocidade do papel (default 25 mm/s — padrão BR).
        gain_mm_per_mV: ganho (default 10 mm/mV — padrão BR).
        n_lead_bands: aceito mas ignorado (detecção de pulso desativada).

    Returns:
        dict com px_per_mm_h/v, px_per_mm, paper_speed, gain, sampling_rate,
        uv_per_pixel e warnings. Veja docstring do módulo para fórmulas.

    Raises:
        ValueError: se o grid não tem pares suficientes pra estimar escala.
    """
    del normalized_image, n_lead_bands  # não usados (pulse detection off)

    warnings_: list[str] = []

    if grid_matrix is None or grid_matrix.ndim != 3 or grid_matrix.shape[2] != 2:
        raise ValueError(
            "grid_matrix inválido: esperado (R, C, 2), recebido %s"
            % (None if grid_matrix is None else grid_matrix.shape,)
        )
    if grid_matrix.shape[0] < 2 or grid_matrix.shape[1] < 2:
        raise ValueError(
            "grid_matrix muito pequeno: %s (mínimo 2x2)" % (grid_matrix.shape,)
        )

    # ---------- 1. Calibração pelo grid ----------
    spacing_h_px, _ = _measure_axis_spacing(grid_matrix, "h")
    spacing_v_px, _ = _measure_axis_spacing(grid_matrix, "v")

    if spacing_h_px is None and spacing_v_px is None:
        raise ValueError("Grid sem pares adjacentes válidos pra estimar escala")

    if spacing_h_px is None:
        warnings_.append("Sem pares horizontais válidos — usando apenas vertical")
        spacing_h_px = spacing_v_px
    if spacing_v_px is None:
        warnings_.append("Sem pares verticais válidos — usando apenas horizontal")
        spacing_v_px = spacing_h_px

    px_per_mm_h = float(spacing_h_px) / float(GRID_MAJOR_MM)
    px_per_mm_v = float(spacing_v_px) / float(GRID_MAJOR_MM)

    diff_pct = (
        abs(px_per_mm_h - px_per_mm_v)
        / max(px_per_mm_h, px_per_mm_v) * 100.0
    )
    if diff_pct > 10.0:
        msg = (
            f"Grid não-quadrado: h={px_per_mm_h:.2f} vs v={px_per_mm_v:.2f} px/mm "
            f"(diferença {diff_pct:.1f}%) — pode indicar erro de detecção"
        )
        warnings_.append(msg)
        logger.warning(msg)

    px_per_mm = (px_per_mm_h + px_per_mm_v) / 2.0

    # ---------- 2. Quantidades derivadas (ganho/velocidade FIXOS) ----------
    sampling_rate_hz = px_per_mm * float(paper_speed_mm_s)
    uv_per_pixel = 1000.0 / (px_per_mm * float(gain_mm_per_mV))

    result = {
        "px_per_mm_h": float(px_per_mm_h),
        "px_per_mm_v": float(px_per_mm_v),
        "px_per_mm": float(px_per_mm),
        "paper_speed_mm_s": float(paper_speed_mm_s),
        "gain_mm_per_mV": float(gain_mm_per_mV),
        "sampling_rate_hz": float(sampling_rate_hz),
        "uv_per_pixel": float(uv_per_pixel),
        "calibration_source": "grid",
        "warnings": warnings_,
    }

    logger.info(
        "Calibração: px/mm=%.2f (h=%.2f, v=%.2f), sr=%.0fHz, uV/px=%.1f, "
        "ganho=%g mm/mV (fixo)",
        result["px_per_mm"], result["px_per_mm_h"], result["px_per_mm_v"],
        result["sampling_rate_hz"], result["uv_per_pixel"],
        result["gain_mm_per_mV"],
    )

    return result


# ===========================================================================
# CÓDIGO DESATIVADO — Detecção do pulso de calibração de 1 mV
# ===========================================================================
#
# Mantido pra reativação futura. O pulso é um RETÂNGULO geométrico de 10 mm
# de altura (= 1 mV no ganho padrão) por ~5 mm de largura, normalmente nas
# bordas esquerda/direita de cada linha de derivações. Quando detectado com
# rigor (forma retangular, bordas verticais, topo/baseline planos), permite:
#   1. Validar a escala vertical do grid
#   2. Detectar ganhos não-padrão (5 ou 20 mm/mV)
#
# Para reativar:
#   - Descomentar as funções abaixo
#   - Adicionar import: `import cv2`
#   - Modificar `calibrate()` pra chamar `_detect_calibration_pulse(...)`
#     usando `normalized_image` e `n_lead_bands` (já presentes na assinatura)
#   - Adicionar de volta os campos `calibration_pulse_detected` e
#     `calibration_pulse_height_px` ao result
#
# ---------------------------------------------------------------------------
#
# def _trace_mask_from_normalized(image_bgr: np.ndarray) -> np.ndarray:
#     """Extrai máscara binária do traçado (pixels escuros)."""
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     thresh_val = float(np.percentile(gray, 8))
#     thresh_val = min(thresh_val, 110.0)
#     mask = (gray < thresh_val).astype(np.uint8) * 255
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     return mask
#
#
# def _detect_pulse_in_band(band_mask, px_per_mm):
#     """Procura pulso retangular numa faixa horizontal de derivações.
#
#     Filtros (todos obrigatórios):
#       1. Largura total (rise→fall): 3–8 mm
#       2. Altura: 6–16 mm (cobre 1 mV em ganhos 5/10/20 mm/mV)
#       3. Subida/descida quase verticais (≤ 2 px de variação horizontal)
#       4. Topo plano por ≥ 2 mm (variação vertical ≤ 1 px)
#       5. Baseline plana antes E depois por ≥ 2 mm (variação ≤ 1 px)
#     """
#     bh, bw = band_mask.shape
#     flat_baseline_min = max(3, int(round(2.0 * px_per_mm)))
#     flat_plateau_min = max(3, int(round(2.0 * px_per_mm)))
#     flat_var_max = 1
#     edge_max_width = 2
#     pulse_min_h = int(round(6.0 * px_per_mm))
#     pulse_max_h = int(round(16.0 * px_per_mm))
#     pulse_min_w = max(2, int(round(3.0 * px_per_mm)))
#     pulse_max_w = int(round(8.0 * px_per_mm))
#
#     if bh < pulse_min_h + 5 or bw < flat_baseline_min + pulse_min_w + flat_baseline_min:
#         return None
#
#     rows_idx = np.arange(bh)
#     top_y = np.full(bw, -1, dtype=np.int32)
#     for c in range(bw):
#         col = band_mask[:, c]
#         if np.any(col > 0):
#             ys = rows_idx[col > 0]
#             top_y[c] = int(ys[0])
#
#     if np.sum(top_y >= 0) < bw * 0.4:
#         return None
#
#     candidates = []
#     min_xr = flat_baseline_min
#     max_xr = bw - (pulse_min_w + flat_baseline_min)
#
#     for x_rise in range(min_xr, max_xr):
#         left = top_y[x_rise - flat_baseline_min:x_rise]
#         if np.any(left < 0) or int(left.max() - left.min()) > flat_var_max:
#             continue
#         y_base_left = float(np.median(left))
#
#         x_top = -1
#         for offset in range(edge_max_width + 1):
#             c_try = x_rise + offset
#             if c_try >= bw or top_y[c_try] < 0:
#                 continue
#             drop = y_base_left - top_y[c_try]
#             if pulse_min_h <= drop <= pulse_max_h:
#                 x_top = c_try
#                 break
#         if x_top < 0:
#             continue
#
#         y_top_seed = top_y[x_top]
#         plateau_end = x_top
#         for c_try in range(x_top + 1, min(x_top + pulse_max_w + 1, bw)):
#             if top_y[c_try] < 0 or abs(int(top_y[c_try]) - int(y_top_seed)) > flat_var_max:
#                 break
#             plateau_end = c_try
#
#         if plateau_end - x_top + 1 < flat_plateau_min:
#             continue
#         plateau_vals = top_y[x_top:plateau_end + 1]
#         if int(plateau_vals.max() - plateau_vals.min()) > flat_var_max:
#             continue
#         y_top = float(np.median(plateau_vals))
#
#         x_after = -1
#         for offset in range(1, edge_max_width + 2):
#             c_try = plateau_end + offset
#             if c_try >= bw or top_y[c_try] < 0:
#                 continue
#             if abs(int(top_y[c_try]) - y_base_left) <= flat_var_max:
#                 x_after = c_try
#                 break
#         if x_after < 0:
#             continue
#
#         right_end = x_after + flat_baseline_min
#         if right_end > bw:
#             continue
#         right = top_y[x_after:right_end]
#         if np.any(right < 0) or int(right.max() - right.min()) > flat_var_max:
#             continue
#         if abs(float(np.median(right)) - y_base_left) > flat_var_max:
#             continue
#
#         total_width = x_after - x_rise
#         if total_width < pulse_min_w or total_width > pulse_max_w:
#             continue
#
#         candidates.append(float(y_base_left - y_top))
#
#     if not candidates:
#         return None
#     return float(np.median(candidates))
#
#
# def _detect_calibration_pulse(image_bgr, px_per_mm, n_lead_bands=3):
#     """Tenta detectar o pulso nas bordas esquerda/direita."""
#     h, w = image_bgr.shape[:2]
#     if h < 100 or w < 100:
#         return None
#
#     mask = _trace_mask_from_normalized(image_bgr)
#     slice_w_px = max(40, int(round(12 * px_per_mm)))
#     slice_w_px = min(slice_w_px, w // 4)
#
#     detections = []
#     for slc in (mask[:, :slice_w_px], mask[:, w - slice_w_px:]):
#         band_h = h // n_lead_bands
#         if band_h < 20:
#             continue
#         for b in range(n_lead_bands):
#             y0 = b * band_h
#             y1 = (b + 1) * band_h if b < n_lead_bands - 1 else h
#             height_px = _detect_pulse_in_band(slc[y0:y1], px_per_mm)
#             if height_px is not None:
#                 detections.append(height_px)
#
#     return float(np.median(detections)) if detections else None
#
#
# # Lógica de override de ganho (no calibrate()) — quando reativar:
# #
# # if pulse_detected:
# #     pulse_height_mm = pulse_height_px / px_per_mm
# #     expected_default_px = float(gain_mm_per_mV) * px_per_mm
# #     height_diff_pct = abs(pulse_height_px - expected_default_px) / expected_default_px * 100
# #     if height_diff_pct > 25.0:
# #         candidates = (5.0, 10.0, 20.0)
# #         dists = [abs(pulse_height_mm - g) / g for g in candidates]
# #         best_gain = candidates[int(np.argmin(dists))]
# #         if best_gain != gain_mm_per_mV:
# #             detected_gain = best_gain
# #     calibration_source = "grid+pulse"
