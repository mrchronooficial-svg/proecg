"""
M4 — Identificação de Leads (derivações).

Encontra o bounding box do ECG via masks (trace), aplica layout template.
Clipa todas as bboxes dentro da borda do papel ECG.

Input:  masks dewarped, CalibrationResult
Output: lista de LeadRegion (name, bbox, row, col)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .constants import LEAD_ORDER

logger = logging.getLogger(__name__)


@dataclass
class LeadRegion:
    """Região de uma derivação no ECG."""
    name: str
    bbox: Tuple[int, int, int, int]  # (y1, x1, y2, x2)
    row: int
    col: int


# Layouts brasileiros
LAYOUT_3x4_DII = {
    "name": "3x4+DII longo",
    "rows": 4,
    "cols": [4, 4, 4, 1],
    "leads": [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
        ["II"],
    ],
}

LAYOUT_3x4 = {
    "name": "3x4 sem ritmo",
    "rows": 3,
    "cols": [4, 4, 4],
    "leads": [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ],
}

KNOWN_LAYOUTS = [LAYOUT_3x4_DII, LAYOUT_3x4]


def _find_paper_bbox(masks: dict) -> Tuple[int, int, int, int]:
    """Encontra o bounding box do PAPEL ECG (não da imagem toda).

    Estratégia:
      1. Dilatar trace para criar blob contínuo da área de sinal
      2. Encontrar o maior contorno (= área do papel com sinais)
      3. Bounding rect desse contorno = limites do papel
      4. Expandir 3% para margem

    Retorna (y1, x1, y2, x2).
    """
    H, W = None, None
    for mask in masks.values():
        if mask is not None:
            H, W = mask.shape
            break
    if H is None:
        return (0, 0, 100, 100)

    trace_mask = masks.get("trace")

    if trace_mask is not None and trace_mask.sum() > 50:
        # Dilatar bastante para unir todas as regiões de trace num único blob
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated = cv2.dilate(trace_mask.astype(np.uint8), kernel, iterations=4)

        # Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Maior contorno = área principal do ECG
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)

            # Expandir 3%
            pad_x = int(bw * 0.03)
            pad_y = int(bh * 0.03)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(W, bx + bw + pad_x)
            y2 = min(H, by + bh + pad_y)

            # Sanidade: deve cobrir pelo menos 20% da imagem
            area_frac = (x2 - x1) * (y2 - y1) / (H * W)
            if area_frac > 0.15:
                logger.info("Paper bbox (contour): (%d,%d)→(%d,%d), %.0f%% da imagem",
                            y1, x1, y2, x2, area_frac * 100)
                return (y1, x1, y2, x2)

    # Fallback: usar grid com percentis conservadores
    grid_mask = masks.get("grid")
    if grid_mask is not None and grid_mask.sum() > 100:
        g_h = grid_mask.astype(np.float64).sum(axis=0)
        g_v = grid_mask.astype(np.float64).sum(axis=1)
        g_h_nz = np.where(g_h > g_h.max() * 0.15)[0]
        g_v_nz = np.where(g_v > g_v.max() * 0.15)[0]

        if len(g_h_nz) > 10 and len(g_v_nz) > 10:
            x1 = int(np.percentile(g_h_nz, 5))
            x2 = int(np.percentile(g_h_nz, 95))
            y1 = int(np.percentile(g_v_nz, 5))
            y2 = int(np.percentile(g_v_nz, 95))
            logger.info("Paper bbox (grid): (%d,%d)→(%d,%d)", y1, x1, y2, x2)
            return (y1, x1, y2, x2)

    # Fallback final
    margin_y = int(H * 0.10)
    margin_x = int(W * 0.05)
    return (margin_y, margin_x, H - margin_y, W - margin_x)


def _detect_row_count(trace_mask: np.ndarray, ecg_bbox: Tuple[int, int, int, int]) -> int:
    """Detecta se o ECG tem 3 ou 4 linhas analisando o perfil vertical."""
    y1, x1, y2, x2 = ecg_bbox
    roi = trace_mask[y1:y2, x1:x2]

    if roi.sum() < 20:
        return 3

    H_roi = y2 - y1
    v_profile = roi.astype(np.float64).sum(axis=1)

    kernel = max(5, H_roi // 20)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = np.convolve(v_profile, np.ones(kernel) / kernel, mode="same")

    threshold = smoothed.max() * 0.1 if smoothed.max() > 0 else 0
    above = smoothed > threshold

    bands = 0
    in_band = False
    for v in above:
        if v and not in_band:
            bands += 1
            in_band = True
        elif not v:
            in_band = False

    logger.info("Perfil vertical: %d bandas de trace detectadas", bands)

    if bands >= 4:
        return 4
    return 3


def _clip_bbox(bbox: Tuple[int, int, int, int],
               clip: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Intersecta bbox com clip region."""
    y1, x1, y2, x2 = bbox
    cy1, cx1, cy2, cx2 = clip
    return (max(y1, cy1), max(x1, cx1), min(y2, cy2), min(x2, cx2))


def detect_leads(
    masks: dict,
    calibration=None,
) -> List[LeadRegion]:
    """Detecta e mapeia derivações no ECG.

    Estratégia:
      1. Encontrar bounding box do papel ECG
      2. Detectar número de linhas (3 ou 4)
      3. Aplicar template layout
      4. Clipar todas as bboxes dentro do papel
    """
    H, W = None, None
    for mask in masks.values():
        if mask is not None:
            H, W = mask.shape
            break
    if H is None:
        return []

    trace_mask = masks.get("trace")

    # 1. Encontrar bbox do papel
    paper_bbox = _find_paper_bbox(masks)
    py1, px1, py2, px2 = paper_bbox
    paper_h = py2 - py1
    paper_w = px2 - px1

    if paper_h < 20 or paper_w < 20:
        logger.warning("Paper bbox muito pequeno: %dx%d", paper_w, paper_h)
        return _uniform_layout(0, 0, H, W, LAYOUT_3x4_DII, (0, 0, H, W))

    # 2. Detectar número de linhas
    n_rows = 3
    if trace_mask is not None and trace_mask.sum() > 50:
        n_rows = _detect_row_count(trace_mask, paper_bbox)

    # 3. Escolher layout
    layout = LAYOUT_3x4_DII if n_rows >= 4 else LAYOUT_3x4

    # 4. Dividir uniformemente + clipar dentro do papel
    regions = _uniform_layout(py1, px1, py2, px2, layout, paper_bbox)

    detected_names = {r.name for r in regions}
    missing = [l for l in LEAD_ORDER if l not in detected_names]
    if missing:
        logger.warning("Derivações não cobertas: %s", missing)

    logger.info("Detectadas %d derivações (%s) no paper bbox (%d,%d)→(%d,%d)",
                len(regions), layout["name"], py1, px1, py2, px2)
    return regions


def _uniform_layout(
    y1: int, x1: int, y2: int, x2: int,
    layout: dict,
    paper_clip: Tuple[int, int, int, int],
) -> List[LeadRegion]:
    """Divide uma região uniformemente, clipando dentro do papel."""
    ecg_h = y2 - y1
    ecg_w = x2 - x1
    n_rows = layout["rows"]
    row_h = ecg_h // n_rows

    regions = []
    for row_idx, lead_names in enumerate(layout["leads"]):
        ry1 = y1 + row_idx * row_h
        ry2 = ry1 + row_h
        n_cols = len(lead_names)

        for col_idx, name in enumerate(lead_names):
            if n_cols == 1:
                cx1 = x1
                cx2 = x2
            else:
                col_w = ecg_w // n_cols
                cx1 = x1 + col_idx * col_w
                cx2 = cx1 + col_w

            # Clipar dentro do papel
            clipped = _clip_bbox((ry1, cx1, ry2, cx2), paper_clip)
            cy1, cx1c, cy2, cx2c = clipped

            # Verificar tamanho mínimo
            clipped_w = cx2c - cx1c
            clipped_h = cy2 - cy1
            expected_w = ecg_w // max(n_cols, 1)
            expected_h = row_h

            if clipped_w < expected_w * 0.30 or clipped_h < expected_h * 0.30:
                logger.warning("Lead %s: bbox muito pequena após clip (%dx%d), parcialmente obstruído",
                               name, clipped_w, clipped_h)

            regions.append(LeadRegion(
                name=name,
                bbox=(cy1, cx1c, cy2, cx2c),
                row=row_idx,
                col=col_idx,
            ))

    return regions
