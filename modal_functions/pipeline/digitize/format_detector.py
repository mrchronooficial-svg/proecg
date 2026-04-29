"""Detecção do layout (linhas × colunas + rhythm leads) de um ECG normalizado.

Entra após undistort + correção de orientação. Identifica o layout do ECG.

Estratégia: TEMPLATE MATCHING contra layouts conhecidos.
  Para cada candidato (n_rows, n_cols, n_rhythm), calcula um score que
  mede quanto a imagem se encaixa naquela estrutura:
    - Bandas horizontais com alta atividade nos centros esperados
    - Gaps verticais com baixa atividade nas bordas entre bandas
    - Dentro das bandas multi-coluna: colunas de tinta separadas por gaps
  O candidato com maior score vence.

Layouts testados (do mais comum no Brasil pro mais raro):
  3x4+1, 3x4+0, 6x2+1, 6x2+0, 12x1+0, 3x4+2, 3x4+3.

Saída: dict {format, n_rows, n_cols, n_rhythm, row_boundaries,
col_boundaries, rhythm_indices, fallback, score}.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# (n_rows, n_cols, n_rhythm) → string canônica.
CANDIDATES = [
    (3, 4, 1, "3x4+1"),
    (3, 4, 0, "3x4+0"),
    (6, 2, 1, "6x2+1"),
    (6, 2, 0, "6x2+0"),
    (12, 1, 0, "12x1+0"),
    (3, 4, 2, "3x4+2"),
    (3, 4, 3, "3x4+3"),
]


def detect_layout(
    img: np.ndarray,
    px_per_mm: float | None = None,
    *,
    dark_threshold: int = 100,
    smooth_frac: float = 0.02,
    side_margin_frac: float = 0.10,
    band_center_frac: float = 0.50,
    gap_zone_frac: float = 0.20,
) -> dict:
    """Detecta o layout via template matching contra candidatos conhecidos.

    Args:
        img: imagem BGR (ou grayscale) pós-undistort.
        px_per_mm: aceito por compat; não usado.
        dark_threshold: pixels com gray < threshold contam como traçado.
        smooth_frac: janela do suavizador (fração da dimensão).
        side_margin_frac: recorta esse % das laterais para perfil vertical.
        band_center_frac: % central da banda candidata onde mede atividade.
        gap_zone_frac: % da banda que vira "zona de gap" entre bandas.
    """
    del px_per_mm

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape
    trace_mask = (gray < dark_threshold).astype(np.uint8)

    # Perfis suavizados.
    sx = int(side_margin_frac * w)
    central = trace_mask[:, sx:w - sx] if sx * 2 < w else trace_mask
    smooth_y = max(3, int(round(smooth_frac * h)))
    smooth_x = max(3, int(round(smooth_frac * w)))
    profile_y = uniform_filter1d(central.sum(axis=1).astype(float), size=smooth_y)

    if profile_y.max() <= profile_y.min():
        logger.warning("Format detector: perfil chato — fallback 3x4+1")
        return _fallback()

    # Pula cabeçalho (topo 8% se densidade alta) na avaliação dos templates.
    head_h = int(0.08 * h)
    if profile_y[:head_h].mean() > profile_y[head_h:].mean() * 1.5:
        body_y0 = head_h
    else:
        body_y0 = 0
    body_y1 = h
    # Pula rodapé com calibração se houver banda densa nos últimos 5%.
    foot_start = int(0.95 * h)
    if profile_y[foot_start:].mean() > profile_y[body_y0:foot_start].mean() * 1.5:
        body_y1 = foot_start

    # Avalia cada candidato.
    results = []
    for n_rows, n_cols, n_rhythm, name in CANDIDATES:
        score, info = _score_template(
            trace_mask, profile_y, body_y0, body_y1, smooth_x,
            n_rows, n_cols, n_rhythm,
            band_center_frac=band_center_frac,
            gap_zone_frac=gap_zone_frac,
        )
        results.append((score, name, n_rows, n_cols, n_rhythm, info))

    results.sort(key=lambda r: -r[0])
    best_score, best_name, best_nr, best_nc, best_nrh, best_info = results[0]

    # Ranking de top-3 pra log.
    ranking = ", ".join(f"{r[1]}={r[0]:.2f}" for r in results[:3])

    logger.info(
        "Format detector: %s (score=%.2f) | top3: %s",
        best_name, best_score, ranking,
    )

    return {
        "format": best_name,
        "n_rows": best_nr,
        "n_cols": best_nc,
        "n_rhythm": best_nrh,
        "row_boundaries": best_info["row_boundaries"],
        "col_boundaries": best_info["col_boundaries"],
        "rhythm_indices": best_info["rhythm_indices"],
        "score": float(best_score),
        "fallback": False,
    }


def _score_template(
    trace_mask: np.ndarray,
    profile_y: np.ndarray,
    body_y0: int,
    body_y1: int,
    smooth_x: int,
    n_rows: int,
    n_cols: int,
    n_rhythm: int,
    *,
    band_center_frac: float,
    gap_zone_frac: float,
) -> tuple[float, dict]:
    """Score de quanto a imagem se ajusta ao layout (n_rows, n_cols, n_rhythm).

    Score combina:
      - Contraste banda/gap no perfil vertical (mede separação de linhas).
      - Contraste coluna/gap no perfil horizontal de cada lead row.
      - Continuidade de tinta na rhythm row (full-width).
    Score normalizado em [0, ~3].
    """
    h, w = trace_mask.shape
    n_total = n_rows + n_rhythm
    if n_total <= 0:
        return -1.0, {"row_boundaries": [], "col_boundaries": [], "rhythm_indices": []}

    body_h = body_y1 - body_y0
    if body_h <= n_total * 4:
        return -1.0, {"row_boundaries": [], "col_boundaries": [], "rhythm_indices": []}

    band_h = body_h / n_total
    row_bounds = []
    rhythm_indices = []
    col_bounds_per_row = []

    # 1. Score vertical: contraste banda vs gap entre bandas.
    band_means = []
    gap_means = []
    for r in range(n_total):
        y_top = body_y0 + int(r * band_h)
        y_bot = body_y0 + int((r + 1) * band_h)
        row_bounds.append((y_top, y_bot))

        center_h = (y_bot - y_top) * band_center_frac
        cy0 = int(y_top + (y_bot - y_top - center_h) / 2)
        cy1 = int(cy0 + center_h)
        band_means.append(profile_y[cy0:cy1].mean())

        # Gap zone entre essa banda e a próxima.
        if r < n_total - 1:
            gap_h = (y_bot - y_top) * gap_zone_frac
            gy0 = int(y_bot - gap_h / 2)
            gy1 = int(y_bot + gap_h / 2)
            gap_means.append(profile_y[max(0, gy0):min(h, gy1)].mean())

    if not band_means or max(band_means) <= 0:
        return -1.0, {"row_boundaries": row_bounds, "col_boundaries": [], "rhythm_indices": []}

    bm = float(np.mean(band_means))
    gm = float(np.mean(gap_means)) if gap_means else 0.0
    # Score 0-1: 1 quando gap = 0, 0 quando gap = band.
    vertical_score = max(0.0, (bm - gm) / max(bm, 1.0))

    # 2. Score horizontal por banda.
    rhythm_indices_idx = list(range(n_rows, n_total))  # rhythms vêm depois das lead rows
    horizontal_scores = []

    for i, (y_top, y_bot) in enumerate(row_bounds):
        is_rhythm = i in rhythm_indices_idx
        mh = max(1, int(0.10 * (y_bot - y_top)))
        band = trace_mask[y_top + mh:y_bot - mh, :]
        if band.size == 0:
            horizontal_scores.append(0.0)
            col_bounds_per_row.append([])
            continue

        ink_x = uniform_filter1d(band.sum(axis=0).astype(float), size=smooth_x)
        if ink_x.max() <= 0:
            horizontal_scores.append(0.0)
            col_bounds_per_row.append([])
            continue

        if is_rhythm:
            # Rhythm: trace contínuo. Score = % de cols com tinta (acima de 10% do max).
            ink_threshold = ink_x.max() * 0.10
            inky_cols = (ink_x > ink_threshold).sum()
            score_h = inky_cols / float(w)
            rhythm_indices.append(i)
            col_bounds_per_row.append([(0, w)])
        else:
            # Lead row: contraste entre n_cols centros e (n_cols-1) gaps.
            col_w = w / n_cols
            col_means = []
            col_segments = []
            for c in range(n_cols):
                x0 = int(c * col_w)
                x1 = int((c + 1) * col_w)
                # Centro da coluna esperada.
                cx0 = int(x0 + (x1 - x0) * 0.20)
                cx1 = int(x1 - (x1 - x0) * 0.20)
                col_means.append(ink_x[cx0:cx1].mean())
                col_segments.append((x0, x1))
            col_bounds_per_row.append(col_segments)

            inter_means = []
            for c in range(n_cols - 1):
                x_mid = int((c + 1) * col_w)
                gap_w = col_w * gap_zone_frac
                gx0 = max(0, int(x_mid - gap_w / 2))
                gx1 = min(w, int(x_mid + gap_w / 2))
                inter_means.append(ink_x[gx0:gx1].mean())

            if not col_means or max(col_means) <= 0:
                score_h = 0.0
            else:
                cm = float(np.mean(col_means))
                im = float(np.mean(inter_means)) if inter_means else 0.0
                score_h = max(0.0, (cm - im) / max(cm, 1.0))
        horizontal_scores.append(score_h)

    h_score = float(np.mean(horizontal_scores)) if horizontal_scores else 0.0

    # Score final: vertical * horizontal (ambos têm que ser bons).
    # Bonus se vertical_score > 0.3 (separação clara).
    final = vertical_score * h_score
    if vertical_score > 0.3:
        final *= 1.2
    return final, {
        "row_boundaries": row_bounds,
        "col_boundaries": col_bounds_per_row,
        "rhythm_indices": rhythm_indices,
    }


def _fallback() -> dict:
    """Layout default no Brasil quando detecção falha."""
    return {
        "format": "3x4+1",
        "n_rows": 3,
        "n_cols": 4,
        "n_rhythm": 1,
        "row_boundaries": [],
        "col_boundaries": [],
        "rhythm_indices": [],
        "fallback": True,
        "score": 0.0,
    }
