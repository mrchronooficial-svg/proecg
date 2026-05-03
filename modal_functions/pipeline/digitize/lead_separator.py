"""
Módulo 6 — Lead Separator (Post-processing do Leader)
=====================================================

Recebe a máscara binária do Leader + a imagem normalizada + output do
calibrador. Faz:

  1. **Faixas horizontais** — projeção vertical de atividade da máscara,
     pega 3 ou 4 picos. Centro da faixa = posição do pico. Fronteira
     entre faixas = ponto médio entre centros adjacentes (não usa gaps).
  2. **Largura útil** — primeiro/último pixel com traçado na projeção
     horizontal. Dividida em 4 colunas IGUAIS de 25 mm cada (= padrão
     brasileiro 25 mm/s, 1 segundo por coluna).
  3. **Rhythm strip** — 4ª faixa (se detectada) ocupa a largura inteira,
     não é dividida em colunas.
  4. **Extração com baseline filter** — pra cada coluna de pixel dentro
     de uma derivação, agrupa pixels conectados verticalmente e escolhe
     o grupo MAIS PRÓXIMO DA BASELINE da faixa (resolve overlapping
     entre faixas vizinhas).

Layout 3×4+1 brasileiro:
    Linha 1: I, aVR, V1, V4
    Linha 2: II, aVL, V2, V5
    Linha 3: III, aVF, V3, V6
    Linha 4: DII longo (rhythm strip)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

from .stenhede_adapter import (
    extract_cell_signal_uv,
    extract_signal_probabilities,
    signal_prob_to_binary_mask,
)

logger = logging.getLogger(__name__)

LAYOUT_3x4 = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD_NAME = "II_rhythm"


# ---------------------------------------------------------------------------
# 1. Detecção de faixas horizontais (centros via picos, fronteiras via midpoints)
# ---------------------------------------------------------------------------

def _find_band_centers(
    mask: np.ndarray, px_per_mm: float, target_n: int
) -> np.ndarray:
    """Encontra `target_n` centros de banda na projeção vertical da máscara.

    Usa janelas de suavização decrescentes até produzir picos suficientes;
    seleciona top-N por proeminência.
    """
    h, w = mask.shape
    row_proj = (mask > 0).sum(axis=1).astype(np.float32)

    min_distance = max(15, int(round(12.0 * px_per_mm)))
    candidates_mm = [10.0, 7.0, 5.0, 3.0, 2.0]

    best: Optional[np.ndarray] = None
    for win_mm in candidates_mm:
        win = max(3, int(round(win_mm * px_per_mm)) | 1)
        if win >= h:
            continue
        smooth = np.convolve(row_proj, np.ones(win) / win, mode="same")
        peaks, props = find_peaks(
            smooth, distance=min_distance, prominence=max(1.0, w * 0.003)
        )
        if len(peaks) == 0:
            continue
        if len(peaks) >= target_n:
            idx = np.argsort(props["prominences"])[::-1][:target_n]
            return np.sort(peaks[idx])
        if best is None or len(peaks) > len(best):
            best = peaks

    return best if best is not None else np.array([], dtype=int)


def _centers_to_bands(centers: np.ndarray, image_h: int) -> list[tuple[int, int]]:
    """Converte centros de banda em (y1, y2) usando midpoints como fronteiras.

    Bordas externas (topo da 1ª, base da última): espelha a metade da
    distância ao vizinho mais próximo.
    """
    if len(centers) == 0:
        return []
    centers = sorted(int(c) for c in centers)
    bands: list[tuple[int, int]] = []
    for i, c in enumerate(centers):
        # Topo
        if i == 0:
            if len(centers) > 1:
                half = (centers[1] - c) // 2
                y1 = max(0, c - half)
            else:
                y1 = 0
        else:
            y1 = (centers[i - 1] + c) // 2

        # Base
        if i == len(centers) - 1:
            if len(centers) > 1:
                half = (c - centers[i - 1]) // 2
                y2 = min(image_h, c + half)
            else:
                y2 = image_h
        else:
            y2 = (c + centers[i + 1]) // 2

        bands.append((y1, y2))
    return bands


# ---------------------------------------------------------------------------
# 2. Largura útil + divisão geométrica em 4 colunas iguais
# ---------------------------------------------------------------------------

def _useful_horizontal_extent(mask: np.ndarray) -> tuple[int, int]:
    """Primeiro e último pixel com traçado em projeção horizontal global."""
    col_proj = (mask > 0).sum(axis=0)
    active = col_proj > 0
    if not np.any(active):
        return 0, mask.shape[1]
    x1 = int(np.argmax(active))
    x2 = int(len(active) - np.argmax(active[::-1]))
    return x1, x2


def _split_into_n_equal_cols(
    x1: int, x2: int, n: int
) -> list[tuple[int, int]]:
    """Divide [x1, x2) em n intervalos iguais."""
    step = (x2 - x1) / n
    return [
        (int(round(x1 + i * step)), int(round(x1 + (i + 1) * step)))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 3. Extração do sinal com filtro de baseline (resolve overlapping)
# ---------------------------------------------------------------------------

def _band_baseline_y(
    mask: np.ndarray, band: tuple[int, int], x_start: int, x_end: int
) -> float:
    """Mediana da coordenada Y dos pixels de traçado dentro da faixa.

    Computada SOBRE A FAIXA INTEIRA (todas as 4 derivações da linha) pra
    ser estável. Usada como referência pra filtrar overlapping vertical.
    """
    y1, y2 = band
    region = mask[y1:y2, x_start:x_end]
    ys, _ = np.where(region > 0)
    if len(ys) == 0:
        # Sem traçado → usa centro do bbox
        return float((y2 - y1) / 2.0)
    return float(np.median(ys))  # já em coords locais (relativas a y1)


def _remove_grid_pattern(crop_gray: np.ndarray) -> np.ndarray:
    """Mantido só para compatibilidade — não é usado mais. Veja
    `_isolate_trace_morph`."""
    crop = crop_gray.astype(np.float32)
    row_mean = crop.mean(axis=1, keepdims=True)
    col_mean = crop.mean(axis=0, keepdims=True)
    global_mean = float(crop.mean())
    return crop - row_mean - col_mean + global_mean


def _build_grid_mask(
    grid_matrix: np.ndarray,
    img_shape: tuple[int, int],
    n_minor: int = 4,
    major_thickness: int = 3,
    minor_thickness: int = 2,
) -> np.ndarray:
    """Renderiza uma máscara binária do grid (major + minor) usando a
    matriz do Gridder.

    `grid_matrix[r, c] = (x, y)` são as interseções das linhas maiores
    (5 mm). Entre cada par de linhas maiores existem 4 linhas menores
    (1 mm cada) — interpoladas linearmente nos 2 eixos.

    Espessura: 3 px nas maiores, 2 px nas menores.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if grid_matrix is None or grid_matrix.size == 0:
        return mask
    n_rows, n_cols = grid_matrix.shape[:2]

    def _line(p1: np.ndarray, p2: np.ndarray, thickness: int) -> None:
        if (
            np.isnan(p1[0]) or np.isnan(p1[1])
            or np.isnan(p2[0]) or np.isnan(p2[1])
        ):
            return
        cv2.line(
            mask,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            255, thickness, cv2.LINE_AA,
        )

    # Major horizontais — segmentos entre interseções consecutivas em cada linha
    for r in range(n_rows):
        for c in range(n_cols - 1):
            _line(grid_matrix[r, c], grid_matrix[r, c + 1], major_thickness)

    # Major verticais
    for c in range(n_cols):
        for r in range(n_rows - 1):
            _line(grid_matrix[r, c], grid_matrix[r + 1, c], major_thickness)

    # Minor horizontais (4 entre cada par de linhas maiores consecutivas)
    n_total = n_minor + 1  # major a major em (n_minor + 1) passos
    for r in range(n_rows - 1):
        for k in range(1, n_total):
            t = k / n_total
            for c in range(n_cols - 1):
                p_a = grid_matrix[r, c]
                p_b = grid_matrix[r + 1, c]
                p_a2 = grid_matrix[r, c + 1]
                p_b2 = grid_matrix[r + 1, c + 1]
                if (
                    np.isnan(p_a[0]) or np.isnan(p_b[0])
                    or np.isnan(p_a2[0]) or np.isnan(p_b2[0])
                ):
                    continue
                p_left = p_a + t * (p_b - p_a)
                p_right = p_a2 + t * (p_b2 - p_a2)
                _line(p_left, p_right, minor_thickness)

    # Minor verticais
    for c in range(n_cols - 1):
        for k in range(1, n_total):
            t = k / n_total
            for r in range(n_rows - 1):
                p_a = grid_matrix[r, c]
                p_b = grid_matrix[r, c + 1]
                p_a2 = grid_matrix[r + 1, c]
                p_b2 = grid_matrix[r + 1, c + 1]
                if (
                    np.isnan(p_a[0]) or np.isnan(p_b[0])
                    or np.isnan(p_a2[0]) or np.isnan(p_b2[0])
                ):
                    continue
                p_top = p_a + t * (p_b - p_a)
                p_bot = p_a2 + t * (p_b2 - p_a2)
                _line(p_top, p_bot, minor_thickness)

    return mask


def _isolate_trace_morph(
    crop_gray: np.ndarray, px_per_mm: float,
) -> np.ndarray:
    """Isola o traçado via top-hat morfológico vertical.

    O traçado de tinta é uma feição ESTREITA verticalmente (~3-5 px de
    espessura). O grid e o papel são as feições "largas" verticalmente
    (linhas horizontais do grid e o fundo do papel ocupam muitos px
    quando vistas em janela vertical alta).

    Pipeline:
      1. closing vertical com kernel 1×(2·px_per_mm): janela alta o
         suficiente pra "engolir" o traço — o resultado é uma estimativa
         do FUNDO (papel + grid sem o traço).
      2. fundo − original: positivo SÓ onde o original era mais escuro
         que o fundo, ou seja, exatamente nos pixels do traço.

    Retorna uma imagem uint8 onde valores altos = pixels do traço,
    valores baixos = grid/papel ≈ 0.
    """
    if crop_gray.size == 0:
        return crop_gray.copy()
    kh = max(3, 2 * int(round(px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    bg = cv2.morphologyEx(crop_gray, cv2.MORPH_CLOSE, kernel)
    trace = bg.astype(np.int16) - crop_gray.astype(np.int16)
    return np.clip(trace, 0, 255).astype(np.uint8)


def _despike_y_series(
    points: np.ndarray, spike_thr_px: float,
) -> np.ndarray:
    """Substitui spikes isolados de 1-2 colunas pela média dos vizinhos.

    Spike = ponto onde |y[i]−y[i−1]| > thr E |y[i]−y[i+1]| > thr E os
    deltas têm sinais OPOSTOS (sobe e volta, ou desce e volta). Não
    atinge QRS real, que sobe ao longo de várias colunas consecutivas.
    """
    out = points.copy()
    n = len(out)
    for i in range(1, n - 1):
        a, b, c = out[i - 1], out[i], out[i + 1]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        d_prev = b - a
        d_next = c - b
        if (
            abs(d_prev) > spike_thr_px
            and abs(d_next) > spike_thr_px
            and (d_prev * d_next) < 0  # sinais opostos
        ):
            out[i] = (a + c) / 2.0
    return out


def _extract_signal_via_tracking(
    mask: np.ndarray,
    intensity_image: Optional[np.ndarray],
    col_x1: int,
    col_x2: int,
    baseline_y: float,
    upper_limit_y: float,
    lower_limit_y: float,
    px_per_mm: float,
    uv_per_pixel: float,
    max_interp_gap_mm: float = 2.0,
    spike_threshold_mm: Optional[float] = None,
    grid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """**LEGADO — não é mais o caminho padrão.** Mantido como fallback
    de referência. O caminho atual usa o `SignalExtractor` do Stenhede
    via `stenhede_adapter.extract_cell_signal_uv` (ver
    `separate_and_extract`).

    Extrai sinal usando a MÁSCARA como guia e a IMAGEM original para o Y,
    com remoção do grid antes da seleção do pixel mais escuro.

    Pipeline (legado):
      1. Crop da grayscale dentro do bbox da derivação.
      2. Remove o grid: subtrai a média por linha + média por coluna
         (`_remove_grid_pattern`). Sobra praticamente só o traçado.
      3. Para cada coluna: nos pixels brancos da máscara, escolhe o
         pixel com MENOR valor corrigido (mais escuro depois do
         desgridding). Sem média, sem percentil, sem centroide e SEM
         restrição de distância com a coluna anterior — picos R afiados
         podem subir vários mm em uma única coluna.
      4. (Opcional) despike só se `spike_threshold_mm` for fornecido.
         Por default desligado — grid já foi removido no passo 2.
      5. Interpola gaps curtos (< 2 mm) na máscara.

    Sem outra suavização. Conversão final:
      amplitude_µV = (baseline_y − y_pix) × uv_per_pixel.
    """
    H, W = mask.shape
    w = col_x2 - col_x1
    if w <= 0:
        return np.zeros(0, dtype=np.float64)

    # Imagem em grayscale
    gray: Optional[np.ndarray] = None
    if intensity_image is not None:
        if intensity_image.ndim == 3:
            gray = cv2.cvtColor(intensity_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = intensity_image
        if gray.shape[:2] != mask.shape[:2]:
            gray = None

    upper = max(0, int(round(upper_limit_y)))
    lower = min(H - 1, int(round(lower_limit_y)))
    max_gap_px = max(2, int(round(max_interp_gap_mm * px_per_mm)))

    # Modo 1 (preferencial): se grid_mask fornecido, exclui pixels do
    # grid e procura o mais escuro do restante na imagem grayscale.
    # Modo 2 (fallback): top-hat morfológico vertical pra isolar o
    # traço, depois argmax. Modo 3: mediana da máscara (sem imagem).
    use_grid_exclude = (
        gray is not None
        and grid_mask is not None
        and grid_mask.shape[:2] == mask.shape[:2]
    )
    if gray is not None and not use_grid_exclude:
        x0 = max(0, col_x1)
        x_end = min(W, col_x2)
        crop_gray = gray[upper:lower + 1, x0:x_end]
        crop_offset_x = x0
        if crop_gray.size > 0:
            trace_img = _isolate_trace_morph(crop_gray, px_per_mm)
        else:
            trace_img = None
    else:
        trace_img = None
        crop_offset_x = col_x1

    def _pick_y(x: int) -> float:
        col_mask = mask[upper:lower + 1, x] > 0
        if not np.any(col_mask):
            return float("nan")
        ys_full = np.arange(upper, lower + 1)
        ys_in_mask = ys_full[col_mask]

        if use_grid_exclude:
            # Exclui pixels que pertencem ao grid
            not_grid = grid_mask[ys_in_mask, x] == 0
            ys_trace = ys_in_mask[not_grid]
            if len(ys_trace) == 0:
                return float("nan")  # sobra coluna sem traço puro — interpolar depois
            intens = gray[ys_trace, x].astype(np.int32)
            return float(ys_trace[int(np.argmin(intens))])

        if trace_img is not None:
            x_local = x - crop_offset_x
            if 0 <= x_local < trace_img.shape[1]:
                ys_local = ys_in_mask - upper
                intens = trace_img[ys_local, x_local]
                return float(ys_in_mask[int(np.argmax(intens))])

        return float(np.median(ys_in_mask))

    # ---- 1. Tracking por coluna (pixel mais escuro pós-desgrid) ----
    points = np.full(w, np.nan, dtype=np.float64)
    for i in range(w):
        x = col_x1 + i
        if x < 0 or x >= W:
            continue
        y_new = _pick_y(x)
        if not np.isfinite(y_new):
            continue
        y_new = max(float(upper), min(float(lower), y_new))
        points[i] = y_new

    if np.all(np.isnan(points)):
        return np.full(w, np.nan, dtype=np.float64)

    # ---- 2. (Opcional) despike — desligado por default ----
    if spike_threshold_mm is not None:
        spike_thr_px = max(1.5, spike_threshold_mm * px_per_mm)
        points = _despike_y_series(points, spike_thr_px)

    # ---- 3. Interpolação só de gaps curtos (< 2mm) ----
    valid = ~np.isnan(points)
    in_gap = False
    gap_start = 0
    for i in range(w):
        if not valid[i]:
            if not in_gap:
                gap_start = i
                in_gap = True
        elif in_gap:
            gap_len = i - gap_start
            if gap_len <= max_gap_px and gap_start > 0:
                y_left = points[gap_start - 1]
                y_right = points[i]
                for xi in range(gap_start, i):
                    t = (xi - gap_start + 1) / (gap_len + 1)
                    points[xi] = y_left + t * (y_right - y_left)
            in_gap = False

    # ---- 4. Conversão pra µV (Y cresce pra baixo → inverter sinal) ----
    signal_uv = np.full(w, np.nan, dtype=np.float64)
    valid_final = ~np.isnan(points)
    signal_uv[valid_final] = (baseline_y - points[valid_final]) * uv_per_pixel
    return signal_uv



# ---------------------------------------------------------------------------
# 4. Função pública
# ---------------------------------------------------------------------------

def separate_and_extract(
    mask: Optional[np.ndarray],
    normalized_image: Optional[np.ndarray],
    calibration: dict,
    grid_mask: Optional[np.ndarray] = None,
    signal_prob: Optional[np.ndarray] = None,
    mask_threshold: float = 0.1,
) -> dict:
    """Pipeline: máscara → derivações nomeadas + sinais em µV.

    Caminho padrão (NOVO): usa o U-Net + SignalExtractor do Stenhede
    (Open-ECG-Digitizer) pra gerar o mapa de probabilidade do traço a
    partir de `normalized_image`, e depois extrai uma linha por cell.
    A máscara binária (usada APENAS pra detectar bandas/baselines/largura)
    é derivada do mapa de probabilidade via threshold se `mask` não for
    fornecida.

    O extrator antigo (`_extract_signal_via_tracking`, baseado em "pixel
    mais escuro" + grid removal morfológico) está comentado mais abaixo
    como fallback de referência — não é mais o caminho padrão.

    Args:
        mask: máscara binária 0/255 do traço, shape (H, W). **Opcional**:
            se None, é derivada do `signal_prob` (Stenhede) via threshold.
        normalized_image: imagem undistorted BGR uint8, shape (H, W, 3).
            Obrigatória — é a entrada do U-Net Stenhede.
        calibration: dict do calibrator (px_per_mm, uv_per_pixel,
            sampling_rate_hz, ...).
        grid_mask: ignorado pelo caminho Stenhede; mantido por compat.
        signal_prob: feature map do Stenhede (H, W) float32 [0,1]. Se
            fornecido, pula a inferência do U-Net (útil pra debug/teste).
        mask_threshold: limiar pra binarizar `signal_prob` em máscara.

    Returns:
        dict com leads, layout, lead_bands, rhythm_band, useful_width,
        baselines, warnings — formato idêntico ao original.
    """
    t_total_start = time.perf_counter()

    if normalized_image is None and signal_prob is None:
        raise ValueError(
            "separate_and_extract: precisa de normalized_image OU signal_prob "
            "(o caminho Stenhede roda o U-Net na imagem undistorted)."
        )

    px_per_mm = float(calibration["px_per_mm"])
    uv_per_pixel = float(calibration["uv_per_pixel"])
    sampling_rate_hz = float(calibration["sampling_rate_hz"])

    # ---------- 0. Mapa de probabilidade do Stenhede (uma vez por foto) ----------
    if signal_prob is None:
        t_unet = time.perf_counter()
        signal_prob = extract_signal_probabilities(normalized_image)
        logger.info(
            "Stenhede signal_prob: shape=%s range=[%.3f, %.3f] (%.0f ms)",
            signal_prob.shape, float(signal_prob.min()),
            float(signal_prob.max()),
            (time.perf_counter() - t_unet) * 1000.0,
        )

    # ---------- 0b. Deriva máscara binária se não veio externamente ----------
    if mask is None:
        mask = signal_prob_to_binary_mask(signal_prob, threshold=mask_threshold)
        logger.info(
            "Máscara binária derivada do signal_prob (thr=%.2f): %d px ativos",
            mask_threshold, int(np.sum(mask > 0)),
        )
    elif mask.ndim != 2:
        raise ValueError("mask deve ser 2D (H, W) — recebi %s" % (mask.shape,))

    h, w = mask.shape
    if signal_prob.shape != (h, w):
        raise ValueError(
            "signal_prob %s incompatível com mask %s" % (signal_prob.shape, mask.shape)
        )

    warnings_: list[str] = []

    # ---------- 1. Centros de bandas (alvo: 4, fallback 3) ----------
    centers = _find_band_centers(mask, px_per_mm, target_n=4)
    if len(centers) < 3:
        # Tenta com alvo 3
        centers = _find_band_centers(mask, px_per_mm, target_n=3)
    if len(centers) < 3:
        raise ValueError(
            f"Banda detection falhou: apenas {len(centers)} picos. "
            "Mascara pode estar muito esparsa ou ECG nao tem layout 3x4."
        )

    has_rhythm = len(centers) >= 4
    layout_name = "3x4+1" if has_rhythm else "3x4"

    bands_all = _centers_to_bands(centers[:4 if has_rhythm else 3], h)
    lead_bands = bands_all[:3]
    rhythm_band = bands_all[3] if has_rhythm else None

    # ---------- 2. Largura útil + 4 colunas geométricas iguais ----------
    x_useful_start, x_useful_end = _useful_horizontal_extent(mask)
    useful_width_mm = (x_useful_end - x_useful_start) / px_per_mm if px_per_mm > 0 else 0.0
    column_bboxes = _split_into_n_equal_cols(x_useful_start, x_useful_end, n=4)

    # Sanity check: cada coluna no padrão BR (10s totais / 4 colunas, 25mm/s)
    # tem 2.5s = 62.5mm. Aceita 1s (25mm) também pra ECGs curtos.
    actual_col_w_px = (x_useful_end - x_useful_start) / 4
    col_w_mm = actual_col_w_px / px_per_mm if px_per_mm > 0 else 0.0
    if col_w_mm < 20.0 or col_w_mm > 90.0:
        warnings_.append(
            f"Largura por coluna {col_w_mm:.1f}mm fora do esperado (20-90mm) — "
            f"verificar calibracao ou crop"
        )

    # ---------- 3. Calcula baselines pra TODAS as bandas (lead + rhythm) ----------
    all_band_yranges: list[tuple[int, int]] = list(lead_bands)
    if has_rhythm and rhythm_band is not None:
        all_band_yranges.append(rhythm_band)

    all_baselines_global: list[float] = []
    for band in all_band_yranges:
        bl_local = _band_baseline_y(mask, band, x_useful_start, x_useful_end)
        all_baselines_global.append(float(bl_local + band[0]))

    # ---------- 4. Limites verticais por banda = midpoints entre baselines ----------
    def _band_limits(idx: int) -> tuple[float, float]:
        if idx == 0:
            upper = 0.0
        else:
            upper = (all_baselines_global[idx - 1] + all_baselines_global[idx]) / 2.0
        if idx == len(all_baselines_global) - 1:
            lower = float(h - 1)
        else:
            lower = (all_baselines_global[idx] + all_baselines_global[idx + 1]) / 2.0
        return upper, lower

    # ---------- 5. Extrai cada derivação via Stenhede SignalExtractor ----------
    # Caminho padrão (NOVO). Para cada cell:
    #   • recorta signal_prob no bbox da cell
    #   • SignalExtractor desenha a linha (Y em pixels, NaN onde não detecta)
    #   • converte Y → µV com nosso uv_per_pixel
    leads: dict[str, dict] = {}
    baselines: dict[str, float] = {}
    t_cells = time.perf_counter()
    n_cells_processed = 0

    for row_idx, _band in enumerate(lead_bands):
        baseline_global = all_baselines_global[row_idx]
        upper_y, lower_y = _band_limits(row_idx)
        names = LAYOUT_3x4[row_idx] if row_idx < len(LAYOUT_3x4) else [
            f"L{row_idx}_{i}" for i in range(4)
        ]
        for col_idx, (x1, x2) in enumerate(column_bboxes):
            name = names[col_idx]
            t_cell = time.perf_counter()
            signal_uv = extract_cell_signal_uv(
                signal_prob_global=signal_prob,
                bbox_xyxy=(int(x1), int(round(upper_y)),
                           int(x2), int(round(lower_y))),
                baseline_y_global=baseline_global,
                uv_per_pixel=uv_per_pixel,
            )
            n_cells_processed += 1
            logger.debug(
                "Cell %s: %d samples, %d/%d não-NaN (%.1fms)",
                name, signal_uv.shape[0],
                int(np.sum(~np.isnan(signal_uv))), signal_uv.shape[0],
                (time.perf_counter() - t_cell) * 1000.0,
            )
            duration_s = (
                signal_uv.shape[0] / sampling_rate_hz if sampling_rate_hz > 0 else 0.0
            )
            leads[name] = {
                "signal_uv": signal_uv,
                "bbox": (int(x1), int(round(upper_y)), int(x2), int(round(lower_y))),
                "samples": int(signal_uv.shape[0]),
                "duration_s": float(duration_s),
                "baseline_y": baseline_global,
            }
            baselines[name] = baseline_global

    if has_rhythm and rhythm_band is not None:
        rhythm_idx = len(lead_bands)
        baseline_global = all_baselines_global[rhythm_idx]
        upper_y, lower_y = _band_limits(rhythm_idx)
        rx1, rx2 = x_useful_start, x_useful_end
        t_cell = time.perf_counter()
        signal_uv = extract_cell_signal_uv(
            signal_prob_global=signal_prob,
            bbox_xyxy=(int(rx1), int(round(upper_y)),
                       int(rx2), int(round(lower_y))),
            baseline_y_global=baseline_global,
            uv_per_pixel=uv_per_pixel,
        )
        n_cells_processed += 1
        logger.debug(
            "Cell %s (rhythm): %d samples, %d/%d não-NaN (%.1fms)",
            RHYTHM_LEAD_NAME, signal_uv.shape[0],
            int(np.sum(~np.isnan(signal_uv))), signal_uv.shape[0],
            (time.perf_counter() - t_cell) * 1000.0,
        )
        duration_s = (
            signal_uv.shape[0] / sampling_rate_hz if sampling_rate_hz > 0 else 0.0
        )
        leads[RHYTHM_LEAD_NAME] = {
            "signal_uv": signal_uv,
            "bbox": (int(rx1), int(round(upper_y)), int(rx2), int(round(lower_y))),
            "samples": int(signal_uv.shape[0]),
            "duration_s": float(duration_s),
            "baseline_y": baseline_global,
        }
        baselines[RHYTHM_LEAD_NAME] = baseline_global

    cells_ms = (time.perf_counter() - t_cells) * 1000.0
    total_ms = (time.perf_counter() - t_total_start) * 1000.0
    logger.info(
        "Lead separator: layout=%s, bandas=%d, leads=%d, "
        "largura_util=%.0fpx (%.1fmm), col=%.0fpx (%.1fmm)",
        layout_name, len(bands_all), len(leads),
        x_useful_end - x_useful_start, useful_width_mm,
        actual_col_w_px, actual_col_w_px / px_per_mm if px_per_mm > 0 else 0.0,
    )
    logger.info(
        "Lead separator timing: %d cells em %.0fms (~%.1fms/cell), "
        "total %.0fms",
        n_cells_processed, cells_ms,
        cells_ms / max(n_cells_processed, 1), total_ms,
    )

    return {
        "leads": leads,
        "layout": layout_name,
        "n_lead_bands": len(lead_bands),
        "lead_bands": lead_bands,
        "rhythm_band": rhythm_band,
        "useful_width": (x_useful_start, x_useful_end),
        "column_bboxes": column_bboxes,
        "baselines": baselines,
        "signal_prob_shape": signal_prob.shape,
        "warnings": warnings_,
    }
