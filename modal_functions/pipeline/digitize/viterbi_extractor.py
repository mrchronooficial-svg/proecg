"""
Extracao de sinal 1D de mascara binaria de ECG via Viterbi DP.

Base: Fortune et al. (2022) — paper-ecg/ecgdigitize.
Modificacoes Rahimi et al. (2025), ja embutidas no Fortune:
  MOD 1 — Centros de clusters por coluna (nao pixels individuais)
  MOD 2 — Custo = alpha*dist_euclidiana + (1-alpha)*mudanca_angulo (alpha=0.5)

Post-processing (Rahimi):
  - Interpolacao linear de gaps
  - Remocao de baseline (mediana movel)
  - Resampling opcional pra taxa padrao

Entrada: mascara 2D (PNG carregado), fundo BRANCO, tracado PRETO
Saida: array 1D float (Y em pixels por coluna). NaN preservado pra gaps que
       cobrem TODAS as colunas (raro pos-interpolacao).
"""

from __future__ import annotations

from math import asin, isnan, pi, sqrt
from typing import Optional

import numpy as np

ALPHA = 0.5  # peso da distancia no custo (Rahimi MOD 2)
MINIMUM_LOOKBACK = 1
OPTIMAL_ENDING_WIDTH = 20  # janela final pra escolher melhor ponto terminal


# ---------------------------------------------------------------------------
# Helpers geometricos
# ---------------------------------------------------------------------------

def _euclid(dx: float, dy: float) -> float:
    return sqrt(dx * dx + dy * dy)


def _angle_from_offsets(dx: float, dy: float) -> float:
    """Angulo em graus do vetor (dx, dy). Convencao: asin(dy/hypot)/pi*180."""
    h = _euclid(dx, dy)
    if h == 0:
        return 0.0
    return asin(dy / h) / pi * 180.0


def _angle_between_points(p_from: tuple, p_to: tuple) -> float:
    return _angle_from_offsets(p_to[0] - p_from[0], p_to[1] - p_from[1])


def _angle_similarity(a1: float, a2: float) -> float:
    """Retorna [0,1] onde 1 = mesmo angulo, 0 = oposto."""
    return (180.0 - abs(a2 - a1)) / 180.0


def _transition_cost(current_pt: tuple, candidate_pt: tuple,
                     candidate_angle: float) -> float:
    """Custo de transicao Rahimi MOD 2:
       cost = alpha * dist_euclidiana + (1-alpha) * mudanca_angulo
       Mudanca_angulo = 1 - similarity(angle_chegada, angle_proximo).
    """
    arrival_angle = _angle_between_points(candidate_pt, current_pt)
    angle_change = 1.0 - _angle_similarity(arrival_angle, candidate_angle)
    dist = _euclid(current_pt[0] - candidate_pt[0],
                   current_pt[1] - candidate_pt[1])
    return ALPHA * dist + (1.0 - ALPHA) * angle_change


# ---------------------------------------------------------------------------
# Rahimi MOD 1 — cluster centers per column
# ---------------------------------------------------------------------------

def _find_contiguous_centers(column: np.ndarray) -> list[int]:
    """Por coluna (1D bool/int), retorna centros (rows) de cada regiao contigua
    de pixels True. Reduz traco grosso a 1 ponto por cluster vertical."""
    centers = []
    start = None
    for i, px in enumerate(column):
        if px > 0 and start is None:
            start = i
        elif px == 0 and start is not None:
            centers.append((start + i - 1) // 2)
            start = None
    if start is not None:  # cluster aberto no fim da coluna
        centers.append((start + len(column) - 1) // 2)
    return centers


def _get_points_by_column(mask: np.ndarray) -> list[list[tuple]]:
    """Retorna lista de listas: cada sublista = pontos (x, y) na coluna x."""
    h, w = mask.shape
    points_by_col = []
    for col in range(w):
        rows = _find_contiguous_centers(mask[:, col])
        points_by_col.append([(col, r) for r in rows])
    return points_by_col


# ---------------------------------------------------------------------------
# Viterbi DP
# ---------------------------------------------------------------------------

def _get_adjacent(points_by_col, best_path, current_col: int,
                  min_lookback: int):
    """Itera por pontos da(s) coluna(s) anterior(es). Se vazias, expande
    lookback ate achar algum (gap handling)."""
    right = current_col
    left = max(0, current_col - min_lookback)
    while True:
        any_found = False
        for col_idx in range(left, right):
            for pt in points_by_col[col_idx]:
                if pt in best_path:
                    score, parent, angle = best_path[pt]
                    yield score, pt, angle
                    any_found = True
        if any_found or left == 0:
            return
        left -= 1


def _interpolate_segment(p_far: tuple, p_near: tuple, signal: np.ndarray) -> None:
    """Interpola Y linearmente entre p_far (x maior) e p_near (x menor)
    e escreve em signal (in-place) preenchendo posicoes intermediarias."""
    x0, y0 = p_near
    x1, y1 = p_far
    if x1 - x0 < 2:
        return
    slope = (y1 - y0) / (x1 - x0)
    for x in range(x0 + 1, x1):
        signal[x] = slope * (x - x1) + y1


def _convert_points_to_signal(points: list[tuple], width: int) -> np.ndarray:
    """Best path (reversed: indice 0 = ponto mais a direita) -> array 1D.
    Interpola gaps entre pontos consecutivos."""
    signal = np.full(width, np.nan, dtype=float)
    if not points:
        return signal
    first = points[0]
    signal[first[0]] = first[1]
    prior = first
    for pt in points[1:]:
        # pt[0] < prior[0] (estamos andando pra esquerda no best path)
        if pt[0] + 1 < prior[0]:
            _interpolate_segment(prior, pt, signal)
        signal[pt[0]] = pt[1]
        prior = pt
    return signal


def _run_viterbi(mask_bool: np.ndarray) -> Optional[np.ndarray]:
    """Roda Viterbi DP completo. Retorna sinal 1D ou None se mascara vazia."""
    h, w = mask_bool.shape
    points_by_col = _get_points_by_column(mask_bool)
    total_points = sum(len(c) for c in points_by_col)
    if total_points == 0:
        return None

    best_path: dict = {}  # point -> (score, parent, arrival_angle)

    # Base case: primeira coluna NAO-VAZIA
    first_non_empty = None
    for i, col in enumerate(points_by_col):
        if col:
            first_non_empty = i
            break
    if first_non_empty is None:
        return None
    for pt in points_by_col[first_non_empty]:
        best_path[pt] = (0.0, None, 0.0)

    # Forward pass
    for col_idx in range(first_non_empty + 1, w):
        for pt in points_by_col[col_idx]:
            adjacent = list(_get_adjacent(points_by_col, best_path,
                                          pt[0], MINIMUM_LOOKBACK))
            if not adjacent:
                best_path[pt] = (0.0, None, 0.0)
                continue
            best_score = float("inf")
            best_parent = None
            for cand_score, cand_pt, cand_angle in adjacent:
                cost = _transition_cost(pt, cand_pt, cand_angle)
                total = cand_score + cost
                if total < best_score:
                    best_score = total
                    best_parent = cand_pt
            arrival_angle = _angle_between_points(best_parent, pt) if best_parent else 0.0
            best_path[pt] = (best_score, best_parent, arrival_angle)

    # Backtrack: comeca do melhor ponto na janela final
    ending = list(_get_adjacent(points_by_col, best_path,
                                w, OPTIMAL_ENDING_WIDTH))
    if not ending:
        return None
    _, current, _ = min(ending, key=lambda t: t[0])

    path = []
    while current is not None:
        path.append(current)
        _, current, _ = best_path[current]

    return _convert_points_to_signal(path, w)


# ---------------------------------------------------------------------------
# Post-processing (Rahimi)
# ---------------------------------------------------------------------------

def _interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Interpolacao linear de NaN. Edges com nearest valid."""
    s = signal.copy()
    isn = np.isnan(s)
    if isn.all():
        return s
    x = np.arange(len(s))
    valid = ~isn
    s[isn] = np.interp(x[isn], x[valid], s[valid])
    return s


def _remove_baseline(signal: np.ndarray, window_frac: float = 0.1) -> np.ndarray:
    """Subtrai mediana movel (baseline) do sinal.
    window_frac = fracao do comprimento total como janela (default 10%)."""
    from scipy.ndimage import median_filter
    n = len(signal)
    window = max(3, int(n * window_frac) | 1)  # forca impar
    baseline = median_filter(signal, size=window, mode="nearest")
    return signal - baseline


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def extrair_sinal_viterbi(mask: np.ndarray, invert: bool = True) -> np.ndarray:
    """Extrai sinal 1D de mascara binaria de ECG via Viterbi DP.

    Args:
        mask: imagem 2D (uint8 ou float). Pode ser BGR/RGB/gray;
              se >2D, converte pra grayscale via media de canais.
        invert: se True (default), inverte: pixels ESCUROS viram tracado.
                Use quando mascara tem fundo BRANCO e tracado PRETO.

    Returns:
        np.ndarray 1D float com sinal extraido (Y em pixels por coluna).
        Negado pra QRS apontar pra cima. Baseline removida.
    """
    if mask.ndim == 3:
        mask = mask.mean(axis=2)

    # Binariza: True = tracado
    if invert:
        binary = mask < 128
    else:
        binary = mask >= 128

    signal = _run_viterbi(binary.astype(np.uint8))
    if signal is None:
        return np.full(mask.shape[1], np.nan, dtype=float)

    # Post-processing
    signal_filled = _interpolate_nans(signal)
    signal_centered = _remove_baseline(signal_filled, window_frac=0.1)
    signal_flipped = -signal_centered  # Y cresce pra baixo na imagem -> nega
    return signal_flipped
