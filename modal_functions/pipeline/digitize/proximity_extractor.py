"""
ProECG — Extracao de sinal por caminhada de proximidade.

Em vez de varrer coluna por coluna e pegar o meio da faixa (centroide),
percorre o tracado pixel a pixel como um dedo seguindo a linha. A cada
passo vai pro pixel de tracado nao-visitado mais proximo. Isso preserva
picos QRS porque o dedo sobe ate o topo antes de descer.

REGRA INVIOLAVEL:
  Todo Y do sinal final esta DENTRO da mascara de calor (canal 2 UNet).
  Nenhum pixel e inventado por interpolacao. Gaps na mascara = NaN no sinal.

Performance:
  Usa scipy.spatial.cKDTree pra nearest-neighbor O(log N).
  ~5-15s pra heatmap com 100-200k pixels de tracado.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def extract_signal_proximity(
    trace_mask: np.ndarray,
    threshold: float = 0.1,
    start_side: str = "left",
    max_column_revisits: int = 3,
    initial_radius: float = 1.5,
    radius_growth: float = 2.0,
    max_radius_factor: float = 0.5,
) -> dict:
    """Caminha pelo tracado pixel a pixel, escolhendo o nao-visitado mais
    proximo a cada passo.

    Args:
        trace_mask: (H, W) float prob ou bool.
        threshold: threshold pra binarizar (se for float).
        start_side: 'left' ou 'right'.
        max_column_revisits: depois de N visitas na mesma coluna, forca
                             avancar pra evitar loops.
        initial_radius: raio inicial de busca por vizinho.
        radius_growth: fator de expansao do raio quando nao acha.
        max_radius_factor: limite do raio (max(H,W) * factor).

    Returns:
        dict com:
          'signal': np.ndarray (W,) Y por coluna, NaN onde sem tracado.
          'path': lista de (y, x) na ordem percorrida.
          'steps': int.
          'multi_y_columns': int.
          'time_s': float.
          'in_mask': int — pontos do sinal dentro da mascara (deve ser == coverage_pts).
          'out_of_mask': int — DEVE SER ZERO.
    """
    t0 = time.perf_counter()
    H, W = trace_mask.shape

    # 1. Binarizar
    if trace_mask.dtype in (np.float32, np.float64):
        binary = trace_mask > threshold
    else:
        binary = trace_mask > 0

    n_trace = int(binary.sum())
    if n_trace == 0:
        return {
            "signal": np.full(W, np.nan, dtype=np.float32),
            "path": [],
            "steps": 0,
            "multi_y_columns": 0,
            "time_s": time.perf_counter() - t0,
            "in_mask": 0,
            "out_of_mask": 0,
            "binary": binary,
        }

    # 2. Coords dos pixels (y, x). KDTree usa estas coordenadas.
    ys, xs = np.where(binary)
    pixels = np.stack([ys, xs], axis=1).astype(np.float64)  # (N, 2)
    N = pixels.shape[0]
    logger.info(
        "Proximity: %d trace pixels (%.1f%% da imagem)",
        N, 100.0 * N / (H * W),
    )

    tree = cKDTree(pixels)
    visited = np.zeros(N, dtype=bool)
    column_visit_count = np.zeros(W, dtype=int)

    # 3. Achar ponto de partida (lado escolhido, Y mediano)
    if start_side == "left":
        x_extreme = int(xs.min())
        start_candidates_mask = (xs == x_extreme)
    else:
        x_extreme = int(xs.max())
        start_candidates_mask = (xs == x_extreme)
    start_idx_candidates = np.where(start_candidates_mask)[0]
    start_ys = ys[start_idx_candidates]
    median_idx = int(np.argsort(start_ys)[len(start_ys) // 2])
    start_idx = int(start_idx_candidates[median_idx])

    # 4. Caminhar
    path_indices: list[int] = [start_idx]
    visited[start_idx] = True
    column_visit_count[int(pixels[start_idx, 1])] = 1
    n_visited = 1

    current_idx = start_idx
    # Otimizacao: tree.query (k-nearest) ao inves de query_ball_point.
    # Comeca com k pequeno (vizinhos imediatos sao quase sempre o caso),
    # cresce exponencialmente se todos visitados.
    while n_visited < N:
        cy, cx = pixels[current_idx]
        chosen_idx: Optional[int] = None
        k = 10
        max_k = N

        while chosen_idx is None and k <= max_k:
            k_eff = min(k, N)
            dists_arr, idxs_arr = tree.query([cy, cx], k=k_eff)
            # k_eff=1 retorna scalar; converte pra array
            if np.isscalar(idxs_arr):
                dists_arr = np.array([dists_arr])
                idxs_arr = np.array([idxs_arr])
            best_cost = float("inf")
            best_i = None
            for d, i in zip(dists_arr, idxs_arr):
                i = int(i)
                if visited[i]:
                    continue
                ny, nx = pixels[i]
                dy_ = ny - cy
                dx_ = nx - cx
                cost = abs(dy_) + abs(dx_)
                if cost == 0:
                    continue
                # Penalidade direcional
                if start_side == "left" and dx_ < 0:
                    cost += 0.5
                elif start_side == "right" and dx_ > 0:
                    cost += 0.5
                # Limite de revisitas por coluna — forca avancar
                if column_visit_count[int(nx)] >= max_column_revisits:
                    if start_side == "left" and dx_ <= 0:
                        continue
                    if start_side == "right" and dx_ >= 0:
                        continue
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
            if best_i is not None:
                chosen_idx = best_i
                break
            # Todos os k vizinhos estao visitados — aumenta k
            if k == max_k:
                break
            k = min(k * 5, max_k)

        if chosen_idx is None:
            # Sem nao-visitados — fim
            break

        current_idx = chosen_idx
        path_indices.append(current_idx)
        visited[current_idx] = True
        column_visit_count[int(pixels[current_idx, 1])] += 1
        n_visited += 1

    # 5. Converter path em sinal — SPLITA por jumps grandes (mudanca de lead).
    path = [(int(pixels[i, 0]), int(pixels[i, 1])) for i in path_indices]
    segments = _split_path_by_jumps(path, jump_threshold=50)
    logger.info("Proximity: %d passos, %d segmentos detectados", len(path), len(segments))

    # Gera uma linha por segmento (uma derivacao por segmento)
    lines = []
    multi_y_total = 0
    for seg in segments:
        if len(seg) < 30:
            continue  # segmento muito curto, ignora
        sig, multi_y = _path_to_signal(seg, W)
        if int((~np.isnan(sig)).sum()) < 30:
            continue
        lines.append(sig)
        multi_y_total += multi_y

    # 6. Verificacao final — REGRA INVIOLAVEL (por linha)
    in_mask_total = 0
    out_of_mask_total = 0
    cleaned_lines = []
    for ln in lines:
        in_m = 0
        out_m = 0
        for x in range(W):
            if np.isnan(ln[x]):
                continue
            y = int(round(float(ln[x])))
            if 0 <= y < H and binary[y, x]:
                in_m += 1
            else:
                out_m += 1
                ln[x] = np.nan
        in_mask_total += in_m
        out_of_mask_total += out_m
        cleaned_lines.append(ln)

    # Signal global pra visualizacao 1D (concatena tudo num so array — para
    # cada coluna, pega o primeiro valor valido entre as linhas)
    signal_global = np.full(W, np.nan, dtype=np.float32)
    for ln in cleaned_lines:
        nan_in_global = np.isnan(signal_global)
        copy_from = (~np.isnan(ln)) & nan_in_global
        signal_global[copy_from] = ln[copy_from]

    elapsed = time.perf_counter() - t0
    logger.info(
        "Proximity: %d segmentos validos, %d cols multi-Y, "
        "in_mask=%d, out_of_mask=%d, %.1fs",
        len(cleaned_lines), multi_y_total, in_mask_total, out_of_mask_total, elapsed,
    )

    return {
        "signal": signal_global,
        "lines": cleaned_lines,        # uma linha por segmento (lead)
        "segments": segments,           # paths por segmento
        "path": path,
        "steps": len(path),
        "multi_y_columns": multi_y_total,
        "time_s": elapsed,
        "in_mask": in_mask_total,
        "out_of_mask": out_of_mask_total,
        "binary": binary,
    }


def _split_path_by_jumps(
    path: list[tuple[int, int]],
    jump_threshold: float = 50.0,
) -> list[list[tuple[int, int]]]:
    """Splita o path em segmentos quando ha um pulo Manhattan > threshold
    entre passos consecutivos. Cada segmento = uma derivacao (componente
    visitado continuamente)."""
    if not path:
        return []
    segments: list[list[tuple[int, int]]] = [[path[0]]]
    for i in range(1, len(path)):
        py, px = path[i - 1]
        y, x = path[i]
        dist = abs(y - py) + abs(x - px)
        if dist > jump_threshold:
            segments.append([(y, x)])
        else:
            segments[-1].append((y, x))
    return segments


def _path_to_signal(
    path: list[tuple[int, int]],
    width: int,
) -> tuple[np.ndarray, int]:
    """Converte a sequencia de (y, x) em sinal temporal (1 Y por coluna X).

    Regra pra colunas com multiplas visitas:
      - Estima baseline = mediana dos Ys das colunas com 1 visita
      - Escolhe o Y mais EXTREMO em relacao a baseline (pico)

    NAO INTERPOLA gaps — colunas sem visita ficam NaN.
    """
    column_visits: dict[int, list[int]] = {}
    for y, x in path:
        column_visits.setdefault(x, []).append(y)

    # Baseline = mediana dos Y de colunas com visita unica
    single_visit_ys = [
        visits[0] for x, visits in column_visits.items() if len(visits) == 1
    ]
    if single_visit_ys:
        baseline_y = float(np.median(single_visit_ys))
    else:
        # Fallback: mediana de TODOS os Y
        all_ys = [y for visits in column_visits.values() for y in visits]
        baseline_y = float(np.median(all_ys)) if all_ys else 0.0

    signal = np.full(width, np.nan, dtype=np.float32)
    multi_y_count = 0

    for x, visits in column_visits.items():
        if len(visits) == 1:
            signal[x] = float(visits[0])
        else:
            multi_y_count += 1
            y_min, y_max = min(visits), max(visits)
            dist_up = abs(y_min - baseline_y)
            dist_down = abs(y_max - baseline_y)
            if dist_up >= dist_down:
                signal[x] = float(y_min)  # pico positivo (R-wave)
            else:
                signal[x] = float(y_max)  # pico negativo (S/Q)

    return signal, multi_y_count
