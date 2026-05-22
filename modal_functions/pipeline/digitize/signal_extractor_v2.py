"""
ProECG Signal Extractor v2 — combina 3 abordagens da literatura para
extracao robusta de sinal ECG.

Fases (cada uma pode ser desligada independentemente):
  Fase 1: Fragmented extraction (UMMISCO/ecgtizer)
    -> Para cada coluna, agrupa pixels lit em fragmentos. Se ha multiplos,
       escolhe o de maior probabilidade media (anti-texto/grid residual).
       Y final = centroide ponderado por prob do fragmento escolhido.
  Fase 2: Viterbi bidirecional (Tereshchenkolab/paper-ecg)
    -> Refina o sinal da Fase 1 com programacao dinamica global.
       Score = alpha * distancia + (1-alpha) * penalidade_angular.
       Roda forward + backward, media os 2 = mais preciso nas bordas.
  Fase 3: QRS peak correction (Fortune et al. 2022)
    -> Onde o slope e alto (deflexao rapida), substitui o centroide
       pelo Y do extremo da mascara (topo se subindo, base se descendo).

Input: signal_prob (H, W) torch.Tensor ou np.ndarray — output do canal 2
       da UNet Stenhede apos process_sparse_prob (valores em [0, 1]).

Output (de extract_signal_v2):
  dict com:
    'lines': list[np.ndarray (W,)] — Y positions por coluna, NaN onde sem
                                    sinal. Uma linha por connected component.
    'method': str — quais fases foram usadas.
    'stats': dict — n_components, coverage, etc.

Output (de extract_lines_v2_with_offset):
  (lines_tensor (N, W_trimmed), x_offset: int) — mesmo formato do
  _signal_extractor_with_offset do stenhede_adapter.py
"""

from __future__ import annotations

import logging
from math import asin, pi, sqrt
from typing import Optional

import numpy as np
import torch
from skimage.measure import label as sk_label

logger = logging.getLogger(__name__)


# =====================================================================
# FASE 1 — Fragmented Extraction (ECGtizer)
# Referencia: ecgtizer/ecgtizer/extraction_functions.py::fragmented_extraction
# =====================================================================

def _phase1_fragmented_column(
    positions: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Single-column Fragmented extraction.

    - 0 positions: None (NaN)
    - 1 position: returns it
    - Single contiguous fragment: returns mean (ou centroide por prob se given)
    - Multiple fragments:
        - Com probs: pega o de MAIOR prob media (mais provavel ser o sinal)
        - Sem probs: pega o ULTIMO fragmento (heuristica ECGtizer anti-texto)
    """
    if positions.size == 0:
        return None
    if positions.size == 1:
        return float(positions[0])
    breaks = np.where(np.diff(positions) > 1)[0] + 1
    if breaks.size == 0:
        # Single fragment
        if probs is not None and probs.sum() > 1e-9:
            return float(np.sum(positions * probs) / probs.sum())
        return float(np.mean(positions))
    # Multiple fragments
    fragments = np.split(positions, breaks)
    if probs is not None:
        fragment_probs = np.split(probs, breaks)
        means = np.array([p.mean() if p.size > 0 else 0.0 for p in fragment_probs])
        best = int(np.argmax(means))
        frag = fragments[best]
        prob_frag = fragment_probs[best]
        if prob_frag.sum() > 1e-9:
            return float(np.sum(frag * prob_frag) / prob_frag.sum())
        return float(np.mean(frag))
    # Sem probs: ECGtizer original — last fragment
    return float(np.mean(fragments[-1]))


def _phase1_extract_fragmented(
    component_mask: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-component Phase 1 fragmented extraction.

    Args:
        component_mask: (H, W) bool — True onde esse component esta presente.
        prob_map: (H, W) float — prob da UNet (usada como peso pro centroide
                                  e pra escolher fragment quando multiplos).

    Returns:
        signal: (W,) float32 — Y per column, NaN onde sem sinal.
    """
    H, W = component_mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        col_mask = component_mask[:, x]
        if not col_mask.any():
            continue
        positions = np.where(col_mask)[0]
        probs = prob_map[positions, x] if prob_map is not None else None
        result = _phase1_fragmented_column(positions, probs)
        if result is not None:
            signal[x] = result
    return signal


def _phase1_skeleton_per_component(
    component_mask: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Skeletoniza o component e le Y por coluna.

    O esqueleto reduz a "faixa grossa" da mascara para 1 pixel de largura.
    Para colunas com >1 pixel do skeleton (vertical strokes do QRS), pega
    o mais proximo do Y anterior valido (continuity).

    Validado nos overlays_per_lead — produz linhas finas e fiéis ao traçado.
    """
    from skimage.morphology import skeletonize as _skel
    skel = _skel(component_mask > 0)
    H, W = component_mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(skel[:, x])[0]
        if ys.size == 0:
            continue
        elif ys.size == 1:
            signal[x] = float(ys[0])
        else:
            if x > 0 and not np.isnan(signal[x - 1]):
                signal[x] = float(ys[np.argmin(np.abs(ys - signal[x - 1]))])
            else:
                signal[x] = float(ys[ys.size // 2])
    return signal


def _fill_small_gaps_in_signal(
    signal: np.ndarray,
    max_gap_size: int = 200,
    max_y_diff: float = 80.0,
) -> np.ndarray:
    """Interpolacao linear em gaps PEQUENOS de NaN no sinal quando os Y
    antes/depois sao similares (= mesma derivacao, mesma altura). Respeita
    o espirito da regra inviolavel: nao inventa crossings entre leads."""
    out = signal.copy().astype(np.float32)
    W = len(out)
    x = 0
    while x < W:
        if not np.isnan(out[x]):
            x += 1
            continue
        gap_start = x
        while x < W and np.isnan(out[x]):
            x += 1
        gap_end = x
        gap_size = gap_end - gap_start
        if gap_start == 0 or gap_end >= W:
            continue
        y_left = out[gap_start - 1]
        y_right = out[gap_end]
        if np.isnan(y_left) or np.isnan(y_right):
            continue
        if gap_size > max_gap_size:
            continue
        if abs(y_right - y_left) > max_y_diff:
            continue
        for i in range(gap_size):
            t = (i + 1) / float(gap_size + 1)
            out[gap_start + i] = float(y_left + t * (y_right - y_left))
    return out


def _phase1_argmax_per_column(
    component_mask: np.ndarray,
    prob_map: np.ndarray,
    median_window: int = 5,
) -> np.ndarray:
    """Phase 1 alternativo: pra cada coluna, Y = argmax do prob dentro da
    mascara. Garante que cada Y esta EXATAMENTE em cima do pixel onde a
    UNet tem maior confianca de ser tracado.

    NAO sofre do bias do centroide (que cai no meio do span vertical em
    colunas com QRS). NAO sofre do bias do cluster center (que cai no meio
    do cluster vertical).

    Aplica median filter de janela `median_window` cols pra suavizar
    oscilacoes 1-px na espessura do tracado.
    """
    H, W = component_mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    masked_prob = np.where(component_mask, prob_map, -1.0)
    for x in range(W):
        col = masked_prob[:, x]
        max_v = col.max()
        if max_v <= 0:
            continue
        signal[x] = float(np.argmax(col))
    # Median filter pra suavizar (preserva picos, remove ruido 1-px)
    if median_window > 1:
        from scipy.ndimage import median_filter
        valid = ~np.isnan(signal)
        if valid.any():
            sig_filled = np.nan_to_num(signal, nan=0.0)
            filtered = median_filter(sig_filled, size=median_window).astype(np.float32)
            signal[valid] = filtered[valid]
    return signal


# =====================================================================
# FASE 2 — Viterbi Bidirectional (PaperECG)
# Referencia: paper-ecg/.../signal/extraction/viterbi.py::extractSignal
# =====================================================================

def _find_cluster_centers(column_mask: np.ndarray) -> list[int]:
    """Centros de regioes contiguas True em uma coluna 1D.

    Adaptado de paper-ecg::findContiguousRegionCenters.
    Para coluna |--###--##---|, retorna centros [(0+2)/2, ...] etc.
    """
    centers: list[int] = []
    start: Optional[int] = None
    for i, val in enumerate(column_mask):
        if val and start is None:
            start = i
        elif (not val) and start is not None:
            centers.append((start + i - 1) // 2)
            start = None
    if start is not None:
        centers.append((start + len(column_mask) - 1) // 2)
    return centers


def _angle_deg(dx: float, dy: float) -> float:
    """Angulo em graus de um vetor (dx, dy)."""
    norm = sqrt(dx * dx + dy * dy)
    if norm < 1e-9:
        return 0.0
    return asin(dy / norm) * 180.0 / pi


def _phase2_viterbi_forward(
    component_mask: np.ndarray,
    signal_hint: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    hint_margin: int = 80,
    max_lookback: int = 5,
) -> np.ndarray:
    """Viterbi DP forward pass.

    Args:
        component_mask: (H, W) bool.
        signal_hint: (W,) optional — se dado, restringe candidates a Y
                     dentro de hint_margin pixels do hint.
        alpha: peso distancia vs angulo (0=so angulo, 1=so distancia).
        hint_margin: pixels permitidos em torno do hint.
        max_lookback: quantas colunas pra tras pode olhar se a anterior
                      esta vazia.

    Returns:
        (W,) float32 — Y per column, NaN onde sem path.
    """
    H, W = component_mask.shape

    # Build per-column candidates (cluster centers)
    candidates_per_col: list[list[int]] = []
    for x in range(W):
        centers = _find_cluster_centers(component_mask[:, x])
        if signal_hint is not None and not np.isnan(signal_hint[x]):
            centers = [c for c in centers if abs(c - signal_hint[x]) <= hint_margin]
        candidates_per_col.append(centers)

    has_any = any(len(c) > 0 for c in candidates_per_col)
    if not has_any:
        return np.full(W, np.nan, dtype=np.float32)

    # DP table: best_to[(x, y)] = (cumulative_score, predecessor, angle)
    best_to: dict[tuple[int, int], tuple[float, Optional[tuple[int, int]], float]] = {}

    # Base case
    first_col = next(i for i, c in enumerate(candidates_per_col) if len(c) > 0)
    for y in candidates_per_col[first_col]:
        best_to[(first_col, y)] = (0.0, None, 0.0)

    # Forward DP
    for x in range(first_col + 1, W):
        for y in candidates_per_col[x]:
            best_score = float("inf")
            best_prev: Optional[tuple[int, int]] = None
            best_angle = 0.0
            for lookback in range(1, max_lookback + 1):
                px = x - lookback
                if px < first_col:
                    break
                if len(candidates_per_col[px]) == 0:
                    continue  # expand to find non-empty col
                found_in_this_lookback = False
                for py in candidates_per_col[px]:
                    if (px, py) not in best_to:
                        continue
                    prev_score, _, prev_angle = best_to[(px, py)]
                    current_angle = _angle_deg(x - px, y - py)
                    angle_diff = abs(current_angle - prev_angle) / 180.0
                    dist = sqrt((x - px) ** 2 + (y - py) ** 2)
                    edge_cost = alpha * dist + (1.0 - alpha) * angle_diff * dist
                    total = prev_score + edge_cost
                    if total < best_score:
                        best_score = total
                        best_prev = (px, py)
                        best_angle = current_angle
                    found_in_this_lookback = True
                if found_in_this_lookback:
                    break  # don't expand further, found adjacency
            if best_prev is not None:
                best_to[(x, y)] = (best_score, best_prev, best_angle)
            else:
                # Isolated start — like a new path starting here
                best_to[(x, y)] = (0.0, None, 0.0)

    # Find best ending
    last_col_iter = [i for i, c in enumerate(candidates_per_col) if len(c) > 0]
    if not last_col_iter:
        return np.full(W, np.nan, dtype=np.float32)
    last_col = last_col_iter[-1]
    end_candidates = [
        (y, best_to[(last_col, y)][0])
        for y in candidates_per_col[last_col]
        if (last_col, y) in best_to
    ]
    if not end_candidates:
        return np.full(W, np.nan, dtype=np.float32)
    best_end_y, _ = min(end_candidates, key=lambda t: t[1])

    # Backtrack
    path: list[tuple[int, int]] = []
    cur: Optional[tuple[int, int]] = (last_col, best_end_y)
    while cur is not None:
        path.append(cur)
        _, prev, _ = best_to[cur]
        cur = prev
    path.reverse()

    # Build signal — preenche path points + interpolacao linear entre eles,
    # MAS apenas em colunas onde a mascara tem presenca (evita "blocos"
    # visuais por interpolacao sobre regioes vazias).
    signal = np.full(W, np.nan, dtype=np.float32)
    if not path:
        return signal
    xs = np.array([p[0] for p in path], dtype=np.int64)
    ys = np.array([p[1] for p in path], dtype=np.float32)
    for px, py in zip(xs, ys):
        signal[px] = py
    if xs.size >= 2:
        full_range = np.arange(int(xs[0]), int(xs[-1]) + 1)
        interp_values = np.interp(full_range, xs, ys).astype(np.float32)
        # So preenche colunas onde a mascara do componente tem algum pixel
        mask_has_pixel = component_mask.any(axis=0)  # (W,) bool
        for i, x in enumerate(full_range):
            if mask_has_pixel[x]:
                signal[x] = interp_values[i]
            # else: deixa NaN — nao interpola sobre regiao vazia
    return signal


def _phase2_bidirectional(
    component_mask: np.ndarray,
    signal_hint: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Roda Viterbi forward e backward, retorna a media (ignorando NaN)."""
    fwd = _phase2_viterbi_forward(component_mask, signal_hint, alpha)
    # Backward: espelha mask + hint, roda, des-espelha
    mask_flipped = np.fliplr(component_mask)
    hint_flipped = np.flip(signal_hint).copy() if signal_hint is not None else None
    bwd_flipped = _phase2_viterbi_forward(mask_flipped, hint_flipped, alpha)
    bwd = np.flip(bwd_flipped).copy()
    # Media ignorando NaN
    stacked = np.stack([fwd, bwd])
    with np.errstate(all="ignore"):
        out = np.nanmean(stacked, axis=0)
    return out.astype(np.float32)


# =====================================================================
# FASE 3 — QRS Peak Correction
# Referencia: Fortune et al. 2022 (descrito textualmente, sem code)
# =====================================================================

def _phase3_peak_correction(
    signal: np.ndarray,
    component_mask: np.ndarray,
    slope_threshold: float = 5.0,
) -> np.ndarray:
    """Onde o slope da linha extraida e alto, substitui pelo extremo do
    cluster (topo se subindo, base se descendo).

    Em pixel coords (Y cresce pra baixo):
      - slope < -threshold (Y diminuindo, sinal subindo) -> usa MIN Y (topo)
      - slope > +threshold (Y aumentando, sinal descendo) -> usa MAX Y (base)
    """
    out = signal.copy()
    # gradient sobre NaN-replaced signal
    filled = np.nan_to_num(signal, nan=0.0)
    slope = np.gradient(filled)
    for x in range(len(signal)):
        if np.isnan(signal[x]):
            continue
        s = slope[x]
        if abs(s) <= slope_threshold:
            continue
        col = component_mask[:, x]
        positions = np.where(col)[0]
        if positions.size == 0:
            continue
        if s < -slope_threshold:
            # Subindo — captura o topo (menor Y)
            out[x] = float(positions.min())
        elif s > slope_threshold:
            # Descendo — captura a base (maior Y)
            out[x] = float(positions.max())
    return out


# =====================================================================
# SNAP-TO-MASK — garante Y dentro da mascara do componente em CADA coluna
# =====================================================================

def _snap_line_to_mask(
    line: np.ndarray,
    component_mask: np.ndarray,
    max_snap_px: int = 30,
) -> np.ndarray:
    """Pra cada Y nao-NaN da linha, snap pra pixel mais proximo da mascara
    naquela coluna (mesma columna X).

    Se mascara da coluna esta vazia: vira NaN (sinal nao pode estar la).
    Se distancia ate o pixel mais proximo > max_snap_px: tambem vira NaN
    (suspeita de Y errado — melhor NaN que pixel fora do trace).
    """
    out = line.copy().astype(np.float32)
    W = component_mask.shape[1]
    for x in range(W):
        if np.isnan(out[x]):
            continue
        col_mask = component_mask[:, x]
        if not col_mask.any():
            out[x] = np.nan
            continue
        mask_ys = np.where(col_mask)[0]
        target = out[x]
        dists = np.abs(mask_ys - target)
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] > max_snap_px:
            out[x] = np.nan
        else:
            out[x] = float(mask_ys[nearest_idx])
    return out


# =====================================================================
# POST-PROCESSING — Merge fragmentos da mesma derivacao (conservador)
# =====================================================================

def _merge_xdisjoint_fragments(
    lines: list[np.ndarray],
    max_y_distance_px: float = 30.0,
    max_x_gap_px: int = 200,
) -> tuple[list[np.ndarray], list[str]]:
    """Merge conservador de fragmentos X-disjuntos da MESMA derivacao.

    Combina o melhor dos 2 mundos:
      - Como Stenhede match_and_merge_lines: mescla fragmentos da mesma linha
      - Mas com restricao Y estrita: so mescla se Y dos endpoints sao MUITO
        proximos (max_y_distance_px). Isso evita mesclar cross-row.

    Dois fragmentos A e B sao da MESMA derivacao se:
      1. A termina em X_a_end com Y_a_end E B comeca em X_b_start com Y_b_start
      2. X_a_end < X_b_start (B vem depois de A em X)
      3. |X_b_start - X_a_end| <= max_x_gap_px (gap pequeno)
      4. |Y_b_start - Y_a_end| <= max_y_distance_px (Y proximo nos endpoints)

    Quando faz merge: concatena os trechos validos, interpola gap pequeno.

    Args:
        lines: lista (W,) com NaN nos gaps.
        max_y_distance_px: distancia Y maxima entre endpoints adjacentes (px).
        max_x_gap_px: gap X maximo entre fragmentos (px).

    Returns:
        merged: lista de linhas com merges aplicados.
        merge_log: descricoes textuais.
    """
    if len(lines) <= 1:
        return list(lines), []

    # Para cada linha, calcula (x_start, x_end, y_at_start, y_at_end)
    def line_endpoints(line):
        valid = ~np.isnan(line)
        if not valid.any():
            return None
        idx = np.where(valid)[0]
        return int(idx[0]), int(idx[-1]), float(line[idx[0]]), float(line[idx[-1]])

    eps = [line_endpoints(ln) for ln in lines]
    # Indices com endpoints validos
    valid_idx = [i for i, e in enumerate(eps) if e is not None]
    if len(valid_idx) <= 1:
        return list(lines), []

    # Grafo de matches: i -> j se eps[i].end_x < eps[j].start_x e Y/X proximos
    # Greedy: pra cada linha, procura sucessor mais proximo
    parent = {i: i for i in valid_idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merge_log: list[str] = []
    for i in valid_idx:
        x_start_i, x_end_i, y_start_i, y_end_i = eps[i]
        best_j = None
        best_cost = float("inf")
        for j in valid_idx:
            if j == i:
                continue
            x_start_j, x_end_j, y_start_j, y_end_j = eps[j]
            # j vem depois de i?
            if x_start_j <= x_end_i:
                continue
            gap_x = x_start_j - x_end_i
            if gap_x > max_x_gap_px:
                continue
            dy = abs(y_start_j - y_end_i)
            if dy > max_y_distance_px:
                continue
            # cost = gap_x + dy*5 (favor proximidade Y)
            cost = gap_x + dy * 5
            if cost < best_cost:
                best_cost = cost
                best_j = j
        if best_j is not None and find(i) != find(best_j):
            # Union (i gets merged into best_j's set, or vice versa)
            ra, rb = find(i), find(best_j)
            parent[ra] = rb
            merge_log.append(
                f"linha {i} (X=[{x_start_i},{x_end_i}], Y_end={y_end_i:.0f}) "
                f"-> linha {best_j} (X=[{eps[best_j][0]},{eps[best_j][1]}], "
                f"Y_start={eps[best_j][2]:.0f}, dy={abs(eps[best_j][2]-y_end_i):.0f}px)"
            )

    # Agrupar por root
    groups: dict[int, list[int]] = {}
    for i in valid_idx:
        r = find(i)
        groups.setdefault(r, []).append(i)

    # Pra cada grupo, mescla os fragmentos
    W = lines[0].shape[0]
    merged: list[np.ndarray] = []
    for root, members in groups.items():
        if len(members) == 1:
            merged.append(lines[members[0]])
            continue
        # Combina: pra cada coluna, pega o primeiro valor valido entre os membros
        # (em ordem por x_start)
        members_sorted = sorted(members, key=lambda i: eps[i][0])
        combined = np.full(W, np.nan, dtype=np.float32)
        for m in members_sorted:
            ln = lines[m]
            valid = ~np.isnan(ln)
            new_valid = valid & np.isnan(combined)
            combined[new_valid] = ln[new_valid]
        # Interpola gaps pequenos entre fragmentos (so dentro do range total)
        valid = ~np.isnan(combined)
        if valid.any():
            first = int(np.where(valid)[0][0])
            last = int(np.where(valid)[0][-1])
            seg = combined[first:last + 1]
            nan_in_seg = np.isnan(seg)
            if nan_in_seg.any() and not nan_in_seg.all():
                idx = np.arange(seg.shape[0])
                seg[nan_in_seg] = np.interp(
                    idx[nan_in_seg], idx[~nan_in_seg], seg[~nan_in_seg],
                ).astype(np.float32)
                combined[first:last + 1] = seg
        merged.append(combined)

    return merged, merge_log


# =====================================================================
# POST-PROCESSING — Merge linhas duplicadas (cross-row contamination)
# =====================================================================

def _merge_duplicate_lines(
    lines: list[np.ndarray],
    max_y_distance_px: float = 50.0,
    min_x_overlap_pct: float = 0.5,
) -> tuple[list[np.ndarray], list[str]]:
    """Identifica e mescla linhas que extraem a mesma derivacao.

    Duas linhas sao duplicatas se:
      1. Cobrem >min_x_overlap_pct% das mesmas colunas (overlap horizontal)
      2. Distancia media vertical entre elas e < max_y_distance_px

    Quando ha um grupo de duplicatas, mantem a linha com MAIOR cobertura
    (menos NaN) e descarta as outras.

    Args:
        lines: lista de np.ndarray (W,) — uma linha por componente.
        max_y_distance_px: threshold de proximidade vertical (px).
        min_x_overlap_pct: threshold de overlap horizontal (0..1).

    Returns:
        merged_lines: lista deduplicada.
        merge_log: lista de strings descrevendo merges.
    """
    if len(lines) <= 1:
        return list(lines), []

    merged: list[np.ndarray] = []
    used: set[int] = set()
    merge_log: list[str] = []

    for i in range(len(lines)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(lines)):
            if j in used:
                continue
            line_i = lines[i]
            line_j = lines[j]
            valid_i = ~np.isnan(line_i)
            valid_j = ~np.isnan(line_j)
            overlap = int((valid_i & valid_j).sum())
            min_cov = min(int(valid_i.sum()), int(valid_j.sum()))
            if min_cov == 0:
                continue
            overlap_pct = overlap / float(min_cov)
            if overlap_pct < min_x_overlap_pct:
                continue
            both = valid_i & valid_j
            if not both.any():
                continue
            y_distance = float(np.mean(np.abs(line_i[both] - line_j[both])))
            if y_distance < max_y_distance_px:
                group.append(j)
                used.add(j)
        # Manter a linha com maior cobertura do grupo
        best_idx = max(group, key=lambda idx: int((~np.isnan(lines[idx])).sum()))
        merged.append(lines[best_idx])
        if len(group) > 1:
            merge_log.append(
                f"Linhas {group} -> mantida {best_idx} "
                f"({int((~np.isnan(lines[best_idx])).sum())} pts validos, "
                f"y_dist medio={y_distance:.1f}px, overlap={overlap_pct*100:.0f}%)"
            )
        used.add(i)
    return merged, merge_log


# =====================================================================
# ORQUESTRADOR — Roda as 3 fases por connected component
# =====================================================================

def extract_signal_v2(
    signal_prob: np.ndarray,
    label_thresh: float = 0.02,      # validado nos overlays — captura sinal fraco em sombras
    threshold_sum: float = 100.0,    # validado — filtra header text noise
    min_line_width: int = 800,       # validado — apenas linhas full-width (rejeita fragmentos)
    phase1_method: str = "skeleton", # "skeleton" (validado, melhor visual) | "argmax" | "fragmented"
    use_phase2: bool = False,        # Viterbi — desativado (cluster-center bias)
    use_phase3: bool = False,        # QRS correction — desativado (cria "blocos" square-wave)
    viterbi_alpha: float = 0.5,
    slope_threshold: float = 5.0,
    argmax_median_window: int = 5,
    dedup_max_y_distance_px: float = 50.0,
    dedup_min_x_overlap_pct: float = 0.5,
    gap_fill_max_size: int = 200,
    gap_fill_max_y_diff: float = 80.0,
    return_per_phase: bool = False,
) -> dict:
    """Main v2 entrypoint.

    Args:
        signal_prob: (H, W) float — prob map do canal 2 da UNet em [0,1].
        label_thresh: threshold pra binarizar (default 0.1).
        threshold_sum: massa minima de prob pra manter um component.
        min_line_width: min colunas nao-NaN pra manter uma linha extraida.
        use_phase2: aplica Viterbi bidirecional.
        use_phase3: aplica QRS peak correction.
        viterbi_alpha: 0..1, peso distancia vs angulo.
        slope_threshold: px/col, threshold pra ativar peak correction.
        return_per_phase: se True, inclui signals de cada fase no resultado.

    Returns:
        dict com 'lines', 'method', 'stats' (e opcional 'phase1'/'2'/'3').
    """
    H, W = signal_prob.shape

    # Binarize
    binary = signal_prob > label_thresh
    if not binary.any():
        return {
            "lines": [],
            "method": "v2_empty",
            "stats": {"n_components": 0, "n_lines": 0, "coverage_pct": 0.0},
        }

    # Opening morfologico DESLIGADO — erodia Lead I (trace fino) nos overlays.
    # Confiar so no threshold baixo (0.02) + connected components.

    # Label connected components (4-connectivity, igual Stenhede)
    labeled = sk_label(binary, connectivity=1)
    n_components = int(labeled.max())

    lines: list[np.ndarray] = []
    phase1_signals: list[np.ndarray] = []
    phase2_signals: list[np.ndarray] = []
    phase3_signals: list[np.ndarray] = []

    # Maxima extensao vertical aceitavel pra um component "single-lead"
    # (em layout 12x1 com px_per_mm~12, cada band tem ~120 px de altura
    # incluindo QRS extremo. 200 da folga).
    max_component_vertical_extent = 200
    n_split_tall = 0
    n_dropped_tall = 0

    def _split_by_valleys(comp_m: np.ndarray) -> list[np.ndarray]:
        """Tenta split component alto em sub-components por valleys da
        Y-projection (sum por linha)."""
        rows_count = comp_m.sum(axis=1)  # (H,) pixels per Y row
        try:
            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import find_peaks
            smoothed = gaussian_filter1d(rows_count.astype(float), sigma=5)
            # Valleys = minimos locais. Procura nos rows_count invertido.
            # Threshold: vale precisa ter "fundo" significativo (<= 30% dos
            # valores em torno)
            inv = -smoothed
            valleys, _ = find_peaks(inv, distance=40, prominence=smoothed.max() * 0.3)
            nonzero_y = np.where(rows_count > 0)[0]
            if nonzero_y.size == 0:
                return [comp_m]
            y_min, y_max = int(nonzero_y.min()), int(nonzero_y.max())
            valleys = sorted([int(v) for v in valleys if y_min < v < y_max])
            if not valleys:
                return [comp_m]
            cuts = [y_min] + valleys + [y_max + 1]
            sub_masks = []
            for i in range(len(cuts) - 1):
                sub = comp_m.copy()
                sub[:cuts[i]] = False
                sub[cuts[i + 1]:] = False
                if sub.any():
                    sub_masks.append(sub)
            return sub_masks if len(sub_masks) > 1 else [comp_m]
        except Exception:
            return [comp_m]

    # Coleta lista de masks a processar (apos split de tall components)
    masks_to_process: list[np.ndarray] = []
    for lid in range(1, n_components + 1):
        comp_mask = labeled == lid
        comp_sum = signal_prob[comp_mask].sum()
        if comp_sum < threshold_sum:
            continue
        rows_with_mask = np.where(comp_mask.any(axis=1))[0]
        if rows_with_mask.size == 0:
            continue
        v_extent = int(rows_with_mask.max() - rows_with_mask.min())
        if v_extent <= max_component_vertical_extent:
            masks_to_process.append(comp_mask)
            continue
        # Tall — tenta split
        subs = _split_by_valleys(comp_mask)
        if len(subs) == 1:
            # Split falhou — drop
            n_dropped_tall += 1
            continue
        # Split OK — adiciona sub-components que passam no extent check
        added = 0
        for sub in subs:
            rows_sub = np.where(sub.any(axis=1))[0]
            if rows_sub.size == 0:
                continue
            sub_extent = int(rows_sub.max() - rows_sub.min())
            sub_sum = signal_prob[sub].sum()
            if sub_extent <= max_component_vertical_extent and sub_sum >= threshold_sum:
                masks_to_process.append(sub)
                added += 1
        if added > 0:
            n_split_tall += 1
        else:
            n_dropped_tall += 1

    logger.info(
        "v2 components: %d total -> %d aposfiltro/split (split_tall=%d, drop_tall=%d)",
        n_components, len(masks_to_process), n_split_tall, n_dropped_tall,
    )

    for comp_mask in masks_to_process:

        # Fase 1: extracao primaria
        if phase1_method == "skeleton":
            sig_p1 = _phase1_skeleton_per_component(comp_mask, signal_prob)
        elif phase1_method == "argmax":
            sig_p1 = _phase1_argmax_per_column(
                comp_mask, signal_prob, median_window=argmax_median_window,
            )
        else:
            sig_p1 = _phase1_extract_fragmented(comp_mask, signal_prob)
        n_valid = int((~np.isnan(sig_p1)).sum())
        if n_valid < min_line_width:
            continue
        sig = sig_p1
        if return_per_phase:
            phase1_signals.append(sig_p1.copy())

        # Fase 2: Viterbi (opcional)
        if use_phase2:
            sig_p2 = _phase2_bidirectional(comp_mask, signal_hint=sig, alpha=viterbi_alpha)
            # Se Viterbi falhou (tudo NaN), mantem Fase 1
            if (~np.isnan(sig_p2)).sum() >= min_line_width:
                sig = sig_p2
            if return_per_phase:
                phase2_signals.append(sig.copy())

        # Fase 3: Peak correction (opcional)
        if use_phase3:
            sig_p3 = _phase3_peak_correction(sig, comp_mask, slope_threshold)
            sig = sig_p3
            if return_per_phase:
                phase3_signals.append(sig.copy())

        lines.append(sig)

    # Post-processing 1: merge fragmentos X-disjuntos da MESMA derivacao
    # (combina fragmentos consecutivos em X com Y proximo - aumenta cobertura)
    n_before_xmerge = len(lines)
    lines, xmerge_log = _merge_xdisjoint_fragments(
        lines, max_y_distance_px=30.0, max_x_gap_px=200,
    )
    if xmerge_log:
        logger.info(
            "v2 xmerge (fragments): %d -> %d linhas (%d merges)",
            n_before_xmerge, len(lines), len(xmerge_log),
        )

    # Post-processing 2: deduplicar linhas que extraem a mesma derivacao
    # (cross-row duplicates com overlap horizontal)
    n_before_dedup = len(lines)
    lines, merge_log = _merge_duplicate_lines(
        lines,
        max_y_distance_px=dedup_max_y_distance_px,
        min_x_overlap_pct=dedup_min_x_overlap_pct,
    )
    if merge_log:
        logger.info(
            "v2 dedup: %d -> %d linhas (%d merges)",
            n_before_dedup, len(lines), len(merge_log),
        )

    # Post-processing 3: gap-fill condicional (curtos + Y similar = mesma derivacao)
    n_filled_total = 0
    filled_lines = []
    for ln in lines:
        n_nan_before = int(np.isnan(ln).sum())
        ln_filled = _fill_small_gaps_in_signal(
            ln, max_gap_size=gap_fill_max_size, max_y_diff=gap_fill_max_y_diff,
        )
        n_nan_after = int(np.isnan(ln_filled).sum())
        n_filled_total += (n_nan_before - n_nan_after)
        filled_lines.append(ln_filled)
    if n_filled_total > 0:
        logger.info("v2 gap-fill: preencheu %d cols", n_filled_total)
    lines = filled_lines

    # Post-processing 4: SNAP TO TRACE MASK final
    snapped: list[np.ndarray] = []
    n_snap_removed = 0
    for ln in lines:
        snapped_ln = _snap_line_to_mask(ln, binary, max_snap_px=30)
        if (~np.isnan(snapped_ln)).sum() >= min_line_width:
            snapped.append(snapped_ln)
        else:
            n_snap_removed += 1
    if n_snap_removed > 0:
        logger.info(
            "v2 snap-to-mask: %d linhas eliminadas (curtas demais apos snap)",
            n_snap_removed,
        )
    lines = snapped

    if n_dropped_tall > 0:
        logger.info(
            "v2: %d components dropados por altura > %d px (multi-lead)",
            n_dropped_tall, max_component_vertical_extent,
        )

    method_parts = ["v2", phase1_method]
    if use_phase2:
        method_parts.append("viterbi")
    if use_phase3:
        method_parts.append("qrs")
    method_parts.append("xmerge")
    method_parts.append("dedup")
    method_parts.append("snap")
    method = "_".join(method_parts)

    coverage = (
        float(np.mean([(~np.isnan(ln)).mean() for ln in lines]) * 100)
        if lines else 0.0
    )

    result: dict = {
        "lines": lines,
        "method": method,
        "stats": {
            "n_components": n_components,
            "n_lines": len(lines),
            "coverage_pct": coverage,
        },
    }
    if return_per_phase:
        result["phase1"] = phase1_signals
        result["phase2"] = phase2_signals
        result["phase3"] = phase3_signals
    return result


# =====================================================================
# Adapter pra stenhede_adapter — mesmo formato de _signal_extractor_with_offset
# =====================================================================

def extract_lines_v2_with_offset(
    signal_prob_t: torch.Tensor,
    use_stenhede_merge: bool = False,  # FALSE — Stenhede merge mescla cross-row indevidamente em layout 12x1
    **kwargs,
) -> tuple[torch.Tensor, int]:
    """Wrapper que produz output identico ao _signal_extractor_with_offset
    do stenhede_adapter.py — pra drop-in replacement via flag.

    Pipeline:
      1. extract_signal_v2 (3 fases per component) -> linhas curtas, NaN nos gaps
      2. _merge_duplicate_lines (interno) -> remove cross-row duplicates
      3. SignalExtractor.match_and_merge_lines (Hungarian) -> mescla fragmentos
         X-disjuntos da mesma derivacao em UMA linha completa (alta cobertura)
      4. Trim final + retorna

    Args:
        signal_prob_t: (H, W) prob map da UNet canal 2 (apos process_sparse_prob).
        use_stenhede_merge: se True, aplica match_and_merge_lines do Stenhede
                            apos as 3 fases. Default True (recomendado).
        **kwargs: passados pra extract_signal_v2 (use_phase2, use_phase3, etc).

    Returns:
        (lines_tensor: (N, W_trimmed) float32, x_offset: int)
    """
    signal_prob_np = signal_prob_t.detach().cpu().numpy().astype(np.float32)
    W = signal_prob_np.shape[1]
    result = extract_signal_v2(signal_prob_np, **kwargs)
    lines = result["lines"]
    n_components = result["stats"]["n_components"]
    n_lines = result["stats"]["n_lines"]
    logger.info(
        "v2 extract: %d components -> %d lines (method=%s, coverage=%.1f%%)",
        n_components, n_lines, result["method"], result["stats"]["coverage_pct"],
    )
    if not lines:
        return torch.empty((0, W), dtype=torch.float32), 0
    lines_arr = np.stack(lines)  # (N, W)

    # Detecta first/last valid col agregando todas as linhas
    chk = lines_arr.copy()
    chk[chk == 0] = np.nan
    abs_sum = np.nansum(np.abs(chk), axis=0)
    valid = abs_sum > 0
    if not valid.any():
        return torch.empty((0, W), dtype=torch.float32), 0
    nonzero_idx = np.where(valid)[0]
    offset = int(nonzero_idx[0])
    last_idx = int(nonzero_idx[-1])

    # Stack as tensor (NaN preservado) pra passar pro Stenhede merge
    lines_t = torch.from_numpy(lines_arr.astype(np.float32))

    if use_stenhede_merge and lines_t.shape[0] > 0:
        # Aplica match_and_merge_lines do Stenhede:
        #   - Calcula endpoints + heights por linha
        #   - Hungarian matching com cost = dist + diff_height * 30
        #   - Mescla components conectados no grafo de matches
        # Resultado: fragmentos da MESMA derivacao (Y similar, X adjacente)
        # viram UMA linha so com cobertura alta. Linhas em Y diferentes
        # NAO mesclam (penalty alta).
        try:
            # Import vendor SignalExtractor pra ter match_and_merge_lines
            from pathlib import Path as _P
            import sys as _sys
            vendor_root = _P(__file__).resolve().parents[2] / "vendor" / "open_ecg_digitizer"
            if str(vendor_root) not in _sys.path:
                _sys.path.insert(0, str(vendor_root))
            from src.model.signal_extractor import SignalExtractor  # type: ignore

            sextractor = SignalExtractor()
            n_before_merge = lines_t.shape[0]
            merged_list, _overlaps = sextractor.match_and_merge_lines(lines_t)
            if merged_list:
                merged_t = torch.stack(merged_list, dim=0)
                # match_and_merge_lines RETORNA ja trimado (preprocess_lines
                # interno). O offset relativo a imagem original e o `offset`
                # que detectamos antes.
                n_after = merged_t.shape[0]
                w_after = merged_t.shape[1]
                # Coverage: % colunas nao-NaN agregadas
                cov = float((~torch.isnan(merged_t)).float().mean().item() * 100)
                logger.info(
                    "v2 stenhede_merge: %d -> %d linhas, W_trim=%d, "
                    "coverage=%.1f%%",
                    n_before_merge, n_after, w_after, cov,
                )
                return merged_t, offset
            else:
                logger.warning(
                    "stenhede match_and_merge_lines retornou vazio — "
                    "fallback pra v2 sem merge",
                )
        except Exception as e:
            logger.warning(
                "Falha em stenhede_merge (%s) — fallback pra v2 sem merge",
                e,
            )

    # Fallback (sem stenhede_merge): trim manual
    trimmed = lines_arr[:, offset:last_idx + 1]
    # IMPORTANTE: manter NaN (NAO converter para 0). O LeadIdentifier
    # _merge_nonoverlapping_lines downstream usa torch.isnan pra detectar
    # quais colunas se sobrepoem. Se converter NaN->0, todas as colunas
    # parecem "ocupadas" e nada e mesclado.
    return torch.from_numpy(trimmed.astype(np.float32)), offset
