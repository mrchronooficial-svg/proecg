"""
Benchmark de 5 signal extractors no mesmo heatmap UNet (canal 2) do IMG_1407.

Carrega o cache pre-SignalExtractor ja gerado (signal_prob + normalized image)
e roda cada extractor sobre os MESMOS connected components do heatmap.

Extractors:
  1. Stenhede ORIGINAL (vendor SignalExtractor sem tuning)
  2. ECG-Digitiser MeanY (mean Y de pixels non-zero por coluna)
  3. ECGtizer Fragmented (gap-based fragment selection)
  4. PaperECG Viterbi (DP global, distancia+angulo)
  5. ECG-code Overlap Viterbi (NAO DISPONIVEL — repo nao acessivel)

Output:
  ~/Desktop/Projeto ECG/resultados_teste_v1/benchmark_extractors/
    IMG_1407_benchmark.png   — subplots comparativos
    metricas.txt             — tabela de metricas
    logs.txt                 — logs de execucao
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
import traceback
from math import asin, pi, sqrt
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import label as sk_label

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

# Add repos paths
REPOS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\repos_referencia")
sys.path.insert(0, str(REPOS_ROOT / "ecgtizer"))  # for ecgtizer module
sys.path.insert(0, str(REPOS_ROOT / "paper-ecg" / "src" / "main" / "python"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("benchmark_extractors")

CACHE_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407"
    r"\_cache_pre_signal_extractor.pkl"
)
OUTPUT_DIR = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\benchmark_extractors"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Threshold de binarizacao do heatmap (canal 2)
TRACE_THRESHOLD = 0.1
# Filtros de componente (mesma logica do Stenhede)
MIN_COMPONENT_SUM = 10.0  # massa minima de prob pra manter componente
MIN_LINE_WIDTH = 30        # min colunas com sinal pra manter linha


# =====================================================================
# Wrappers — cada um implementa a logica de UM repo
# =====================================================================

def extract_per_component_meany(
    component_mask: np.ndarray,
    prob_map: np.ndarray,
) -> np.ndarray:
    """ECG-Digitiser — mean Y de pixels non-zero por coluna.

    Source: ECG-Digitiser/src/run/digitize.py linhas 251-256 (vectorise).
    """
    H, W = component_mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        positions = np.where(component_mask[:, x])[0]
        if positions.size > 0:
            signal[x] = float(np.mean(positions))
    return signal


def extract_per_component_fragmented(
    component_mask: np.ndarray,
    prob_map: np.ndarray,
) -> np.ndarray:
    """ECGtizer Fragmented — pega ultimo fragmento contiguo por coluna.

    Source: ecgtizer/ecgtizer/extraction_functions.py linhas 91-129
    (fragmented_extraction).
    Adaptado pra rodar sobre um component mask (em vez do binary global).
    """
    H, W = component_mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        positions = np.where(component_mask[:, x])[0]
        if positions.size == 0:
            continue
        if positions.size == 1:
            signal[x] = float(positions[0])
            continue
        breaks = np.where(np.diff(positions) > 1)[0] + 1
        if breaks.size == 0:
            signal[x] = float(np.mean(positions))
        else:
            fragments = np.split(positions, breaks)
            # ECGtizer original: ultimo fragmento (anti-text heuristic)
            signal[x] = float(np.mean(fragments[-1]))
    return signal


def _find_centers_1d(col: np.ndarray) -> list[int]:
    """Centros de regioes contiguas True (PaperECG findContiguousRegionCenters)."""
    centers = []
    start = None
    for i, v in enumerate(col):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            centers.append((start + i - 1) // 2)
            start = None
    if start is not None:
        centers.append((start + len(col) - 1) // 2)
    return centers


def _angle_deg(dx: float, dy: float) -> float:
    n = sqrt(dx * dx + dy * dy)
    return 0.0 if n < 1e-9 else asin(dy / n) * 180.0 / pi


def extract_per_component_viterbi(
    component_mask: np.ndarray,
    prob_map: np.ndarray,
    alpha: float = 0.5,
    max_lookback: int = 5,
) -> np.ndarray:
    """PaperECG Viterbi — DP global distancia+angulo.

    Source: paper-ecg/.../signal/extraction/viterbi.py linhas 178-243
    (extractSignal). Adaptado: opera por componente, retorna 1 linha (W,).
    """
    H, W = component_mask.shape
    cand_per_col = [_find_centers_1d(component_mask[:, x]) for x in range(W)]
    if not any(cand_per_col):
        return np.full(W, np.nan, dtype=np.float32)

    best_to: dict[tuple[int, int], tuple[float, Optional[tuple[int, int]], float]] = {}
    first_col = next(i for i, c in enumerate(cand_per_col) if len(c) > 0)
    for y in cand_per_col[first_col]:
        best_to[(first_col, y)] = (0.0, None, 0.0)

    for x in range(first_col + 1, W):
        for y in cand_per_col[x]:
            best_score = float("inf")
            best_prev: Optional[tuple[int, int]] = None
            best_angle = 0.0
            for lb in range(1, max_lookback + 1):
                px = x - lb
                if px < first_col:
                    break
                if len(cand_per_col[px]) == 0:
                    continue
                found = False
                for py in cand_per_col[px]:
                    if (px, py) not in best_to:
                        continue
                    prev_s, _, prev_a = best_to[(px, py)]
                    cur_a = _angle_deg(x - px, y - py)
                    angle_pen = abs(cur_a - prev_a) / 180.0
                    dist = sqrt((x - px) ** 2 + (y - py) ** 2)
                    edge = alpha * dist + (1.0 - alpha) * angle_pen * dist
                    tot = prev_s + edge
                    if tot < best_score:
                        best_score = tot
                        best_prev = (px, py)
                        best_angle = cur_a
                    found = True
                if found:
                    break
            if best_prev is not None:
                best_to[(x, y)] = (best_score, best_prev, best_angle)
            else:
                best_to[(x, y)] = (0.0, None, 0.0)

    last_iter = [i for i, c in enumerate(cand_per_col) if len(c) > 0]
    if not last_iter:
        return np.full(W, np.nan, dtype=np.float32)
    last_col = last_iter[-1]
    end_cands = [
        (y, best_to[(last_col, y)][0])
        for y in cand_per_col[last_col]
        if (last_col, y) in best_to
    ]
    if not end_cands:
        return np.full(W, np.nan, dtype=np.float32)
    best_end_y, _ = min(end_cands, key=lambda t: t[1])

    path = []
    cur: Optional[tuple[int, int]] = (last_col, best_end_y)
    while cur is not None:
        path.append(cur)
        _, prev, _ = best_to[cur]
        cur = prev
    path.reverse()

    signal = np.full(W, np.nan, dtype=np.float32)
    if not path:
        return signal
    xs = np.array([p[0] for p in path], dtype=np.int64)
    ys = np.array([p[1] for p in path], dtype=np.float32)
    for px, py in zip(xs, ys):
        signal[px] = py
    if xs.size >= 2:
        full = np.arange(int(xs[0]), int(xs[-1]) + 1)
        interp = np.interp(full, xs, ys).astype(np.float32)
        mask_has = component_mask.any(axis=0)  # mask-restricted interp
        for i, x in enumerate(full):
            if mask_has[x]:
                signal[x] = interp[i]
    return signal


def extract_stenhede_original(signal_prob: np.ndarray) -> list[np.ndarray]:
    """Stenhede SignalExtractor ORIGINAL (vendor, sem tuning, sem skeleton).

    Returns a list de np.ndarrays — uma linha por componente detectado.
    """
    from pipeline.digitize.stenhede_adapter import _ensure_vendor_on_path
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore

    extractor = SignalExtractor()  # defaults puros
    fmap = torch.from_numpy(np.ascontiguousarray(signal_prob)).float()
    lines_t = extractor(fmap)
    if lines_t.shape[0] == 0:
        return []
    return [lines_t[i].cpu().numpy().astype(np.float32) for i in range(lines_t.shape[0])]


def run_per_component_method(
    method_name: str,
    extractor_fn,
    component_masks: list[np.ndarray],
    prob_map: np.ndarray,
) -> tuple[list[np.ndarray], float]:
    """Roda um extractor (single-line) em cada component, retorna lista de
    linhas (W,) e tempo total.
    """
    t0 = time.perf_counter()
    lines: list[np.ndarray] = []
    for cm in component_masks:
        try:
            sig = extractor_fn(cm, prob_map)
        except Exception as e:
            logger.warning("%s falhou num componente: %s", method_name, e)
            continue
        if (~np.isnan(sig)).sum() < MIN_LINE_WIDTH:
            continue
        lines.append(sig)
    return lines, time.perf_counter() - t0


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    if not CACHE_PATH.is_file():
        logger.error("Cache nao encontrado em %s", CACHE_PATH)
        return 1

    logger.info("Carregando cache do IMG_1407...")
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    signal_prob: np.ndarray = cache["signal_prob"]
    normalized: np.ndarray = cache["normalized"]
    logger.info(
        "signal_prob shape=%s dtype=%s, normalized shape=%s",
        signal_prob.shape, signal_prob.dtype, normalized.shape,
    )

    H, W = signal_prob.shape
    binary = signal_prob > TRACE_THRESHOLD
    if not binary.any():
        logger.error("Nenhum pixel acima do threshold %.2f", TRACE_THRESHOLD)
        return 1

    # Label connected components (mesma base pra todos os extractors single-line)
    labeled = sk_label(binary, connectivity=1)
    n_comp_total = int(labeled.max())

    # Filtra components por massa de prob (mesma logica do Stenhede)
    component_masks: list[np.ndarray] = []
    for lid in range(1, n_comp_total + 1):
        m = labeled == lid
        if signal_prob[m].sum() >= MIN_COMPONENT_SUM:
            component_masks.append(m)
    logger.info(
        "Components: %d total -> %d apos filtro de massa",
        n_comp_total, len(component_masks),
    )

    # Run all extractors
    results: dict[str, tuple[list[np.ndarray], float, str]] = {}

    # 1. Stenhede ORIGINAL (vendor, no tuning) — opera sobre signal_prob direto
    logger.info("[1/5] Stenhede ORIGINAL...")
    try:
        t0 = time.perf_counter()
        lines = extract_stenhede_original(signal_prob)
        dt = time.perf_counter() - t0
        results["1_stenhede_original"] = (lines, dt, "OK")
    except Exception as e:
        logger.error("Stenhede ORIGINAL falhou: %s", e)
        results["1_stenhede_original"] = ([], 0.0, f"ERR: {e}")

    # 2. ECG-Digitiser MeanY
    logger.info("[2/5] ECG-Digitiser MeanY...")
    try:
        lines, dt = run_per_component_method(
            "MeanY", extract_per_component_meany, component_masks, signal_prob,
        )
        results["2_ecgdigitiser_meany"] = (lines, dt, "OK")
    except Exception as e:
        logger.error("MeanY falhou: %s", e)
        results["2_ecgdigitiser_meany"] = ([], 0.0, f"ERR: {e}")

    # 3. ECGtizer Fragmented
    logger.info("[3/5] ECGtizer Fragmented...")
    try:
        lines, dt = run_per_component_method(
            "Fragmented", extract_per_component_fragmented, component_masks, signal_prob,
        )
        results["3_ecgtizer_fragmented"] = (lines, dt, "OK")
    except Exception as e:
        logger.error("Fragmented falhou: %s", e)
        results["3_ecgtizer_fragmented"] = ([], 0.0, f"ERR: {e}")

    # 4. PaperECG Viterbi
    logger.info("[4/5] PaperECG Viterbi...")
    try:
        lines, dt = run_per_component_method(
            "Viterbi", extract_per_component_viterbi, component_masks, signal_prob,
        )
        results["4_paperecg_viterbi"] = (lines, dt, "OK")
    except Exception as e:
        logger.error("Viterbi falhou: %s", e)
        results["4_paperecg_viterbi"] = ([], 0.0, f"ERR: {e}")

    # 5. ECG-code Overlap Viterbi (NAO DISPONIVEL)
    results["5_ecgcode_overlap_viterbi"] = (
        [], 0.0,
        "SKIP: repo masoudrahimi39/ECG-code nao acessivel (404 nao existe)",
    )
    logger.warning("[5/5] ECG-code: SKIPPED (repo 404 - nao acessivel)")

    # =====================================================================
    # Generate visualization: 5 subplots verticais
    # =====================================================================
    logger.info("Gerando visualizacao comparativa...")

    color_map = {
        "1_stenhede_original": "#1f77b4",       # azul
        "2_ecgdigitiser_meany": "#2ca02c",      # verde
        "3_ecgtizer_fragmented": "#ff7f0e",     # laranja
        "4_paperecg_viterbi": "#9467bd",        # roxo
        "5_ecgcode_overlap_viterbi": "#d62728", # vermelho (n/a)
    }
    label_map = {
        "1_stenhede_original": "Stenhede ORIGINAL (vendor, sem tuning)",
        "2_ecgdigitiser_meany": "ECG-Digitiser — Mean Y nao-zero",
        "3_ecgtizer_fragmented": "ECGtizer — Fragmented (last fragment)",
        "4_paperecg_viterbi": "PaperECG — Viterbi DP (dist+angulo)",
        "5_ecgcode_overlap_viterbi": "ECG-code — Overlap Viterbi (n/a)",
    }

    methods_order = list(color_map.keys())
    n_methods = len(methods_order)

    fig = plt.figure(figsize=(16, 4 * n_methods), dpi=110)
    fig.suptitle(
        "BENCHMARK SIGNAL EXTRACTORS — IMG_1407\n"
        f"Heatmap canal 2 da UNet (binarizado em {TRACE_THRESHOLD}), "
        f"{len(component_masks)} components",
        fontsize=14, fontweight="bold", y=0.998,
    )

    for i, method_id in enumerate(methods_order):
        ax = fig.add_subplot(n_methods, 1, i + 1)
        lines, dt, status = results[method_id]
        color = color_map[method_id]
        label = label_map[method_id]

        # Background: imagem undistorted em cinza claro
        ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB), alpha=0.55)

        # Overlay: linhas extraidas
        n_valid_pts = 0
        n_total_pts = 0
        for line in lines:
            n_valid_pts += int((~np.isnan(line)).sum())
            n_total_pts += int(line.size)
            xs = np.arange(line.size).astype(np.float64)
            ax.plot(xs, line, color=color, linewidth=0.9, alpha=0.95)

        cov = (n_valid_pts / max(n_total_pts, 1)) * 100.0
        nan_pct = 100.0 - cov
        n_lines = len(lines)

        title = (
            f"{i+1}. {label}  —  {n_lines} linhas  |  "
            f"cobertura {cov:.1f}%  |  NaN {nan_pct:.1f}%  |  "
            f"{dt:.2f}s  |  status: {status[:60]}"
        )
        ax.set_title(title, fontsize=10, loc="left")
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = OUTPUT_DIR / "IMG_1407_benchmark.png"
    fig.savefig(str(out_png), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("Salvo: %s", out_png)

    # =====================================================================
    # Individual figures — uma imagem grande por extractor
    # =====================================================================
    for i, method_id in enumerate(methods_order):
        lines, dt, status = results[method_id]
        color = color_map[method_id]
        label = label_map[method_id]

        fig_i, ax = plt.subplots(figsize=(20, 11), dpi=120)
        ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB), alpha=0.55)

        n_valid_pts = 0
        n_total_pts = 0
        for line in lines:
            n_valid_pts += int((~np.isnan(line)).sum())
            n_total_pts += int(line.size)
            xs = np.arange(line.size).astype(np.float64)
            ax.plot(xs, line, color=color, linewidth=1.0, alpha=0.95)

        cov = (n_valid_pts / max(n_total_pts, 1)) * 100.0
        nan_pct = 100.0 - cov
        n_lines = len(lines)

        ax.set_title(
            f"{i+1}. {label}\n"
            f"{n_lines} linhas  |  cobertura {cov:.1f}%  |  NaN {nan_pct:.1f}%  |  "
            f"tempo {dt:.2f}s  |  {status[:50]}",
            fontsize=12, fontweight="bold",
        )
        ax.axis("off")
        fig_i.tight_layout()
        out_path = OUTPUT_DIR / f"IMG_1407_{method_id}.png"
        fig_i.savefig(str(out_path), bbox_inches="tight", dpi=120, facecolor="white")
        plt.close(fig_i)
        logger.info("Salvo: %s", out_path)

    # =====================================================================
    # Save metrics.txt
    # =====================================================================
    metrics_path = OUTPUT_DIR / "metricas.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("BENCHMARK SIGNAL EXTRACTORS — IMG_1407\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Heatmap shape: {signal_prob.shape}\n")
        f.write(f"Trace threshold (binarizacao): {TRACE_THRESHOLD}\n")
        f.write(f"Components totais: {n_comp_total}\n")
        f.write(f"Components apos filtro de massa (>{MIN_COMPONENT_SUM}): "
                f"{len(component_masks)}\n\n")
        f.write(
            f"{'Extractor':<42} {'Linhas':>7} {'Cobertura':>10} "
            f"{'NaN %':>7} {'Tempo':>9} Status\n"
        )
        f.write("-" * 100 + "\n")
        for method_id in methods_order:
            lines, dt, status = results[method_id]
            label = label_map[method_id]
            n = len(lines)
            n_valid = sum(int((~np.isnan(l)).sum()) for l in lines)
            n_total = sum(int(l.size) for l in lines)
            cov = (n_valid / max(n_total, 1)) * 100.0
            nan_pct = 100.0 - cov
            f.write(
                f"{label[:42]:<42} {n:>7d} {cov:>9.1f}% "
                f"{nan_pct:>6.1f}% {dt:>8.2f}s  {status[:40]}\n"
            )
    logger.info("Salvo: %s", metrics_path)

    # =====================================================================
    # logs.txt
    # =====================================================================
    logs_path = OUTPUT_DIR / "logs.txt"
    with logs_path.open("w", encoding="utf-8") as f:
        f.write("BENCHMARK — Logs de execucao\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input: {CACHE_PATH}\n")
        f.write(f"signal_prob.shape: {signal_prob.shape}\n")
        f.write(f"signal_prob range: [{signal_prob.min():.3f}, {signal_prob.max():.3f}]\n")
        f.write(f"normalized.shape: {normalized.shape}\n\n")
        f.write(f"Components apos filtro: {len(component_masks)}\n\n")
        f.write("Por extractor:\n")
        for method_id in methods_order:
            lines, dt, status = results[method_id]
            f.write(f"  {method_id}: {len(lines)} linhas, {dt:.2f}s, status={status}\n")
    logger.info("Salvo: %s", logs_path)

    logger.info("Benchmark concluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
