"""
Sobreposicao limpa por derivacao: cada lead band identificado via
connected components, cada um com cor distinta.

Para cada metodo:
  - Identifica connected components no binary (cada um = uma derivacao)
  - Filtra components por massa e altura
  - Roda o extractor em CADA componente -> 1 linha por lead
  - Plota cada linha em cor distinta sobre o ECG

Outputs em ~/Desktop/Projeto ECG/resultados_teste_v1/benchmark_extractors/overlays_per_lead/
  signal_skeleton.png
  signal_thinning.png
  signal_borda_superior.png
  signal_media_bordas.png
  signal_4_metodos_overlay.png
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.measure import label as sk_label
from skimage.morphology import closing, opening, rectangle, skeletonize, thin

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_overlay_per_lead")

ECG_NAMES = ["IMG_1407", "IMG_1473", "IMG_1471"]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_ROOT = RESULTS_ROOT / "benchmark_extractors" / "overlays_per_lead"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.02       # threshold sensivel sem ser excessivo
MIN_MASS = 100.0
MIN_EXTENT_X = 30
MAX_EXTENT_Y = 200
MIN_LINE_COVERAGE = 800
HORIZONTAL_CLOSING_PX = 1  # sem closing — gap-fill condicional no signal e mais seguro


# Extracao per-component (todos retornam (W,) com NaN fora do component)
def extract_skeleton(component_mask):
    skel = skeletonize(component_mask > 0)
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


def extract_thinning(component_mask):
    thinned = thin(component_mask > 0)
    H, W = component_mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(thinned[:, x])[0]
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


def extract_borda_superior(component_mask):
    H, W = component_mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(component_mask[:, x] > 0)[0]
        if ys.size > 0:
            signal[x] = float(ys[0])
    return signal


def extract_media_bordas(component_mask):
    H, W = component_mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(component_mask[:, x] > 0)[0]
        if ys.size > 0:
            signal[x] = float((ys[0] + ys[-1]) / 2.0)
    return signal


def _enforce_in_mask(signal, mask):
    H, W = mask.shape
    out = signal.copy()
    fora = 0
    for x in range(W):
        if np.isnan(out[x]):
            continue
        y = int(round(float(out[x])))
        if not (0 <= y < H and mask[y, x]):
            out[x] = np.nan
            fora += 1
    return out, fora


def _fill_small_gaps_in_signal(
    signal: np.ndarray,
    max_gap_size: int = 30,
    max_y_diff: float = 20.0,
) -> np.ndarray:
    """Interpolacao linear em gaps PEQUENOS de NaN no sinal, onde os Y
    antes/depois do gap sao similares (= mesma derivacao, mesma altura).

    Respeita o espirito da regra inviolavel: nao inventa crossings entre
    leads. So fecha pequenas descontinuidades dentro de UM mesmo lead.
    """
    out = signal.copy().astype(np.float32)
    W = len(out)
    x = 0
    while x < W:
        if not np.isnan(out[x]):
            x += 1
            continue
        # Encontrou inicio de gap (NaN)
        gap_start = x
        while x < W and np.isnan(out[x]):
            x += 1
        gap_end = x  # exclusive
        gap_size = gap_end - gap_start
        # Bordas validas
        if gap_start == 0 or gap_end >= W:
            continue  # gap nas pontas, deixa
        y_left = out[gap_start - 1]
        y_right = out[gap_end]
        if np.isnan(y_left) or np.isnan(y_right):
            continue
        if gap_size > max_gap_size:
            continue
        if abs(y_right - y_left) > max_y_diff:
            continue
        # Interpola linearmente
        for i in range(gap_size):
            t = (i + 1) / float(gap_size + 1)
            out[gap_start + i] = float(y_left + t * (y_right - y_left))
    return out


def _split_tall_by_valleys(comp_mask: np.ndarray) -> list[np.ndarray]:
    """Splita component tall por valleys da Y-projection."""
    rows_count = comp_mask.sum(axis=1)
    smoothed = gaussian_filter1d(rows_count.astype(float), sigma=5)
    inv = -smoothed
    valleys, _ = find_peaks(inv, distance=40, prominence=smoothed.max() * 0.3)
    nonzero_y = np.where(rows_count > 0)[0]
    if nonzero_y.size == 0:
        return [comp_mask]
    y_min, y_max = int(nonzero_y.min()), int(nonzero_y.max())
    vals = sorted([int(v) for v in valleys if y_min < v < y_max])
    if not vals:
        return [comp_mask]
    cuts = [y_min] + vals + [y_max + 1]
    subs = []
    for i in range(len(cuts) - 1):
        sub = comp_mask.copy()
        sub[:cuts[i]] = False
        sub[cuts[i + 1]:] = False
        if sub.any():
            subs.append(sub)
    return subs if len(subs) > 1 else [comp_mask]


def _get_component_masks(signal_prob: np.ndarray):
    """Labela components, splita tall por valleys, filtra por massa/altura."""
    binary = signal_prob > THRESHOLD
    # Closing morfologico HORIZONTAL — DESLIGADO. Closing achata picos QRS
    # porque a dilatacao + erosao 1xN exige N cols contiguas no mesmo Y, e
    # picos QRS tem apenas 1-3 cols na altura do pico. Apaga os picos.
    # Em vez disso confiar so no threshold baixo (0.02) pra capturar sinal
    # fraco no vinco/sombra do papel.
    if HORIZONTAL_CLOSING_PX > 1:
        try:
            binary_closed = closing(binary, rectangle(1, HORIZONTAL_CLOSING_PX))
            if binary_closed.any():
                logger.info(
                    "Closing horizontal (1x%d): %d -> %d pixels",
                    HORIZONTAL_CLOSING_PX, int(binary.sum()), int(binary_closed.sum()),
                )
                binary = binary_closed
        except Exception as e:
            logger.warning("Closing falhou: %s", e)

    labeled = sk_label(binary, connectivity=1)
    n = int(labeled.max())
    n_dropped = 0
    n_split = 0

    masks_to_process: list[np.ndarray] = []
    for lid in range(1, n + 1):
        cm = labeled == lid
        if signal_prob[cm].sum() < MIN_MASS:
            continue
        rows = np.where(cm.any(axis=1))[0]
        cols = np.where(cm.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        ex_x = cols.max() - cols.min()
        ex_y = rows.max() - rows.min()
        if ex_x < MIN_EXTENT_X:
            continue
        if ex_y <= MAX_EXTENT_Y:
            masks_to_process.append(cm)
            continue
        # Tall — tenta split por valleys
        subs = _split_tall_by_valleys(cm)
        if len(subs) == 1:
            # Split falhou — em vez de dropar, divide manualmente em
            # bandas horizontais de ~120 px (altura tipica de lead band)
            band_h = 120
            n_bands = int(np.ceil(ex_y / band_h))
            for b in range(n_bands):
                y0 = int(rows.min() + b * band_h)
                y1 = int(min(rows.min() + (b + 1) * band_h, rows.max() + 1))
                sub = cm.copy()
                sub[:y0] = False
                sub[y1:] = False
                if not sub.any():
                    continue
                sub_rows = np.where(sub.any(axis=1))[0]
                sub_cols = np.where(sub.any(axis=0))[0]
                if sub_rows.size == 0 or sub_cols.size == 0:
                    continue
                sub_ex_x = sub_cols.max() - sub_cols.min()
                sub_mass = signal_prob[sub].sum()
                if sub_ex_x >= MIN_EXTENT_X and sub_mass >= MIN_MASS:
                    masks_to_process.append(sub)
            logger.info(
                "  Tall comp (y=%d-%d, x=%d) dividido em %d bandas horizontais (split-by-valleys falhou)",
                rows.min(), rows.max(), ex_x, n_bands,
            )
            n_split += 1
            continue
        added = 0
        for sub in subs:
            rows_s = np.where(sub.any(axis=1))[0]
            cols_s = np.where(sub.any(axis=0))[0]
            if rows_s.size == 0 or cols_s.size == 0:
                continue
            sub_ex_x = cols_s.max() - cols_s.min()
            sub_ex_y = rows_s.max() - rows_s.min()
            sub_mass = signal_prob[sub].sum()
            if (sub_ex_x >= MIN_EXTENT_X and sub_ex_y <= MAX_EXTENT_Y
                    and sub_mass >= MIN_MASS):
                masks_to_process.append(sub)
                added += 1
        if added > 0:
            n_split += 1
        else:
            n_dropped += 1

    logger.info(
        "Components: %d totais -> %d validos (split=%d, drop=%d)",
        n, len(masks_to_process), n_split, n_dropped,
    )
    # Ordena por Y medio (top-to-bottom)
    masks_to_process.sort(key=lambda m: float(np.where(m.any(axis=1))[0].mean()))
    # Debug: log bounds de cada component
    for i, m in enumerate(masks_to_process):
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        y_mid = float(rows.mean())
        x_w = cols.max() - cols.min()
        mass = signal_prob[m].sum()
        logger.info(
            "  Comp #%d: y_mid=%.0f, x_extent=%d, mass=%.0f",
            i + 1, y_mid, x_w, mass,
        )
    return masks_to_process, binary


def _xmerge_fragments(
    lines: list[np.ndarray],
    max_y_distance_px: float = 30.0,
    max_x_gap_px: int = 400,
) -> list[np.ndarray]:
    """Merge X-disjoint fragments com Y proximo (= mesma derivacao)."""
    if len(lines) <= 1:
        return lines

    def ep(line):
        valid = ~np.isnan(line)
        if not valid.any():
            return None
        idx = np.where(valid)[0]
        return int(idx[0]), int(idx[-1]), float(line[idx[0]]), float(line[idx[-1]])

    eps = [ep(ln) for ln in lines]
    parent = {i: i for i, e in enumerate(eps) if e is not None}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in parent.keys():
        x_si, x_ei, _, y_ei = eps[i]
        best_j = None
        best_cost = float("inf")
        for j in parent.keys():
            if j == i:
                continue
            x_sj, _, y_sj, _ = eps[j]
            if x_sj <= x_ei:
                continue
            gap_x = x_sj - x_ei
            if gap_x > max_x_gap_px:
                continue
            dy = abs(y_sj - y_ei)
            if dy > max_y_distance_px:
                continue
            cost = gap_x + dy * 5
            if cost < best_cost:
                best_cost = cost
                best_j = j
        if best_j is not None and find(i) != find(best_j):
            parent[find(i)] = find(best_j)

    groups: dict[int, list[int]] = {}
    for i in parent.keys():
        r = find(i)
        groups.setdefault(r, []).append(i)

    W = lines[0].shape[0]
    merged: list[np.ndarray] = []
    for root, members in groups.items():
        if len(members) == 1:
            merged.append(lines[members[0]])
            continue
        members_sorted = sorted(members, key=lambda i: eps[i][0])
        combined = np.full(W, np.nan, dtype=np.float32)
        for m in members_sorted:
            ln = lines[m]
            valid = ~np.isnan(ln)
            new_valid = valid & np.isnan(combined)
            combined[new_valid] = ln[new_valid]
        merged.append(combined)
    return merged


def _save_per_lead_overlay(
    lines, normalized, palette_cmap, title, out_path,
):
    fig, ax = plt.subplots(figsize=(18, 10), dpi=120)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    palette = plt.get_cmap(palette_cmap, max(len(lines), 12))
    total_cov = 0
    total_w = 0
    for i, sig in enumerate(lines):
        color = palette(i % palette.N)
        xs = np.arange(sig.size).astype(np.float64)
        ax.plot(xs, sig, color=color, linewidth=1.0, alpha=0.95,
                label=f"Lead {i+1}")
        total_cov += int((~np.isnan(sig)).sum())
        total_w = sig.size
    ax.set_title(
        f"{title}  —  {len(lines)} derivacoes  "
        f"(total cov: {total_cov} cols)",
        fontsize=14, fontweight="bold",
    )
    if len(lines) <= 20:
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)


def _save_per_lead_overlay_vibrant(
    lines, normalized, colors, title, out_path,
):
    """Overlay com paleta de cores vibrantes (lista fixa)."""
    fig, ax = plt.subplots(figsize=(18, 10), dpi=120)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    total_cov = 0
    for i, sig in enumerate(lines):
        color = colors[i % len(colors)]
        xs = np.arange(sig.size).astype(np.float64)
        ax.plot(xs, sig, color=color, linewidth=1.4, alpha=0.95,
                label=f"Lead {i+1}")
        total_cov += int((~np.isnan(sig)).sum())
    ax.set_title(
        f"{title}  —  {len(lines)} derivacoes  (total cov: {total_cov} cols)",
        fontsize=14, fontweight="bold",
    )
    if len(lines) <= 20:
        ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)


def _process_ecg(stem: str) -> None:
    cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
    if not cache_path.is_file():
        logger.error("Cache nao existe pra %s: %s", stem, cache_path)
        return
    output_dir = OUTPUT_ROOT / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("===== %s =====", stem)
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    signal_prob = cache["signal_prob"]
    normalized = cache["normalized"]
    H, W = signal_prob.shape

    component_masks, binary = _get_component_masks(signal_prob)
    logger.info("Components validos: %d (filtros: massa>%g, x>%d, y<%d)",
                len(component_masks), MIN_MASS, MIN_EXTENT_X, MAX_EXTENT_Y)

    methods = [
        ("skeleton",       extract_skeleton,       "tab20"),
        ("thinning",       extract_thinning,       "tab20"),
        ("borda_superior", extract_borda_superior, "tab20"),
        ("media_bordas",   extract_media_bordas,   "tab20"),
    ]

    # Paleta de 12 cores VIBRANTES e distintas (anti-tab20 que tem pretos)
    VIBRANT_COLORS = [
        "#e6194B",  # vermelho vivido
        "#3cb44b",  # verde
        "#ffe119",  # amarelo
        "#4363d8",  # azul
        "#f58231",  # laranja
        "#911eb4",  # roxo
        "#42d4f4",  # cyan
        "#f032e6",  # magenta
        "#bfef45",  # lima
        "#469990",  # teal
        "#9A6324",  # marrom
        "#800000",  # vinho
    ]

    results: dict[str, list[np.ndarray]] = {}
    for name, fn, _cmap in methods:
        t0 = time.perf_counter()
        lines = []
        total_fora = 0
        for cm in component_masks:
            sig = fn(cm.astype(np.uint8))
            sig, fora = _enforce_in_mask(sig, binary)
            total_fora += fora
            if int((~np.isnan(sig)).sum()) >= MIN_EXTENT_X:
                lines.append(sig)
        # xmerge: junta fragmentos de mesma derivacao (V3 splitado, etc)
        n_before = len(lines)
        lines = _xmerge_fragments(lines, max_y_distance_px=30, max_x_gap_px=600)
        if len(lines) != n_before:
            logger.info("  xmerge: %d -> %d lines", n_before, len(lines))
        # Filtra linhas curtas (fragmentos que nao mesclaram)
        n_before_cov = len(lines)
        lines = [s for s in lines if int((~np.isnan(s)).sum()) >= MIN_LINE_COVERAGE]
        if len(lines) != n_before_cov:
            logger.info("  coverage filter (>=%d cols): %d -> %d lines",
                        MIN_LINE_COVERAGE, n_before_cov, len(lines))
        # Fill condicional de gaps pequenos no sinal (respeita continuidade
        # do lead — nao inventa crossings entre leads diferentes).
        before_gap_total = sum(int(np.isnan(s).sum()) for s in lines)
        lines = [_fill_small_gaps_in_signal(s, max_gap_size=200, max_y_diff=80)
                 for s in lines]
        after_gap_total = sum(int(np.isnan(s).sum()) for s in lines)
        filled = before_gap_total - after_gap_total
        if filled > 0:
            logger.info("  fill_small_gaps: preencheu %d cols", filled)
        # Reordena por Y medio top-to-bottom
        lines.sort(key=lambda s: float(np.nanmean(s)) if not np.all(np.isnan(s)) else 0)
        dt = time.perf_counter() - t0
        results[name] = lines
        total_cov = sum(int((~np.isnan(s)).sum()) for s in lines)
        logger.info(
            "%s: %d lines, cov=%d cols, fora=%d, %.1fs",
            name, len(lines), total_cov, total_fora, dt,
        )
        _save_per_lead_overlay_vibrant(
            lines, normalized, VIBRANT_COLORS,
            f"{stem} — {name} — sinal por derivacao",
            output_dir / f"signal_{name}.png",
        )

    # Imagem combinada
    fig, ax = plt.subplots(figsize=(20, 11), dpi=120)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    method_colors = {
        "skeleton":       "#00aa44",
        "thinning":       "#9933cc",
        "borda_superior": "#ff8800",
        "media_bordas":   "#cc0000",
    }
    legend_handles = []
    for name, lines in results.items():
        color = method_colors[name]
        total_cov = sum(int((~np.isnan(s)).sum()) for s in lines)
        for sig in lines:
            xs = np.arange(sig.size).astype(np.float64)
            ax.plot(xs, sig, color=color, linewidth=0.7, alpha=0.6)
        legend_handles.append(plt.Line2D(
            [0], [0], color=color, linewidth=2,
            label=f"{name} ({len(lines)} leads, {total_cov} cols)",
        ))
    ax.set_title(
        f"Comparacao por derivacao — 4 metodos simples ({stem})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(output_dir / "signal_4_metodos_overlay.png"),
                bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    logger.info("[%s] Concluido. Outputs em %s", stem, output_dir)


def main() -> int:
    for stem in ECG_NAMES:
        _process_ecg(stem)
    logger.info("Tudo concluido. Outputs em %s/<ECG>/", OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
