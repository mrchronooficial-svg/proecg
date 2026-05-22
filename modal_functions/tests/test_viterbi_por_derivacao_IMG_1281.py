"""
Testa Viterbi por derivacao na mascara do canal 2 do IMG_1281.

Pipeline:
  1. Carrega mascara B&W (fundo branco, tracado preto)
  2. Detecta bandas (derivacoes) via row-projection + find_peaks
  3. Pra cada banda: crop -> Viterbi -> plot
  4. Salva painel com todas as derivacoes
"""

from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

from pipeline.digitize.viterbi_extractor import extrair_sinal_viterbi  # noqa: E402

MASK_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1281_stenhede_steps\05_canal_2_signal_PB.png")
OUTPUT_DIR = MASK_PATH.parent
OUTPUT_PANEL = OUTPUT_DIR / "09_viterbi_por_derivacao.png"
OUTPUT_BANDS = OUTPUT_DIR / "09_bandas_detectadas.png"


def detect_bands(binary_mask: np.ndarray,
                 sigma: float = 10.0,
                 distance: int = 50,
                 prominence_factor: float = 0.1,
                 buffer_factor: float = 1.2) -> list[tuple[int, int]]:
    """Detecta bandas horizontais via row-projection.
    binary_mask: 2D bool/uint8 (True = tracado).
    Retorna lista de (y_start, y_end)."""
    h = binary_mask.shape[0]
    row_count = binary_mask.sum(axis=1).astype(float)
    if row_count.sum() == 0:
        return []
    smoothed = gaussian_filter1d(row_count, sigma=sigma)
    peaks, _ = find_peaks(smoothed, distance=distance,
                          prominence=smoothed.max() * prominence_factor)
    if peaks.size == 0:
        return []

    # Valleys entre peaks consecutivos
    inner_valleys = []
    for i in range(len(peaks) - 1):
        segment = smoothed[peaks[i]:peaks[i + 1]]
        local_min = int(peaks[i] + np.argmin(segment))
        inner_valleys.append(local_min)

    bands = []
    for i, peak in enumerate(peaks):
        # Boundary superior
        if i == 0:
            if inner_valleys:
                half = int((inner_valleys[0] - peak) * buffer_factor)
                y0 = max(0, peak - half)
            else:
                y0 = 0
        else:
            y0 = inner_valleys[i - 1]
        # Boundary inferior
        if i == len(peaks) - 1:
            if inner_valleys:
                half = int((peak - inner_valleys[-1]) * buffer_factor)
                y1 = min(h, peak + half)
            else:
                y1 = h
        else:
            y1 = inner_valleys[i]
        bands.append((y0, y1))
    return bands


def main() -> int:
    print(f"Carregando: {MASK_PATH}")
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERRO: nao consegui ler {MASK_PATH}")
        return 1
    H, W = mask.shape
    print(f"Mascara: {W}x{H}")

    binary = mask < 128
    bands = detect_bands(binary)
    n_bands = len(bands)
    print(f"Bandas detectadas: {n_bands}")
    for i, (y0, y1) in enumerate(bands):
        print(f"  banda {i+1}: y[{y0}:{y1}] altura={y1-y0}px")

    # --- 1. Visualizacao das bandas detectadas (sanity check) ---
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    ax.imshow(mask, cmap="gray", aspect="auto")
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_bands, 1)))
    for i, ((y0, y1), c) in enumerate(zip(bands, colors)):
        ax.axhspan(y0, y1, color=c, alpha=0.15)
        ax.text(20, (y0 + y1) // 2, f"banda {i+1}",
                color=c * 0.7, fontweight="bold", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_title(f"{n_bands} bandas detectadas via row-projection", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_BANDS), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"Bandas salvas: {OUTPUT_BANDS}")

    # --- 2. Viterbi por banda ---
    signals = []
    stats = []
    for i, (y0, y1) in enumerate(bands):
        band_mask = mask[y0:y1, :]
        t0 = time.perf_counter()
        sig = extrair_sinal_viterbi(band_mask, invert=True)
        dt = time.perf_counter() - t0

        binary_band = band_mask < 128
        cols_with = int(np.any(binary_band, axis=0).sum())
        n_gaps = W - cols_with
        n_valid = int((~np.isnan(sig)).sum())

        signals.append(sig)
        stats.append({
            "band": i + 1, "y0": y0, "y1": y1, "h": y1 - y0,
            "time_s": dt, "n_valid": n_valid, "n_gaps_raw": n_gaps,
            "min": float(np.nanmin(sig)), "max": float(np.nanmax(sig)),
        })
        print(f"  banda {i+1}: {dt:.2f}s | gaps raw={n_gaps} | "
              f"min={stats[-1]['min']:.1f} max={stats[-1]['max']:.1f}")

    # --- 3. Painel: 1 subplot por banda ---
    fig, axes = plt.subplots(n_bands, 1, figsize=(16, 1.6 * n_bands), dpi=100,
                             sharex=True)
    if n_bands == 1:
        axes = [axes]
    fig.suptitle("Viterbi DP por derivacao — IMG_1281",
                 fontsize=14, fontweight="bold", y=0.995)
    for i, (sig, st, ax) in enumerate(zip(signals, stats, axes)):
        ax.plot(sig, color="#1f78b4", linewidth=0.7)
        ax.axhline(0, color="#aaa", linewidth=0.4, linestyle="--")
        ax.set_ylabel(f"banda {st['band']}\n(y={st['y0']}-{st['y1']})",
                      fontsize=8, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, W)
        title = (f"banda {st['band']}: {st['n_valid']}/{W} validos | "
                 f"gaps raw {st['n_gaps_raw']} | range [{st['min']:.0f}, {st['max']:.0f}] | "
                 f"{st['time_s']:.2f}s")
        ax.set_title(title, fontsize=9, loc="left")
    axes[-1].set_xlabel("coluna (x)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(str(OUTPUT_PANEL), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"Painel salvo: {OUTPUT_PANEL}")

    total_time = sum(s["time_s"] for s in stats)
    print(f"\nTotal Viterbi (todas as bandas): {total_time:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
