"""
Sobreposicao limpa: sinal extraido por cada metodo simples diretamente
sobre o ECG undistorted. Sem comparacao, sem heatmap — apenas a linha.

Outputs em ~/Desktop/Projeto ECG/resultados_teste_v1/benchmark_extractors/overlays/
  signal_skeleton.png         — esqueleto verde
  signal_thinning.png         — thinning roxo
  signal_borda_superior.png   — borda superior laranja
  signal_media_bordas.png     — midpoint vermelho (com pontos FORA)
  signal_4_metodos_overlay.png — todos os 4 em cores distintas no mesmo ECG
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
from skimage.morphology import skeletonize, thin

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_overlay")

CACHE_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407"
    r"\_cache_pre_signal_extractor.pkl"
)
OUTPUT_DIR = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\benchmark_extractors\overlays"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.1


def extract_skeleton(mask):
    skel = skeletonize(mask > 0)
    H, W = mask.shape
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


def extract_thinning(mask):
    thinned = thin(mask > 0)
    H, W = mask.shape
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


def extract_borda_superior(mask):
    H, W = mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size > 0:
            signal[x] = float(ys[0])
    return signal


def extract_media_bordas(mask):
    H, W = mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size > 0:
            signal[x] = float((ys[0] + ys[-1]) / 2.0)
    return signal


def _save_overlay(signal, normalized, color, title, out_path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    xs = np.arange(len(signal))
    ax.plot(xs, signal, color=color, linewidth=0.9, alpha=0.95)
    coverage = int((~np.isnan(signal)).sum())
    ax.set_title(
        f"{title}  —  {coverage}/{len(signal)} cols ({100.0*coverage/len(signal):.1f}%)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)


def main() -> int:
    if not CACHE_PATH.is_file():
        logger.error("Cache nao existe")
        return 1
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    signal_prob = cache["signal_prob"]
    normalized = cache["normalized"]
    mask = (signal_prob > THRESHOLD).astype(np.uint8)
    logger.info("Mascara: %d pixels", int(mask.sum()))

    methods = [
        ("skeleton",       extract_skeleton,       "#00aa44"),  # verde
        ("thinning",       extract_thinning,       "#9933cc"),  # roxo
        ("borda_superior", extract_borda_superior, "#ff8800"),  # laranja
        ("media_bordas",   extract_media_bordas,   "#cc0000"),  # vermelho
    ]

    signals: dict[str, np.ndarray] = {}
    for name, fn, color in methods:
        t0 = time.perf_counter()
        sig = fn(mask)
        dt = time.perf_counter() - t0
        # Garante regra inviolavel: NaN pontos fora da mascara
        H, W = mask.shape
        for x in range(W):
            if np.isnan(sig[x]):
                continue
            y = int(round(float(sig[x])))
            if not (0 <= y < H and mask[y, x]):
                sig[x] = np.nan
        signals[name] = sig
        coverage = int((~np.isnan(sig)).sum())
        logger.info("%s: cobertura %d/%d (%.1f%%) em %.1fs",
                    name, coverage, W, 100.0*coverage/W, dt)

        _save_overlay(
            sig, normalized, color,
            f"{name}",
            OUTPUT_DIR / f"signal_{name}.png",
        )

    # Imagem combinada — todos os 4 metodos no mesmo ECG, cor distinta
    fig, ax = plt.subplots(figsize=(18, 10), dpi=120)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    xs_all = np.arange(mask.shape[1])
    for name, _fn, color in methods:
        sig = signals[name]
        cov = int((~np.isnan(sig)).sum())
        ax.plot(xs_all, sig, color=color, linewidth=0.8, alpha=0.7,
                label=f"{name} ({100.0*cov/mask.shape[1]:.1f}% cov)")
    ax.set_title(
        "Comparacao: 4 metodos simples sobrepostos no ECG (IMG_1407)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "signal_4_metodos_overlay.png"),
                bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    logger.info("Concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
