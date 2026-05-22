"""
Testa Viterbi extractor na mascara do canal 2 do IMG_1281.

Carrega: Projeto ECG/IMG_1281_stenhede_steps/05_canal_2_signal_PB.png
Roda:    extrair_sinal_viterbi(mask, invert=True)
Plota:   mascara (top) + sinal 1D (bottom)
Salva:   IMG_1281_viterbi_output.png
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

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

from pipeline.digitize.viterbi_extractor import extrair_sinal_viterbi  # noqa: E402

MASK_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1281_stenhede_steps\05_canal_2_signal_PB.png")
OUTPUT_PATH = MASK_PATH.parent / "08_viterbi_output.png"


def main() -> int:
    print(f"Carregando: {MASK_PATH}")
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERRO: nao consegui ler {MASK_PATH}")
        return 1
    print(f"Mascara: shape={mask.shape}, dtype={mask.dtype}, "
          f"min={mask.min()}, max={mask.max()}, mean={mask.mean():.1f}")

    t0 = time.perf_counter()
    sinal = extrair_sinal_viterbi(mask, invert=True)
    dt = time.perf_counter() - t0
    print(f"Viterbi rodou em {dt:.1f}s")

    # Stats
    n_total = sinal.size
    n_nan = int(np.isnan(sinal).sum())
    n_valid = n_total - n_nan
    # Gaps na mascara raw: colunas inteiramente brancas (sem pixel do tracado)
    binary = mask < 128
    cols_with_signal = int(np.any(binary, axis=0).sum())
    n_gaps_raw = mask.shape[1] - cols_with_signal

    print(f"Shape do sinal: {sinal.shape}")
    print(f"Total de pontos: {n_total}")
    print(f"Pontos validos: {n_valid} ({100.0*n_valid/n_total:.1f}%)")
    print(f"NaN apos extracao: {n_nan}")
    print(f"Min: {np.nanmin(sinal):.2f}")
    print(f"Max: {np.nanmax(sinal):.2f}")
    print(f"Gaps na mascara raw (colunas sem pixel): {n_gaps_raw} de {mask.shape[1]}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=100)
    fig.suptitle("Viterbi DP extraction (Fortune 2022 + Rahimi 2025) — IMG_1281",
                 fontsize=14, fontweight="bold")

    ax1.imshow(mask, cmap="gray", aspect="auto")
    ax1.set_title(f"Mascara original (canal 2 Stenhede, B&W) — {mask.shape[1]}x{mask.shape[0]}",
                  fontsize=11)
    ax1.set_xlabel("coluna (x)")
    ax1.set_ylabel("linha (y)")

    ax2.plot(sinal, color="#1f78b4", linewidth=0.6)
    ax2.axhline(0, color="#aaa", linewidth=0.5, linestyle="--")
    ax2.set_title(f"Sinal 1D extraido (Y negado, baseline removida) — "
                  f"{n_valid}/{n_total} pontos validos",
                  fontsize=11)
    ax2.set_xlabel("coluna (x)")
    ax2.set_ylabel("amplitude (px, invertida)")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, mask.shape[1])

    plt.tight_layout()
    plt.savefig(str(OUTPUT_PATH), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"Plot salvo: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
