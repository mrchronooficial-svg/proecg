"""
test_single_image.py — Testa pipeline digitize_ecg() com uma foto real.

Uso: python scripts/test_single_image.py caminho/para/ecg.jpg
"""

import os
import sys
import time

_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _modal_dir)

import numpy as np
from PIL import Image

from pipeline.digitize import digitize_ecg

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/test_single_image.py <caminho_ecg.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Arquivo nao encontrado: {image_path}")
        sys.exit(1)

    # Carregar imagem
    img = Image.open(image_path)
    print(f"Imagem: {image_path}")
    print(f"  Tamanho: {img.size}, Modo: {img.mode}")
    print()

    # Rodar pipeline
    print("Rodando digitize_ecg()...")
    t0 = time.perf_counter()
    signal = digitize_ecg(img)
    elapsed = time.perf_counter() - t0
    print(f"Concluido em {elapsed:.2f}s\n")

    # Resumo
    print(f"Shape: {signal.shape}")
    print(f"Dtype: {signal.dtype}")
    print(f"NaN: {np.any(np.isnan(signal))}")
    print()

    # Por lead
    print(f"{'Lead':>5s}  {'Min':>8s}  {'Max':>8s}  {'Std':>8s}  Ativo")
    print("-" * 50)
    active_count = 0
    for i, name in enumerate(LEAD_NAMES):
        row = signal[i]
        is_active = np.any(row != 0)
        if is_active:
            active_count += 1
        print(f"{name:>5s}  {row.min():8.4f}  {row.max():8.4f}  {row.std():8.4f}  {'sim' if is_active else '---'}")

    print(f"\nLeads ativos: {active_count}/12")

    # Plot
    output_path = os.path.join(_modal_dir, "scripts", "test_output.png")
    _save_plot(signal, output_path)
    print(f"Plot salvo: {output_path}")


def _save_plot(signal, path):
    """Salva plot dos 12 leads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)
    t = np.arange(signal.shape[1]) / 500.0  # segundos

    for i, (ax, name) in enumerate(zip(axes, LEAD_NAMES)):
        ax.plot(t, signal[i], color="black", linewidth=0.5)
        ax.set_ylabel(name, rotation=0, labelpad=30, fontsize=9)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.tick_params(labelsize=7)
        # Grid leve tipo papel ECG
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Tempo (s)")
    fig.suptitle("ProECG Digitizer — 12 Leads", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
