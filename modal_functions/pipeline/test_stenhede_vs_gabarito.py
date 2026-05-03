"""
Teste comparativo: U-Net Stenhede vs Máscara Gabarito
=====================================================

Roda DUAS extrações em paralelo na mesma imagem undistorted IMG_1279:

  Teste A — usa o feature map do U-Net do Stenhede como signal_prob
  Teste B — usa a máscara gabarito (Leader Masks Teste) como signal_prob

Em ambos:
  • mesma calibração (gerada do Dotter+Gridder na imagem undistorted)
  • mesmo SignalExtractor (do vendor)
  • mesmo conversor pixel→µV
  • mesmo plot 3×4+rhythm

Salva:
  - _visualizations/IMG_1279_stenhede_unet.png
  - _visualizations/IMG_1279_gabarito_mask.png

Imprime tabela comparativa de NaN% e range mV por derivação.

Uso:
    python -m modal_functions.pipeline.test_stenhede_vs_gabarito
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.lead_separator import separate_and_extract
from .digitize.stenhede_adapter import extract_signal_probabilities

UNDIST_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted\IMG_1279.png")
MASK_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\Leader Masks Teste\IMG_1279.png")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")

LAYOUT_3x4 = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II_rhythm"


def _calibrate_normalized(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    _, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        _, keypoints = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _plot_lead(ax: plt.Axes, sig_uv: np.ndarray, fs: float, name: str) -> None:
    n = sig_uv.shape[0]
    if n == 0:
        ax.set_title(f"{name} (vazio)", fontsize=9, color="red")
        return
    t = np.arange(n) / float(fs)
    ax.plot(t, sig_uv, color="black", lw=0.8)
    ax.text(
        0.012, 0.93, name, transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        va="top", ha="left",
    )
    ax.grid(True, alpha=0.25)


def _render(signals: dict[str, np.ndarray], fs: float, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(20, 12), facecolor="white")
    gs = fig.add_gridspec(
        4, 4, height_ratios=[1, 1, 1, 1.05], hspace=0.35, wspace=0.18,
    )
    for r, row in enumerate(LAYOUT_3x4):
        for c, name in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            sig = signals.get(name, np.array([], dtype=np.float64))
            _plot_lead(ax, sig, fs, name)
            if c == 0:
                ax.set_ylabel(r"$\mu$V", fontsize=8)
            if r == 2:
                ax.set_xlabel("t (s)", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    ax_r = fig.add_subplot(gs[3, :])
    rhy = signals.get(RHYTHM_LEAD, np.array([], dtype=np.float64))
    _plot_lead(ax_r, rhy, fs, "II (rhythm)")
    ax_r.set_xlabel("t (s)", fontsize=8)
    ax_r.set_ylabel(r"$\mu$V", fontsize=8)
    ax_r.tick_params(axis="both", labelsize=7)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _stats(signals: dict[str, np.ndarray]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, sig in signals.items():
        n = sig.shape[0]
        valid = ~np.isnan(sig)
        nan_pct = 100.0 * (1.0 - valid.sum() / max(n, 1))
        if valid.any():
            mn = float(np.nanmin(sig)) / 1000.0
            mx = float(np.nanmax(sig)) / 1000.0
        else:
            mn, mx = float("nan"), float("nan")
        out[name] = {"nan_pct": nan_pct, "min_mv": mn, "max_mv": mx, "n": n}
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    if not UNDIST_PATH.exists():
        print(f"ERRO: undistorted nao encontrada: {UNDIST_PATH}", file=sys.stderr)
        return 1
    if not MASK_PATH.exists():
        print(f"ERRO: mascara gabarito nao encontrada: {MASK_PATH}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(" COMPARATIVO Stenhede UNet vs Mascara Gabarito -- IMG_1279")
    print("=" * 78)
    print(f"  undistorted: {UNDIST_PATH}")
    print(f"  gabarito   : {MASK_PATH}")

    img = cv2.imread(str(UNDIST_PATH))
    mask_raw = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("ERRO: falha ao ler imagem undistorted", file=sys.stderr)
        return 3
    if mask_raw is None:
        print("ERRO: falha ao ler mascara gabarito", file=sys.stderr)
        return 4
    print(f"  img shape  : {img.shape}")
    print(f"  mask shape : {mask_raw.shape}")
    if mask_raw.shape[:2] != img.shape[:2]:
        mask_raw = cv2.resize(
            mask_raw, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        print(f"  mask resized to {mask_raw.shape}")

    cal = _calibrate_normalized(img)
    fs = float(cal["sampling_rate_hz"])
    print(
        f"\n  calibracao: px/mm={cal['px_per_mm']:.3f} fs={fs:.1f}Hz "
        f"uv/px={cal['uv_per_pixel']:.2f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- TESTE A: U-Net do Stenhede como signal_prob ---
    print("\n[A] Stenhede UNet -> SignalExtractor")
    t0 = time.perf_counter()
    try:
        prob_unet = extract_signal_probabilities(img)
    except Exception as e:
        print(f"[A][ERRO] extract_signal_probabilities: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 5
    print(f"    UNet inference: {(time.perf_counter()-t0)*1000:.0f} ms")
    print(
        f"    prob_unet shape={prob_unet.shape} dtype={prob_unet.dtype} "
        f"range=[{prob_unet.min():.3f}, {prob_unet.max():.3f}]"
    )
    try:
        sep_a = separate_and_extract(
            mask=None, normalized_image=img, calibration=cal,
            signal_prob=prob_unet,
        )
    except Exception as e:
        print(f"[A][ERRO] separate_and_extract: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 6
    signals_a: dict[str, np.ndarray] = {
        n: i["signal_uv"] for n, i in sep_a["leads"].items()
    }

    # --- TESTE B: Mascara Gabarito como signal_prob ---
    print("\n[B] Mascara gabarito -> SignalExtractor (sem UNet)")
    prob_gab = (mask_raw.astype(np.float32) / 255.0).astype(np.float32)
    print(
        f"    prob_gab shape={prob_gab.shape} dtype={prob_gab.dtype} "
        f"range=[{prob_gab.min():.3f}, {prob_gab.max():.3f}] "
        f"px_ativos={int(np.sum(prob_gab > 0.5))}"
    )
    try:
        sep_b = separate_and_extract(
            mask=None, normalized_image=img, calibration=cal,
            signal_prob=prob_gab,
        )
    except Exception as e:
        print(f"[B][ERRO] separate_and_extract: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 7
    signals_b: dict[str, np.ndarray] = {
        n: i["signal_uv"] for n, i in sep_b["leads"].items()
    }

    # --- Renders ---
    out_a = OUT_DIR / "IMG_1279_stenhede_unet.png"
    out_b = OUT_DIR / "IMG_1279_gabarito_mask.png"
    print(f"\n[Render] {out_a.name}")
    _render(signals_a, fs, "ProECG -- IMG_1279 -- Stenhede U-Net", out_a)
    print(f"[Render] {out_b.name}")
    _render(signals_b, fs, "ProECG -- IMG_1279 -- Mascara Gabarito", out_b)

    # --- Tabela comparativa ---
    stats_a = _stats(signals_a)
    stats_b = _stats(signals_b)
    LEAD_PRINT = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6", "II_rhythm"]
    print("\n=== TABELA COMPARATIVA ===")
    print(
        f"{'Lead':<10}"
        f"{'NaN% UNet':>11}{'NaN% Gab':>11}"
        f"{'Range UNet (mV)':>22}{'Range Gab (mV)':>22}"
    )
    print("-" * 78)
    for name in LEAD_PRINT:
        a = stats_a.get(name)
        b = stats_b.get(name)
        if a is None or b is None:
            print(f"{name:<10}  (ausente)")
            continue
        rng_a = f"[{a['min_mv']:+.2f}, {a['max_mv']:+.2f}]"
        rng_b = f"[{b['min_mv']:+.2f}, {b['max_mv']:+.2f}]"
        print(
            f"{name:<10}"
            f"{a['nan_pct']:>10.1f}%{b['nan_pct']:>10.1f}%"
            f"{rng_a:>22}{rng_b:>22}"
        )
    print(f"\nSalvos em: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
