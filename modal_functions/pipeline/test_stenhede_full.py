"""
Teste do pipeline FULL Stenhede em IMG_1279 undistorted.
========================================================

Roda extract_signals_stenhede (U-Net inteira -> SignalExtractor inteiro
-> LeadIdentifier) e gera:
  • IMG_1279_overlay_stenhede_full.png — raw_lines em pixels sobre a foto
  • Tabela comparativa cell-by-cell vs full pipeline

Uso:
    python -m modal_functions.pipeline.test_stenhede_full
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
from .digitize.stenhede_adapter import (
    extract_signals_stenhede,
    extract_signal_probabilities,
)

_IMG_NAME = sys.argv[1] if len(sys.argv) > 1 else "IMG_1279"
if not _IMG_NAME.lower().endswith(".png"):
    _IMG_NAME = f"{_IMG_NAME}.png"
UNDIST_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted") / _IMG_NAME
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
_IMG_STEM = Path(_IMG_NAME).stem
LEAD_CHANNEL_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]


def _calibrate_normalized(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    _, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        _, keypoints = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _render_overlay(
    image_bgr: np.ndarray,
    raw_lines_pixel: np.ndarray,
    title: str,
    out_path: Path,
    blend_white: float = 0.40,
) -> None:
    """Plota raw_lines (N, W) em pixel-Y por cima da imagem."""
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = (1.0 - blend_white) * img_rgb + blend_white * 1.0
    img_blend = np.clip(img_blend, 0.0, 1.0)

    fig_w = max(16.0, w / 200.0)
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    n_lines = raw_lines_pixel.shape[0] if raw_lines_pixel.ndim == 2 else 0
    if n_lines > 0:
        # Desenha cada linha em vermelho
        x = np.arange(raw_lines_pixel.shape[1])
        for i in range(n_lines):
            y = raw_lines_pixel[i]
            ax.plot(x, y, color="red", linewidth=1.5, alpha=0.85, zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _stats_signals(signals: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for name, sig in signals.items():
        if sig is None or len(sig) == 0:
            out[name] = {"nan_pct": 100.0, "min_mv": float("nan"),
                         "max_mv": float("nan"), "n": 0}
            continue
        valid = ~np.isnan(sig)
        nan_pct = 100.0 * (1.0 - valid.sum() / max(sig.shape[0], 1))
        if valid.any():
            mn = float(np.nanmin(sig)) / 1000.0
            mx = float(np.nanmax(sig)) / 1000.0
        else:
            mn, mx = float("nan"), float("nan")
        out[name] = {"nan_pct": nan_pct, "min_mv": mn, "max_mv": mx,
                     "n": int(sig.shape[0])}
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    if not UNDIST_PATH.exists():
        print(f"ERRO: {UNDIST_PATH} nao encontrada", file=sys.stderr)
        return 1

    img = cv2.imread(str(UNDIST_PATH))
    if img is None:
        print(f"ERRO: falha ao ler {UNDIST_PATH}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(f" TESTE FULL Stenhede pipeline -- {_IMG_STEM} undistorted")
    print("=" * 78)
    print(f"  imagem: {UNDIST_PATH} shape={img.shape}")

    cal = _calibrate_normalized(img)
    fs_native = float(cal["sampling_rate_hz"])
    print(
        f"\n  calibracao: px/mm={cal['px_per_mm']:.3f} fs={fs_native:.1f}Hz "
        f"uv/px={cal['uv_per_pixel']:.2f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============== TESTE FULL ==============
    print("\n[FULL] extract_signals_stenhede ...")
    t0 = time.perf_counter()
    try:
        full_result = extract_signals_stenhede(
            image_bgr=img,
            px_per_mm=float(cal["px_per_mm"]),
            paper_speed=25.0,
            voltage_gain=10.0,
        )
    except Exception as e:
        print(f"\n[FULL][ERRO] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 3
    full_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\n[FULL] OK em {full_ms:.0f} ms")
    print(f"   layout match    : {full_result['match'].get('layout')}")
    print(f"   match cost      : {full_result['match'].get('cost'):.3f}")
    print(f"   n_lines detected: {full_result['n_lines_detected']}")
    print(f"   fs efetivo      : {full_result['sampling_rate_hz']:.1f} Hz")

    # ============== Render overlay ==============
    out_overlay = OUT_DIR / f"{_IMG_STEM}_overlay_stenhede_full.png"
    print(f"\n[Render] {out_overlay.name}")
    _render_overlay(
        img,
        full_result["raw_lines_pixel"],
        f"{_IMG_STEM} -- Overlay FULL Stenhede ({full_result['n_lines_detected']} linhas)",
        out_overlay,
    )

    # ============== TESTE CELL-BY-CELL (referencia) ==============
    print("\n[CELL] separate_and_extract (caminho cell-by-cell) ...")
    t0 = time.perf_counter()
    try:
        prob_cell = extract_signal_probabilities(img)
        sep = separate_and_extract(
            mask=None, normalized_image=img, calibration=cal,
            signal_prob=prob_cell,
        )
    except Exception as e:
        print(f"\n[CELL][ERRO] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 4
    cell_ms = (time.perf_counter() - t0) * 1000.0
    signals_cell = {n: i["signal_uv"] for n, i in sep["leads"].items()}
    print(f"[CELL] OK em {cell_ms:.0f} ms")

    # ============== Tabela comparativa ==============
    s_cell = _stats_signals(signals_cell)
    s_full = _stats_signals(full_result["signals"])
    print("\n=== TABELA COMPARATIVA ===")
    print(
        f"{'Lead':<10}"
        f"{'NaN% cell':>11}{'NaN% full':>11}"
        f"{'Range cell (mV)':>22}{'Range full (mV)':>22}"
    )
    print("-" * 78)
    LEAD_PRINT = LEAD_CHANNEL_ORDER + ["II_rhythm"]
    for name in LEAD_PRINT:
        c = s_cell.get(name)
        f = s_full.get(name)
        if c is None:
            c = {"nan_pct": float("nan"), "min_mv": float("nan"),
                 "max_mv": float("nan"), "n": 0}
        if f is None:
            f = {"nan_pct": float("nan"), "min_mv": float("nan"),
                 "max_mv": float("nan"), "n": 0}
        rc = f"[{c['min_mv']:+.2f}, {c['max_mv']:+.2f}]" if c["n"] else "(ausente)"
        rf = f"[{f['min_mv']:+.2f}, {f['max_mv']:+.2f}]" if f["n"] else "(ausente)"
        print(
            f"{name:<10}"
            f"{c['nan_pct']:>10.1f}%{f['nan_pct']:>10.1f}%"
            f"{rc:>22}{rf:>22}"
        )

    print(f"\n[Tempo] full={full_ms:.0f} ms, cell={cell_ms:.0f} ms")
    print(f"\nSalvos em: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
