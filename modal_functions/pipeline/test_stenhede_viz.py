"""
Visualização 12-derivações estilo SBC do output do pipeline Stenhede
====================================================================

  1. Roda digitizer.run em ECGs Reais3/IMG_1279.jpg (foto crua, NÃO undistorted)
  2. Gera duas visualizações dos sinais extraídos:
       a) IMG_1279_stenhede_12leads.png — layout 3×4+rhythm com grid ECG
          (papel rosa, linhas finas a 1mm, grossas a 5mm). Padrão SBC
          (25 mm/s horizontal, 10 mm/mV vertical).
       b) IMG_1279_stenhede_simple.png  — subplots matplotlib puros.
  3. Imprime tempo total + min/max em mV + %NaN por derivação.

Uso:
    python -m modal_functions.pipeline.test_stenhede_viz
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .digitize.ecg_digitizer import ECGDigitizer

IMG_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3\IMG_1279.jpg")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")

# Layout 3×4 — uma cell = uma derivação
LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II_rhythm"


# ---------------------------------------------------------------------------
# Helpers de plotagem
# ---------------------------------------------------------------------------

def _draw_ecg_grid(ax: plt.Axes, t_max_s: float, y_lim_mv: tuple[float, float]) -> None:
    """Desenha papel ECG: linhas finas a cada 1mm (40ms / 0.1mV) e
    grossas a cada 5mm (200ms / 0.5mV). Cor padrão papel rosa."""
    minor_color = (1.0, 0.85, 0.85)   # rosa claro
    major_color = (1.0, 0.55, 0.55)   # rosa mais saturado

    # Verticais (tempo)
    t_minor = np.arange(0.0, t_max_s + 0.04, 0.04)
    t_major = np.arange(0.0, t_max_s + 0.20, 0.20)
    for t in t_minor:
        ax.axvline(t, color=minor_color, lw=0.4, zorder=0)
    for t in t_major:
        ax.axvline(t, color=major_color, lw=0.7, zorder=0)

    # Horizontais (amplitude)
    y_min, y_max = y_lim_mv
    y_minor = np.arange(np.floor(y_min * 10) / 10, y_max + 0.1, 0.1)
    y_major = np.arange(np.floor(y_min * 2) / 2, y_max + 0.5, 0.5)
    for y in y_minor:
        ax.axhline(y, color=minor_color, lw=0.4, zorder=0)
    for y in y_major:
        ax.axhline(y, color=major_color, lw=0.7, zorder=0)

    ax.set_xlim(0.0, t_max_s)
    ax.set_ylim(y_lim_mv)


def _plot_lead_with_gaps(
    ax: plt.Axes,
    sig_uv: np.ndarray,
    fs: float,
    color: str = "black",
    lw: float = 0.8,
) -> None:
    """Plot do sinal em mV vs tempo (s); NaN deixa gap visível."""
    n = sig_uv.shape[0]
    if n == 0:
        return
    t = np.arange(n) / float(fs)
    sig_mv = sig_uv / 1000.0
    # NaN preserva gap (matplotlib quebra a linha automaticamente nos NaN)
    ax.plot(t, sig_mv, color=color, linewidth=lw, zorder=3)


# ---------------------------------------------------------------------------
# Visualização 1: estilo SBC (papel ECG)
# ---------------------------------------------------------------------------

def render_ecg_paper(
    signals: dict[str, np.ndarray], fs: float, out_path: Path,
    title: str,
) -> None:
    # Ajusta limites Y por linha pra englobar todos os sinais daquela linha
    fig = plt.figure(figsize=(20, 12), facecolor="white")
    gs = fig.add_gridspec(
        4, 4, height_ratios=[1, 1, 1, 1.05], hspace=0.35, wspace=0.18,
    )

    # 3 linhas × 4 cols
    for row_idx, row in enumerate(LAYOUT):
        # Define limite Y da linha = max abs entre todos os 4 leads
        max_abs_mv = 0.0
        for name in row:
            sig = signals.get(name)
            if sig is None:
                continue
            valid = ~np.isnan(sig)
            if valid.any():
                m = float(np.nanmax(np.abs(sig[valid])) / 1000.0)
                max_abs_mv = max(max_abs_mv, m)
        if max_abs_mv < 0.5:
            max_abs_mv = 0.5
        # arredonda pra cima em 0.5 mV
        y_half = np.ceil(max_abs_mv * 2 + 0.4) / 2.0
        y_lim = (-y_half, y_half)

        for col_idx, name in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            sig = signals.get(name)
            if sig is None:
                ax.text(0.5, 0.5, f"{name}\n(ausente)", ha="center", va="center",
                        transform=ax.transAxes, color="red")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            n = sig.shape[0]
            t_max = n / float(fs)
            _draw_ecg_grid(ax, t_max, y_lim)
            _plot_lead_with_gaps(ax, sig, fs)
            ax.text(
                0.012, 0.95, name, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="black",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.85),
            )
            ax.tick_params(axis="both", labelsize=7)
            ax.set_xlabel("t (s)" if row_idx == 2 else "", fontsize=8)
            ax.set_ylabel("mV" if col_idx == 0 else "", fontsize=8)

    # Linha 4: rhythm strip (II_rhythm)
    ax_rhythm = fig.add_subplot(gs[3, :])
    rhy = signals.get(RHYTHM_LEAD)
    if rhy is not None and rhy.shape[0] > 0:
        valid = ~np.isnan(rhy)
        if valid.any():
            max_abs_mv = float(np.nanmax(np.abs(rhy[valid])) / 1000.0)
        else:
            max_abs_mv = 0.5
        y_half = max(0.5, np.ceil(max_abs_mv * 2 + 0.4) / 2.0)
        y_lim = (-y_half, y_half)
        n = rhy.shape[0]
        t_max = n / float(fs)
        _draw_ecg_grid(ax_rhythm, t_max, y_lim)
        _plot_lead_with_gaps(ax_rhythm, rhy, fs)
        ax_rhythm.text(
            0.005, 0.92, "II (rhythm)", transform=ax_rhythm.transAxes,
            fontsize=11, fontweight="bold", color="black",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="none", alpha=0.85),
        )
        ax_rhythm.tick_params(axis="both", labelsize=7)
        ax_rhythm.set_xlabel("t (s)", fontsize=8)
        ax_rhythm.set_ylabel("mV", fontsize=8)
    else:
        ax_rhythm.text(0.5, 0.5, "II_rhythm (ausente)",
                       ha="center", va="center", transform=ax_rhythm.transAxes,
                       color="red")
        ax_rhythm.set_xticks([]); ax_rhythm.set_yticks([])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.005,
        "Padrao SBC: 25 mm/s, 10 mm/mV",
        ha="center", fontsize=8, color="dimgray",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualização 2: simples (sem grid)
# ---------------------------------------------------------------------------

def render_simple(
    signals: dict[str, np.ndarray], fs: float, out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 11),
                             sharex="col", sharey="row")
    # As 3 primeiras linhas: 12 leads
    for r in range(3):
        for c in range(4):
            ax = axes[r, c]
            name = LAYOUT[r][c]
            sig = signals.get(name)
            if sig is None or sig.shape[0] == 0:
                ax.set_title(f"{name} (ausente)", fontsize=9, color="red")
                continue
            t = np.arange(sig.shape[0]) / float(fs)
            ax.plot(t, sig / 1000.0, color="black", lw=0.8)
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel("mV", fontsize=8)
            if r == 2:
                ax.set_xlabel("t (s)", fontsize=8)
    # Linha 4: rhythm full-width
    for c in range(4):
        axes[3, c].axis("off")
    ax_r = fig.add_subplot(4, 1, 4)
    rhy = signals.get(RHYTHM_LEAD)
    if rhy is not None and rhy.shape[0] > 0:
        t = np.arange(rhy.shape[0]) / float(fs)
        ax_r.plot(t, rhy / 1000.0, color="black", lw=0.8)
        ax_r.set_title("II (rhythm)", fontsize=10, fontweight="bold")
        ax_r.grid(True, alpha=0.3)
        ax_r.set_xlabel("t (s)", fontsize=8)
        ax_r.set_ylabel("mV", fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    if not IMG_PATH.exists():
        print(f"ERRO: imagem nao encontrada: {IMG_PATH}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f" PIPELINE STENHEDE -- {IMG_PATH}")
    print("=" * 78)

    digitizer = ECGDigitizer(use_mock=False)
    t0 = time.perf_counter()
    try:
        result = digitizer.run(str(IMG_PATH))
    except Exception as e:
        print(f"\n[ERRO] digitizer.run: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2
    elapsed = time.perf_counter() - t0

    signals = result["signals"]
    fs = float(result["sampling_rate"])

    print(f"\n[OK] digitizer.run finalizou em {elapsed:.1f} s")
    print(f"    sampling_rate = {fs:.0f} Hz")
    print(f"    px_per_mm     = {result['px_per_mm']:.3f}")
    print(f"    grid_shape    = {result['grid_shape']}")
    print(f"    segmenter     = {result['quality_flags'].get('segmenter')}")

    # Imprime tabela mV e %NaN
    print(f"\n{'Lead':<10}{'samples':>10}{'min_mV':>12}{'max_mV':>12}{'NaN%':>10}")
    print("-" * 54)
    LEAD_PRINT = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6", "II_rhythm"]
    for name in LEAD_PRINT:
        if name not in signals:
            print(f"{name:<10}{'(ausente)':>10}")
            continue
        s = signals[name]
        valid = ~np.isnan(s)
        nan_pct = 100.0 * (1.0 - valid.sum() / max(s.shape[0], 1))
        if valid.any():
            mn = float(np.nanmin(s)) / 1000.0
            mx = float(np.nanmax(s)) / 1000.0
        else:
            mn, mx = float("nan"), float("nan")
        print(
            f"{name:<10}{s.shape[0]:>10d}{mn:>12.3f}{mx:>12.3f}{nan_pct:>9.2f}%"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    title = "ProECG -- IMG_1279 -- Extracao Stenhede"

    out1 = OUT_DIR / "IMG_1279_stenhede_12leads.png"
    out2 = OUT_DIR / "IMG_1279_stenhede_simple.png"

    print(f"\n[Render] {out1.name} ...")
    try:
        render_ecg_paper(signals, fs, out1, title)
    except Exception as e:
        print(f"[ERRO] render_ecg_paper: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 3

    print(f"[Render] {out2.name} ...")
    try:
        render_simple(signals, fs, out2, title)
    except Exception as e:
        print(f"[ERRO] render_simple: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 4

    print(f"\n[OK] PNGs salvos em {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
