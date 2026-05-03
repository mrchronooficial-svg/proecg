"""
Delineação NK2 em DII (IMG_1303) com brackets de intervalos
CORRIGIDOS para o padrão cardiológico.

Definições (output do `nk.ecg_delineate`):
  PR  = ECG_P_Onsets   → ECG_Q_Peaks
  QRS = ECG_Q_Peaks    → ECG_R_Offsets   (J point)
  QT  = ECG_Q_Peaks    → ECG_T_Offsets
  Pdur= ECG_P_Onsets   → ECG_P_Offsets
  ST  = ECG_R_Offsets  → ECG_T_Onsets    (segmento entre J e início de T)
  RR  = R_Peak[i]      → R_Peak[i+1]

Uso:
    python -m modal_functions.pipeline.test_stenhede_delineate_v2
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.stenhede_adapter import extract_signals_stenhede

UNDIST_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted\IMG_1303.png")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calibrate_normalized(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    _, kps = digitizer.dotter(img)
    if len(kps) < 4:
        _, kps = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(kps, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _segment_color_per_sample(
    n: int, waves: dict, rpeaks: list,
) -> np.ndarray:
    """Devolve array (n,) com a cor de cada sample: P, QRS, T, baseline.

    Convenção (idem v1):
      [P_Onsets..P_Offsets]  = "P"
      [R_Onsets..R_Offsets]  = "QRS"   (mantemos a coloração visual do
                                        complexo inteiro)
      [T_Onsets..T_Offsets]  = "T"
      caso contrário          = "baseline"
    """
    seg = np.full(n, "baseline", dtype=object)

    def _fill(starts_key: str, ends_key: str, label: str) -> None:
        starts = waves.get(starts_key, [])
        ends = waves.get(ends_key, [])
        for s, e in zip(starts, ends):
            if s is None or e is None:
                continue
            try:
                si = int(s); ei = int(e)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(si) and np.isfinite(ei)):
                continue
            if si < 0 or ei < 0 or ei <= si:
                continue
            si = max(0, min(n - 1, si))
            ei = max(0, min(n, ei + 1))
            seg[si:ei] = label

    _fill("ECG_P_Onsets", "ECG_P_Offsets", "P")
    _fill("ECG_R_Onsets", "ECG_R_Offsets", "QRS")
    _fill("ECG_T_Onsets", "ECG_T_Offsets", "T")
    return seg


def _delineate_dii(sig_uv: np.ndarray, fs: float) -> dict | None:
    valid = ~np.isnan(sig_uv)
    if valid.sum() < int(2 * fs):
        return None
    sig_mv = np.where(valid, sig_uv / 1000.0, 0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ecg_clean(sig_mv, sampling_rate=int(fs))
            _, rd = nk.ecg_peaks(cleaned, sampling_rate=int(fs))
            rpeaks = list(rd.get("ECG_R_Peaks", []))
            if len(rpeaks) < 2:
                return None
            _, waves = nk.ecg_delineate(
                cleaned, rpeaks, sampling_rate=int(fs), method="dwt",
            )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "NK delineate falhou: %s: %s", type(e).__name__, e,
        )
        return None
    return {
        "cleaned": cleaned, "rpeaks": rpeaks, "waves": waves,
        "valid_mask": valid,
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _render_v2(
    sig_uv: np.ndarray, fs: float, deli: dict, out_path: Path,
) -> None:
    rpeaks = deli["rpeaks"]
    waves = deli["waves"]

    n = len(sig_uv)
    t = np.arange(n) / fs
    sig_mv = sig_uv / 1000.0

    # ----------- Pega listas das ondas (filtra NaN/None/inválidos) -----------
    def _ints(arr_key: str) -> list[int]:
        vals = waves.get(arr_key, [])
        out: list[int] = []
        for v in vals:
            if v is None:
                continue
            try:
                vi = int(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(vi) and 0 <= vi < n:
                out.append(vi)
        return out

    p_ons = _ints("ECG_P_Onsets")
    p_offs = _ints("ECG_P_Offsets")
    p_pks = _ints("ECG_P_Peaks")
    q_pks = _ints("ECG_Q_Peaks")        # = "ponto I" do cardiologista
    s_pks = _ints("ECG_S_Peaks")
    qrs_offs = _ints("ECG_R_Offsets")    # = ponto J
    t_pks = _ints("ECG_T_Peaks")
    t_offs = _ints("ECG_T_Offsets")
    t_ons = _ints("ECG_T_Onsets")

    # ----------- Janela de zoom: 3 batimentos centrais -----------
    if len(rpeaks) >= 3:
        center_r = rpeaks[len(rpeaks) // 2]
        rr_med = int(np.median(np.diff(rpeaks))) if len(rpeaks) >= 2 else int(0.8 * fs)
        x_start = max(0, int(center_r - 1.4 * rr_med))
        x_end = min(n, int(center_r + 1.6 * rr_med))
    else:
        x_start, x_end = 0, n

    # ----------- Plot -----------
    fig, ax = plt.subplots(figsize=(20, 7), facecolor="white")

    # Cores por segmento (idem v1)
    color_map = {
        "P": "#1f77b4", "QRS": "#2ca02c", "T": "#ff7f0e",
        "baseline": "#7f7f7f",
    }
    seg = _segment_color_per_sample(n, waves, rpeaks)
    for label, color in color_map.items():
        mask = seg == label
        if not np.any(mask):
            continue
        ys = np.where(mask, sig_mv, np.nan)
        ax.plot(
            t, ys, color=color,
            lw=2.0 if label != "baseline" else 1.0,
            zorder=3,
        )

    # ----------- Pontos marcados -----------
    def _scatter(idx_list, color, marker, size, label_pt):
        if not idx_list:
            return
        xs = [t[i] for i in idx_list]
        ys = [sig_mv[i] for i in idx_list]
        ax.scatter(
            xs, ys, color=color, marker=marker, s=size, zorder=5,
            edgecolors="black", linewidths=0.4, label=label_pt,
        )

    _scatter(p_pks, "#1f77b4", "o", 60, "P peak")
    _scatter(q_pks, "#1a5d1a", "o", 60, "Q peak (ponto I)")
    _scatter(rpeaks, "red", "o", 100, "R peak")
    _scatter(s_pks, "#7be07b", "o", 60, "S peak")
    _scatter(t_pks, "#ff7f0e", "o", 60, "T peak")
    _scatter(p_ons, "#1f77b4", "^", 50, "P on")
    _scatter(p_offs, "#1f77b4", "v", 50, "P off")
    _scatter(qrs_offs, "#2ca02c", "v", 50, "QRS off (J)")
    _scatter(t_ons, "#ff7f0e", "^", 50, "T on")
    _scatter(t_offs, "#ff7f0e", "v", 50, "T off")

    # ----------- Brackets CORRIGIDOS -----------
    sig_window = sig_mv[x_start:x_end] if x_end > x_start else sig_mv
    y_min = float(np.nanmin(sig_window))
    y_max = float(np.nanmax(sig_window))
    y_range = max(0.5, y_max - y_min)

    bracket_y_pr  = y_min - 0.20 * y_range
    bracket_y_qrs = y_min - 0.34 * y_range
    bracket_y_qt  = y_min - 0.48 * y_range
    bracket_y_rr  = y_min - 0.62 * y_range
    bracket_y_pdur= y_min - 0.13 * y_range
    bracket_y_st  = y_min - 0.07 * y_range

    def _bracket(x1: float, x2: float, y: float, label: str, color: str,
                 below: bool = True) -> None:
        if not (np.isfinite(x1) and np.isfinite(x2)) or x2 <= x1:
            return
        ax.plot([x1, x2], [y, y], color=color, lw=1.5, zorder=4)
        tick_h = y_range * 0.025
        tick_dir = -1 if below else 1
        ax.plot([x1, x1], [y, y + tick_dir * tick_h], color=color, lw=1.5, zorder=4)
        ax.plot([x2, x2], [y, y + tick_dir * tick_h], color=color, lw=1.5, zorder=4)
        ax.text(
            (x1 + x2) / 2, y - tick_h * 1.5, label,
            ha="center", va="top" if below else "bottom",
            fontsize=8, color=color, zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        )

    def _nearest_before(vals: list[int], target: int) -> int | None:
        cands = [v for v in vals if v <= target]
        return max(cands) if cands else None

    def _nearest_after(vals: list[int], target: int) -> int | None:
        cands = [v for v in vals if v >= target]
        return min(cands) if cands else None

    # Iteração por batimento dentro do zoom
    for i_beat, r in enumerate(rpeaks):
        if r < x_start or r > x_end:
            continue

        # Landmarks do batimento atual (CORRIGIDOS):
        p_on = _nearest_before(p_ons, r)
        p_off = _nearest_before(p_offs, r)
        q_pk = _nearest_before(q_pks, r)        # Q peak = início "real" do QRS
        q_off = _nearest_after(qrs_offs, r)      # J point
        t_on = _nearest_after(t_ons, r)
        t_off = _nearest_after(t_offs, r)

        # Sanity: Q peak deve estar logo antes do R peak (até 80ms)
        if q_pk is not None and (r - q_pk) > int(0.10 * fs):
            q_pk = None

        # P duration: P_onset → P_offset
        if p_on is not None and p_off is not None and p_off > p_on:
            pd_ms = (p_off - p_on) / fs * 1000.0
            _bracket(t[p_on], t[p_off], bracket_y_pdur,
                     f"P: {pd_ms:.0f}ms", "#1f77b4")

        # PR (CORRIGIDO): P_onset → Q_peak
        if p_on is not None and q_pk is not None and q_pk > p_on:
            pr_ms = (q_pk - p_on) / fs * 1000.0
            _bracket(t[p_on], t[q_pk], bracket_y_pr,
                     f"PR: {pr_ms:.0f}ms", "#1f77b4")

        # QRS (CORRIGIDO): Q_peak → R_offset (J)
        if q_pk is not None and q_off is not None and q_off > q_pk:
            qrs_ms = (q_off - q_pk) / fs * 1000.0
            _bracket(t[q_pk], t[q_off], bracket_y_qrs,
                     f"QRS: {qrs_ms:.0f}ms", "#2ca02c")

        # QT (CORRIGIDO): Q_peak → T_offset
        if q_pk is not None and t_off is not None and t_off > q_pk:
            qt_ms = (t_off - q_pk) / fs * 1000.0
            _bracket(t[q_pk], t[t_off], bracket_y_qt,
                     f"QT: {qt_ms:.0f}ms", "#9467bd")

        # ST: J point → T onset (segmento isoelétrico/ascendente)
        if q_off is not None and t_on is not None and t_on > q_off:
            _bracket(t[q_off], t[t_on], bracket_y_st,
                     "ST", "#ff7f0e", below=False)

        # RR: R_peak[i] → R_peak[i+1]
        if i_beat + 1 < len(rpeaks):
            r_next = rpeaks[i_beat + 1]
            if r_next <= x_end:
                rr_ms = (r_next - r) / fs * 1000.0
                hr = 60000.0 / max(rr_ms, 1)
                _bracket(t[r], t[r_next], bracket_y_rr,
                         f"RR: {rr_ms:.0f}ms (FC: {hr:.0f} bpm)", "red")

    # ----------- Eixos / legenda -----------
    ax.set_xlim(t[x_start], t[max(x_start + 1, x_end - 1)])
    ax.set_ylim(bracket_y_rr - 0.05 * y_range, y_max + 0.05 * y_range)
    ax.set_xlabel("tempo (s)", fontsize=10)
    ax.set_ylabel("mV", fontsize=10)
    ax.grid(True, which="major", color="lightcoral", alpha=0.3)
    ax.set_title(
        "ProECG — IMG_1303 — DII — Delineação NeuroKit2 (v2 — brackets corrigidos)",
        fontsize=13, fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(color="#1f77b4", label="Onda P"),
        mpatches.Patch(color="#2ca02c", label="QRS"),
        mpatches.Patch(color="#ff7f0e", label="Onda T"),
        mpatches.Patch(color="#7f7f7f", label="Baseline"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=10, label="R peak"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a5d1a",
                   markersize=8, label="Q peak (ponto I)"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#2ca02c",
                   markersize=8, label="Ponto J (QRS off)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    if not UNDIST_PATH.exists():
        print(f"ERRO: {UNDIST_PATH} nao encontrada", file=sys.stderr)
        return 1

    img = cv2.imread(str(UNDIST_PATH))
    if img is None:
        print("ERRO: falha ao ler imagem", file=sys.stderr)
        return 2

    print("=" * 78)
    print(" Delineação DII v2 — brackets corrigidos — IMG_1303")
    print("=" * 78)

    cal = _calibrate_normalized(img)
    print(f"calibracao: px/mm={cal['px_per_mm']:.3f} fs={cal['sampling_rate_hz']:.1f}Hz")

    print("\n[Stenhede] extract_signals_stenhede ...")
    t0 = time.perf_counter()
    full_result = extract_signals_stenhede(
        image_bgr=img, px_per_mm=float(cal["px_per_mm"]),
    )
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s, "
          f"fs={full_result['sampling_rate_hz']:.1f}Hz, "
          f"layout={full_result['match'].get('layout')}")

    fs = float(full_result["sampling_rate_hz"])
    sig_dii = full_result["signals"].get("II")
    if sig_dii is None or sig_dii.size == 0:
        print("[ERRO] DII nao disponivel"); return 3
    print(f"DII: {len(sig_dii)} samples, NaN={np.isnan(sig_dii).mean()*100:.1f}%, "
          f"range=[{np.nanmin(sig_dii):+.0f}, {np.nanmax(sig_dii):+.0f}] uV")

    deli = _delineate_dii(sig_dii, fs)
    if deli is None:
        print("[ERRO] NK delineate falhou"); return 4

    rpeaks = deli["rpeaks"]
    waves = deli["waves"]
    print(f"\nLandmarks detectados:")
    print(f"  R-peaks    : {len(rpeaks)}")
    for k in [
        "ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets",
        "ECG_Q_Peaks", "ECG_S_Peaks",
        "ECG_R_Onsets", "ECG_R_Offsets",
        "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets",
    ]:
        vals = waves.get(k, [])
        n_valid = sum(1 for v in vals if v is not None and np.isfinite(v))
        print(f"  {k:18s}: {n_valid}/{len(vals)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "IMG_1303_DII_delineacao_v2.png"
    _render_v2(sig_dii, fs, deli, out_path)
    print(f"\n[Render] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
