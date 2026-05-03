"""
Delineação NK2 v2 (brackets cardiologicamente corretos) para TODAS as
derivações do IMG_1303.

Para cada lead (I, II, III, aVR, aVL, aVF, V1–V6, II_rhythm):
  • extrai o sinal canônico do output do Stenhede
  • recorta a região válida (canonical tem ~75% NaN para 11 dos 12 leads
    porque cada lead ocupa só 2.5s do buffer canônico de 10s)
  • roda nk.ecg_clean → ecg_peaks → ecg_delineate(method="dwt")
    com fallback para method="peak" se DWT falhar (DWT precisa ≥3 beats)
  • renderiza no formato v2 (brackets PR/QRS/QT/RR + cores por onda)
  • salva como IMG_1303_<lead>_delineacao_v2.png

Pula leads onde a delineação falha — reporta no console o motivo.

Uso:
    python -m modal_functions.pipeline.test_stenhede_delineate_all
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

LEAD_ORDER = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6", "II_rhythm",
]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _calibrate_normalized(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    _, kps = digitizer.dotter(img)
    if len(kps) < 4:
        _, kps = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(kps, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _trim_valid_region(sig_uv: np.ndarray) -> tuple[np.ndarray, int]:
    """Devolve (sig_trim, start_idx) — o trecho contíguo de valores não-NaN.

    Para canonical leads não-rhythm, sig_uv tem ~75% NaN (apenas o chunk
    2.5s do lead específico tem dados). Esta função pega o intervalo entre
    o primeiro e último não-NaN e devolve sem os zeros do meio (pra NK
    funcionar).
    """
    n = sig_uv.shape[0]
    valid = ~np.isnan(sig_uv)
    if not valid.any():
        return np.zeros(0, dtype=np.float64), 0
    first = int(np.argmax(valid))
    last = n - int(np.argmax(valid[::-1])) - 1
    return sig_uv[first : last + 1].copy(), first


def _delineate(
    sig_uv: np.ndarray, fs: float,
) -> tuple[dict | None, str]:
    """Devolve (resultado, status_str).

    Padding: chunks curtos (< ~5s) são estendidos com zeros antes/depois
    pra que o `ecg_segment` do NeuroKit2 (que precisa de ~0.5s de padding
    em volta de cada R-peak) consiga rodar. Os indices retornados são
    remapeados de volta às coords originais (sem padding).
    """
    if sig_uv is None or sig_uv.size == 0:
        return None, "sinal vazio"
    valid = ~np.isnan(sig_uv) & ~np.isinf(sig_uv)
    n_valid = int(valid.sum())
    if n_valid < int(1.5 * fs):
        return None, f"poucos samples válidos: {n_valid}"

    sig_mv = np.where(valid, sig_uv / 1000.0, 0.0).astype(np.float64)
    n_orig = len(sig_mv)

    # Pad simétrico com zeros até atingir pelo menos 5s totais
    target_total = max(int(5.0 * fs), n_orig + int(2.0 * fs))
    pad_total = max(0, target_total - n_orig)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    sig_padded = np.concatenate(
        [np.zeros(pad_left, dtype=np.float64),
         sig_mv,
         np.zeros(pad_right, dtype=np.float64)]
    )

    methods = ["dwt", "peak"]
    last_err = ""
    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cleaned_pad = nk.ecg_clean(sig_padded, sampling_rate=int(fs))
                _, rd = nk.ecg_peaks(cleaned_pad, sampling_rate=int(fs))
                rpeaks_pad = list(rd.get("ECG_R_Peaks", []))
                # Filtra R-peaks dentro da região válida (sem padding)
                rpeaks_in_orig = [
                    int(r) - pad_left for r in rpeaks_pad
                    if (pad_left <= int(r) < pad_left + n_orig)
                ]
                if len(rpeaks_in_orig) < 1:
                    last_err = f"{method}: 0 R-peaks na região válida"
                    continue
                _, waves_pad = nk.ecg_delineate(
                    cleaned_pad, rpeaks_pad, sampling_rate=int(fs),
                    method=method,
                )
        except Exception as e:
            last_err = f"{method}: {type(e).__name__}: {e}"
            continue

        # Remap indices: subtract pad_left, drop os fora de [0, n_orig)
        def _remap(arr_key: str) -> list:
            out: list = []
            for v in waves_pad.get(arr_key, []):
                if v is None:
                    out.append(None)
                    continue
                try:
                    vi = int(v) - pad_left
                except (TypeError, ValueError):
                    out.append(None)
                    continue
                if 0 <= vi < n_orig:
                    out.append(vi)
                else:
                    out.append(None)
            return out

        waves = {k: _remap(k) for k in waves_pad.keys()}
        cleaned = cleaned_pad[pad_left : pad_left + n_orig]
        return {
            "cleaned": cleaned,
            "rpeaks": rpeaks_in_orig,
            "waves": waves,
            "valid_mask": valid,
            "method": method,
            "pad_info": {
                "pad_left": pad_left, "pad_right": pad_right,
                "n_orig": n_orig, "n_padded": len(sig_padded),
            },
        }, f"OK ({method}, {len(rpeaks_in_orig)} R-peaks, pad={pad_left}+{pad_right})"
    return None, last_err


# ---------------------------------------------------------------------------
# Render v2 — uma derivação
# ---------------------------------------------------------------------------

def _segment_color_per_sample(
    n: int, waves: dict, rpeaks: list,
) -> np.ndarray:
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
            if not (np.isfinite(si) and np.isfinite(ei)) or ei <= si:
                continue
            si = max(0, min(n - 1, si))
            ei = max(0, min(n, ei + 1))
            seg[si:ei] = label

    _fill("ECG_P_Onsets", "ECG_P_Offsets", "P")
    _fill("ECG_R_Onsets", "ECG_R_Offsets", "QRS")
    _fill("ECG_T_Onsets", "ECG_T_Offsets", "T")
    return seg


def _render_lead_v2(
    sig_uv: np.ndarray, fs: float, deli: dict, lead_name: str,
    out_path: Path,
) -> None:
    rpeaks = deli["rpeaks"]
    waves = deli["waves"]
    method = deli.get("method", "?")

    n = len(sig_uv)
    t = np.arange(n) / fs
    sig_mv = sig_uv / 1000.0

    def _ints(key: str) -> list[int]:
        out: list[int] = []
        for v in waves.get(key, []):
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
    q_pks = _ints("ECG_Q_Peaks")
    s_pks = _ints("ECG_S_Peaks")
    qrs_offs = _ints("ECG_R_Offsets")
    t_pks = _ints("ECG_T_Peaks")
    t_offs = _ints("ECG_T_Offsets")
    t_ons = _ints("ECG_T_Onsets")

    # Janela de zoom: 3 batimentos centrais (ou todos se forem poucos)
    if len(rpeaks) >= 3:
        center_r = rpeaks[len(rpeaks) // 2]
        rr_med = int(np.median(np.diff(rpeaks))) if len(rpeaks) >= 2 else int(0.8 * fs)
        x_start = max(0, int(center_r - 1.4 * rr_med))
        x_end = min(n, int(center_r + 1.6 * rr_med))
    elif len(rpeaks) >= 1:
        x_start, x_end = 0, n
    else:
        x_start, x_end = 0, n

    fig, ax = plt.subplots(figsize=(20, 7), facecolor="white")

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
        ax.plot(t, ys, color=color,
                lw=2.0 if label != "baseline" else 1.0, zorder=3)

    def _scatter(idx_list, color, marker, size, label_pt):
        if not idx_list:
            return
        xs = [t[i] for i in idx_list]
        ys = [sig_mv[i] for i in idx_list]
        ax.scatter(xs, ys, color=color, marker=marker, s=size, zorder=5,
                   edgecolors="black", linewidths=0.4, label=label_pt)

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

    # Brackets corrigidos
    sig_window = sig_mv[x_start:x_end] if x_end > x_start else sig_mv
    valid_w = ~np.isnan(sig_window)
    if valid_w.any():
        y_min = float(np.nanmin(sig_window))
        y_max = float(np.nanmax(sig_window))
    else:
        y_min, y_max = -0.5, 0.5
    y_range = max(0.5, y_max - y_min)

    bracket_y_pdur = y_min - 0.13 * y_range
    bracket_y_pr = y_min - 0.20 * y_range
    bracket_y_qrs = y_min - 0.34 * y_range
    bracket_y_qt = y_min - 0.48 * y_range
    bracket_y_rr = y_min - 0.62 * y_range
    bracket_y_st = y_min - 0.07 * y_range

    def _bracket(x1: float, x2: float, y: float, label: str, color: str,
                 below: bool = True) -> None:
        if not (np.isfinite(x1) and np.isfinite(x2)) or x2 <= x1:
            return
        ax.plot([x1, x2], [y, y], color=color, lw=1.5, zorder=4)
        tick_h = y_range * 0.025
        tick_dir = -1 if below else 1
        ax.plot([x1, x1], [y, y + tick_dir * tick_h], color=color, lw=1.5, zorder=4)
        ax.plot([x2, x2], [y, y + tick_dir * tick_h], color=color, lw=1.5, zorder=4)
        ax.text((x1 + x2) / 2, y - tick_h * 1.5, label,
                ha="center", va="top" if below else "bottom",
                fontsize=8, color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    def _nb(vals, target):
        cs = [v for v in vals if v <= target]
        return max(cs) if cs else None

    def _na(vals, target):
        cs = [v for v in vals if v >= target]
        return min(cs) if cs else None

    for i_beat, r in enumerate(rpeaks):
        if r < x_start or r > x_end:
            continue
        p_on = _nb(p_ons, r)
        p_off = _nb(p_offs, r)
        q_pk = _nb(q_pks, r)
        if q_pk is not None and (r - q_pk) > int(0.10 * fs):
            q_pk = None
        q_off = _na(qrs_offs, r)
        t_on = _na(t_ons, r)
        t_off = _na(t_offs, r)

        if p_on is not None and p_off is not None and p_off > p_on:
            pd_ms = (p_off - p_on) / fs * 1000.0
            _bracket(t[p_on], t[p_off], bracket_y_pdur,
                     f"P: {pd_ms:.0f}ms", "#1f77b4")
        if p_on is not None and q_pk is not None and q_pk > p_on:
            pr_ms = (q_pk - p_on) / fs * 1000.0
            _bracket(t[p_on], t[q_pk], bracket_y_pr,
                     f"PR: {pr_ms:.0f}ms", "#1f77b4")
        if q_pk is not None and q_off is not None and q_off > q_pk:
            qrs_ms = (q_off - q_pk) / fs * 1000.0
            _bracket(t[q_pk], t[q_off], bracket_y_qrs,
                     f"QRS: {qrs_ms:.0f}ms", "#2ca02c")
        if q_pk is not None and t_off is not None and t_off > q_pk:
            qt_ms = (t_off - q_pk) / fs * 1000.0
            _bracket(t[q_pk], t[t_off], bracket_y_qt,
                     f"QT: {qt_ms:.0f}ms", "#9467bd")
        if q_off is not None and t_on is not None and t_on > q_off:
            _bracket(t[q_off], t[t_on], bracket_y_st,
                     "ST", "#ff7f0e", below=False)
        if i_beat + 1 < len(rpeaks):
            r_next = rpeaks[i_beat + 1]
            if r_next <= x_end:
                rr_ms = (r_next - r) / fs * 1000.0
                hr = 60000.0 / max(rr_ms, 1)
                _bracket(t[r], t[r_next], bracket_y_rr,
                         f"RR: {rr_ms:.0f}ms (FC: {hr:.0f} bpm)", "red")

    ax.set_xlim(t[x_start], t[max(x_start + 1, x_end - 1)])
    ax.set_ylim(bracket_y_rr - 0.05 * y_range, y_max + 0.05 * y_range)
    ax.set_xlabel("tempo (s)", fontsize=10)
    ax.set_ylabel("mV", fontsize=10)
    ax.grid(True, which="major", color="lightcoral", alpha=0.3)
    ax.set_title(
        f"ProECG — IMG_1303 — {lead_name} — Delineação NeuroKit2 v2 "
        f"(method={method})",
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
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(name)s] %(message)s")
    if not UNDIST_PATH.exists():
        print(f"ERRO: {UNDIST_PATH} nao encontrada", file=sys.stderr)
        return 1

    img = cv2.imread(str(UNDIST_PATH))
    if img is None:
        print("ERRO: falha ao ler imagem", file=sys.stderr); return 2

    print("=" * 78)
    print(" Delineação v2 — TODAS as derivações — IMG_1303")
    print("=" * 78)

    cal = _calibrate_normalized(img)
    print(f"calibracao: px/mm={cal['px_per_mm']:.3f} fs={cal['sampling_rate_hz']:.1f}Hz")

    print("\n[Stenhede] extract_signals_stenhede ...")
    t0 = time.perf_counter()
    full_result = extract_signals_stenhede(
        image_bgr=img, px_per_mm=float(cal["px_per_mm"]),
    )
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s, "
          f"fs={full_result['sampling_rate_hz']:.1f}Hz")

    fs = float(full_result["sampling_rate_hz"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Lead':<10}{'samples':>9}{'NaN%':>7}{'status':>50}")
    print("-" * 76)

    summary: list[tuple[str, str]] = []
    for name in LEAD_ORDER:
        sig = full_result["signals"].get(name)
        if sig is None:
            print(f"{name:<10}{'-':>9}{'-':>7}{'(ausente)':>50}")
            summary.append((name, "ausente"))
            continue
        nan_pct = float(np.isnan(sig).mean() * 100)

        # Trima até o trecho válido contíguo (canonical tem ~75% NaN
        # nos 11 leads não-rhythm)
        sig_trim, _start = _trim_valid_region(sig)

        deli, status = _delineate(sig_trim, fs)
        if deli is None:
            print(f"{name:<10}{len(sig_trim):>9d}{nan_pct:>6.1f}%{status:>50}")
            summary.append((name, status))
            continue

        out_path = OUT_DIR / f"IMG_1303_{name}_delineacao_v2.png"
        try:
            _render_lead_v2(sig_trim, fs, deli, name, out_path)
            print(f"{name:<10}{len(sig_trim):>9d}{nan_pct:>6.1f}%{status:>50}")
            summary.append((name, status))
        except Exception as e:
            err = f"render falhou: {type(e).__name__}: {e}"
            print(f"{name:<10}{len(sig_trim):>9d}{nan_pct:>6.1f}%{err:>50}")
            summary.append((name, err))

    print("\n" + "=" * 78)
    print(f"Resumo: {sum(1 for _, s in summary if s.startswith('OK'))}/{len(summary)} OK")
    print("=" * 78)
    print(f"\nPNGs em: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
