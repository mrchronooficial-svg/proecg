"""
Debug do segmento ST -- ProECG
==============================

Diagnostica a superestimacao de ST em fotos reais.
Para IMG_1279 (foco principal):
  - imprime baseline uV por derivacao + onde foi detectado (TP segs)
  - imprime st_uv, uv_per_pixel, st_mm com fator de erro
  - gera 6 PNGs de debug pra V2: signal, R-peaks, baseline, QRS/J,
    ponto J+60ms, T-peak/T-end

Uso:
    python -m modal_functions.pipeline.debug_st
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.lead_separator import (
    _build_grid_mask,
    _despike_y_series,
    _isolate_trace_morph,
    _remove_grid_pattern,
    separate_and_extract,
)
from .measure import (
    _baseline_uv,
    _detect_p_peaks,
    _detect_qrs_onsets_offsets,
    _detect_r_peaks,
    _detect_t_peaks_offsets,
    _strip_nans,
    LEAD_NAMES,
    UV_PER_MM,
)

NORMALIZED_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Normalizados Leader")
MASKS_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\Leader Masks Teste")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
TARGETS = ["IMG_1275.png", "IMG_1279.png", "IMG_1303.png"]


def _calibrate_normalized(img: np.ndarray) -> tuple[dict, np.ndarray]:
    digitizer = ECGDigitizer(use_mock=False)
    grid_mask, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        grid_mask, keypoints = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    cal = calibrate(grid_matrix=grid_matrix, normalized_image=img)
    return cal, grid_matrix


def _per_lead_detect(sig: np.ndarray, fs: int) -> dict:
    """Roda detecao R/P/QRS/T POR LEAD, sem assumir DII como referencia."""
    sig = _strip_nans(sig)
    r_peaks = _detect_r_peaks(sig, fs)
    p_peaks, _ = _detect_p_peaks(sig, r_peaks, fs)
    qrs_on, qrs_off = _detect_qrs_onsets_offsets(sig, r_peaks, fs)
    t_peaks, t_off = _detect_t_peaks_offsets(sig, r_peaks, qrs_off, fs)
    baseline = _baseline_uv(sig, t_off, p_peaks, r_peaks, fs)
    return {
        "r_peaks": r_peaks, "p_peaks": p_peaks,
        "qrs_on": qrs_on, "qrs_off": qrs_off,
        "t_peaks": t_peaks, "t_off": t_off,
        "baseline_uv": baseline, "sig": sig,
    }


def _tp_segments_used(t_off: np.ndarray, p_peaks: np.ndarray, fs: int) -> list:
    """Retorna list de (start_ms, end_ms) dos segmentos TP usados pra baseline."""
    segs = []
    margin = int(round(0.020 * fs))
    n = min(len(t_off), len(p_peaks) - 1) if len(p_peaks) > 1 else 0
    for i in range(n):
        ti = int(t_off[i])
        pi = int(p_peaks[i + 1])
        if ti < 0 or pi < 0 or pi <= ti:
            continue
        s = ti + margin
        e = pi - margin
        if e - s >= 5:
            segs.append((s, e))
    return segs


def _st_at_j60(
    sig: np.ndarray, qrs_off: np.ndarray, baseline: float, fs: int
) -> tuple[float, list[float]]:
    """ST = sig(J+60ms) - baseline; retorna (uv_median, list per beat)."""
    j_offset = int(round(0.060 * fs))
    vals = []
    for j in qrs_off:
        j = int(j)
        if j < 0:
            continue
        idx = j + j_offset
        if idx >= len(sig):
            continue
        vals.append(float(sig[idx]) - baseline)
    if not vals:
        return float("nan"), []
    return float(np.median(vals)), vals


def _print_lead_diagnostics(
    leads_signals: dict[str, np.ndarray],
    fs: int,
    cal: dict,
    image_name: str,
) -> dict:
    """Imprime baseline + ST por derivacao usando 2 estrategias:
    A) modo atual (R-peaks da DII compartilhados com todas)
    B) modo per-lead (cada derivacao detecta seus R-peaks)
    """
    sig_ii = _strip_nans(leads_signals["II"])

    # ----- Estrategia A: usa DII como referencia (compartilhada) -----
    r_peaks_ii = _detect_r_peaks(sig_ii, fs)
    p_peaks_ii, _ = _detect_p_peaks(sig_ii, r_peaks_ii, fs)
    qrs_on_ii, qrs_off_ii = _detect_qrs_onsets_offsets(sig_ii, r_peaks_ii, fs)
    t_peaks_ii, t_off_ii = _detect_t_peaks_offsets(
        sig_ii, r_peaks_ii, qrs_off_ii, fs
    )

    print(
        f"\n[A] Estrategia atual (DII compartilhada): {len(r_peaks_ii)} R-peaks em DII"
    )
    print(f"    R-peaks DII (idx) = {r_peaks_ii.tolist()}")
    print(
        f"    QRS_off DII (idx) = "
        f"{[int(x) for x in qrs_off_ii.tolist()]}"
    )

    print(
        f"\n[B] Estrategia per-lead: cada derivacao detecta seus proprios R-peaks"
    )

    print(
        f"\n--- Calibracao ---"
        f"\n    px_per_mm = {cal['px_per_mm']:.3f}"
        f"\n    uv_per_pixel = {cal['uv_per_pixel']:.3f} uV/px"
        f"\n    sampling_rate_hz = {cal['sampling_rate_hz']:.1f} Hz  (fs usado: {fs})"
        f"\n    UV_PER_MM = {UV_PER_MM} (= 1mm)"
    )

    rows_out = {}

    print(
        f"\n{'Lead':<6}{'BL_A_uv':>10}{'BL_B_uv':>10}"
        f"{'ST_A_uv':>10}{'ST_B_uv':>10}{'ST_A_mm':>10}{'ST_B_mm':>10}"
        f"{'#R_B':>6}{'TP_segs_B':>11}"
    )
    for name in LEAD_NAMES:
        if name not in leads_signals:
            continue
        sig = _strip_nans(leads_signals[name])

        # Estrategia A: usa DII
        # baseline pode falhar se TP segs nao se alinham com este lead
        bl_a = _baseline_uv(sig, t_off_ii, p_peaks_ii, r_peaks_ii, fs)
        st_a_uv, _ = _st_at_j60(sig, qrs_off_ii, bl_a, fs)

        # Estrategia B: per-lead
        per = _per_lead_detect(leads_signals[name], fs)
        bl_b = per["baseline_uv"]
        st_b_uv, _ = _st_at_j60(sig, per["qrs_off"], bl_b, fs)
        n_r = len(per["r_peaks"])
        tp_segs = _tp_segments_used(per["t_off"], per["p_peaks"], fs)

        st_a_mm = st_a_uv / UV_PER_MM if not np.isnan(st_a_uv) else float("nan")
        st_b_mm = st_b_uv / UV_PER_MM if not np.isnan(st_b_uv) else float("nan")

        print(
            f"{name:<6}{bl_a:>10.1f}{bl_b:>10.1f}"
            f"{st_a_uv:>10.1f}{st_b_uv:>10.1f}"
            f"{st_a_mm:>10.2f}{st_b_mm:>10.2f}"
            f"{n_r:>6d}{len(tp_segs):>11d}"
        )
        rows_out[name] = {
            "bl_a_uv": bl_a, "bl_b_uv": bl_b,
            "st_a_uv": st_a_uv, "st_b_uv": st_b_uv,
            "st_a_mm": st_a_mm, "st_b_mm": st_b_mm,
            "per_lead": per,
        }

    return {
        "rows": rows_out,
        "ii": {
            "r_peaks": r_peaks_ii, "p_peaks": p_peaks_ii,
            "qrs_on": qrs_on_ii, "qrs_off": qrs_off_ii,
            "t_peaks": t_peaks_ii, "t_off": t_off_ii,
        },
    }


def _ms_axis(n: int, fs: int) -> np.ndarray:
    return np.arange(n) / fs * 1000.0


def _save_v2_steps(
    sig_v2: np.ndarray, fs: int, image_name: str, out_dir: Path,
    image_bgr: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    baseline_y_global: float | None = None,
    uv_per_pixel: float | None = None,
) -> None:
    """Gera os 7 PNGs de step para V2 (modo per-lead)."""
    sig = _strip_nans(sig_v2)
    per = _per_lead_detect(sig, fs)
    rp = per["r_peaks"]
    pp = per["p_peaks"]
    qon = per["qrs_on"]
    qoff = per["qrs_off"]
    tp = per["t_peaks"]
    toff = per["t_off"]
    baseline = per["baseline_uv"]
    j60 = int(round(0.060 * fs))

    t_ms = _ms_axis(len(sig), fs)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_title = f"{image_name} V2 (fs={fs}Hz)"

    # ---- Step 0: foto V2 (undistorted) + mask Leader sobreposta ----
    if image_bgr is not None and mask is not None and bbox is not None:
        x1, y1, x2, y2 = bbox
        crop_img = image_bgr[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        overlay = crop_rgb.copy()
        m = crop_mask > 0
        overlay[m] = (0.45 * overlay[m] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.imshow(overlay)
        ax.set_title(
            f"{base_title} - step0: foto V2 + mascara Leader (vermelho)  "
            f"bbox=({x1},{y1})-({x2},{y2})"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{image_name}_V2_step0_mask_overlay.png", dpi=110)
        plt.close(fig)

        # Comparação: máscara vs sinal extraído (mesma escala X em px)
        # O sinal é convertido de volta para coords de pixel da máscara
        # (y_local = baseline_local - sig/uv_per_pixel) e sobreposto.
        h_crop, w_crop = crop_mask.shape
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].imshow(crop_mask, cmap="gray", aspect="auto")
        axes[0].set_title(f"{base_title} - mascara V2 (recortada {w_crop}x{h_crop}px)")
        axes[0].set_ylabel("y (px)")

        # Painel intermediário: sinal sobreposto à máscara em coords de pixel
        axes[1].imshow(crop_mask, cmap="gray", aspect="auto", alpha=0.55)
        if (
            baseline_y_global is not None
            and uv_per_pixel
            and uv_per_pixel > 0
        ):
            baseline_local = baseline_y_global - y1
            sig_y_local = baseline_local - sig / uv_per_pixel
            axes[1].plot(
                np.arange(len(sig)), sig_y_local, color="lime", lw=1.2,
                label="sinal extraido",
            )
            axes[1].axhline(
                baseline_local, color="cyan", lw=0.8, ls="--",
                label=f"baseline_y = {baseline_local:.1f}px",
            )
            axes[1].legend(loc="upper right")
        axes[1].set_title("sobreposicao: mascara + sinal extraido (em coords de pixel)")
        axes[1].set_ylabel("y (px)")
        axes[1].set_ylim(h_crop, 0)  # mantem orientacao da imagem

        # Painel inferior: sinal em uV
        axes[2].plot(np.arange(len(sig)), sig, color="black", lw=0.9)
        axes[2].axhline(0, color="gray", lw=0.5)
        axes[2].set_title("sinal extraido em uV (mesma escala X)")
        axes[2].set_xlabel("x (px / coluna)")
        axes[2].set_ylabel(r"amplitude ($\mu$V)")
        axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{image_name}_V2_extraction_comparison.png", dpi=110)
        plt.close(fig)

    # ---- Step 1: signal raw ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, sig, color="black", lw=0.9)
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel(r"amplitude ($\mu$V)")
    ax.set_title(f"{base_title} - step1: sinal V2 bruto")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_name}_V2_step1_signal.png", dpi=110)
    plt.close(fig)

    # ---- Step 2: R-peaks ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, sig, color="black", lw=0.9)
    if len(rp) > 0:
        ax.scatter(t_ms[rp], sig[rp], color="red", s=60, zorder=5, label="R-peaks")
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel(r"amplitude ($\mu$V)")
    ax.set_title(f"{base_title} - step2: R-peaks detectados ({len(rp)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_name}_V2_step2_rpeaks.png", dpi=110)
    plt.close(fig)

    # ---- Step 3: baseline + TP segs ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, sig, color="black", lw=0.9)
    ax.axhline(baseline, color="green", lw=1.5, label=f"baseline = {baseline:+.1f} uV")
    tp_segs = _tp_segments_used(toff, pp, fs)
    for i, (s, e) in enumerate(tp_segs):
        ax.axvspan(
            t_ms[s], t_ms[e - 1], color="yellow", alpha=0.35,
            label="TP segment" if i == 0 else None,
        )
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel(r"amplitude ($\mu$V)")
    ax.set_title(
        f"{base_title} - step3: baseline ({len(tp_segs)} segs TP)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_name}_V2_step3_baseline.png", dpi=110)
    plt.close(fig)

    # ---- Step 4: QRS on/off (J point) ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, sig, color="black", lw=0.9)
    for i in range(len(qon)):
        on = int(qon[i])
        off = int(qoff[i])
        if on >= 0:
            ax.axvline(
                t_ms[on], color="blue", lw=1.0, alpha=0.7,
                label="QRS onset" if i == 0 else None,
            )
        if off >= 0:
            ax.axvline(
                t_ms[off], color="orange", lw=1.0, alpha=0.7,
                label="J point (QRS off)" if i == 0 else None,
            )
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel(r"amplitude ($\mu$V)")
    ax.set_title(f"{base_title} - step4: QRS onset/offset")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_name}_V2_step4_qrs.png", dpi=110)
    plt.close(fig)

    # ---- Step 5: ST point (J+60ms) ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, sig, color="black", lw=0.9)
    ax.axhline(baseline, color="green", lw=1.5, label=f"baseline = {baseline:+.1f} uV")
    st_uv_per_beat = []
    for i in range(len(qoff)):
        off = int(qoff[i])
        if off < 0:
            continue
        idx = off + j60
        if idx >= len(sig):
            continue
        ax.axvline(
            t_ms[off], color="orange", lw=1.0, alpha=0.5,
            label="J point" if i == 0 else None,
        )
        ax.scatter(
            t_ms[idx], sig[idx], color="red", s=70, zorder=5,
            label="J+60ms" if i == 0 else None,
        )
        # seta da baseline ate o ponto ST
        ax.annotate(
            "", xy=(t_ms[idx], sig[idx]), xytext=(t_ms[idx], baseline),
            arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
        )
        st_uv = float(sig[idx]) - baseline
        st_mm = st_uv / UV_PER_MM
        st_uv_per_beat.append(st_uv)
        ax.text(
            t_ms[idx] + 10, (sig[idx] + baseline) / 2,
            f"{st_mm:+.2f}mm",
            color="purple", fontsize=9, fontweight="bold",
        )
    st_med_uv = float(np.median(st_uv_per_beat)) if st_uv_per_beat else float("nan")
    st_med_mm = st_med_uv / UV_PER_MM if not np.isnan(st_med_uv) else float("nan")
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel(r"amplitude ($\mu$V)")
    ax.set_title(
        f"{base_title} - step5: ST em J+60ms  median = {st_med_mm:+.2f} mm "
        f"({st_med_uv:+.1f} uV)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_name}_V2_step5_st.png", dpi=110)
    plt.close(fig)

    # ---- Step 6: T peak / T end ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, sig, color="black", lw=0.9)
    for i in range(len(tp)):
        tpi = int(tp[i])
        tei = int(toff[i]) if i < len(toff) else -1
        if tpi >= 0:
            ax.scatter(
                t_ms[tpi], sig[tpi], color="purple", s=60, zorder=5,
                label="T peak" if i == 0 else None,
            )
        if tei >= 0 and tei < len(sig):
            ax.scatter(
                t_ms[tei], sig[tei], color="gray", s=60, zorder=5,
                label="T end" if i == 0 else None,
            )
    ax.set_xlabel("tempo (ms)")
    ax.set_ylabel(r"amplitude ($\mu$V)")
    ax.set_title(f"{base_title} - step6: T peak / T end")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{image_name}_V2_step6_twave.png", dpi=110)
    plt.close(fig)

    print(
        f"\n  -> Step PNGs salvos em: {out_dir}"
        f"\n  V2 per-lead: R={len(rp)}  baseline={baseline:+.1f}uV  "
        f"ST_med={st_med_uv:+.1f}uV ({st_med_mm:+.2f}mm)"
    )


def _save_lead_grid_mapped(
    image_bgr: np.ndarray,
    grid_matrix: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_name: str,
    lead_name: str,
    out_dir: Path,
) -> None:
    """Renderiza linhas do grid (major em vermelho grosso, minor em
    vermelho fino) por cima do recorte da imagem original."""
    x1, y1, x2, y2 = bbox
    h_full, w_full = image_bgr.shape[:2]
    # Constrói máscara de grid full-image em duas espessuras
    if grid_matrix is None or grid_matrix.size == 0:
        canvas = image_bgr[y1:y2, x1:x2].copy()
        cv2.imwrite(
            str(out_dir / f"{image_name}_{lead_name}_grid_mapped.png"), canvas,
        )
        return
    major_only = _build_grid_mask(
        grid_matrix, (h_full, w_full),
        n_minor=0, major_thickness=3, minor_thickness=2,
    )
    full_grid = _build_grid_mask(
        grid_matrix, (h_full, w_full),
        n_minor=4, major_thickness=3, minor_thickness=2,
    )
    minor_only = cv2.bitwise_and(full_grid, cv2.bitwise_not(major_only))

    canvas = image_bgr[y1:y2, x1:x2].copy()
    crop_major = major_only[y1:y2, x1:x2] > 0
    crop_minor = minor_only[y1:y2, x1:x2] > 0
    # BGR vermelho puro pra major; vermelho mais claro pra minor
    canvas[crop_minor] = (60, 60, 220)
    canvas[crop_major] = (0, 0, 255)
    cv2.imwrite(
        str(out_dir / f"{image_name}_{lead_name}_grid_mapped.png"), canvas,
    )


def _save_lead_grid_excluded(
    image_bgr: np.ndarray,
    grid_full_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_name: str,
    lead_name: str,
    out_dir: Path,
) -> None:
    """Recorte da derivação em grayscale com pixels do grid apagados
    (set to white). Visualmente sobra só o traço."""
    x1, y1, x2, y2 = bbox
    crop_bgr = image_bgr[y1:y2, x1:x2]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    crop_grid = grid_full_mask[y1:y2, x1:x2] > 0
    cleaned = crop_gray.copy()
    cleaned[crop_grid] = 245  # cinza bem claro (quase branco) pra ficar visível como "papel"
    out = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(
        str(out_dir / f"{image_name}_{lead_name}_grid_excluded.png"), out,
    )


def _save_lead_extraction(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_name: str,
    lead_name: str,
    out_dir: Path,
    px_per_mm: float = 1.0,
    spike_threshold_mm: Optional[float] = None,
    grid_full_mask: Optional[np.ndarray] = None,
) -> None:
    """PNG único: recorte da derivação em grayscale com a linha de
    extração desenhada por cima (linha contínua verde).

    Pipeline:
      - se `grid_full_mask` estiver disponível: por coluna, exclui pixels
        do grid e pega o mais escuro do restante (raw grayscale)
      - senão: top-hat morfológico vertical + argmax por coluna
      - gaps onde só sobra grid → NaN, depois interpolação linear
      - sem tracking, sem suavização, sem despike (a menos que pedido).
    """
    x1, y1, x2, y2 = bbox
    crop_mask = mask[y1:y2, x1:x2]
    crop_bgr = image_bgr[y1:y2, x1:x2]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h_crop, w_crop = crop_mask.shape
    if h_crop == 0 or w_crop == 0:
        return

    crop_grid = (
        grid_full_mask[y1:y2, x1:x2] > 0
        if grid_full_mask is not None
        else None
    )

    if crop_grid is None:
        trace_img = _isolate_trace_morph(crop_gray, px_per_mm)
    else:
        trace_img = None

    points = np.full(w_crop, np.nan, dtype=np.float64)
    for i in range(w_crop):
        col_mask = crop_mask[:, i] > 0
        if not np.any(col_mask):
            continue
        ys = np.arange(h_crop)[col_mask]
        if crop_grid is not None:
            not_grid = ~crop_grid[ys, i]
            ys_trace = ys[not_grid]
            if len(ys_trace) == 0:
                continue
            intens = crop_gray[ys_trace, i].astype(np.int32)
            points[i] = float(ys_trace[int(np.argmin(intens))])
        else:
            intens = trace_img[ys, i]
            points[i] = float(ys[int(np.argmax(intens))])

    if spike_threshold_mm is not None:
        spike_thr_px = max(1.5, spike_threshold_mm * px_per_mm)
        points = _despike_y_series(points, spike_thr_px)

    # Interpolação linear pra gaps curtos onde só havia grid (< 2mm)
    valid = ~np.isnan(points)
    if valid.any():
        max_gap = max(2, int(round(2.0 * px_per_mm)))
        in_gap = False
        gap_start = 0
        for i in range(w_crop):
            if not valid[i]:
                if not in_gap:
                    gap_start = i
                    in_gap = True
            elif in_gap:
                gap_len = i - gap_start
                if gap_len <= max_gap and gap_start > 0:
                    y_left = points[gap_start - 1]
                    y_right = points[i]
                    for xi in range(gap_start, i):
                        t = (xi - gap_start + 1) / (gap_len + 1)
                        points[xi] = y_left + t * (y_right - y_left)
                in_gap = False

    canvas = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    valid = ~np.isnan(points)
    if valid.any():
        xs = np.arange(w_crop)[valid]
        ys = points[valid].round().astype(np.int32)
        pts = np.stack([xs.astype(np.int32), ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(
            canvas, [pts], isClosed=False, color=(0, 255, 0),
            thickness=1, lineType=cv2.LINE_AA,
        )

    out_path = out_dir / f"{image_name}_{lead_name}_new_extraction.png"
    cv2.imwrite(str(out_path), canvas)


def _save_v2_new_extraction(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    baseline_y_global: float,
    uv_per_pixel: float,
    sig: np.ndarray,
    image_name: str,
    out_dir: Path,
    px_per_mm: float = 1.0,
    spike_threshold_mm: Optional[float] = None,
) -> None:
    """Compat: mantém a rota antiga mas delega pro helper genérico."""
    del baseline_y_global, uv_per_pixel, sig
    _save_lead_extraction(
        image_bgr=image_bgr, mask=mask, bbox=bbox,
        image_name=image_name, lead_name="V2",
        out_dir=out_dir, px_per_mm=px_per_mm,
        spike_threshold_mm=spike_threshold_mm,
    )


def _process_one(image_path: Path, mask_path: Path) -> None:
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"  ERRO ao carregar {image_path.name}")
        return
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    print(f"\n{'#' * 80}")
    print(f"# {image_path.name}")
    print(f"{'#' * 80}")

    cal, grid_matrix = _calibrate_normalized(img)
    grid_bin = _build_grid_mask(grid_matrix, img.shape[:2])
    sep = separate_and_extract(mask, img, cal, grid_mask=grid_bin)

    leads_signals = {
        n: i["signal_uv"]
        for n, i in sep["leads"].items() if n != "II_rhythm"
    }
    if "II" not in leads_signals and "II_rhythm" in sep["leads"]:
        leads_signals["II"] = sep["leads"]["II_rhythm"]["signal_uv"]

    fs = int(round(cal["sampling_rate_hz"]))
    diag = _print_lead_diagnostics(leads_signals, fs, cal, image_path.stem)

    # IMG_1275 e IMG_1279: só new_extraction por derivacao
    if image_path.stem in {"IMG_1275", "IMG_1279"}:
        for ld_name, ld_info in sep["leads"].items():
            ld_bbox = ld_info.get("bbox")
            if ld_bbox is None:
                continue
            _save_lead_extraction(
                image_bgr=img, mask=mask, bbox=ld_bbox,
                image_name=image_path.stem, lead_name=ld_name,
                out_dir=OUT_DIR, px_per_mm=float(cal["px_per_mm"]),
                grid_full_mask=grid_bin,
            )
        print(f"  -> {len(sep['leads'])} PNGs de new_extraction salvos")

    # IMG_1303: 3 PNGs (grid_mapped, grid_excluded, new_extraction) por derivacao
    if image_path.stem == "IMG_1303":
        for ld_name, ld_info in sep["leads"].items():
            ld_bbox = ld_info.get("bbox")
            if ld_bbox is None:
                continue
            _save_lead_grid_mapped(
                image_bgr=img, grid_matrix=grid_matrix, bbox=ld_bbox,
                image_name=image_path.stem, lead_name=ld_name, out_dir=OUT_DIR,
            )
            _save_lead_grid_excluded(
                image_bgr=img, grid_full_mask=grid_bin, bbox=ld_bbox,
                image_name=image_path.stem, lead_name=ld_name, out_dir=OUT_DIR,
            )
            _save_lead_extraction(
                image_bgr=img, mask=mask, bbox=ld_bbox,
                image_name=image_path.stem, lead_name=ld_name,
                out_dir=OUT_DIR, px_per_mm=float(cal["px_per_mm"]),
                grid_full_mask=grid_bin,
            )
        print(f"  -> {len(sep['leads'])} x 3 PNGs (grid_mapped/excluded/extraction) salvos")

    # V2 step PNGs + comparacao mascara vs sinal (rodar nas 3 imagens)
    if "V2" in leads_signals:
        v2_info = sep["leads"].get("V2") or {}
        bbox = v2_info.get("bbox")
        baseline_y_global = v2_info.get("baseline_y")
        _save_v2_steps(
            leads_signals["V2"], fs, image_path.stem, OUT_DIR,
            image_bgr=img, mask=mask, bbox=bbox,
            baseline_y_global=baseline_y_global,
            uv_per_pixel=float(cal["uv_per_pixel"]),
        )
        # Nova validacao: mascara como guia + intensidade da imagem original
        if bbox is not None and baseline_y_global is not None:
            _save_v2_new_extraction(
                image_bgr=img, mask=mask, bbox=bbox,
                baseline_y_global=float(baseline_y_global),
                uv_per_pixel=float(cal["uv_per_pixel"]),
                sig=leads_signals["V2"],
                image_name=image_path.stem, out_dir=OUT_DIR,
                px_per_mm=float(cal["px_per_mm"]),
            )


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    if not MASKS_DIR.exists() or not NORMALIZED_DIR.exists():
        print("ERRO: pastas nao encontradas", file=sys.stderr)
        return 1

    for name in TARGETS:
        ip = NORMALIZED_DIR / name
        mp = MASKS_DIR / name
        if not ip.exists() or not mp.exists():
            print(f"AVISO: faltam arquivos pra {name}", file=sys.stderr)
            continue
        _process_one(ip, mp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
