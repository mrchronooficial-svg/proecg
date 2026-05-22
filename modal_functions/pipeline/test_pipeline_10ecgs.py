"""Teste end-to-end do pipeline ProECG nos ECGs de `ECGs teste v1`.

Para cada ECG:
  1. Dotter (grid detection) -> PNG com keypoints sobrepostos
  2. Undistortion (correção de perspectiva) -> PNG da imagem normalizada
  3. Stenhede (extração de sinal) -> PNG sinal sobreposto + % de cobertura
  4. CNN ResNet1d v2 (24 classes) -> ECG digital + diagnóstico categorizado

Salva em ~/Desktop/Projeto ECG/resultados_teste_v1/
Uso (do diretório do projeto):
    python -m modal_functions.pipeline.test_pipeline_10ecgs
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .classify_v2 import (
    CLASSES_24,
    DIAG_TEXT,
    build_cnn_input,
    classify_signal,
    load_cnn_model,
)
from .digitize.constants import GAIN_DEFAULT, PAPER_SPEED_DEFAULT, LEAD_ORDER
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.stenhede_adapter import extract_signals_stenhede

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Cores fixas por derivação (12 distintas + RHY pra II longo)
LEAD_COLORS: dict[str, str] = {
    "I":   "#e6194B",  # vermelho
    "II":  "#3cb44b",  # verde
    "III": "#ffd700",  # dourado
    "aVR": "#4363d8",  # azul
    "aVL": "#f58231",  # laranja
    "aVF": "#911eb4",  # roxo
    "V1":  "#42d4f4",  # ciano
    "V2":  "#f032e6",  # magenta
    "V3":  "#bfef45",  # verde-limão
    "V4":  "#fabed4",  # rosa claro
    "V5":  "#469990",  # teal
    "V6":  "#9A6324",  # marrom
    "RHY": "#000000",  # preto pra rhythm strip
}

# Cell layouts dos formatos suportados (mapeamento row -> [leads por coluna])
CELL_LAYOUTS: dict[str, dict] = {
    "3x4+1": {
        "main": [
            ["I",   "aVR", "V1", "V4"],
            ["II",  "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ],
        "rhythm": "II",
    },
    "6x2+1": {
        "main": [
            ["I",   "V1"],
            ["II",  "V2"],
            ["III", "V3"],
            ["aVR", "V4"],
            ["aVL", "V5"],
            ["aVF", "V6"],
        ],
        "rhythm": "II",
    },
    "12x1": {
        "main": [[n] for n in LEAD_NAMES],
        "rhythm": None,
    },
}


# ---------------------------------------------------------------------------
# 4 visualizadores
# ---------------------------------------------------------------------------

def _save_dotter_viz(
    cropped_bgr: np.ndarray,
    keypoints: list[tuple[int, int]] | np.ndarray,
    out_path: Path,
) -> None:
    img = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    fig_w = max(14.0, w / 220.0)
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img)
    if len(keypoints) > 0:
        kp = np.asarray(keypoints, dtype=np.float64)
        if kp.ndim == 2 and kp.shape[1] >= 2:
            ax.scatter(
                kp[:, 0], kp[:, 1],
                s=8, color="#00DDFF", edgecolors="#003355",
                linewidths=0.4, alpha=0.85, zorder=3,
            )
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"Dotter — Grid Detection ({len(keypoints)} keypoints)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_undistorted_viz(undistorted_bgr: np.ndarray, out_path: Path) -> None:
    img = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    fig_w = max(14.0, w / 220.0)
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Undistorted", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_stenhede_overlay_plain(
    undistorted_bgr: np.ndarray,
    raw_lines_pixel: np.ndarray,
    coverage_pct: float,
    out_path: Path,
    *,
    ecg_format: str = "3x4+1",
) -> None:
    """Sobreposição em cor única (vermelho) — fácil ver onde o pipeline pegou."""
    img = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = 0.55 * img + 0.45
    img_blend = np.clip(img_blend, 0, 1)
    h, w = img.shape[:2]
    fig_w = max(14.0, w / 220.0)
    fig_h = fig_w * (h / w) + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    if raw_lines_pixel.ndim == 2 and raw_lines_pixel.size > 0:
        xs = np.arange(raw_lines_pixel.shape[1])
        for i in range(raw_lines_pixel.shape[0]):
            ax.plot(xs, raw_lines_pixel[i],
                    color="#FF3B30", lw=1.4, alpha=0.9, zorder=3)
    ax.set_title(
        f"Stenhede — sinal sobreposto (cor única)  "
        f"(cobertura: {coverage_pct:.1f}%, formato: {ecg_format.upper()})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_stenhede_overlay(
    undistorted_bgr: np.ndarray,
    raw_lines_pixel: np.ndarray,
    coverage_pct: float,
    out_path: Path,
    *,
    ecg_format: str = "3x4+1",
    chunk_px: int = 0,
    x_offset: int = 0,
) -> None:
    """Pinta cada cell de cada row com a cor da derivação correspondente."""
    img = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = 0.55 * img + 0.45
    img_blend = np.clip(img_blend, 0, 1)
    h, w = img.shape[:2]
    fig_w = max(14.0, w / 220.0)
    fig_h = fig_w * (h / w) + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])

    seen_leads: set[str] = set()
    if raw_lines_pixel.ndim == 2 and raw_lines_pixel.size > 0 and chunk_px > 0:
        layout_def = CELL_LAYOUTS.get(ecg_format, CELL_LAYOUTS["3x4+1"])
        rows_layout = layout_def["main"]
        rhythm_lead = layout_def.get("rhythm")
        n_rows_lines = raw_lines_pixel.shape[0]
        n_main_rows = len(rows_layout)

        # Cells principais
        for row_idx in range(min(n_rows_lines, n_main_rows)):
            row_leads = rows_layout[row_idx]
            line = raw_lines_pixel[row_idx]
            for col_idx, lead_name in enumerate(row_leads):
                color = LEAD_COLORS.get(lead_name, "#999999")
                x_start = x_offset + col_idx * chunk_px
                x_end = x_start + chunk_px
                # Plot só o segmento de coluna correspondente
                xs = np.arange(x_start, min(x_end, len(line)))
                if len(xs) == 0:
                    continue
                ys = line[xs]
                ax.plot(xs, ys, color=color, lw=1.6, alpha=0.95, zorder=3,
                        label=lead_name if lead_name not in seen_leads else None)
                seen_leads.add(lead_name)
                # Label da derivação no início da cell
                valid = ~np.isnan(ys)
                if valid.any():
                    fx = xs[np.argmax(valid)]
                    fy = ys[np.argmax(valid)]
                    ax.text(fx + 8, max(20, fy - 30), lead_name,
                            fontsize=10, fontweight="bold", color=color,
                            bbox=dict(boxstyle="round,pad=0.18",
                                      fc="white", ec=color, alpha=0.9),
                            zorder=5)

        # Rhythm strip (última linha do raw_lines, ocupa cols × chunk_px)
        if rhythm_lead is not None and n_rows_lines > n_main_rows:
            rhythm_row_idx = n_main_rows
            line = raw_lines_pixel[rhythm_row_idx]
            color = LEAD_COLORS["RHY"]
            n_cols = len(rows_layout[0])
            x_start = x_offset
            x_end = x_offset + n_cols * chunk_px
            xs = np.arange(x_start, min(x_end, len(line)))
            ys = line[xs]
            ax.plot(xs, ys, color=color, lw=1.6, alpha=0.95, zorder=3,
                    label=f"{rhythm_lead} (rhythm)")
            valid = ~np.isnan(ys)
            if valid.any():
                fx = xs[np.argmax(valid)]
                fy = ys[np.argmax(valid)]
                ax.text(fx + 8, max(20, fy - 30), f"{rhythm_lead} long",
                        fontsize=10, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.18",
                                  fc=color, ec="black", alpha=0.95),
                        zorder=5)

        ax.legend(loc="upper right", ncol=4, fontsize=8, framealpha=0.92)
    else:
        # Fallback: cor única por row (sem layout info)
        if raw_lines_pixel.ndim == 2 and raw_lines_pixel.size > 0:
            xs = np.arange(raw_lines_pixel.shape[1])
            for i in range(raw_lines_pixel.shape[0]):
                ax.plot(xs, raw_lines_pixel[i],
                        color="#FF3B30", lw=1.4, alpha=0.85, zorder=3)

    ax.set_title(
        f"Stenhede — sinal extraído  "
        f"(cobertura: {coverage_pct:.1f}%, formato: {ecg_format.upper()})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


LAYOUT_3x4 = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
LAYOUT_6x2 = [
    ["I",   "V1"],
    ["II",  "V2"],
    ["III", "V3"],
    ["aVR", "V4"],
    ["aVL", "V5"],
    ["aVF", "V6"],
]
LAYOUT_12x1 = [[name] for name in LEAD_NAMES]


def _detect_format(layout_hint: str | None) -> str:
    """Mapeia o layout retornado pelo Stenhede para um dos 3 formatos."""
    if not layout_hint:
        return "3x4+1"
    h = layout_hint.lower()
    if "3x4" in h or "3_x_4" in h or "12x1" not in h and "6x2" not in h:
        return "3x4+1"
    if "6x2" in h:
        return "6x2+1"
    if "12x1" in h:
        return "12x1"
    return "3x4+1"


def _render_ecg_with_diagnosis(
    signals: dict[str, np.ndarray],
    sampling_rate: float,
    diagnosis: dict,
    title: str,
    out_path: Path,
    ecg_format: str = "3x4+1",
) -> None:
    """Renderiza ECG no formato original (3x4+1 default BR) + diagnóstico."""
    if ecg_format == "3x4+1":
        grid_layout = LAYOUT_3x4
        has_rhythm = True
        fig_w = 18
        fig_ecg_h = 9.5
    elif ecg_format == "6x2+1":
        grid_layout = LAYOUT_6x2
        has_rhythm = True
        fig_w = 14
        fig_ecg_h = 13
    else:
        grid_layout = LAYOUT_12x1
        has_rhythm = False
        fig_w = 18
        fig_ecg_h = 15

    n_cols = len(grid_layout[0])

    # Sinal por cell -> mV
    cell_arrays: dict[str, np.ndarray] = {}
    max_cell_n = 0
    for row in grid_layout:
        for name in row:
            s = signals.get(name)
            if s is None:
                continue
            arr = np.asarray(s, dtype=np.float64)
            if arr.size == 0:
                continue
            arr = np.nan_to_num(arr, nan=0.0)
            if np.abs(arr).max() > 50:
                arr = arr / 1000.0
            cell_arrays[name] = arr
            max_cell_n = max(max_cell_n, len(arr))

    if max_cell_n == 0:
        max_cell_n = 1024

    # Duração de cada cell em segundos (pelo sampling rate atual)
    cell_duration = max_cell_n / float(sampling_rate)
    total_width_s = cell_duration * n_cols

    # Rhythm strip — usar II_rhythm se existir, senão DII concatenado
    rhythm_arr: np.ndarray | None = None
    if has_rhythm:
        rh = signals.get("II_rhythm")
        if rh is None:
            rh = signals.get("II_long")
        if rh is None:
            rh = cell_arrays.get("II")
        if rh is not None:
            rhythm_arr = np.asarray(rh, dtype=np.float64)
            rhythm_arr = np.nan_to_num(rhythm_arr, nan=0.0)
            if np.abs(rhythm_arr).max() > 50:
                rhythm_arr = rhythm_arr / 1000.0

    lead_rows = len(grid_layout) + (1 if has_rhythm and rhythm_arr is not None else 0)
    lead_height_mv = 3.0
    total_height = lead_rows * lead_height_mv

    fig_height = fig_ecg_h + 4.5
    fig = plt.figure(figsize=(fig_w, fig_height), facecolor="white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[fig_ecg_h, 4.5], hspace=0.08)
    ax_ecg = fig.add_subplot(gs[0])
    ax_ecg.set_facecolor("#FFF5F5")

    # Grid milimetrado
    for t in np.arange(0, total_width_s + 0.01, 0.04):
        ax_ecg.axvline(t, color="#FFD0D0", linewidth=0.3, zorder=0)
    for v in np.arange(-total_height, lead_height_mv, 0.1):
        ax_ecg.axhline(v, color="#FFD0D0", linewidth=0.3, zorder=0)
    for t in np.arange(0, total_width_s + 0.01, 0.2):
        ax_ecg.axvline(t, color="#FFA0A0", linewidth=0.6, zorder=0)
    for v in np.arange(-total_height, lead_height_mv, 0.5):
        ax_ecg.axhline(v, color="#FFA0A0", linewidth=0.6, zorder=0)

    # Separadores entre colunas
    for col in range(1, n_cols):
        ax_ecg.axvline(col * cell_duration, color="#CC8888",
                       linewidth=1.2, zorder=1)
    # Separadores entre linhas
    for row in range(1, lead_rows):
        y_sep = -(row * lead_height_mv)
        ax_ecg.axhline(y_sep, color="#CC8888", linewidth=1.0, zorder=1)

    # Plot dos cells
    clip_amp = lead_height_mv / 2 * 0.9
    for row_idx, row_leads in enumerate(grid_layout):
        for col_idx, lead_name in enumerate(row_leads):
            arr = cell_arrays.get(lead_name)
            if arr is None:
                continue
            # cada cell ocupa sua coluna inteira; eixo X relativo à coluna
            x_offset = col_idx * cell_duration
            t = np.arange(len(arr)) / float(sampling_rate) + x_offset
            y_offset = -(row_idx * lead_height_mv) - (lead_height_mv / 2)
            clipped = np.clip(arr, -clip_amp, clip_amp)
            ax_ecg.plot(t, clipped + y_offset, "k-",
                        linewidth=0.8, zorder=3)
            ax_ecg.text(
                x_offset + 0.05, y_offset + clip_amp,
                lead_name, fontsize=9, fontweight="bold",
                color="#333333", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", alpha=0.85,
                          edgecolor="none"),
            )

    # Rhythm strip
    if has_rhythm and rhythm_arr is not None:
        rhythm_row = len(grid_layout)
        y_offset = -(rhythm_row * lead_height_mv) - (lead_height_mv / 2)
        # Estica/limita para ocupar a largura total
        n_max = int(total_width_s * sampling_rate)
        if len(rhythm_arr) > n_max:
            rhythm_arr = rhythm_arr[:n_max]
        t_r = np.arange(len(rhythm_arr)) / float(sampling_rate)
        clipped = np.clip(rhythm_arr, -clip_amp, clip_amp)
        ax_ecg.plot(t_r, clipped + y_offset, "k-",
                    linewidth=0.8, zorder=3)
        ax_ecg.text(
            0.05, y_offset + clip_amp,
            "II (rhythm)", fontsize=9, fontweight="bold",
            color="#333333", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor="white", alpha=0.85,
                      edgecolor="none"),
        )

    ax_ecg.set_xlim(-0.3, total_width_s + 0.1)
    ax_ecg.set_ylim(-total_height - 0.5, lead_height_mv / 2 + 0.5)
    ax_ecg.set_xlabel("Tempo (s)", fontsize=9, color="#666666")
    ax_ecg.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax_ecg.tick_params(axis="y", left=False, labelleft=False)
    ax_ecg.tick_params(axis="x", labelsize=8)
    ax_ecg.text(
        total_width_s + 0.05, lead_height_mv / 2,
        f"25 mm/s   10 mm/mV\nFormato: {ecg_format.upper()}",
        fontsize=7, color="#999999", va="top",
    )

    # ---- Diagnóstico ----
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    bbox = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
        facecolor="#F8F9FA", edgecolor="#DEE2E6", linewidth=1.5,
    )
    ax.add_patch(bbox)
    ax.text(0.5, 0.93, "DIAGNOSTICO CNN — ProECG",
            fontsize=13, fontweight="bold", ha="center", va="top",
            color="#212529")

    y = 0.80

    def line(x: float, txt: str, *, fs=10, color="#333333", bold=False):
        ax.text(x, y, txt, fontsize=fs, va="top", color=color,
                fontweight="bold" if bold else "normal")

    # Isquemia
    line(0.05, "[ISQUEMIA]", fs=11, color="#C0392B", bold=True)
    y -= 0.06
    if diagnosis["isquemia"]:
        for it in diagnosis["isquemia"]:
            text = DIAG_TEXT.get(it["code"], it["code"])
            line(0.08, f"- {text}  ({it['prob']:.0%})")
            y -= 0.06
    else:
        line(0.08, "- Não detectada", color="#95A5A6"); y -= 0.06

    y -= 0.02
    # Arritmia
    line(0.05, "[ARRITMIA]", fs=11, color="#2980B9", bold=True); y -= 0.06
    if diagnosis["arritmia"]:
        for it in diagnosis["arritmia"]:
            text = DIAG_TEXT.get(it["code"], it["code"])
            line(0.08, f"- {text}  ({it['prob']:.0%})")
            y -= 0.06
    else:
        line(0.08, "- Não detectada", color="#95A5A6"); y -= 0.06

    y -= 0.02
    # Outras
    line(0.05, "[OUTRAS ALTERAÇÕES]", fs=11, color="#8E44AD", bold=True)
    y -= 0.06
    if diagnosis["outras"]:
        for it in diagnosis["outras"]:
            text = DIAG_TEXT.get(it["code"], it["code"])
            line(0.08, f"- {text}  ({it['prob']:.0%})")
            y -= 0.06
    else:
        if diagnosis["is_normal"]:
            line(0.08, "- ECG dentro dos limites da normalidade",
                 color="#27AE60")
        else:
            line(0.08, "- Nenhuma detectada", color="#95A5A6")

    ax.text(0.5, 0.04,
            "Ferramenta de apoio à decisão clínica — correlacionar com dados clínicos",
            fontsize=9, ha="center", va="bottom", color="#E74C3C",
            fontstyle="italic")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coverage_from_raw_lines(raw_lines_pixel: np.ndarray) -> float:
    if raw_lines_pixel.size == 0:
        return 0.0
    valid = ~np.isnan(raw_lines_pixel)
    any_per_col = valid.any(axis=0)
    if not any_per_col.any():
        return 0.0
    first = int(np.argmax(any_per_col))
    last = len(any_per_col) - int(np.argmax(any_per_col[::-1])) - 1
    trim = raw_lines_pixel[:, first:last + 1]
    n_total = trim.size
    n_valid = int((~np.isnan(trim)).sum())
    return 100.0 * n_valid / max(n_total, 1)


def _interp_nan(arr: np.ndarray) -> np.ndarray:
    """Interpola NaN linearmente. Se tudo NaN, retorna como veio."""
    a = arr.astype(np.float64).copy()
    nans = np.isnan(a)
    if nans.all() or not nans.any():
        return a
    x = np.arange(len(a))
    a[nans] = np.interp(x[nans], x[~nans], a[~nans])
    return a


def _build_signals_faithful(
    raw_lines_pixel: np.ndarray,
    ecg_format: str,
    x_offset: int,
    chunk_px: int,
    avg_pixel_per_mm: float,
    voltage_gain: float = 10.0,
) -> dict[str, np.ndarray]:
    """Constrói signals dict FIEL ao raw_lines_pixel.

    Diferenças vs _canonicalize_calibrated do stenhede_adapter:
      • Baseline ROW-LEVEL (mediana do row inteiro), NÃO per-cell.
        Cells vizinhas no mesmo row compartilham baseline (igual ao papel).
      • Interpola NaN linearmente dentro de cada cell — sem buraco.
      • Sem mean-subtract isolada que distorce morfologia.

    Output em µV. Mesma chave: dict{lead: array}.
    """
    layout_def = CELL_LAYOUTS.get(ecg_format, CELL_LAYOUTS["3x4+1"])
    rows_layout = layout_def["main"]
    rhythm_lead = layout_def.get("rhythm")

    mv_per_mm = 1.0 / float(voltage_gain)
    uv_per_pixel = (mv_per_mm / float(avg_pixel_per_mm)) * 1000.0

    signals: dict[str, np.ndarray] = {}
    n_rows_lines = raw_lines_pixel.shape[0]
    n_main_rows = len(rows_layout)
    n_cols = len(rows_layout[0]) if rows_layout else 1

    def _to_uv(pixel_y: np.ndarray, baseline_y: float) -> np.ndarray:
        # Y cresce pra baixo na imagem -> negar pra ter R-peak positivo
        return -(pixel_y - baseline_y) * uv_per_pixel

    # Cells principais
    for row_idx in range(min(n_rows_lines, n_main_rows)):
        row_leads = rows_layout[row_idx]
        line = raw_lines_pixel[row_idx]
        valid = ~np.isnan(line)
        if not valid.any():
            for lead_raw in row_leads:
                ln = str(lead_raw).lstrip("-")
                signals[ln] = np.zeros(chunk_px, dtype=np.float64)
            continue

        # Baseline = mediana do row INTEIRO (compartilhada entre cells)
        baseline = float(np.nanmedian(line))

        for col_idx, lead_name_raw in enumerate(row_leads):
            lead_name = str(lead_name_raw).lstrip("-")
            sign = -1 if str(lead_name_raw).startswith("-") else 1
            x_start = x_offset + col_idx * chunk_px
            x_end = x_start + chunk_px

            if x_start >= len(line):
                signals[lead_name] = np.zeros(chunk_px, dtype=np.float64)
                continue
            actual_end = min(x_end, len(line))
            cell_pixel = np.full(chunk_px, np.nan, dtype=np.float64)
            cell_pixel[: actual_end - x_start] = line[x_start:actual_end]
            # Interpola NaN dentro da cell (preserva forma, fecha buracos)
            cell_pixel = _interp_nan(cell_pixel)
            chunk_uv = _to_uv(cell_pixel, baseline) * sign
            signals[lead_name] = chunk_uv

    # Rhythm strip (ocupa n_cols × chunk_px)
    if rhythm_lead is not None and n_rows_lines > n_main_rows:
        line = raw_lines_pixel[n_main_rows]
        valid = ~np.isnan(line)
        if valid.any():
            baseline = float(np.nanmedian(line))
            full_w = n_cols * chunk_px
            x_start = x_offset
            x_end = min(x_start + full_w, len(line))
            rhythm_pixel = np.full(full_w, np.nan, dtype=np.float64)
            rhythm_pixel[: x_end - x_start] = line[x_start:x_end]
            rhythm_pixel = _interp_nan(rhythm_pixel)
            signals["II_rhythm"] = _to_uv(rhythm_pixel, baseline)

    # Garante 12 leads canônicos no dict (NaN se layout não cobriu)
    for canon in LEAD_NAMES:
        if canon not in signals:
            signals[canon] = np.full(chunk_px, np.nan, dtype=np.float64)

    return signals


def _signals_to_array(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Converte dict{lead: array} -> (12, N) na ordem LEAD_ORDER."""
    arrays = []
    target_n = 0
    for name in LEAD_NAMES:
        s = signals.get(name)
        if s is None:
            arrays.append(None)
        else:
            arrays.append(np.asarray(s, dtype=np.float32))
            target_n = max(target_n, len(arrays[-1]))
    if target_n == 0:
        # nada extraído
        return np.zeros((12, 1024), dtype=np.float32)
    out = np.zeros((12, target_n), dtype=np.float32)
    for i, a in enumerate(arrays):
        if a is None or a.size == 0:
            continue
        n = min(len(a), target_n)
        out[i, :n] = np.nan_to_num(a[:n], nan=0.0)
    return out


# ---------------------------------------------------------------------------
# Pipeline por ECG (substitui ECGDigitizer.run para capturar intermediarios)
# ---------------------------------------------------------------------------

def _run_pipeline_with_intermediates(
    image_path: Path, digitizer: ECGDigitizer,
) -> dict:
    """Replica ECGDigitizer.run capturando intermediarios para visualização."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    cropped, _crop_info = digitizer.preprocess(img)

    if digitizer.use_mock:
        _grid_mask, keypoints = digitizer.dotter_mock(cropped)
    else:
        _grid_mask, keypoints = digitizer.dotter(cropped)
        if len(keypoints) == 0:
            _grid_mask, keypoints = digitizer.dotter_mock(cropped)

    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])

    if len(keypoints) >= 100:
        normalized = digitizer.undistort(
            cropped, grid_matrix, grid_info["px_per_mm"],
        )
    else:
        normalized = cropped.copy()

    stenhede = extract_signals_stenhede(
        image_bgr=normalized,
        px_per_mm=float(grid_info["px_per_mm"]),
        paper_speed=float(PAPER_SPEED_DEFAULT),
        voltage_gain=float(GAIN_DEFAULT),
    )
    signals: dict[str, np.ndarray] = stenhede["signals"]
    raw_lines = np.asarray(stenhede["raw_lines_pixel"], dtype=np.float64)
    sampling_rate = float(stenhede["sampling_rate_hz"])

    layout_name = (stenhede.get("match") or {}).get("layout") or "standard_3x4_with_r1"
    canonical = stenhede.get("canonical_lines_uv")
    chunk_px = int(canonical.shape[1]) if canonical is not None and canonical.size else 0
    x_offset = int(stenhede.get("raw_lines_x_offset", 0))
    return {
        "cropped": cropped,
        "keypoints": keypoints,
        "normalized": normalized,
        "raw_lines_pixel": raw_lines,
        "signals": signals,
        "sampling_rate": sampling_rate,
        "px_per_mm": float(grid_info["px_per_mm"]),
        "layout": layout_name,
        "chunk_px": chunk_px,
        "x_offset": x_offset,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_input_dir() -> Path:
    base = Path.home() / "Desktop" / "Projeto ECG" / "ECGs teste v1"
    if base.exists():
        return base
    user = os.getenv("USERNAME") or os.getenv("USER") or "user"
    win = Path(f"C:/Users/{user}/Desktop/Projeto ECG/ECGs teste v1")
    if win.exists():
        return win
    raise FileNotFoundError(
        "Pasta 'ECGs teste v1' não encontrada — esperada em "
        f"{base} ou {win}"
    )


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    in_dir = _resolve_input_dir()
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff",
            "*.JPG", "*.JPEG", "*.PNG")
    images: list[Path] = []
    for ext in exts:
        images.extend(in_dir.glob(ext))
    images = sorted(set(images))
    print(f"[*] {len(images)} ECG(s) em {in_dir}")
    if not images:
        return 1

    out_dir = in_dir.parent / "resultados_teste_v1"
    out_dir.mkdir(exist_ok=True)
    print(f"[*] Saída: {out_dir}\n")

    print("[*] Carregando CNN ResNet1d v2...")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = load_cnn_model(device=device)
    print(f"   pronto em {time.time()-t0:.1f}s (device={device})\n")

    digitizer = ECGDigitizer()

    summary: list[dict] = []
    for idx, img_path in enumerate(images, 1):
        name = img_path.stem
        print(f"{'='*60}\n[{idx}/{len(images)}] {name}\n{'='*60}")
        t_total = time.time()
        try:
            result = _run_pipeline_with_intermediates(img_path, digitizer)
        except Exception as e:
            print(f"   ERRO no pipeline de digitalização: "
                  f"{type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            print()
            continue

        n_kp = len(result["keypoints"])
        print(f"   [1/4] Dotter — {n_kp} keypoints")
        _save_dotter_viz(
            result["cropped"], result["keypoints"],
            out_dir / f"{name}_01_dotter.png",
        )

        print(f"   [2/4] Undistortion — shape={result['normalized'].shape[:2]}")
        _save_undistorted_viz(
            result["normalized"],
            out_dir / f"{name}_02_undistorted.png",
        )

        cov = _coverage_from_raw_lines(result["raw_lines_pixel"])
        ecg_fmt = _detect_format(result.get("layout"))
        print(f"   [3/4] Stenhede — {result['raw_lines_pixel'].shape[0]} linhas, "
              f"cobertura {cov:.1f}%  fmt={ecg_fmt}")
        _save_stenhede_overlay_plain(
            result["normalized"],
            result["raw_lines_pixel"],
            cov,
            out_dir / f"{name}_03a_stenhede_plain.png",
            ecg_format=ecg_fmt,
        )
        _save_stenhede_overlay(
            result["normalized"],
            result["raw_lines_pixel"],
            cov,
            out_dir / f"{name}_03b_stenhede_colored.png",
            ecg_format=ecg_fmt,
            chunk_px=result["chunk_px"],
            x_offset=result["x_offset"],
        )

        # --- Constrói signals FIÉIS ao raw_lines (sem alterar morfologia) ---
        signals_faithful = _build_signals_faithful(
            result["raw_lines_pixel"],
            ecg_fmt,
            result["x_offset"],
            result["chunk_px"],
            result["px_per_mm"],
        )
        # CNN: usa o faithful (II_rhythm 10s real, tile 4× nos outros)
        cnn_array = build_cnn_input(
            signals_faithful, result["sampling_rate"],
        )
        try:
            diagnosis = classify_signal(
                cnn, cnn_array, original_hz=result["sampling_rate"],
            )
        except Exception as e:
            print(f"   ERRO na CNN: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

        n_isq = len(diagnosis["isquemia"])
        n_arr = len(diagnosis["arritmia"])
        n_otr = len(diagnosis["outras"])
        print(f"   [4/4] CNN — isq={n_isq} arr={n_arr} outras={n_otr} "
              f"normal={diagnosis['is_normal']}")
        if n_isq:
            short = [(d["code"], round(d["prob"], 2)) for d in diagnosis["isquemia"]]
            print(f"          isquemia: {short}")
        if n_arr:
            short = [(d["code"], round(d["prob"], 2)) for d in diagnosis["arritmia"]]
            print(f"          arritmia: {short}")
        if n_otr:
            short = [(d["code"], round(d["prob"], 2)) for d in diagnosis["outras"]]
            print(f"          outras:   {short}")

        sr = result["sampling_rate"] if result["sampling_rate"] > 50 else 400.0
        ecg_format = _detect_format(result.get("layout"))
        _render_ecg_with_diagnosis(
            signals_faithful, sr, diagnosis,
            title=f"ProECG — {name}",
            out_path=out_dir / f"{name}_04_diagnostico.png",
            ecg_format=ecg_format,
        )

        elapsed = time.time() - t_total
        print(f"   {elapsed:.1f}s total\n")
        summary.append({
            "name": name, "kp": n_kp, "coverage": cov,
            "isquemia": diagnosis["isquemia"],
            "arritmia": diagnosis["arritmia"],
            "outras": diagnosis["outras"],
            "is_normal": diagnosis["is_normal"],
        })

    print(f"\n[*] Resultados em: {out_dir}")
    print(f"[*] Processados: {len(summary)}/{len(images)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
