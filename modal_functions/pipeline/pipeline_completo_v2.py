"""
Pipeline UNIFICADO ProECG v2 — foto -> 12 sinais 1D extraidos (em mV).

Versão v2 (otimizada pra produção):
  - Etapa [9] (OCR de calibração) removida — usa defaults fixos BR
    (25 mm/s, 10 mm/mV). O OCR só confirmava os defaults.
  - Flag `save_plots` controla a geração de PNGs/CSV de debug.
    Em produção (save_plots=False) cai de ~15-20s pra <1s na fase de plots.
  - O OCR de orientação (dentro de digitizer.undistort) é mantido — é o
    que de fato corrige fotos rotacionadas 90/180/270°.

Engloba TODAS as etapas anteriormente espalhadas em:
  - pipeline_completo_IMG_*.py
  - separacao_derivacoes_IMG_*.py
  - overlay_sinal_no_ecg_real_IMG_*.py

Etapas (todas em ordem):
   1. Load HEIC/JPG
   2. Rotaciona se portrait
   3. Preprocess (crop + perspectiva)
   4. Dotter UNet (keypoints)
   5. Gridder (malha regular)
   6. Undistort (homografia por celula)
   7. Stenhede UNet (4 canais)
   8. Salva 4 canais heatmap + B&W mask + painel combinado
   9. PixelSizeFinder
  10. OCR full image -> calibracao (vel, ganho, formato)
  11. Band detection (row projection)
  12. find_trace_bounds_per_band (skip pulso calibracao)
  13. Range comum max(starts)/min(ends)
  14. Build leads structure (com bounds reais)
  15. Crop por lead com margem vertical 40%
  16. Viterbi por lead
  17. Conversao px -> mV
  18. Salva .npy individuais + matriz (12,N) + CSV
  19. Plots: bboxes, 12-lead grid, comparacao mascara/sinal, overlay no ECG real

Uso:
    python tests/pipeline_unificado.py <caminho_imagem> [pasta_output]

Exemplo:
    python tests/pipeline_unificado.py "C:/.../IMG_1283.jpg"
    -> output em "C:/.../IMG_1283_unificado/"
"""

from __future__ import annotations

import logging
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample as scipy_resample

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))
sys.path.insert(0, str(MODAL_ROOT.parent))

from pipeline.digitize.ecg_digitizer import ECGDigitizer  # noqa: E402
from pipeline.digitize.stenhede_adapter import (  # noqa: E402
    _get_lead_identifier,
    _get_pixel_size_finder,
    _process_sparse_prob_torch,
    _signal_extractor_with_offset,
    get_unet,
)
from pipeline.digitize.viterbi_extractor import extrair_sinal_viterbi  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline_unificado")

# ----- Constantes -----
SIGNAL_THRESHOLD = 0.3
VERTICAL_MARGIN_FRAC = 0.40
TARGET_HZ = 400          # taxa de amostragem alvo dos sinais em mV (CNN 1D espera 400Hz)
TARGET_SAMPLES = 4096    # comprimento dos canais no .npy unificado (~10.24s @ 400Hz)
LEAD_LAYOUT_3x4 = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
ALL_LEADS = [lead for row in LEAD_LAYOUT_3x4 for lead in row]
RHYTHM_LABEL = "II_rhythm"

LEAD_ALIASES = {
    "I": "I", "II": "II", "III": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",
}


# =====================================================================
# Helpers (band detection, bounds, OCR)
# =====================================================================

def detect_bands(binary_mask, sigma=10.0, distance=50, prominence_factor=0.1,
                 buffer_factor=1.2):
    h = binary_mask.shape[0]
    row_count = binary_mask.sum(axis=1).astype(float)
    if row_count.sum() == 0:
        return []
    smoothed = gaussian_filter1d(row_count, sigma=sigma)
    peaks, _ = find_peaks(smoothed, distance=distance,
                          prominence=smoothed.max() * prominence_factor)
    if peaks.size == 0:
        return []
    inner_valleys = []
    for i in range(len(peaks) - 1):
        seg = smoothed[peaks[i]:peaks[i + 1]]
        inner_valleys.append(int(peaks[i] + np.argmin(seg)))
    bands = []
    for i, peak in enumerate(peaks):
        if i == 0:
            y0 = max(0, peak - int((inner_valleys[0] - peak) * buffer_factor)) if inner_valleys else 0
        else:
            y0 = inner_valleys[i - 1]
        if i == len(peaks) - 1:
            y1 = min(h, peak + int((peak - inner_valleys[-1]) * buffer_factor)) if inner_valleys else h
        else:
            y1 = inner_valleys[i]
        bands.append((y0, y1))
    return bands


def find_trace_bounds_per_band(binary, y0, y1, px_per_mm=13.0,
                                cal_pulse_max_mm=12.0, sustained_window=50,
                                sustained_min_cols=25, col_min_pixels=3):
    """Acha x_start, x_end do tracado REAL na banda, pulando pulso de cal."""
    band = binary[y0:y1, :]
    col_sums = band.sum(axis=0)
    W = band.shape[1]
    active = col_sums >= col_min_pixels
    kernel = np.ones(sustained_window, dtype=np.int32)
    rolling = np.convolve(active.astype(np.int32), kernel, mode="same")
    sustained = rolling >= sustained_min_cols
    if not sustained.any():
        return 0, W
    sustained_idx = np.where(sustained)[0]
    cal_pulse_max_px = int(cal_pulse_max_mm * px_per_mm)
    candidate_start = int(sustained_idx[0])
    if candidate_start < cal_pulse_max_px:
        for i in range(candidate_start, W):
            if not sustained[i]:
                for j in range(i, W):
                    if sustained[j]:
                        candidate_start = j
                        break
                break
    x_end = min(W, int(sustained_idx[-1]) + sustained_window // 2)
    return candidate_start, x_end


VALID_SPEEDS_MM_S = (12.5, 25.0, 50.0)
VALID_GAINS_MM_MV = (5.0, 10.0, 20.0)


# =====================================================================
# Detecção dinâmica de formato + bboxes precisos
# (mesma lógica validada em test_combined_format.py)
# =====================================================================

LAYOUT_6X2 = [
    ["I", "V1"], ["II", "V2"], ["III", "V3"],
    ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"],
]
LAYOUT_12X1 = [
    ["I"], ["II"], ["III"], ["aVR"], ["aVL"], ["aVF"],
    ["V1"], ["V2"], ["V3"], ["V4"], ["V5"], ["V6"],
]
QRS_MARGIN_FRAC = 0.05  # 5% folga vertical pros picos do QRS


def detector_bandas(signal_binary, bands):
    """Propõe formato pela contagem de bandas:
      <=5 -> 3x4   |   <=8 -> 6x2   |   >8 -> 12x1
    Rhythm: bandas além das main (3, 6 ou 0).
    """
    n = len(bands)
    if n == 0:
        return {"format": "unknown", "base": "unknown", "n_bands": 0,
                "n_main": 0, "n_cols": 0, "n_rhythm": 0}
    if n <= 5:
        base, n_main, n_cols = "3x4", 3, 4
        n_rhythm = max(0, n - 3)
    elif n <= 8:
        base, n_main, n_cols = "6x2", 6, 2
        n_rhythm = max(0, n - 6)
    else:
        base, n_main, n_cols = "12x1", 12, 1
        n_rhythm = 0
    return {
        "format": base + (f"+{n_rhythm}" if n_rhythm > 0 else ""),
        "base": base, "n_bands": n,
        "n_main": n_main, "n_cols": n_cols, "n_rhythm": n_rhythm,
    }


def detector_lead_identifier_full(signal_prob, grid_prob, text_prob):
    """PixelSizeFinder + SignalExtractor + LeadIdentifier do Stenhede."""
    pxsize = _get_pixel_size_finder()
    with torch.no_grad():
        grid_t = torch.from_numpy(np.ascontiguousarray(grid_prob)).float()
        mm_px_x, mm_px_y = pxsize(grid_t)
    avg_pixel_per_mm = float((1.0 / float(mm_px_x) + 1.0 / float(mm_px_y)) / 2.0)
    signal_prob_t = torch.from_numpy(np.ascontiguousarray(signal_prob)).float()
    raw_lines_t, x_offset = _signal_extractor_with_offset(signal_prob_t)
    identifier = _get_lead_identifier(target_num_samples=5000)
    if raw_lines_t.shape[0] > 1:
        raw_lines_t = identifier._merge_nonoverlapping_lines(raw_lines_t.clone())
    text_prob_t = (
        torch.from_numpy(np.ascontiguousarray(text_prob))
        .unsqueeze(0).unsqueeze(0).float()
    )
    result = identifier(
        raw_lines_t.clone(), text_prob_t,
        avg_pixel_per_mm=avg_pixel_per_mm,
        threshold=0.8, mv_per_mm=0.1,
    )
    layout_name = result.get("layout") or ""
    cost = float(result.get("cost", float("nan")))
    layouts = identifier.layouts
    if layout_name in layouts:
        desc = layouts[layout_name]
        cols = int(desc["layout"]["cols"])
        rows = int(desc["layout"]["rows"])
        n_rhythm = len(desc.get("rhythm_leads", []))
        base = f"{rows}x{cols}"
        fmt = base + (f"+{n_rhythm}" if n_rhythm else "")
    else:
        cols, rows, n_rhythm = 0, 0, 0
        fmt = "unknown"
        base = "unknown"
    return {
        "format": fmt, "base": base, "layout_name": layout_name, "cost": cost,
        "n_main": rows, "n_cols": cols, "n_rhythm": n_rhythm,
        "avg_pixel_per_mm": avg_pixel_per_mm,
    }


def combinar_formato(bandas, li):
    """Prioridade: bandas > LeadIdentifier.
    Confiança ALTA se bandas e LI concordam, MÉDIA se discordam (bandas decide).
    """
    if bandas["base"] != "unknown":
        final_base = bandas["base"]
        n_rhythm = bandas["n_rhythm"]
    elif li["base"] != "unknown":
        final_base = li["base"]
        n_rhythm = li["n_rhythm"]
    else:
        return "unknown", "BAIXA"
    final_fmt = final_base + (f"+{n_rhythm}" if n_rhythm > 0 else "")
    if bandas["base"] != "unknown" and li["base"] != "unknown":
        if bandas["base"] == li["base"]:
            conf = "ALTA (bandas+LI concordam)"
        else:
            conf = (
                f"MÉDIA (bandas={bandas['base']}, LI={li['base']} "
                "— bandas decide)"
            )
    else:
        conf = "BAIXA (só 1 detector funcionou)"
    return final_fmt, conf


def build_precise_bboxes(bands, final_base, signal_binary, px_per_mm):
    """Para cada banda + col, calcula bbox preciso:
      X: trim do traçado real dentro do range da coluna (signal_binary)
      Y: topo/base do traçado + 5% margem, clampado ao midpoint entre bandas
    """
    H, W = signal_binary.shape
    if not bands:
        return [], 0, 0, 0

    if final_base == "3x4":
        layout = LEAD_LAYOUT_3x4
        n_main, n_cols = 3, 4
    elif final_base == "6x2":
        layout = LAYOUT_6X2
        n_main, n_cols = 6, 2
    elif final_base == "12x1":
        layout = LAYOUT_12X1
        n_main, n_cols = 12, 1
    else:
        return [], 0, 0, 0

    # Bounds X comuns às bandas main
    band_bounds = []
    for (y0, y1) in bands[:n_main]:
        xs, xe = find_trace_bounds_per_band(signal_binary, y0, y1, px_per_mm)
        band_bounds.append((xs, xe))
    if band_bounds:
        x_start_common = max(b[0] for b in band_bounds)
        x_end_common = min(b[1] for b in band_bounds)
    else:
        x_start_common, x_end_common = 0, W
    chunk_w = max(1, (x_end_common - x_start_common) // n_cols)

    n_bands = len(bands)
    band_centers = [(y0 + y1) // 2 for (y0, y1) in bands]
    cap_top = [0] * n_bands
    cap_bot = [H] * n_bands
    for i in range(n_bands):
        if i > 0:
            cap_top[i] = (band_centers[i - 1] + band_centers[i]) // 2
        if i < n_bands - 1:
            cap_bot[i] = (band_centers[i] + band_centers[i + 1]) // 2

    def _trim_xy(x_s, x_e, row_idx):
        y_search_top = cap_top[row_idx]
        y_search_bot = cap_bot[row_idx]
        x_s = max(0, min(W, x_s))
        x_e = max(0, min(W, x_e))
        if x_e <= x_s or y_search_bot <= y_search_top:
            return x_s, x_e, y_search_top, y_search_bot
        region = signal_binary[y_search_top:y_search_bot, x_s:x_e]
        cols_w = np.any(region, axis=0)
        rows_w = np.any(region, axis=1)
        if not rows_w.any() or not cols_w.any():
            return x_s, x_e, y_search_top, y_search_bot
        col_idx = np.where(cols_w)[0]
        x_trim_s = x_s + int(col_idx[0])
        x_trim_e = x_s + int(col_idx[-1]) + 1
        row_idx_sig = np.where(rows_w)[0]
        y_sig_top = y_search_top + int(row_idx_sig[0])
        y_sig_bot = y_search_top + int(row_idx_sig[-1]) + 1
        band_h = bands[row_idx][1] - bands[row_idx][0]
        margin = max(1, int(round(band_h * QRS_MARGIN_FRAC)))
        y_top = max(y_search_top, y_sig_top - margin)
        y_bot = min(y_search_bot, y_sig_bot + margin)
        return x_trim_s, x_trim_e, int(y_top), int(y_bot)

    leads = []
    for row_idx in range(min(n_main, n_bands)):
        row_leads = layout[row_idx]
        y0_b, y1_b = bands[row_idx]
        for col_idx, lead_name in enumerate(row_leads):
            x_cell_s = x_start_common + col_idx * chunk_w
            x_cell_e = (
                x_start_common + (col_idx + 1) * chunk_w
                if col_idx < n_cols - 1 else x_end_common
            )
            x_s, x_e, y_top, y_bot = _trim_xy(x_cell_s, x_cell_e, row_idx)
            leads.append({
                "name": lead_name, "row": row_idx,
                "x_start": int(x_s), "x_end": int(x_e),
                "y0": int(y_top), "y1": int(y_bot),
                "y_center": int((y_top + y_bot) // 2),
                "y_top": int(y_top), "y_bot": int(y_bot),
                "is_rhythm": False, "anchor": "split_real",
            })
    # Apenas 1º rhythm strip -> II_rhythm
    if n_bands > n_main:
        k = n_main
        y0_r, y1_r = bands[k]
        xs_r, xe_r = find_trace_bounds_per_band(
            signal_binary, y0_r, y1_r, px_per_mm,
        )
        x_s, x_e, y_top, y_bot = _trim_xy(xs_r, xe_r, k)
        leads.append({
            "name": RHYTHM_LABEL, "row": k,
            "x_start": int(x_s), "x_end": int(x_e),
            "y0": int(y_top), "y1": int(y_bot),
            "y_center": int((y_top + y_bot) // 2),
            "y_top": int(y_top), "y_bot": int(y_bot),
            "is_rhythm": True, "anchor": "rhythm",
        })
    return leads, x_start_common, x_end_common, chunk_w


def _snap_to_clinical(value, valid_set, tol=5.0):
    """Snap valor lido pelo OCR pro valor clinico mais proximo se dentro de
    tolerancia. Se nao, retorna None (caller usa default)."""
    if value is None:
        return None
    closest = min(valid_set, key=lambda v: abs(v - value))
    return closest if abs(closest - value) <= tol else None


def parse_calibration(ocr_results):
    cal = {"speed_mm_s": 25.0, "gain_limb_mm_mV": 10.0,
           "gain_chest_mm_mV": None, "format": "3x4+1"}
    items = []
    for bbox, text, _ in ocr_results:
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        items.append((int(np.mean(xs)), int(np.mean(ys)), text.strip()))

    def find_num_neighbor(keyword):
        for i, (cx, cy, txt) in enumerate(items):
            if keyword.lower() not in txt.lower():
                continue
            best = None; best_dx = float("inf")
            for j, (cx2, cy2, txt2) in enumerate(items):
                if j == i or abs(cy2 - cy) > 25 or cx2 <= cx:
                    continue
                # Tenta extrair PRIMEIRO grupo de digitos (1-3 chars)
                # com possivel ponto decimal, pra evitar "258" quando era "25 8"
                m = re.search(r"\d{1,3}(?:[.,]\d)?", txt2)
                if not m:
                    continue
                dx = cx2 - cx
                if dx > 500:
                    continue
                if dx < best_dx:
                    best_dx = dx
                    best = float(m.group(0).replace(",", "."))
            return best
        return None

    raw_speed = find_num_neighbor("Veloc")
    raw_limb = find_num_neighbor("Membr")
    raw_chest = find_num_neighbor("Torax") or find_num_neighbor("Tórax")

    snapped = _snap_to_clinical(raw_speed, VALID_SPEEDS_MM_S, tol=5.0)
    if snapped is not None:
        cal["speed_mm_s"] = snapped
    snapped = _snap_to_clinical(raw_limb, VALID_GAINS_MM_MV, tol=3.0)
    if snapped is not None:
        cal["gain_limb_mm_mV"] = snapped
    snapped = _snap_to_clinical(raw_chest, VALID_GAINS_MM_MV, tol=3.0)
    if snapped is not None:
        cal["gain_chest_mm_mV"] = snapped
    if cal["gain_chest_mm_mV"] is None:
        cal["gain_chest_mm_mV"] = cal["gain_limb_mm_mV"]
    text_combined = " ".join(t for _, _, t in items)
    if re.search(r"12\s+deriv", text_combined, re.IGNORECASE):
        cal["format"] = "3x4+1"
    cal["_raw_ocr_values"] = {"speed": raw_speed, "limb": raw_limb, "chest": raw_chest}
    return cal


def find_signal_baseline_y(mask, lead):
    y0, y1 = lead["y0"], lead["y1"]
    x0, x1 = lead["x_start"], lead["x_end"]
    crop_binary = mask[y0:y1, x0:x1] < 128
    if not crop_binary.any():
        return (y0 + y1) // 2
    ys = np.where(crop_binary)[0]
    return int(np.median(ys)) + y0


# =====================================================================
# Plot helpers
# =====================================================================

def save_channel_heatmap(prob, cmap, title, out_path):
    fig = plt.figure(figsize=(14, 9), dpi=110)
    plt.imshow(prob, cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.colorbar(fraction=0.04, pad=0.02)
    plt.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def plot_12_leads_grid(signals_px, leads, cal, px_per_mm, out_path):
    fig = plt.figure(figsize=(16, 11), dpi=110)
    fig.suptitle(
        f"ECG digitalizado | vel={cal['speed_mm_s']:.0f} mm/s, "
        f"gain={cal['gain_limb_mm_mV']:.0f} mm/mV | px/mm={px_per_mm:.2f}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    for row_idx, row_leads in enumerate(LEAD_LAYOUT_3x4):
        for col_idx, lead_name in enumerate(row_leads):
            ax = plt.subplot2grid((4, 4), (row_idx, col_idx))
            sig_px = signals_px.get(lead_name)
            if sig_px is None or len(sig_px) == 0:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(lead_name, fontsize=10)
                continue
            gain = (cal["gain_chest_mm_mV"] if lead_name.startswith("V")
                    else cal["gain_limb_mm_mV"])
            mv = sig_px / (gain * px_per_mm)
            t = np.arange(len(sig_px)) / (cal["speed_mm_s"] * px_per_mm)
            ax.plot(t, mv, color="#1f78b4", linewidth=0.7)
            ax.axhline(0, color="#aaa", linewidth=0.3, linestyle="--")
            ax.set_title(lead_name, fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3, linewidth=0.4)
            ax.set_xlabel("t (s)", fontsize=8)
            ax.set_ylabel("mV", fontsize=8)
            ax.tick_params(labelsize=7)
    ax = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    sig = signals_px.get(RHYTHM_LABEL)
    if sig is not None:
        mv = sig / (cal["gain_limb_mm_mV"] * px_per_mm)
        t = np.arange(len(sig)) / (cal["speed_mm_s"] * px_per_mm)
        ax.plot(t, mv, color="#1f78b4", linewidth=0.6)
        ax.axhline(0, color="#aaa", linewidth=0.3, linestyle="--")
        ax.set_title("II (rhythm strip)", fontsize=10, fontweight="bold")
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("mV", fontsize=8)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def plot_bboxes(mask, leads, out_path):
    img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colors = [(255, 80, 80), (80, 200, 80), (80, 80, 255), (200, 200, 0),
              (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
              (0, 128, 128), (200, 100, 100), (100, 200, 100), (100, 100, 200),
              (50, 50, 50)]
    for i, lead in enumerate(leads):
        c = colors[i % len(colors)]
        # y_top/y_bot já vêm trimados precisos pelo signal_binary +5% margem
        y_top = lead.get("y_top", lead["y0"])
        y_bot = lead.get("y_bot", lead["y1"])
        cv2.rectangle(img, (lead["x_start"], y_top),
                      (lead["x_end"], y_bot), c, 2)
        cv2.putText(img, f"{lead['name']} [{lead['anchor']}]",
                    (lead["x_start"] + 6, y_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def plot_overlay_on_real(norm_bgr, mask, leads, signals_px, out_path):
    fig, ax = plt.subplots(figsize=(20, 7), dpi=120)
    ax.imshow(cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB), aspect="equal")
    for L in leads:
        sig = signals_px.get(L["name"])
        if sig is None:
            continue
        y_base = find_signal_baseline_y(mask, L)
        x = np.arange(L["x_start"], L["x_start"] + len(sig))
        y = y_base - sig
        color = "#22cc44" if L["name"] == RHYTHM_LABEL else "#ff2222"
        lw = 0.5 if L["name"] == RHYTHM_LABEL else 0.7
        ax.plot(x, y, color=color, linewidth=lw, alpha=0.85)
        ax.text(L["x_start"] + 8, y_base - 50, L["name"],
                color="red", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
    ax.set_title("Sinais Viterbi (vermelho/verde) sobre ECG real undistorted",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)


# =====================================================================
# Main pipeline
# =====================================================================

def run_pipeline(
    img_path: Path,
    out_dir: Path,
    save_plots: bool = True,
) -> int:
    """Roda o pipeline end-to-end.

    Args:
        img_path: caminho da foto (JPG/PNG/HEIC).
        out_dir:  diretório onde salvar os outputs.
        save_plots: se True (default), gera todos os PNGs intermediários
            + CSV + summary.txt. Em produção, passar False corta ~10-15s
            de matplotlib/I/O sem afetar o ecg_*_leads_mv.npy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("PIPELINE UNIFICADO — %s", img_path.name)
    logger.info("=" * 70)

    # ----- 1. Load -----
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        # Tenta HEIC
        try:
            import pillow_heif  # noqa
            from PIL import Image
            pillow_heif.register_heif_opener()
            pil = Image.open(str(img_path)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error("Falha ao ler imagem: %s", e)
            return 1
    h, w = img_bgr.shape[:2]
    if h > w * 1.2:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    logger.info("[1] Original %dx%d", img_bgr.shape[1], img_bgr.shape[0])

    # ----- 2-5. Preprocess + Dotter + Gridder + Undistort -----
    logger.info("[2-5] Preprocess + Dotter + Gridder + Undistort...")
    digitizer = ECGDigitizer(use_mock=False)
    cropped, _ = digitizer.preprocess(img_bgr)
    _, keypoints = digitizer.dotter(cropped)
    if len(keypoints) == 0:
        _, keypoints = digitizer.dotter_mock(cropped)
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])
    if len(keypoints) >= 100:
        normalized = digitizer.undistort(cropped, grid_matrix, grid_info["px_per_mm"])
    else:
        normalized = cropped.copy()
    h_n, w_n = normalized.shape[:2]
    if save_plots:
        cv2.imwrite(str(out_dir / "00_normalized.png"), normalized)
    logger.info("    Normalized: %dx%d", w_n, h_n)

    # ----- 6-7. Stenhede UNet + PixelSizeFinder -----
    logger.info("[6-7] Stenhede UNet (4 canais) + PixelSizeFinder...")
    image_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    max_side = max(h_n, w_n)
    if max_side > 3000:
        scale = 3000 / float(max_side)
        new_h, new_w = int(round(h_n * scale)), int(round(w_n * scale))
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_float = image_rgb.astype(np.float32)
    img_float = (img_float - img_float.min()) / max(img_float.max() - img_float.min(), 1e-8)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        grid_p = _process_sparse_prob_torch(probs[:, 0])
        text_p = _process_sparse_prob_torch(probs[:, 1])
        signal_p = _process_sparse_prob_torch(probs[:, 2])
        bg_p = _process_sparse_prob_torch(probs[:, 3])

    def _to_np(t):
        arr = t.squeeze(0).cpu().numpy().astype(np.float32)
        if arr.shape != (h_n, w_n):
            arr = cv2.resize(arr, (w_n, h_n), interpolation=cv2.INTER_LINEAR)
        return arr

    grid_prob = _to_np(grid_p); text_prob = _to_np(text_p)
    signal_prob = _to_np(signal_p); bg_prob = _to_np(bg_p)

    if save_plots:
        save_channel_heatmap(grid_prob, "Blues", "Canal 0: GRID",
                              out_dir / "01_canal_0_grid.png")
        save_channel_heatmap(text_prob, "Oranges", "Canal 1: TEXTO/FUNDO",
                              out_dir / "02_canal_1_text.png")
        save_channel_heatmap(signal_prob, "Reds", "Canal 2: TRACADO",
                              out_dir / "03_canal_2_signal_heatmap.png")
        save_channel_heatmap(bg_prob, "Greys", "Canal 3: FUNDO",
                              out_dir / "04_canal_3_bg.png")

    signal_binary = (signal_prob > SIGNAL_THRESHOLD).astype(np.uint8) * 255
    mask_bw = 255 - signal_binary
    if save_plots:
        cv2.imwrite(str(out_dir / "05_canal_2_signal_PB.png"), mask_bw)
    H, W = mask_bw.shape
    n_signal = int((signal_prob > SIGNAL_THRESHOLD).sum())
    pct = 100.0 * n_signal / signal_prob.size
    logger.info("    Mascara B&W canal 2: %d px (%.2f%%)", n_signal, pct)

    # PixelSizeFinder
    pxsize = _get_pixel_size_finder()
    with torch.no_grad():
        grid_t = torch.from_numpy(np.ascontiguousarray(grid_prob)).float()
        mm_per_pixel_x, mm_per_pixel_y = pxsize(grid_t)
    px_per_mm = float((1.0 / float(mm_per_pixel_x) + 1.0 / float(mm_per_pixel_y)) / 2.0)
    logger.info("    px/mm: %.3f", px_per_mm)

    # ----- 8. Painel 4 canais (matplotlib) -----
    if save_plots:
        fig = plt.figure(figsize=(15, 11), dpi=110)
        fig.suptitle("STENHEDE — UNet 4 canais", fontsize=14, fontweight="bold", y=0.98)
        for i, (p, t, cm) in enumerate(zip(
                [grid_prob, text_prob, signal_prob, bg_prob],
                ["Canal 0: GRID", "Canal 1: TEXTO/FUNDO", "Canal 2: TRACADO", "Canal 3: FUNDO"],
                ["Blues", "Oranges", "Reds", "Greys"])):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.imshow(p, cmap=cm, vmin=0, vmax=1)
            ax.set_title(t, fontsize=11); ax.axis("off")
        fig.text(0.5, 0.01, f"Tracado: {n_signal} px ({pct:.1f}%)",
                 ha="center", fontsize=10, color="#444")
        fig.savefig(str(out_dir / "06_unet_4canais_combinado.png"),
                    bbox_inches="tight", dpi=110, facecolor="white")
        plt.close(fig)

    # ----- 9. Calibração — defaults BR fixos (OCR removido em v2) -----
    # Brasil é sempre 25 mm/s + 10 mm/mV (padrão SBC). O OCR antigo só
    # confirmava esses valores e gastava ~5-7s carregando um Reader novo.
    cal = {
        "speed_mm_s": 25.0,
        "gain_limb_mm_mV": 10.0,
        "gain_chest_mm_mV": 10.0,
        "_raw_ocr_values": {"speed": None, "limb": None, "chest": None},
    }
    logger.info("[9] Calibração: defaults fixos (25mm/s, 10mm/mV)")

    # ----- 10. Band detection -----
    logger.info("[10] Band detection...")
    binary = mask_bw < 128
    bands = detect_bands(binary)
    logger.info("    Bandas: %d", len(bands))
    for i, (y0, y1) in enumerate(bands):
        logger.info("      banda %d: y[%d:%d] h=%d", i, y0, y1, y1 - y0)
    if len(bands) < 4:
        logger.warning("Esperava 4 bandas, achei %d", len(bands))

    # ----- 11. Detecção de formato (bandas + LeadIdentifier) -----
    logger.info("[11] Detecção dinâmica de formato...")
    bandas_info = detector_bandas(binary, bands)
    li_info = detector_lead_identifier_full(signal_prob, grid_prob, text_prob)
    final_fmt, final_conf = combinar_formato(bandas_info, li_info)
    logger.info(
        "    bandas: %s | LI: %s (cost=%.3f) | FINAL: %s [%s]",
        bandas_info["format"], li_info["format"], li_info["cost"],
        final_fmt, final_conf,
    )
    # px/mm: prefere o do LI se foi calculado; senão usa o do calibrator
    final_px_per_mm = li_info.get("avg_pixel_per_mm") or px_per_mm

    # ----- 12-13. Build leads + bboxes precisos (X/Y trim no signal_binary) -----
    final_base = final_fmt.split("+")[0] if final_fmt != "unknown" else "3x4"
    leads, x_start_common, x_end_common, chunk_w = build_precise_bboxes(
        bands=bands, final_base=final_base,
        signal_binary=binary, px_per_mm=final_px_per_mm,
    )
    logger.info(
        "    formato=%s, %d derivações mapeadas (range x[%d..%d] chunk_w=%d)",
        final_fmt, len(leads), x_start_common, x_end_common, chunk_w,
    )

    # ----- 14. Bboxes overlay -----
    if save_plots:
        plot_bboxes(mask_bw, leads, out_dir / "07_bboxes_overlay.png")

    # ----- 15-17. Viterbi por lead + conversao mV -----
    logger.info("[15-17] Viterbi por derivacao...")
    signals_px = {}; signals_mv = {}; stats = {}
    for L in leads:
        # y_top/y_bot já vêm trimados pelo signal_binary + 5% margem
        # (build_precise_bboxes) — NÃO aplicar VERTICAL_MARGIN_FRAC de novo.
        y_top = L["y_top"]
        y_bot = L["y_bot"]
        crop = mask_bw[y_top:y_bot, L["x_start"]:L["x_end"]]
        sig_px = extrair_sinal_viterbi(crop, invert=True)
        signals_px[L["name"]] = sig_px
        gain = (cal["gain_chest_mm_mV"] if L["name"].startswith("V")
                else cal["gain_limb_mm_mV"])
        sig_mv = sig_px / (gain * px_per_mm)
        signals_mv[L["name"]] = sig_mv
        bin_crop = crop < 128
        cols_with = int(np.any(bin_crop, axis=0).sum())
        stats[L["name"]] = {
            "n": len(sig_px), "valid": int((~np.isnan(sig_px)).sum()),
            "gaps_raw": crop.shape[1] - cols_with,
            "min_mv": float(np.nanmin(sig_mv)),
            "max_mv": float(np.nanmax(sig_mv)),
            "dur_s": len(sig_px) / (cal["speed_mm_s"] * px_per_mm),
            "gain_mm_mV": gain,
        }
        logger.info("    %s: n=%d gaps=%d range=[%.2f,%.2f]mV dur=%.2fs",
                    L["name"], len(sig_px), stats[L["name"]]["gaps_raw"],
                    stats[L["name"]]["min_mv"], stats[L["name"]]["max_mv"],
                    stats[L["name"]]["dur_s"])

    # ----- 17.5. Resample signals_mv para TARGET_HZ (CNN 1D espera 400Hz) -----
    # signals_px continua na taxa original (pixel-rate) — só os mV são
    # reamostrados pra alimentar o classificador.
    n_cols_layout = bandas_info["n_cols"] if bandas_info["n_cols"] > 0 else 4
    sec_per_lead = 10.0 / n_cols_layout       # 2.5 (3x4) / 5 (6x2) / 10 (12x1)
    samples_per_lead = int(sec_per_lead * TARGET_HZ)
    samples_per_rhythm = int(10.0 * TARGET_HZ)
    logger.info(
        "[17.5] Resample signals_mv -> %d Hz (lead=%d, rhythm=%d amostras)",
        TARGET_HZ, samples_per_lead, samples_per_rhythm,
    )
    for lead_name, sig in list(signals_mv.items()):
        if lead_name == RHYTHM_LABEL or lead_name.startswith("rhythm"):
            target = samples_per_rhythm
        else:
            target = samples_per_lead
        if len(sig) > 0 and len(sig) != target:
            signals_mv[lead_name] = scipy_resample(sig, target).astype(np.float32)

    # ----- 18. Save .npy / .csv -----
    logger.info("[18] Save .npy/.csv...")
    if save_plots:
        for name, sig in signals_px.items():
            np.save(out_dir / f"sig_px_{name}.npy", sig)    # taxa original (px)
        for name, sig in signals_mv.items():
            np.save(out_dir / f"sig_mv_{name}.npy", sig)    # 400Hz

    has_rhythm = RHYTHM_LABEL in signals_mv

    # Arquivo unificado pra CNN: (N, TARGET_SAMPLES) com zero-pad
    #   12x1   -> ecg_12_leads_mv.npy   shape (12, 4096)
    #   3x4+1 / 6x2+1 -> ecg_13_leads_mv.npy shape (13, 4096)
    #     canais 0..11 = leads canônicos (ordem ALL_LEADS), com seu len real + pad
    #     canal 12 = II_rhythm (10s); fallback = cópia de DII se ausente
    ecg_channels = 13 if has_rhythm else 12
    ecg_array = np.zeros((ecg_channels, TARGET_SAMPLES), dtype=np.float32)
    for ch_idx, lead_name in enumerate(ALL_LEADS):
        sig = signals_mv.get(lead_name)
        if sig is None or len(sig) == 0:
            continue
        n = min(len(sig), TARGET_SAMPLES)
        ecg_array[ch_idx, :n] = sig[:n]
    if has_rhythm:
        rhythm = signals_mv.get(RHYTHM_LABEL)
        if rhythm is not None and len(rhythm) > 0:
            n = min(len(rhythm), TARGET_SAMPLES)
            ecg_array[12, :n] = rhythm[:n]
        else:
            ecg_array[12] = ecg_array[1]  # fallback: cópia de DII
            logger.warning("II_rhythm ausente — canal 12 = cópia de DII")
    unified_name = "ecg_13_leads_mv.npy" if has_rhythm else "ecg_12_leads_mv.npy"
    np.save(out_dir / unified_name, ecg_array)
    logger.info(
        "Salvo %s shape=%s (%s)",
        unified_name, ecg_array.shape, final_fmt,
    )

    # CSV: 12 leads canônicos sempre; II_rhythm só se o formato detectado
    # tiver rhythm strip (3x4+1 / 6x2+1). 12x1 fica sem coluna rhythm.
    # t_s agora derivado de TARGET_HZ (signals_mv já foi reamostrado).
    if save_plots:
        columns = list(ALL_LEADS) + ([RHYTHM_LABEL] if has_rhythm else [])
        csv_lines = ["t_s," + ",".join(columns)]
        n_min = min(
            (len(signals_mv[L]) for L in ALL_LEADS if L in signals_mv),
            default=0,
        )
        n_rhythm_len = len(signals_mv.get(RHYTHM_LABEL, np.array([])))
        n_csv = max(n_min, n_rhythm_len)
        for i in range(n_csv):
            t_s = i / float(TARGET_HZ)
            vals = []
            for L in ALL_LEADS:
                s = signals_mv.get(L)
                v = s[i] if s is not None and i < len(s) else np.nan
                vals.append(f"{v:.4f}" if not np.isnan(v) else "")
            if has_rhythm:
                r = signals_mv.get(RHYTHM_LABEL)
                rv = r[i] if r is not None and i < len(r) else np.nan
                vals.append(f"{rv:.4f}" if not np.isnan(rv) else "")
            csv_lines.append(f"{t_s:.4f}," + ",".join(vals))
        (out_dir / "ecg_signals.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # ----- 19. Plots finais -----
    if save_plots:
        logger.info("[19] Plots finais...")
        plot_12_leads_grid(signals_px, leads, cal, px_per_mm,
                           out_dir / "08_ecg_digitalizado.png")
        plot_overlay_on_real(normalized, mask_bw, leads, signals_px,
                             out_dir / "09_overlay_sinal_no_ecg_real.png")

    # ----- Summary -----
    fs_original_hz = cal["speed_mm_s"] * px_per_mm
    sum_lines = [
        f"Pipeline unificado — {img_path.name}",
        "=" * 60,
        f"Formato (detectado): {final_fmt}  [{final_conf}]",
        f"  detect_bands: {bandas_info['format']} ({bandas_info['n_bands']} bandas)",
        f"  LeadIdentifier: {li_info['layout_name']} (cost {li_info['cost']:.3f})",
        f"Velocidade: {cal['speed_mm_s']} mm/s",
        f"Ganho membros: {cal['gain_limb_mm_mV']} mm/mV",
        f"Ganho precordiais: {cal['gain_chest_mm_mV']} mm/mV",
        f"px/mm: {px_per_mm:.3f}",
        f"Derivações mapeadas: {len(leads)}",
        f"Range comum: x[{x_start_common}..{x_end_common}] chunk_w={chunk_w}",
        "",
        f"Sampling rate original (pixel-based): ~{fs_original_hz:.0f} Hz",
        f"Sampling rate final (signals_mv):    {TARGET_HZ} Hz (resampleado)",
        f"Amostras por lead: {samples_per_lead}  | rhythm: {samples_per_rhythm if has_rhythm else 0}",
        f"Arquivo unificado: {unified_name} shape={ecg_array.shape} (zero-pad até {TARGET_SAMPLES})",
        "",
        "Derivacoes:",
    ]
    for L in leads:
        s = stats[L["name"]]
        sum_lines.append(
            f"  {L['name']:11s} row={L['row']} x[{L['x_start']}..{L['x_end']}] "
            f"y={L['y_center']} | n={s['n']} valid={s['valid']} gaps={s['gaps_raw']} | "
            f"[{s['min_mv']:+.2f},{s['max_mv']:+.2f}] mV | dur={s['dur_s']:.2f}s"
        )
    if save_plots:
        (out_dir / "summary.txt").write_text("\n".join(sum_lines), encoding="utf-8")

    total = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info("PIPELINE em %.1fs. Outputs em %s", total, out_dir)
    return 0


def main(argv) -> int:
    if len(argv) < 2:
        print("Uso: python pipeline_completo_v2.py <caminho_imagem> [out_dir]")
        return 1
    img_path = Path(argv[1])
    if not img_path.is_file():
        print(f"Arquivo nao existe: {img_path}")
        return 1
    if len(argv) >= 3:
        out_dir = Path(argv[2])
    else:
        out_dir = Path(r"C:\Users\rafae\Desktop\Projeto ECG") / f"{img_path.stem}_unificado"
    try:
        return run_pipeline(img_path, out_dir)
    except Exception:
        logger.error(traceback.format_exc())
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
