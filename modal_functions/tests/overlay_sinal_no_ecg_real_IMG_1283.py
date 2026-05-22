"""
Sobrepoe sinais Viterbi extraidos no ECG real (normalized) pra comparacao visual.

Para cada derivacao, o sinal (Y em pixels, baseline removida, +Y = pra cima)
e plotado em vermelho sobre a imagem normalizada, no Y do centro da banda
correspondente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

NORM_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1283_pipeline_completo\00_normalized.png")
MASK_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1283_pipeline_completo\05_canal_2_signal_PB.png")
SIG_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1283_separacao_ocr")
OUTPUT_DIR = SIG_DIR
OUT_PATH = OUTPUT_DIR / "04_overlay_sinal_no_ecg_real.png"
OUT_MASK_PATH = OUTPUT_DIR / "05_overlay_sinal_na_mascara.png"

LEAD_LAYOUT_3x4 = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LABEL = "II_rhythm"


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


def find_signal_baseline_y(mask: np.ndarray, lead) -> int:
    """Encontra a linha Y onde o tracado real esta concentrado (mediana das
    linhas com pixel do tracado, dentro do crop). Usa pra alinhar o overlay
    melhor que o centro geometrico da banda."""
    y0, y1 = lead["y0"], lead["y1"]
    x0, x1 = lead["x_start"], lead["x_end"]
    crop_binary = mask[y0:y1, x0:x1] < 128
    if not crop_binary.any():
        return (y0 + y1) // 2
    ys = np.where(crop_binary)[0]
    return int(np.median(ys)) + y0


def find_trace_bounds_per_band(binary, y0, y1, px_per_mm=13.0,
                                cal_pulse_max_mm=12.0, sustained_window=50,
                                sustained_min_cols=25, col_min_pixels=3):
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
                        candidate_start = j; break
                break
    x_end = min(W, int(sustained_idx[-1]) + sustained_window // 2)
    return candidate_start, x_end


def main() -> int:
    norm = cv2.imread(str(NORM_PATH))
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if norm is None or mask is None:
        print("ERRO: nao consegui ler imagens")
        return 1
    H, W = mask.shape

    # Rebuild leads structure (mesma logica do script de separacao corrigido)
    binary = mask < 128
    bands = detect_bands(binary)
    # bounds reais por banda (skip pulso calibracao)
    band_bounds = []
    for i, (y0, y1) in enumerate(bands[:3]):
        xs, xe = find_trace_bounds_per_band(binary, y0, y1)
        band_bounds.append((xs, xe))
    x_start_common = max(b[0] for b in band_bounds) if band_bounds else 0
    x_end_common = min(b[1] for b in band_bounds) if band_bounds else W
    chunk_w = (x_end_common - x_start_common) // 4
    print(f"Range comum: x[{x_start_common}..{x_end_common}] chunk_w={chunk_w}")

    leads = []
    for row_idx, row_leads in enumerate(LEAD_LAYOUT_3x4):
        if row_idx >= len(bands):
            break
        y0, y1 = bands[row_idx]
        for col_idx, name in enumerate(row_leads):
            x_start = x_start_common + col_idx * chunk_w
            x_end = (x_start_common + (col_idx + 1) * chunk_w
                     if col_idx < 3 else x_end_common)
            leads.append({
                "name": name, "row": row_idx,
                "x_start": x_start, "x_end": x_end,
                "y0": y0, "y1": y1,
            })
    if len(bands) >= 4:
        y0, y1 = bands[3]
        xs, xe = find_trace_bounds_per_band(binary, y0, y1)
        leads.append({
            "name": RHYTHM_LABEL, "row": 3,
            "x_start": xs, "x_end": xe,
            "y0": y0, "y1": y1,
        })

    # Compute baseline Y per lead (where signal real esta no canal 2)
    for L in leads:
        L["y_baseline"] = find_signal_baseline_y(mask, L)
        print(f"{L['name']:11s} x[{L['x_start']:4d}..{L['x_end']:4d}] "
              f"y_band=[{L['y0']}..{L['y1']}] y_baseline={L['y_baseline']}")

    # ----- Plot 1: overlay no ECG real (normalized) -----
    print("\nGerando overlay sobre ECG real...")
    fig, ax = plt.subplots(figsize=(20, 7), dpi=120)
    ax.imshow(cv2.cvtColor(norm, cv2.COLOR_BGR2RGB), aspect="equal")
    for L in leads:
        sig_path = SIG_DIR / f"sig_px_{L['name']}.npy"
        if not sig_path.is_file():
            print(f"  AVISO: nao achei {sig_path.name}")
            continue
        sig = np.load(sig_path)
        x = np.arange(L["x_start"], L["x_start"] + len(sig))
        # Y na imagem = baseline - sinal (sinal positivo = QRS pra cima = Y menor)
        y = L["y_baseline"] - sig
        color = "#22cc44" if L["name"] == RHYTHM_LABEL else "#ff2222"
        lw = 0.5 if L["name"] == RHYTHM_LABEL else 0.7
        ax.plot(x, y, color=color, linewidth=lw, alpha=0.85,
                label=L["name"] if L["row"] == 0 else None)
        # Label discreto
        ax.text(L["x_start"] + 8, L["y_baseline"] - 50, L["name"],
                color="red", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
    ax.set_title("Sinais Viterbi (vermelho) sobre ECG real undistorted (verde = rhythm strip)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(OUT_PATH), bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Salvo: {OUT_PATH}")

    # ----- Plot 2: overlay so na mascara canal 2 -----
    print("Gerando overlay sobre mascara canal 2...")
    fig, ax = plt.subplots(figsize=(20, 7), dpi=120)
    ax.imshow(mask, cmap="gray", aspect="equal")
    for L in leads:
        sig_path = SIG_DIR / f"sig_px_{L['name']}.npy"
        if not sig_path.is_file():
            continue
        sig = np.load(sig_path)
        x = np.arange(L["x_start"], L["x_start"] + len(sig))
        y = L["y_baseline"] - sig
        color = "#22cc44" if L["name"] == RHYTHM_LABEL else "#ff2222"
        lw = 0.6 if L["name"] == RHYTHM_LABEL else 0.8
        ax.plot(x, y, color=color, linewidth=lw, alpha=0.85)
        ax.text(L["x_start"] + 8, L["y_baseline"] - 50, L["name"],
                color="red", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
    ax.set_title("Sinais Viterbi (vermelho) sobre mascara canal 2",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(OUT_MASK_PATH), bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Salvo: {OUT_MASK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
