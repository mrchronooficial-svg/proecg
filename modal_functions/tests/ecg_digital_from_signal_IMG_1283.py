"""
Monta ECG digital a partir do sinal extraido via Viterbi (IMG_1283).

Pipeline:
  1. Carrega mascara B&W
  2. Detecta bandas (3 lead rows + 1 rhythm)
  3. Roda Viterbi POR LEAD (split de cada banda em 4 chunks)
  4. Banda 4 (rhythm) -> Viterbi na largura inteira
  5. Constroi canvas 3 px/mm com grid 232/245 + tracado nos seus lugares

Escala:
  - 25 mm/s (paper speed)
  - 10 mm/mV (gain)
  - 3 px/mm output (igual v14)
  - 1mm = 3x3 px | 5mm = 15x15 px

Layout 3x4+1 padrao:
  Row 1: I    aVR  V1  V4
  Row 2: II   aVL  V2  V5
  Row 3: III  aVF  V3  V6
  Row 4: II (rhythm strip, largura inteira)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

from pipeline.digitize.viterbi_extractor import extrair_sinal_viterbi  # noqa: E402

# ----- Input / output -----
MASK_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1283_pipeline_completo\05_canal_2_signal_PB.png")
OUTPUT_DIR = MASK_PATH.parent
OUTPUT_PATH = OUTPUT_DIR / "10_ecg_digital_reconstruido.png"

# ----- Constants -----
PX_PER_MM_ORIG = 13.0  # do PixelSizeFinder
PX_PER_MM_OUT = 3.0
MM_PER_MV = 10.0  # gain padrao
MM_PER_SEC = 25.0  # paper speed

LEAD_ROW_HEIGHT_MM = 30  # altura alocada por linha de leads (3mV range)
RHYTHM_HEIGHT_MM = 30
ROW_GAP_MM = 5
MARGIN_MM_X = 10
MARGIN_MM_Y = 8

LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LABEL = "II"

GRID_MINOR_GRAY = 245
GRID_MAJOR_GRAY = 232
TRACE_GRAY = 0
LABEL_COLOR = (0, 0, 0)


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


def draw_grid(canvas, px_per_mm, offset_x=0, offset_y=0):
    H, W = canvas.shape
    p_min = px_per_mm
    p_maj = 5 * px_per_mm
    y = offset_y % p_min
    while y < H:
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = np.minimum(canvas[yi, :], GRID_MINOR_GRAY)
        y += p_min
    x = offset_x % p_min
    while x < W:
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = np.minimum(canvas[:, xi], GRID_MINOR_GRAY)
        x += p_min
    y = offset_y % p_maj
    while y < H:
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = np.minimum(canvas[yi, :], GRID_MAJOR_GRAY)
        y += p_maj
    x = offset_x % p_maj
    while x < W:
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = np.minimum(canvas[:, xi], GRID_MAJOR_GRAY)
        x += p_maj


def plot_signal(canvas, signal, x_start_px, y_center_px, scale_x_per_col, y_scale):
    """Plota signal (1D, valores em px originais, baseline em 0) no canvas.
    signal[col] = valor em pixels do orig (negativo = pra baixo, ja negado -> +up).
    y_scale = PX_PER_MM_OUT / PX_PER_MM_ORIG (preserva relacao mm)."""
    n = len(signal)
    pts = []
    for col in range(n):
        v = signal[col]
        if np.isnan(v):
            continue
        x = x_start_px + col * scale_x_per_col
        y = y_center_px - v * y_scale  # px_orig -> px out preservando mm
        pts.append((int(round(x)), int(round(y))))
    if len(pts) < 2:
        return
    pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts_arr], isClosed=False, color=TRACE_GRAY,
                  thickness=1, lineType=cv2.LINE_AA)


def main() -> int:
    print(f"Carregando: {MASK_PATH}")
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERRO: nao consegui ler {MASK_PATH}")
        return 1
    H, W = mask.shape
    print(f"Mascara: {W}x{H}")

    binary = mask < 128
    bands = detect_bands(binary)
    if len(bands) < 4:
        print(f"AVISO: detectei {len(bands)} bandas (esperava 4 pra 3x4+1)")
    print(f"Bandas: {bands}")

    chunk_px = W // 4
    print(f"Chunk px (largura/4): {chunk_px}")

    # ----- Extracao por lead -----
    print("\n--- Viterbi por lead ---")
    lead_signals = {}
    for row_idx, (band, row_leads) in enumerate(zip(bands[:3], LEAD_LAYOUT)):
        y0, y1 = band
        band_mask = mask[y0:y1, :]
        for col_idx, lead_name in enumerate(row_leads):
            x_start = col_idx * chunk_px
            x_end = (col_idx + 1) * chunk_px if col_idx < 3 else W
            chunk_mask = band_mask[:, x_start:x_end]
            sig = extrair_sinal_viterbi(chunk_mask, invert=True)
            lead_signals[lead_name] = sig
            print(f"  {lead_name}: {len(sig)} cols | range "
                  f"[{np.nanmin(sig):.1f}, {np.nanmax(sig):.1f}]")

    # Rhythm
    if len(bands) >= 4:
        y0, y1 = bands[3]
        rhythm_mask = mask[y0:y1, :]
        rhythm_sig = extrair_sinal_viterbi(rhythm_mask, invert=True)
        print(f"  {RHYTHM_LABEL} (rhythm): {len(rhythm_sig)} cols | range "
              f"[{np.nanmin(rhythm_sig):.1f}, {np.nanmax(rhythm_sig):.1f}]")
    else:
        rhythm_sig = None

    # ----- Dimensoes do canvas -----
    # Largura: tempo total em mm × 3 px/mm + margens
    total_width_mm = W / PX_PER_MM_ORIG  # ex: 3835/13 = 295 mm
    chunk_width_mm = chunk_px / PX_PER_MM_ORIG  # ~73.7 mm
    content_width_px = int(round(total_width_mm * PX_PER_MM_OUT))
    margin_x_px = int(round(MARGIN_MM_X * PX_PER_MM_OUT))
    canvas_w = content_width_px + 2 * margin_x_px

    row_h_px = int(round(LEAD_ROW_HEIGHT_MM * PX_PER_MM_OUT))
    rhythm_h_px = int(round(RHYTHM_HEIGHT_MM * PX_PER_MM_OUT))
    gap_px = int(round(ROW_GAP_MM * PX_PER_MM_OUT))
    margin_y_px = int(round(MARGIN_MM_Y * PX_PER_MM_OUT))
    content_height = 3 * row_h_px + 2 * gap_px + gap_px + rhythm_h_px
    canvas_h = content_height + 2 * margin_y_px
    print(f"\nCanvas: {canvas_w}x{canvas_h} @ {PX_PER_MM_OUT} px/mm")

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    draw_grid(canvas, PX_PER_MM_OUT, offset_x=margin_x_px, offset_y=margin_y_px)

    # ----- Plot leads -----
    chunk_w_out = int(round(chunk_width_mm * PX_PER_MM_OUT))
    y_scale = PX_PER_MM_OUT / PX_PER_MM_ORIG  # converte px_orig -> px_out preservando mm
    scale_x = PX_PER_MM_OUT / PX_PER_MM_ORIG  # conversao col_orig -> col_out

    for row_idx, row_leads in enumerate(LEAD_LAYOUT):
        y_center = margin_y_px + row_idx * (row_h_px + gap_px) + row_h_px // 2
        for col_idx, lead_name in enumerate(row_leads):
            sig = lead_signals.get(lead_name)
            if sig is None:
                continue
            x_start = margin_x_px + col_idx * chunk_w_out
            plot_signal(canvas, sig, x_start, y_center, scale_x, y_scale)
            # Label do lead
            cv2.putText(canvas, lead_name,
                        (x_start + 4, y_center - row_h_px // 2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1, cv2.LINE_AA)

    # Rhythm
    if rhythm_sig is not None:
        y_center = margin_y_px + 3 * (row_h_px + gap_px) + rhythm_h_px // 2
        plot_signal(canvas, rhythm_sig, margin_x_px, y_center, scale_x, y_scale)
        cv2.putText(canvas, RHYTHM_LABEL + " (rhythm)",
                    (margin_x_px + 4, y_center - rhythm_h_px // 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1, cv2.LINE_AA)

    cv2.imwrite(str(OUTPUT_PATH), canvas)
    print(f"\nECG digital salvo: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
