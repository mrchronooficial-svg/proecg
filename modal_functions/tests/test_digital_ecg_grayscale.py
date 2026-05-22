"""
Renderiza ECG digital limpo do IMG_1407:
  1. Fundo branco
  2. Grid cinza claro (1mm e 5mm)
  3. Tracado preto a partir do canal 2 da UNet
  4. Labels das derivacoes (I, II, III, aVR, aVL, aVF, V1-V6)
  5. Grayscale 300x300

Output:
  ~/Desktop/Projeto ECG/resultados_teste_v1/ecg_digital_1407.png
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks
from skimage.measure import label as sk_label

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("digital_ecg_grayscale")

ECG_STEM = "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661"
CACHE_PATH = Path(
    rf"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\{ECG_STEM}"
    r"\_cache_pre_signal_extractor.pkl"
)
OUTPUT_PATH = Path(
    rf"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\ecg_digital_{ECG_STEM}.png"
)

TRACE_THRESHOLD = 0.05
MIN_COMP_MASS = 100.0
MIN_COMP_EXTENT_X = 30
MAX_COMP_EXTENT_Y = 200

# Tons de cinza
COLOR_BG = (255, 255, 255)
COLOR_GRID_MINOR = (230, 230, 230)  # cinza muito claro (1mm)
COLOR_GRID_MAJOR = (180, 180, 180)  # cinza medio (5mm)
COLOR_TRACE = (0, 0, 0)              # preto
COLOR_LABEL = (50, 50, 50)           # quase preto

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def _detect_phase(projection: np.ndarray, period: float) -> int:
    L = len(projection)
    P = int(round(period))
    if P < 2 or L < 3 * P:
        return 0
    comb = np.zeros(L, dtype=np.float32)
    for i in range(0, L, P):
        comb[i] = 1.0
    proj = projection - projection.mean()
    mx = float(np.abs(proj).max())
    if mx > 0:
        proj = proj / mx
    corr = correlate(proj, comb, mode="full")
    center = len(corr) // 2
    best_lag = 0
    best_val = -float("inf")
    for lag in range(-P, P + 1):
        v = float(corr[center + lag])
        if v > best_val:
            best_val = v
            best_lag = lag
    return int(best_lag) % P


def _draw_grid(canvas, px_per_mm, off_y, off_x):
    H, W, _ = canvas.shape
    p_min = px_per_mm
    p_maj = 5 * px_per_mm
    # Minor 1mm
    for y in np.arange(off_y, H, p_min):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = COLOR_GRID_MINOR
    for y in np.arange(off_y - p_min, -1, -p_min):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = COLOR_GRID_MINOR
    for x in np.arange(off_x, W, p_min):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = COLOR_GRID_MINOR
    for x in np.arange(off_x - p_min, -1, -p_min):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = COLOR_GRID_MINOR
    # Major 5mm — sobrescreve
    for y in np.arange(off_y, H, p_maj):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = COLOR_GRID_MAJOR
    for y in np.arange(off_y - p_maj, -1, -p_maj):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = COLOR_GRID_MAJOR
    for x in np.arange(off_x, W, p_maj):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = COLOR_GRID_MAJOR
    for x in np.arange(off_x - p_maj, -1, -p_maj):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = COLOR_GRID_MAJOR


def _split_tall_by_valleys(comp_mask):
    rows_count = comp_mask.sum(axis=1)
    smoothed = gaussian_filter1d(rows_count.astype(float), sigma=5)
    inv = -smoothed
    valleys, _ = find_peaks(inv, distance=40, prominence=smoothed.max() * 0.3)
    nz = np.where(rows_count > 0)[0]
    if nz.size == 0:
        return [comp_mask]
    y_min, y_max = int(nz.min()), int(nz.max())
    vals = sorted([int(v) for v in valleys if y_min < v < y_max])
    if not vals:
        return [comp_mask]
    cuts = [y_min] + vals + [y_max + 1]
    subs = []
    for i in range(len(cuts) - 1):
        sub = comp_mask.copy()
        sub[:cuts[i]] = False
        sub[cuts[i + 1]:] = False
        if sub.any():
            subs.append(sub)
    return subs if len(subs) > 1 else [comp_mask]


def _detect_lead_band_ys(signal_prob, threshold):
    """Detecta Y central de cada lead band — com split-tall pra recuperar
    leads mesclados pelo vinco do papel."""
    binary = signal_prob > threshold
    labeled = sk_label(binary, connectivity=1)
    n = int(labeled.max())
    bands = []

    def _accept(cm):
        if signal_prob[cm].sum() < MIN_COMP_MASS:
            return
        rows = np.where(cm.any(axis=1))[0]
        cols = np.where(cm.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return
        ex_x = cols.max() - cols.min()
        if ex_x < MIN_COMP_EXTENT_X:
            return
        bands.append((float(rows.mean()), cm))

    for lid in range(1, n + 1):
        cm = labeled == lid
        if signal_prob[cm].sum() < MIN_COMP_MASS:
            continue
        rows = np.where(cm.any(axis=1))[0]
        if rows.size == 0:
            continue
        ex_y = rows.max() - rows.min()
        if ex_y <= MAX_COMP_EXTENT_Y:
            _accept(cm)
            continue
        # Tall — split por valleys, ou fallback bands de 120 px
        subs = _split_tall_by_valleys(cm)
        if len(subs) == 1:
            band_h = 120
            n_bands = int(np.ceil(ex_y / band_h))
            for b in range(n_bands):
                y0 = int(rows.min() + b * band_h)
                y1 = int(min(rows.min() + (b + 1) * band_h, rows.max() + 1))
                sub = cm.copy()
                sub[:y0] = False
                sub[y1:] = False
                if sub.any():
                    _accept(sub)
        else:
            for sub in subs:
                _accept(sub)
    bands.sort(key=lambda t: t[0])
    return bands


def main() -> int:
    if not CACHE_PATH.is_file():
        logger.error("Cache nao encontrado")
        return 1
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)

    signal_prob = cache["signal_prob"]
    grid_prob = cache["grid_prob"]
    px_per_mm = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape
    logger.info("Shape %dx%d, px_per_mm=%.3f", H, W, px_per_mm)

    # 1. Canvas branco
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # 2. Grid alinhado com o original
    off_y = _detect_phase(grid_prob.sum(axis=1), 5 * px_per_mm)
    off_x = _detect_phase(grid_prob.sum(axis=0), 5 * px_per_mm)
    _draw_grid(canvas, px_per_mm, off_y, off_x)
    logger.info("Grid plotado (fase y=%d x=%d)", off_y, off_x)

    # 3. Tracado preto solido
    trace_mask = signal_prob > TRACE_THRESHOLD
    canvas[trace_mask] = COLOR_TRACE
    logger.info("Tracado plotado: %d pixels", int(trace_mask.sum()))

    # 4. Labels DESATIVADOS — somente grid + tracado

    # 5. Grayscale + pad pra aspect 2:1 (= 1200/600) + resize final
    TARGET_W, TARGET_H = 1200, 600
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    target_aspect = TARGET_W / TARGET_H  # 2.0
    src_aspect = w / h                    # ~1.575
    if src_aspect < target_aspect:
        # Pad horizontal pra ficar mais wide
        new_w = int(round(h * target_aspect))
        pad_lef = (new_w - w) // 2
        pad_rig = new_w - w - pad_lef
        pad_top = pad_bot = 0
    else:
        # Pad vertical pra ficar mais tall
        new_h = int(round(w / target_aspect))
        pad_top = (new_h - h) // 2
        pad_bot = new_h - h - pad_top
        pad_lef = pad_rig = 0
    gray_padded = cv2.copyMakeBorder(
        gray, pad_top, pad_bot, pad_lef, pad_rig,
        cv2.BORDER_CONSTANT, value=255,
    )
    gray_resized = cv2.resize(
        gray_padded, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), gray_resized)
    logger.info(
        "Salvo: %s (%dx%d grayscale, original %dx%d com pad pra aspect %.2f)",
        OUTPUT_PATH, TARGET_W, TARGET_H, w, h, target_aspect,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
