"""
v9: igual ao v8 (escala 3 px/mm + grid 232/245) MAS com skeletonize
final pra trace de 1 px exato sem perder topologia.

Pipeline:
  1. Renderiza igual v8 (alpha + boost + resize + grid)
  2. Identifica trace pixels (img < 100)
  3. Skeletonize → 1 px
  4. Repinta: trace original vira branco, skeleton vira preto puro
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecg_training_format_v9")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CANVAS_W = 900
MIN_CANVAS_H = 450
MARGIN = 40
TRACE_THRESHOLD = 0.05
BBOX_THRESHOLD = 0.5
ALPHA_BOOST = 1.5
FIXED_PX_PER_MM = 3.0
TRACE_DARK_THRESHOLD = 200  # pra capturar core+halo no rendered image


def make_v9(cache):
    signal_prob = cache["signal_prob"]
    px_per_mm_orig = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape

    # Bbox
    core = signal_prob > BBOX_THRESHOLD
    rows_with = np.any(core, axis=1)
    cols_with = np.any(core, axis=0)
    if not rows_with.any() or not cols_with.any():
        return np.full((MIN_CANVAS_H, MIN_CANVAS_W), 255, dtype=np.uint8), 0, 0
    y_min = int(np.argmax(rows_with))
    y_max = H - int(np.argmax(rows_with[::-1]))
    x_min = int(np.argmax(cols_with))
    x_max = W - int(np.argmax(cols_with[::-1]))
    bbox_w_px = x_max - x_min
    bbox_h_px = y_max - y_min
    width_mm = bbox_w_px / px_per_mm_orig
    height_mm = bbox_h_px / px_per_mm_orig

    new_w = int(round(width_mm * FIXED_PX_PER_MM))
    new_h = int(round(height_mm * FIXED_PX_PER_MM))

    # Render alpha gradient
    sp_crop = signal_prob[y_min:y_max, x_min:x_max]
    trace_full = np.full((bbox_h_px, bbox_w_px), 255.0, dtype=np.float32)
    trace_mask = sp_crop > TRACE_THRESHOLD
    trace_intensity = np.clip(
        sp_crop / max(signal_prob.max(), 1e-6) * ALPHA_BOOST, 0, 1,
    )
    alpha_t = trace_intensity * trace_mask.astype(np.float32)
    trace_full = trace_full * (1.0 - alpha_t)
    trace_full = trace_full.astype(np.uint8)
    trace_resized = cv2.resize(trace_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # === NOVO: skeletonize do trace_resized ===
    trace_binary = trace_resized < TRACE_DARK_THRESHOLD
    trace_skeleton = skeletonize(trace_binary)
    # Rebuild trace layer: branco com skeleton em preto puro
    trace_thin = np.full_like(trace_resized, 255)
    trace_thin[trace_skeleton] = 0

    # Canvas
    canvas_w = max(MIN_CANVAS_W, new_w + 2 * MARGIN)
    canvas_h = max(MIN_CANVAS_H, new_h + 2 * MARGIN)
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    offset_x = (canvas_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2

    # Grid 232/245 a 3 px/mm
    p_min = FIXED_PX_PER_MM
    p_maj = 5 * FIXED_PX_PER_MM
    y = offset_y % p_min
    while y < canvas_h:
        yi = int(round(y))
        if 0 <= yi < canvas_h:
            canvas[yi, :] = np.minimum(canvas[yi, :], 245)
        y += p_min
    x = offset_x % p_min
    while x < canvas_w:
        xi = int(round(x))
        if 0 <= xi < canvas_w:
            canvas[:, xi] = np.minimum(canvas[:, xi], 245)
        x += p_min
    y = offset_y % p_maj
    while y < canvas_h:
        yi = int(round(y))
        if 0 <= yi < canvas_h:
            canvas[yi, :] = np.minimum(canvas[yi, :], 232)
        y += p_maj
    x = offset_x % p_maj
    while x < canvas_w:
        xi = int(round(x))
        if 0 <= xi < canvas_w:
            canvas[:, xi] = np.minimum(canvas[:, xi], 232)
        x += p_maj

    # Compose trace skeleton
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.minimum(
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w],
        trace_thin,
    )

    return canvas, new_w, new_h, int(trace_skeleton.sum())


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.warning("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        img, new_w, new_h, n_skel = make_v9(cache)
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), img)
        canvas_h, canvas_w = img.shape
        logger.info(
            "Salvo: %s (canvas %dx%d, ECG %dx%d a %.1f px/mm, skeleton %d px)",
            out_path.name, canvas_w, canvas_h, new_w, new_h, FIXED_PX_PER_MM, n_skel,
        )
    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
