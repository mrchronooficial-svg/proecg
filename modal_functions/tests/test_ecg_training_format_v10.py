"""
v10: igual ao v9 + binary_dilation(iterations=1) pra trace ~2 px (formato treino).

Pipeline:
  1. Renderiza igual v8 (alpha + boost + resize + grid)
  2. Skeletonize (1 px)
  3. binary_dilation iterations=1 nos pixels < 128 -> ~2 px
  4. Repinta: trace dilatado em preto puro, resto canvas branco com grid
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecg_training_format_v10")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CANVAS_W = 900
MIN_CANVAS_H = 450
MARGIN = 40
TRACE_THRESHOLD = 0.05
BBOX_THRESHOLD = 0.5
ALPHA_BOOST = 1.5
FIXED_PX_PER_MM = 3.0
TRACE_DARK_THRESHOLD = 200
DILATE_THRESHOLD = 128  # spec: pixels < 128


def make_v10(cache):
    signal_prob = cache["signal_prob"]
    px_per_mm_orig = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape

    core = signal_prob > BBOX_THRESHOLD
    rows_with = np.any(core, axis=1)
    cols_with = np.any(core, axis=0)
    if not rows_with.any() or not cols_with.any():
        return np.full((MIN_CANVAS_H, MIN_CANVAS_W), 255, dtype=np.uint8), 0, 0, 0
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

    trace_binary = trace_resized < TRACE_DARK_THRESHOLD
    trace_skeleton = skeletonize(trace_binary)
    # Skeleton vira preto puro num layer branco
    trace_skel_img = np.full_like(trace_resized, 255)
    trace_skel_img[trace_skeleton] = 0

    # Dilatation: pixels < 128 -> dilate iter=1
    dilate_input = trace_skel_img < DILATE_THRESHOLD
    dilated = binary_dilation(dilate_input, iterations=1)
    trace_thick = np.full_like(trace_resized, 255)
    trace_thick[dilated] = 0

    canvas_w = max(MIN_CANVAS_W, new_w + 2 * MARGIN)
    canvas_h = max(MIN_CANVAS_H, new_h + 2 * MARGIN)
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    offset_x = (canvas_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2

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

    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.minimum(
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w],
        trace_thick,
    )

    # Medicao da espessura final
    dist = distance_transform_edt(dilated)
    if dilated.any():
        thickness_med = float(np.median(2 * dist[dilated]))
        thickness_max = float(np.max(2 * dist[dilated]))
    else:
        thickness_med = thickness_max = 0.0

    return canvas, new_w, new_h, int(dilated.sum()), thickness_med, thickness_max


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.warning("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        img, new_w, new_h, n_dil, t_med, t_max = make_v10(cache)
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), img)
        canvas_h, canvas_w = img.shape
        logger.info(
            "Salvo: %s (canvas %dx%d, ECG %dx%d a %.1f px/mm, dilated %d px, espessura med=%.1f max=%.1f)",
            out_path.name, canvas_w, canvas_h, new_w, new_h, FIXED_PX_PER_MM,
            n_dil, t_med, t_max,
        )
    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
