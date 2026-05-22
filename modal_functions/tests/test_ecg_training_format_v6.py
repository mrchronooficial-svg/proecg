"""
v6: trace preenche EXATAMENTE a area util 820x370 (sem preservar aspect).

  - bbox do tracado resized pra 820x370 com INTER_AREA
  - Grid 232/245 desenhado no canvas com:
      x_px_per_mm = 820 / width_mm
      y_px_per_mm = 370 / height_mm
    (px_per_mm diferente em X e Y se aspect nao corresponder)
  - 40 px margem todos os lados
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecg_training_format_v6")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 900
CANVAS_H = 450
MARGIN = 40
USABLE_W = CANVAS_W - 2 * MARGIN  # 820
USABLE_H = CANVAS_H - 2 * MARGIN  # 370
TRACE_THRESHOLD = 0.05
BBOX_THRESHOLD = 0.5
ALPHA_BOOST = 1.5


def make_v6(cache):
    signal_prob = cache["signal_prob"]
    px_per_mm_orig = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape

    # 1. Bbox do core
    core = signal_prob > BBOX_THRESHOLD
    rows_with = np.any(core, axis=1)
    cols_with = np.any(core, axis=0)
    if not rows_with.any() or not cols_with.any():
        return np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8), 0, 0
    y_min = int(np.argmax(rows_with))
    y_max = H - int(np.argmax(rows_with[::-1]))
    x_min = int(np.argmax(cols_with))
    x_max = W - int(np.argmax(cols_with[::-1]))
    bbox_w_px = x_max - x_min
    bbox_h_px = y_max - y_min
    width_mm = bbox_w_px / px_per_mm_orig
    height_mm = bbox_h_px / px_per_mm_orig

    # 2. Trace alpha gradient na bbox em full res
    sp_crop = signal_prob[y_min:y_max, x_min:x_max]
    trace_full = np.full((bbox_h_px, bbox_w_px), 255.0, dtype=np.float32)
    trace_mask = sp_crop > TRACE_THRESHOLD
    trace_intensity = np.clip(
        sp_crop / max(signal_prob.max(), 1e-6) * ALPHA_BOOST, 0, 1,
    )
    alpha_t = trace_intensity * trace_mask.astype(np.float32)
    trace_full = trace_full * (1.0 - alpha_t)
    trace_full = trace_full.astype(np.uint8)

    # 3. STRETCH RESIZE pra 820x370 (sem preservar aspect)
    trace_resized = cv2.resize(
        trace_full, (USABLE_W, USABLE_H), interpolation=cv2.INTER_AREA,
    )

    # 4. Canvas branco 900x450
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)

    # 5. Grid com px_per_mm DIFERENTE em X e Y
    x_px_per_mm = USABLE_W / width_mm
    y_px_per_mm = USABLE_H / height_mm

    # 1mm (245)
    y = MARGIN
    while y < CANVAS_H:
        yi = int(round(y))
        if 0 <= yi < CANVAS_H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 245)
        y += y_px_per_mm
    y = MARGIN - y_px_per_mm
    while y >= 0:
        yi = int(round(y))
        if 0 <= yi < CANVAS_H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 245)
        y -= y_px_per_mm
    x = MARGIN
    while x < CANVAS_W:
        xi = int(round(x))
        if 0 <= xi < CANVAS_W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 245)
        x += x_px_per_mm
    x = MARGIN - x_px_per_mm
    while x >= 0:
        xi = int(round(x))
        if 0 <= xi < CANVAS_W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 245)
        x -= x_px_per_mm
    # 5mm (232)
    y = MARGIN
    while y < CANVAS_H:
        yi = int(round(y))
        if 0 <= yi < CANVAS_H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 232)
        y += 5 * y_px_per_mm
    y = MARGIN - 5 * y_px_per_mm
    while y >= 0:
        yi = int(round(y))
        if 0 <= yi < CANVAS_H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 232)
        y -= 5 * y_px_per_mm
    x = MARGIN
    while x < CANVAS_W:
        xi = int(round(x))
        if 0 <= xi < CANVAS_W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 232)
        x += 5 * x_px_per_mm
    x = MARGIN - 5 * x_px_per_mm
    while x >= 0:
        xi = int(round(x))
        if 0 <= xi < CANVAS_W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 232)
        x -= 5 * x_px_per_mm

    # 6. Posiciona trace em (40, 40) e compoe (np.minimum)
    canvas[MARGIN:MARGIN + USABLE_H, MARGIN:MARGIN + USABLE_W] = np.minimum(
        canvas[MARGIN:MARGIN + USABLE_H, MARGIN:MARGIN + USABLE_W],
        trace_resized,
    )

    return canvas, x_px_per_mm, y_px_per_mm


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.warning("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        img, xpx, ypx = make_v6(cache)
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), img)
        logger.info(
            "Salvo: %s (900x450, x_px/mm=%.2f y_px/mm=%.2f)",
            out_path.name, xpx, ypx,
        )
    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
