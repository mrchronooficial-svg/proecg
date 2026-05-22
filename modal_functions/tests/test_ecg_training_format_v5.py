"""
v5: trace alpha gradient (v4) + grid desenhado NO TARGET com px_per_mm spec.

Pipeline:
  1. Trace alpha gradient * boost em full res (SEM grid)
  2. Bounding box do tracado (signal_prob > 0.5)
  3. Converter bbox pra mm com px_per_mm original
  4. new_px_per_mm = min(820/width_mm, 370/height_mm)
  5. Resize trace pra (new_w, new_h) INTER_AREA
  6. Centralizar em 900x450 com 40px margem
  7. Desenhar grid 232/245 NO 900x450 final usando new_px_per_mm
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
logger = logging.getLogger("ecg_training_format_v5")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 900
CANVAS_H = 450
MARGIN = 40
USABLE_W = CANVAS_W - 2 * MARGIN  # 820
USABLE_H = CANVAS_H - 2 * MARGIN  # 370
TRACE_THRESHOLD = 0.05
BBOX_THRESHOLD = 0.5   # threshold pra bbox do core (alta confianca)
ALPHA_BOOST = 1.5


def make_v5(cache):
    signal_prob = cache["signal_prob"]
    px_per_mm_orig = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape

    # 1. Bounding box do core (signal_prob > 0.5)
    core_mask = signal_prob > BBOX_THRESHOLD
    rows_with = np.any(core_mask, axis=1)
    cols_with = np.any(core_mask, axis=0)
    if not rows_with.any() or not cols_with.any():
        return np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8), 0, 0, 0
    y_min = int(np.argmax(rows_with))
    y_max = H - int(np.argmax(rows_with[::-1]))
    x_min = int(np.argmax(cols_with))
    x_max = W - int(np.argmax(cols_with[::-1]))
    bbox_w_px = x_max - x_min
    bbox_h_px = y_max - y_min
    width_mm = bbox_w_px / px_per_mm_orig
    height_mm = bbox_h_px / px_per_mm_orig

    # 2. Spec: new_px_per_mm = min(820/width_mm, 370/height_mm)
    new_px_per_mm = min(USABLE_W / width_mm, USABLE_H / height_mm)
    new_w = int(round(width_mm * new_px_per_mm))
    new_h = int(round(height_mm * new_px_per_mm))

    # 3. Trace alpha gradient em full res (sem grid) — soh dentro do bbox
    trace_canvas_full = np.full((bbox_h_px, bbox_w_px), 255.0, dtype=np.float32)
    sp_crop = signal_prob[y_min:y_max, x_min:x_max]
    trace_mask = sp_crop > TRACE_THRESHOLD
    trace_intensity = np.clip(
        sp_crop / max(signal_prob.max(), 1e-6) * ALPHA_BOOST, 0, 1,
    )
    alpha_t = trace_intensity * trace_mask.astype(np.float32)
    trace_canvas_full = trace_canvas_full * (1.0 - alpha_t)
    trace_canvas_full = trace_canvas_full.astype(np.uint8)

    # 4. Resize trace pro tamanho final (new_w, new_h) INTER_AREA
    trace_resized = cv2.resize(
        trace_canvas_full, (new_w, new_h), interpolation=cv2.INTER_AREA,
    )

    # 5. Canvas branco final 900x450
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    offset_x = (CANVAS_W - new_w) // 2
    offset_y = (CANVAS_H - new_h) // 2

    # 6. Grid 232/245 desenhado NO TARGET 900x450 usando new_px_per_mm
    p_min = new_px_per_mm
    p_maj = 5 * new_px_per_mm
    # 1mm (245)
    y = offset_y % p_min
    while y < CANVAS_H:
        yi = int(round(y))
        if 0 <= yi < CANVAS_H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 245)
        y += p_min
    x = offset_x % p_min
    while x < CANVAS_W:
        xi = int(round(x))
        if 0 <= xi < CANVAS_W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 245)
        x += p_min
    # 5mm (232) — sobrescrevem
    y = offset_y % p_maj
    while y < CANVAS_H:
        yi = int(round(y))
        if 0 <= yi < CANVAS_H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 232)
        y += p_maj
    x = offset_x % p_maj
    while x < CANVAS_W:
        xi = int(round(x))
        if 0 <= xi < CANVAS_W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 232)
        x += p_maj

    # 7. Compor: trace resized POR CIMA do canvas com grid, usando minimum
    #    (pixels escuros do trace sobrescrevem os mais claros do grid)
    h_eff = min(new_h, CANVAS_H - offset_y)
    w_eff = min(new_w, CANVAS_W - offset_x)
    canvas[offset_y:offset_y + h_eff, offset_x:offset_x + w_eff] = np.minimum(
        canvas[offset_y:offset_y + h_eff, offset_x:offset_x + w_eff],
        trace_resized[:h_eff, :w_eff],
    )

    return canvas, new_px_per_mm, new_w, new_h


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.warning("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        img, new_pxmm, new_w, new_h = make_v5(cache)
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), img)
        logger.info(
            "Salvo: %s (900x450, ECG %dx%d em %.3f px/mm)",
            out_path.name, new_w, new_h, new_pxmm,
        )
    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
