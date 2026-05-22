"""
v3: producao = formato treino
  - Threshold 0.5 (core da mascara, ~1px de espessura)
  - INTER_AREA resize (anti-aliased, evita escadas)
  - Re-threshold pixels escuros → preto puro
  - Grid 232/245 desenhado por cima no canvas final
  - Margens 40px, centralizado, 900x450
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
logger = logging.getLogger("ecg_training_format_v3")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 900
CANVAS_H = 450
MARGIN = 40
USABLE_W = CANVAS_W - 2 * MARGIN  # 820
USABLE_H = CANVAS_H - 2 * MARGIN  # 370
TRACE_THRESHOLD = 0.5            # core de alta confianca
REBINARIZE_THRESHOLD = 100        # apos INTER_AREA, pixels < 100 = preto puro


def make_v3(cache):
    signal_prob = cache["signal_prob"]
    px_per_mm_orig = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape

    # 1. Core da mascara (high threshold = ~1px de espessura)
    core_mask = signal_prob > TRACE_THRESHOLD

    # 2. Bounding box do core
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

    # 3. mm da bbox
    width_mm = bbox_w_px / px_per_mm_orig
    height_mm = bbox_h_px / px_per_mm_orig

    # 4. Novo px_per_mm pra caber em 820x370
    new_px_per_mm = min(USABLE_W / width_mm, USABLE_H / height_mm)
    new_w = int(round(width_mm * new_px_per_mm))
    new_h = int(round(height_mm * new_px_per_mm))

    # 5. Cria layer com tracado preto solido em full res (so dentro do bbox)
    trace_layer_full = np.full((bbox_h_px, bbox_w_px), 255, dtype=np.uint8)
    core_crop = core_mask[y_min:y_max, x_min:x_max]
    trace_layer_full[core_crop] = 0

    # 6. INTER_AREA resize do trace layer (anti-aliased, suaviza)
    trace_resized = cv2.resize(
        trace_layer_full, (new_w, new_h), interpolation=cv2.INTER_AREA,
    )

    # 7. Re-binariza: pixels escuros → preto puro
    trace_binary = trace_resized < REBINARIZE_THRESHOLD

    # 8. Canvas branco
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)

    # 9. Posicao centralizada
    offset_x = (CANVAS_W - new_w) // 2
    offset_y = (CANVAS_H - new_h) // 2

    # 10. Grid 232/245 sobre canvas inteiro, fase alinhada com offset
    p_min = new_px_per_mm
    p_maj = 5 * new_px_per_mm
    # Linhas 1mm
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
    # Linhas 5mm (mais escuras, sobrescrevem)
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

    # 11. Pinta tracado preto puro por cima
    h_eff = min(new_h, CANVAS_H - offset_y)
    w_eff = min(new_w, CANVAS_W - offset_x)
    canvas_slice = canvas[offset_y:offset_y + h_eff, offset_x:offset_x + w_eff]
    trace_slice = trace_binary[:h_eff, :w_eff]
    canvas_slice[trace_slice] = 0

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
        img, new_pxmm, new_w, new_h = make_v3(cache)
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
