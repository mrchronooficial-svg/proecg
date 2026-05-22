"""
Gera ECG digital com:
  - Tracado EXATO da mascara UNet canal 2 (preto solido, sem skeletonize/alpha)
  - Grid matematico cores treino (232/245)
  - Fundo branco puro
  - Sem labels
  - 900x450 grayscale

Faz pra 3 ECGs: IMG_1407, IMG_1316, 0a8c7db0-...
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecg_training_format")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 900
CANVAS_H = 450
TRACE_THRESHOLD = 0.05


def make_training_format(cache):
    """Tracado: alpha gradient pela prob da UNet (MESMO metodo da img2 fina)
    + grid matematico 232/245 + INTER_AREA anti-aliased resize pra 900x450.

    Pipeline:
      1. Canvas branco FULL RES
      2. Grid matematico em full res (cores 232/245)
      3. Tracado em full res com ALPHA GRADIENT pela prob
      4. Grayscale + INTER_AREA resize pra 900x450 (anti-aliased)
    """
    signal_prob = cache["signal_prob"]
    px_per_mm = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape

    # 1. Canvas branco em full res (grayscale)
    canvas = np.full((H, W), 255, dtype=np.float32)

    # 2. Grid matematico 232/245 em full res (np.minimum nao sobrescreve)
    p_min = px_per_mm
    p_maj = 5 * px_per_mm
    for y in np.arange(0, H, p_min):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 245)
    for x in np.arange(0, W, p_min):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 245)
    for y in np.arange(0, H, p_maj):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 232)
    for x in np.arange(0, W, p_maj):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 232)

    # 3. Tracado ALPHA GRADIENT (mesmo metodo da ecg_digital.png img2)
    trace_mask = signal_prob > TRACE_THRESHOLD
    trace_intensity = np.clip(
        signal_prob / max(signal_prob.max(), 1e-6), 0, 1,
    )
    alpha_t = trace_intensity * trace_mask.astype(np.float32)
    # canvas = canvas * (1 - alpha) + 0 * alpha  (preto opaco mixado)
    canvas = canvas * (1.0 - alpha_t)
    canvas_full = canvas.astype(np.uint8)

    # 4. Anti-aliased resize pra 900x450 (preserva detalhe sem "escadas")
    target_w, target_h = CANVAS_W, CANVAS_H
    target_aspect = target_w / target_h
    src_aspect = W / H
    if src_aspect < target_aspect:
        new_w = int(round(H * target_aspect))
        pad_lef = (new_w - W) // 2
        pad_rig = new_w - W - pad_lef
        canvas_full = cv2.copyMakeBorder(
            canvas_full, 0, 0, pad_lef, pad_rig,
            cv2.BORDER_CONSTANT, value=255,
        )
    else:
        new_h = int(round(W / target_aspect))
        pad_top = (new_h - H) // 2
        pad_bot = new_h - H - pad_top
        canvas_full = cv2.copyMakeBorder(
            canvas_full, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=255,
        )
    img = cv2.resize(canvas_full, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return img, int(trace_mask.sum()), int((trace_intensity > 0.5).sum())


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.warning("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        img, n_orig, n_canvas = make_training_format(cache)
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), img)
        logger.info(
            "Salvo: %s (%dx%d grayscale, %d -> %d pixels apos resize)",
            out_path.name, CANVAS_W, CANVAS_H, n_orig, n_canvas,
        )

    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
