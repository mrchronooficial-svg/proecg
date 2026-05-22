"""
v4: combina o que ficou perfeito no ecg_digital com formato treino.

Pipeline:
  1. Full resolution canvas branco
  2. Grid cinza 232/245 (formato treino, NAO rosa)
  3. Tracado com ALPHA GRADIENT * 1.5 boost (halo escurece, core preto puro)
  4. INTER_AREA resize pra 900x450 (anti-aliased preserva fineza)
  5. Output grayscale 8-bit
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
logger = logging.getLogger("ecg_training_format_v4")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 900
CANVAS_H = 450
TRACE_THRESHOLD = 0.05
ALPHA_BOOST = 1.5  # faz halo ficar mais escuro, core ja era 1.0


def make_v4(cache):
    signal_prob = cache["signal_prob"]
    px_per_mm_orig = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape

    # 1. Canvas branco em full res (float pra alpha)
    canvas = np.full((H, W), 255.0, dtype=np.float32)

    # 2. Grid matematico 232/245 em full res (preserva alinhamento)
    p_min = px_per_mm_orig
    p_maj = 5 * px_per_mm_orig
    # 1mm (245)
    for y in np.arange(0, H, p_min):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 245)
    for x in np.arange(0, W, p_min):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 245)
    # 5mm (232) sobrescrevem
    for y in np.arange(0, H, p_maj):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 232)
    for x in np.arange(0, W, p_maj):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 232)

    # 3. Tracado ALPHA GRADIENT com BOOST
    trace_mask = signal_prob > TRACE_THRESHOLD
    trace_intensity = np.clip(
        signal_prob / max(signal_prob.max(), 1e-6) * ALPHA_BOOST, 0, 1,
    )
    alpha_t = trace_intensity * trace_mask.astype(np.float32)
    canvas = canvas * (1.0 - alpha_t)  # preto opaco mixado

    canvas_u8 = canvas.astype(np.uint8)

    # 4. Pad pra aspect 2:1 + resize INTER_AREA pra 900x450
    target_aspect = CANVAS_W / CANVAS_H
    src_aspect = W / H
    if src_aspect < target_aspect:
        new_w = int(round(H * target_aspect))
        pad_lef = (new_w - W) // 2
        pad_rig = new_w - W - pad_lef
        canvas_u8 = cv2.copyMakeBorder(
            canvas_u8, 0, 0, pad_lef, pad_rig,
            cv2.BORDER_CONSTANT, value=255,
        )
    else:
        new_h = int(round(W / target_aspect))
        pad_top = (new_h - H) // 2
        pad_bot = new_h - H - pad_top
        canvas_u8 = cv2.copyMakeBorder(
            canvas_u8, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=255,
        )
    img = cv2.resize(canvas_u8, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    return img


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.warning("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        img = make_v4(cache)
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), img)
        logger.info("Salvo: %s (900x450 grayscale)", out_path.name)
    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
