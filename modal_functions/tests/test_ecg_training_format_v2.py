"""
Producao ajustada pra match EXATO o formato de treino:

Specs:
  - Canvas 900x450 grayscale 8-bit
  - Fundo 255 puro
  - Grid 1mm = 245 (cinza claro), grid 5mm = 232 (cinza)
  - px_per_mm = min(820/width_mm, 370/height_mm)
  - Tracado cor 0 PRETO PURO
  - Espessura 1px (skeleton + preenchimento de gaps verticais)
  - Margem 40px todos os lados
  - Area util 820x370
  - ECG centralizado
  - Sem labels, sem separadores, sem pulso de calibracao
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecg_training_format_v2")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 900
CANVAS_H = 450
MARGIN = 40
USABLE_W = CANVAS_W - 2 * MARGIN  # 820
USABLE_H = CANVAS_H - 2 * MARGIN  # 370
TRACE_THRESHOLD = 0.05


def make_training_format_v2(cache):
    signal_prob = cache["signal_prob"]
    px_per_mm_orig = float(cache["pixel_size"]["avg_pixel_per_mm"])
    H, W = signal_prob.shape

    # 1. Binarizar + skeletonize (1px de espessura)
    binary = (signal_prob > TRACE_THRESHOLD).astype(np.uint8)
    skel = skeletonize(binary > 0)

    # 2. Bounding box do esqueleto em pixels
    rows_with = np.any(skel, axis=1)
    cols_with = np.any(skel, axis=0)
    if not rows_with.any() or not cols_with.any():
        return np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    y_min = int(np.argmax(rows_with))
    y_max = H - int(np.argmax(rows_with[::-1]))
    x_min = int(np.argmax(cols_with))
    x_max = W - int(np.argmax(cols_with[::-1]))
    bbox_w_px = x_max - x_min
    bbox_h_px = y_max - y_min

    # 3. Converter bbox pra mm usando px_per_mm da imagem original
    width_mm = bbox_w_px / px_per_mm_orig
    height_mm = bbox_h_px / px_per_mm_orig

    # 4. Calcular novo px_per_mm pra caber em 820x370
    new_px_per_mm = min(USABLE_W / width_mm, USABLE_H / height_mm)
    new_w = int(round(width_mm * new_px_per_mm))
    new_h = int(round(height_mm * new_px_per_mm))

    # 5. Crop esqueleto pro bbox e redimensiona pro (new_w, new_h)
    skel_crop = skel[y_min:y_max, x_min:x_max].astype(np.uint8) * 255
    skel_pil = Image.fromarray(skel_crop)
    skel_resized_pil = skel_pil.resize((new_w, new_h), Image.NEAREST)
    skel_resized = np.array(skel_resized_pil) > 128

    # 6. Canvas branco 900x450
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)

    # 7. Offsets pra centralizar na area util
    offset_x = (CANVAS_W - new_w) // 2
    offset_y = (CANVAS_H - new_h) // 2

    # 8. Grid matematico (1mm = 245, 5mm = 232) cobrindo CANVAS INTEIRO
    p_min = new_px_per_mm
    p_maj = 5 * new_px_per_mm
    # Linhas 1mm (245) — primeiro pra que 5mm sobrescreva nos pontos comuns
    y = offset_y % p_min  # fase do grid alinhada ao offset
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
    # Linhas 5mm (232) — sobrescrevem 1mm onde coincidem
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

    # 9. Coloca esqueleto no canvas (pinta PRETO PURO = 0)
    skel_full = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    y2 = offset_y + new_h
    x2 = offset_x + new_w
    if y2 > CANVAS_H or x2 > CANVAS_W:
        # Clipar caso saia (nao deveria com a logica acima)
        h_eff = min(new_h, CANVAS_H - offset_y)
        w_eff = min(new_w, CANVAS_W - offset_x)
        skel_full[offset_y:offset_y + h_eff, offset_x:offset_x + w_eff] = skel_resized[:h_eff, :w_eff]
    else:
        skel_full[offset_y:y2, offset_x:x2] = skel_resized
    canvas[skel_full] = 0

    # 10. Preenche gaps verticais: em cada coluna, conecta pixels consecutivos
    #     do esqueleto que tem diff Y > 1
    for col in range(CANVAS_W):
        ys = np.where(skel_full[:, col])[0]
        if ys.size >= 2:
            for i in range(ys.size - 1):
                y1, y2 = int(ys[i]), int(ys[i + 1])
                if y2 - y1 > 1:
                    canvas[y1:y2 + 1, col] = 0

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
        img, new_pxmm, new_w, new_h = make_training_format_v2(cache)
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
