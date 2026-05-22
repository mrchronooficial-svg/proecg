"""
Reconstroi ECG de producao no MESMO formato visual da imagem de treino:
  - Esqueletonizar mascara UNet canal 2 (afina pra 1 px)
  - Grid matematico (mesmos tons 232/245 do treino)
  - Canvas 900x450 grayscale
  - Fundo branco puro 255

Gera lado-a-lado: PRODUCAO ANTIGA vs PRODUCAO NOVA.
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("producao_ajustada")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "producao_ajustada"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 900
CANVAS_H = 450
TRACE_THRESHOLD = 0.05


def reconstruct_production_ecg(
    unet_channel2: np.ndarray,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> Image.Image:
    """Reconstroi imagem de ECG identica ao formato de treino.

    Passos:
      1. Binarizar mascara
      2. Esqueletonizar (1 px de espessura)
      3. Redimensionar pro canvas (NEAREST pra preservar 1 px)
      4. Desenhar grid matematico (mesmas cores 232/245 do treino)
      5. Pintar esqueleto preto
      6. Fechar gaps verticais pequenos
    """
    # 1. Binarizar
    if unet_channel2.max() <= 1.0:
        mask = (unet_channel2 > TRACE_THRESHOLD).astype(np.uint8)
    else:
        mask = (unet_channel2 > 0).astype(np.uint8)

    # 2. Esqueletonizar (afina pra 1 px)
    skel = skeletonize(mask > 0).astype(np.uint8)

    # 3. Redimensionar esqueleto preservando 1 px (NEAREST)
    skel_pil = Image.fromarray(skel * 255)
    skel_resized = skel_pil.resize((canvas_w, canvas_h), Image.NEAREST)
    skel_array = np.array(skel_resized) > 128

    # 4. Canvas branco puro
    img_array = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    # 5. Bounding box do tracado
    rows_with = np.any(skel_array, axis=1)
    cols_with = np.any(skel_array, axis=0)
    if rows_with.sum() > 0 and cols_with.sum() > 0:
        y_min = int(np.argmax(rows_with))
        y_max = canvas_h - int(np.argmax(rows_with[::-1]))
        x_min = int(np.argmax(cols_with))
        x_max = canvas_w - int(np.argmax(cols_with[::-1]))
        ecg_w = x_max - x_min
        ecg_h = y_max - y_min
        offset_x = x_min
        offset_y = y_min
        # ECG padrao: 10s × 25mm/s = 250mm de largura
        px_per_mm = ecg_w / 250.0

        # Grid vertical (linhas a cada 1mm)
        x = offset_x
        mm_count = 0
        while x <= offset_x + ecg_w:
            xi = min(int(x), canvas_w - 1)
            color = 232 if mm_count % 5 == 0 else 245
            img_array[offset_y:offset_y + ecg_h, xi] = np.minimum(
                img_array[offset_y:offset_y + ecg_h, xi], color,
            )
            x += px_per_mm
            mm_count += 1

        # Grid horizontal (linhas a cada 1mm)
        y = offset_y
        mm_count = 0
        while y <= offset_y + ecg_h:
            yi = min(int(y), canvas_h - 1)
            color = 232 if mm_count % 5 == 0 else 245
            img_array[yi, offset_x:offset_x + ecg_w] = np.minimum(
                img_array[yi, offset_x:offset_x + ecg_w], color,
            )
            y += px_per_mm
            mm_count += 1

    # 6. Pintar esqueleto preto
    img_array[skel_array] = 0

    # 7. Fechar gaps verticais pequenos
    for col in range(canvas_w):
        col_pixels = np.where(skel_array[:, col])[0]
        if len(col_pixels) >= 2:
            for k in range(len(col_pixels) - 1):
                y1, y2 = col_pixels[k], col_pixels[k + 1]
                if y2 - y1 <= 3:
                    img_array[y1:y2 + 1, col] = 0

    return Image.fromarray(img_array, mode="L")


def reconstruct_old_production_ecg(
    signal_prob: np.ndarray,
    grid_prob: np.ndarray,
    px_per_mm: float,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> Image.Image:
    """Versao ANTIGA da producao — mascara grossa + grid sintetico nao
    matematico (usado pra comparacao A/B)."""
    H, W = signal_prob.shape
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    # Grid claro (sem alinhamento perfeito)
    period = px_per_mm
    for y in np.arange(0, H, period):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = (230, 230, 230)
    for x in np.arange(0, W, period):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = (230, 230, 230)
    period_maj = 5 * px_per_mm
    for y in np.arange(0, H, period_maj):
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = (180, 180, 180)
    for x in np.arange(0, W, period_maj):
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = (180, 180, 180)
    # Tracado preto sem afinar
    trace_mask = signal_prob > TRACE_THRESHOLD
    canvas[trace_mask] = (0, 0, 0)
    # Grayscale + resize pra mesma dimensao
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    target_aspect = canvas_w / canvas_h
    h, w = gray.shape
    src_aspect = w / h
    if src_aspect < target_aspect:
        new_w = int(round(h * target_aspect))
        pad_lef = (new_w - w) // 2
        pad_rig = new_w - w - pad_lef
        gray = cv2.copyMakeBorder(gray, 0, 0, pad_lef, pad_rig,
                                   cv2.BORDER_CONSTANT, value=255)
    else:
        new_h = int(round(w / target_aspect))
        pad_top = (new_h - h) // 2
        pad_bot = new_h - h - pad_top
        gray = cv2.copyMakeBorder(gray, pad_top, pad_bot, 0, 0,
                                   cv2.BORDER_CONSTANT, value=255)
    gray = cv2.resize(gray, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(gray, mode="L")


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.error("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        signal_prob = cache["signal_prob"]
        grid_prob = cache["grid_prob"]
        px_per_mm = float(cache["pixel_size"]["avg_pixel_per_mm"])

        # NOVA producao (esqueletonizada + grid matematico)
        new_img = reconstruct_production_ecg(signal_prob)
        new_path = OUTPUT_DIR / f"{stem}_NOVA.png"
        new_img.save(str(new_path))
        logger.info("Salvo: %s (%dx%d)", new_path.name, *new_img.size)

        # ANTIGA producao (pra comparacao)
        old_img = reconstruct_old_production_ecg(signal_prob, grid_prob, px_per_mm)
        old_path = OUTPUT_DIR / f"{stem}_ANTIGA.png"
        old_img.save(str(old_path))
        logger.info("Salvo: %s (%dx%d)", old_path.name, *old_img.size)

        # Lado-a-lado
        fig, axes = plt.subplots(2, 1, figsize=(14, 11), dpi=120)
        axes[0].imshow(np.array(old_img), cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"{stem} — PRODUCAO ANTIGA (mascara grossa + grid irregular)",
                           fontsize=12, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(np.array(new_img), cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"{stem} — PRODUCAO NOVA (skeleton 1px + grid matematico)",
                           fontsize=12, fontweight="bold")
        axes[1].axis("off")
        fig.tight_layout()
        comp_path = OUTPUT_DIR / f"{stem}_comparacao.png"
        fig.savefig(str(comp_path), bbox_inches="tight", dpi=120, facecolor="white")
        plt.close(fig)
        logger.info("Comparacao: %s", comp_path.name)

    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
