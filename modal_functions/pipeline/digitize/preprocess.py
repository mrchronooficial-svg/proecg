"""
M0 — Pré-processamento de imagem de ECG.

Input:  PIL.Image.Image (qualquer modo/tamanho/orientação)
Output: np.ndarray float32 shape (H, W, 3), range [0, 1]
"""

import logging

import numpy as np
from PIL import Image, ImageOps

from .constants import MAX_INPUT_SIZE

logger = logging.getLogger(__name__)


def preprocess(image: Image.Image) -> np.ndarray:
    """Prepara foto do ECG para segmentação UNet.

    1. Corrige orientação EXIF (celulares salvam rotação como metadata)
    2. Converte para RGB
    3. Resize (lado maior -> MAX_INPUT_SIZE px)
    4. Normaliza para float32 [0, 1]
    """
    # 1. Corrigir orientação EXIF
    image = ImageOps.exif_transpose(image)

    # 2. Converter para RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 3. Resize mantendo aspect ratio
    w, h = image.size
    max_side = max(w, h)
    if max_side > MAX_INPUT_SIZE:
        scale = MAX_INPUT_SIZE / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.info("Resize: %dx%d -> %dx%d", w, h, new_w, new_h)

    # 4. Normalizar para float32 [0, 1]
    img_np = np.array(image, dtype=np.float32) / 255.0

    logger.info("Preprocessed: shape %s, range [%.2f, %.2f]",
                img_np.shape, img_np.min(), img_np.max())
    return img_np
