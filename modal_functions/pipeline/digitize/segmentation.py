"""
M1 — Segmentação UNet.

Input:  np.ndarray float32 (H, W, 3), range [0, 1]
Output: dict com 4 masks booleanas shape (H, W):
        "background", "grid", "trace", "text"
        Retorna None se pesos não disponíveis.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch

from .constants import (
    SEG_BACKGROUND,
    SEG_GRID,
    SEG_TEXT,
    SEG_TRACE,
    UNET_ENCODER_WIDTHS,
    UNET_IN_CHANNELS,
    UNET_NUM_CLASSES,
    UNET_OVERLAP,
    UNET_PATCH_SIZE,
    UNET_WEIGHTS_FILENAME,
)
from .unet import ResidualUNet, load_weights

logger = logging.getLogger(__name__)

# Cache do modelo (carrega uma vez por container Modal)
_model_cache: Optional[ResidualUNet] = None


def _resolve_weights_path() -> str:
    """Resolve caminho dos pesos a partir da raiz de modal_functions/."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "models", "digitizer", UNET_WEIGHTS_FILENAME)


def _get_device() -> str:
    """Retorna 'cuda' se disponível, senão 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model() -> Optional[ResidualUNet]:
    """Carrega o modelo UNet (com cache)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    weights_path = _resolve_weights_path()
    if not os.path.exists(weights_path):
        logger.warning("Pesos UNet não encontrados: %s", weights_path)
        return None

    device = _get_device()
    model = ResidualUNet(
        in_channels=UNET_IN_CHANNELS,
        num_classes=UNET_NUM_CLASSES,
        encoder_widths=UNET_ENCODER_WIDTHS,
    )
    model = load_weights(model, weights_path, device=device)
    model = model.to(device)
    _model_cache = model
    logger.info("UNet carregado em %s", device)
    return model


def _run_single_pass(model: ResidualUNet, tensor: torch.Tensor) -> np.ndarray:
    """Forward pass direto para imagens <= patch_size."""
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        return output.detach().cpu().numpy()


def _run_sliding_window(model: ResidualUNet, tensor: torch.Tensor) -> np.ndarray:
    """Sliding window com overlap para imagens grandes.

    Acumula logits nos patches e faz média nas regiões de overlap.
    """
    device = next(model.parameters()).device
    _, C, H, W = tensor.shape
    patch = UNET_PATCH_SIZE
    overlap = UNET_OVERLAP
    stride = patch - overlap

    # Acumuladores
    logits_sum = np.zeros((1, UNET_NUM_CLASSES, H, W), dtype=np.float32)
    count = np.zeros((1, 1, H, W), dtype=np.float32)

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + patch, H)
            x1 = min(x0 + patch, W)
            # Ajustar início se o patch ultrapassa a borda
            y0_adj = max(0, y1 - patch)
            x0_adj = max(0, x1 - patch)

            patch_tensor = tensor[:, :, y0_adj:y1, x0_adj:x1].to(device)

            with torch.no_grad():
                out = model(patch_tensor)
                out_np = out.detach().cpu().numpy()

            logits_sum[:, :, y0_adj:y1, x0_adj:x1] += out_np
            count[:, :, y0_adj:y1, x0_adj:x1] += 1.0

    # Média nas regiões de overlap
    count = np.maximum(count, 1.0)
    return logits_sum / count


def segment(image: np.ndarray) -> Optional[dict]:
    """Segmenta imagem de ECG em 4 classes via UNet.

    Args:
        image: np.ndarray float32 (H, W, 3), range [0, 1]

    Returns:
        dict com masks booleanas (H, W): "background", "grid", "trace", "text"
        None se modelo não disponível.
    """
    model = _load_model()
    if model is None:
        return None

    H, W, _ = image.shape

    # Converter para tensor: (1, 3, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    # Escolher single pass ou sliding window
    if H <= UNET_PATCH_SIZE and W <= UNET_PATCH_SIZE:
        logits = _run_single_pass(model, tensor)
    else:
        logits = _run_sliding_window(model, tensor)
        logger.info("Sliding window: imagem %dx%d processada em patches", H, W)

    # Argmax -> class_map (H, W)
    class_map = np.argmax(logits[0], axis=0)  # (H, W)

    masks = {
        "background": class_map == SEG_BACKGROUND,
        "grid": class_map == SEG_GRID,
        "trace": class_map == SEG_TRACE,
        "text": class_map == SEG_TEXT,
    }

    # Log estatísticas
    total_px = H * W
    for name, mask in masks.items():
        pct = mask.sum() / total_px * 100
        logger.info("Mask '%s': %.1f%% (%d px)", name, pct, mask.sum())

    return masks
