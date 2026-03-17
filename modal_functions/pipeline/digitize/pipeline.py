"""
Pipeline principal do digitalizador — orquestra M0→M6.

Tenta pipeline UNet completo; se falhar, cai para fallback clássico.
"""

import logging
import os

import cv2
import numpy as np
from PIL import Image as PILImage

from .calibration import calibrate
from .classic_fallback import digitize_classic_fallback
from .constants import LEAD_ORDER, OUTPUT_LENGTH
from .lead_detection import detect_leads
from .postprocess import postprocess
from .preprocess import preprocess
from .perspective import correct_perspective
from .segmentation import segment
from .signal_extraction import extract_signals

logger = logging.getLogger(__name__)


_DEBUG_DIR = os.environ.get("PROECG_DEBUG_DIR", "")


def _save_debug(name: str, data, is_mask: bool = False):
    """Salva imagem de debug se PROECG_DEBUG_DIR estiver definido."""
    if not _DEBUG_DIR:
        return
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    path = os.path.join(_DEBUG_DIR, name)
    if is_mask:
        img = PILImage.fromarray((data.astype(np.uint8) * 255))
    elif data.dtype == np.float32 or data.dtype == np.float64:
        img = PILImage.fromarray((np.clip(data, 0, 1) * 255).astype(np.uint8))
    else:
        img = PILImage.fromarray(data)
    img.save(path)
    logger.info("Debug salvo: %s", path)


def digitize_ecg(image: PILImage.Image) -> np.ndarray:
    """Converte foto de ECG em papel -> sinal digital de 12 derivações.

    Args:
        image: PIL Image (RGB) da foto do ECG.

    Returns:
        np.ndarray shape (12, 5000) com sinal em mV a 500 Hz.
        Ordem das derivações: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        Duração: 10 segundos (5000 amostras a 500 Hz)
        NaN substituídos por 0.0

    Raises:
        RuntimeError: se não foi possível digitalizar.
    """
    # Tentar pipeline UNet
    try:
        result = _pipeline_unet(image)
        if result is not None:
            leads_active = sum(1 for i in range(12) if np.any(result[i] != 0))
            if leads_active >= 4:
                logger.info("Pipeline UNet: sucesso — %d/12 leads ativos", leads_active)
                return result
            logger.warning("Pipeline UNet: apenas %d/12 leads — tentando fallback", leads_active)
    except Exception as e:
        logger.warning("Pipeline UNet falhou: %s — tentando fallback clássico", e)

    # Fallback clássico
    try:
        result = digitize_classic_fallback(image)
        if result is not None:
            leads_active = sum(1 for i in range(12) if np.any(result[i] != 0))
            logger.info("Fallback clássico: sucesso — %d/12 leads", leads_active)
            return result
    except Exception as e:
        logger.error("Fallback clássico também falhou: %s", e)

    raise RuntimeError(
        "Não foi possível digitalizar o ECG. "
        "Verifique a qualidade da foto e tente novamente."
    )


def _dilate_mask(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Dilata mask booleana."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)


def _pipeline_unet(image: PILImage.Image) -> np.ndarray:
    """Pipeline completo UNet: M0 → M1 → M2 → M3 → M4 → M5 → M6."""

    # M0: Pré-processamento
    img_np = preprocess(image)

    # M1: Segmentação UNet
    masks = segment(img_np)
    if masks is None:
        logger.warning("UNet não disponível (pesos não encontrados)")
        return None

    # Verificar que trace tem pixels
    trace_px = masks["trace"].sum()
    if trace_px < 50:
        logger.warning("Segmentação: trace mask quase vazia (%d px)", trace_px)
        return None

    _save_debug("debug_01_mask_grid.png", masks["grid"], is_mask=True)
    _save_debug("debug_02_mask_trace.png", masks["trace"], is_mask=True)

    # M2: Correção de perspectiva
    img_np, masks = correct_perspective(img_np, masks)

    # M3: Calibração
    cal = calibrate(masks, img_np)
    logger.info("Calibração: px/mm_h=%.2f, px/mm_v=%.2f, speed=%d, gain=%d (%s, %s)",
                cal.px_per_mm_h, cal.px_per_mm_v, cal.paper_speed, cal.gain,
                cal.method, cal.confidence)

    # M4: Identificação de leads (usa grid+trace bbox + template layout)
    lead_regions = detect_leads(masks, cal)
    if not lead_regions:
        logger.warning("Nenhuma região de lead detectada")
        return None

    # Debug: salvar leads detectados sobre a imagem
    if _DEBUG_DIR:
        _save_leads_debug(img_np, masks["trace"], lead_regions)

    # M5: Extração de sinal (subtrai grid, dilata, spline, Savitzky-Golay)
    signals = extract_signals(masks["trace"], lead_regions, cal,
                               grid_mask=masks.get("grid"))
    if not signals:
        logger.warning("Nenhum sinal extraído")
        return None

    # Debug: salvar sinais extraídos
    if _DEBUG_DIR:
        _save_signals_debug(signals)

    # M6: Pós-processamento (filtros + montagem final)
    result = postprocess(signals)

    return result


def _save_leads_debug(img_np, trace_mask, lead_regions):
    """Salva imagem com bboxes dos leads detectados."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    H, W = trace_mask.shape
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img_np)

    colors = plt.cm.Set3(np.linspace(0, 1, 13))
    for i, region in enumerate(lead_regions):
        y1, x1, y2, x2 = region.bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor=colors[i],
                                  facecolor="none")
        ax.add_patch(rect)
        ax.text(x1 + 3, y1 + 15, region.name, color=colors[i],
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    ax.set_title(f"Leads detectados: {len(lead_regions)}")
    fig.tight_layout()
    fig.savefig(os.path.join(_DEBUG_DIR, "debug_04_leads_detected.png"), dpi=150)
    plt.close(fig)


def _save_signals_debug(signals):
    """Salva plot dos sinais extraídos (pré-postprocess)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lead_order = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=False)
    for i, (ax, name) in enumerate(zip(axes, lead_order)):
        if name in signals:
            sig = signals[name]
            ax.plot(sig, color="black", linewidth=0.5)
            ax.set_ylabel(name, rotation=0, labelpad=25, fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{name} — não extraído", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=9)
            ax.set_ylabel(name, rotation=0, labelpad=25, fontsize=8, color="red")
        ax.tick_params(labelsize=6)

    fig.suptitle(f"Sinais extraídos: {len(signals)}/12", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_DEBUG_DIR, "debug_05_signals_extracted.png"), dpi=150)
    plt.close(fig)
