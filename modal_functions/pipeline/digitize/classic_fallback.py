"""
Fallback clássico — binarização + morfologia + varredura vertical.

Preservado do digitize.py original como último recurso quando o
pipeline UNet falha completamente.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from .constants import LEAD_ORDER, OUTPUT_LENGTH, SAMPLING_RATE

logger = logging.getLogger(__name__)

_SHORT_SIGNAL_SEC = 2.5


def digitize_classic_fallback(image: Image.Image) -> Optional[np.ndarray]:
    """Abordagem clássica: binarização -> remoção de grid -> segmentação -> varredura.

    Args:
        image: PIL Image (RGB)

    Returns:
        np.ndarray (12, 5000) em mV aproximado, ou None se falhar.
    """
    img_np = np.array(image)

    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    h, w = gray.shape

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10,
    )

    binary = _remove_grid(binary)

    lead_layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    margin_top = int(h * 0.12)
    margin_bottom = int(h * 0.05)
    usable_h = h - margin_top - margin_bottom

    margin_left = int(w * 0.05)
    margin_right = int(w * 0.05)
    usable_w = w - margin_left - margin_right

    row_height = usable_h // 3
    col_width = usable_w // 4

    signals = {}
    for row_idx, row_leads in enumerate(lead_layout):
        for col_idx, lead_name in enumerate(row_leads):
            y1 = margin_top + row_idx * row_height
            y2 = y1 + row_height
            x1 = margin_left + col_idx * col_width
            x2 = x1 + col_width

            roi = binary[y1:y2, x1:x2]
            sig = _extract_signal_from_roi(roi)
            signals[lead_name] = sig

    target_per_lead = int(_SHORT_SIGNAL_SEC * SAMPLING_RATE)  # 1250

    result = np.zeros((12, OUTPUT_LENGTH), dtype=np.float64)
    for i, lead_name in enumerate(LEAD_ORDER):
        if lead_name in signals:
            sig = signals[lead_name]
            resampled = _resample_signal(sig, target_per_lead)
            result[i, :target_per_lead] = resampled
            result[i, target_per_lead:] = resampled[-1] if len(resampled) > 0 else 0.0

    result = _normalize_to_mv(result)

    leads_found = sum(1 for i in range(12) if np.any(result[i] != 0))
    if leads_found == 0:
        return None

    logger.info("Fallback clássico: %d/12 derivações extraídas", leads_found)
    return result


def _remove_grid(binary: np.ndarray) -> np.ndarray:
    """Remove linhas de grid do ECG usando operações morfológicas."""
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    binary = cv2.subtract(binary, horizontal_lines)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    binary = cv2.subtract(binary, vertical_lines)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)

    return binary


def _extract_signal_from_roi(roi: np.ndarray) -> np.ndarray:
    """Extrai sinal 1D de uma ROI binarizada por varredura vertical."""
    h, w = roi.shape
    signal = np.full(w, np.nan, dtype=np.float64)

    for x in range(w):
        col = roi[:, x]
        white_pixels = np.where(col > 0)[0]
        if len(white_pixels) > 0:
            weights = col[white_pixels].astype(np.float64)
            center = np.average(white_pixels, weights=weights)
            signal[x] = h - center

    valid = ~np.isnan(signal)
    if valid.sum() < 2:
        return np.zeros(w, dtype=np.float64)

    x_valid = np.where(valid)[0]
    f = interp1d(
        x_valid, signal[x_valid],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    signal = f(np.arange(w))

    if len(signal) > 5:
        signal = medfilt(signal, kernel_size=5)

    return signal


def _resample_signal(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Resample sinal 1D para target_len amostras via interpolação linear."""
    if len(signal) == target_len:
        return signal
    if len(signal) < 2:
        return np.zeros(target_len, dtype=np.float64)

    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, signal, kind="linear")
    return f(x_new)


def _normalize_to_mv(signal: np.ndarray) -> np.ndarray:
    """Normaliza sinal de pixels para escala aproximada de mV."""
    for i in range(signal.shape[0]):
        lead = signal[i]
        lead = lead - np.median(lead)
        peak = np.percentile(np.abs(lead), 99)
        if peak > 0:
            lead = lead * (1.5 / peak)
        signal[i] = lead

    return signal
