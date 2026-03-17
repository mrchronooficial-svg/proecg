"""
M5 — Extração de Sinal.

Para cada lead: mask trace na bbox → centroide por coluna → array Y em mV.
Inclui: subtração de grid, dilatação, janela de vizinhos, spline, Savitzky-Golay.

Input:  mask de trace dewarped, lista de LeadRegion, CalibrationResult
Output: dict {lead_name: np.ndarray em mV}
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt, resample, savgol_filter

from .calibration import CalibrationResult
from .constants import SAMPLING_RATE
from .lead_detection import LeadRegion

logger = logging.getLogger(__name__)

# NaN tolerance: descartar lead somente se >80% NaN
_MAX_GAP_FRAC = 0.80

# Janela de vizinhos para colunas sem trace
_NEIGHBOR_WINDOW = 3

# Dilatação da mask antes da extração
_DILATE_KERNEL_SIZE = 5
_DILATE_ITERATIONS = 3


def _clean_trace_mask(trace_mask: np.ndarray, grid_mask: Optional[np.ndarray]) -> np.ndarray:
    """Remove pixels de grid que foram classificados como trace e dilata.

    1. Subtrair grid da trace (remover pixels sobrepostos)
    2. Dilatar para preencher micro-gaps
    """
    cleaned = trace_mask.copy()

    # Subtrair grid: pixels que são tanto trace quanto grid provavelmente são grid
    if grid_mask is not None:
        overlap = trace_mask & grid_mask
        if overlap.sum() > 0:
            cleaned = trace_mask & ~grid_mask
            logger.info("Removidos %d pixels grid-trace sobrepostos", overlap.sum())

    # Dilatar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (_DILATE_KERNEL_SIZE, _DILATE_KERNEL_SIZE))
    dilated = cv2.dilate(cleaned.astype(np.uint8), kernel, iterations=_DILATE_ITERATIONS)

    result = dilated.astype(bool)
    logger.info("Trace: original=%d, limpo=%d, dilatado=%d px",
                trace_mask.sum(), cleaned.sum(), result.sum())
    return result


def _extract_trace_from_roi(trace_roi: np.ndarray) -> np.ndarray:
    """Extrai sinal 1D de uma ROI da mask de trace via centroide vertical.

    Usa janela de vizinhos quando uma coluna não tem pixels.
    """
    H, W = trace_roi.shape
    signal = np.full(W, np.nan, dtype=np.float64)

    for x in range(W):
        col = trace_roi[:, x]
        trace_pixels = np.where(col)[0]

        if len(trace_pixels) > 0:
            center = float(np.mean(trace_pixels))
            signal[x] = H - center
            continue

        # Sem pixels — olhar vizinhos
        neighbors_y = []
        for dx in range(1, _NEIGHBOR_WINDOW + 1):
            for nx in [x - dx, x + dx]:
                if 0 <= nx < W:
                    ncol = trace_roi[:, nx]
                    npx = np.where(ncol)[0]
                    if len(npx) > 0:
                        neighbors_y.append(float(np.mean(npx)))

        if neighbors_y:
            center = np.mean(neighbors_y)
            signal[x] = H - center

    return signal


def _interpolate_gaps(signal: np.ndarray) -> Optional[np.ndarray]:
    """Interpola gaps NaN: cúbica no interior, constante nas bordas.

    NUNCA extrapola — bordas são preenchidas com o valor válido mais próximo.
    Usa spline cúbica entre pontos válidos para suavidade.
    """
    valid = ~np.isnan(signal)
    n_valid = valid.sum()
    n_total = len(signal)

    if n_valid < 4:
        return None

    nan_frac = 1 - n_valid / n_total
    if nan_frac > _MAX_GAP_FRAC:
        logger.warning("Lead com %.0f%% NaN (>%.0f%% limite) — descartado",
                       nan_frac * 100, _MAX_GAP_FRAC * 100)
        return None

    if n_valid == n_total:
        return signal

    x_valid = np.where(valid)[0]
    y_valid = signal[x_valid]

    # Interpolar apenas ENTRE o primeiro e último ponto válido (sem extrapolar)
    first_valid = x_valid[0]
    last_valid = x_valid[-1]

    kind = "cubic" if n_valid >= 4 else "linear"
    f = interp1d(
        x_valid, y_valid,
        kind=kind,
        bounds_error=False,
        fill_value=(y_valid[0], y_valid[-1]),  # bordas = valor mais próximo
    )

    result = signal.copy()
    # Preencher tudo com interpolação
    all_x = np.arange(n_total)
    result = f(all_x)

    # Bordas: constante (repetir primeiro/último valor válido)
    result[:first_valid] = y_valid[0]
    result[last_valid + 1:] = y_valid[-1]

    return result


def _remove_outliers(signal: np.ndarray, threshold_factor: float = 3.0) -> np.ndarray:
    """Remove outliers (pontos > threshold_factor × IQR da mediana).

    Substitui outliers pelo valor mediano local.
    """
    n = len(signal)
    if n < 20:
        return signal

    median = np.median(signal)
    q25, q75 = np.percentile(signal, [25, 75])
    iqr = q75 - q25
    if iqr < 1e-6:
        return signal

    lower = median - threshold_factor * iqr
    upper = median + threshold_factor * iqr

    outliers = (signal < lower) | (signal > upper)
    n_outliers = outliers.sum()

    if n_outliers > 0 and n_outliers < n * 0.3:
        # Substituir outliers com mediana local
        result = signal.copy()
        for i in np.where(outliers)[0]:
            # Janela de 20 vizinhos
            start = max(0, i - 10)
            end = min(n, i + 10)
            local = signal[start:end]
            local_valid = local[(local >= lower) & (local <= upper)]
            if len(local_valid) > 0:
                result[i] = np.median(local_valid)
            else:
                result[i] = median
        logger.debug("Removidos %d outliers de %d pontos", n_outliers, n)
        return result

    return signal


def _smooth_signal(signal: np.ndarray) -> np.ndarray:
    """Suaviza sinal com median filter + Savitzky-Golay."""
    n = len(signal)
    if n < 10:
        return signal

    # 1. Median filter para remover spikes isolados
    if n > 5:
        signal = medfilt(signal, kernel_size=5)

    # 2. Savitzky-Golay para suavizar sem perder QRS
    # window = ~2% do sinal, mínimo 7, sempre ímpar
    window = max(7, min(n // 50, 31))
    if window % 2 == 0:
        window += 1
    if window < n:
        signal = savgol_filter(signal, window_length=window, polyorder=3)

    return signal


def _convert_px_to_mv(signal_px: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    """Converte sinal de pixels para mV."""
    median_val = np.nanmedian(signal_px)
    if not np.isnan(median_val):
        signal_px = signal_px - median_val

    mm_per_px = 1.0 / calibration.px_per_mm_v if calibration.px_per_mm_v > 0 else 1.0
    mv_per_mm = 1.0 / calibration.gain if calibration.gain > 0 else 0.1
    return signal_px * mm_per_px * mv_per_mm


def _resample_to_target(signal: np.ndarray, calibration: CalibrationResult,
                         target_sr: int = SAMPLING_RATE) -> np.ndarray:
    """Resamplea sinal para a frequência alvo (500 Hz)."""
    native_sr = calibration.px_per_mm_h * calibration.paper_speed
    if native_sr <= 0:
        native_sr = 75.0

    n = len(signal)
    duration_sec = n / native_sr
    if duration_sec <= 0:
        return signal

    n_target = int(duration_sec * target_sr)
    if n_target < 2:
        n_target = 2
    if n_target == n:
        return signal

    return resample(signal, n_target)


def extract_signals(
    trace_mask: np.ndarray,
    lead_regions: List[LeadRegion],
    calibration: CalibrationResult,
    grid_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Extrai sinais de todas as derivações.

    Pipeline por lead:
      1. Limpar trace (subtrair grid + dilatar)
      2. Extrair centroide por coluna (com vizinhos)
      3. Interpolar gaps NaN com spline cúbica
      4. Suavizar (median + Savitzky-Golay)
      5. Converter px → mV
      6. Resamplear para 500 Hz
    """
    # Limpar e dilatar trace mask (compartilhada entre todos os leads)
    trace_clean = _clean_trace_mask(trace_mask, grid_mask)

    signals = {}

    for region in lead_regions:
        name = region.name
        y1, x1, y2, x2 = region.bbox

        if y2 <= y1 or x2 <= x1:
            logger.warning("Lead %s: bbox inválida (%d,%d,%d,%d)", name, y1, x1, y2, x2)
            continue

        # Recortar ROI
        roi = trace_clean[y1:y2, x1:x2]
        if roi.size == 0 or roi.sum() == 0:
            logger.warning("Lead %s: ROI sem trace", name)
            continue

        # Extrair trace por centroide (com vizinhos)
        signal_px = _extract_trace_from_roi(roi)

        # Interpolar gaps com spline cúbica
        signal_interp = _interpolate_gaps(signal_px)
        if signal_interp is None:
            logger.warning("Lead %s: muitos gaps — descartado", name)
            continue

        # Remover outliers (pixels espúrios fora do papel)
        signal_interp = _remove_outliers(signal_interp)

        # Suavizar (median + Savitzky-Golay)
        signal_smooth = _smooth_signal(signal_interp)

        # Converter px → mV
        signal_mv = _convert_px_to_mv(signal_smooth, calibration)

        # Resamplear para 500 Hz
        signal_resampled = _resample_to_target(signal_mv, calibration)

        signals[name] = signal_resampled

    logger.info("Extraídos %d/%d leads", len(signals), len(lead_regions))
    return signals
