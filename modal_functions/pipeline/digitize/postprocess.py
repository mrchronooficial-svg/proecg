"""
M6 — Pós-processamento.

Filtra, valida e monta o array final (12, 5000).

Input:  dict {lead_name: np.ndarray em mV a 500 Hz}
Output: np.ndarray shape (12, 5000) em mV, dtype float64
"""

import logging
from typing import Dict

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .constants import (
    BASELINE_HIGHPASS,
    LEAD_ORDER,
    LOWPASS_FREQ,
    OUTPUT_LENGTH,
    POWERLINE_FREQ,
    SAMPLING_RATE,
)

logger = logging.getLogger(__name__)

# Amplitude máxima plausível em mV (acima disso é artefato)
_MAX_AMPLITUDE_MV = 5.0


def _highpass_filter(signal: np.ndarray, cutoff: float, fs: int, order: int = 2) -> np.ndarray:
    """Filtro high-pass Butterworth para remover baseline wander."""
    nyq = fs / 2.0
    if cutoff >= nyq:
        return signal
    b, a = butter(order, cutoff / nyq, btype="high")
    # padlen precisa ser menor que o sinal
    padlen = min(3 * max(len(b), len(a)), len(signal) - 1)
    if padlen < 1:
        return signal
    return filtfilt(b, a, signal, padlen=padlen)


def _notch_filter(signal: np.ndarray, freq: float, fs: int, Q: float = 30.0) -> np.ndarray:
    """Filtro notch IIR para remover ruído da rede elétrica."""
    nyq = fs / 2.0
    if freq >= nyq:
        return signal
    b, a = iirnotch(freq, Q, fs)
    padlen = min(3 * max(len(b), len(a)), len(signal) - 1)
    if padlen < 1:
        return signal
    return filtfilt(b, a, signal, padlen=padlen)


def _lowpass_filter(signal: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """Filtro low-pass Butterworth anti-aliasing."""
    nyq = fs / 2.0
    if cutoff >= nyq:
        return signal
    b, a = butter(order, cutoff / nyq, btype="low")
    padlen = min(3 * max(len(b), len(a)), len(signal) - 1)
    if padlen < 1:
        return signal
    return filtfilt(b, a, signal, padlen=padlen)


def _pad_or_truncate(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Ajusta sinal para exatamente target_len amostras."""
    n = len(signal)
    if n == target_len:
        return signal
    if n > target_len:
        return signal[:target_len]
    # Pad com último valor (evita descontinuidade)
    padded = np.full(target_len, signal[-1] if n > 0 else 0.0, dtype=np.float64)
    padded[:n] = signal
    return padded


def postprocess(
    signals: Dict[str, np.ndarray],
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """Aplica filtros e monta array final (12, 5000).

    Args:
        signals: dict {lead_name: np.ndarray em mV a fs Hz}
        fs: frequência de amostragem

    Returns:
        np.ndarray shape (12, OUTPUT_LENGTH), dtype float64, em mV.
        NaN substituídos por 0.0. Leads ausentes = zeros.
    """
    result = np.zeros((12, OUTPUT_LENGTH), dtype=np.float64)

    for i, lead_name in enumerate(LEAD_ORDER):
        if lead_name not in signals:
            logger.debug("Lead %s ausente — preenchido com zeros", lead_name)
            continue

        sig = signals[lead_name].astype(np.float64)

        if len(sig) < 10:
            logger.warning("Lead %s muito curto (%d amostras) — zeros", lead_name, len(sig))
            continue

        # Filtros (só aplicar se sinal suficientemente longo)
        sig = _highpass_filter(sig, BASELINE_HIGHPASS, fs)
        sig = _notch_filter(sig, POWERLINE_FREQ, fs)
        sig = _lowpass_filter(sig, LOWPASS_FREQ, fs)

        # Clipar artefatos
        max_amp = np.max(np.abs(sig))
        if max_amp > _MAX_AMPLITUDE_MV:
            logger.warning("Lead %s: amplitude %.1f mV > %.1f mV — clipando",
                           lead_name, max_amp, _MAX_AMPLITUDE_MV)
            sig = np.clip(sig, -_MAX_AMPLITUDE_MV, _MAX_AMPLITUDE_MV)

        # Pad/truncar para OUTPUT_LENGTH
        sig = _pad_or_truncate(sig, OUTPUT_LENGTH)

        result[i] = sig

    # Substituir NaN residuais por 0.0
    result = np.nan_to_num(result, nan=0.0)

    # Garantir dtype
    result = result.astype(np.float64)

    leads_active = sum(1 for i in range(12) if np.any(result[i] != 0))
    logger.info("Pós-processamento: %d/12 leads ativos, shape %s", leads_active, result.shape)

    return result
