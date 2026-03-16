"""
Medições automáticas de ECG — ProECG

Recebe sinal digital de 12 derivações (array numpy) proveniente do
ECG-Digitiser e calcula: FC, eixo, PR, QRS, QT, QTc (Bazett), ritmo.

Usa neurokit2 para detecção de ondas P/QRS/T e scipy para cálculos
auxiliares.
"""

from __future__ import annotations

import numpy as np
import neurokit2 as nk
from scipy import signal as scipy_signal

# Ordem padrão das 12 derivações (mesma saída do ECG-Digitiser)
LEAD_NAMES = ["DI", "DII", "DIII", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Frequência de amostragem padrão do ECG-Digitiser (500 Hz)
DEFAULT_FS = 500


def _samples_to_ms(samples: int | float, fs: int) -> float:
    """Converte amostras para milissegundos."""
    return (samples / fs) * 1000


def _compute_heart_rate(r_peaks: np.ndarray, fs: int) -> float | None:
    """Calcula FC média a partir dos R-peaks.
    Fórmula: FC = 60 / intervalo RR médio (em segundos)."""
    if len(r_peaks) < 2:
        return None
    rr_intervals = np.diff(r_peaks) / fs  # em segundos
    rr_mean = np.mean(rr_intervals)
    if rr_mean <= 0:
        return None
    return 60.0 / rr_mean


def _compute_axis(signal_di: np.ndarray, signal_avf: np.ndarray,
                  r_peaks: np.ndarray, fs: int) -> float | None:
    """Calcula eixo elétrico baseado na amplitude do QRS em DI e aVF.
    Usa janela de 40ms antes e 60ms depois do R-peak para medir
    amplitude líquida do QRS."""
    if len(r_peaks) == 0:
        return None

    pre_samples = int(0.040 * fs)   # 40ms antes do R
    post_samples = int(0.060 * fs)  # 60ms depois do R

    amplitudes_di = []
    amplitudes_avf = []

    for rpeak in r_peaks:
        start = max(0, rpeak - pre_samples)
        end = min(len(signal_di), rpeak + post_samples)
        if end <= start:
            continue

        # Amplitude líquida = max - min dentro da janela QRS
        qrs_di = signal_di[start:end]
        qrs_avf = signal_avf[start:end]

        amplitudes_di.append(np.max(qrs_di) - np.min(qrs_di))
        amplitudes_avf.append(np.max(qrs_avf) - np.min(qrs_avf))

    if not amplitudes_di:
        return None

    amp_di = np.mean(amplitudes_di)
    amp_avf = np.mean(amplitudes_avf)

    # Eixo = arctan(aVF / DI) convertido para graus
    axis_rad = np.arctan2(amp_avf, amp_di)
    return float(np.degrees(axis_rad))


def _compute_intervals(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
) -> dict:
    """Calcula intervalos PR, QRS, QT usando neurokit2.
    Retorna dict com pr_interval, qrs_duration, qt_interval (em ms)."""
    result = {"pr_interval": None, "qrs_duration": None, "qt_interval": None}

    if len(r_peaks) < 2:
        return result

    try:
        # Delinear ondas P, QRS, T
        _, waves = nk.ecg_delineate(
            ecg_signal,
            rpeaks=r_peaks,
            sampling_rate=fs,
            method="dwt",
            show=False,
        )

        # PR interval: início de P até início de QRS
        p_onsets = waves.get("ECG_P_Onsets", [])
        qrs_onsets = waves.get("ECG_R_Onsets", [])

        pr_intervals = []
        for p_on, qrs_on in zip(p_onsets, qrs_onsets):
            if p_on is not None and qrs_on is not None and not (
                np.isnan(p_on) or np.isnan(qrs_on)
            ):
                pr_ms = _samples_to_ms(qrs_on - p_on, fs)
                if 50 < pr_ms < 500:  # sanity check
                    pr_intervals.append(pr_ms)

        if pr_intervals:
            result["pr_interval"] = float(np.median(pr_intervals))

        # QRS duration: início a fim do QRS
        qrs_offsets = waves.get("ECG_R_Offsets", [])

        qrs_durations = []
        for qrs_on, qrs_off in zip(qrs_onsets, qrs_offsets):
            if qrs_on is not None and qrs_off is not None and not (
                np.isnan(qrs_on) or np.isnan(qrs_off)
            ):
                qrs_ms = _samples_to_ms(qrs_off - qrs_on, fs)
                if 30 < qrs_ms < 300:  # sanity check
                    qrs_durations.append(qrs_ms)

        if qrs_durations:
            result["qrs_duration"] = float(np.median(qrs_durations))

        # QT interval: início do QRS até fim da onda T
        t_offsets = waves.get("ECG_T_Offsets", [])

        qt_intervals = []
        for qrs_on, t_off in zip(qrs_onsets, t_offsets):
            if qrs_on is not None and t_off is not None and not (
                np.isnan(qrs_on) or np.isnan(t_off)
            ):
                qt_ms = _samples_to_ms(t_off - qrs_on, fs)
                if 200 < qt_ms < 700:  # sanity check
                    qt_intervals.append(qt_ms)

        if qt_intervals:
            result["qt_interval"] = float(np.median(qt_intervals))

    except Exception:
        # Se delineamento falhar, retornar None para os intervalos
        pass

    return result


def _compute_qtc_bazett(qt_ms: float | None, hr: float | None) -> float | None:
    """QTc por Bazett: QTc = QT / sqrt(RR).
    RR em segundos = 60 / FC."""
    if qt_ms is None or hr is None or hr <= 0:
        return None
    rr_sec = 60.0 / hr
    if rr_sec <= 0:
        return None
    return qt_ms / np.sqrt(rr_sec)


def _detect_rhythm(
    r_peaks: np.ndarray,
    p_wave_present: bool,
    hr: float | None,
    fs: int,
) -> str:
    """Determina o ritmo baseado na regularidade dos RR e presença de onda P.
    Retorna string descritiva."""
    if len(r_peaks) < 3:
        return "indeterminado"

    rr_intervals = np.diff(r_peaks) / fs  # em segundos
    rr_mean = np.mean(rr_intervals)
    rr_std = np.std(rr_intervals)

    # Coeficiente de variação dos RR
    rr_cv = rr_std / rr_mean if rr_mean > 0 else 0

    # RR irregular (CV > 15%) sem P → sugestivo de FA
    if rr_cv > 0.15 and not p_wave_present:
        return "irregular_sem_p"

    # RR regular com P → sinusal
    if p_wave_present:
        return "sinus"

    # RR regular sem P → pode ser juncional ou outro
    return "regular_sem_p"


def _detect_p_wave(ecg_lead_ii: np.ndarray, r_peaks: np.ndarray, fs: int) -> bool:
    """Detecta presença de onda P em DII (derivação mais confiável para P)."""
    if len(r_peaks) < 2:
        return True  # assumir presente se não puder avaliar

    try:
        _, waves = nk.ecg_delineate(
            ecg_lead_ii,
            rpeaks=r_peaks,
            sampling_rate=fs,
            method="dwt",
            show=False,
        )
        p_peaks = waves.get("ECG_P_Peaks", [])
        valid_p = [p for p in p_peaks if p is not None and not np.isnan(p)]
        # Se detectou P em pelo menos 60% dos batimentos, considerar presente
        return len(valid_p) >= 0.6 * len(r_peaks)
    except Exception:
        return True  # na dúvida, assumir presente


def measure_ecg(
    signal_12lead: np.ndarray,
    fs: int = DEFAULT_FS,
) -> dict:
    """Função principal de medição.

    Args:
        signal_12lead: array numpy (12, N) com o sinal das 12 derivações.
            Ordem: DI, DII, DIII, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
        fs: frequência de amostragem em Hz (padrão 500).

    Returns:
        dict com:
            - heart_rate (float): FC em bpm
            - heart_rate_unit (str): "bpm"
            - axis (float): eixo elétrico em graus
            - axis_unit (str): "°"
            - pr_interval (float): intervalo PR em ms
            - pr_unit (str): "ms"
            - qrs_duration (float): duração QRS em ms
            - qrs_unit (str): "ms"
            - qt_interval (float): intervalo QT em ms
            - qt_unit (str): "ms"
            - qtc_bazett (float): QTc corrigido em ms
            - qtc_unit (str): "ms"
            - rhythm (str): descrição do ritmo
            - p_wave_present (bool): presença de onda P
    """
    if signal_12lead.ndim != 2 or signal_12lead.shape[0] != 12:
        raise ValueError(
            f"Esperado array (12, N), recebeu {signal_12lead.shape}"
        )

    # Usar DII como derivação de referência (melhor para detecção de P, QRS, T)
    lead_ii = signal_12lead[1]  # DII
    lead_i = signal_12lead[0]   # DI
    lead_avf = signal_12lead[5]  # aVF

    # Pré-processamento: filtro bandpass 0.5-40 Hz
    lead_ii_clean = nk.ecg_clean(lead_ii, sampling_rate=fs, method="neurokit")

    # Detectar R-peaks
    r_peaks_info = nk.ecg_peaks(lead_ii_clean, sampling_rate=fs, method="neurokit")
    r_peaks = r_peaks_info[1]["ECG_R_Peaks"]

    # Frequência cardíaca
    heart_rate = _compute_heart_rate(r_peaks, fs)

    # Eixo elétrico (DI e aVF)
    axis = _compute_axis(lead_i, lead_avf, r_peaks, fs)

    # Intervalos (PR, QRS, QT)
    intervals = _compute_intervals(lead_ii_clean, r_peaks, fs)
    pr_interval = intervals["pr_interval"]
    qrs_duration = intervals["qrs_duration"]
    qt_interval = intervals["qt_interval"]

    # QTc Bazett
    qtc_bazett = _compute_qtc_bazett(qt_interval, heart_rate)

    # Onda P
    p_wave_present = _detect_p_wave(lead_ii_clean, r_peaks, fs)

    # Ritmo
    rhythm = _detect_rhythm(r_peaks, p_wave_present, heart_rate, fs)

    return {
        "heart_rate": round(heart_rate, 0) if heart_rate is not None else None,
        "heart_rate_unit": "bpm",
        "axis": round(axis, 0) if axis is not None else None,
        "axis_unit": "°",
        "pr_interval": round(pr_interval, 0) if pr_interval is not None else None,
        "pr_unit": "ms",
        "qrs_duration": round(qrs_duration, 0) if qrs_duration is not None else None,
        "qrs_unit": "ms",
        "qt_interval": round(qt_interval, 0) if qt_interval is not None else None,
        "qt_unit": "ms",
        "qtc_bazett": round(qtc_bazett, 0) if qtc_bazett is not None else None,
        "qtc_unit": "ms",
        "rhythm": rhythm,
        "p_wave_present": p_wave_present,
    }
