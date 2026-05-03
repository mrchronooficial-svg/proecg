"""
Módulo de Medições Automáticas — ProECG
=======================================

Recebe os 12 sinais digitais (em µV) + sampling_rate do calibrador e
calcula todas as medições necessárias para as regras clínicas.

Detecção de ondas via scipy.signal.find_peaks (base) + neurokit2 (delineamento
fino quando disponível). Retorna dict estruturado conforme spec.

Layout 3×4+1 brasileiro: derivações nomeadas I, II, III, aVR, aVL, aVF,
V1, V2, V3, V4, V5, V6 + (opcional) "II_rhythm" para o DII longo.

Padrão BR de calibração:
    velocidade do papel = 25 mm/s
    ganho               = 10 mm/mV
    1 mm vertical       = 100 µV
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

try:
    import neurokit2 as nk  # type: ignore
    HAS_NK2 = True
except Exception:  # pragma: no cover
    HAS_NK2 = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Aliases pra suportar nomenclatura legada (DI/DII/DIII)
LEAD_ALIASES = {
    "DI": "I", "DII": "II", "DIII": "III",
    "I": "I", "II": "II", "III": "III",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",
    "II_rhythm": "II_rhythm",
}

DEFAULT_FS = 500
GAIN_MM_PER_MV = 10.0  # padrão BR
UV_PER_MM = 1000.0 / GAIN_MM_PER_MV  # 100 µV/mm

# Mapeamento entre nome canônico (I/II/III) e legado (DI/DII/DIII)
LEAD_NEW_TO_LEGACY = {"I": "DI", "II": "DII", "III": "DIII"}


# ---------------------------------------------------------------------------
# Helpers de I/O
# ---------------------------------------------------------------------------

def _normalize_input(
    signal_input, fs: Optional[int] = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Aceita np.array (12, N) ou dict[str, np.ndarray] e retorna dict
    canônico {lead_name: np.array em µV} + fs."""
    if fs is None:
        fs = DEFAULT_FS

    if isinstance(signal_input, dict):
        leads: dict[str, np.ndarray] = {}
        for raw_name, arr in signal_input.items():
            if arr is None:
                continue
            canonical = LEAD_ALIASES.get(raw_name, raw_name)
            if canonical in LEAD_NAMES or canonical == "II_rhythm":
                # Limpa NaN no início/fim que podem vir do trim de borda
                clean = np.asarray(arr, dtype=np.float64)
                leads[canonical] = clean
        return leads, int(fs)

    arr = np.asarray(signal_input, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 12:
        raise ValueError(
            f"Esperado dict ou array (12, N), recebi shape={arr.shape}"
        )
    return {LEAD_NAMES[i]: arr[i] for i in range(12)}, int(fs)


def _strip_nans(sig: np.ndarray) -> np.ndarray:
    """Substitui NaN por 0 (tracking pode deixar NaN no início/fim por trim)."""
    if np.any(np.isnan(sig)):
        clean = sig.copy()
        clean[np.isnan(clean)] = 0.0
        return clean
    return sig


def _samples_to_ms(samples: float, fs: int) -> float:
    return float(samples) / float(fs) * 1000.0


# ---------------------------------------------------------------------------
# 1. Detecção de ondas
# ---------------------------------------------------------------------------

def _detect_r_peaks(
    signal_uv: np.ndarray, fs: int, bipolar: bool = False,
) -> np.ndarray:
    """find_peaks robusto + filtro de outliers RR.

    Critérios:
      - distance ≥ 300 ms (FC máxima fisiológica = 200 bpm; nunca menor)
      - prominence ≥ 40% do maior pico (R é SEMPRE o ponto mais
        proeminente; T-waves têm proeminência menor)
      - height ≥ max(50µV, 50% do p99)

    Se `bipolar=True`, detecta no sinal absoluto — aceita QRS positivo OU
    negativo. Necessário para leads como aVR / V1 onde a maior deflexão
    do QRS é negativa (R minúsculo + S grande). Sem isso, find_peaks
    ignora a deflexão dominante e perde os batimentos.

    Após detecção, aplica filtro: RR < 50% do mediano = falso positivo
    (T-wave sendo detectada como R). Remove iterativamente o pico
    espúrio até todos os RR ficarem consistentes.
    """
    sig = _strip_nans(signal_uv)
    min_dist = max(1, int(round(0.300 * fs)))  # 300ms
    abs_sig = np.abs(sig)
    p99 = float(np.percentile(abs_sig, 99))
    max_amp = float(np.max(abs_sig))
    height = max(50.0, p99 * 0.5)
    prominence = max(40.0, max_amp * 0.4)  # 40% do maior pico
    target = abs_sig if bipolar else sig
    peaks, _ = find_peaks(
        target, height=height, distance=min_dist, prominence=prominence,
    )
    peaks = peaks.astype(np.int64)

    # Filtro de outliers: remove picos que criam RR < 50% da mediana
    # Itera até estabilizar (max 10 iterações por segurança)
    for _ in range(10):
        if len(peaks) < 3:
            break
        rr = np.diff(peaks)
        median_rr = float(np.median(rr))
        if median_rr <= 0:
            break
        threshold = median_rr * 0.5
        short = np.where(rr < threshold)[0]
        if len(short) == 0:
            break
        # Remove o pico DIREITO do RR mais curto (frequentemente é T-wave
        # detectada após o R real)
        i = int(short[np.argmin(rr[short])])
        peaks = np.delete(peaks, i + 1)

    return peaks


def _detect_p_peaks(
    signal_uv: np.ndarray, r_peaks: np.ndarray, fs: int,
    min_amp_uv: float = 50.0,
) -> tuple[np.ndarray, list[bool]]:
    """Pra cada R, procura pico positivo entre 120-300 ms ANTES de R.

    Returns:
        (p_peak_indices, found_flags) — usa NaN/False quando não detecta.
    """
    sig = _strip_nans(signal_uv)
    p_search_back = int(round(0.300 * fs))
    p_search_near = int(round(0.080 * fs))
    p_peaks = np.full(len(r_peaks), -1, dtype=np.int64)
    found = []
    for i, r in enumerate(r_peaks):
        win_start = max(0, int(r) - p_search_back)
        win_end = max(0, int(r) - p_search_near)
        if win_end <= win_start + 2:
            found.append(False)
            continue
        seg = sig[win_start:win_end]
        # Pico local > min_amp_uv
        loc = int(np.argmax(seg))
        if seg[loc] >= min_amp_uv:
            p_peaks[i] = win_start + loc
            found.append(True)
        else:
            found.append(False)
    return p_peaks, found


def _detect_qrs_onsets_offsets(
    signal_uv: np.ndarray, r_peaks: np.ndarray, fs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detecta onset/offset do QRS via derivada com threshold adaptativo.

    Threshold inicial: 10% do pico de derivada local (era 20%, gerava QRS
    curto demais). Onset = último ponto onde |d/dt| cai abaixo do threshold
    (walking back). Offset (ponto J) = primeiro ponto onde |d/dt| cai
    abaixo do threshold (walking forward).

    Validação: se QRS < 60ms (impossível fisiologicamente), repete a
    detecção pra esse batimento com threshold de 5%.

    Janela ampliada (120ms back, 200ms forward) pra acomodar QRS largos
    de bloqueios de ramo.
    """
    sig = _strip_nans(signal_uv)
    deriv = np.abs(np.gradient(sig))
    qrs_window_back = int(round(0.120 * fs))
    qrs_window_fwd = int(round(0.200 * fs))
    min_qrs_samples = int(round(0.060 * fs))  # 60ms

    onsets = np.full(len(r_peaks), -1, dtype=np.int64)
    offsets = np.full(len(r_peaks), -1, dtype=np.int64)

    def _walk(r: int, threshold_pct: float) -> tuple[int, int]:
        win_start = max(0, r - qrs_window_back)
        win_end = min(len(sig), r + qrs_window_fwd)
        local_deriv = deriv[win_start:win_end]
        if len(local_deriv) < 5:
            return -1, -1
        thr = float(np.max(local_deriv)) * threshold_pct
        on = r
        for k in range(r - 1, win_start - 1, -1):
            if deriv[k] <= thr:
                on = k
                break
        off = r
        for k in range(r + 1, win_end):
            if deriv[k] <= thr:
                off = k
                break
        return on, off

    # Primeira passada: threshold 10%
    for i, r in enumerate(r_peaks):
        r = int(r)
        on, off = _walk(r, 0.10)
        onsets[i], offsets[i] = on, off

    # Segunda passada: pra QRS < 60ms, retenta com threshold 5%
    for i, r in enumerate(r_peaks):
        r = int(r)
        if onsets[i] < 0 or offsets[i] < 0:
            continue
        if (offsets[i] - onsets[i]) < min_qrs_samples:
            on2, off2 = _walk(r, 0.05)
            if on2 >= 0 and off2 >= 0 and (off2 - on2) > (offsets[i] - onsets[i]):
                onsets[i], offsets[i] = on2, off2

    return onsets, offsets


def _detect_t_peaks_offsets(
    signal_uv: np.ndarray, r_peaks: np.ndarray, qrs_offsets: np.ndarray, fs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detecta pico T (positivo ou negativo) e fim da onda T.

    T peak: maior |amplitude| entre 80-400 ms após o ponto J.
    T offset: ponto onde |d/dt| volta abaixo de threshold após o pico T.
    """
    sig = _strip_nans(signal_uv)
    t_search_start = int(round(0.080 * fs))
    t_search_end = int(round(0.400 * fs))

    t_peaks = np.full(len(r_peaks), -1, dtype=np.int64)
    t_offsets = np.full(len(r_peaks), -1, dtype=np.int64)

    deriv = np.abs(np.gradient(sig))

    for i, j_pt in enumerate(qrs_offsets):
        j_pt = int(j_pt)
        if j_pt < 0:
            continue
        win_start = j_pt + t_search_start
        win_end = min(len(sig), j_pt + t_search_end)
        if win_end <= win_start + 5:
            continue
        seg = sig[win_start:win_end]
        # Pico = maior |amplitude|
        idx = int(np.argmax(np.abs(seg)))
        t_peak = win_start + idx
        t_peaks[i] = t_peak

        # T offset: walk forward até derivada cair
        local_deriv = deriv[t_peak:win_end]
        if len(local_deriv) < 3:
            continue
        thr = float(np.max(local_deriv)) * 0.20
        off = t_peak
        for k in range(t_peak + 1, win_end):
            if deriv[k] <= thr:
                off = k
                break
        t_offsets[i] = off

    return t_peaks, t_offsets


def _baseline_uv(
    signal_uv: np.ndarray, t_offsets: np.ndarray, p_peaks: np.ndarray,
    r_peaks: np.ndarray, fs: int,
) -> float:
    """Mediana do sinal nos segmentos TP (entre fim de T e próxima P)."""
    sig = _strip_nans(signal_uv)
    samples: list[float] = []
    for i in range(len(r_peaks) - 1):
        t_off = int(t_offsets[i]) if i < len(t_offsets) else -1
        next_p = int(p_peaks[i + 1]) if i + 1 < len(p_peaks) else -1
        if t_off < 0 or next_p < 0 or next_p <= t_off:
            continue
        # Margem de 20ms entre T_off e P_peak
        margin = int(round(0.020 * fs))
        seg_start = t_off + margin
        seg_end = next_p - margin
        if seg_end - seg_start < 5:
            continue
        samples.extend(sig[seg_start:seg_end].tolist())
    if not samples:
        # Fallback: mediana do sinal todo
        return float(np.median(sig))
    return float(np.median(samples))


def _detect_per_lead(
    signal_uv: np.ndarray, fs: int,
) -> dict:
    """Detecção independente por derivação.

    Cada cell do layout 3×4 é uma fatia temporal diferente do registro
    (DII = 0–2.5s, V2 = 5–7.5s, etc.), então R-peaks/QRS/T detectados
    em DII NÃO correspondem aos índices de batimento das outras
    derivações. Esta função roda todo o pipeline de detecção sobre o
    sinal de uma única derivação para obter J-points e baseline corretos
    para medir ST naquela derivação.

    Para leads com QRS predominantemente negativo (aVR, V1 com rS),
    `_detect_r_peaks` falha por procurar apenas picos positivos. Aqui
    fazemos fallback automático para detecção bipolar (sinal absoluto).

    Returns:
        dict com r_peaks, p_peaks, qrs_on, qrs_off, t_peaks, t_off,
        baseline_uv. Listas/arrays vazios se a detecção falhar.
    """
    sig = _strip_nans(signal_uv)
    r_peaks = _detect_r_peaks(sig, fs)
    if len(r_peaks) < 2:
        # Fallback bipolar pra leads com QRS negativo dominante
        r_peaks = _detect_r_peaks(sig, fs, bipolar=True)
    if len(r_peaks) < 2:
        n = len(sig)
        return {
            "r_peaks": np.zeros(0, dtype=np.int64),
            "p_peaks": np.zeros(0, dtype=np.int64),
            "qrs_on": np.zeros(0, dtype=np.int64),
            "qrs_off": np.zeros(0, dtype=np.int64),
            "t_peaks": np.zeros(0, dtype=np.int64),
            "t_off": np.zeros(0, dtype=np.int64),
            "baseline_uv": float(np.median(sig)) if n else 0.0,
        }
    p_peaks, _ = _detect_p_peaks(sig, r_peaks, fs)
    qrs_on, qrs_off = _detect_qrs_onsets_offsets(sig, r_peaks, fs)
    t_peaks, t_off = _detect_t_peaks_offsets(sig, r_peaks, qrs_off, fs)
    baseline = _baseline_uv(sig, t_off, p_peaks, r_peaks, fs)
    return {
        "r_peaks": r_peaks, "p_peaks": p_peaks,
        "qrs_on": qrs_on, "qrs_off": qrs_off,
        "t_peaks": t_peaks, "t_off": t_off,
        "baseline_uv": baseline,
    }


# ---------------------------------------------------------------------------
# 2. Medições
# ---------------------------------------------------------------------------

def _heart_rate_stats(r_peaks: np.ndarray, fs: int) -> dict:
    """FC mean/min/max + RR intervals + flag de regularidade."""
    if len(r_peaks) < 2:
        return {
            "mean_bpm": None, "min_bpm": None, "max_bpm": None,
            "regular": None, "rr_intervals_ms": [],
        }
    rr_samples = np.diff(r_peaks)
    rr_ms = (rr_samples / fs * 1000.0).tolist()
    rr_arr = np.array(rr_ms)
    rr_mean = float(np.mean(rr_arr))
    rr_min = float(np.min(rr_arr))
    rr_max = float(np.max(rr_arr))
    cv = float(np.std(rr_arr)) / rr_mean if rr_mean > 0 else 0.0
    return {
        "mean_bpm": round(60000.0 / rr_mean, 1) if rr_mean > 0 else None,
        "min_bpm": round(60000.0 / rr_max, 1) if rr_max > 0 else None,
        "max_bpm": round(60000.0 / rr_min, 1) if rr_min > 0 else None,
        "regular": bool(cv < 0.20),
        "rr_intervals_ms": [round(x, 1) for x in rr_ms],
    }


def _classify_axis(degrees: Optional[float]) -> str:
    if degrees is None:
        return "indeterminado"
    if -30.0 <= degrees <= 90.0:
        return "normal"
    if -90.0 <= degrees < -30.0:
        return "desvio_esquerdo"
    if 90.0 < degrees <= 180.0:
        return "desvio_direito"
    return "indeterminado"  # -180° a -90° = extreme/northwest


def _compute_axis(
    sig_di: np.ndarray, sig_avf: np.ndarray,
    r_peaks_di: np.ndarray, r_peaks_avf: np.ndarray, fs: int,
) -> Optional[float]:
    """Eixo elétrico via amplitude líquida do QRS em DI vs aVF.

    DI e aVF estão em colunas temporais diferentes do layout 3×4 (col 1
    e col 2), então cada um precisa dos seus próprios R-peaks pra que
    a soma do QRS caia sobre o batimento certo.
    """
    pre = int(round(0.040 * fs))
    post = int(round(0.060 * fs))

    def _mean_qrs_sum(sig: np.ndarray, peaks: np.ndarray) -> Optional[float]:
        sums = []
        for r in peaks:
            r = int(r)
            s = max(0, r - pre)
            e = min(len(sig), r + post)
            if e <= s:
                continue
            sums.append(float(np.sum(sig[s:e])))
        return float(np.mean(sums)) if sums else None

    amp_di = _mean_qrs_sum(sig_di, r_peaks_di)
    amp_avf = _mean_qrs_sum(sig_avf, r_peaks_avf)
    if amp_di is None or amp_avf is None:
        return None
    return float(np.degrees(np.arctan2(amp_avf, amp_di)))


def _intervals(
    r_peaks: np.ndarray,
    p_peaks: np.ndarray,
    qrs_onsets: np.ndarray,
    qrs_offsets: np.ndarray,
    t_offsets: np.ndarray,
    fs: int,
    qrs_per_lead: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
) -> dict:
    """Calcula intervalos PR, QRS, QT, PP.

    QRS é medido na derivação onde sai mais largo (padrão clínico).
    Se `qrs_per_lead` fornecido, usa o máximo entre as derivações por
    batimento; senão usa onsets/offsets do DII.
    """
    out = {"pr_ms": None, "qrs_ms": None, "qt_ms": None, "qtc_ms": None, "pp_ms": None}

    # PR: P_peak -> QRS_onset (DII)
    pr_vals = []
    for i in range(len(r_peaks)):
        p = int(p_peaks[i]) if i < len(p_peaks) else -1
        qon = int(qrs_onsets[i]) if i < len(qrs_onsets) else -1
        if p < 0 or qon < 0 or qon <= p:
            continue
        pr_ms = _samples_to_ms(qon - p, fs)
        if 80.0 < pr_ms < 400.0:
            pr_vals.append(pr_ms)
    if pr_vals:
        out["pr_ms"] = float(np.median(pr_vals))

    # QRS: pega o MAIOR valor por batimento entre todas as derivações
    qrs_vals = []
    if qrs_per_lead:
        n_beats = len(r_peaks)
        for i in range(n_beats):
            widest = 0
            for ons, offs in qrs_per_lead.values():
                if i >= len(ons) or i >= len(offs):
                    continue
                if ons[i] < 0 or offs[i] < 0 or offs[i] <= ons[i]:
                    continue
                dur = int(offs[i]) - int(ons[i])
                if dur > widest:
                    widest = dur
            if widest > 0:
                qrs_ms = _samples_to_ms(widest, fs)
                if 30.0 < qrs_ms < 300.0:
                    qrs_vals.append(qrs_ms)
    else:
        for qon, qoff in zip(qrs_onsets, qrs_offsets):
            if qon < 0 or qoff < 0 or qoff <= qon:
                continue
            qrs_ms = _samples_to_ms(qoff - qon, fs)
            if 30.0 < qrs_ms < 300.0:
                qrs_vals.append(qrs_ms)
    if qrs_vals:
        out["qrs_ms"] = float(np.median(qrs_vals))

    # QT: QRS_onset -> T_offset (DII)
    qt_vals = []
    for qon, toff in zip(qrs_onsets, t_offsets):
        if qon < 0 or toff < 0 or toff <= qon:
            continue
        qt_ms = _samples_to_ms(toff - qon, fs)
        if 200.0 < qt_ms < 700.0:
            qt_vals.append(qt_ms)
    if qt_vals:
        out["qt_ms"] = float(np.median(qt_vals))

    # PP: distância entre ondas P consecutivas
    valid_p = [int(p) for p in p_peaks if p >= 0]
    if len(valid_p) >= 2:
        pp_diffs = np.diff(valid_p) / fs * 1000.0
        out["pp_ms"] = float(np.median(pp_diffs))

    # QTc Bazett
    if out["qt_ms"] is not None and len(r_peaks) >= 2:
        rr_sec = float(np.median(np.diff(r_peaks))) / fs
        if rr_sec > 0:
            out["qtc_ms"] = round(out["qt_ms"] / np.sqrt(rr_sec), 1)

    return {k: round(v, 1) if isinstance(v, float) else v for k, v in out.items()}


def _p_wave_summary(
    sig_di_ii: np.ndarray, p_peaks: np.ndarray, fs: int,
    sig_v1: Optional[np.ndarray] = None,
    duration_search_mm: float = 1.5,
) -> dict:
    """Duração + amplitude da onda P em DII; checa P bifásica em V1."""
    valid_p = [int(p) for p in p_peaks if p >= 0]
    if not valid_p:
        return {
            "duration_ms": None, "amplitude_uv": None,
            "present": False, "bifasic_v1": False,
        }
    sig = _strip_nans(sig_di_ii)
    # Amplitude P = mediana dos picos
    amps = [float(sig[p]) for p in valid_p]
    amp_uv = float(np.median(amps))

    # Duração P: estima walking back/forward até cruzar 20% do pico
    durations = []
    for p in valid_p:
        peak_amp = float(sig[p])
        if peak_amp < 30.0:
            continue
        thr = peak_amp * 0.20
        # Walk back
        start = p
        for k in range(p - 1, max(0, p - int(0.100 * fs)) - 1, -1):
            if sig[k] < thr:
                start = k
                break
        # Walk forward
        end = p
        for k in range(p + 1, min(len(sig), p + int(0.100 * fs))):
            if sig[k] < thr:
                end = k
                break
        if end > start:
            durations.append(_samples_to_ms(end - start, fs))
    duration_ms = float(np.median(durations)) if durations else None

    # P bifásica em V1: componente negativo > 1mm × 1mm
    bifasic_v1 = False
    if sig_v1 is not None and len(valid_p) > 0:
        sigv1 = _strip_nans(sig_v1)
        for p in valid_p:
            # Janela: 100ms ao redor do P
            half = int(round(0.060 * fs))
            ws, we = max(0, p - half), min(len(sigv1), p + half)
            seg = sigv1[ws:we]
            if len(seg) < 5:
                continue
            # Componente negativo > 100µV (1mm) por > 40ms (1mm a 25mm/s)
            neg = seg[seg < -100.0]
            if len(neg) >= int(round(0.040 * fs)):
                bifasic_v1 = True
                break

    return {
        "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
        "amplitude_uv": round(amp_uv, 1),
        "present": True,
        "bifasic_v1": bifasic_v1,
    }


def _st_per_lead(
    leads: dict[str, np.ndarray],
    qrs_offsets_per_lead: dict[str, np.ndarray],
    baseline_per_lead: dict[str, float],
    fs: int,
) -> dict:
    """Mede ST em ponto J + 60ms vs baseline pra cada derivação.

    Os J-points (qrs_offsets) DEVEM vir do próprio sinal da derivação —
    o layout 3×4 coloca cada lead em uma fatia temporal diferente do
    registro, então qrs_offsets compartilhados (de DII) caem em fases
    erradas do batimento e fazem o ST ser medido sobre a onda T do
    batimento anterior.
    """
    j_offset = int(round(0.060 * fs))
    out: dict[str, dict] = {}
    for name in LEAD_NAMES:
        if name not in leads:
            continue
        sig = _strip_nans(leads[name])
        baseline = baseline_per_lead.get(name, 0.0)
        qrs_offsets = qrs_offsets_per_lead.get(name)
        if qrs_offsets is None or len(qrs_offsets) == 0:
            out[name] = {"value_mm": None, "value_uv": None, "class": "indeterminado"}
            continue
        st_uvs = []
        for j_pt in qrs_offsets:
            j_pt = int(j_pt)
            if j_pt < 0:
                continue
            idx = j_pt + j_offset
            if idx >= len(sig):
                continue
            st_uvs.append(float(sig[idx]) - baseline)
        if not st_uvs:
            out[name] = {"value_mm": None, "value_uv": None, "class": "indeterminado"}
            continue
        st_uv = float(np.median(st_uvs))
        st_mm = st_uv / UV_PER_MM
        # Thresholds
        if name in ("V1", "V2", "V3"):
            supra_thr_mm = 2.0
        else:
            supra_thr_mm = 1.0
        infra_thr_mm = -0.5
        if st_mm >= supra_thr_mm:
            cls = "supra"
        elif st_mm <= infra_thr_mm:
            cls = "infra"
        else:
            cls = "normal"
        out[name] = {
            "value_mm": round(st_mm, 2),
            "value_uv": round(st_uv, 1),
            "class": cls,
        }
    return out


def _t_per_lead(
    leads: dict[str, np.ndarray],
    qrs_offsets_per_lead: dict[str, np.ndarray],
    fs: int,
) -> dict:
    """Polaridade + amplitude da onda T por derivação.

    Igual ao ST: busca a onda T relativa ao J-point da própria derivação,
    senão a janela cai em fase errada quando o lead está em outra coluna
    temporal do layout 3×4.
    """
    out: dict[str, dict] = {}
    t_search_start = int(round(0.080 * fs))
    t_search_end = int(round(0.400 * fs))
    flat_thr_uv = 50.0  # < 0.5mm = achatada

    for name in LEAD_NAMES:
        if name not in leads:
            continue
        sig = _strip_nans(leads[name])
        amps = []
        polarities = []
        qrs_offsets = qrs_offsets_per_lead.get(name)
        if qrs_offsets is None:
            qrs_offsets = np.zeros(0, dtype=np.int64)
        for j_pt in qrs_offsets:
            j_pt = int(j_pt)
            if j_pt < 0:
                continue
            ws = j_pt + t_search_start
            we = min(len(sig), j_pt + t_search_end)
            if we <= ws + 5:
                continue
            seg = sig[ws:we]
            # Pico de maior amplitude (positivo OU negativo)
            idx_pos = int(np.argmax(seg))
            idx_neg = int(np.argmin(seg))
            pos_amp = float(seg[idx_pos])
            neg_amp = float(seg[idx_neg])
            amp = pos_amp if abs(pos_amp) >= abs(neg_amp) else neg_amp
            amps.append(amp)
            # Polaridade do batimento
            if abs(amp) < flat_thr_uv:
                polarities.append("flat")
            elif pos_amp > flat_thr_uv and neg_amp < -flat_thr_uv:
                polarities.append("biphasic")
            elif amp > 0:
                polarities.append("positive")
            else:
                polarities.append("negative")
        if not amps:
            out[name] = {"polarity": "unknown", "amplitude_uv": None}
            continue
        # Polaridade dominante = moda
        from collections import Counter
        polarity = Counter(polarities).most_common(1)[0][0]
        out[name] = {
            "polarity": polarity,
            "amplitude_uv": round(float(np.median(amps)), 1),
        }
    return out


def _q_per_lead(
    leads: dict[str, np.ndarray],
    qrs_onsets_per_lead: dict[str, np.ndarray],
    r_peaks_per_lead: dict[str, np.ndarray],
    fs: int,
) -> dict:
    """Detecta onda Q + classifica como patológica.

    Q patológica: duração > 40ms OU amplitude > 25% da onda R.
    Q = primeira deflexão negativa antes do pico R no QRS.
    Usa landmarks do próprio lead (cada cell tem fatia temporal diferente).
    """
    out: dict[str, dict] = {}
    for name in LEAD_NAMES:
        if name not in leads:
            continue
        sig = _strip_nans(leads[name])
        durations = []
        amplitudes = []
        r_amps = []
        path_flags = []
        qrs_onsets = qrs_onsets_per_lead.get(name)
        r_peaks = r_peaks_per_lead.get(name)
        if qrs_onsets is None or r_peaks is None:
            out[name] = {
                "pathological": False, "duration_ms": None,
                "amplitude_uv": None,
            }
            continue
        for qon, r in zip(qrs_onsets, r_peaks):
            qon, r = int(qon), int(r)
            if qon < 0 or r < 0 or r <= qon:
                continue
            # Pixels do QRS antes do R: qon → r
            seg = sig[qon:r]
            if len(seg) < 3:
                continue
            min_idx = int(np.argmin(seg))
            q_amp = float(seg[min_idx])
            r_amp = float(sig[r])
            # Q só existe se há deflexão negativa significativa antes do R
            if q_amp >= 0:
                # Sem Q (QRS começa positivo)
                continue
            # Duração da Q: do qon até voltar a 0
            q_duration_samples = 0
            for k in range(qon + min_idx + 1, r):
                if sig[k] >= 0:
                    q_duration_samples = k - qon
                    break
            if q_duration_samples == 0:
                q_duration_samples = min_idx
            q_duration_ms = _samples_to_ms(q_duration_samples, fs)
            durations.append(q_duration_ms)
            amplitudes.append(q_amp)
            r_amps.append(r_amp if r_amp > 0 else 0.0)
            # Patológica?
            is_path = False
            if q_duration_ms > 40.0:
                is_path = True
            if r_amp > 0 and abs(q_amp) > r_amp * 0.25:
                is_path = True
            path_flags.append(is_path)

        if not durations:
            out[name] = {
                "pathological": False, "duration_ms": None,
                "amplitude_uv": None,
            }
            continue
        # Q patológica se >= 50% dos batimentos forem patológicos
        pathological = sum(path_flags) >= len(path_flags) * 0.5
        out[name] = {
            "pathological": bool(pathological),
            "duration_ms": round(float(np.median(durations)), 1),
            "amplitude_uv": round(float(np.median(amplitudes)), 1),
        }
    return out


def _r_s_ratio(
    leads: dict[str, np.ndarray],
    qrs_onsets_per_lead: dict[str, np.ndarray],
    qrs_offsets_per_lead: dict[str, np.ndarray],
    r_peaks_per_lead: dict[str, np.ndarray],
    fs: int,
) -> dict:
    """R/S ratio em V1 e V6.

    R = amplitude positiva máxima no QRS
    S = amplitude negativa máxima no QRS após o R
    """
    out: dict[str, Optional[float]] = {"V1": None, "V6": None}
    for name in ("V1", "V6"):
        if name not in leads:
            continue
        sig = _strip_nans(leads[name])
        ratios = []
        qrs_onsets = qrs_onsets_per_lead.get(name)
        qrs_offsets = qrs_offsets_per_lead.get(name)
        r_peaks = r_peaks_per_lead.get(name)
        if qrs_onsets is None or qrs_offsets is None or r_peaks is None:
            continue
        for qon, qoff, r in zip(qrs_onsets, qrs_offsets, r_peaks):
            qon, qoff, r = int(qon), int(qoff), int(r)
            if qon < 0 or qoff < 0 or r < 0:
                continue
            r_amp = float(np.max(sig[qon:qoff + 1])) if qoff > qon else 0.0
            # S = mín entre R e qoff
            if qoff > r:
                s_amp = float(np.min(sig[r:qoff + 1]))
            else:
                s_amp = float(np.min(sig[qon:qoff + 1]))
            if s_amp >= 0 or r_amp <= 0:
                ratios.append(float("inf") if r_amp > 0 else 0.0)
                continue
            ratios.append(r_amp / abs(s_amp))
        if ratios:
            finite = [x for x in ratios if np.isfinite(x)]
            out[name] = round(float(np.median(finite)), 2) if finite else None
    return out


# ---------------------------------------------------------------------------
# 3. Função pública
# ---------------------------------------------------------------------------

def measure_ecg(
    signal_input,
    fs: Optional[int] = None,
    calibration: Optional[dict] = None,
) -> dict:
    """Pipeline de medições. Aceita:
        signal_input: np.array (12, N) OU dict[lead_name, np.array em µV].
        fs: sampling rate em Hz (sobrescreve calibration).
        calibration: dict do calibrador (sampling_rate_hz, etc).

    Returns:
        Dict estruturado conforme spec (heart_rate, axis, intervals,
        p_wave, st_segment, t_wave, q_wave, r_s_ratio, r_peaks_ms,
        quality, warnings).
    """
    if fs is None and calibration is not None:
        fs = int(round(float(calibration.get("sampling_rate_hz", DEFAULT_FS))))
    if fs is None:
        fs = DEFAULT_FS

    leads, fs = _normalize_input(signal_input, fs)
    warnings_: list[str] = []

    if "II" not in leads:
        raise ValueError("Sinal de DII (II) é obrigatório como referência")

    sig_ii = _strip_nans(leads["II"])

    # ---- 1. R-peaks na DII ----
    r_peaks = _detect_r_peaks(sig_ii, fs)
    if len(r_peaks) < 2:
        warnings_.append(f"Apenas {len(r_peaks)} R-peak(s) detectado(s) em DII")

    # ---- 2. P-peaks na DII ----
    p_peaks, p_found = _detect_p_peaks(sig_ii, r_peaks, fs)
    p_wave_present = sum(p_found) >= 0.6 * max(len(p_found), 1)
    p_wave_confidence = (
        sum(p_found) / len(p_found) if p_found else 0.0
    )

    # ---- 3. QRS onsets/offsets na DII (referência pra PR e QT) ----
    qrs_onsets, qrs_offsets = _detect_qrs_onsets_offsets(sig_ii, r_peaks, fs)

    # ---- 3b. Detecção POR LEAD (R/P/QRS/T/baseline) ----
    # Cada cell do layout 3×4 é uma fatia temporal diferente do registro
    # (DII em 0–2.5s, V2 em 5–7.5s, …), então R-peaks da DII NÃO marcam os
    # batimentos das outras derivações. Detectar os landmarks no próprio
    # sinal de cada lead é o que garante que ST/T/Q sejam medidos no
    # ponto certo do batimento daquele lead.
    per_lead: dict[str, dict] = {}
    r_peaks_per_lead: dict[str, np.ndarray] = {}
    qrs_onsets_per_lead: dict[str, np.ndarray] = {}
    qrs_offsets_per_lead: dict[str, np.ndarray] = {}
    qrs_per_lead: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    baseline_per_lead: dict[str, float] = {}
    for name in LEAD_NAMES:
        if name not in leads:
            continue
        d = _detect_per_lead(leads[name], fs)
        per_lead[name] = d
        r_peaks_per_lead[name] = d["r_peaks"]
        qrs_onsets_per_lead[name] = d["qrs_on"]
        qrs_offsets_per_lead[name] = d["qrs_off"]
        qrs_per_lead[name] = (d["qrs_on"], d["qrs_off"])
        baseline_per_lead[name] = d["baseline_uv"]

    # ---- 4. T peaks/offsets na DII (referência para QT) ----
    t_peaks, t_offsets = _detect_t_peaks_offsets(
        sig_ii, r_peaks, qrs_offsets, fs,
    )
    t_wave_confidence = float(np.sum(t_offsets >= 0)) / max(len(r_peaks), 1)

    # ---- 6. Métricas ----
    hr_stats = _heart_rate_stats(r_peaks, fs)

    # Eixo (DI vs aVF) — usa R-peaks do próprio lead, pois I e aVF estão
    # em colunas temporais distintas do layout 3×4.
    axis_deg = None
    if "I" in leads and "aVF" in leads:
        axis_deg = _compute_axis(
            _strip_nans(leads["I"]), _strip_nans(leads["aVF"]),
            r_peaks_per_lead.get("I", r_peaks),
            r_peaks_per_lead.get("aVF", r_peaks),
            fs,
        )
    axis_class = _classify_axis(axis_deg)

    intervals = _intervals(
        r_peaks, p_peaks, qrs_onsets, qrs_offsets, t_offsets, fs,
        qrs_per_lead=qrs_per_lead,
    )

    # Onda P (em DII; checa bifásica em V1 se disponível)
    p_summary = _p_wave_summary(
        sig_ii, p_peaks, fs, sig_v1=leads.get("V1"),
    )
    p_summary["present"] = p_wave_present

    st_seg = _st_per_lead(leads, qrs_offsets_per_lead, baseline_per_lead, fs)
    t_per = _t_per_lead(leads, qrs_offsets_per_lead, fs)
    q_per = _q_per_lead(leads, qrs_onsets_per_lead, r_peaks_per_lead, fs)
    rs_ratio = _r_s_ratio(
        leads, qrs_onsets_per_lead, qrs_offsets_per_lead, r_peaks_per_lead, fs,
    )

    r_peaks_ms = [round(float(r) / fs * 1000.0, 1) for r in r_peaks]

    # ---- Qualidade ----
    if p_wave_confidence >= 0.7 and t_wave_confidence >= 0.7:
        overall = "good"
    elif p_wave_confidence >= 0.4 or t_wave_confidence >= 0.4:
        overall = "fair"
    else:
        overall = "poor"

    if intervals.get("qt_ms") is None:
        warnings_.append("QT não mensurável (onda T não detectada)")
    if intervals.get("pr_ms") is None and not p_wave_present:
        warnings_.append("PR não mensurável (onda P não identificada)")

    # ---- Validações clínicas (limites fisiológicos) ----
    qrs_ms = intervals.get("qrs_ms")
    pr_ms = intervals.get("pr_ms")
    qtc_ms = intervals.get("qtc_ms")
    hr_bpm = hr_stats.get("mean_bpm")

    if qrs_ms is not None and (qrs_ms < 60.0 or qrs_ms > 200.0):
        warnings_.append(f"QRS não confiável ({qrs_ms:.0f} ms fora de 60-200)")
    if pr_ms is not None and (pr_ms < 80.0 or pr_ms > 400.0):
        warnings_.append(f"PR não confiável ({pr_ms:.0f} ms fora de 80-400)")
    if qtc_ms is not None and (qtc_ms < 300.0 or qtc_ms > 600.0):
        warnings_.append(f"QTc não confiável ({qtc_ms:.0f} ms fora de 300-600)")
    if hr_bpm is not None and (hr_bpm < 20.0 or hr_bpm > 220.0):
        warnings_.append(f"FC não confiável ({hr_bpm:.0f} bpm fora de 20-220)")

    # ---- QRS amplitudes por lead (pra critérios de sobrecarga, ex: Sokolow) ----
    qrs_amplitudes: dict[str, dict] = {}
    for name in LEAD_NAMES:
        if name not in leads:
            continue
        sig = _strip_nans(leads[name])
        r_amps, s_amps, qrs_net = [], [], []
        ons_arr, offs_arr = qrs_per_lead.get(name, (qrs_onsets, qrs_offsets))
        rpk = r_peaks_per_lead.get(name, r_peaks)
        for ron, roff, r in zip(ons_arr, offs_arr, rpk):
            ron, roff, r = int(ron), int(roff), int(r)
            if ron < 0 or roff < 0 or roff <= ron:
                continue
            seg = sig[ron:roff + 1]
            if len(seg) < 2:
                continue
            r_amp = float(np.max(seg))
            s_amp = float(np.min(seg))
            r_amps.append(max(0.0, r_amp))
            s_amps.append(abs(min(0.0, s_amp)))
            qrs_net.append(r_amp + s_amp)  # signed sum (pra Sgarbossa)
        if r_amps:
            qrs_amplitudes[name] = {
                "r_uv": float(np.median(r_amps)),
                "s_uv": float(np.median(s_amps)),
                "r_mm": round(float(np.median(r_amps)) / UV_PER_MM, 2),
                "s_mm": round(float(np.median(s_amps)) / UV_PER_MM, 2),
                "net_uv": float(np.median(qrs_net)),
            }

    return {
        "heart_rate": hr_stats,
        "axis": {
            "degrees": round(axis_deg, 1) if axis_deg is not None else None,
            "classification": axis_class,
        },
        "intervals": intervals,
        "p_wave": p_summary,
        "st_segment": st_seg,
        "t_wave": t_per,
        "q_wave": q_per,
        "r_s_ratio": rs_ratio,
        "qrs_amplitudes": qrs_amplitudes,
        "r_peaks_ms": r_peaks_ms,
        "quality": {
            "p_wave_confidence": round(p_wave_confidence, 2),
            "t_wave_confidence": round(t_wave_confidence, 2),
            "overall": overall,
        },
        "warnings": warnings_,
    }


# ---------------------------------------------------------------------------
# Adapter: novo formato → formato legacy (rules.py + report.py)
# ---------------------------------------------------------------------------

def to_legacy_format(m: dict, sex: str = "M") -> dict:
    """Converte o output de measure_ecg (novo, estruturado) pro formato
    plano esperado por rules.py e report.py.

    Mapeamento:
      heart_rate.mean_bpm  -> heart_rate (escalar)
      axis.degrees         -> axis (escalar)
      intervals.pr_ms      -> pr_interval
      intervals.qrs_ms     -> qrs_duration
      intervals.qt_ms      -> qt_interval
      intervals.qtc_ms     -> qtc_bazett
      p_wave.present       -> p_wave_present
      st_segment[lead]     -> leads[DI/DII/...].st_elevation/st_depression
      t_wave[lead]         -> leads[...].t_wave (polarity)
      q_wave[lead]         -> leads[...].has_q_wave (pathological)
      qrs_amplitudes[lead] -> leads[...].qrs_amplitude (mm)

    Ritmo (heurística):
      regular + p_present  -> "sinus"
      irregular + sem p    -> "irregular_sem_p"
      regular + sem p      -> "regular_sem_p"
      caso contrário       -> "indeterminado"
    """
    if not isinstance(m, dict):
        return m  # já em legacy ou inválido

    hr = m.get("heart_rate") or {}
    ax = m.get("axis") or {}
    iv = m.get("intervals") or {}
    pw = m.get("p_wave") or {}
    st = m.get("st_segment") or {}
    tw = m.get("t_wave") or {}
    qw = m.get("q_wave") or {}
    qrs_a = m.get("qrs_amplitudes") or {}
    rs = m.get("r_s_ratio") or {}

    regular = hr.get("regular")
    p_present = bool(pw.get("present"))
    if p_present and regular:
        rhythm = "sinus"
    elif p_present and regular is False:
        rhythm = "sinus"  # P presente, RR irregular leve = ainda sinusal
    elif (not p_present) and regular:
        rhythm = "regular_sem_p"
    elif (not p_present) and (regular is False):
        rhythm = "irregular_sem_p"
    else:
        rhythm = "indeterminado"

    # Constrói leads dict no formato legacy
    leads_legacy: dict[str, dict] = {}
    for new_name in LEAD_NAMES:
        legacy_name = LEAD_NEW_TO_LEGACY.get(new_name, new_name)
        st_data = st.get(new_name, {}) or {}
        t_data = tw.get(new_name, {}) or {}
        q_data = qw.get(new_name, {}) or {}
        amp_data = qrs_a.get(new_name, {}) or {}

        st_value_mm = st_data.get("value_mm") or 0.0
        st_elevation = max(0.0, st_value_mm)
        st_depression = max(0.0, -st_value_mm)

        leads_legacy[legacy_name] = {
            "qrs_amplitude": amp_data.get("net_uv", 0.0) / UV_PER_MM,  # mm signed
            "qrs_amplitude_mm": amp_data.get("r_mm", 0.0) + amp_data.get("s_mm", 0.0),
            "r_amplitude_mm": amp_data.get("r_mm", 0.0),
            "s_amplitude_mm": amp_data.get("s_mm", 0.0),
            "st_elevation": float(st_elevation),
            "st_depression": float(st_depression),
            "t_wave": t_data.get("polarity", "unknown"),
            "t_wave_amplitude": (t_data.get("amplitude_uv", 0.0) or 0.0) / UV_PER_MM,
            "p_wave_present": p_present,
            "p_wave_amplitude": (pw.get("amplitude_uv", 0.0) or 0.0) / UV_PER_MM,
            "pr_depression": 0.0,  # não computado
            "has_delta_wave": False,  # não detectado ainda
            "has_rsr_prime": False,  # não detectado
            "has_wide_s": False,  # não detectado
            "has_q_wave": bool(q_data.get("pathological", False)),
            "has_monophasic_r": False,  # não detectado
            "has_prominent_s": amp_data.get("s_mm", 0.0) >= 5.0,  # heurística
            "has_u_wave": False,
            "st_morphology": "",
            "qrs_alternans": False,
            "st_segment_ratio": None,
        }

    legacy: dict = {
        "heart_rate": hr.get("mean_bpm"),
        "heart_rate_min": hr.get("min_bpm"),
        "heart_rate_max": hr.get("max_bpm"),
        "axis": ax.get("degrees"),
        "axis_classification": ax.get("classification"),
        "pr_interval": iv.get("pr_ms"),
        "qrs_duration": iv.get("qrs_ms"),
        "qt_interval": iv.get("qt_ms"),
        "qtc_bazett": iv.get("qtc_ms"),
        "rhythm": rhythm,
        "p_wave_present": p_present,
        "p_wave_duration": pw.get("duration_ms"),
        "p_wave_amplitude": pw.get("amplitude_uv"),
        "p_bifasic_v1": pw.get("bifasic_v1", False),
        "sex": sex,
        "leads": leads_legacy,
        "r_s_ratio_v1": rs.get("V1"),
        "r_s_ratio_v6": rs.get("V6"),
        "r_peaks_ms": m.get("r_peaks_ms", []),
        "quality_flags": m.get("quality", {}),
        "warnings_measure": m.get("warnings", []),
        # Compat: ST por derivação no formato legacy ({lead: {st_amplitude_mv, st_classification}})
        "st_segment": {
            LEAD_NEW_TO_LEGACY.get(name, name): {
                "st_amplitude_mv": (
                    (st.get(name, {}).get("value_uv") or 0.0) / 1000.0
                ),
                "st_classification": st.get(name, {}).get("class", "indeterminado"),
            }
            for name in LEAD_NAMES if name in st
        },
    }
    return legacy
