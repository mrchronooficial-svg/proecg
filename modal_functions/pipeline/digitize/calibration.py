"""
M3 — Calibração (módulo mais crítico).

Converte pixels → milímetros usando 4 métodos em cascata:
  1. Autocorrelação no grid — centro da imagem + FFT fallback
  2. OCR do cabeçalho (velocidade/ganho)
  3. Pulso de calibração
  4. Fallback final (assumir padrão BR)

Output: CalibrationResult com px_per_mm, velocidade, ganho, confiança.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from .constants import (
    AUTOCORR_HEIGHT_THRESH,
    AUTOCORR_PROMINENCE_THRESH,
    GAIN_DEFAULT,
    GRID_MAJOR_MM,
    GRID_MINOR_MM,
    PAPER_SPEED_DEFAULT,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Resultado da calibração pixel → mm."""
    px_per_mm_h: float          # pixels por mm horizontal
    px_per_mm_v: float          # pixels por mm vertical
    paper_speed: float          # mm/s (25 ou 50)
    gain: float                 # mm/mV (10, 5, ou 20)
    confidence: str             # "high", "medium", "low"
    method: str                 # qual método foi usado

    @property
    def mm_per_px_h(self) -> float:
        return 1.0 / self.px_per_mm_h if self.px_per_mm_h > 0 else 0.0

    @property
    def mm_per_px_v(self) -> float:
        return 1.0 / self.px_per_mm_v if self.px_per_mm_v > 0 else 0.0


# =========================================================================
# Utilitários de detecção de periodicidade
# =========================================================================

def _autocorr_spacing(signal_1d: np.ndarray,
                      h_thresh: float = AUTOCORR_HEIGHT_THRESH,
                      p_thresh: float = AUTOCORR_PROMINENCE_THRESH) -> Optional[float]:
    """Encontra espaçamento periódico via autocorrelação."""
    if len(signal_1d) < 20:
        return None

    sig = signal_1d.astype(np.float64)
    sig = sig - sig.mean()
    std = sig.std()
    if std < 1e-6:
        return None
    sig = sig / std

    n = len(sig)
    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[n - 1:]
    autocorr = autocorr / autocorr[0]

    peaks, props = find_peaks(autocorr, height=h_thresh, prominence=p_thresh, distance=3)

    if len(peaks) < 2:
        return None

    minor_spacing = float(peaks[0])
    if minor_spacing < 2:
        return None

    # Validar com major grid (5mm)
    expected_major = minor_spacing * (GRID_MAJOR_MM / GRID_MINOR_MM)
    for p in peaks:
        if abs(p - expected_major) / expected_major < 0.25:
            logger.info("Autocorrelação: minor=%.1f px, major confirmado ~%.1f px",
                         minor_spacing, expected_major)
            return minor_spacing

    logger.info("Autocorrelação: minor=%.1f px (major não confirmado)", minor_spacing)
    return minor_spacing


def _fft_spacing(signal_1d: np.ndarray) -> Optional[float]:
    """Encontra espaçamento periódico via FFT — mais robusto a ruído."""
    if len(signal_1d) < 40:
        return None

    sig = signal_1d.astype(np.float64)
    sig = sig - sig.mean()
    if sig.std() < 1e-6:
        return None

    n = len(sig)
    # FFT real
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1.0)  # em ciclos/pixel

    # Ignorar DC e frequências muito baixas (< 2 ciclos no sinal)
    min_freq_idx = max(2, int(2 * n / n))  # pelo menos 2 ciclos
    # Ignorar frequências muito altas (espaçamento < 3px)
    max_freq_idx = min(len(freqs) - 1, int(n / 3))

    if min_freq_idx >= max_freq_idx:
        return None

    fft_slice = fft[min_freq_idx:max_freq_idx]
    freqs_slice = freqs[min_freq_idx:max_freq_idx]

    if len(fft_slice) == 0:
        return None

    # Encontrar pico dominante
    peaks, props = find_peaks(fft_slice, height=fft_slice.max() * 0.3, prominence=fft_slice.max() * 0.1)

    if len(peaks) == 0:
        # Sem picos claros — pegar o máximo absoluto
        peak_idx = np.argmax(fft_slice)
    else:
        # Pegar o pico mais forte
        peak_idx = peaks[np.argmax(props["peak_heights"])]

    dominant_freq = freqs_slice[peak_idx]
    if dominant_freq <= 0:
        return None

    spacing = 1.0 / dominant_freq

    # Sanidade: espaçamento entre 3 e 30 px
    if not (3.0 <= spacing <= 30.0):
        return None

    # Verificar se há harmônica a 5× (major grid)
    # ou sub-harmônica a 1/5 (minor = 1mm, major = 5mm)
    for harmonics_name, factor in [("5x", 5.0), ("1/5", 0.2)]:
        target_freq = dominant_freq * factor
        if target_freq < freqs_slice[0] or target_freq > freqs_slice[-1]:
            continue
        idx = np.argmin(np.abs(freqs_slice - target_freq))
        if fft_slice[idx] > fft_slice.max() * 0.15:
            logger.info("FFT: spacing=%.1f px, harmônica %s confirmada", spacing, harmonics_name)
            return spacing

    logger.info("FFT: spacing=%.1f px (sem harmônica confirmada)", spacing)
    return spacing


def _try_spacing_detection(profile: np.ndarray, axis_name: str) -> Optional[float]:
    """Tenta detectar espaçamento com múltiplos métodos progressivos."""
    # 1. Autocorrelação com thresholds originais
    spacing = _autocorr_spacing(profile)
    if spacing is not None:
        logger.info("Calibração %s: autocorrelação original = %.1f px", axis_name, spacing)
        return spacing

    # 2. Autocorrelação com thresholds relaxados
    spacing = _autocorr_spacing(profile, h_thresh=0.05, p_thresh=0.01)
    if spacing is not None:
        logger.info("Calibração %s: autocorrelação relaxada = %.1f px", axis_name, spacing)
        return spacing

    # 3. Autocorrelação muito relaxada
    spacing = _autocorr_spacing(profile, h_thresh=0.02, p_thresh=0.005)
    if spacing is not None:
        logger.info("Calibração %s: autocorrelação muito relaxada = %.1f px", axis_name, spacing)
        return spacing

    # 4. FFT
    spacing = _fft_spacing(profile)
    if spacing is not None:
        logger.info("Calibração %s: FFT = %.1f px", axis_name, spacing)
        return spacing

    return None


# =========================================================================
# Método 1: Autocorrelação no grid (com recorte central)
# =========================================================================

def _calibrate_autocorrelation(grid_mask: np.ndarray) -> Optional[CalibrationResult]:
    """Método 1: Autocorrelação na mask do grid → px/mm.

    Tenta primeiro na região central (60% interno) para evitar bordas ruidosas,
    depois na imagem toda se falhar.
    """
    H, W = grid_mask.shape

    # Definir região central (recortar 20% de cada lado)
    margin_h = int(H * 0.20)
    margin_w = int(W * 0.20)
    center_mask = grid_mask[margin_h:H - margin_h, margin_w:W - margin_w]

    # Tentar primeiro no centro
    spacing_h = None
    spacing_v = None

    if center_mask.sum() > 50:
        h_profile = center_mask.astype(np.float64).sum(axis=0)
        spacing_h = _try_spacing_detection(h_profile, "H_centro")

        v_profile = center_mask.astype(np.float64).sum(axis=1)
        spacing_v = _try_spacing_detection(v_profile, "V_centro")

    # Se centro falhou, tentar imagem toda
    if spacing_h is None and spacing_v is None:
        logger.info("Centro falhou, tentando imagem toda")
        h_profile = grid_mask.astype(np.float64).sum(axis=0)
        spacing_h = _try_spacing_detection(h_profile, "H_total")

        v_profile = grid_mask.astype(np.float64).sum(axis=1)
        spacing_v = _try_spacing_detection(v_profile, "V_total")

    if spacing_h is None and spacing_v is None:
        return None

    # Se só um eixo funcionou, assumir grid quadrado
    if spacing_h is None:
        spacing_h = spacing_v
        confidence = "medium"
    elif spacing_v is None:
        spacing_v = spacing_h
        confidence = "medium"
    else:
        ratio = spacing_h / spacing_v if spacing_v > 0 else 999
        if 0.7 < ratio < 1.4:
            confidence = "high"
        else:
            confidence = "medium"
            logger.warning("Grid não-quadrado: h=%.1f px, v=%.1f px (ratio=%.2f)",
                           spacing_h, spacing_v, ratio)

    px_per_mm_h = spacing_h
    px_per_mm_v = spacing_v

    # Sanidade: px_per_mm deve estar entre 1 e 15
    if not (1.0 <= px_per_mm_h <= 15.0) or not (1.0 <= px_per_mm_v <= 15.0):
        logger.warning("px_per_mm fora do range plausível: h=%.1f, v=%.1f",
                       px_per_mm_h, px_per_mm_v)
        return None

    return CalibrationResult(
        px_per_mm_h=px_per_mm_h,
        px_per_mm_v=px_per_mm_v,
        paper_speed=PAPER_SPEED_DEFAULT,
        gain=GAIN_DEFAULT,
        confidence=confidence,
        method="autocorrelation",
    )


# =========================================================================
# Método 1b: Calibração por área do trace
# =========================================================================

def _calibrate_trace_extent(trace_mask: np.ndarray) -> Optional[CalibrationResult]:
    """Estima px/mm pela extensão horizontal do trace.

    Se o trace ocupa uma faixa larga da imagem, podemos estimar:
    - Largura do trace ≈ 10s × 25mm/s = 250mm (layout 3×4 com 4 colunas de 2.5s)
    - px_per_mm = largura_do_trace / 250
    """
    if trace_mask is None or trace_mask.sum() < 50:
        return None

    H, W = trace_mask.shape
    h_profile = trace_mask.astype(np.float64).sum(axis=0)

    # Encontrar extensão horizontal do trace (onde há pixels)
    nonzero_cols = np.where(h_profile > 0)[0]
    if len(nonzero_cols) < W * 0.3:  # Trace deve cobrir pelo menos 30% da largura
        return None

    trace_left = nonzero_cols[0]
    trace_right = nonzero_cols[-1]
    trace_width = trace_right - trace_left

    if trace_width < W * 0.3:
        return None

    # Assumir que essa largura = 250mm (10s a 25mm/s)
    px_per_mm_h = trace_width / 250.0
    px_per_mm_v = px_per_mm_h  # Grid quadrado

    if not (1.0 <= px_per_mm_h <= 15.0):
        return None

    logger.info("Calibração por extensão do trace: largura=%d px, px/mm=%.2f",
                trace_width, px_per_mm_h)

    return CalibrationResult(
        px_per_mm_h=px_per_mm_h,
        px_per_mm_v=px_per_mm_v,
        paper_speed=PAPER_SPEED_DEFAULT,
        gain=GAIN_DEFAULT,
        confidence="medium",
        method="trace_extent",
    )


# =========================================================================
# Método 2: OCR do cabeçalho (simplificado para MVP)
# =========================================================================

def _calibrate_ocr(text_mask: np.ndarray, image: np.ndarray,
                    trace_mask: Optional[np.ndarray] = None) -> Optional[CalibrationResult]:
    """Método 2: OCR para extrair velocidade e ganho do cabeçalho.

    Busca padrões como "25 mm/s", "25 mm/seg", "10 mm/mV" no texto
    do cabeçalho da imagem.
    """
    import re

    H, W = image.shape[:2]

    # Região do cabeçalho: top 25% da imagem (onde ficam dados do paciente/aparelho)
    header_h = int(H * 0.25)
    header_img = image[:header_h, :]

    # Converter para uint8 se necessário
    if header_img.dtype == np.float32 or header_img.dtype == np.float64:
        header_uint8 = (header_img * 255).clip(0, 255).astype(np.uint8)
    else:
        header_uint8 = header_img

    try:
        import easyocr
        reader = easyocr.Reader(["pt", "en"], gpu=False, verbose=False)
        results = reader.readtext(header_uint8, detail=0, paragraph=False)
        text = " ".join(results).lower()
        logger.info("OCR header: '%s'", text[:200])
    except Exception as e:
        logger.warning("OCR falhou: %s", e)
        return None

    if not text.strip():
        return None

    # Extrair velocidade: "25 mm/s", "25mm/s", "25 mm/seg", "50 mm/s"
    speed = None
    speed_patterns = [
        r"(\d{2,3})\s*mm\s*/\s*s(?:eg)?",
        r"(\d{2,3})\s*mm/s",
        r"vel[a-z]*\s*[:=]?\s*(\d{2,3})",
    ]
    for pattern in speed_patterns:
        m = re.search(pattern, text)
        if m:
            val = int(m.group(1))
            if val in (25, 50):
                speed = val
                logger.info("OCR: velocidade = %d mm/s", speed)
                break

    # Extrair ganho: "10 mm/mV", "10mm/mv", "5 mm/mV", "20 mm/mV"
    gain = None
    gain_patterns = [
        r"(\d{1,2})\s*mm\s*/\s*mv",
        r"ganho\s*[:=]?\s*(\d{1,2})",
        r"(\d{1,2})\s*mm/mv",
    ]
    for pattern in gain_patterns:
        m = re.search(pattern, text)
        if m:
            val = int(m.group(1))
            if val in (5, 10, 20):
                gain = val
                logger.info("OCR: ganho = %d mm/mV", gain)
                break

    if speed is None and gain is None:
        logger.info("OCR: velocidade/ganho não encontrados no texto")
        return None

    # Usar defaults para o que não encontrou
    if speed is None:
        speed = PAPER_SPEED_DEFAULT
    if gain is None:
        gain = GAIN_DEFAULT

    # Calcular px_per_mm usando trace extent se disponível
    if trace_mask is not None and trace_mask.sum() > 50:
        t_h = trace_mask.astype(np.float64).sum(axis=0)
        t_h_nz = np.where(t_h > 0)[0]
        if len(t_h_nz) > 10:
            trace_left = np.percentile(t_h_nz, 2)
            trace_right = np.percentile(t_h_nz, 98)
            trace_width = trace_right - trace_left
            total_mm = 10.0 * speed  # 10s × speed mm/s
            px_per_mm = trace_width / total_mm
            if 1.0 <= px_per_mm <= 15.0:
                logger.info("OCR + trace: px/mm=%.2f (trace_width=%.0f, total_mm=%.0f)",
                            px_per_mm, trace_width, total_mm)
                return CalibrationResult(
                    px_per_mm_h=px_per_mm,
                    px_per_mm_v=px_per_mm,
                    paper_speed=speed,
                    gain=gain,
                    confidence="high",
                    method="ocr",
                )

    # Sem trace para calcular px_per_mm — fallback com largura da imagem
    usable_width = W * 0.85
    total_mm = 10.0 * speed
    px_per_mm = usable_width / total_mm

    return CalibrationResult(
        px_per_mm_h=px_per_mm,
        px_per_mm_v=px_per_mm,
        paper_speed=speed,
        gain=gain,
        confidence="medium",
        method="ocr",
    )


# =========================================================================
# Método 3: Pulso de calibração
# =========================================================================

def _calibrate_pulse(trace_mask: np.ndarray,
                     px_per_mm_v: Optional[float] = None) -> Optional[CalibrationResult]:
    """Método 3: Detectar pulso de calibração (quadradinho 1mV)."""
    return None


# =========================================================================
# Método 4: Fallback final
# =========================================================================

def _calibrate_fallback(image_width: int, image_height: int) -> CalibrationResult:
    """Método 4: Assumir padrão BR (250mm de largura total)."""
    usable_width = image_width * 0.90
    px_per_mm_h = usable_width / 250.0
    px_per_mm_v = px_per_mm_h

    logger.warning("Usando fallback: assumindo 250mm de largura (padrão BR 25mm/s)")

    return CalibrationResult(
        px_per_mm_h=px_per_mm_h,
        px_per_mm_v=px_per_mm_v,
        paper_speed=PAPER_SPEED_DEFAULT,
        gain=GAIN_DEFAULT,
        confidence="low",
        method="fallback",
    )


# =========================================================================
# API pública
# =========================================================================

def calibrate(
    masks: dict,
    image: np.ndarray,
) -> CalibrationResult:
    """Calibra pixels → mm usando métodos em cascata."""
    H, W = image.shape[:2]
    grid_mask = masks.get("grid")
    text_mask = masks.get("text")
    trace_mask = masks.get("trace")

    # Método 1: Autocorrelação no grid (centro + FFT)
    if grid_mask is not None and grid_mask.sum() > 100:
        result = _calibrate_autocorrelation(grid_mask)
        if result is not None:
            logger.info("Calibração via autocorrelação: px/mm_h=%.2f, px/mm_v=%.2f (%s)",
                        result.px_per_mm_h, result.px_per_mm_v, result.confidence)
            return result

    # Método 1b: Extensão do trace
    if trace_mask is not None:
        result = _calibrate_trace_extent(trace_mask)
        if result is not None:
            return result

    # Método 2: OCR do cabeçalho
    if text_mask is not None:
        result = _calibrate_ocr(text_mask, image, trace_mask)
        if result is not None:
            logger.info("Calibração via OCR: speed=%d mm/s, gain=%d mm/mV, px/mm=%.2f",
                        result.paper_speed, result.gain, result.px_per_mm_h)
            return result

    # Método 3: Pulso de calibração
    if trace_mask is not None:
        result = _calibrate_pulse(trace_mask)
        if result is not None:
            return result

    # Método 4: Fallback final
    return _calibrate_fallback(W, H)
