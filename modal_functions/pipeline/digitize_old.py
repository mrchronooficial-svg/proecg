"""
Digitalização de ECG — ProECG

Converte foto de ECG em papel → sinal digital de 12 derivações.

Camada principal: Open-ECG-Digitizer (Ahus-AIM) — U-Net segmentation
Fallback: abordagem clássica (binarização adaptativa + varredura vertical)

Output: numpy array (12, N) a 500 Hz
Ordem das derivações: I, II, III, aVR, aVL, aVF, V1–V6
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SAMPLING_RATE = 500  # Hz
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SHORT_SIGNAL_SEC = 2.5
LONG_SIGNAL_SEC = 10.0
OPEN_ECG_DIGITIZER_PATH = "/root/open_ecg_digitizer"
OPEN_ECG_CONFIG = "src/config/inference_wrapper.yml"

# Cache do modelo (carrega uma vez por container Modal)
_wrapper_cache = None


# =========================================================================
# API pública
# =========================================================================

def digitize_ecg(image: Image.Image) -> np.ndarray:
    """Converte foto de ECG em papel → sinal digital (12, N) a 500 Hz.

    Tenta Open-ECG-Digitizer primeiro; se falhar, usa fallback clássico.

    Args:
        image: PIL Image (RGB) da foto do ECG.

    Returns:
        np.ndarray shape (12, N) com sinal em mV.

    Raises:
        RuntimeError: se ambos os métodos falharem.
    """
    try:
        signal = _digitize_with_open_ecg_digitizer(image)
        if signal is not None and signal.shape[0] == 12 and signal.shape[1] > 0:
            logger.info("Open-ECG-Digitizer: sucesso — shape %s", signal.shape)
            return signal
    except Exception as e:
        logger.warning("Open-ECG-Digitizer falhou: %s. Tentando fallback clássico.", e)

    try:
        signal = _digitize_classic_fallback(image)
        if signal is not None and signal.shape[0] == 12 and signal.shape[1] > 0:
            logger.info("Fallback clássico: sucesso — shape %s", signal.shape)
            return signal
    except Exception as e:
        logger.error("Fallback clássico também falhou: %s", e)

    raise RuntimeError(
        "Não foi possível digitalizar o ECG. "
        "Verifique a qualidade da foto e tente novamente."
    )


# =========================================================================
# Open-ECG-Digitizer (U-Net segmentation)
# =========================================================================

def _get_inference_wrapper():
    """Carrega (ou retorna do cache) o InferenceWrapper do Open-ECG-Digitizer."""
    global _wrapper_cache
    if _wrapper_cache is not None:
        return _wrapper_cache

    import torch

    repo_path = Path(OPEN_ECG_DIGITIZER_PATH)

    # Adicionar ao sys.path para importar src.* do Open-ECG-Digitizer
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Os caminhos no config YAML são relativos ao repo (ex: ./weights/...)
    # Precisamos mudar o cwd temporariamente para que resolvam corretamente.
    original_cwd = os.getcwd()
    os.chdir(repo_str)

    try:
        from src.config.default import get_cfg
        from src.utils import import_class_from_path

        cfg = get_cfg(OPEN_ECG_CONFIG)

        # Instanciar via import dinâmico (mesmo padrão do digitize.py do repo)
        wrapper_class = import_class_from_path(cfg.MODEL.class_path)
        wrapper = wrapper_class(**cfg.MODEL.KWARGS)
        wrapper.eval()
    finally:
        os.chdir(original_cwd)

    _wrapper_cache = wrapper
    return wrapper


def _digitize_with_open_ecg_digitizer(image: Image.Image) -> Optional[np.ndarray]:
    """Usa o Open-ECG-Digitizer (Ahus-AIM) para extrair sinal de 12 derivações.

    Pipeline:
      1. U-Net segmenta imagem em 4 classes (grid, texto, sinal, fundo)
      2. Correção de perspectiva via grid detectado
      3. Estimativa de espaçamento de pixels (mm/pixel) via autocorrelação
      4. Extração de traços do mapa de probabilidade do sinal
      5. Identificação de layout e mapeamento para 12 derivações canônicas
      6. Conversão de pixel → mV usando calibração do grid
    """
    repo_path = Path(OPEN_ECG_DIGITIZER_PATH)
    if not repo_path.exists():
        logger.warning("Open-ECG-Digitizer não encontrado em %s", OPEN_ECG_DIGITIZER_PATH)
        return None

    weights_dir = repo_path / "weights"
    if not weights_dir.exists() or not any(weights_dir.glob("*.pt")):
        logger.warning("Pesos do Open-ECG-Digitizer não encontrados em %s", weights_dir)
        return None

    import torch

    wrapper = _get_inference_wrapper()

    # --- Converter PIL Image → tensor [1, 3, H, W] ---
    # Mesmo formato usado por torchvision.io.decode_image (uint8)
    img_np = np.array(image)  # (H, W, 3) uint8
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

    # --- Rodar inferência ---
    with torch.no_grad():
        result = wrapper(img_tensor, layout_should_include_substring=None)

    # --- Extrair canonical_lines: tensor (12, W) ---
    # Valores em coordenadas de pixel (y-position no mapa de sinal alinhado)
    canonical = result.get("signal", {}).get("canonical_lines", None)
    if canonical is None:
        logger.warning("Open-ECG-Digitizer não detectou derivações canônicas.")
        return None

    canonical_np = canonical.squeeze().cpu().numpy()  # (12, W) ou (n_leads, W)
    if canonical_np.ndim == 1:
        canonical_np = canonical_np[np.newaxis, :]

    n_leads, n_points = canonical_np.shape
    logger.info("canonical_lines shape: (%d, %d)", n_leads, n_points)

    if n_leads == 0 or n_points < 2:
        return None

    # --- Converter pixel → mV ---
    # pixel_spacing_mm retorna mm/pixel para x e y
    # ECG padrão: 10 mm/mV (sensibilidade vertical)
    # mV = deslocamento_pixels * mm_per_pixel_y / 10
    pixel_spacing = result.get("pixel_spacing_mm", {})
    mm_per_pixel_y = pixel_spacing.get("y", None)
    if hasattr(mm_per_pixel_y, "item"):
        mm_per_pixel_y = mm_per_pixel_y.item()  # tensor → float

    if mm_per_pixel_y is not None and mm_per_pixel_y > 0:
        signal_mv = canonical_np * mm_per_pixel_y / 10.0
    else:
        # Sem calibração: normalizar heuristicamente
        signal_mv = canonical_np.copy()
        for i in range(signal_mv.shape[0]):
            lead = signal_mv[i]
            valid = ~np.isnan(lead)
            if valid.sum() < 2:
                continue
            lead[valid] = lead[valid] - np.nanmedian(lead)
            peak = np.nanpercentile(np.abs(lead[valid]), 99)
            if peak > 0:
                lead[valid] = lead[valid] * (1.5 / peak)
            signal_mv[i] = lead

    # --- Montar array (12, target_len) a 500 Hz ---
    target_len = SAMPLING_RATE * int(LONG_SIGNAL_SEC)  # 5000

    # Pad para 12 derivações se menos foram detectadas
    result_array = np.full((12, target_len), np.nan, dtype=np.float64)

    for i in range(min(n_leads, 12)):
        sig = signal_mv[i]
        valid = ~np.isnan(sig)
        if valid.sum() < 2:
            result_array[i] = 0.0
            continue

        # Interpolar NaN gaps
        x_valid = np.where(valid)[0]
        f = interp1d(
            x_valid, sig[x_valid],
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        sig_clean = f(np.arange(n_points))

        # Resample para target_len
        if len(sig_clean) == target_len:
            result_array[i] = sig_clean
        else:
            x_old = np.linspace(0, 1, len(sig_clean))
            x_new = np.linspace(0, 1, target_len)
            f2 = interp1d(x_old, sig_clean, kind="linear", fill_value="extrapolate")
            result_array[i] = f2(x_new)

    # Substituir NaN residuais por zero
    result_array = np.nan_to_num(result_array, nan=0.0)

    leads_found = sum(1 for i in range(12) if np.any(result_array[i] != 0))
    logger.info("Open-ECG-Digitizer: %d/12 derivações extraídas.", leads_found)

    if leads_found == 0:
        return None

    return result_array


# =========================================================================
# Fallback clássico (sem modelo)
# =========================================================================

def _digitize_classic_fallback(image: Image.Image) -> Optional[np.ndarray]:
    """Abordagem clássica: binarização → remoção de grid → segmentação → varredura.

    Mais simples e menos preciso que o Open-ECG-Digitizer, mas funciona
    como fallback quando o modelo não está disponível ou falha.

    Pipeline:
      1. Converter para grayscale
      2. Binarização adaptativa (Otsu + adaptiveThreshold)
      3. Remover grid (filtro morfológico)
      4. Segmentar em 12 derivações (dividir em 4 linhas × 3 colunas)
      5. Extrair sinal por varredura vertical (center of mass do traço)
      6. Resample para 500 Hz
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

    target_per_lead = int(SHORT_SIGNAL_SEC * SAMPLING_RATE)  # 1250
    target_total = int(LONG_SIGNAL_SEC * SAMPLING_RATE)  # 5000

    result = np.zeros((12, target_total), dtype=np.float64)
    for i, lead_name in enumerate(LEAD_ORDER):
        if lead_name in signals:
            sig = signals[lead_name]
            resampled = _resample_signal(sig, target_per_lead)
            result[i, :target_per_lead] = resampled
            result[i, target_per_lead:] = resampled[-1]

    result = _normalize_to_mv(result)

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
