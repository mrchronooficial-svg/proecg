"""
Digitalização de ECG — ProECG

Converte foto de ECG em papel → sinal digital de 12 derivações.

Camada principal: Open-ECG-Digitizer (Ahus-AIM) — U-Net segmentation
Fallback: abordagem clássica (binarização adaptativa + varredura vertical)

Output: numpy array (12, N) a 500 Hz
Ordem das derivações: DI, DII, DIII, aVR, aVL, aVF, V1–V6
"""

from __future__ import annotations

import logging
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
    # Tentar Open-ECG-Digitizer
    try:
        signal = _digitize_with_open_ecg_digitizer(image)
        if signal is not None and signal.shape[0] == 12 and signal.shape[1] > 0:
            logger.info("Open-ECG-Digitizer: sucesso — shape %s", signal.shape)
            return signal
    except Exception as e:
        logger.warning("Open-ECG-Digitizer falhou: %s. Tentando fallback clássico.", e)

    # Fallback clássico
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

def _digitize_with_open_ecg_digitizer(image: Image.Image) -> Optional[np.ndarray]:
    """Usa o Open-ECG-Digitizer (Ahus-AIM) para extrair sinal de 12 derivações.

    Pipeline do Open-ECG-Digitizer:
      1. U-Net segmenta imagem em 4 classes (grid, texto, sinal, fundo)
      2. Correção de perspectiva via grid detectado
      3. Estimativa de espaçamento de pixels (mm/pixel) via autocorrelação
      4. Extração de traços do mapa de probabilidade do sinal
      5. Identificação de layout e mapeamento para derivações padrão
      6. Conversão para microvolts calibrados
    """
    repo_path = Path(OPEN_ECG_DIGITIZER_PATH)
    if not repo_path.exists():
        logger.warning("Open-ECG-Digitizer não encontrado em %s", OPEN_ECG_DIGITIZER_PATH)
        return None

    weights_dir = repo_path / "weights"
    if not weights_dir.exists() or not any(weights_dir.glob("*.pt")):
        logger.warning("Pesos do Open-ECG-Digitizer não encontrados em %s", weights_dir)
        return None

    # Adicionar ao path para importar módulos do Open-ECG-Digitizer
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    import torch
    from src.config.default import get_cfg
    from src.model.inference_wrapper import InferenceWrapper

    # --- 1. Carregar config e modelo ---
    config_path = str(repo_path / OPEN_ECG_CONFIG)
    cfg = get_cfg(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InferenceWrapper(
        config=cfg.MODEL.KWARGS.config,
        device=device,
        resample_size=3000,
        rotate_on_resample=True,
        apply_dewarping=False,
    )
    model.eval()

    # --- 2. Converter PIL Image → tensor [1, 3, H, W] float32 ---
    img_np = np.array(image)  # (H, W, 3) uint8
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

    # --- 3. Rodar inferência ---
    with torch.no_grad():
        result = model(img_tensor)

    # --- 4. Extrair sinais das derivações canônicas ---
    canonical_lines = result.get("signal", {}).get("canonical_lines", {})
    if not canonical_lines:
        logger.warning("Open-ECG-Digitizer não detectou derivações canônicas.")
        return None

    # --- 5. Obter calibração (pixel spacing) ---
    pixel_spacing = result.get("pixel_spacing_mm", {})
    mm_per_pixel_y = pixel_spacing.get("average_pixel_per_mm", None)

    # ECG padrão: 10 mm/mV → converter de pixel para mV
    # Se mm_per_pixel_y disponível: mV = pixel_displacement / (mm_per_pixel_y * 10)
    # Senão: usar normalização heurística
    has_calibration = mm_per_pixel_y is not None and mm_per_pixel_y > 0

    # --- 6. Montar array (12, N) a 500 Hz ---
    target_len = SAMPLING_RATE * int(LONG_SIGNAL_SEC)  # 5000 amostras (10s)

    result_array = np.zeros((12, target_len), dtype=np.float64)
    leads_found = 0

    for i, lead_name in enumerate(LEAD_ORDER):
        if lead_name not in canonical_lines:
            continue

        sig = canonical_lines[lead_name]
        if hasattr(sig, "numpy"):
            sig = sig.cpu().numpy()
        sig = np.asarray(sig, dtype=np.float64).flatten()

        if len(sig) < 2:
            continue

        # Converter para mV
        if has_calibration:
            # Open-ECG-Digitizer retorna em uV quando calibrado
            sig_mv = sig / 1000.0
        else:
            # Sem calibração: normalizar heuristicamente
            sig_mv = sig.copy()
            sig_mv = sig_mv - np.median(sig_mv)
            peak = np.percentile(np.abs(sig_mv), 99)
            if peak > 0:
                sig_mv = sig_mv * (1.5 / peak)

        # Resample para target_len
        if len(sig_mv) == target_len:
            result_array[i] = sig_mv
        else:
            x_old = np.linspace(0, 1, len(sig_mv))
            x_new = np.linspace(0, 1, target_len)
            f = interp1d(x_old, sig_mv, kind="linear", fill_value="extrapolate")
            result_array[i] = f(x_new)

        leads_found += 1

    if leads_found == 0:
        return None

    logger.info("Open-ECG-Digitizer: %d/12 derivações extraídas.", leads_found)
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

    # --- 1. Grayscale ---
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    h, w = gray.shape

    # --- 2. Binarização adaptativa ---
    # Inverter: traço do ECG fica branco
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10,
    )

    # --- 3. Remover grid ---
    binary = _remove_grid(binary)

    # --- 4. Segmentar em 12 derivações ---
    # Layout 4 colunas × 3 linhas (padrão internacional)
    lead_layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    # Margens estimadas (10% superior/inferior para header/footer)
    margin_top = int(h * 0.12)
    margin_bottom = int(h * 0.05)
    usable_h = h - margin_top - margin_bottom

    # Margem lateral (5% de cada lado)
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

    # --- 5. Montar array (12, N) ---
    target_per_lead = int(SHORT_SIGNAL_SEC * SAMPLING_RATE)  # 1250
    target_total = int(LONG_SIGNAL_SEC * SAMPLING_RATE)  # 5000

    result = np.zeros((12, target_total), dtype=np.float64)
    for i, lead_name in enumerate(LEAD_ORDER):
        if lead_name in signals:
            sig = signals[lead_name]
            resampled = _resample_signal(sig, target_per_lead)
            result[i, :target_per_lead] = resampled
            result[i, target_per_lead:] = resampled[-1]

    # Normalizar para escala aproximada de mV
    result = _normalize_to_mv(result)

    return result


def _remove_grid(binary: np.ndarray) -> np.ndarray:
    """Remove linhas de grid do ECG usando operações morfológicas."""
    # Remover linhas horizontais finas (grid)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    binary = cv2.subtract(binary, horizontal_lines)

    # Remover linhas verticais finas (grid)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    binary = cv2.subtract(binary, vertical_lines)

    # Limpar ruído residual
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)

    return binary


def _extract_signal_from_roi(roi: np.ndarray) -> np.ndarray:
    """Extrai sinal 1D de uma ROI binarizada por varredura vertical.

    Para cada coluna x, calcula o centro de massa dos pixels brancos.
    Onde não há pixels, interpola linearmente.
    """
    h, w = roi.shape
    signal = np.full(w, np.nan, dtype=np.float64)

    for x in range(w):
        col = roi[:, x]
        white_pixels = np.where(col > 0)[0]
        if len(white_pixels) > 0:
            weights = col[white_pixels].astype(np.float64)
            center = np.average(white_pixels, weights=weights)
            signal[x] = h - center

    # Interpolar gaps (NaN)
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

    # Suavizar com filtro mediano para remover ruído
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
    """Normaliza sinal de pixels para escala aproximada de mV.

    ECG normal: QRS ~1-2 mV, onda T ~0.3 mV.
    Normaliza para que o pico mediano do QRS fique em ~1.0 mV.
    """
    for i in range(signal.shape[0]):
        lead = signal[i]
        lead = lead - np.median(lead)
        peak = np.percentile(np.abs(lead), 99)
        if peak > 0:
            lead = lead * (1.5 / peak)
        signal[i] = lead

    return signal
