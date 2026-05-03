"""
Adapter Stenhede (Open-ECG-Digitizer) → ProECG.
================================================

Faz a ponte entre o pipeline ProECG (que já entrega imagem undistorted +
calibração) e o motor de extração de sinal do Open-ECG-Digitizer
(`vendor/open_ecg_digitizer/`):

  • U-Net (3→4 canais: grid, text+background, signal, background) gera o
    mapa de probabilidade do traço (canal 2).
  • SignalExtractor desenha as linhas do traçado a partir desse mapa via
    connected components + matching húngaro.
  • A conversão pixel→µV usa o `uv_per_pixel` do nosso calibrator (não
    precisa do PixelSizeFinder do Stenhede — já temos px_per_mm).

API pública:
  • get_unet()                          — modelo U-Net pré-carregado (cached)
  • extract_signal_probabilities(image) — imagem BGR → mapa (H, W) [0,1]
  • extract_lines_from_cell(prob_cell)  — recorte de cell → linha (W,) px
  • convert_line_to_uv(line, baseline, uv_per_px) → array µV

TODO(licença): o Open-ECG-Digitizer está em
`modal_functions/vendor/open_ecg_digitizer/` como dependência vendorizada.
Confirmar e documentar a licença do projeto antes de distribuir o ProECG
publicamente — a licença do Open-ECG-Digitizer pode exigir atribuição
e/ou compatibilidade de licença do nosso código.
"""

from __future__ import annotations

import logging
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Diretório do repo vendorizado (modal_functions/vendor/open_ecg_digitizer/)
_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "open_ecg_digitizer"
_VENDOR_WEIGHTS = _VENDOR_ROOT / "weights" / "unet_weights_07072025.pt"
_VENDOR_LEAD_UNET_WEIGHTS = (
    _VENDOR_ROOT / "weights" / "lead_name_unet_weights_07072025.pt"
)
_VENDOR_LAYOUTS_YAML = _VENDOR_ROOT / "src" / "config" / "lead_layouts_all.yml"

# Constantes do U-Net Ahus-AIM (devem bater com config/unet.yml do vendor)
_UNET_KWARGS = {
    "num_in_channels": 3,
    "num_out_channels": 4,
    "dims": [32, 64, 128, 256, 320, 320, 320, 320],
    "depth": 2,
}
# Lead-name U-Net (1 canal text-prob → 13 canais, 12 leads + 1 outro/grid)
_LEAD_UNET_KWARGS = {
    "num_in_channels": 1,
    "num_out_channels": 13,
    "dims": [32, 64, 128, 256, 256],
    "depth": 2,
}
# Canais do U-Net principal (4 classes): grid=0, text+background=1, signal=2, background=3
_GRID_CLASS = 0
_TEXT_BG_CLASS = 1
_SIGNAL_CLASS = 2
_BG_CLASS = 3
_DEFAULT_MAX_DIM = 3000

# Ordem canônica dos 12 leads (mesma do LeadIdentifier)
LEAD_CHANNEL_ORDER: list[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def _ensure_vendor_on_path() -> None:
    """Coloca o root do repo vendorizado no `sys.path` para que
    `from src.model.unet import UNet` funcione."""
    p = str(_VENDOR_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


@lru_cache(maxsize=1)
def get_unet() -> torch.nn.Module:
    """Carrega a U-Net do Stenhede com pesos pré-treinados (cacheado).

    - Strip de prefixo `_orig_mod.` nas chaves (pesos foram salvos com
      `torch.compile`).
    - Move para CUDA se disponível, senão CPU.
    - Modelo em modo `eval()`.
    """
    _ensure_vendor_on_path()
    from src.model.unet import UNet  # type: ignore

    if not _VENDOR_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Pesos da U-Net Stenhede não encontrados em {_VENDOR_WEIGHTS}. "
            "Execute `git lfs pull` em modal_functions/vendor/open_ecg_digitizer/."
        )

    model: torch.nn.Module = UNet(**_UNET_KWARGS)
    state_dict = torch.load(
        str(_VENDOR_WEIGHTS), map_location="cpu", weights_only=True,
    )
    if isinstance(state_dict, tuple):
        state_dict = state_dict[0]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info(
        "U-Net Stenhede carregada: %s (%.1f MB) em device=%s",
        _VENDOR_WEIGHTS.name,
        _VENDOR_WEIGHTS.stat().st_size / (1024 * 1024),
        device.type,
    )
    return model


@lru_cache(maxsize=1)
def _get_signal_extractor():  # type: ignore[no-untyped-def]
    """Retorna instância do SignalExtractor do Stenhede (com kwargs default)."""
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore
    return SignalExtractor()


def extract_signal_probabilities(
    image_bgr: np.ndarray,
    max_dim: int = _DEFAULT_MAX_DIM,
) -> np.ndarray:
    """Roda a U-Net na imagem undistorted e devolve o mapa de
    probabilidade do canal "signal" na resolução ORIGINAL.

    Args:
        image_bgr: imagem undistorted em BGR uint8, shape `(H, W, 3)`.
        max_dim: max(H, W) usado pra inferência. Acima disso a imagem é
            redimensionada antes do U-Net (depois o mapa é interpolado
            de volta pra resolução original).

    Returns:
        `np.ndarray` float32, shape `(H, W)`, valores em `[0, 1]`. Após
        `softmax → process_sparse_prob` (subtrai média, clamp 0, divide
        pelo max) — pixels de traço têm valores altos, fundo ~0.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Esperado imagem BGR (H, W, 3) uint8; recebi shape={image_bgr.shape}"
        )

    h_orig, w_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1. Resize se max(H, W) > max_dim (o U-Net foi calibrado pra ~3000 px)
    max_side = max(h_orig, w_orig)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_h, new_w = int(round(h_orig * scale)), int(round(w_orig * scale))
        image_rgb = cv2.resize(
            image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA,
        )
        logger.debug(
            "U-Net input resize: %dx%d -> %dx%d (scale=%.3f)",
            w_orig, h_orig, new_w, new_h, scale,
        )

    # 2. uint8 [0,255] -> float32 normalizado [0,1] via min-max
    img_float = image_rgb.astype(np.float32)
    img_min = float(img_float.min())
    img_max = float(img_float.max())
    img_float = (img_float - img_min) / max(img_max - img_min, 1e-8)

    # 3. (H, W, 3) -> (1, 3, H, W)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    # 4. Inferência
    t_inf = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                 # (1, 4, H, W)
        probs = torch.softmax(logits, dim=1)
        signal_prob = probs[:, _SIGNAL_CLASS, :, :]   # (1, H, W)

        # process_sparse_prob: subtrai média, clamp(min=0), divide pelo max
        signal_prob = signal_prob - signal_prob.mean()
        signal_prob = torch.clamp(signal_prob, min=0)
        signal_prob = signal_prob / (signal_prob.max() + 1e-9)
    inf_ms = (time.perf_counter() - t_inf) * 1000.0
    logger.info(
        "U-Net Stenhede inferência: %.1f ms (input %dx%d, device=%s)",
        inf_ms, tensor.shape[3], tensor.shape[2], device.type,
    )

    # 5. (1, H, W) -> (H, W) numpy float32
    sp_np: np.ndarray = signal_prob.squeeze(0).cpu().numpy().astype(np.float32)

    # 6. Resize de volta pra resolução original (pra coords de cell baterem)
    if sp_np.shape != (h_orig, w_orig):
        sp_np = cv2.resize(
            sp_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR,
        )

    return sp_np


def extract_lines_from_cell(
    signal_prob_cell: np.ndarray,
) -> np.ndarray:
    """Roda o `SignalExtractor` do Stenhede em um recorte do feature map
    correspondente a UMA derivação (uma cell do layout 3×4).

    Args:
        signal_prob_cell: shape `(H_cell, W_cell)`, float32 em `[0, 1]`.
            É o recorte do mapa global em torno da cell daquela
            derivação (banda horizontal × coluna do layout).

    Returns:
        `np.ndarray` float32, shape `(W_cell,)`. Cada posição é o `Y` do
        traço (em pixels, coords da cell, com a origem no topo do
        recorte). Colunas onde o extractor não detectou linha viram
        `NaN`. Se nenhuma linha for detectada, retorna array todo `NaN`.
    """
    if signal_prob_cell.ndim != 2:
        raise ValueError(
            f"Esperado shape (H, W); recebi {signal_prob_cell.shape}"
        )
    h_cell, w_cell = signal_prob_cell.shape
    if h_cell < 4 or w_cell < 4:
        return np.full(w_cell, np.nan, dtype=np.float32)

    extractor = _get_signal_extractor()
    feature_map = torch.from_numpy(np.ascontiguousarray(signal_prob_cell)).float()

    t0 = time.perf_counter()
    try:
        lines = extractor(feature_map)         # (N_lines, W_cell)
    except Exception as exc:  # robustez — não derruba o pipeline
        logger.warning(
            "SignalExtractor falhou em cell %dx%d: %s: %s",
            h_cell, w_cell, type(exc).__name__, exc,
        )
        return np.full(w_cell, np.nan, dtype=np.float32)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if lines.shape[0] == 0:
        logger.debug(
            "SignalExtractor: 0 linhas em cell %dx%d (%.1fms)",
            h_cell, w_cell, dt_ms,
        )
        return np.full(w_cell, np.nan, dtype=np.float32)

    # Pega a primeira (e tipicamente única) linha. Numa cell do layout
    # 3×4, esperamos 1 linha por derivação. Múltiplas linhas seriam
    # overlap entre derivações vizinhas — pegamos a mais "principal".
    line: np.ndarray = lines[0].cpu().numpy().astype(np.float32)
    logger.debug(
        "SignalExtractor: %d linhas em cell %dx%d, escolhida [0] "
        "(%d/%d cols não-NaN, %.1fms)",
        lines.shape[0], h_cell, w_cell,
        int(np.sum(~np.isnan(line))), w_cell, dt_ms,
    )
    return line


def convert_line_to_uv(
    line_y_pixels: np.ndarray,
    baseline_y: float,
    uv_per_pixel: float,
) -> np.ndarray:
    """Converte a linha extraída (Y em pixels) para amplitude em µV.

    Como Y cresce pra baixo na imagem mas amplitude cresce pra cima no
    ECG, invertemos: `signal_uv = (baseline_y − line_y) × uv_per_pixel`.

    `baseline_y` e `line_y_pixels` DEVEM estar no MESMO sistema de
    coordenadas (geralmente: ambos em coordenadas relativas à cell, ou
    ambos em coordenadas globais da imagem).

    Args:
        line_y_pixels: shape `(W,)`, float, valores em pixels (NaN
            preservado).
        baseline_y: linha de base em pixels, mesmo sistema de coords.
        uv_per_pixel: do calibrator (`1000 / (px_per_mm × gain_mm/mV)`).

    Returns:
        `np.ndarray` float64 shape `(W,)` em µV (NaN preservado).
    """
    line = np.asarray(line_y_pixels, dtype=np.float64)
    return (float(baseline_y) - line) * float(uv_per_pixel)


# ---------------------------------------------------------------------------
# Helper de conveniência: do feature map global ao sinal em µV de uma cell
# ---------------------------------------------------------------------------

def extract_cell_signal_uv(
    signal_prob_global: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    baseline_y_global: float,
    uv_per_pixel: float,
) -> np.ndarray:
    """Pipeline completo de UMA cell: recorta → SignalExtractor →
    converte pra µV (no sistema de coords GLOBAL).

    Args:
        signal_prob_global: feature map (H, W) na resolução original.
        bbox_xyxy: `(x1, y1, x2, y2)` da cell em coords globais.
        baseline_y_global: linha de base em coords globais.
        uv_per_pixel: do calibrator.

    Returns:
        `np.ndarray` float64 shape `(x2 − x1,)` em µV.
    """
    x1, y1, x2, y2 = bbox_xyxy
    H, W = signal_prob_global.shape
    x1c = max(0, int(x1))
    y1c = max(0, int(y1))
    x2c = min(W, int(x2))
    y2c = min(H, int(y2))
    if x2c <= x1c or y2c <= y1c:
        return np.full(max(0, int(x2) - int(x1)), np.nan, dtype=np.float64)

    cell = signal_prob_global[y1c:y2c, x1c:x2c]
    line_local = extract_lines_from_cell(cell)
    # converte Y local da cell -> Y global da imagem
    line_global = line_local.astype(np.float64) + float(y1c)
    signal_uv = convert_line_to_uv(line_global, baseline_y_global, uv_per_pixel)

    # padding caso a cell tenha sido clipada à esquerda/direita
    full_w = int(x2) - int(x1)
    if signal_uv.shape[0] != full_w:
        out = np.full(full_w, np.nan, dtype=np.float64)
        offset = x1c - int(x1)
        out[offset:offset + signal_uv.shape[0]] = signal_uv
        signal_uv = out
    return signal_uv


def signal_prob_to_binary_mask(
    signal_prob: np.ndarray, threshold: float = 0.1,
) -> np.ndarray:
    """Converte o feature map em máscara binária (0/255 uint8).

    Útil pra reuso da lógica de detecção de bandas/baselines do
    `lead_separator` (que historicamente recebia uma máscara do Leader).
    """
    mask = (signal_prob > float(threshold)).astype(np.uint8) * 255
    return mask


# ===========================================================================
# Pipeline FULL Stenhede (sem cell-by-cell)
# ===========================================================================

def _process_sparse_prob_torch(prob: torch.Tensor) -> torch.Tensor:
    """`process_sparse_prob` do inference_wrapper: subtrai média, clamp 0,
    normaliza pelo max."""
    prob = prob - prob.mean()
    prob = torch.clamp(prob, min=0)
    prob = prob / (prob.max() + 1e-9)
    return prob


def get_unet_feature_maps(
    image_bgr: np.ndarray, max_dim: int = _DEFAULT_MAX_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roda a U-Net principal e devolve os 3 feature maps usados pelo
    pipeline Stenhede (signal, grid, text_bg) na resolução ORIGINAL.

    Returns:
        (signal_prob, grid_prob, text_prob, bg_prob), todas
        `np.ndarray` float32 shape `(H, W)`, valores em [0, 1] após
        `softmax → process_sparse_prob`.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Esperado BGR (H, W, 3); recebi shape={image_bgr.shape}"
        )
    h_orig, w_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    max_side = max(h_orig, w_orig)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_h, new_w = int(round(h_orig * scale)), int(round(w_orig * scale))
        image_rgb = cv2.resize(
            image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA,
        )

    img_float = image_rgb.astype(np.float32)
    img_float = (img_float - img_float.min()) / max(
        img_float.max() - img_float.min(), 1e-8,
    )
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                    # (1, 4, H, W)
        probs = torch.softmax(logits, dim=1)
        signal_p = _process_sparse_prob_torch(probs[:, _SIGNAL_CLASS])
        grid_p = _process_sparse_prob_torch(probs[:, _GRID_CLASS])
        text_p = _process_sparse_prob_torch(probs[:, _TEXT_BG_CLASS])
        bg_p = _process_sparse_prob_torch(probs[:, _BG_CLASS])
    inf_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "U-Net Stenhede inferência (3 mapas): %.1f ms (input %dx%d, device=%s)",
        inf_ms, tensor.shape[3], tensor.shape[2], device.type,
    )

    def _to_np(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze(0).cpu().numpy().astype(np.float32)
        if arr.shape != (h_orig, w_orig):
            arr = cv2.resize(arr, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return arr

    return _to_np(signal_p), _to_np(grid_p), _to_np(text_p), _to_np(bg_p)


@lru_cache(maxsize=4)
def _get_lead_identifier(target_num_samples: int = 5000) -> object:
    """Carrega o LeadIdentifier do Stenhede com:
      • a U-Net pequena 1→13 (lead_name_unet) com pesos
      • o conjunto completo de layouts (`lead_layouts_all.yml`)
    """
    _ensure_vendor_on_path()
    import yaml
    from src.model.lead_identifier import LeadIdentifier  # type: ignore
    from src.model.unet import UNet  # type: ignore

    if not _VENDOR_LEAD_UNET_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Pesos lead-name U-Net não encontrados: {_VENDOR_LEAD_UNET_WEIGHTS}"
        )
    if not _VENDOR_LAYOUTS_YAML.is_file():
        raise FileNotFoundError(
            f"Arquivo de layouts não encontrado: {_VENDOR_LAYOUTS_YAML}"
        )

    lead_unet = UNet(**_LEAD_UNET_KWARGS)
    state = torch.load(
        str(_VENDOR_LEAD_UNET_WEIGHTS), map_location="cpu", weights_only=True,
    )
    if isinstance(state, tuple):
        state = state[0]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    lead_unet.load_state_dict(state)
    lead_unet.eval()

    with open(_VENDOR_LAYOUTS_YAML, "r", encoding="utf-8") as f:
        layouts = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identifier = LeadIdentifier(
        layouts=layouts,
        unet=lead_unet,
        device=device,
        possibly_flipped=False,           # nossas fotos já vêm orientadas
        target_num_samples=int(target_num_samples),
        required_valid_samples=2,
        debug=False,
    )
    logger.info(
        "LeadIdentifier carregado: %d layouts, lead_name_unet em %s, "
        "target_num_samples=%d",
        len(layouts), device.type, target_num_samples,
    )
    return identifier


def extract_signals_stenhede(
    image_bgr: np.ndarray,
    px_per_mm: float,
    paper_speed: float = 25.0,
    voltage_gain: float = 10.0,
    target_num_samples: Optional[int] = None,
) -> dict:
    """Pipeline FULL Stenhede end-to-end.

    Sequência (sem perspective/cropper/dewarper — já fizemos no nosso
    pipeline):
      1. U-Net Stenhede → signal_prob, grid_prob, text_prob (resolução
         original)
      2. SignalExtractor(signal_prob FULL IMAGE) → linhas em pixel Y
      3. LeadIdentifier(lines, text_prob, px_per_mm, mv_per_mm) →
         canonical_lines (12, target_num_samples) em µV + match info

    Args:
        image_bgr: imagem undistorted BGR uint8 (H, W, 3).
        px_per_mm: do nosso calibrator (sobrescreve PixelSizeFinder do
            Stenhede — nosso é mais preciso porque vem do Gridder).
        paper_speed: mm/s, default 25 (BR).
        voltage_gain: mm/mV, default 10 (BR). Determina mv_per_mm = 1/gain.
        target_num_samples: se None, usa W da imagem (sem interpolação).
            Caso contrário, o LeadIdentifier interpola pra esse valor.

    Returns:
        dict:
          {
            "signals": {"I": np.ndarray µV, ..., "V6": ..., "II_rhythm": ...},
            "sampling_rate_hz": float,
            "raw_lines_pixel": np.ndarray (N_lines, W) — Y em pixels
                              (pré-canonicalize, pra overlay/debug),
            "match": dict (layout, flip, cost, leads, ...),
            "canonical_lines_uv": np.ndarray (12, target_num_samples) em µV,
            "n_lines_detected": int,
          }
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Esperado BGR (H, W, 3); recebi {image_bgr.shape}")
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm inválido: {px_per_mm}")

    h_orig, w_orig = image_bgr.shape[:2]
    if target_num_samples is None:
        target_num_samples = int(w_orig)

    t_total = time.perf_counter()

    # 1. Feature maps
    signal_prob, grid_prob, text_prob, _bg_prob = get_unet_feature_maps(
        image_bgr,
    )

    # 2. SignalExtractor na imagem inteira
    extractor = _get_signal_extractor()
    signal_prob_t = torch.from_numpy(np.ascontiguousarray(signal_prob)).float()
    t_se = time.perf_counter()
    raw_lines_t: torch.Tensor = extractor(signal_prob_t)   # (N, W) Y em pixels
    se_ms = (time.perf_counter() - t_se) * 1000.0
    n_lines = int(raw_lines_t.shape[0])
    logger.info(
        "SignalExtractor full image: %d linhas detectadas em %.0f ms",
        n_lines, se_ms,
    )
    if n_lines == 0:
        logger.warning(
            "SignalExtractor não detectou linhas — pipeline degenerado."
        )

    # 3. LeadIdentifier
    identifier = _get_lead_identifier(target_num_samples=target_num_samples)
    text_prob_t = (
        torch.from_numpy(np.ascontiguousarray(text_prob))
        .unsqueeze(0).unsqueeze(0).float()
    )
    mv_per_mm = 1.0 / float(voltage_gain)
    t_li = time.perf_counter()
    try:
        result = identifier(
            raw_lines_t.clone(),
            text_prob_t,
            avg_pixel_per_mm=float(px_per_mm),
            threshold=0.8,
            mv_per_mm=mv_per_mm,
        )
    except Exception as e:
        logger.error(
            "LeadIdentifier falhou: %s: %s", type(e).__name__, e,
        )
        raise
    li_ms = (time.perf_counter() - t_li) * 1000.0
    logger.info(
        "LeadIdentifier: layout=%s flip=%s cost=%.3f n_detected=%d (%.0f ms)",
        result.get("layout"), result.get("flip"), result.get("cost", float("nan")),
        result.get("n_detected", 0), li_ms,
    )

    canonical: torch.Tensor = result["canonical_lines"]  # (12, T) em µV
    if canonical is None:
        canonical = torch.full(
            (12, target_num_samples), float("nan"), dtype=torch.float32,
        )

    canonical_np = canonical.cpu().numpy().astype(np.float64)

    # 4. fs efetivo (target_num_samples mapeia a duração efetiva)
    duration_total_s = w_orig / (px_per_mm * paper_speed)
    fs_eff = canonical_np.shape[1] / duration_total_s if duration_total_s > 0 else 0.0
    logger.info(
        "fs efetivo: %.1f Hz (target_num_samples=%d, duracao=%.2f s)",
        fs_eff, canonical_np.shape[1], duration_total_s,
    )

    # 5. Monta dict de saída
    signals: dict[str, np.ndarray] = {}
    for i, name in enumerate(LEAD_CHANNEL_ORDER):
        signals[name] = canonical_np[i].copy()

    # II_rhythm: se o layout tem rhythm leads, o LeadIdentifier já moveu o
    # rhythm strip pra dentro de canonical_lines (geralmente substitui II).
    # Pra preservar o contrato com measure.py, copiamos II como II_rhythm.
    # No futuro: extrair rhythm separado de result["lines"] olhando a linha
    # mais larga.
    signals["II_rhythm"] = signals["II"].copy()

    total_ms = (time.perf_counter() - t_total) * 1000.0
    logger.info("extract_signals_stenhede TOTAL: %.0f ms", total_ms)

    return {
        "signals": signals,
        "sampling_rate_hz": float(fs_eff),
        "raw_lines_pixel": raw_lines_t.cpu().numpy().astype(np.float64),
        "match": {
            "layout": result.get("layout"),
            "flip": bool(result.get("flip", False)),
            "cost": float(result.get("cost", float("nan"))),
            "leads": result.get("leads"),
            "rows_in_layout": int(result.get("rows_in_layout", 0)),
            "n_detected": int(result.get("n_detected", 0)),
        },
        "canonical_lines_uv": canonical_np,
        "n_lines_detected": n_lines,
    }
