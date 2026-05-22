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
_VENDOR_LAYOUTS_YAML = _VENDOR_ROOT / "src" / "config" / "lead_layouts_all.yml"  # reduced so tem 2 layouts e nao cobre 12x1

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

# A/B flag — se True, usa o novo signal_extractor_v2 (3 fases: Fragmented +
# Viterbi bidirecional + QRS peak correction). Se False, usa o caminho
# original (Stenhede SignalExtractor / SkeletonSignalExtractor).
USE_EXTRACTOR_V2 = True

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
    """Retorna SkeletonSignalExtractor — subclasse do Stenhede com extracao
    primaria por SKELETONIZAÇÃO (eixo medial) + Regra B de continuidade.

    Por que skeletonização:
      O canal 2 da UNet pinta o tracado com espessura (~5-10 px). O algoritmo
      original (centroide ponderado Y por coluna) afoga picos verticais como
      QRS no meio do span entre baseline e pico. A skeletonização eh a operacao
      morfologica classica pra reduzir uma "pintura grossa" a 1 px de espessura
      preservando a forma original — literalmente o "espinhaço" da pintura.

    Algoritmo:
      1. skeletonize(mask) → produz linha de 1 px ao longo do eixo medial
      2. Em trechos horizontais (baseline): 1 Y por X. Leitura direta.
      3. Em trechos verticais (QRS): N Y por X. Tiebreak pela Regra B
         (continuidade) — escolhe Y mais proximo do Y da coluna anterior.
      4. Ancora na coluna de prob maxima no esqueleto, walk left + right.

    Parametros mantidos da rodada anterior:
      label_thresh        0.05  (aceita sinal mais fraco)
      threshold_sum       5.0   (manter fragmentos menores)
      candidate_span      5     (anti-confusao na horizontal_path fallback)
      split_num_stripes   8     (quebrar conexoes cross-row no label)
      min_line_width      20    (manter trechos curtos)
    """
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore
    from skimage.morphology import skeletonize  # type: ignore
    import numpy as _np

    class SkeletonSignalExtractor(SignalExtractor):  # type: ignore[misc]
        def _extract_line_from_region(
            self,
            fmap: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            H, W = fmap.shape
            mask_np = mask.cpu().numpy().astype(bool)
            fmap_np = fmap.cpu().numpy()

            # 1. Skeletonização — extrai o eixo medial da pintura do canal 2
            skel = skeletonize(mask_np)  # bool (H, W), True onde tem esqueleto
            if not skel.any():
                return torch.full((W,), float("nan"), dtype=torch.float32)

            # 2. Pra cada X, qual o "valor" maximo do esqueleto (prob).
            #    Usado pra escolher ancora.
            skel_masked = _np.where(skel, fmap_np, -1.0)
            col_max = skel_masked.max(axis=0)  # (W,)
            if col_max.max() <= 0:
                return torch.full((W,), float("nan"), dtype=torch.float32)

            # 3. Ancora: coluna com prob max no esqueleto.
            #    Anchor Y: entre os Y do esqueleto naquela coluna, o de maior prob.
            anchor_x = int(col_max.argmax())
            anchor_col_ys = _np.where(skel[:, anchor_x])[0]
            anchor_y = int(
                anchor_col_ys[
                    fmap_np[anchor_col_ys, anchor_x].argmax()
                ]
            )

            line_np = _np.full(W, _np.nan, dtype=_np.float32)
            line_np[anchor_x] = float(anchor_y)

            # 4. Walk left + right aplicando Regra B (continuidade).
            #    Em colunas com 1 Y no esqueleto: leitura direta.
            #    Em colunas com N Y (trechos verticais): pega o mais proximo do
            #    Y da coluna anterior.
            #    Em colunas SEM esqueleto mas COM mask: usa o centroide Y da
            #    mascara naquela coluna (fallback pra evitar gaps que fariam
            #    _classify_line rejeitar a linha inteira).
            def _walk(start: int, end: int, step: int, init_prev: int) -> None:
                prev_y = init_prev
                for x in range(start, end, step):
                    ys = _np.where(skel[:, x])[0]
                    if len(ys) == 0:
                        # Fallback: usa centroide Y da mascara nesta coluna
                        ys_mask = _np.where(mask_np[:, x])[0]
                        if len(ys_mask) == 0:
                            continue  # nem mascara, mantem NaN
                        # Pega Y na mascara mais proximo do prev_y
                        best = int(ys_mask[_np.argmin(_np.abs(ys_mask - prev_y))])
                    elif len(ys) == 1:
                        best = int(ys[0])
                    else:
                        best = int(ys[_np.argmin(_np.abs(ys - prev_y))])
                    line_np[x] = float(best)
                    prev_y = best

            _walk(anchor_x + 1, W, 1, anchor_y)
            _walk(anchor_x - 1, -1, -1, anchor_y)

            # 5. Interpolar quaisquer gaps remanescentes (NaN dentro do range
            #    onde a mascara existe) — protege contra rejeicao pelo
            #    _classify_line por causa de buracos pontuais.
            valid_mask_cols = mask_np.any(axis=0)
            valid_idx = _np.where(valid_mask_cols)[0]
            if len(valid_idx) > 0:
                first_v, last_v = int(valid_idx[0]), int(valid_idx[-1])
                seg = line_np[first_v:last_v + 1]
                nan_in_seg = _np.isnan(seg)
                if nan_in_seg.any() and not nan_in_seg.all():
                    idx = _np.arange(seg.shape[0])
                    seg[nan_in_seg] = _np.interp(
                        idx[nan_in_seg],
                        idx[~nan_in_seg],
                        seg[~nan_in_seg],
                    )
                    line_np[first_v:last_v + 1] = seg

            line_t = torch.from_numpy(line_np)
            line_t[line_t < 5] = float("nan")
            return line_t

    return SkeletonSignalExtractor(
        label_thresh=0.05,
        threshold_sum=5.0,
        candidate_span=5,
        split_num_stripes=8,
        min_line_width=20,
    )


def _signal_extractor_with_offset(
    signal_prob_t: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Como `SignalExtractor.__call__`, mas também devolve o `x_offset`
    em coords da imagem original.

    Se USE_EXTRACTOR_V2 estiver True, delega pra extract_lines_v2_with_offset
    (modulo signal_extractor_v2) — 3 fases: Fragmented + Viterbi + QRS.

    Senao usa o caminho original (Stenhede SignalExtractor / Skeleton).
    """
    if USE_EXTRACTOR_V2:
        from .signal_extractor_v2 import extract_lines_v2_with_offset
        return extract_lines_v2_with_offset(signal_prob_t)

    extractor = _get_signal_extractor()
    fmap = signal_prob_t.cpu().clone()
    W = signal_prob_t.shape[-1]

    # Lines pre-merge (full-width) só pra calcular o offset
    lines_list = extractor._iterative_extraction(fmap.clone())
    extractor.num_peaks = extractor._autodetect_num_peaks(fmap)
    lines_list = [
        ln for ln in lines_list
        if (~torch.isnan(ln)).sum() > extractor.min_line_width
    ]
    if len(lines_list) == 0:
        return torch.empty((0, W), dtype=torch.float32), 0

    lines_full = torch.stack(lines_list, dim=0)
    # Replica EXATAMENTE preprocess_lines pra detectar first_valid_col
    chk = lines_full.clone()
    chk[chk == 0] = float("nan")
    valid_cols = chk.nan_to_num(0.0).abs().sum(0) > 0
    nonzero = torch.nonzero(valid_cols, as_tuple=True)[0]
    offset = int(nonzero[0].item()) if nonzero.numel() > 0 else 0

    # Roda match-and-merge no full-width (vai re-trimar internamente
    # com o mesmo first/last)
    merged_list, _ = extractor.match_and_merge_lines(lines_full)
    if len(merged_list) == 0:
        return torch.empty((0, W), dtype=torch.float32), offset
    merged = torch.stack(merged_list, dim=0)
    return merged, offset


# Segundos cobertos por cada coluna do layout (padrão BR a 25 mm/s)
# Para layout R×C com 10s totais: cada coluna tem 10/C segundos
_SECONDS_PER_CELL_BY_COLS = {
    1: 10.0,    # 12x1, 6x1, 3x1 — uma coluna inteira = 10 s
    2: 5.0,     # 6x2 — duas colunas = 5 s cada
    4: 2.5,     # 3x4 (BR padrão) — 4 colunas = 2.5 s cada
}


def _canonicalize_calibrated(
    raw_lines: np.ndarray,
    layouts: dict,
    layout_name: str,
    x_offset: int,
    w_orig: int,
    avg_pixel_per_mm: float,
    paper_speed: float = 25.0,
    voltage_gain: float = 10.0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Divide `raw_lines` (saída do SignalExtractor, em pixel-Y) em leads
    usando CALIBRAÇÃO REAL em mm em vez de `W // cols`.

    Args:
        raw_lines: ndarray `(N_rows, lines_trimmed_W)` com Y em pixels da
            imagem (já trimmada pelo preprocess_lines do SignalExtractor —
            inicia em coords de cell, NÃO de imagem).
        layouts: dict de layouts (carregado do YAML do vendor).
        layout_name: nome do layout escolhido pelo LeadIdentifier.
        x_offset: primeira coluna da imagem onde raw_lines começa
            (margem esquerda já trimmada).
        w_orig: largura total da imagem original (px).
        avg_pixel_per_mm: do PixelSizeFinder.
        paper_speed: 25 mm/s (BR padrão).
        voltage_gain: 10 mm/mV (BR padrão).

    Returns:
        (signals_dict, canonical_array)
          • signals_dict: {lead_name: np.ndarray em µV} com 12 leads + (se
            houver) "II_rhythm" no contrato esperado por measure.py.
          • canonical_array: (12, chunk_px) em µV, mesma ordem de
            LEAD_CHANNEL_ORDER (None se layout não-suportado).
    """
    if layout_name not in layouts:
        # Fallback: layout desconhecido → empty dict
        empty = {n: np.zeros(0, dtype=np.float64) for n in LEAD_CHANNEL_ORDER}
        return empty, np.full((12, 0), np.nan, dtype=np.float64)

    layout_def = layouts[layout_name]
    cols = int(layout_def["layout"]["cols"])
    leads_def = layout_def["leads"]
    rhythm_leads = list(layout_def.get("rhythm_leads", []))

    # Largura real de cada cell em px, calibrada em mm
    spc = _SECONDS_PER_CELL_BY_COLS.get(cols, 10.0 / cols)
    chunk_mm = spc * float(paper_speed)
    chunk_px = int(round(chunk_mm * float(avg_pixel_per_mm)))
    if chunk_px <= 0:
        empty = {n: np.zeros(0, dtype=np.float64) for n in LEAD_CHANNEL_ORDER}
        return empty, np.full((12, 0), np.nan, dtype=np.float64)

    mv_per_mm = 1.0 / float(voltage_gain)
    # Fator de conversão pixel-Y → µV.
    # Idêntico ao normalize() do LeadIdentifier: (mv_per_mm/px_per_mm) × 1000.
    # O sinal é negado porque Y cresce pra baixo na imagem.
    uv_per_pixel = (mv_per_mm / float(avg_pixel_per_mm)) * 1000.0

    n_rows_lines, lines_w = raw_lines.shape

    def _chunk_to_uv(chunk: np.ndarray) -> np.ndarray:
        """Centra e converte pra µV (com sinal invertido)."""
        if chunk.size == 0 or np.all(np.isnan(chunk)):
            return np.full_like(chunk, np.nan, dtype=np.float64)
        # Centra na média (igual ao normalize do Stenhede)
        c = chunk.astype(np.float64)
        mean = float(np.nanmean(c))
        c = c - mean
        # µV (negado: Y cresce pra baixo)
        return -c * uv_per_pixel

    signals: dict[str, np.ndarray] = {}
    canonical_arr = np.full((12, chunk_px), np.nan, dtype=np.float64)

    is_matrix = isinstance(leads_def[0], list) if leads_def else False
    if is_matrix:
        rows_iter = leads_def
    else:
        # Flat list — reshape usando rows/cols do layout YAML.
        # Ex: standard_12x1 tem leads=[I,II,...,V6] (flat) e rows=12, cols=1
        # → devemos tratar como 12 rows × 1 col, não 1 row × 12 cols.
        layout_rows = int(layout_def["layout"].get("rows", 1))
        layout_cols_check = int(layout_def["layout"].get("cols", len(leads_def)))
        if layout_rows * layout_cols_check == len(leads_def) and layout_rows > 1:
            rows_iter = [
                leads_def[i * layout_cols_check:(i + 1) * layout_cols_check]
                for i in range(layout_rows)
            ]
        else:
            rows_iter = [leads_def]

    n_main_rows = len(rows_iter)
    for row_idx, row_leads in enumerate(rows_iter):
        if row_idx >= n_rows_lines:
            break
        if not isinstance(row_leads, list):
            row_leads = [row_leads]
        line = raw_lines[row_idx]
        for col_idx, lead_name_raw in enumerate(row_leads):
            lead_name = str(lead_name_raw).lstrip("-")
            sign = -1 if str(lead_name_raw).startswith("-") else 1
            x_start_local = col_idx * chunk_px
            x_end_local = x_start_local + chunk_px
            if x_start_local >= lines_w:
                chunk_line = np.zeros(chunk_px, dtype=np.float64)
            else:
                actual_end = min(x_end_local, lines_w)
                chunk_line = np.full(chunk_px, np.nan, dtype=np.float64)
                chunk_line[: actual_end - x_start_local] = line[x_start_local:actual_end]
            chunk_uv = _chunk_to_uv(chunk_line) * sign
            signals[lead_name] = chunk_uv
            if lead_name in LEAD_CHANNEL_ORDER:
                idx = LEAD_CHANNEL_ORDER.index(lead_name)
                canonical_arr[idx, :] = chunk_uv

    # Rhythm strip(s): última linha(s) do raw_lines, ocupando cols × chunk_px
    full_w = cols * chunk_px
    if len(rhythm_leads) > 0 and n_rows_lines > n_main_rows:
        for k, _rhy_lead_def in enumerate(rhythm_leads):
            line_idx = n_main_rows + k
            if line_idx >= n_rows_lines:
                break
            line = raw_lines[line_idx]
            actual_w = min(full_w, lines_w)
            chunk = np.full(full_w, np.nan, dtype=np.float64)
            chunk[:actual_w] = line[:actual_w]
            chunk_uv = _chunk_to_uv(chunk)
            # Por convenção: primeiro rhythm = II_rhythm
            key = "II_rhythm" if k == 0 else f"rhythm_{k}"
            signals[key] = chunk_uv

    # Garante que todos os 12 leads canônicos estão no dict (NaN se ausente)
    for canon_name in LEAD_CHANNEL_ORDER:
        if canon_name not in signals:
            signals[canon_name] = np.full(chunk_px, np.nan, dtype=np.float64)

    return signals, canonical_arr


def _pad_lines_to_image_width(
    lines_trimmed: np.ndarray, x_offset: int, w_orig: int,
) -> np.ndarray:
    """Devolve `(N, w_orig)` com NaN nas bordas e os dados em
    `[x_offset : x_offset + L]`. Plotar contra `np.arange(w_orig)` agora
    cai exatamente em cima da imagem original."""
    if lines_trimmed.ndim != 2 or lines_trimmed.shape[1] == 0:
        return np.full((lines_trimmed.shape[0], w_orig), np.nan, dtype=np.float64)
    n, lw = lines_trimmed.shape
    padded = np.full((n, w_orig), np.nan, dtype=np.float64)
    end = min(x_offset + lw, w_orig)
    valid_l = max(0, end - x_offset)
    padded[:, x_offset:end] = lines_trimmed[:, :valid_l]
    return padded


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


@lru_cache(maxsize=1)
def _get_cropper():  # type: ignore[no-untyped-def]
    """Cropper do Stenhede com kwargs do `inference_wrapper.yml` (gm2024).

    Usado APENAS pelo `apply_perspective` (modo "Versão B"). Não chamamos
    `cropper.forward()` pra calcular source_points — passamos os 4 cantos
    da imagem como source (a imagem já está undistorted), de modo que o
    cropper só aplica o blending alpha=0.85 entre image-corners e center.
    """
    _ensure_vendor_on_path()
    from src.model.cropper import Cropper  # type: ignore
    return Cropper(granularity=80, percentiles=(0.02, 0.98), alpha=0.85)


@lru_cache(maxsize=1)
def _get_pixel_size_finder():  # type: ignore[no-untyped-def]
    """PixelSizeFinder do Stenhede (autocorrelação da projeção do grid)."""
    _ensure_vendor_on_path()
    from src.model.pixel_size_finder import PixelSizeFinder  # type: ignore
    return PixelSizeFinder(
        min_number_of_grid_lines=30,
        max_number_of_grid_lines=70,
        lower_grid_line_factor=0.3,  # default Stenhede (restaurado de 0.5)
    )


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
        required_valid_samples=3,         # default Stenhede (restaurado de 2)
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
    px_per_mm: Optional[float] = None,
    paper_speed: float = 25.0,
    voltage_gain: float = 10.0,
    target_num_samples: Optional[int] = None,
    use_cropper: bool = False,
    use_internal_pixel_size: bool = True,
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

    h_orig, w_orig = image_bgr.shape[:2]
    if target_num_samples is None:
        target_num_samples = 5000  # default Stenhede (restaurado de int(w_orig))

    t_total = time.perf_counter()

    # 1. U-Net Stenhede → 4 feature maps em resolução original
    signal_prob, grid_prob, text_prob, _bg_prob = get_unet_feature_maps(
        image_bgr,
    )

    # 2. (Opcional) Cropper.apply_perspective — Versão B do pipeline.
    #    Não chamamos cropper.forward() (precisaria do perspective_detector
    #    que pulamos). Em vez disso passamos os 4 cantos da imagem como
    #    source_points; o cropper aplica blending alpha=0.85 com o center,
    #    encolhendo ~7.5% pra dentro de cada lado. Pra Versão A pulamos.
    if use_cropper:
        cropper = _get_cropper()
        H, W = signal_prob.shape[:2]
        source_points = torch.tensor(
            [[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]],
            dtype=torch.float32,
        )

        def _warp(arr_2d: np.ndarray) -> np.ndarray:
            t = (
                torch.from_numpy(np.ascontiguousarray(arr_2d))
                .unsqueeze(0).unsqueeze(0).float()
            )
            warped = cropper.apply_perspective(t, source_points, fill_value=0)
            return warped.squeeze().cpu().numpy().astype(np.float32)

        t_crop = time.perf_counter()
        signal_prob_use = _warp(signal_prob)
        grid_prob_use = _warp(grid_prob)
        text_prob_use = _warp(text_prob)
        logger.info(
            "Cropper.apply_perspective (Versão B, alpha=0.85): %.0f ms",
            (time.perf_counter() - t_crop) * 1000.0,
        )
    else:
        signal_prob_use = signal_prob
        grid_prob_use = grid_prob
        text_prob_use = text_prob

    # 3. PixelSizeFinder — px/mm a partir do grid_prob do Stenhede
    if use_internal_pixel_size:
        pxsize = _get_pixel_size_finder()
        t_ps = time.perf_counter()
        with torch.no_grad():
            grid_t = torch.from_numpy(
                np.ascontiguousarray(grid_prob_use)
            ).float()
            mm_per_pixel_x, mm_per_pixel_y = pxsize(grid_t)
        avg_pixel_per_mm = float(
            (1.0 / float(mm_per_pixel_x) + 1.0 / float(mm_per_pixel_y)) / 2.0
        )
        logger.info(
            "PixelSizeFinder Stenhede: mm/px x=%.4f y=%.4f -> "
            "avg_px/mm=%.3f (%.0f ms)",
            float(mm_per_pixel_x), float(mm_per_pixel_y), avg_pixel_per_mm,
            (time.perf_counter() - t_ps) * 1000.0,
        )
    else:
        if px_per_mm is None or px_per_mm <= 0:
            raise ValueError(
                "px_per_mm é obrigatório quando use_internal_pixel_size=False"
            )
        avg_pixel_per_mm = float(px_per_mm)
        logger.info(
            "Pixel size externo (do calibrator): avg_px/mm=%.3f",
            avg_pixel_per_mm,
        )

    # 4. SignalExtractor (com detecção do x_offset)
    signal_prob_t = torch.from_numpy(
        np.ascontiguousarray(signal_prob_use)
    ).float()
    t_se = time.perf_counter()
    raw_lines_t, x_offset = _signal_extractor_with_offset(signal_prob_t)
    se_ms = (time.perf_counter() - t_se) * 1000.0
    n_lines_raw = int(raw_lines_t.shape[0])
    logger.info(
        "SignalExtractor full image: %d linhas (pre-sort), x_offset=%d "
        "(trim de %d/%d cols), %.0f ms",
        n_lines_raw, x_offset,
        signal_prob_use.shape[1] - raw_lines_t.shape[1] if n_lines_raw > 0 else 0,
        signal_prob_use.shape[1], se_ms,
    )
    if n_lines_raw == 0:
        logger.warning(
            "SignalExtractor não detectou linhas — pipeline degenerado."
        )

    # 4.5. Sort+merge das raw_lines IGUAL ao LeadIdentifier faz internamente.
    # Sem isso, raw_lines saem do SignalExtractor em ordem arbitraria (ordem
    # do Hungarian matching), e o _canonicalize_calibrated abaixo associa
    # row_idx do layout (top-to-bottom) a linhas em ordem aleatoria —
    # fazendo lead I cair na linha errada do ECG, etc.
    # `_merge_nonoverlapping_lines` faz: (1) sort por mean Y top-to-bottom,
    # (2) mescla pares de linhas que nao se sobrepoe horizontalmente
    # (fragmentos da mesma linha fisica).
    identifier = _get_lead_identifier(target_num_samples=target_num_samples)
    if raw_lines_t.shape[0] > 1:
        raw_lines_t = identifier._merge_nonoverlapping_lines(raw_lines_t.clone())
        n_lines = int(raw_lines_t.shape[0])
        logger.info(
            "raw_lines apos sort+merge (igual LeadIdentifier interno): %d -> %d",
            n_lines_raw, n_lines,
        )
    else:
        n_lines = n_lines_raw

    # 5. LeadIdentifier
    text_prob_t = (
        torch.from_numpy(np.ascontiguousarray(text_prob_use))
        .unsqueeze(0).unsqueeze(0).float()
    )
    mv_per_mm = 1.0 / float(voltage_gain)
    t_li = time.perf_counter()
    try:
        result = identifier(
            raw_lines_t.clone(),
            text_prob_t,
            avg_pixel_per_mm=avg_pixel_per_mm,
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

    # 6. Canonicalização CALIBRADA — ignora canonical_lines do LeadIdentifier
    #    (que usava W//cols, hardcoded). Usa o LAYOUT escolhido pelo
    #    matching húngaro (que é robusto), mas divide as cells em pedaços
    #    de 62.5 mm × px_per_mm = chunk real do papel BR a 25 mm/s.
    layouts_dict = (
        identifier.layouts if hasattr(identifier, "layouts") else {}
    )
    raw_lines_np = raw_lines_t.cpu().numpy().astype(np.float64)
    layout_name = result.get("layout") or "standard_3x4_with_r1"
    signals, canonical_arr_calibrated = _canonicalize_calibrated(
        raw_lines=raw_lines_np,
        layouts=layouts_dict,
        layout_name=str(layout_name),
        x_offset=int(x_offset),
        w_orig=int(signal_prob_use.shape[1]),
        avg_pixel_per_mm=float(avg_pixel_per_mm),
        paper_speed=float(paper_speed),
        voltage_gain=float(voltage_gain),
    )

    canonical_np = canonical_arr_calibrated  # já em np
    chunk_px = canonical_np.shape[1] if canonical_np.size else 0

    # fs efetivo: chunk_px samples cobrem seconds_per_cell segundos
    cols_layout = (
        int(layouts_dict.get(layout_name, {}).get("layout", {}).get("cols", 4))
        if layout_name in layouts_dict else 4
    )
    spc = _SECONDS_PER_CELL_BY_COLS.get(cols_layout, 10.0 / max(cols_layout, 1))
    fs_eff = chunk_px / spc if spc > 0 else 0.0
    logger.info(
        "Canonicalize CALIBRADO: layout=%s cols=%d chunk_px=%d "
        "(seconds/cell=%.1fs) -> fs=%.1f Hz",
        layout_name, cols_layout, chunk_px, spc, fs_eff,
    )

    # Garante "II_rhythm" no dict (mesmo que não venha do layout)
    if "II_rhythm" not in signals:
        # Não há rhythm strip detectada — copia DII como fallback (compat)
        signals["II_rhythm"] = signals.get(
            "II", np.full(chunk_px, np.nan, dtype=np.float64),
        ).copy()

    # raw_lines_pixel padded pra largura do signal_prob_use — plotar
    # contra np.arange(W) cai sobre o traçado. Em Versão A coincide com
    # imagem original; em B coincide com a versão warped pelo cropper.
    raw_lines_padded = _pad_lines_to_image_width(
        raw_lines_t.cpu().numpy().astype(np.float64),
        x_offset=x_offset,
        w_orig=int(signal_prob_use.shape[1]),
    )

    total_ms = (time.perf_counter() - t_total) * 1000.0
    logger.info("extract_signals_stenhede TOTAL: %.0f ms", total_ms)

    return {
        "signals": signals,
        "sampling_rate_hz": float(fs_eff),
        "raw_lines_pixel": raw_lines_padded,
        "raw_lines_x_offset": int(x_offset),
        "avg_pixel_per_mm": avg_pixel_per_mm,
        "use_cropper": bool(use_cropper),
        "use_internal_pixel_size": bool(use_internal_pixel_size),
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
