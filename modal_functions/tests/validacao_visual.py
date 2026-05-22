"""
Validacao visual do pipeline ProECG — gera 16 imagens por ECG mostrando o
output de cada etapa.

Rodar com:
    cd modal_functions
    python -m tests.validacao_visual

Ajustes ja aplicados no pipeline:
  1. format_detector nosso DESATIVADO (layout vem do LeadIdentifier do Stenhede)
  2. calibrator nosso DESATIVADO (px/mm vem do PixelSizeFinder do Stenhede)
"""

from __future__ import annotations

import logging
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# Caminho da raiz do projeto e do modal_functions
HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))
sys.path.insert(0, str(MODAL_ROOT.parent))

from pipeline.digitize.ecg_digitizer import ECGDigitizer, LEAD_LAYOUT_3x4
from pipeline.digitize.constants import GAIN_DEFAULT, GRID_MAJOR_MM, PAPER_SPEED_DEFAULT
from pipeline.digitize.stenhede_adapter import (
    LEAD_CHANNEL_ORDER,
    _SECONDS_PER_CELL_BY_COLS,
    _canonicalize_calibrated,
    _ensure_vendor_on_path,
    _get_lead_identifier,
    _get_pixel_size_finder,
    _process_sparse_prob_torch,
    _signal_extractor_with_offset,
    get_unet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("validacao_visual")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3")
OUTPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")

ECG_NAMES = ["0a8c7db0-e31a-4e64-a15e-f7d29c1f7661"]

# Sufixo pra adicionar ao nome da imagem 12 — usado pra comparar metodos
# de extracao diferentes lado a lado sem sobrescrever resultados anteriores.
# Ex: "_v2_frag_viterbi_qrs" -> "12_signal_extractor_v2_frag_viterbi_qrs.png"
# Vazio = sobrescreve "12_signal_extractor.png".
GEN12_SUFFIX = "_v2_argmax_opening_splitTall"

# Cache de intermediarios (tudo ate antes do SignalExtractor) — permite
# rerodar so a parte de extracao em ~30s (vs ~12min full).
# Setar pra False pra forcar full rerun (ignora cache existente, sobrescreve).
USE_CACHE = True

# Cores para 13 linhas (12 leads + 1 rhythm)
LEAD_COLORS = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
    "#9A6324",
]


def find_image(stem: str) -> Path | None:
    """Procura imagem com o stem dado em INPUT_DIR (qualquer extensao)."""
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".heic", ".HEIC"):
        p = INPUT_DIR / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def setup_fig(title: str, figsize: tuple = (14, 10)) -> tuple[plt.Figure, Any]:
    """Cria figura com titulo grande na parte superior."""
    fig = plt.figure(figsize=figsize, dpi=110)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    return fig, fig


def save_fig(fig: plt.Figure, path: Path, footer: str = "") -> None:
    """Salva figura com footer opcional e fecha."""
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9, color="#444")
    fig.savefig(str(path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def text_only_image(path: Path, lines: list[str], color_top: str = "#cc0000") -> None:
    """Gera imagem com apenas texto (para etapas desativadas/removidas)."""
    fig, _ = plt.subplots(figsize=(14, 8), dpi=110)
    plt.axis("off")
    y = 0.85
    for i, line in enumerate(lines):
        size = 22 if i == 0 else 14
        weight = "bold" if i == 0 else "normal"
        col = color_top if i == 0 else "#222"
        plt.text(0.5, y, line, ha="center", va="center", fontsize=size,
                 fontweight=weight, color=col, transform=plt.gca().transAxes)
        y -= 0.13 if i == 0 else 0.10
    plt.savefig(str(path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline instrumentado — captura todos os intermediarios
# ---------------------------------------------------------------------------

def _cache_file_path(output_dir_for_ecg: Path) -> Path:
    """Caminho do arquivo cache (pre-SignalExtractor) por ECG."""
    return output_dir_for_ecg / "_cache_pre_signal_extractor.pkl"


# Keys que vao pro cache (tudo ate antes da SignalExtractor)
_CACHE_KEYS = [
    "original", "cropped", "crop_info",
    "dotter_mask", "keypoints",
    "grid_matrix", "grid_info",
    "normalized", "undistort_applied",
    "signal_prob", "grid_prob", "text_prob", "bg_prob",
    "pixel_size",
]


def _compute_pre_signal_extractor(image_path: Path) -> dict[str, Any]:
    """Roda o pipeline desde o input ate (e incluindo) o UNet Stenhede +
    PixelSizeFinder. Esse e o pedaco cacheavel — nao depende do SignalExtractor.
    """
    out: dict[str, Any] = {}

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Falha ao carregar: {image_path}")

    # Orientar portrait -> landscape
    h, w = img.shape[:2]
    if h > w * 1.2:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    out["original"] = img.copy()

    digitizer = ECGDigitizer(use_mock=False)

    # ----- Etapa 2: crop + perspectiva -----
    cropped, crop_info = digitizer.preprocess(img)
    out["cropped"] = cropped
    out["crop_info"] = crop_info

    # ----- Etapa 3: Dotter -----
    dotter_mask, keypoints = digitizer.dotter(cropped)
    if len(keypoints) == 0:
        logger.warning("Dotter UNet vazio, fallback mock")
        dotter_mask, keypoints = digitizer.dotter_mock(cropped)
    out["dotter_mask"] = dotter_mask
    out["keypoints"] = keypoints

    # ----- Etapa 4: Gridder -----
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])
    out["grid_matrix"] = grid_matrix
    out["grid_info"] = grid_info

    # ----- Etapa 5: Undistort -----
    if len(keypoints) >= 100:
        normalized = digitizer.undistort(cropped, grid_matrix, grid_info["px_per_mm"])
        out["undistort_applied"] = True
    else:
        normalized = cropped.copy()
        out["undistort_applied"] = False
    out["normalized"] = normalized

    # ===== Stenhede begins =====

    # Forcar resolucao maxima 3000
    h_n, w_n = normalized.shape[:2]
    image_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    max_side = max(h_n, w_n)
    if max_side > 3000:
        scale = 3000 / float(max_side)
        new_h, new_w = int(round(h_n * scale)), int(round(w_n * scale))
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_float = image_rgb.astype(np.float32)
    img_float = (img_float - img_float.min()) / max(img_float.max() - img_float.min(), 1e-8)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    # ----- Etapa 8: UNet 4 canais -----
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        signal_p = _process_sparse_prob_torch(probs[:, 2])
        grid_p = _process_sparse_prob_torch(probs[:, 0])
        text_p = _process_sparse_prob_torch(probs[:, 1])
        bg_p = _process_sparse_prob_torch(probs[:, 3])

    def _to_np(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze(0).cpu().numpy().astype(np.float32)
        if arr.shape != (h_n, w_n):
            arr = cv2.resize(arr, (w_n, h_n), interpolation=cv2.INTER_LINEAR)
        return arr

    out["signal_prob"] = _to_np(signal_p)
    out["grid_prob"] = _to_np(grid_p)
    out["text_prob"] = _to_np(text_p)
    out["bg_prob"] = _to_np(bg_p)

    # ----- Etapa 10: PixelSizeFinder Stenhede -----
    pxsize = _get_pixel_size_finder()
    with torch.no_grad():
        grid_t = torch.from_numpy(np.ascontiguousarray(out["grid_prob"])).float()
        mm_per_pixel_x, mm_per_pixel_y = pxsize(grid_t)
    avg_pixel_per_mm = float(
        (1.0 / float(mm_per_pixel_x) + 1.0 / float(mm_per_pixel_y)) / 2.0
    )
    out["pixel_size"] = {
        "mm_per_pixel_x": float(mm_per_pixel_x),
        "mm_per_pixel_y": float(mm_per_pixel_y),
        "avg_pixel_per_mm": avg_pixel_per_mm,
    }

    return out


def _run_signal_extractor_and_downstream(out: dict[str, Any]) -> dict[str, Any]:
    """Roda SignalExtractor + LeadIdentifier + canonicalize. Nao cacheado.
    Modifica `out` in-place adicionando: raw_lines_pre_sort, raw_lines,
    x_offset, lead_identifier_match, layouts_dict, signals_canon, etc.
    """
    avg_pixel_per_mm = out["pixel_size"]["avg_pixel_per_mm"]
    h_n, w_n = out["normalized"].shape[:2]

    # ----- Etapa 12: SignalExtractor (com flag USE_EXTRACTOR_V2 no adapter) -----
    signal_prob_t = torch.from_numpy(np.ascontiguousarray(out["signal_prob"])).float()
    raw_lines_t, x_offset = _signal_extractor_with_offset(signal_prob_t)
    out["raw_lines_pre_sort"] = raw_lines_t.cpu().numpy().astype(np.float64)
    out["n_lines_raw"] = int(raw_lines_t.shape[0])

    # ----- Etapa 12.5: sort+merge (replica logica interna do LeadIdentifier) -----
    identifier = _get_lead_identifier()
    if raw_lines_t.shape[0] > 1:
        raw_lines_t = identifier._merge_nonoverlapping_lines(raw_lines_t.clone())
    raw_lines_np = raw_lines_t.cpu().numpy().astype(np.float64)
    out["raw_lines"] = raw_lines_np
    out["x_offset"] = int(x_offset)
    out["n_lines"] = int(raw_lines_np.shape[0])
    logger.info("raw_lines: %d (pre-sort) -> %d (sort+merge)",
                out["n_lines_raw"], out["n_lines"])

    # ----- Etapa 13: LeadIdentifier -----
    text_prob_t = (
        torch.from_numpy(np.ascontiguousarray(out["text_prob"]))
        .unsqueeze(0).unsqueeze(0).float()
    )
    mv_per_mm = 1.0 / float(GAIN_DEFAULT)
    try:
        match = identifier(
            raw_lines_t.clone(),
            text_prob_t,
            avg_pixel_per_mm=avg_pixel_per_mm,
            threshold=0.8,
            mv_per_mm=mv_per_mm,
        )
    except Exception as e:
        logger.warning("LeadIdentifier falhou: %s: %s", type(e).__name__, e)
        match = {"layout": "standard_3x4_with_r1", "flip": False, "cost": float("nan"),
                 "leads": [], "n_detected": 0, "rows_in_layout": 0}
    out["lead_identifier_match"] = match

    # ----- Etapa 14: canonicalize -----
    layouts_dict = identifier.layouts if hasattr(identifier, "layouts") else {}
    out["layouts_dict"] = layouts_dict
    layout_name = match.get("layout") or "standard_3x4_with_r1"
    out["layout_name"] = layout_name
    cols_layout = (
        int(layouts_dict.get(layout_name, {}).get("layout", {}).get("cols", 4))
        if layout_name in layouts_dict else 4
    )
    out["cols"] = cols_layout

    signals, canonical_arr = _canonicalize_calibrated(
        raw_lines=raw_lines_np,
        layouts=layouts_dict,
        layout_name=str(layout_name),
        x_offset=int(x_offset),
        w_orig=int(out["signal_prob"].shape[1]),
        avg_pixel_per_mm=float(avg_pixel_per_mm),
        paper_speed=float(PAPER_SPEED_DEFAULT),
        voltage_gain=float(GAIN_DEFAULT),
    )
    out["signals_canon"] = signals
    out["canonical_arr"] = canonical_arr
    out["chunk_px"] = canonical_arr.shape[1] if canonical_arr.size else 0

    spc = _SECONDS_PER_CELL_BY_COLS.get(cols_layout, 10.0 / max(cols_layout, 1))
    out["seconds_per_cell"] = spc
    out["fs_eff"] = (out["chunk_px"] / spc) if spc > 0 else 0.0

    return out


def run_instrumented(image_path: Path, output_dir_for_ecg: Optional[Path] = None) -> dict[str, Any]:
    """Roda o pipeline capturando intermediarios. Usa cache se disponivel
    (so re-executa o SignalExtractor + downstream).

    Returns dict com todos os intermediarios necessarios pras visualizacoes.
    """
    cache_path = (
        _cache_file_path(output_dir_for_ecg) if output_dir_for_ecg else None
    )
    pre: Optional[dict[str, Any]] = None

    if USE_CACHE and cache_path is not None and cache_path.is_file():
        try:
            t0 = time.perf_counter()
            with cache_path.open("rb") as f:
                pre = pickle.load(f)
            logger.info(
                "Cache pre-SignalExtractor carregado de %s (%.1fs)",
                cache_path.name, time.perf_counter() - t0,
            )
        except Exception as e:
            logger.warning("Falha ao ler cache (%s) — recomputando", e)
            pre = None

    if pre is None:
        t0 = time.perf_counter()
        pre = _compute_pre_signal_extractor(image_path)
        logger.info(
            "Pre-SignalExtractor computado em %.1fs", time.perf_counter() - t0,
        )
        if USE_CACHE and cache_path is not None:
            try:
                output_dir_for_ecg.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as f:
                    pickle.dump({k: pre[k] for k in _CACHE_KEYS if k in pre}, f)
                logger.info("Cache salvo em %s", cache_path.name)
            except Exception as e:
                logger.warning("Falha ao salvar cache (%s)", e)

    out = dict(pre)
    t_se = time.perf_counter()
    out = _run_signal_extractor_and_downstream(out)
    logger.info("SignalExtractor+downstream em %.1fs", time.perf_counter() - t_se)
    return out


# ---------------------------------------------------------------------------
# Geradores das 16 imagens
# ---------------------------------------------------------------------------

def gen_01_original(data: dict, out_path: Path) -> None:
    img = data["original"]
    fig, _ = setup_fig("01. FOTO ORIGINAL (input bruto)", figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    h, w = img.shape[:2]
    save_fig(fig, out_path, footer=f"Resolucao: {w} x {h} pixels")


def gen_02_crop(data: dict, out_path: Path) -> None:
    orig = data["original"]
    cropped = data["cropped"]
    info = data["crop_info"]

    fig, _ = setup_fig("02. CROP + PERSPECTIVA (codigo nosso)", figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    h, w = orig.shape[:2]
    if info.get("method") == "perspective":
        pts_det = info.get("pts_detected")
        pts_exp = info.get("pts_expanded")
        if pts_det is not None and pts_exp is not None:
            pts_det_arr = np.array(pts_det)
            pts_exp_arr = np.array(pts_exp)
            # Detectado em verde
            poly_det = mpatches.Polygon(
                pts_det_arr, fill=False, edgecolor="#00cc44", linewidth=2.0,
                linestyle="--", label="cantos detectados (approxPolyDP)",
            )
            plt.gca().add_patch(poly_det)
            # Expandido em amarelo
            poly_exp = mpatches.Polygon(
                pts_exp_arr, fill=False, edgecolor="#ffaa00", linewidth=2.5,
                label=f"+{info.get('safety_margin_frac', 0)*100:.0f}% margem (usado no warp)",
            )
            plt.gca().add_patch(poly_exp)
            plt.legend(loc="upper right", fontsize=9)
    elif info.get("method") == "no_crop_fallback":
        plt.gca().add_patch(mpatches.Rectangle(
            (0, 0), w - 1, h - 1,
            fill=False, edgecolor="#cc0000", linewidth=3,
            label="Sem retangulo detectado — imagem inteira"
        ))
        plt.legend(loc="upper right", fontsize=9)
    plt.axis("off")
    plt.title(f"Foto original ({w} x {h})", fontsize=11)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    ch, cw = cropped.shape[:2]
    plt.title(f"Apos crop+perspectiva ({cw} x {ch})", fontsize=11)

    foot = (
        f"Metodo: {info.get('method', 'n/a')} | "
        f"margem seguranca: {info.get('safety_margin_frac', 0)*100:.0f}%"
    )
    save_fig(fig, out_path, footer=foot)


def gen_03_dotter(data: dict, out_path: Path) -> None:
    cropped = data["cropped"]
    keypoints = data["keypoints"]

    fig, _ = setup_fig(
        "03. DOTTER — Keypoints do grid (codigo nosso, UNet propria)",
        figsize=(14, 9),
    )
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    if len(keypoints) > 0:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=8, c="#00aaff",
                    alpha=0.7, edgecolors="none")
    plt.axis("off")
    save_fig(fig, out_path, footer=f"{len(keypoints)} keypoints detectados")


def gen_04_gridder(data: dict, out_path: Path) -> None:
    cropped = data["cropped"]
    grid = data["grid_matrix"]
    info = data["grid_info"]
    keypoints = data["keypoints"]

    fig, _ = setup_fig("04. GRIDDER — Malha regular (codigo nosso)", figsize=(14, 9))
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    if grid.size > 0 and grid.ndim == 3:
        rows, cols, _ = grid.shape
        for r in range(rows):
            row_pts = grid[r]
            valid = ~np.isnan(row_pts[:, 0])
            if valid.sum() >= 2:
                plt.plot(row_pts[valid, 0], row_pts[valid, 1],
                         color="#cc0000", linewidth=0.6, alpha=0.7)
        for c in range(cols):
            col_pts = grid[:, c]
            valid = ~np.isnan(col_pts[:, 0])
            if valid.sum() >= 2:
                plt.plot(col_pts[valid, 0], col_pts[valid, 1],
                         color="#0033cc", linewidth=0.6, alpha=0.7)

    if len(keypoints) > 0:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=4,
                    c="#ffff00", alpha=0.5, edgecolors="none")

    plt.axis("off")
    px_mm = info.get("px_per_mm", 0)
    shape = info.get("shape", (0, 0))
    save_fig(fig, out_path,
             footer=f"Grid: {shape[0]}x{shape[1]} pontos | px/mm: {px_mm:.2f}")


def gen_05_undistort(data: dict, out_path: Path) -> None:
    cropped = data["cropped"]
    normalized = data["normalized"]
    grid = data["grid_matrix"]
    applied = data["undistort_applied"]

    fig, _ = setup_fig("05. UNDISTORTION — Correcao local (codigo nosso)", figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    if grid.size > 0 and grid.ndim == 3:
        rows, cols, _ = grid.shape
        for r in range(rows):
            row_pts = grid[r]
            valid = ~np.isnan(row_pts[:, 0])
            if valid.sum() >= 2:
                plt.plot(row_pts[valid, 0], row_pts[valid, 1],
                         color="#cc0000", linewidth=0.5, alpha=0.6)
        for c in range(cols):
            col_pts = grid[:, c]
            valid = ~np.isnan(col_pts[:, 0])
            if valid.sum() >= 2:
                plt.plot(col_pts[valid, 0], col_pts[valid, 1],
                         color="#0033cc", linewidth=0.5, alpha=0.6)
    plt.axis("off")
    h, w = cropped.shape[:2]
    plt.title(f"ANTES (com malha) — {w}x{h}", fontsize=11)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    nh, nw = normalized.shape[:2]
    plt.title(f"DEPOIS — {nw}x{nh}", fontsize=11)

    status = "APLICADA" if applied else "PULADA (grid esparso < 100 keypoints)"
    save_fig(fig, out_path, footer=f"Status: {status}")


def gen_06_format(data: dict, out_path: Path) -> None:
    normalized = data["normalized"]
    layouts = data["layouts_dict"]
    layout_name = data["layout_name"]
    cols = data["cols"]
    raw_lines = data["raw_lines"]
    x_offset = data["x_offset"]
    chunk_px = data["chunk_px"]

    fig, _ = setup_fig(
        "06. FORMAT DETECTOR (Stenhede — substitui codigo nosso)",
        figsize=(14, 9),
    )
    plt.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))

    # Desenhar bbox de cada cell baseado em raw_lines (Y) + chunk_px
    h_n, w_n = normalized.shape[:2]
    n_lines, lines_w = raw_lines.shape

    if layout_name in layouts and chunk_px > 0:
        layout_def = layouts[layout_name]
        leads_def = layout_def["leads"]
        is_matrix = isinstance(leads_def[0], list) if leads_def else False
        rows_iter = leads_def if is_matrix else [leads_def]

        for row_idx, row_leads in enumerate(rows_iter):
            if row_idx >= n_lines:
                break
            if not isinstance(row_leads, list):
                row_leads = [row_leads]
            line_y = raw_lines[row_idx]
            valid = ~np.isnan(line_y)
            if valid.sum() < 5:
                continue
            y_med = float(np.nanmedian(line_y))
            band_h = chunk_px * 0.4
            for col_idx, lead in enumerate(row_leads):
                x_start = x_offset + col_idx * chunk_px
                x_end = x_start + chunk_px
                if x_start >= w_n:
                    continue
                rect = mpatches.Rectangle(
                    (x_start, y_med - band_h),
                    min(chunk_px, w_n - x_start),
                    band_h * 2,
                    fill=False, edgecolor=LEAD_COLORS[col_idx % len(LEAD_COLORS)],
                    linewidth=2, alpha=0.7,
                )
                plt.gca().add_patch(rect)
                plt.text(x_start + chunk_px * 0.5, y_med - band_h - 10,
                         str(lead).lstrip("-"),
                         ha="center", color="black", fontsize=10,
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.axis("off")
    n_rows = (
        len(layouts.get(layout_name, {}).get("leads", []))
        if layout_name in layouts else 0
    )
    save_fig(fig, out_path,
             footer=f"Formato detectado: {layout_name} ({n_rows} linhas x {cols} colunas) | "
                    f"Metodo: lead_name_unet + matching geometrico")


def gen_07_calibrador_removido(out_path: Path) -> None:
    text_only_image(out_path, [
        "07. CALIBRADOR — REMOVIDO DO PIPELINE",
        "",
        "Motivo: o valor era descartado downstream",
        "(o PixelSizeFinder do Stenhede no passo 10 e quem define px/mm).",
        "",
        "Arquivo: pipeline/digitize/calibrator.py (existe mas nao e mais chamado).",
    ])


def gen_08_unet(data: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 11), dpi=110)
    fig.suptitle(
        "08. UNET SEGMENTACAO — 4 canais (Stenhede original, NAO modificado)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    titles = ["Canal 0: GRID (probabilidade)",
              "Canal 1: TEXTO/FUNDO",
              "Canal 2: TRACADO ECG",
              "Canal 3: FUNDO BRANCO"]
    keys = ["grid_prob", "text_prob", "signal_prob", "bg_prob"]
    cmaps = ["Blues", "Oranges", "Reds", "Greys"]

    for i, (k, t, cm) in enumerate(zip(keys, titles, cmaps)):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(data[k], cmap=cm, vmin=0, vmax=1)
        ax.set_title(t, fontsize=11)
        ax.axis("off")

    sig = data["signal_prob"]
    n_signal = int(np.sum(sig > 0.3))
    pct = 100.0 * n_signal / sig.size
    fig.text(0.5, 0.01,
             f"Pixels de tracado (>0.3): {n_signal} ({pct:.1f}% da imagem)",
             ha="center", fontsize=10, color="#444")
    fig.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def gen_09_cropper_desativado(out_path: Path) -> None:
    text_only_image(out_path, [
        "09. CROPPER STENHEDE — DESATIVADO",
        "",
        "use_cropper=False (default no nosso adapter)",
        "",
        "Motivo: ja fizemos crop + correcao de perspectiva no passo 02",
        "com codigo nosso (preprocess() do ECGDigitizer).",
    ])


def gen_10_pixel_size(data: dict, out_path: Path) -> None:
    normalized = data["normalized"]
    grid_prob = data["grid_prob"]
    psz = data["pixel_size"]

    fig = plt.figure(figsize=(16, 9), dpi=110)
    fig.suptitle(
        "10. PIXEL SIZE FINDER (Stenhede — PARAMETRO MODIFICADO)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    ax1.imshow(grid_prob, cmap="cool", alpha=0.45)
    ax1.set_title("Imagem + grid_prob (overlay)", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    proj_x = grid_prob.sum(axis=0)
    proj_y = grid_prob.sum(axis=1)
    ax2.plot(proj_x / max(proj_x.max(), 1e-6), label="projecao X (vertical)",
             color="#2266cc", linewidth=0.8)
    ax2.plot(proj_y / max(proj_y.max(), 1e-6), label="projecao Y (horizontal)",
             color="#cc6622", linewidth=0.8)
    ax2.set_title("Projecoes do grid_prob (auto-correlacao usada p/ medir periodicidade)", fontsize=10)
    ax2.set_xlabel("pixel")
    ax2.set_ylabel("intensidade normalizada")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.text(0.5, 0.01,
             f"px/mm Stenhede: {psz['avg_pixel_per_mm']:.3f} | "
             f"mm/px X: {psz['mm_per_pixel_x']:.4f} | mm/px Y: {psz['mm_per_pixel_y']:.4f} | "
             f"⚠️ lower_grid_line_factor: 0.5 (DEFAULT 0.3)",
             ha="center", fontsize=9, color="#444")
    fig.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def gen_11_dewarper_desativado(out_path: Path) -> None:
    text_only_image(out_path, [
        "11. DEWARPER STENHEDE — DESATIVADO",
        "",
        "Nunca chamado (apply_dewarping=False no adapter)",
        "",
        "Motivo: ja fizemos undistortion no passo 05",
        "com codigo nosso (homografia por celula do Gridder).",
    ])


def gen_12_signal_extractor(data: dict, out_path: Path) -> None:
    """Mostra o output BRUTO do SignalExtractor (pre sort+merge), pra ver o
    que o algoritmo do Stenhede de fato extraiu antes do reordenamento.
    """
    normalized = data["normalized"]
    raw_lines = data.get("raw_lines_pre_sort", data["raw_lines"])
    x_offset = data["x_offset"]
    n_lines_raw = data.get("n_lines_raw", data["n_lines"])
    n_lines_after = data["n_lines"]

    fig, _ = setup_fig(
        "12. SIGNAL EXTRACTOR — Linhas extraidas (Stenhede, com 5 params tunados)",
        figsize=(16, 9),
    )
    plt.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))

    h_n, w_n = normalized.shape[:2]

    if raw_lines.ndim == 2 and raw_lines.shape[0] > 0:
        n_rows_lines, lines_w = raw_lines.shape
        for i in range(n_rows_lines):
            line = raw_lines[i]
            xs = np.arange(line.shape[0]).astype(np.float64)
            if line.shape[0] != w_n:
                xs = xs + x_offset
            ys = line.astype(np.float64).copy()
            ys[ys == 0] = np.nan
            valid = ~np.isnan(ys)
            if valid.sum() < 5:
                continue
            color = LEAD_COLORS[i % len(LEAD_COLORS)]
            # Plota COM NaN — matplotlib quebra a linha em NaN, evitando
            # "saltos" visuais entre pontos validos distantes.
            plt.plot(xs, ys, color=color, linewidth=0.7,
                     alpha=0.85, label=f"Linha {i+1}")

    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.axis("off")
    foot = (f"{n_lines_raw} linhas extraidas pelo SignalExtractor "
            f"(apos sort+merge no proximo passo: {n_lines_after} linhas).\n"
            f"Cores aqui sao na ORDEM DE EXTRACAO (nao Y-sorted) — apenas pra "
            f"distinguir visualmente.")
    save_fig(fig, out_path, footer=foot)


def gen_13_lead_identifier(data: dict, out_path: Path) -> None:
    normalized = data["normalized"]
    layouts = data["layouts_dict"]
    layout_name = data["layout_name"]
    raw_lines = data["raw_lines"]
    x_offset = data["x_offset"]
    chunk_px = data["chunk_px"]
    match = data["lead_identifier_match"]

    fig, _ = setup_fig(
        "13. LEAD IDENTIFIER (Stenhede — 4 PARAMETROS MODIFICADOS)",
        figsize=(16, 10),
    )
    plt.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))

    h_n, w_n = normalized.shape[:2]
    n_lines = raw_lines.shape[0]

    if layout_name in layouts and chunk_px > 0:
        layout_def = layouts[layout_name]
        leads_def = layout_def["leads"]
        rhythm_leads = list(layout_def.get("rhythm_leads", []))
        is_matrix = isinstance(leads_def[0], list) if leads_def else False
        rows_iter = leads_def if is_matrix else [leads_def]

        cidx = 0
        for row_idx, row_leads in enumerate(rows_iter):
            if row_idx >= n_lines:
                break
            if not isinstance(row_leads, list):
                row_leads = [row_leads]
            line_y = raw_lines[row_idx]
            valid = ~np.isnan(line_y)
            if valid.sum() < 5:
                continue
            for col_idx, lead in enumerate(row_leads):
                x_start = x_offset + col_idx * chunk_px
                x_end = min(x_start + chunk_px, w_n)
                if x_start >= w_n:
                    continue
                seg_x = np.arange(x_start, x_end)
                if line_y.shape[0] == w_n:
                    seg_y = line_y[x_start:x_end]
                else:
                    sl_start = x_start - x_offset
                    sl_end = sl_start + (x_end - x_start)
                    if sl_end > line_y.shape[0]:
                        sl_end = line_y.shape[0]
                        seg_x = seg_x[:sl_end - sl_start]
                    seg_y = line_y[sl_start:sl_end]
                seg_y = seg_y.astype(float).copy()
                seg_y[seg_y == 0] = np.nan
                valid_seg = ~np.isnan(seg_y)
                color = LEAD_COLORS[cidx % len(LEAD_COLORS)]
                cidx += 1
                if valid_seg.sum() > 5:
                    plt.plot(seg_x[valid_seg], seg_y[valid_seg],
                             color=color, linewidth=1.0, alpha=0.85)
                # bbox + nome
                y_med = float(np.nanmedian(seg_y))
                band_h = chunk_px * 0.4
                rect = mpatches.Rectangle(
                    (x_start, y_med - band_h),
                    x_end - x_start, band_h * 2,
                    fill=False, edgecolor=color, linewidth=1.5, alpha=0.5,
                )
                plt.gca().add_patch(rect)
                plt.text(x_start + (x_end - x_start) * 0.5, y_med - band_h - 12,
                         str(lead).lstrip("-"),
                         ha="center", color="white", fontsize=10, fontweight="bold",
                         bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=2))

        # rhythm strip
        n_main_rows = len(rows_iter)
        for k, _rhy in enumerate(rhythm_leads):
            li = n_main_rows + k
            if li >= n_lines:
                break
            line_y = raw_lines[li]
            valid = ~np.isnan(line_y)
            if valid.sum() < 5:
                continue
            xs = np.arange(line_y.shape[0])
            if line_y.shape[0] != w_n:
                xs = xs + x_offset
            color = "#000000"
            plt.plot(xs[valid], line_y[valid],
                     color=color, linewidth=1.0, alpha=0.85)
            y_med = float(np.nanmedian(line_y[valid]))
            plt.text(20, y_med, "DII (rhythm)",
                     ha="left", color="white", fontsize=10, fontweight="bold",
                     bbox=dict(facecolor="black", alpha=0.85, edgecolor="none"))

    plt.axis("off")
    foot = (
        f"Layout: {layout_name} | flip={match.get('flip')} | "
        f"cost={match.get('cost', float('nan')):.3f} | "
        f"n_detected={match.get('n_detected', 0)}\n"
        f"Param. modificados: lead_layouts_all (vs reduced) | "
        f"required_valid_samples=2 (vs 3) | target_num_samples=W (vs 5000) | threshold=0.8 (=)"
    )
    save_fig(fig, out_path, footer=foot)


def gen_14_split(data: dict, out_path: Path) -> None:
    layouts = data["layouts_dict"]
    layout_name = data["layout_name"]
    raw_lines = data["raw_lines"]
    x_offset = data["x_offset"]
    chunk_px = data["chunk_px"]
    cols = data["cols"]
    spc = data["seconds_per_cell"]

    if layout_name not in layouts or chunk_px <= 0:
        text_only_image(out_path, [
            "14. SPLIT EM DERIVACOES — N/A",
            "",
            f"Layout '{layout_name}' nao encontrado ou chunk_px=0",
        ])
        return

    layout_def = layouts[layout_name]
    leads_def = layout_def["leads"]
    rhythm_leads = list(layout_def.get("rhythm_leads", []))
    is_matrix = isinstance(leads_def[0], list) if leads_def else False
    rows_iter = leads_def if is_matrix else [leads_def]

    n_main_rows = len(rows_iter)
    n_rhythm = len(rhythm_leads)
    n_lines = raw_lines.shape[0]
    total_rows = min(n_main_rows + n_rhythm, n_lines)

    fig = plt.figure(figsize=(16, 3 * total_rows + 1), dpi=110)
    fig.suptitle(
        "14. SPLIT EM DERIVACOES (codigo NOSSO — substitui canonical_lines do Stenhede)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    plot_idx = 0
    line_descs: list[str] = []
    for row_idx, row_leads in enumerate(rows_iter):
        if row_idx >= n_lines:
            break
        if not isinstance(row_leads, list):
            row_leads = [row_leads]

        plot_idx += 1
        ax = fig.add_subplot(total_rows, 1, plot_idx)
        line = raw_lines[row_idx].copy()
        line[line == 0] = np.nan
        xs = np.arange(line.shape[0])
        ax.plot(xs, line, color="#222", linewidth=0.7)
        # cortes verticais
        for ci in range(len(row_leads) + 1):
            ax.axvline(ci * chunk_px, color="#cc0000", linewidth=1.2, alpha=0.7)
        # nomes
        for ci, lead in enumerate(row_leads):
            ax.text((ci + 0.5) * chunk_px, ax.get_ylim()[0],
                    str(lead).lstrip("-"),
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color="#0066cc")
            line_descs.append(
                f"Lead {str(lead).lstrip('-')}: amostras {ci*chunk_px} a {(ci+1)*chunk_px} | duracao: {spc:.1f}s"
            )
        ax.invert_yaxis()
        ax.set_title(f"Linha {row_idx+1}: {len(row_leads)} derivacoes", fontsize=10)
        ax.set_xlabel("pixel X (na largura trimmed)", fontsize=9)
        ax.set_ylabel("pixel Y", fontsize=9)
        ax.grid(alpha=0.2)

    for k, _rhy in enumerate(rhythm_leads):
        li = n_main_rows + k
        if li >= n_lines:
            break
        plot_idx += 1
        ax = fig.add_subplot(total_rows, 1, plot_idx)
        line = raw_lines[li].copy()
        line[line == 0] = np.nan
        ax.plot(np.arange(line.shape[0]), line, color="#222", linewidth=0.7)
        ax.invert_yaxis()
        ax.set_title(f"Rhythm strip {k+1}: derivacao continua de 10s", fontsize=10)
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    foot = (f"Metodo nosso: corte por largura calibrada em mm "
            f"(chunk_px={chunk_px}, seconds/cell={spc:.1f}s, cols={cols})\n"
            f"Stenhede original: cortaria por W//cols sem calibracao")
    fig.text(0.5, 0.005, foot, ha="center", fontsize=9, color="#444")
    fig.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def gen_15_uv_conversion(data: dict, out_path: Path) -> None:
    signals = data["signals_canon"]
    psz = data["pixel_size"]
    raw_lines = data["raw_lines"]
    layouts = data["layouts_dict"]
    layout_name = data["layout_name"]
    chunk_px = data["chunk_px"]
    spc = data["seconds_per_cell"]

    avg_px_mm = psz["avg_pixel_per_mm"]
    px_per_1mv = avg_px_mm * GAIN_DEFAULT  # ex: 12.5*10 = 125 px/mV

    n_subplots = 12
    fig = plt.figure(figsize=(16, 14), dpi=110)
    fig.suptitle("15. CONVERSAO PIXEL → µV (codigo NOSSO)",
                 fontsize=14, fontweight="bold", y=0.995)

    leads_to_plot = LEAD_CHANNEL_ORDER[:12]

    # Mapa lead -> (row_idx, col_idx) baseado no layout
    lead_pos: dict[str, tuple[int, int]] = {}
    if layout_name in layouts:
        leads_def = layouts[layout_name]["leads"]
        is_matrix = isinstance(leads_def[0], list) if leads_def else False
        rows_iter = leads_def if is_matrix else [leads_def]
        for ri, rl in enumerate(rows_iter):
            if not isinstance(rl, list):
                rl = [rl]
            for ci, lead in enumerate(rl):
                lead_pos[str(lead).lstrip("-")] = (ri, ci)

    n_rows_lines = raw_lines.shape[0]

    for i, lead in enumerate(leads_to_plot):
        ax = fig.add_subplot(4, 3, i + 1)
        sig_uv = signals.get(lead)
        if sig_uv is None or sig_uv.size == 0:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(lead, fontsize=10)
            ax.axis("off")
            continue

        ts = np.arange(len(sig_uv)) / max(data["fs_eff"], 1.0)

        ax.plot(ts, sig_uv, color="#1f78b4", linewidth=0.8, label="µV")
        ax.axhline(0, color="#888", linewidth=0.5, linestyle="--")

        # eixo Y secundario em pixels
        ax2 = ax.twinx()
        if avg_px_mm > 0:
            sig_px = -sig_uv * (avg_px_mm / 1000.0) * GAIN_DEFAULT
            ax2.plot(ts, sig_px, color="#aaa", linewidth=0.4, alpha=0.0)
            ax2.set_ylabel("Y (px)", fontsize=8, color="#aaa")
            ax2.tick_params(labelsize=7)

        if sig_uv.size > 0 and not np.all(np.isnan(sig_uv)):
            mn, mx = float(np.nanmin(sig_uv)), float(np.nanmax(sig_uv))
        else:
            mn, mx = 0.0, 0.0

        # baseline pixel approx (de raw_lines)
        baseline_px = float("nan")
        if lead in lead_pos:
            ri, _ = lead_pos[lead]
            if ri < n_rows_lines:
                line = raw_lines[ri]
                line[line == 0] = np.nan
                if np.any(~np.isnan(line)):
                    baseline_px = float(np.nanmean(line))

        ax.set_title(f"{lead}: [{mn:.0f}, {mx:.0f}] µV | baseline≈{baseline_px:.0f} px",
                     fontsize=9)
        ax.set_xlabel("tempo (s)", fontsize=8)
        ax.set_ylabel("µV", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    foot = (f"Escala: 1mV = {px_per_1mv:.1f} px = 10mm | "
            f"Inversao Y: Sim (Y cresce p/ baixo) | "
            f"Centra na media + nega + multiplica por (mV/mm)/px_per_mm * 1000")
    fig.text(0.5, 0.005, foot, ha="center", fontsize=9, color="#444")
    fig.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def gen_16_validation(data: dict, out_path: Path) -> None:
    """Sobrepoe sinal extraido (em pixel-Y) sobre tracado original.

    Usamos raw_lines (em pixel-Y) ja em coords da imagem pra desenhar a curva
    em vermelho exatamente em cima do tracado preto.
    """
    normalized = data["normalized"]
    layouts = data["layouts_dict"]
    layout_name = data["layout_name"]
    raw_lines = data["raw_lines"]
    x_offset = data["x_offset"]
    chunk_px = data["chunk_px"]
    cols = data["cols"]
    signals = data["signals_canon"]

    h_n, w_n = normalized.shape[:2]

    if layout_name not in layouts or chunk_px <= 0:
        text_only_image(out_path, [
            "16. VALIDACAO FINAL — N/A",
            "",
            f"Layout invalido ou chunk_px=0 ({layout_name}, {chunk_px})",
        ])
        return

    layout_def = layouts[layout_name]
    leads_def = layout_def["leads"]
    rhythm_leads = list(layout_def.get("rhythm_leads", []))
    is_matrix = isinstance(leads_def[0], list) if leads_def else False
    rows_iter = leads_def if is_matrix else [leads_def]

    # Layout do plot: cada cell = subplot
    n_rows_layout = len(rows_iter)
    cols_layout = max(len(r) if isinstance(r, list) else 1 for r in rows_iter)
    n_rhythm = min(len(rhythm_leads), raw_lines.shape[0] - n_rows_layout)
    total_subplot_rows = n_rows_layout + n_rhythm

    fig = plt.figure(figsize=(16, 3.5 * total_subplot_rows + 1), dpi=100)
    fig.suptitle("16. VALIDACAO FINAL — Sinal extraido vs tracado original",
                 fontsize=14, fontweight="bold", y=0.995)

    n_lines = raw_lines.shape[0]
    valid_pixels_total = 0
    expected_total = 0
    foot_lines: list[str] = []

    plot_idx = 0
    for row_idx, row_leads in enumerate(rows_iter):
        if row_idx >= n_lines:
            break
        if not isinstance(row_leads, list):
            row_leads = [row_leads]
        line_y = raw_lines[row_idx].copy()
        line_y[line_y == 0] = np.nan

        for col_idx, lead_raw in enumerate(row_leads):
            plot_idx += 1
            lead = str(lead_raw).lstrip("-")
            ax = fig.add_subplot(total_subplot_rows, cols_layout, plot_idx)

            x_start = x_offset + col_idx * chunk_px
            x_end = min(x_start + chunk_px, w_n)
            if x_start >= w_n:
                ax.text(0.5, 0.5, f"{lead}: fora da imagem",
                        ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue

            # Tracado original como background
            valid_y = ~np.isnan(line_y)
            if valid_y.sum() < 5:
                ax.text(0.5, 0.5, f"{lead}: sem sinal", ha="center", va="center",
                        transform=ax.transAxes)
                ax.axis("off")
                continue
            y_med = float(np.nanmedian(line_y))
            band_h = int(chunk_px * 0.5)
            y1 = max(0, int(y_med - band_h))
            y2 = min(h_n, int(y_med + band_h))
            crop_img = normalized[y1:y2, x_start:x_end]
            if crop_img.size == 0:
                ax.axis("off")
                continue
            ax.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB),
                      extent=[x_start, x_end, y2, y1], aspect="auto")

            # Sinal extraido em vermelho (em coords da imagem, mesma)
            if line_y.shape[0] == w_n:
                seg_y = line_y[x_start:x_end]
                seg_x = np.arange(x_start, x_end)
            else:
                sl_start = x_start - x_offset
                sl_end = sl_start + (x_end - x_start)
                if sl_end > line_y.shape[0]:
                    sl_end = line_y.shape[0]
                seg_y = line_y[sl_start:sl_end]
                seg_x = np.arange(x_start, x_start + (sl_end - sl_start))

            valid_seg = ~np.isnan(seg_y)
            ax.plot(seg_x[valid_seg], seg_y[valid_seg],
                    color="#ff0000", linewidth=0.8, alpha=0.85)

            # Stats
            sig_uv = signals.get(lead, np.array([]))
            n_samples = sig_uv.size
            n_nan = int(np.sum(np.isnan(sig_uv))) if n_samples else 0
            pct_nan = 100.0 * n_nan / max(n_samples, 1)
            if n_samples > 0 and not np.all(np.isnan(sig_uv)):
                mn = float(np.nanmin(sig_uv))
                mx = float(np.nanmax(sig_uv))
            else:
                mn = mx = 0.0
            valid_pixels_total += int(valid_seg.sum())
            expected_total += int(seg_y.size)

            ax.set_title(f"{lead} | n={n_samples} | [{mn:.0f}, {mx:.0f}] µV | NaN: {pct_nan:.0f}%",
                         fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            foot_lines.append(f"{lead}: {valid_seg.sum()}/{seg_y.size} colunas validas")

    # Rhythm strip
    n_main_rows = len(rows_iter)
    for k in range(n_rhythm):
        li = n_main_rows + k
        if li >= n_lines:
            break
        plot_idx += cols_layout - (plot_idx % cols_layout) if (plot_idx % cols_layout) else 0
        plot_idx += 1
        # Span all cols
        ax = plt.subplot2grid((total_subplot_rows, cols_layout),
                              (n_main_rows + k, 0), colspan=cols_layout, fig=fig)
        line_y = raw_lines[li].copy()
        line_y[line_y == 0] = np.nan
        valid_y = ~np.isnan(line_y)
        if valid_y.sum() < 5:
            ax.axis("off")
            continue
        y_med = float(np.nanmedian(line_y[valid_y]))
        band_h = int(chunk_px * 0.5)
        y1 = max(0, int(y_med - band_h))
        y2 = min(h_n, int(y_med + band_h))

        # Determinar x range
        if line_y.shape[0] == w_n:
            x_full_start, x_full_end = 0, w_n
            xs = np.arange(w_n)
        else:
            x_full_start = x_offset
            x_full_end = x_offset + line_y.shape[0]
            xs = np.arange(x_full_start, x_full_end)

        crop_img = normalized[y1:y2, x_full_start:x_full_end]
        ax.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB),
                  extent=[x_full_start, x_full_end, y2, y1], aspect="auto")
        ax.plot(xs[valid_y[:xs.shape[0]]] if valid_y.shape[0] >= xs.shape[0] else xs,
                line_y[valid_y[:xs.shape[0]]] if valid_y.shape[0] >= xs.shape[0] else line_y[valid_y],
                color="#ff0000", linewidth=0.8)
        ax.set_title("DII (rhythm strip)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    cov_pct = 100.0 * valid_pixels_total / max(expected_total, 1)
    foot = f"COBERTURA TOTAL: {cov_pct:.1f}% das colunas tem sinal valido"
    fig.text(0.5, 0.005, foot, ha="center", fontsize=10, color="#444",
             fontweight="bold")
    fig.savefig(str(out_path), bbox_inches="tight", dpi=100, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_one_ecg(stem: str, output_dir: Path) -> dict[str, Any]:
    """Processa 1 ECG e gera 16 imagens. Retorna info para o resumo."""
    info: dict[str, Any] = {"name": stem, "ok": False, "error": None}

    img_path = find_image(stem)
    if img_path is None:
        info["error"] = f"Imagem '{stem}' nao encontrada em {INPUT_DIR}"
        logger.error(info["error"])
        return info

    info["input_path"] = str(img_path)
    sub = output_dir / stem
    sub.mkdir(parents=True, exist_ok=True)

    logger.info("==================== %s ====================", stem)
    t0 = time.perf_counter()

    try:
        data = run_instrumented(img_path, output_dir_for_ecg=sub)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()
        logger.error("Pipeline falhou: %s", info["error"])
        logger.error(info["traceback"])
        return info

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline OK em %.1fs — gerando 16 imagens...", elapsed)
    info["pipeline_time_s"] = elapsed

    # Gerar todas as 16
    try:
        gen_01_original(data, sub / "01_original.png")
        gen_02_crop(data, sub / "02_crop.png")
        gen_03_dotter(data, sub / "03_dotter.png")
        gen_04_gridder(data, sub / "04_gridder.png")
        gen_05_undistort(data, sub / "05_undistort.png")
        gen_06_format(data, sub / "06_format_stenhede.png")
        gen_07_calibrador_removido(sub / "07_calibrador_removido.png")
        gen_08_unet(data, sub / "08_unet_4canais.png")
        gen_09_cropper_desativado(sub / "09_cropper_desativado.png")
        gen_10_pixel_size(data, sub / "10_pixel_size.png")
        gen_11_dewarper_desativado(sub / "11_dewarper_desativado.png")
        gen_12_signal_extractor(data, sub / f"12_signal_extractor{GEN12_SUFFIX}.png")
        gen_13_lead_identifier(data, sub / "13_lead_identifier.png")
        gen_14_split(data, sub / "14_split.png")
        gen_15_uv_conversion(data, sub / "15_uv_conversion.png")
        gen_16_validation(data, sub / "16_validacao_final.png")
    except Exception as e:
        info["error"] = f"Falha em gerador: {type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()
        logger.error(info["error"])
        logger.error(info["traceback"])
        return info

    info["ok"] = True
    info["layout_name"] = data["layout_name"]
    info["n_lines"] = data["n_lines"]
    info["chunk_px"] = data["chunk_px"]
    info["fs_eff"] = data["fs_eff"]
    info["px_per_mm_stenhede"] = data["pixel_size"]["avg_pixel_per_mm"]
    info["px_per_mm_gridder"] = data["grid_info"].get("px_per_mm", 0.0)
    info["n_keypoints"] = len(data["keypoints"])
    info["match_cost"] = data["lead_identifier_match"].get("cost", float("nan"))
    return info


def write_summary(infos: list[dict], output_dir: Path) -> None:
    summary_path = output_dir / "resumo_validacao.txt"
    lines = [
        "PIPELINE ProECG — Resumo da validacao visual",
        "=" * 60,
        "",
        "AJUSTES APLICADOS NESTA RODADA:",
        "  1. Format Detector: trocado de codigo nosso -> Stenhede (LeadIdentifier)",
        "  2. Calibrador: removido do pipeline (arquivo mantido, chamada desativada)",
        "",
        "Etapas com codigo NOSSO: 1, 2, 3, 4, 5, 14, 15",
        "Etapas com codigo STENHEDE: 6, 8, 10, 12, 13",
        "Etapas DESATIVADAS: 7 (Calibrador), 9 (Cropper), 11 (Dewarper)",
        "Etapas com parametros MODIFICADOS: 10 (PixelSizeFinder), 13 (LeadIdentifier)",
        "",
        "PARAMETROS ALTERADOS:",
        "  PixelSizeFinder.lower_grid_line_factor : 0.5    (default 0.3)",
        "  LeadIdentifier.layouts                  : all    (default reduced)",
        "  LeadIdentifier.required_valid_samples   : 2      (default 3)",
        "  LeadIdentifier.target_num_samples       : W      (default 5000)",
        "  LeadIdentifier.threshold (text_prob)    : 0.8    (default 0.8 — sem mudanca)",
        "",
        "RESULTADOS POR ECG:",
        "-" * 60,
    ]
    for info in infos:
        lines.append(f"\n{info['name']}:")
        if not info["ok"]:
            lines.append(f"  STATUS: FALHOU — {info.get('error', '?')}")
            continue
        lines.append(f"  STATUS: OK ({info.get('pipeline_time_s', 0):.1f}s)")
        lines.append(f"  Input: {info.get('input_path', '?')}")
        lines.append(f"  Keypoints (Dotter): {info.get('n_keypoints', 0)}")
        lines.append(f"  px/mm Gridder (nosso): {info.get('px_per_mm_gridder', 0):.3f}")
        lines.append(f"  px/mm Stenhede (efetivo): {info.get('px_per_mm_stenhede', 0):.3f}")
        lines.append(f"  Layout detectado: {info.get('layout_name', '?')}")
        lines.append(f"  Linhas extraidas: {info.get('n_lines', 0)}")
        lines.append(f"  Chunk px / cell: {info.get('chunk_px', 0)}")
        lines.append(f"  fs efetivo: {info.get('fs_eff', 0):.1f} Hz")
        lines.append(f"  Match cost (LeadIdentifier): {info.get('match_cost', 0):.4f}")

    lines.append("\n\nOBSERVACOES:")
    lines.append(
        "  - Imagem 16 sobrepoe o sinal vermelho diretamente sobre o tracado preto"
    )
    lines.append("    da imagem undistorted, em coords de pixel. Misalignment >2px sinaliza")
    lines.append("    erro num passo upstream (split, px/mm, etc).")
    lines.append(
        "  - Diferenca px/mm Gridder vs Stenhede pode indicar problema de calibracao"
    )
    lines.append("    se for grande (>10%).")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Resumo salvo em %s", summary_path)


def main() -> int:
    if not INPUT_DIR.is_dir():
        logger.error("Pasta de entrada nao encontrada: %s", INPUT_DIR)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", OUTPUT_DIR)

    infos: list[dict] = []
    for stem in ECG_NAMES:
        info = process_one_ecg(stem, OUTPUT_DIR)
        infos.append(info)

    write_summary(infos, OUTPUT_DIR)

    n_ok = sum(1 for i in infos if i["ok"])
    logger.info("Concluido: %d/%d ECGs com sucesso", n_ok, len(infos))
    return 0 if n_ok == len(infos) else 2


if __name__ == "__main__":
    sys.exit(main())
