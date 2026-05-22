"""
Teste combinado: 3 detectores de formato lado a lado
=====================================================

Para cada imagem, roda os 3 detectores e combina:

  1. detect_bands  — conta bandas horizontais na máscara binária (canal 2)
  2. LeadIdentifier — Stenhede (matching húngaro contra layouts conhecidos)
  3. OCR           — easyocr nos labels (I, II, V1-V6, aVR/L/F)

Combinação (prioridade):
  detect_bands > LeadIdentifier > OCR

Saída: overlay PNG com bboxes precisas (X e Y trim pelo signal_binary)
e header mostrando o resultado dos 3 detectores + decisão final.

Reusa código de test_lead_identifier_bboxes.py, test_ocr_format_detection.py
e pipeline_completo_v1.py — NÃO modifica arquivos do pipeline.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))
sys.path.insert(0, str(MODAL_ROOT.parent))

# Reuso direto dos testes anteriores e do pipeline
from pipeline.pipeline_completo_v1 import (  # noqa: E402
    detect_bands,
    find_trace_bounds_per_band,
)
from pipeline.digitize.stenhede_adapter import (  # noqa: E402
    _SECONDS_PER_CELL_BY_COLS,
    _get_lead_identifier,
    _get_pixel_size_finder,
    _signal_extractor_with_offset,
    get_unet_feature_maps,
)
from pipeline.test_lead_identifier_bboxes import (  # noqa: E402
    normalize_image,
)
from pipeline.test_ocr_format_detection import (  # noqa: E402
    CANONICAL_LEADS,
    detect_format as ocr_detect_format,
    extract_lead_labels,
    group_into_rows_and_cols,
    run_ocr,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("test_combined_format")


HOME = Path.home()
OUT_DIR = HOME / "Desktop" / "Projeto ECG" / "teste_formato"

DEFAULT_IMAGES = [
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1461.jpg",
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1310.jpg",
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1400.jpg",
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1383.jpg",
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1454.jpg",
]

SIGNAL_THRESHOLD = 0.3        # binarização do canal 2 (igual pipeline_completo)
QRS_MARGIN_FRAC = 0.05        # 5% da altura inicial como folga vertical

# Layout default de leads por formato (sem texto do OCR)
LAYOUT_3X4 = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
LAYOUT_6X2 = [
    ["I", "V1"], ["II", "V2"], ["III", "V3"],
    ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"],
]
LAYOUT_12X1 = [["I"], ["II"], ["III"], ["aVR"], ["aVL"], ["aVF"],
               ["V1"], ["V2"], ["V3"], ["V4"], ["V5"], ["V6"]]

ROW_COLORS_BGR = [
    (80, 80, 255),    # 0 vermelho
    (80, 200, 80),    # 1 verde
    (255, 80, 80),    # 2 azul
    (0, 200, 220),    # 3 amarelo (rhythm)
    (200, 100, 200),  # 4 magenta
    (200, 200, 100),  # 5 ciano
    (180, 130, 30),
    (130, 0, 130),
    (0, 130, 130),
    (180, 180, 180),
    (50, 200, 200),
    (200, 50, 200),
]


# ---------------------------------------------------------------------------
# Detector 1: detect_bands
# ---------------------------------------------------------------------------

def detector_bandas(signal_binary: np.ndarray) -> dict:
    """Conta bandas e propõe formato.

    Regra:
      <= 5 bandas -> 3x4
      <= 8 bandas -> 6x2
      >  8 bandas -> 12x1
    Rhythm:
      3x4: se n_bands >= 4 -> +(n_bands - 3) rhythm
      6x2: se n_bands >= 7 -> +(n_bands - 6) rhythm
      12x1: sem rhythm
    """
    bands = detect_bands(signal_binary)
    n = len(bands)
    if n == 0:
        return {"format": "unknown", "n_bands": 0, "bands": [],
                "n_main": 0, "n_cols": 0, "n_rhythm": 0}
    if n <= 5:
        base, n_main, n_cols = "3x4", 3, 4
        n_rhythm = max(0, n - 3)
    elif n <= 8:
        base, n_main, n_cols = "6x2", 6, 2
        n_rhythm = max(0, n - 6)
    else:
        base, n_main, n_cols = "12x1", 12, 1
        n_rhythm = 0
    fmt = base + (f"+{n_rhythm}" if n_rhythm > 0 else "")
    return {
        "format": fmt, "base": base, "n_bands": n, "bands": bands,
        "n_main": n_main, "n_cols": n_cols, "n_rhythm": n_rhythm,
    }


# ---------------------------------------------------------------------------
# Detector 2: LeadIdentifier
# ---------------------------------------------------------------------------

def detector_lead_identifier(
    signal_prob: np.ndarray, grid_prob: np.ndarray, text_prob: np.ndarray,
) -> dict:
    """Roda PixelSizeFinder + SignalExtractor + LeadIdentifier."""
    H, W = signal_prob.shape
    pxsize = _get_pixel_size_finder()
    with torch.no_grad():
        grid_t = torch.from_numpy(np.ascontiguousarray(grid_prob)).float()
        mm_per_pixel_x, mm_per_pixel_y = pxsize(grid_t)
    avg_pixel_per_mm = float(
        (1.0 / float(mm_per_pixel_x) + 1.0 / float(mm_per_pixel_y)) / 2.0
    )

    signal_prob_t = torch.from_numpy(
        np.ascontiguousarray(signal_prob)
    ).float()
    raw_lines_t, x_offset = _signal_extractor_with_offset(signal_prob_t)

    identifier = _get_lead_identifier(target_num_samples=5000)
    if raw_lines_t.shape[0] > 1:
        raw_lines_t = identifier._merge_nonoverlapping_lines(
            raw_lines_t.clone()
        )
    text_prob_t = (
        torch.from_numpy(np.ascontiguousarray(text_prob))
        .unsqueeze(0).unsqueeze(0).float()
    )
    result = identifier(
        raw_lines_t.clone(), text_prob_t,
        avg_pixel_per_mm=avg_pixel_per_mm,
        threshold=0.8, mv_per_mm=0.1,
    )
    layout_name = result.get("layout") or ""
    cost = float(result.get("cost", float("nan")))

    # Parser layout_name -> formato "3x4+N" / "6x2+N" / "12x1"
    layouts = identifier.layouts
    if layout_name in layouts:
        desc = layouts[layout_name]
        cols = int(desc["layout"]["cols"])
        rows = int(desc["layout"]["rows"])
        n_rhythm_li = len(desc.get("rhythm_leads", []))
        base = f"{rows}x{cols}"
        fmt = base + (f"+{n_rhythm_li}" if n_rhythm_li else "")
    else:
        cols, rows, n_rhythm_li = 0, 0, 0
        fmt = "unknown"
    return {
        "format": fmt, "layout_name": layout_name, "cost": cost,
        "n_main": rows, "n_cols": cols, "n_rhythm": n_rhythm_li,
        "avg_pixel_per_mm": avg_pixel_per_mm,
        "raw_lines_t": raw_lines_t, "x_offset": x_offset,
    }


# ---------------------------------------------------------------------------
# Detector 3: OCR (reusa funções de test_ocr_format_detection.py)
# ---------------------------------------------------------------------------

def detector_ocr(normalized_bgr: np.ndarray) -> dict:
    H, W = normalized_bgr.shape[:2]
    ocr_results = run_ocr(normalized_bgr)
    labels = extract_lead_labels(ocr_results)
    rows, cols_per_row = group_into_rows_and_cols(labels, H, W)
    fmt, n_main, n_cols, n_rhythm = ocr_detect_format(rows, cols_per_row)
    # OCR é "inconclusivo" se achou < 8 labels (de 12)
    inconclusive = len(labels) < 8
    return {
        "format": fmt if not inconclusive else f"{fmt} (INCONCLUSIVO)",
        "n_labels": len(labels), "labels": labels,
        "rows": rows, "cols_per_row": cols_per_row,
        "n_main": n_main, "n_cols": n_cols, "n_rhythm": n_rhythm,
        "inconclusive": inconclusive,
    }


# ---------------------------------------------------------------------------
# Combinação final
# ---------------------------------------------------------------------------

def base_of(fmt: str) -> str:
    """Extrai a parte base do formato (3x4, 6x2, 12x1) ignorando '+N'
    e sufixos como '(INCONCLUSIVO)'."""
    s = fmt.split(" ")[0].split("+")[0]
    return s if s in {"3x4", "6x2", "12x1"} else "unknown"


def combinar_formato(
    bandas: dict, li: dict, ocr: dict,
) -> tuple[str, str]:
    """Decide formato final + confiança.

    Prioridade: bandas > LeadIdentifier > OCR. Confiança ALTA quando
    pelo menos bandas e LI concordam. MÉDIA se discordam — bandas decide.
    BAIXA se só um detector funcionou.
    """
    b_base = base_of(bandas["format"])
    li_base = base_of(li["format"])
    ocr_base = base_of(ocr["format"]) if not ocr["inconclusive"] else "unknown"

    final_base = b_base if b_base != "unknown" else li_base
    if final_base == "unknown":
        final_base = ocr_base
    if final_base == "unknown":
        return "unknown", "BAIXA"

    # Rhythm vem do detector que decidiu o base
    if final_base == b_base and bandas.get("n_rhythm", 0) > 0:
        suffix = f"+{bandas['n_rhythm']}"
    elif final_base == li_base and li.get("n_rhythm", 0) > 0:
        suffix = f"+{li['n_rhythm']}"
    else:
        suffix = ""
    final_fmt = final_base + suffix

    n_agree = sum(
        1 for x in (b_base, li_base, ocr_base)
        if x == final_base and x != "unknown"
    )
    if b_base != "unknown" and li_base != "unknown" and b_base == li_base:
        conf = "ALTA (bandas+LI concordam)"
    elif n_agree >= 2:
        conf = f"ALTA ({n_agree}/3 detectores concordam)"
    elif b_base != "unknown" and li_base != "unknown" and b_base != li_base:
        conf = f"MÉDIA (bandas diz {b_base}, LI diz {li_base} — bandas decide)"
    else:
        conf = "BAIXA (só 1 detector deu resposta)"
    return final_fmt, conf


# ---------------------------------------------------------------------------
# Construção dos bboxes precisos (X e Y trim pelo signal_binary)
# ---------------------------------------------------------------------------

def build_precise_bboxes(
    bandas: dict, final_base: str, signal_binary: np.ndarray,
    px_per_mm: float,
) -> list[dict]:
    """Para cada banda + col, calcula bbox preciso:
      x_start..x_end: trim do traçado real dentro do range da coluna
      y_top..y_bot:   topo/base do traçado +5% margem, clampado a midpoint
                       entre bandas vizinhas

    Layout de leads por banda:
      3x4: linhas 0-2 = main (4 cols), linhas 3+ = rhythm (1 col total)
      6x2: linhas 0-5 = main (2 cols), linhas 6+ = rhythm (1 col total)
      12x1: todas linhas = main (1 col), sem rhythm
    """
    H, W = signal_binary.shape
    bands: list[tuple[int, int]] = bandas["bands"]
    if not bands:
        return []

    if final_base == "3x4":
        layout = LAYOUT_3X4
        n_main, n_cols = 3, 4
    elif final_base == "6x2":
        layout = LAYOUT_6X2
        n_main, n_cols = 6, 2
    elif final_base == "12x1":
        layout = LAYOUT_12X1
        n_main, n_cols = 12, 1
    else:
        return []

    # Bounds X comuns às bandas main (pula pulso de calibração)
    band_bounds: list[tuple[int, int]] = []
    for (y0, y1) in bands[:n_main]:
        xs, xe = find_trace_bounds_per_band(signal_binary, y0, y1, px_per_mm)
        band_bounds.append((xs, xe))
    if band_bounds:
        x_start_common = max(b[0] for b in band_bounds)
        x_end_common = min(b[1] for b in band_bounds)
    else:
        x_start_common, x_end_common = 0, W
    chunk_w = max(1, (x_end_common - x_start_common) // n_cols)

    # Midpoints Y entre bandas (cap pra não invadir vizinho)
    n_bands = len(bands)
    band_centers = [(y0 + y1) // 2 for (y0, y1) in bands]
    cap_top = [0] * n_bands
    cap_bot = [H] * n_bands
    for i in range(n_bands):
        if i > 0:
            cap_top[i] = (band_centers[i - 1] + band_centers[i]) // 2
        if i < n_bands - 1:
            cap_bot[i] = (band_centers[i] + band_centers[i + 1]) // 2

    def _trim_xy(x_s: int, x_e: int, row_idx: int) -> tuple[int, int, int, int]:
        """Trima x e y a partir do signal_binary, +5% margem, clampado ao cap."""
        y_search_top = cap_top[row_idx]
        y_search_bot = cap_bot[row_idx]
        x_s = max(0, min(W, x_s))
        x_e = max(0, min(W, x_e))
        if x_e <= x_s or y_search_bot <= y_search_top:
            return x_s, x_e, y_search_top, y_search_bot
        region = signal_binary[y_search_top:y_search_bot, x_s:x_e]
        cols_with_signal = np.any(region, axis=0)
        rows_with_signal = np.any(region, axis=1)
        if not rows_with_signal.any() or not cols_with_signal.any():
            return x_s, x_e, y_search_top, y_search_bot
        # X trim: primeiro e último col com sinal
        col_idx = np.where(cols_with_signal)[0]
        x_trim_s = x_s + int(col_idx[0])
        x_trim_e = x_s + int(col_idx[-1]) + 1
        # Y trim: topo e base do sinal + 5% margem
        row_idx_sig = np.where(rows_with_signal)[0]
        y_sig_top = y_search_top + int(row_idx_sig[0])
        y_sig_bot = y_search_top + int(row_idx_sig[-1]) + 1
        band_h = bands[row_idx][1] - bands[row_idx][0]
        margin = max(1, int(round(band_h * QRS_MARGIN_FRAC)))
        y_top = max(y_search_top, y_sig_top - margin)
        y_bot = min(y_search_bot, y_sig_bot + margin)
        return x_trim_s, x_trim_e, int(y_top), int(y_bot)

    bboxes: list[dict] = []
    # Main bands
    for row_idx in range(min(n_main, n_bands)):
        row_leads = layout[row_idx]
        for col_idx, lead_name in enumerate(row_leads):
            x_cell_s = x_start_common + col_idx * chunk_w
            x_cell_e = (
                x_start_common + (col_idx + 1) * chunk_w
                if col_idx < n_cols - 1 else x_end_common
            )
            x_s, x_e, y_top, y_bot = _trim_xy(x_cell_s, x_cell_e, row_idx)
            bboxes.append({
                "name": lead_name, "row": row_idx, "is_rhythm": False,
                "x_start": x_s, "x_end": x_e,
                "y_top": y_top, "y_bot": y_bot,
            })
    # Rhythm bands (full width)
    for k in range(n_main, n_bands):
        y0_r, y1_r = bands[k]
        xs_r, xe_r = find_trace_bounds_per_band(
            signal_binary, y0_r, y1_r, px_per_mm,
        )
        x_s, x_e, y_top, y_bot = _trim_xy(xs_r, xe_r, k)
        r_name = "II_rhythm" if k == n_main else f"rhythm_{k - n_main}"
        bboxes.append({
            "name": r_name, "row": k, "is_rhythm": True,
            "x_start": x_s, "x_end": x_e,
            "y_top": y_top, "y_bot": y_bot,
        })
    return bboxes


# ---------------------------------------------------------------------------
# Overlay visual
# ---------------------------------------------------------------------------

def draw_overlay(
    normalized_bgr: np.ndarray,
    bandas: dict, li: dict, ocr: dict,
    final_fmt: str, final_conf: str,
    bboxes: list[dict],
) -> np.ndarray:
    img = normalized_bgr.copy()
    H, W = img.shape[:2]

    # 1. Faixas horizontais SUTIS marcando as bandas detectadas (debug)
    bands_overlay = img.copy()
    for i, (y0, y1) in enumerate(bandas.get("bands", [])):
        c = ROW_COLORS_BGR[i % len(ROW_COLORS_BGR)]
        c_light = tuple(int(255 - (255 - v) * 0.25) for v in c)
        cv2.rectangle(bands_overlay, (0, y0), (W, y1), c_light, -1)
    img = cv2.addWeighted(bands_overlay, 0.25, img, 0.75, 0)

    # 2. Header com resultado dos 3 detectores
    header_h = 220
    header = np.full((header_h, W, 3), 255, dtype=np.uint8)
    line_y = 38
    cv2.putText(
        header, "Detector            Resultado",
        (12, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2,
        cv2.LINE_AA,
    )
    line_y += 36

    b_ok = base_of(bandas["format"]) == base_of(final_fmt)
    li_ok = base_of(li["format"]) == base_of(final_fmt)
    ocr_ok = (
        not ocr["inconclusive"]
        and base_of(ocr["format"]) == base_of(final_fmt)
    )
    rows_txt = [
        (f"detect_bands:  {bandas['n_bands']} bandas -> {bandas['format']}",
         b_ok),
        (f"LeadIdentifier: {li.get('layout_name', '-')} "
         f"(cost {li.get('cost', float('nan')):.3f}) -> {li['format']}", li_ok),
        (f"OCR:           {ocr['n_labels']}/12 labels -> {ocr['format']}",
         ocr_ok),
    ]
    for txt, ok in rows_txt:
        mark_color = (0, 160, 0) if ok else (0, 0, 200)
        mark = "OK" if ok else "X"
        cv2.putText(header, txt, (12, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(header, mark, (W - 70, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, mark_color, 3,
                    cv2.LINE_AA)
        line_y += 32
    cv2.putText(
        header, f"FORMATO FINAL: {final_fmt}    Confianca: {final_conf}",
        (12, line_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (50, 50, 200),
        2, cv2.LINE_AA,
    )
    out = np.vstack([header, img])

    # 3. Bboxes precisos
    y_offset = header_h
    for b in bboxes:
        color = ROW_COLORS_BGR[b["row"] % len(ROW_COLORS_BGR)]
        if b["is_rhythm"]:
            color = (0, 200, 220)
        x_s, x_e = b["x_start"], b["x_end"]
        y_t, y_b = b["y_top"] + y_offset, b["y_bot"] + y_offset
        cv2.rectangle(out, (x_s, y_t), (x_e, y_b), color, 2)
        cv2.putText(
            out, b["name"], (x_s + 6, y_t + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Pipeline por imagem
# ---------------------------------------------------------------------------

def process_image(img_path: Path) -> None:
    print(f"\n{'=' * 60}")
    print(img_path.stem)
    print('=' * 60)
    t0 = time.perf_counter()

    # Preprocess (Dotter+Undistort) — mesmo dos outros testes
    normalized = normalize_image(img_path)
    H, W = normalized.shape[:2]
    logger.info("Normalized: %dx%d", W, H)

    # UNet -> 4 feature maps (compartilhado pelos 3 detectores que precisam)
    signal_prob, grid_prob, text_prob, _bg = get_unet_feature_maps(normalized)
    signal_binary = signal_prob > SIGNAL_THRESHOLD

    # Detector 1 — bandas
    bandas = detector_bandas(signal_binary)
    print(
        f"  detect_bands:    {bandas['n_bands']} bandas -> {bandas['format']}"
    )

    # Detector 2 — LeadIdentifier
    li = detector_lead_identifier(signal_prob, grid_prob, text_prob)
    print(
        f"  LeadIdentifier:  {li['layout_name']} "
        f"(cost {li['cost']:.3f}) -> {li['format']}"
    )

    # Detector 3 — OCR
    ocr = detector_ocr(normalized)
    print(f"  OCR:             {ocr['n_labels']}/12 labels -> {ocr['format']}")

    # Decisão final
    final_fmt, final_conf = combinar_formato(bandas, li, ocr)
    print(f"\n  FORMATO FINAL:   {final_fmt}    confiança: {final_conf}")

    # Bboxes precisos com base no formato final
    bboxes = build_precise_bboxes(
        bandas=bandas, final_base=base_of(final_fmt),
        signal_binary=signal_binary,
        px_per_mm=li.get("avg_pixel_per_mm", 13.0),
    )

    overlay = draw_overlay(
        normalized, bandas, li, ocr, final_fmt, final_conf, bboxes,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{img_path.stem}_combined.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"  -> {out_path.name}  ({time.perf_counter() - t0:.1f}s)")


def main(argv: list[str]) -> int:
    if argv:
        images = [Path(a).expanduser() for a in argv]
    else:
        images = DEFAULT_IMAGES
    print(f"Saída: {OUT_DIR}")
    for img_path in images:
        if not img_path.is_file():
            print(f"[SKIP] não encontrado: {img_path}")
            continue
        try:
            process_image(img_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERRO: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
