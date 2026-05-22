"""
Teste de detecção de formato com overlay de bboxes
==================================================

Roda Preprocess + Dotter + Undistort + Stenhede UNet + PixelSizeFinder +
SignalExtractor + LeadIdentifier em 3 imagens, e desenha bboxes coloridos
(uma por derivação detectada) sobre o ECG undistorted.

Diferença pro pipeline_completo_v1.py:
  - Bboxes vêm do LeadIdentifier (formato + nomes detectados), não do
    LEAD_LAYOUT_3x4 hardcoded.

Saída em ~/Desktop/Projeto ECG/teste_formato/<NOME>_formato.png

Uso:
    python modal_functions/pipeline/test_lead_identifier_bboxes.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))
sys.path.insert(0, str(MODAL_ROOT.parent))

from pipeline.digitize.ecg_digitizer import ECGDigitizer  # noqa: E402
from pipeline.digitize.stenhede_adapter import (  # noqa: E402
    _SECONDS_PER_CELL_BY_COLS,
    _get_lead_identifier,
    _get_pixel_size_finder,
    _signal_extractor_with_offset,
    get_unet_feature_maps,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("test_lead_identifier_bboxes")


HOME = Path.home()
OUT_DIR = HOME / "Desktop" / "Projeto ECG" / "teste_formato"
IMAGES = [
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1400.jpg",
]


# Cores BGR por linha (linha 0 = vermelho-rosa, linha 1 = verde, linha 2 =
# azul, rhythm = amarelo) — mesmo esquema do pipeline_completo_v1.plot_bboxes.
ROW_COLORS_BGR = {
    0: (80, 80, 255),    # vermelho
    1: (80, 200, 80),    # verde
    2: (255, 80, 80),    # azul
    3: (0, 200, 220),    # amarelo (rhythm)
    4: (200, 100, 200),  # extra rhythm 1
    5: (200, 200, 100),  # extra rhythm 2
}


def load_image_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is not None:
        return img
    import pillow_heif  # type: ignore
    from PIL import Image

    pillow_heif.register_heif_opener()
    pil = Image.open(str(path)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def normalize_image(img_path: Path) -> np.ndarray:
    """Replica os passos 1-5 do pipeline_completo_v1: load + auto-rotate +
    preprocess + dotter + gridder + undistort -> imagem normalizada BGR."""
    img_bgr = load_image_any(img_path)
    h, w = img_bgr.shape[:2]
    if h > w * 1.2:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    digitizer = ECGDigitizer(use_mock=False)
    cropped, _ = digitizer.preprocess(img_bgr)
    _, keypoints = digitizer.dotter(cropped)
    if len(keypoints) == 0:
        _, keypoints = digitizer.dotter_mock(cropped)
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])
    if len(keypoints) >= 100:
        normalized = digitizer.undistort(
            cropped, grid_matrix, grid_info["px_per_mm"],
        )
    else:
        normalized = cropped.copy()
    return normalized


def _patch_identifier_collect_all_scores(identifier) -> list:
    """Monkey-patch `_match_layout` pra coletar o cost de TODOS os
    layouts/flips testados (e não só o vencedor). Retorna a lista que
    será preenchida durante a próxima chamada de identifier(...).
    """
    all_scores: list[dict] = []

    def patched_match_layout(detected_pts, rows_in_layout, layouts, check_flipped):
        all_scores.clear()
        names, xs, ys = zip(*detected_pts)
        pts = np.stack([xs, ys], axis=1)
        n = pts.shape[0]
        best = {"cost": np.inf}
        for layout_name, desc in layouts.items():
            total_rows = desc["total_rows"]
            rows_difference = abs(total_rows - rows_in_layout)
            pos_map = identifier._generate_grid_positions(desc)
            if "I" in pos_map:
                del pos_map["I"]
            grid_leads = list(pos_map.keys())
            grid_pts = np.stack([pos_map[lead] for lead in grid_leads])
            flip_options = (False, True) if check_flipped else (False,)
            for flip in flip_options:
                scaling_factor = (
                    max(len(grid_leads), n) / min(len(grid_leads), n)
                    * (1 + rows_difference * 3)
                )
                P = pts.copy()
                if flip:
                    P = -P
                Pm, Gm, idxs, missing = [], [], [], 0
                for i, lead in enumerate(names):
                    if lead in pos_map:
                        j = grid_leads.index(lead)
                        Pm.append(P[i])
                        Gm.append(grid_pts[j])
                        idxs.append((i, j))
                    else:
                        missing += 1
                Pm_arr = np.array(Pm)
                Gm_arr = np.array(Gm)
                if Pm_arr.shape[0] < 2:
                    all_scores.append({
                        "layout": layout_name, "flip": flip,
                        "cost": float("inf"), "n_matched": int(Pm_arr.shape[0]),
                        "n_missing": int(missing),
                        "total_rows": total_rows, "skipped": True,
                    })
                    continue
                mu_P = Pm_arr.mean(axis=0)
                mu_G = Gm_arr.mean(axis=0)
                Pc = Pm_arr - mu_P
                Gc = Gm_arr - mu_G
                num = np.sum(Pc * Gc, axis=0)
                den = np.sum(Pc ** 2, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = num / den
                s = np.where(np.isfinite(s), s, 0.0)
                if np.any(s < 0):
                    scaling_factor *= 2
                s[s < 1e-4] = 1e-4
                t = mu_G - s * mu_P
                P_scaled = P * s + t
                res = []
                for i, j in idxs:
                    res.append(float(np.linalg.norm(P_scaled[i] - grid_pts[j])))
                PENALTY = 0.5
                res.extend([PENALTY] * missing)
                avg_res = float(np.mean(res)) * scaling_factor
                all_scores.append({
                    "layout": layout_name, "flip": flip,
                    "cost": avg_res, "n_matched": int(Pm_arr.shape[0]),
                    "n_missing": int(missing),
                    "total_rows": total_rows, "skipped": False,
                })
                if avg_res < best["cost"]:
                    best = {
                        "layout": layout_name, "flip": flip,
                        "cost": avg_res, "leads": grid_leads,
                    }
        return best

    identifier._match_layout = patched_match_layout
    return all_scores


def run_lead_identifier(normalized_bgr: np.ndarray) -> dict:
    """Roda UNet + PixelSizeFinder + SignalExtractor + LeadIdentifier e
    devolve tudo o que precisamos pra desenhar bboxes."""
    H, W = normalized_bgr.shape[:2]

    # 1. UNet -> 4 feature maps
    signal_prob, grid_prob, text_prob, _bg_prob = get_unet_feature_maps(
        normalized_bgr,
    )

    # 2. PixelSizeFinder
    pxsize = _get_pixel_size_finder()
    with torch.no_grad():
        grid_t = torch.from_numpy(np.ascontiguousarray(grid_prob)).float()
        mm_per_pixel_x, mm_per_pixel_y = pxsize(grid_t)
    avg_pixel_per_mm = float(
        (1.0 / float(mm_per_pixel_x) + 1.0 / float(mm_per_pixel_y)) / 2.0
    )

    # 3. SignalExtractor -> raw_lines + x_offset
    signal_prob_t = torch.from_numpy(
        np.ascontiguousarray(signal_prob)
    ).float()
    raw_lines_t, x_offset = _signal_extractor_with_offset(signal_prob_t)

    # 4. LeadIdentifier (com sort+merge interno IGUAL ao stenhede_adapter)
    identifier = _get_lead_identifier(target_num_samples=5000)
    if raw_lines_t.shape[0] > 1:
        raw_lines_t = identifier._merge_nonoverlapping_lines(
            raw_lines_t.clone()
        )

    # Instala patch que coleta cost de TODOS os layouts candidatos
    all_scores = _patch_identifier_collect_all_scores(identifier)

    text_prob_t = (
        torch.from_numpy(np.ascontiguousarray(text_prob))
        .unsqueeze(0).unsqueeze(0).float()
    )
    result = identifier(
        raw_lines_t.clone(),
        text_prob_t,
        avg_pixel_per_mm=avg_pixel_per_mm,
        threshold=0.8,
        mv_per_mm=0.1,
    )

    return {
        "layout": result.get("layout"),
        "cost": float(result.get("cost", float("nan"))),
        "flip": bool(result.get("flip", False)),
        "n_detected": int(result.get("n_detected", 0)),
        "raw_lines": raw_lines_t.cpu().numpy().astype(np.float64),
        "x_offset": int(x_offset),
        "avg_pixel_per_mm": avg_pixel_per_mm,
        "layouts": identifier.layouts,
        "signal_prob": signal_prob,
        "image_W": W,
        "image_H": H,
        "all_scores": list(all_scores),
    }


SIGNAL_BINARY_THRESHOLD = 0.3  # mesmo limiar do pipeline_completo_v1
QRS_MARGIN_FRAC = 0.10         # 10% da altura inicial como folga


def build_bboxes(
    raw_lines: np.ndarray,
    x_offset: int,
    image_W: int,
    image_H: int,
    avg_pixel_per_mm: float,
    layouts: dict,
    layout_name: str,
    signal_prob: np.ndarray | None = None,
) -> list[dict]:
    """A partir do layout escolhido pelo LeadIdentifier + raw_lines (já
    sort+merge), calcula um bbox por derivação:
      - x_start, x_end: derivados de x_offset + col_idx * chunk_px
      - y_center: média Y da linha correspondente (raw_lines[row_idx])
      - y_top, y_bot: y_center ± band_h/2 (band_h derivado da distância
        entre linhas vizinhas)
    """
    if layout_name not in layouts:
        logger.warning("Layout '%s' não está no YAML — abortando bbox", layout_name)
        return []

    layout_def = layouts[layout_name]
    cols = int(layout_def["layout"]["cols"])
    leads_def = layout_def["leads"]
    rhythm_leads = list(layout_def.get("rhythm_leads", []))

    # Reshape leads em rows_iter (matriz row × col)
    is_matrix = isinstance(leads_def[0], list) if leads_def else False
    if is_matrix:
        rows_iter = leads_def
    else:
        layout_rows = int(layout_def["layout"].get("rows", 1))
        layout_cols_check = int(layout_def["layout"].get("cols", len(leads_def)))
        if layout_rows * layout_cols_check == len(leads_def) and layout_rows > 1:
            rows_iter = [
                leads_def[i * layout_cols_check:(i + 1) * layout_cols_check]
                for i in range(layout_rows)
            ]
        else:
            rows_iter = [leads_def]

    # chunk_px CALIBRADO (igual _canonicalize_calibrated)
    spc = _SECONDS_PER_CELL_BY_COLS.get(cols, 10.0 / max(cols, 1))
    chunk_mm = spc * 25.0  # paper speed BR
    chunk_px = int(round(chunk_mm * float(avg_pixel_per_mm)))
    if chunk_px <= 0:
        chunk_px = (image_W - x_offset) // max(cols, 1)

    n_lines = raw_lines.shape[0]
    n_main_rows = len(rows_iter)

    # y_center de cada linha (média ignorando NaN)
    y_centers = []
    for i in range(n_lines):
        line = raw_lines[i]
        valid = ~np.isnan(line)
        y_centers.append(float(np.nanmean(line)) if valid.any() else float("nan"))

    # band_h: distância entre y_centers consecutivos (usa o min se houver
    # múltiplos pra ficar conservador). Fallback: image_H / max(n_lines, 4).
    if n_lines >= 2:
        diffs = []
        for i in range(n_lines - 1):
            yc_a, yc_b = y_centers[i], y_centers[i + 1]
            if not (np.isnan(yc_a) or np.isnan(yc_b)):
                diffs.append(abs(yc_b - yc_a))
        band_h = float(min(diffs)) if diffs else image_H / max(n_lines, 4)
    else:
        band_h = image_H / 4.0

    bboxes: list[dict] = []
    full_w = cols * chunk_px

    # Midpoints entre y_centers consecutivos — cap pra não invadir vizinho.
    # midpoint_above[i] = limite superior de expansão (não pode passar)
    # midpoint_below[i] = limite inferior de expansão (não pode passar)
    midpoint_above = [0] * n_lines
    midpoint_below = [image_H] * n_lines
    for i in range(n_lines):
        if i > 0 and not np.isnan(y_centers[i - 1]) and not np.isnan(y_centers[i]):
            midpoint_above[i] = int((y_centers[i - 1] + y_centers[i]) / 2)
        if i < n_lines - 1 and not np.isnan(y_centers[i + 1]) and not np.isnan(y_centers[i]):
            midpoint_below[i] = int((y_centers[i] + y_centers[i + 1]) / 2)

    # Máscara binária do traçado (se signal_prob disponível) — usada pra
    # expandir y_top/y_bot até cobrir os extremos reais do QRS dentro do
    # range de colunas.
    sig_bin: np.ndarray | None = None
    if signal_prob is not None and signal_prob.shape == (image_H, image_W):
        sig_bin = signal_prob > SIGNAL_BINARY_THRESHOLD

    def _expand_bbox(
        row_idx: int, x_s: int, x_e: int, y_top0: int, y_bot0: int,
    ) -> tuple[int, int]:
        """Expande [y_top0, y_bot0] até cobrir todos os pixels de sinal
        em [y_top0..y_bot0, x_s..x_e], +10% margem, clampando ao
        midpoint entre bandas vizinhas (pra não invadir)."""
        cap_top = midpoint_above[row_idx]
        cap_bot = midpoint_below[row_idx]
        if sig_bin is None or x_e <= x_s:
            return (
                int(max(cap_top, y_top0)),
                int(min(cap_bot, y_bot0)),
            )
        # Procura traço dentro do range MAXIMO permitido (entre midpoints),
        # não só dentro do bbox inicial — senão picos já fora do bbox seriam
        # ignorados.
        y_search_top = max(0, cap_top)
        y_search_bot = min(image_H, cap_bot)
        if y_search_bot <= y_search_top:
            return (
                int(max(cap_top, y_top0)),
                int(min(cap_bot, y_bot0)),
            )
        region = sig_bin[y_search_top:y_search_bot, x_s:x_e]
        rows_with_signal = np.any(region, axis=1)
        if not rows_with_signal.any():
            return (
                int(max(cap_top, y_top0)),
                int(min(cap_bot, y_bot0)),
            )
        idx_with = np.where(rows_with_signal)[0]
        top_in_region = int(idx_with[0])
        bot_in_region = int(idx_with[-1])
        sig_top = y_search_top + top_in_region
        sig_bot = y_search_top + bot_in_region
        # Margem de 10% da altura inicial do bbox
        margin = int(max(1, round((y_bot0 - y_top0) * QRS_MARGIN_FRAC)))
        y_top_expanded = min(y_top0, sig_top - margin)
        y_bot_expanded = max(y_bot0, sig_bot + margin)
        # Clampa aos midpoints e à imagem
        y_top_final = int(max(cap_top, max(0, y_top_expanded)))
        y_bot_final = int(min(cap_bot, min(image_H, y_bot_expanded)))
        return y_top_final, y_bot_final

    for row_idx, row_leads in enumerate(rows_iter):
        if row_idx >= n_lines:
            break
        yc = y_centers[row_idx]
        if np.isnan(yc):
            continue
        y_top0 = int(max(0, yc - band_h / 2))
        y_bot0 = int(min(image_H, yc + band_h / 2))
        if not isinstance(row_leads, list):
            row_leads = [row_leads]
        for col_idx, lead_name_raw in enumerate(row_leads):
            lead_name = str(lead_name_raw).lstrip("-")
            x_start = x_offset + col_idx * chunk_px
            x_end = x_offset + (col_idx + 1) * chunk_px
            x_start = int(max(0, min(image_W, x_start)))
            x_end = int(max(0, min(image_W, x_end)))
            y_top, y_bot = _expand_bbox(row_idx, x_start, x_end, y_top0, y_bot0)
            bboxes.append({
                "name": lead_name,
                "row": row_idx,
                "is_rhythm": False,
                "x_start": x_start,
                "x_end": x_end,
                "y_top": y_top,
                "y_bot": y_bot,
                "y_center": int(yc),
            })

    # Rhythm strips: linhas após n_main_rows ocupam toda a largura útil
    for k in range(len(rhythm_leads)):
        line_idx = n_main_rows + k
        if line_idx >= n_lines:
            break
        yc = y_centers[line_idx]
        if np.isnan(yc):
            continue
        y_top0 = int(max(0, yc - band_h / 2))
        y_bot0 = int(min(image_H, yc + band_h / 2))
        if k == 0:
            r_name = "II_rhythm"
        else:
            r_name = f"rhythm_{k}"
        x_start = x_offset
        x_end = x_offset + full_w
        x_start = int(max(0, min(image_W, x_start)))
        x_end = int(max(0, min(image_W, x_end)))
        y_top, y_bot = _expand_bbox(line_idx, x_start, x_end, y_top0, y_bot0)
        bboxes.append({
            "name": r_name,
            "row": line_idx,
            "is_rhythm": True,
            "x_start": x_start,
            "x_end": x_end,
            "y_top": y_top,
            "y_bot": y_bot,
            "y_center": int(yc),
        })

    return bboxes


def draw_overlay(
    normalized_bgr: np.ndarray, bboxes: list[dict], layout_name: str,
    cost: float, n_detected: int,
) -> np.ndarray:
    img = normalized_bgr.copy()
    # Faixa branca translúcida no topo pra escrever metadata
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (255, 255, 255), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(
        img,
        f"Layout: {layout_name} | Cost: {cost:.3f} | Detected: {n_detected}",
        (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA,
    )

    for b in bboxes:
        color = ROW_COLORS_BGR.get(b["row"], (50, 50, 50))
        if b["is_rhythm"]:
            color = ROW_COLORS_BGR[3]
        cv2.rectangle(
            img,
            (b["x_start"], b["y_top"]),
            (b["x_end"], b["y_bot"]),
            color, 2,
        )
        label = b["name"]
        cv2.putText(
            img, label,
            (b["x_start"] + 6, b["y_top"] + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )
    return img


def summarize_layout(layouts: dict, layout_name: str, bboxes: list[dict]) -> str:
    if layout_name not in layouts:
        return "(layout desconhecido)"
    desc = layouts[layout_name]
    cols = int(desc["layout"]["cols"])
    rows = int(desc["layout"]["rows"])
    n_rhythm = len(desc.get("rhythm_leads", []))
    leads_per_row: dict[int, list[str]] = {}
    for b in bboxes:
        leads_per_row.setdefault(b["row"], []).append(b["name"])
    parts = [", ".join(leads_per_row[k]) for k in sorted(leads_per_row.keys())]
    layout_str = f"{rows}x{cols}" + (f"+{n_rhythm}" if n_rhythm else "")
    return (
        f"{layout_str} ({cols} colunas, {rows} linhas de leads, "
        f"{n_rhythm} rhythm)\nLeads: " + " | ".join(parts)
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaída: {OUT_DIR}\n")

    for img_path in IMAGES:
        if not img_path.is_file():
            print(f"[SKIP] não encontrado: {img_path}")
            continue
        stem = img_path.stem
        print(f"=== {stem} ===")
        t0 = time.perf_counter()
        try:
            normalized = normalize_image(img_path)
            res = run_lead_identifier(normalized)
            bboxes = build_bboxes(
                raw_lines=res["raw_lines"],
                x_offset=res["x_offset"],
                image_W=res["image_W"],
                image_H=res["image_H"],
                avg_pixel_per_mm=res["avg_pixel_per_mm"],
                layouts=res["layouts"],
                layout_name=str(res["layout"]),
                signal_prob=res["signal_prob"],
            )
            overlay = draw_overlay(
                normalized, bboxes,
                layout_name=str(res["layout"]),
                cost=res["cost"],
                n_detected=res["n_detected"],
            )
            out_path = OUT_DIR / f"{stem}_formato.png"
            cv2.imwrite(str(out_path), overlay)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERRO: {type(e).__name__}: {e}\n")
            continue
        dt = time.perf_counter() - t0
        layout_str = summarize_layout(
            res["layouts"], str(res["layout"]), bboxes,
        )
        print(
            f"  Formato: {res['layout']}\n"
            f"  Cost:    {res['cost']:.3f}\n"
            f"  Linhas:  {res['raw_lines'].shape[0]}\n"
            f"  Flip:    {res['flip']}\n"
            f"  Layout:  {layout_str}"
        )
        scores = res.get("all_scores", [])
        if scores:
            ranked = sorted(
                scores, key=lambda s: (float("inf") if s["skipped"] else s["cost"])
            )
            print(f"  Ranking de TODOS os layouts (n={len(ranked)}):")
            print(
                f"    {'#':>2} {'layout':<28} {'flip':>5} "
                f"{'cost':>10} {'matched':>7} {'missing':>7} "
                f"{'total_rows':>10}"
            )
            for i, s in enumerate(ranked, 1):
                cost_str = "  SKIPPED" if s["skipped"] else f"{s['cost']:10.4f}"
                print(
                    f"    {i:>2} {s['layout']:<28} {str(s['flip']):>5} "
                    f"{cost_str} {s['n_matched']:>7} {s['n_missing']:>7} "
                    f"{s['total_rows']:>10}"
                )
        print(f"  -> {out_path.name}  ({dt:.1f}s)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
