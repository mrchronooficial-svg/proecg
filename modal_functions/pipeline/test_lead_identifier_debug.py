"""
Diagnóstico do LeadIdentifier do Stenhede:
  • Replica internamente a chamada à lead_name_unet em IMG_1387 e IMG_1303
  • Imprime (x, y) detectado para cada um dos 12 canais
  • Imprime as fronteiras de chunk usadas pelo _canonicalize_lines (W//cols)
  • Imprime o x_offset do SignalExtractor (trim das margens vazias)
  • Compara: posição IDEAL do texto de cada lead vs onde a lead_name_unet
    detectou — desvios em px e em fração da chunk_width
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.stenhede_adapter import (
    LEAD_CHANNEL_ORDER,
    _ensure_vendor_on_path,
    _get_lead_identifier,
    _signal_extractor_with_offset,
    _VENDOR_LAYOUTS_YAML,
    get_unet_feature_maps,
)

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
TARGETS = ["IMG_1387", "IMG_1303"]


def _load_layouts() -> dict:
    with open(_VENDOR_LAYOUTS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _process(stem: str) -> int:
    img_path = UNDIST_DIR / f"{stem}.png"
    if not img_path.exists():
        print(f"[ERRO] {img_path} nao existe")
        return 1
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] falha ao ler {img_path}")
        return 2
    h, w = img.shape[:2]

    print(f"\n{'=' * 78}")
    print(f"  {stem}.png  shape=(H={h}, W={w})")
    print(f"{'=' * 78}")

    # 1. U-Net principal → feature maps
    sig_p, grid_p, text_p, _bg = get_unet_feature_maps(img)
    print(f"\nFeature maps shape = {sig_p.shape} (em coords da imagem original)")

    # 2. SignalExtractor — só pra obter x_offset e n_lines
    sig_t = torch.from_numpy(np.ascontiguousarray(sig_p)).float()
    raw_lines, x_offset = _signal_extractor_with_offset(sig_t)
    n_lines = int(raw_lines.shape[0])
    lines_w = int(raw_lines.shape[1])
    print(
        f"SignalExtractor: {n_lines} linhas, x_offset={x_offset}, "
        f"trimmed_width={lines_w} (cobre cols [{x_offset}, {x_offset + lines_w - 1}])"
    )

    # 3. Roda lead_name_unet manualmente no text_prob
    identifier = _get_lead_identifier(target_num_samples=w)
    text_t = (
        torch.from_numpy(np.ascontiguousarray(text_p))
        .unsqueeze(0).unsqueeze(0).float()
    )
    with torch.no_grad():
        logits = identifier.unet(text_t.to(identifier.device))  # (1,13,H,W)
        probs = torch.softmax(logits, dim=1)[:, :12]
        probs[:, 0] = 0  # I é ignorado pelo Stenhede
    threshold = 0.8
    probs_thr = probs.clone()
    probs_thr[probs_thr < threshold] = 0

    # 4. Centroide por canal — x_center, y_center, peso total
    arr = probs_thr[0].cpu().numpy()
    detected_at_thr = []
    detected_no_thr = []  # sem threshold pra ver tudo
    for ch_idx, lead_name in enumerate(LEAD_CHANNEL_ORDER):
        ch_thr = arr[ch_idx]
        ch_raw = probs[0, ch_idx].cpu().numpy()
        x_grid = np.arange(ch_thr.shape[1]).reshape(1, -1)
        y_grid = np.arange(ch_thr.shape[0]).reshape(-1, 1)

        if ch_thr.sum() > 0:
            xc = float((x_grid * ch_thr).sum() / ch_thr.sum())
            yc = float((y_grid * ch_thr).sum() / ch_thr.sum())
            mass_thr = float(ch_thr.sum())
        else:
            xc, yc, mass_thr = float("nan"), float("nan"), 0.0
        detected_at_thr.append((lead_name, xc, yc, mass_thr))

        if ch_raw.sum() > 0:
            xc2 = float((x_grid * ch_raw).sum() / ch_raw.sum())
            yc2 = float((y_grid * ch_raw).sum() / ch_raw.sum())
            mass_raw = float(ch_raw.sum())
        else:
            xc2, yc2, mass_raw = float("nan"), float("nan"), 0.0
        detected_no_thr.append((lead_name, xc2, yc2, mass_raw))

    print(f"\nlead_name_unet (threshold={threshold}, "
          f"canal 'I' é ignorado pelo Stenhede):")
    print(f"  {'Lead':<6}{'x_thr':>9}{'y_thr':>8}{'mass_thr':>11}"
          f"{'x_raw':>9}{'y_raw':>8}{'mass_raw':>11}")
    for (n_thr, xt, yt, mt), (_, xr, yr, mr) in zip(detected_at_thr, detected_no_thr):
        xt_s = f"{xt:.0f}" if not np.isnan(xt) else "  -"
        yt_s = f"{yt:.0f}" if not np.isnan(yt) else "  -"
        xr_s = f"{xr:.0f}" if not np.isnan(xr) else "  -"
        yr_s = f"{yr:.0f}" if not np.isnan(yr) else "  -"
        print(f"  {n_thr:<6}{xt_s:>9}{yt_s:>8}{mt:>11.2f}"
              f"{xr_s:>9}{yr_s:>8}{mr:>11.2f}")

    # 5. Match layout: usa apenas pontos com prob > threshold
    detected_pts = [
        (n, xc, yc) for n, xc, yc, m in detected_at_thr
        if not np.isnan(xc) and m > 0
    ]
    print(f"\nLeads detectados acima do threshold: {len(detected_pts)}")
    if len(detected_pts) <= 2:
        print("  (insuficientes pra match — fallback default layout)")
        return 0

    rows_in_layout = n_lines  # n linhas extraídas
    layouts = _load_layouts()
    match = identifier._match_layout(
        detected_pts, rows_in_layout, layouts, identifier.possibly_flipped,
    )
    layout_name = match.get("layout")
    cost = match.get("cost", float("inf"))
    print(f"Layout escolhido: {layout_name} (cost={cost:.3f})")

    if layout_name not in layouts:
        return 0

    layout_def = layouts[layout_name]
    cols = int(layout_def["layout"]["cols"])
    leads_def = layout_def["leads"]

    # 6. Fronteiras de chunk usadas pelo _canonicalize_lines
    # canonical width = target_num_samples = W_image
    canonical_w = w
    chunk_w = canonical_w // cols
    print(f"\nDivisão em chunks (rígida em W//cols):")
    print(f"  W (canonical) = {canonical_w}")
    print(f"  cols          = {cols}")
    print(f"  chunk_width   = {chunk_w} px ({chunk_w / 13.0:.1f} mm a 13px/mm)")
    print(f"  Fronteiras:")
    for c in range(cols):
        x_start = c * chunk_w
        x_end = (c + 1) * chunk_w if c < cols - 1 else canonical_w
        print(f"    col {c}: [{x_start:>5d}, {x_end:>5d}]   "
              f"largura={x_end - x_start}")

    # 7. POSIÇÃO IDEAL do texto de cada lead = centro do chunk dele
    # Pra layout 3x4 leads_def é lista de listas [["I","aVR","V1","V4"], ...]
    print(f"\nPosição IDEAL do texto (= centro do chunk) vs DETECTADA:")
    print(f"  {'Lead':<6}{'col':>4}{'x_ideal':>10}{'x_detect':>10}"
          f"{'Δ_px':>9}{'Δ_chunk':>9}")
    is_matrix = isinstance(leads_def[0], list)
    for row_idx, row_leads in enumerate(leads_def if is_matrix else [leads_def]):
        if not isinstance(row_leads, list):
            row_leads = [row_leads]
        for c_idx, lead_name_raw in enumerate(row_leads):
            lead_name = str(lead_name_raw).lstrip("-")
            x_center_ideal = (c_idx + 0.5) * chunk_w
            # Procura na detected_at_thr
            xc = float("nan")
            for n, x, y, m in detected_at_thr:
                if n == lead_name and m > 0:
                    xc = x
                    break
            if not np.isnan(xc):
                delta = xc - x_center_ideal
                delta_chunk = delta / chunk_w
                xc_s = f"{xc:.0f}"
                d_s = f"{delta:+.0f}"
                dc_s = f"{delta_chunk:+.2f}"
            else:
                xc_s = "  -"
                d_s = "  -"
                dc_s = "  -"
            print(f"  {lead_name:<6}{c_idx:>4}"
                  f"{x_center_ideal:>10.0f}{xc_s:>10}{d_s:>9}{dc_s:>9}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(name)s] %(message)s")
    print("=" * 78)
    print(" Diagnóstico LeadIdentifier — fronteiras de chunk vs textos detectados")
    print("=" * 78)
    rc = 0
    for stem in TARGETS:
        rc = _process(stem) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
