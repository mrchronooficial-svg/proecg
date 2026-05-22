"""
Roda UNet Stenhede direto no ecg_digital.png (reconstrucao limpa) e
extrai sinal usando o metodo validado (skeleton + per-lead overlay).

Pula crop/Dotter/Gridder/Undistort (a imagem ja eh limpa, sem vincos).
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

from pipeline.digitize.stenhede_adapter import (
    _process_sparse_prob_torch, get_unet,
)
from tests.test_signal_overlay_per_lead import (
    _get_component_masks, _xmerge_fragments, _enforce_in_mask,
    _fill_small_gaps_in_signal,
    extract_skeleton, extract_thinning,
    extract_borda_superior, extract_media_bordas,
    _save_per_lead_overlay_vibrant,
    MIN_LINE_COVERAGE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("digital_extract")

INPUT_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407\ecg_digital.png"
)
OUTPUT_DIR = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407\digital_extraction"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIBRANT_COLORS = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9A6324", "#800000",
]


def main() -> int:
    logger.info("Carregando %s", INPUT_PATH)
    img_bgr = cv2.imread(str(INPUT_PATH))
    if img_bgr is None:
        logger.error("Falha ao carregar")
        return 1
    h_orig, w_orig = img_bgr.shape[:2]
    logger.info("Shape: %d x %d", h_orig, w_orig)

    # Roda UNet
    logger.info("Rodando UNet Stenhede...")
    t0 = time.perf_counter()
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    max_side = max(h_orig, w_orig)
    if max_side > 3000:
        scale = 3000 / float(max_side)
        new_h, new_w = int(round(h_orig * scale)), int(round(w_orig * scale))
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_float = image_rgb.astype(np.float32)
    img_float = (img_float - img_float.min()) / max(img_float.max() - img_float.min(), 1e-8)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        signal_p = _process_sparse_prob_torch(probs[:, 2])

    sig_prob_np = signal_p.squeeze(0).cpu().numpy().astype(np.float32)
    if sig_prob_np.shape != (h_orig, w_orig):
        sig_prob_np = cv2.resize(sig_prob_np, (w_orig, h_orig),
                                  interpolation=cv2.INTER_LINEAR)
    logger.info("UNet rodada em %.1fs", time.perf_counter() - t0)
    logger.info("signal_prob range: [%.3f, %.3f]",
                float(sig_prob_np.min()), float(sig_prob_np.max()))

    # Encontra components + skeleton extraction
    component_masks, binary = _get_component_masks(sig_prob_np)
    logger.info("Components validos: %d", len(component_masks))

    methods = [
        ("skeleton",       extract_skeleton),
        ("thinning",       extract_thinning),
        ("borda_superior", extract_borda_superior),
        ("media_bordas",   extract_media_bordas),
    ]

    for name, fn in methods:
        t0 = time.perf_counter()
        lines = []
        total_fora = 0
        for cm in component_masks:
            sig = fn(cm.astype(np.uint8))
            sig, fora = _enforce_in_mask(sig, binary)
            total_fora += fora
            if int((~np.isnan(sig)).sum()) >= 30:
                lines.append(sig)
        lines = _xmerge_fragments(lines, max_y_distance_px=30, max_x_gap_px=600)
        lines = [s for s in lines if int((~np.isnan(s)).sum()) >= MIN_LINE_COVERAGE]
        lines = [_fill_small_gaps_in_signal(s, max_gap_size=200, max_y_diff=80) for s in lines]
        lines.sort(key=lambda s: float(np.nanmean(s)) if not np.all(np.isnan(s)) else 0)
        total_cov = sum(int((~np.isnan(s)).sum()) for s in lines)
        dt = time.perf_counter() - t0
        logger.info(
            "%s: %d lines, cov=%d cols, fora=%d, %.1fs",
            name, len(lines), total_cov, total_fora, dt,
        )
        _save_per_lead_overlay_vibrant(
            lines, img_bgr, VIBRANT_COLORS,
            f"DIGITAL ECG — {name}",
            OUTPUT_DIR / f"signal_{name}.png",
        )

    logger.info("Concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
