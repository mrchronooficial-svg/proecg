"""
Comparativo: potencial sinal (signal_prob da U-Net) vs filtro default
do SignalExtractor — para IMG_1279, IMG_1303 e IMG_1387.

Em cada PNG:
  • Imagem undistorted clareada (fundo)
  • signal_prob como heatmap LARANJA-VERMELHO (semi-transparente):
    onde a U-Net vê traço com prob alta
  • Linha extraída pelo SignalExtractor (defaults Stenhede) em VERDE

Onde aparecer LARANJA SEM verde por cima = região que a U-Net detectou
como traçado mas o SignalExtractor descartou.

Saída em modal_functions/pipeline/digitize/_visualizations/:
  IMG_1279_potential_vs_extracted.png
  IMG_1303_potential_vs_extracted.png
  IMG_1387_potential_vs_extracted.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.stenhede_adapter import (
    _DEFAULT_MAX_DIM,
    _SIGNAL_CLASS,
    _signal_extractor_with_offset,
    get_unet,
    _pad_lines_to_image_width,
)

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
TARGETS = ["IMG_1279", "IMG_1303", "IMG_1387"]


def _signal_prob(image_bgr: np.ndarray) -> np.ndarray:
    h_orig, w_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (image_rgb - image_rgb.min()) / max(image_rgb.max() - image_rgb.min(), 1e-8)
    max_side = max(h_orig, w_orig)
    if max_side > _DEFAULT_MAX_DIM:
        scale = _DEFAULT_MAX_DIM / float(max_side)
        new_h = int(round(h_orig * scale))
        new_w = int(round(w_orig * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        sp = probs[0, _SIGNAL_CLASS]
        sp = sp - sp.mean()
        sp = torch.clamp(sp, min=0)
        sp = sp / (sp.max() + 1e-9)
    sp_np = sp.cpu().numpy().astype(np.float32)
    if sp_np.shape != (h_orig, w_orig):
        sp_np = cv2.resize(sp_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return sp_np


def _process(stem: str) -> None:
    img_path = UNDIST_DIR / f"{stem}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] {img_path}")
        return
    h, w = img.shape[:2]
    print(f"\n--- {stem} ---  shape=({h}, {w})")

    # 1. Potencial sinal (U-Net)
    sp = _signal_prob(img)
    px_pot_001 = int((sp > 0.01).sum())
    px_pot_005 = int((sp > 0.05).sum())
    px_pot_01 = int((sp > 0.10).sum())
    print(
        f"  signal_prob: max={sp.max():.3f} mean={sp.mean():.4f} "
        f">0.01: {px_pot_001} px,  >0.05: {px_pot_005} px,  >0.10: {px_pot_01} px"
    )

    # 2. Linha extraída pelo SignalExtractor com defaults
    sp_t = torch.from_numpy(np.ascontiguousarray(sp)).float()
    raw_lines, x_offset = _signal_extractor_with_offset(sp_t)
    raw_padded = _pad_lines_to_image_width(
        raw_lines.cpu().numpy().astype(np.float64),
        x_offset=x_offset, w_orig=w,
    )
    n_lines = raw_padded.shape[0]
    valid = (~np.isnan(raw_padded)).sum()
    print(f"  SignalExtractor (defaults): {n_lines} linhas, "
          f"x_offset={x_offset}, samples válidos={valid}")

    # 3. Render: imagem clareada + heatmap laranja + linha verde
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = 0.35 * img_rgb + 0.65   # clarear bastante
    img_blend = np.clip(img_blend, 0, 1)

    # Heatmap somente onde sp > 0.02 (ignora ruído de fundo)
    sp_norm = sp.copy()
    if sp_norm.max() > 0:
        sp_norm = sp_norm / sp_norm.max()
    sp_thr = 0.02
    sp_mask = (sp_norm > sp_thr).astype(np.float32)
    heat_rgba = cm.get_cmap("YlOrRd")(sp_norm)
    heat_rgb = heat_rgba[..., :3].astype(np.float32)
    # Blend somente onde tem sinal
    alpha = 0.55
    final = (
        img_blend * (1 - alpha * sp_mask[..., None])
        + heat_rgb * (alpha * sp_mask[..., None])
    )
    final = np.clip(final, 0, 1)

    # Plot final + linhas extraídas em verde brilhante
    fig_w = max(20.0, w / 200.0); fig_h = fig_w * (h / w) + 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(final, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])

    if n_lines > 0:
        x_axis = np.arange(w)
        for i in range(n_lines):
            ax.plot(
                x_axis, raw_padded[i],
                color="#00ff66", lw=1.7, alpha=0.95, zorder=5,
            )

    legend = [
        mpatches.Patch(color="#fde68a", label="Potencial sinal U-Net (claro)"),
        mpatches.Patch(color="#dc2626", label="Potencial sinal U-Net (forte)"),
        plt.Line2D([0], [0], color="#00ff66", lw=2,
                   label="Filtro default (SignalExtractor)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=11, framealpha=0.95)
    ax.set_title(
        f"{stem} — Potencial sinal (U-Net, vermelho/amarelo) "
        f"vs filtro default (linha verde)",
        fontsize=12, fontweight="bold",
    )
    out_path = OUT_DIR / f"{stem}_potential_vs_extracted.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Render] {out_path.name}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(" Potencial sinal (U-Net) vs filtro default (SignalExtractor)")
    print("=" * 78)
    for stem in TARGETS:
        _process(stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
