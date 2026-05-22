"""
Roda pipeline ate Stenhede pra IMG_1281 (JPG em ECGs Reais3) e salva
outputs do Stenhede.
"""

from __future__ import annotations

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
sys.path.insert(0, str(MODAL_ROOT.parent))

from pipeline.digitize.ecg_digitizer import ECGDigitizer  # noqa: E402
from pipeline.digitize.stenhede_adapter import (  # noqa: E402
    _get_pixel_size_finder,
    _process_sparse_prob_torch,
    get_unet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("stenhede_steps")

IMG_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3\IMG_1281.jpg")
OUTPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1281_stenhede_steps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_THRESHOLD = 0.3


def main() -> int:
    t0 = time.perf_counter()
    logger.info("Carregando %s", IMG_PATH.name)
    img_bgr = cv2.imread(str(IMG_PATH))
    if img_bgr is None:
        logger.error("Falha ao ler imagem: %s", IMG_PATH)
        return 1
    h, w = img_bgr.shape[:2]
    if h > w * 1.2:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    logger.info("Preprocess + Dotter + Gridder + Undistort...")
    digitizer = ECGDigitizer(use_mock=False)
    cropped, _ = digitizer.preprocess(img_bgr)
    _, keypoints = digitizer.dotter(cropped)
    if len(keypoints) == 0:
        _, keypoints = digitizer.dotter_mock(cropped)
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])
    if len(keypoints) >= 100:
        normalized = digitizer.undistort(cropped, grid_matrix, grid_info["px_per_mm"])
    else:
        normalized = cropped.copy()

    h_n, w_n = normalized.shape[:2]
    logger.info("Normalized: %dx%d (input do Stenhede)", w_n, h_n)
    cv2.imwrite(str(OUTPUT_DIR / "00_normalized.png"), normalized)

    logger.info("Stenhede UNet inference...")
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

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        grid_p = _process_sparse_prob_torch(probs[:, 0])
        text_p = _process_sparse_prob_torch(probs[:, 1])
        signal_p = _process_sparse_prob_torch(probs[:, 2])
        bg_p = _process_sparse_prob_torch(probs[:, 3])

    def _to_np(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze(0).cpu().numpy().astype(np.float32)
        if arr.shape != (h_n, w_n):
            arr = cv2.resize(arr, (w_n, h_n), interpolation=cv2.INTER_LINEAR)
        return arr

    grid_prob = _to_np(grid_p)
    text_prob = _to_np(text_p)
    signal_prob = _to_np(signal_p)
    bg_prob = _to_np(bg_p)

    channels = [
        ("01_canal_0_grid", grid_prob, "Blues", "Canal 0: GRID (probabilidade)"),
        ("02_canal_1_text", text_prob, "Oranges", "Canal 1: TEXTO/FUNDO"),
        ("03_canal_2_signal_heatmap", signal_prob, "Reds", "Canal 2: TRACADO ECG (probabilidade)"),
        ("04_canal_3_bg", bg_prob, "Greys", "Canal 3: FUNDO BRANCO"),
    ]
    for name, prob, cmap, title in channels:
        fig = plt.figure(figsize=(14, 9), dpi=110)
        plt.imshow(prob, cmap=cmap, vmin=0, vmax=1)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.axis("off")
        plt.colorbar(fraction=0.04, pad=0.02)
        plt.savefig(str(OUTPUT_DIR / f"{name}.png"),
                    bbox_inches="tight", dpi=110, facecolor="white")
        plt.close(fig)
        logger.info("Salvo: %s.png", name)

    signal_binary = (signal_prob > SIGNAL_THRESHOLD).astype(np.uint8) * 255
    mask_bw = 255 - signal_binary
    cv2.imwrite(str(OUTPUT_DIR / "05_canal_2_signal_PB.png"), mask_bw)
    n_signal = int(np.sum(signal_prob > SIGNAL_THRESHOLD))
    pct = 100.0 * n_signal / signal_prob.size
    logger.info("Mascara B&W canal 2: %d pixels do tracado (%.2f%%) | threshold=%.2f",
                n_signal, pct, SIGNAL_THRESHOLD)

    fig = plt.figure(figsize=(15, 11), dpi=110)
    fig.suptitle("STENHEDE — UNet 4 canais", fontsize=14, fontweight="bold", y=0.98)
    titles = ["Canal 0: GRID", "Canal 1: TEXTO/FUNDO",
              "Canal 2: TRACADO ECG", "Canal 3: FUNDO BRANCO"]
    probs_list = [grid_prob, text_prob, signal_prob, bg_prob]
    cmaps = ["Blues", "Oranges", "Reds", "Greys"]
    for i, (p, t, cm) in enumerate(zip(probs_list, titles, cmaps)):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(p, cmap=cm, vmin=0, vmax=1)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.text(0.5, 0.01,
             f"Tracado (>{SIGNAL_THRESHOLD}): {n_signal} px ({pct:.1f}%)",
             ha="center", fontsize=10, color="#444")
    fig.savefig(str(OUTPUT_DIR / "06_unet_4canais_combinado.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)

    logger.info("PixelSizeFinder...")
    pxsize = _get_pixel_size_finder()
    with torch.no_grad():
        grid_t = torch.from_numpy(np.ascontiguousarray(grid_prob)).float()
        mm_per_pixel_x, mm_per_pixel_y = pxsize(grid_t)
    avg_pixel_per_mm = float(
        (1.0 / float(mm_per_pixel_x) + 1.0 / float(mm_per_pixel_y)) / 2.0
    )
    logger.info("px/mm: x=%.3f y=%.3f avg=%.3f",
                1.0 / float(mm_per_pixel_x), 1.0 / float(mm_per_pixel_y), avg_pixel_per_mm)

    fig = plt.figure(figsize=(16, 9), dpi=110)
    fig.suptitle("STENHEDE — PixelSizeFinder", fontsize=14, fontweight="bold", y=0.98)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    ax1.imshow(grid_prob, cmap="cool", alpha=0.45)
    ax1.set_title("Imagem + grid_prob (overlay)", fontsize=11)
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    proj_x = grid_prob.sum(axis=0)
    proj_y = grid_prob.sum(axis=1)
    ax2.plot(proj_x / max(proj_x.max(), 1e-6), label="projecao X",
             color="#2266cc", linewidth=0.8)
    ax2.plot(proj_y / max(proj_y.max(), 1e-6), label="projecao Y",
             color="#cc6622", linewidth=0.8)
    ax2.set_title("Projecoes do grid_prob (auto-correlacao -> periodicidade)", fontsize=10)
    ax2.set_xlabel("pixel")
    ax2.set_ylabel("intensidade normalizada")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.text(0.5, 0.01,
             f"px/mm: {avg_pixel_per_mm:.3f} | "
             f"mm/px X: {float(mm_per_pixel_x):.4f} | mm/px Y: {float(mm_per_pixel_y):.4f}",
             ha="center", fontsize=10, color="#444")
    fig.savefig(str(OUTPUT_DIR / "07_pixel_size.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)

    dt = time.perf_counter() - t0
    logger.info("Concluido em %.1fs. Outputs em %s", dt, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
