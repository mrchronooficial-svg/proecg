"""
Pipeline completo pra IMG_1283 (ECGs Reais3): foto -> Stenhede -> Viterbi por derivacao.

Etapas:
  1. preprocess + dotter + gridder + undistort -> normalized
  2. Stenhede UNet -> 4 canais
  3. canal 2 -> mascara B&W
  4. detect bandas -> Viterbi por derivacao
  5. salva todos os outputs em IMG_1283_pipeline_completo/
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

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
from pipeline.digitize.viterbi_extractor import extrair_sinal_viterbi  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline_completo")

IMG_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3\IMG_1283.jpg")
OUTPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1283_pipeline_completo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_THRESHOLD = 0.3


def detect_bands(binary_mask: np.ndarray, sigma=10.0, distance=50,
                 prominence_factor=0.1, buffer_factor=1.2):
    h = binary_mask.shape[0]
    row_count = binary_mask.sum(axis=1).astype(float)
    if row_count.sum() == 0:
        return []
    smoothed = gaussian_filter1d(row_count, sigma=sigma)
    peaks, _ = find_peaks(smoothed, distance=distance,
                          prominence=smoothed.max() * prominence_factor)
    if peaks.size == 0:
        return []
    inner_valleys = []
    for i in range(len(peaks) - 1):
        segment = smoothed[peaks[i]:peaks[i + 1]]
        inner_valleys.append(int(peaks[i] + np.argmin(segment)))
    bands = []
    for i, peak in enumerate(peaks):
        if i == 0:
            y0 = max(0, peak - int((inner_valleys[0] - peak) * buffer_factor)) if inner_valleys else 0
        else:
            y0 = inner_valleys[i - 1]
        if i == len(peaks) - 1:
            y1 = min(h, peak + int((peak - inner_valleys[-1]) * buffer_factor)) if inner_valleys else h
        else:
            y1 = inner_valleys[i]
        bands.append((y0, y1))
    return bands


def main() -> int:
    t0 = time.perf_counter()
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETO — %s", IMG_PATH.name)
    logger.info("=" * 70)

    # ----- ETAPA 1: load -----
    img_bgr = cv2.imread(str(IMG_PATH))
    if img_bgr is None:
        logger.error("Falha ao ler %s", IMG_PATH)
        return 1
    h, w = img_bgr.shape[:2]
    if h > w * 1.2:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    logger.info("Original: %dx%d", img_bgr.shape[1], img_bgr.shape[0])

    # ----- ETAPA 2-5: preprocess + dotter + gridder + undistort -----
    logger.info("--- Preprocess + Dotter + Gridder + Undistort ---")
    digitizer = ECGDigitizer(use_mock=False)
    cropped, _ = digitizer.preprocess(img_bgr)
    _, keypoints = digitizer.dotter(cropped)
    if len(keypoints) == 0:
        _, keypoints = digitizer.dotter_mock(cropped)
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])
    if len(keypoints) >= 100:
        normalized = digitizer.undistort(cropped, grid_matrix, grid_info["px_per_mm"])
        undistort_applied = True
    else:
        normalized = cropped.copy()
        undistort_applied = False
    h_n, w_n = normalized.shape[:2]
    logger.info("Normalized: %dx%d | undistort: %s", w_n, h_n, undistort_applied)
    cv2.imwrite(str(OUTPUT_DIR / "00_normalized.png"), normalized)

    # ----- ETAPA 8: Stenhede UNet -----
    logger.info("--- Stenhede UNet inference ---")
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

    # 4 canais individuais (heatmap)
    channels = [
        ("01_canal_0_grid", grid_prob, "Blues", "Canal 0: GRID"),
        ("02_canal_1_text", text_prob, "Oranges", "Canal 1: TEXTO/FUNDO"),
        ("03_canal_2_signal_heatmap", signal_prob, "Reds", "Canal 2: TRACADO"),
        ("04_canal_3_bg", bg_prob, "Greys", "Canal 3: FUNDO"),
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

    # Mascara B&W do canal 2
    signal_binary = (signal_prob > SIGNAL_THRESHOLD).astype(np.uint8) * 255
    mask_bw = 255 - signal_binary
    cv2.imwrite(str(OUTPUT_DIR / "05_canal_2_signal_PB.png"), mask_bw)
    n_signal = int((signal_prob > SIGNAL_THRESHOLD).sum())
    pct = 100.0 * n_signal / signal_prob.size
    logger.info("Mascara B&W canal 2: %d px do tracado (%.2f%%)", n_signal, pct)

    # 4 canais combinado
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
    fig.text(0.5, 0.01, f"Tracado: {n_signal} px ({pct:.1f}%)",
             ha="center", fontsize=10, color="#444")
    fig.savefig(str(OUTPUT_DIR / "06_unet_4canais_combinado.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)

    # ----- ETAPA 10: PixelSizeFinder -----
    logger.info("--- PixelSizeFinder ---")
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
    ax1.set_title("Imagem + grid_prob overlay", fontsize=11)
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    proj_x = grid_prob.sum(axis=0)
    proj_y = grid_prob.sum(axis=1)
    ax2.plot(proj_x / max(proj_x.max(), 1e-6), label="X", color="#2266cc", linewidth=0.8)
    ax2.plot(proj_y / max(proj_y.max(), 1e-6), label="Y", color="#cc6622", linewidth=0.8)
    ax2.set_title("Projecoes do grid_prob", fontsize=10)
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.text(0.5, 0.01,
             f"px/mm: {avg_pixel_per_mm:.3f} | "
             f"mm/px X: {float(mm_per_pixel_x):.4f} | mm/px Y: {float(mm_per_pixel_y):.4f}",
             ha="center", fontsize=10, color="#444")
    fig.savefig(str(OUTPUT_DIR / "07_pixel_size.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)

    # ----- ETAPA: detect bandas + Viterbi por derivacao -----
    logger.info("--- Deteccao de bandas + Viterbi por derivacao ---")
    binary_mask = mask_bw < 128  # True = tracado
    bands = detect_bands(binary_mask)
    n_bands = len(bands)
    logger.info("Bandas detectadas: %d", n_bands)
    for i, (y0, y1) in enumerate(bands):
        logger.info("  banda %d: y[%d:%d] altura=%dpx", i + 1, y0, y1, y1 - y0)

    # Visualizacao bandas
    H, W = mask_bw.shape
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    ax.imshow(mask_bw, cmap="gray", aspect="auto")
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_bands, 1)))
    for i, ((y0, y1), c) in enumerate(zip(bands, colors)):
        ax.axhspan(y0, y1, color=c, alpha=0.15)
        ax.text(20, (y0 + y1) // 2, f"banda {i+1}",
                color=c * 0.7, fontweight="bold", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_title(f"{n_bands} bandas detectadas", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "08_bandas_detectadas.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)

    # Viterbi por banda
    signals = []
    stats = []
    for i, (y0, y1) in enumerate(bands):
        band_mask = mask_bw[y0:y1, :]
        t_b = time.perf_counter()
        sig = extrair_sinal_viterbi(band_mask, invert=True)
        dt_b = time.perf_counter() - t_b
        cols_with = int(np.any(band_mask < 128, axis=0).sum())
        n_gaps = W - cols_with
        n_valid = int((~np.isnan(sig)).sum())
        signals.append(sig)
        stats.append({
            "band": i + 1, "y0": y0, "y1": y1,
            "time_s": dt_b, "n_valid": n_valid, "n_gaps_raw": n_gaps,
            "min": float(np.nanmin(sig)), "max": float(np.nanmax(sig)),
        })
        logger.info("  banda %d: %.2fs | gaps raw=%d | min=%.1f max=%.1f",
                    i + 1, dt_b, n_gaps, stats[-1]["min"], stats[-1]["max"])

    # Painel
    fig, axes = plt.subplots(n_bands, 1, figsize=(16, 1.6 * n_bands), dpi=100,
                             sharex=True)
    if n_bands == 1:
        axes = [axes]
    fig.suptitle(f"Viterbi DP por derivacao — {IMG_PATH.stem}",
                 fontsize=14, fontweight="bold", y=0.995)
    for i, (sig, st, ax) in enumerate(zip(signals, stats, axes)):
        ax.plot(sig, color="#1f78b4", linewidth=0.7)
        ax.axhline(0, color="#aaa", linewidth=0.4, linestyle="--")
        ax.set_ylabel(f"banda {st['band']}\n(y={st['y0']}-{st['y1']})",
                      fontsize=8, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, W)
        title = (f"banda {st['band']}: {st['n_valid']}/{W} validos | "
                 f"gaps raw {st['n_gaps_raw']} | "
                 f"range [{st['min']:.0f}, {st['max']:.0f}] | {st['time_s']:.2f}s")
        ax.set_title(title, fontsize=9, loc="left")
    axes[-1].set_xlabel("coluna (x)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(str(OUTPUT_DIR / "09_viterbi_por_derivacao.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)

    total_dt = time.perf_counter() - t0
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETO em %.1fs", total_dt)
    logger.info("Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
