"""
Pipeline completo HEIC -> ECG digital v14 (formato treino).

Le HEIC de Downloads, roda pre-SignalExtractor (Stenhede UNet + PixelSizeFinder),
renderiza no formato v14 aprovado:
  - 3 px/mm fixo
  - Canvas adaptativo (>=900x450)
  - Grid 232/245
  - Alpha boost 1.5x
  - Skeletonize (1 px) + halo cinza 200 (engorso perceptivo ~20%)

Salva apenas a imagem final em pasta nova dentro de Projeto ECG.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

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
logger = logging.getLogger("batch_v14")

# Habilita PIL ler HEIC
pillow_heif.register_heif_opener()

DOWNLOADS = Path(r"C:\Users\rafae\Downloads")
OUTPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ecgs_digitais_v14_batch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ECG_FILES = [
    "IMG_1583", "IMG_1582", "IMG_1581", "IMG_1579", "IMG_1578", "IMG_1556",
    "IMG_1575", "IMG_1532", "IMG_1531", "IMG_1510", "IMG_1577", "IMG_1572",
    "IMG_1565", "IMG_1562", "IMG_1560", "IMG_1559", "IMG_1558", "IMG_1534",
    "IMG_1511", "IMG_1503", "IMG_1491", "IMG_1490", "IMG_1478", "IMG_1462",
    "IMG_1461", "IMG_1455", "IMG_1454", "IMG_1453",
]

# Params v14 (aprovados)
MIN_CANVAS_W = 900
MIN_CANVAS_H = 450
MARGIN = 40
TRACE_THRESHOLD = 0.05
BBOX_THRESHOLD = 0.5
ALPHA_BOOST = 1.5
FIXED_PX_PER_MM = 3.0
TRACE_DARK_THRESHOLD = 200
DILATE_THRESHOLD = 128
HALO_GRAY = 200


def heic_to_bgr(heic_path: Path) -> np.ndarray:
    """Converte HEIC pra BGR ndarray usando pillow_heif."""
    img = Image.open(str(heic_path)).convert("RGB")
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def compute_signal_prob(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Roda pipeline pre-SignalExtractor. Retorna (signal_prob, avg_pixel_per_mm)."""
    h, w = img_bgr.shape[:2]
    if h > w * 1.2:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    digitizer = ECGDigitizer(use_mock=False)
    cropped, _ = digitizer.preprocess(img_bgr)
    dotter_mask, keypoints = digitizer.dotter(cropped)
    if len(keypoints) == 0:
        dotter_mask, keypoints = digitizer.dotter_mock(cropped)
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])
    if len(keypoints) >= 100:
        normalized = digitizer.undistort(cropped, grid_matrix, grid_info["px_per_mm"])
    else:
        normalized = cropped.copy()

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

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        signal_p = _process_sparse_prob_torch(probs[:, 2])
        grid_p = _process_sparse_prob_torch(probs[:, 0])

    def _to_np(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze(0).cpu().numpy().astype(np.float32)
        if arr.shape != (h_n, w_n):
            arr = cv2.resize(arr, (w_n, h_n), interpolation=cv2.INTER_LINEAR)
        return arr

    signal_prob = _to_np(signal_p)
    grid_prob = _to_np(grid_p)

    pxsize = _get_pixel_size_finder()
    with torch.no_grad():
        grid_t = torch.from_numpy(np.ascontiguousarray(grid_prob)).float()
        mm_per_pixel_x, mm_per_pixel_y = pxsize(grid_t)
    avg_pixel_per_mm = float(
        (1.0 / float(mm_per_pixel_x) + 1.0 / float(mm_per_pixel_y)) / 2.0
    )
    return signal_prob, avg_pixel_per_mm


def render_v14(signal_prob: np.ndarray, px_per_mm_orig: float) -> np.ndarray | None:
    """Renderiza ECG digital v14. Retorna None se nao houver tracado."""
    H, W = signal_prob.shape
    core = signal_prob > BBOX_THRESHOLD
    rows_with = np.any(core, axis=1)
    cols_with = np.any(core, axis=0)
    if not rows_with.any() or not cols_with.any():
        return None
    y_min = int(np.argmax(rows_with))
    y_max = H - int(np.argmax(rows_with[::-1]))
    x_min = int(np.argmax(cols_with))
    x_max = W - int(np.argmax(cols_with[::-1]))
    bbox_w_px = x_max - x_min
    bbox_h_px = y_max - y_min
    width_mm = bbox_w_px / px_per_mm_orig
    height_mm = bbox_h_px / px_per_mm_orig
    new_w = int(round(width_mm * FIXED_PX_PER_MM))
    new_h = int(round(height_mm * FIXED_PX_PER_MM))

    sp_crop = signal_prob[y_min:y_max, x_min:x_max]
    trace_full = np.full((bbox_h_px, bbox_w_px), 255.0, dtype=np.float32)
    trace_mask = sp_crop > TRACE_THRESHOLD
    trace_intensity = np.clip(
        sp_crop / max(signal_prob.max(), 1e-6) * ALPHA_BOOST, 0, 1,
    )
    alpha_t = trace_intensity * trace_mask.astype(np.float32)
    trace_full = trace_full * (1.0 - alpha_t)
    trace_full = trace_full.astype(np.uint8)
    trace_resized = cv2.resize(trace_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

    trace_binary = trace_resized < TRACE_DARK_THRESHOLD
    trace_skeleton = skeletonize(trace_binary)
    trace_skel_img = np.full_like(trace_resized, 255)
    trace_skel_img[trace_skeleton] = 0

    skel_mask = trace_skel_img < DILATE_THRESHOLD
    dilated_full = binary_dilation(skel_mask, iterations=1)
    halo_only = dilated_full & ~skel_mask

    trace_layer = np.full_like(trace_resized, 255)
    trace_layer[halo_only] = HALO_GRAY
    trace_layer[skel_mask] = 0

    canvas_w = max(MIN_CANVAS_W, new_w + 2 * MARGIN)
    canvas_h = max(MIN_CANVAS_H, new_h + 2 * MARGIN)
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    offset_x = (canvas_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2

    p_min = FIXED_PX_PER_MM
    p_maj = 5 * FIXED_PX_PER_MM
    y = offset_y % p_min
    while y < canvas_h:
        yi = int(round(y))
        if 0 <= yi < canvas_h:
            canvas[yi, :] = np.minimum(canvas[yi, :], 245)
        y += p_min
    x = offset_x % p_min
    while x < canvas_w:
        xi = int(round(x))
        if 0 <= xi < canvas_w:
            canvas[:, xi] = np.minimum(canvas[:, xi], 245)
        x += p_min
    y = offset_y % p_maj
    while y < canvas_h:
        yi = int(round(y))
        if 0 <= yi < canvas_h:
            canvas[yi, :] = np.minimum(canvas[yi, :], 232)
        y += p_maj
    x = offset_x % p_maj
    while x < canvas_w:
        xi = int(round(x))
        if 0 <= xi < canvas_w:
            canvas[:, xi] = np.minimum(canvas[:, xi], 232)
        x += p_maj

    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.minimum(
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w],
        trace_layer,
    )
    return canvas


def process_one(stem: str) -> tuple[bool, str]:
    heic_path = DOWNLOADS / f"{stem}.HEIC"
    if not heic_path.is_file():
        return False, f"HEIC nao encontrado: {heic_path}"
    try:
        t0 = time.perf_counter()
        img_bgr = heic_to_bgr(heic_path)
        signal_prob, px_per_mm = compute_signal_prob(img_bgr)
        canvas = render_v14(signal_prob, px_per_mm)
        if canvas is None:
            return False, "Sem tracado detectado (signal_prob vazio)"
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), canvas)
        dt = time.perf_counter() - t0
        h, w = canvas.shape
        return True, f"{w}x{h} em {dt:.1f}s"
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{type(e).__name__}: {e}\n{tb}"


def main() -> int:
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Total: %d ECGs", len(ECG_FILES))
    n_ok = 0
    n_fail = 0
    fails: list[tuple[str, str]] = []
    t_start = time.perf_counter()
    for i, stem in enumerate(ECG_FILES, 1):
        logger.info("[%d/%d] %s", i, len(ECG_FILES), stem)
        ok, msg = process_one(stem)
        if ok:
            n_ok += 1
            logger.info("  OK -> %s", msg)
        else:
            n_fail += 1
            fails.append((stem, msg))
            logger.error("  FALHOU: %s", msg.splitlines()[0])
    total_dt = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("Concluido em %.1fs: %d OK / %d falhas", total_dt, n_ok, n_fail)
    if fails:
        logger.info("FALHAS:")
        for stem, msg in fails:
            logger.info("  %s: %s", stem, msg.splitlines()[0])
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
