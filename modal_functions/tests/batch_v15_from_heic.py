"""
Pipeline HEIC -> ECG digital v15 (formato treino, canvas fixo 900x650).

Especificacao:
  - Canvas: 900x650 px (fixo, todos os layouts)
  - px_per_mm: 3.0 (fixo) -> 1 quadradinho (1mm) = 3x3 px | 5mm = 15x15 px
  - Grid 232/245 cobrindo todo canvas
  - Area do ECG dentro do canvas (centralizado, padding branco ao redor):
      3x4+1  : 750 x 240 px
      6x2+1  : 750 x 525 px
      12x1   : 750 x 540 px
  - Render: alpha boost 1.5x + skeletonize 1 px + halo cinza 200

Layout detectado por contagem de bandas (row-projection peaks):
  <=4 bandas -> 3x4+1
  <=8 bandas -> 6x2+1
   else      -> 12x1

Cache: signal_prob+px_per_mm pickle por ECG em _pipeline_cache/ (evita rerodar UNet).
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.signal import find_peaks
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
logger = logging.getLogger("batch_v15")

pillow_heif.register_heif_opener()

DOWNLOADS = Path(r"C:\Users\rafae\Downloads")
OUTPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ecgs_digitais_v15_batch")
CACHE_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\_pipeline_cache_v15")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ECG_FILES = [
    "IMG_1583", "IMG_1582", "IMG_1581", "IMG_1579", "IMG_1578", "IMG_1556",
    "IMG_1575", "IMG_1532", "IMG_1531", "IMG_1510", "IMG_1577", "IMG_1572",
    "IMG_1565", "IMG_1562", "IMG_1560", "IMG_1559", "IMG_1558", "IMG_1534",
    "IMG_1511", "IMG_1503", "IMG_1491", "IMG_1490", "IMG_1478", "IMG_1462",
    "IMG_1461", "IMG_1455", "IMG_1454", "IMG_1453",
]

# === Canvas / grid (fixo, todos os layouts) ===
CANVAS_W = 900
CANVAS_H = 650
PX_PER_MM = 3.0

# === Area do ECG por layout (LxA, centralizada) ===
LAYOUT_DIMS = {
    "3x4+1": (750, 240),
    "6x2+1": (750, 525),
    "12x1": (750, 540),
}

# === Render params (heranca v14) ===
TRACE_THRESHOLD = 0.05
BBOX_THRESHOLD = 0.5
ALPHA_BOOST = 1.5
TRACE_DARK_THRESHOLD = 200
DILATE_THRESHOLD = 128
HALO_GRAY = 200


def heic_to_bgr(heic_path: Path) -> np.ndarray:
    img = Image.open(str(heic_path)).convert("RGB")
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def compute_signal_prob(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Pipeline pre-SignalExtractor. Retorna (signal_prob, avg_pixel_per_mm)."""
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


def detect_layout(signal_prob: np.ndarray) -> tuple[str, int, int]:
    """Detecta layout via row-projection. Retorna (nome, target_w, target_h)."""
    binary = signal_prob > BBOX_THRESHOLD
    row_count = binary.sum(axis=1).astype(float)
    if row_count.sum() == 0:
        return ("3x4+1", *LAYOUT_DIMS["3x4+1"])
    smoothed = gaussian_filter1d(row_count, sigma=10)
    peaks, _ = find_peaks(smoothed, distance=50, prominence=smoothed.max() * 0.1)
    n_bands = len(peaks)
    if n_bands <= 4:
        name = "3x4+1"
    elif n_bands <= 8:
        name = "6x2+1"
    else:
        name = "12x1"
    tw, th = LAYOUT_DIMS[name]
    return (name, tw, th)


def render_v15(signal_prob: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Renderiza ECG digital v15 (canvas 900x650 fixo). Retorna (canvas, layout)."""
    H, W = signal_prob.shape
    layout_name, target_w, target_h = detect_layout(signal_prob)

    core = signal_prob > BBOX_THRESHOLD
    rows_with = np.any(core, axis=1)
    cols_with = np.any(core, axis=0)
    if not rows_with.any() or not cols_with.any():
        return None, layout_name
    y_min = int(np.argmax(rows_with))
    y_max = H - int(np.argmax(rows_with[::-1]))
    x_min = int(np.argmax(cols_with))
    x_max = W - int(np.argmax(cols_with[::-1]))

    # Render alpha gradient na bbox em res original
    sp_crop = signal_prob[y_min:y_max, x_min:x_max]
    bbox_h_px, bbox_w_px = sp_crop.shape
    trace_full = np.full((bbox_h_px, bbox_w_px), 255.0, dtype=np.float32)
    trace_mask = sp_crop > TRACE_THRESHOLD
    trace_intensity = np.clip(
        sp_crop / max(signal_prob.max(), 1e-6) * ALPHA_BOOST, 0, 1,
    )
    alpha_t = trace_intensity * trace_mask.astype(np.float32)
    trace_full = trace_full * (1.0 - alpha_t)
    trace_full = trace_full.astype(np.uint8)

    # Resize NAO-uniforme pra target_w x target_h (dimensions ja embutem aspect do layout)
    trace_resized = cv2.resize(
        trace_full, (target_w, target_h), interpolation=cv2.INTER_AREA,
    )

    # Skeletonize + halo
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

    # Canvas 900x650 com grid em todo o canvas
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    offset_x = (CANVAS_W - target_w) // 2
    offset_y = (CANVAS_H - target_h) // 2

    p_min = PX_PER_MM
    p_maj = 5 * PX_PER_MM
    # Grid alinhado em (offset_x, offset_y) -> cantos do ECG ficam em interseccao do grid
    y = offset_y % p_min
    while y < CANVAS_H:
        yi = int(round(y))
        if 0 <= yi < CANVAS_H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 245)
        y += p_min
    x = offset_x % p_min
    while x < CANVAS_W:
        xi = int(round(x))
        if 0 <= xi < CANVAS_W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 245)
        x += p_min
    y = offset_y % p_maj
    while y < CANVAS_H:
        yi = int(round(y))
        if 0 <= yi < CANVAS_H:
            canvas[yi, :] = np.minimum(canvas[yi, :], 232)
        y += p_maj
    x = offset_x % p_maj
    while x < CANVAS_W:
        xi = int(round(x))
        if 0 <= xi < CANVAS_W:
            canvas[:, xi] = np.minimum(canvas[:, xi], 232)
        x += p_maj

    canvas[offset_y:offset_y + target_h, offset_x:offset_x + target_w] = np.minimum(
        canvas[offset_y:offset_y + target_h, offset_x:offset_x + target_w],
        trace_layer,
    )
    return canvas, layout_name


def load_or_compute_signal_prob(stem: str, heic_path: Path) -> tuple[np.ndarray, float]:
    """Carrega cache se existir, senao computa e salva."""
    cache_path = CACHE_DIR / f"{stem}.pkl"
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                d = pickle.load(f)
            return d["signal_prob"], d["avg_pixel_per_mm"]
        except Exception as e:
            logger.warning("Cache invalido (%s) — recomputando", e)
    img_bgr = heic_to_bgr(heic_path)
    signal_prob, px_per_mm = compute_signal_prob(img_bgr)
    try:
        with cache_path.open("wb") as f:
            pickle.dump({"signal_prob": signal_prob, "avg_pixel_per_mm": px_per_mm}, f)
    except Exception as e:
        logger.warning("Falha ao salvar cache: %s", e)
    return signal_prob, px_per_mm


def process_one(stem: str) -> tuple[bool, str]:
    heic_path = DOWNLOADS / f"{stem}.HEIC"
    if not heic_path.is_file():
        return False, f"HEIC nao encontrado: {heic_path}"
    try:
        t0 = time.perf_counter()
        signal_prob, _ = load_or_compute_signal_prob(stem, heic_path)
        canvas, layout = render_v15(signal_prob)
        if canvas is None:
            return False, f"Sem tracado detectado (layout={layout})"
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), canvas)
        dt = time.perf_counter() - t0
        return True, f"layout={layout} em {dt:.1f}s"
    except Exception:
        return False, traceback.format_exc()


def main() -> int:
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Cache:  %s", CACHE_DIR)
    logger.info("Canvas: %dx%d @ %.1f px/mm | Total: %d ECGs",
                CANVAS_W, CANVAS_H, PX_PER_MM, len(ECG_FILES))
    n_ok = 0
    fails: list[tuple[str, str]] = []
    t_start = time.perf_counter()
    for i, stem in enumerate(ECG_FILES, 1):
        logger.info("[%d/%d] %s", i, len(ECG_FILES), stem)
        ok, msg = process_one(stem)
        if ok:
            n_ok += 1
            logger.info("  OK -> %s", msg)
        else:
            fails.append((stem, msg))
            logger.error("  FALHOU: %s", msg.splitlines()[0])
    dt = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("Concluido em %.1fs: %d OK / %d falhas", dt, n_ok, len(fails))
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
