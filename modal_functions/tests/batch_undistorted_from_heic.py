"""
Salva apenas as imagens UNDISTORTED (alta resolucao) das HEIC em Downloads.

Pipeline ate o passo 5 (preprocess + dotter + gridder + undistort).
Pula UNet Stenhede + SignalExtractor + render -> mais rapido (~3min/ECG).

Output: PNG na resolucao natural da undistorcao (similar a original, ~3-4K).
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))
sys.path.insert(0, str(MODAL_ROOT.parent))

from pipeline.digitize.ecg_digitizer import ECGDigitizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("batch_undistorted")

pillow_heif.register_heif_opener()

DOWNLOADS = Path(r"C:\Users\rafae\Downloads")
OUTPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\undistorted_batch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ECG_FILES = [
    "IMG_1583", "IMG_1582", "IMG_1581", "IMG_1579", "IMG_1578", "IMG_1556",
    "IMG_1575", "IMG_1532", "IMG_1531", "IMG_1510", "IMG_1577", "IMG_1572",
    "IMG_1565", "IMG_1562", "IMG_1560", "IMG_1559", "IMG_1558", "IMG_1534",
    "IMG_1511", "IMG_1503", "IMG_1491", "IMG_1490", "IMG_1478", "IMG_1462",
    "IMG_1461", "IMG_1455", "IMG_1454", "IMG_1453",
]


def heic_to_bgr(heic_path: Path) -> np.ndarray:
    img = Image.open(str(heic_path)).convert("RGB")
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process_one(stem: str) -> tuple[bool, str]:
    heic_path = DOWNLOADS / f"{stem}.HEIC"
    if not heic_path.is_file():
        return False, f"HEIC nao encontrado: {heic_path}"
    try:
        t0 = time.perf_counter()
        img_bgr = heic_to_bgr(heic_path)
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
            applied = True
        else:
            normalized = cropped.copy()
            applied = False

        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), normalized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        dt = time.perf_counter() - t0
        h, w = normalized.shape[:2]
        status = "undistorted" if applied else "cropped (sem undistort, keypts<100)"
        return True, f"{w}x{h} [{status}] em {dt:.1f}s"
    except Exception:
        return False, traceback.format_exc()


def main() -> int:
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Total: %d ECGs", len(ECG_FILES))
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
