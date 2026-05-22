"""Pipeline parcial nas 14 imagens de ECGs Reais3.

Por ECG, gera só 3 visualizações:
  • _02_undistorted.png       (imagem após Dotter + undistortion)
  • _03a_stenhede_plain.png   (overlay Stenhede em vermelho)
  • _03b_stenhede_colored.png (overlay colorido por derivação)

Skip Dotter visualization, CNN, ECG digital render.

Uso:
    python -m modal_functions.pipeline.test_undistort_stenhede_only
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.constants import GAIN_DEFAULT, PAPER_SPEED_DEFAULT
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.stenhede_adapter import extract_signals_stenhede
from .test_pipeline_10ecgs import (
    _coverage_from_raw_lines,
    _detect_format,
    _save_stenhede_overlay,
    _save_stenhede_overlay_plain,
    _save_undistorted_viz,
)

INPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3")
OUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_reais3")
TARGETS = [
    "IMG_1449", "IMG_1473", "IMG_1474", "IMG_1491", "IMG_1460",
    "IMG_1407", "IMG_1472", "IMG_1405", "IMG_1475", "IMG_1412",
    "IMG_1383", "IMG_1471", "IMG_1459", "IMG_1386",
]


def _process_one(stem: str, digitizer: ECGDigitizer) -> dict:
    img_path = INPUT_DIR / f"{stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return {"name": stem, "error": "imagem não encontrada"}

    # 1. Preprocess (crop + perspective)
    cropped, _ = digitizer.preprocess(img)

    # 2. Dotter (grid keypoints) — necessário pra undistortion
    if digitizer.use_mock:
        _, keypoints = digitizer.dotter_mock(cropped)
    else:
        _, keypoints = digitizer.dotter(cropped)
        if len(keypoints) == 0:
            _, keypoints = digitizer.dotter_mock(cropped)

    # 3. Gridder
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])

    # 4. Undistortion
    if len(keypoints) >= 100:
        normalized = digitizer.undistort(
            cropped, grid_matrix, grid_info["px_per_mm"],
        )
    else:
        normalized = cropped.copy()

    # Salva undistorted
    _save_undistorted_viz(
        normalized, OUT_DIR / f"{stem}_02_undistorted.png",
    )

    # 5. Stenhede full
    stenhede = extract_signals_stenhede(
        image_bgr=normalized,
        px_per_mm=float(grid_info["px_per_mm"]),
        paper_speed=float(PAPER_SPEED_DEFAULT),
        voltage_gain=float(GAIN_DEFAULT),
    )
    raw_lines = np.asarray(stenhede["raw_lines_pixel"], dtype=np.float64)
    layout_name = (stenhede.get("match") or {}).get("layout") or "standard_3x4_with_r1"
    canonical = stenhede.get("canonical_lines_uv")
    chunk_px = int(canonical.shape[1]) if canonical is not None and canonical.size else 0
    x_offset = int(stenhede.get("raw_lines_x_offset", 0))
    cov = _coverage_from_raw_lines(raw_lines)
    ecg_fmt = _detect_format(layout_name)

    _save_stenhede_overlay_plain(
        normalized, raw_lines, cov,
        OUT_DIR / f"{stem}_03a_stenhede_plain.png",
        ecg_format=ecg_fmt,
    )
    _save_stenhede_overlay(
        normalized, raw_lines, cov,
        OUT_DIR / f"{stem}_03b_stenhede_colored.png",
        ecg_format=ecg_fmt,
        chunk_px=chunk_px,
        x_offset=x_offset,
    )
    return {
        "name": stem, "kp": len(keypoints), "layout": layout_name,
        "fmt": ecg_fmt, "coverage": cov, "shape": normalized.shape,
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] {len(TARGETS)} ECGs em {INPUT_DIR}")
    print(f"[*] Saída: {OUT_DIR}\n")
    digitizer = ECGDigitizer()
    summary = []
    t_global = time.time()
    for idx, stem in enumerate(TARGETS, 1):
        print(f"[{idx}/{len(TARGETS)}] {stem}")
        t0 = time.time()
        try:
            r = _process_one(stem, digitizer)
            dt = time.time() - t0
            if "error" in r:
                print(f"   ERRO: {r['error']}")
            else:
                print(f"   kp={r['kp']}  layout={r['layout']}  fmt={r['fmt']}  "
                      f"cov={r['coverage']:.1f}%  {dt:.1f}s")
                summary.append(r)
        except Exception as e:
            print(f"   ERRO: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    total = time.time() - t_global
    print(f"\n[*] Total: {total/60:.1f}min  ({len(summary)}/{len(TARGETS)} OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
