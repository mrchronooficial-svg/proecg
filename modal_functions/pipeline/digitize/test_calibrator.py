"""
Teste do Calibrador
===================

Roda o pipeline (preprocess -> dotter -> gridder -> undistort) em 10 fotos
reais (pasta ECGs Reais3, seed=42) e imprime o output completo do calibrador
pra cada uma.

Uso:
    cd proecg
    python -m modal_functions.pipeline.digitize.test_calibrator
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from .calibrator import calibrate
from .ecg_digitizer import ECGDigitizer
from .format_detector import detect_layout

ECGS_REAIS3_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3")
N_IMAGES = 10
RANDOM_SEED = 42


def _print_dict(d: dict, indent: int = 2) -> None:
    """Pretty-print de dict (com floats formatados em 2 casas)."""

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, dict):
            return {k: _fmt(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_fmt(x) for x in v]
        return v

    print(json.dumps({k: _fmt(v) for k, v in d.items()}, indent=indent, ensure_ascii=False))


def run_calibration_for_image(image_path: str, use_mock: bool = True) -> dict:
    """Roda preprocess -> dotter -> gridder -> undistort -> calibrate.

    Returns:
        dict do calibrador, ou {"error": "..."} se algo falhar.
    """
    print(f"\n{'=' * 70}")
    print(f"IMAGEM: {image_path}")
    print(f"{'=' * 70}")

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"não foi possível abrir: {image_path}"}

    print(f"  shape original: {img.shape}")

    digitizer = ECGDigitizer(use_mock=use_mock)

    # 1. Preprocess
    cropped, crop_info = digitizer.preprocess(img)
    print(f"  preprocess: {crop_info.get('method')} -> shape {cropped.shape}")

    # 2. Dotter (mock = HSV/morfologia)
    if use_mock:
        grid_mask, keypoints = digitizer.dotter_mock(cropped)
    else:
        grid_mask, keypoints = digitizer.dotter(cropped)
        if len(keypoints) == 0:
            grid_mask, keypoints = digitizer.dotter_mock(cropped)
    print(f"  dotter: {len(keypoints)} keypoints")

    if len(keypoints) < 4:
        return {"error": f"keypoints insuficientes ({len(keypoints)})"}

    # 3. Gridder
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])
    print(
        f"  gridder: shape={grid_info['shape']}, "
        f"px_per_mm(grid)={grid_info['px_per_mm']:.3f}, "
        f"interpolated={grid_info['interpolated']}"
    )

    if grid_info["shape"][0] < 2 or grid_info["shape"][1] < 2:
        return {"error": "gridder não produziu matriz válida"}

    # 4. Undistort (skip se grid muito esparso)
    if len(keypoints) >= 100:
        normalized = digitizer.undistort(cropped, grid_matrix, grid_info["px_per_mm"])
        print(f"  undistort: aplicado -> shape {normalized.shape}")
    else:
        normalized = cropped.copy()
        print(f"  undistort: pulado (grid esparso, {len(keypoints)} kp)")

    # 4.5. Layout (pra n_lead_bands)
    layout = detect_layout(normalized, grid_info["px_per_mm"])
    n_bands = layout.get("n_rows", 3)
    print(f"  layout: {layout.get('format')} ({n_bands} linhas de derivações)")

    # 5. Calibrate
    print("\n  -> CALIBRADOR:")
    try:
        cal = calibrate(
            grid_matrix=grid_matrix,
            normalized_image=normalized,
            n_lead_bands=n_bands,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    _print_dict(cal, indent=4)
    return cal


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    if not ECGS_REAIS3_DIR.exists():
        print(f"ERRO: pasta não encontrada: {ECGS_REAIS3_DIR}", file=sys.stderr)
        return 1

    all_imgs = sorted(
        p for p in ECGS_REAIS3_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not all_imgs:
        print(f"ERRO: nenhuma imagem em {ECGS_REAIS3_DIR}", file=sys.stderr)
        return 1

    # Mesma seleção do test_quality_scorer (mesma seed → mesmas 10 imagens)
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(all_imgs, k=min(N_IMAGES, len(all_imgs)))

    print(f"\n{'=' * 70}")
    print(f"Pasta: {ECGS_REAIS3_DIR}")
    print(f"Amostra: {len(sample)} imagens (seed={RANDOM_SEED})")
    print(f"{'=' * 70}")

    results: list[tuple[str, dict]] = []
    for path in sample:
        cal = run_calibration_for_image(str(path), use_mock=False)
        results.append((str(path), cal))

    # Resumo focado em ganho (mm/mV) e px_per_mm
    print(f"\n{'=' * 78}")
    print("RESUMO — ganho (mm/mV) e px_per_mm por imagem")
    print(f"{'=' * 78}")
    print(f"  {'Imagem':45s}  {'px/mm':>6s}  {'mm/mV':>6s}  {'Fonte':>8s}")
    print(f"  {'-' * 45}  {'-' * 6}  {'-' * 6}  {'-' * 8}")
    for path, cal in results:
        name = Path(path).name[:45]
        if "error" in cal:
            print(f"  {name:45s}  ERRO: {cal['error']}")
            continue
        ppm = cal["px_per_mm"]
        gain = cal["gain_mm_per_mV"]
        src = cal["calibration_source"]
        print(f"  {name:45s}  {ppm:>6.2f}  {gain:>6.1f}  {src:>8s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
