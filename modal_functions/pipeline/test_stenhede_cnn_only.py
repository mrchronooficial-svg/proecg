"""Stenhede + CNN nas 4 imagens já undistorted (skip Dotter/undistortion).

Imprime tabela com diagnóstico CNN por ECG.
Uso:
    python -m modal_functions.pipeline.test_stenhede_cnn_only
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .classify_v2 import (
    CLASSES_24,
    DIAG_TEXT,
    build_cnn_input,
    classify_signal,
    load_cnn_model,
)
from .digitize.constants import GAIN_DEFAULT, PAPER_SPEED_DEFAULT
from .digitize.stenhede_adapter import extract_signals_stenhede
from .test_pipeline_10ecgs import (
    CELL_LAYOUTS,
    LEAD_NAMES,
    _build_signals_faithful,
    _detect_format,
)

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
TARGETS = ["IMG_1275", "IMG_1279", "IMG_1303", "IMG_1312"]


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")
    print(f"[*] Carregando CNN...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = load_cnn_model(device=device)
    print(f"   pronto (device={device})\n")

    rows: list[dict] = []
    for stem in TARGETS:
        img_path = UNDIST_DIR / f"{stem}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERRO] {img_path}")
            continue
        print(f"\n=== {stem}  shape={img.shape} ===")
        t0 = time.time()
        try:
            stenhede = extract_signals_stenhede(
                image_bgr=img,
                use_cropper=False,
                use_internal_pixel_size=True,
                paper_speed=float(PAPER_SPEED_DEFAULT),
                voltage_gain=float(GAIN_DEFAULT),
            )
        except Exception as e:
            print(f"   ERRO Stenhede: {type(e).__name__}: {e}")
            continue

        raw_lines = np.asarray(stenhede["raw_lines_pixel"], dtype=np.float64)
        layout_name = (stenhede.get("match") or {}).get("layout") or "standard_3x4_with_r1"
        canonical = stenhede.get("canonical_lines_uv")
        chunk_px = int(canonical.shape[1]) if canonical is not None and canonical.size else 0
        x_offset = int(stenhede.get("raw_lines_x_offset", 0))
        sampling_rate = float(stenhede["sampling_rate_hz"])
        px_per_mm = float(stenhede["avg_pixel_per_mm"])
        ecg_fmt = _detect_format(layout_name)
        print(f"   layout={layout_name}  fmt={ecg_fmt}  px/mm={px_per_mm:.2f}  fs={sampling_rate:.1f}Hz")
        print(f"   raw_lines: shape={raw_lines.shape}  x_offset={x_offset}  chunk_px={chunk_px}")

        # signals fiéis
        signals_faithful = _build_signals_faithful(
            raw_lines, ecg_fmt, x_offset, chunk_px, px_per_mm,
        )

        # CNN
        cnn_array = build_cnn_input(signals_faithful, sampling_rate)
        diagnosis = classify_signal(cnn, cnn_array, original_hz=sampling_rate)
        dt = time.time() - t0
        print(f"   tempo: {dt:.1f}s")

        # Imprime probabilidades top-5
        sorted_probs = sorted(
            diagnosis["all_probs"].items(), key=lambda kv: kv[1], reverse=True
        )[:8]
        print(f"   top-8 probs: {[(c, round(p, 3)) for c, p in sorted_probs]}")
        print(f"   isquemia: {[(d['code'], round(d['prob'], 2)) for d in diagnosis['isquemia']]}")
        print(f"   arritmia: {[(d['code'], round(d['prob'], 2)) for d in diagnosis['arritmia']]}")
        print(f"   outras:   {[(d['code'], round(d['prob'], 2)) for d in diagnosis['outras']]}")
        print(f"   normal:   {diagnosis['is_normal']}")

        rows.append({
            "name": stem,
            "fmt": ecg_fmt,
            "isquemia": diagnosis["isquemia"],
            "arritmia": diagnosis["arritmia"],
            "outras": diagnosis["outras"],
            "is_normal": diagnosis["is_normal"],
            "top": sorted_probs[:5],
            "time_s": dt,
        })

    # Tabela final
    print("\n" + "=" * 78)
    print("RESUMO — Diagnóstico CNN por ECG")
    print("=" * 78)
    print(f"{'ECG':<10} {'Fmt':<6} {'Isquemia':<25} {'Arritmia':<25} {'Outras':<20}")
    print("-" * 78)
    for r in rows:
        isq_s = ", ".join(f"{d['code']}({d['prob']:.2f})" for d in r["isquemia"]) or "—"
        arr_s = ", ".join(f"{d['code']}({d['prob']:.2f})" for d in r["arritmia"]) or "—"
        out_s = ", ".join(f"{d['code']}({d['prob']:.2f})" for d in r["outras"]) or "—"
        if r["is_normal"]:
            print(f"{r['name']:<10} {r['fmt']:<6} NORMAL — sem alterações")
        else:
            print(f"{r['name']:<10} {r['fmt']:<6} {isq_s:<25} {arr_s:<25} {out_s:<20}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
