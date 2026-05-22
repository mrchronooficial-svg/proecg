"""
Valida a correção do offset horizontal do overlay Stenhede.

Roda em IMG_1378 e IMG_1303 da pasta ECGs Undistorted, gera:
  - IMG_1378_overlay_corrigido.png
  - IMG_1303_overlay_corrigido.png

Imprime o `x_offset` detectado e diagnóstico de alinhamento.
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

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.stenhede_adapter import extract_signals_stenhede

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
TARGETS = ["IMG_1378", "IMG_1303"]


def _calibrate(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    _, kps = digitizer.dotter(img)
    if len(kps) < 4:
        _, kps = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(kps, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _check_alignment_dx(
    image_bgr: np.ndarray, raw_lines_padded: np.ndarray,
    search_px: int = 30,
) -> tuple[int, float]:
    """Sweep X-offset em [-search_px, +search_px] e mede onde os pixels
    da linha extraída caem sobre os pixels mais ESCUROS da imagem
    original. Retorna (best_dx, improvement_pct)."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dxs = np.arange(-search_px, search_px + 1)
    scores = []
    valid_mask = ~np.isnan(raw_lines_padded)
    for dx in dxs:
        sc = 0.0
        for li in range(raw_lines_padded.shape[0]):
            row_valid = valid_mask[li]
            if row_valid.sum() < 50:
                continue
            xs_orig = np.where(row_valid)[0]
            xs_shift = xs_orig + dx
            mask = (xs_shift >= 0) & (xs_shift < w)
            xs_shift = xs_shift[mask]
            xs_use = xs_orig[mask]
            ys = raw_lines_padded[li, xs_use]
            ys_int = np.clip(np.round(ys).astype(int), 0, h - 1)
            sc += float((255.0 - gray[ys_int, xs_shift]).sum())
        scores.append(sc)
    scores_arr = np.array(scores)
    best_i = int(np.argmax(scores_arr))
    best_dx = int(dxs[best_i])
    score_zero = float(scores_arr[search_px])
    score_best = float(scores_arr[best_i])
    impr = (score_best / max(score_zero, 1.0) - 1.0) * 100.0
    return best_dx, impr


def _render_overlay(
    image_bgr: np.ndarray, raw_lines_padded: np.ndarray,
    title: str, out_path: Path,
    blend_white: float = 0.40, color: tuple = (0.85, 0, 0),
) -> None:
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = (1 - blend_white) * img_rgb + blend_white
    img_blend = np.clip(img_blend, 0, 1)
    fig_w = max(16.0, w / 200.0); fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    if raw_lines_padded.ndim == 2 and raw_lines_padded.size > 0:
        x = np.arange(raw_lines_padded.shape[1])
        for i in range(raw_lines_padded.shape[0]):
            ax.plot(x, raw_lines_padded[i], color=color, lw=1.5,
                    alpha=0.85, zorder=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _process(stem: str) -> int:
    img_path = UNDIST_DIR / f"{stem}.png"
    if not img_path.exists():
        print(f"[ERRO] {img_path} nao existe")
        return 1
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] falha ao ler {img_path}")
        return 2
    h, w = img.shape[:2]
    print(f"\n{'=' * 78}")
    print(f" {stem} -- shape=(H={h}, W={w})")
    print("=" * 78)

    cal = _calibrate(img)
    print(f"calibracao: px/mm={cal['px_per_mm']:.3f} fs={cal['sampling_rate_hz']:.1f}Hz")

    t0 = time.perf_counter()
    result = extract_signals_stenhede(
        image_bgr=img, px_per_mm=float(cal["px_per_mm"]),
    )
    dt = time.perf_counter() - t0

    raw = result["raw_lines_pixel"]
    x_off = result.get("raw_lines_x_offset", 0)
    n = raw.shape[0]
    valid_per_line = (~np.isnan(raw)).sum(axis=1) if n > 0 else np.array([])
    layout = result["match"].get("layout", "?")
    cost = result["match"].get("cost", float("nan"))

    print(
        f"\nextract_signals_stenhede em {dt:.1f}s -- layout={layout} cost={cost:.2f} "
        f"linhas={n} x_offset={x_off}"
    )
    print(f"raw_lines_pixel.shape={raw.shape}  (W esperado={w})")
    if n > 0:
        for i in range(n):
            v = raw[i]
            valid = ~np.isnan(v)
            if valid.any():
                first_x = int(np.argmax(valid))
                last_x = len(v) - int(np.argmax(valid[::-1])) - 1
                print(
                    f"  linha {i}: cobre x=[{first_x:5d}, {last_x:5d}] "
                    f"({int(valid_per_line[i])} samples válidos), "
                    f"primeiros 5 Y: "
                    f"{[f'{y:.1f}' for y in v[first_x:first_x+5].tolist()]}"
                )

    # Diagnóstico de alinhamento residual
    if n > 0:
        best_dx, impr = _check_alignment_dx(img, raw, search_px=30)
        print(
            f"\nAlinhamento residual: best dx={best_dx:+d} px "
            f"({best_dx / cal['px_per_mm']:+.2f} mm) "
            f"melhora vs dx=0: {impr:.1f}%"
        )
        if abs(best_dx) <= 3:
            print("  -> alinhamento OK (drift sub-3px)")
        else:
            print(f"  -> drift residual de {best_dx} px (não corrigido)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stem}_overlay_corrigido.png"
    _render_overlay(
        img, raw,
        f"{stem} -- Overlay corrigido (x_offset={x_off})",
        out_path,
    )
    print(f"\n[Render] {out_path.name}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rc = 0
    for stem in TARGETS:
        rc = _process(stem) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
