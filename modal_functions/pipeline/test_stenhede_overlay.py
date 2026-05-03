"""
Overlay do sinal extraído por cima da imagem undistorted IMG_1279.
==================================================================

Roda DUAS extrações (U-Net Stenhede vs máscara gabarito) e plota cada
linha extraída — em coordenadas de PIXEL na imagem original — sobreposta
à foto undistorted. Se a extração estiver fiel, a linha vermelha cai
sobre o traçado preto do ECG.

Saídas em modal_functions/pipeline/digitize/_visualizations/:
  - IMG_1279_overlay_stenhede_unet.png
  - IMG_1279_overlay_gabarito_mask.png

Uso:
    python -m modal_functions.pipeline.test_stenhede_overlay
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

from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.lead_separator import separate_and_extract
from .digitize.stenhede_adapter import (
    extract_lines_from_cell,
    extract_signal_probabilities,
)

# Permite escolher imagem via arg CLI (default IMG_1279).
_IMG_NAME = sys.argv[1] if len(sys.argv) > 1 else "IMG_1279"
if not _IMG_NAME.lower().endswith(".png"):
    _IMG_NAME = f"{_IMG_NAME}.png"

UNDIST_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted") / _IMG_NAME
MASK_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\Leader Masks Teste") / _IMG_NAME
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
_IMG_STEM = Path(_IMG_NAME).stem


def _calibrate_normalized(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    _, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        _, keypoints = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _extract_lines_in_pixel_coords(
    signal_prob: np.ndarray,
    leads_info: dict[str, dict],
) -> dict[str, dict]:
    """Para cada lead em `leads_info` (saída de separate_and_extract),
    re-executa o SignalExtractor no recorte do `signal_prob` correspondente
    e devolve a linha em COORDS GLOBAIS de pixel (NÃO em µV).

    Returns:
        dict[name] = {"x_global": np.ndarray, "y_global": np.ndarray (NaN
        onde não há linha), "n_samples": int, "nan_pct": float}
    """
    H, W = signal_prob.shape
    out: dict[str, dict] = {}
    for name, info in leads_info.items():
        x1, y1, x2, y2 = info["bbox"]
        x1c = max(0, int(x1)); y1c = max(0, int(y1))
        x2c = min(W, int(x2)); y2c = min(H, int(y2))
        if x2c <= x1c or y2c <= y1c:
            out[name] = {
                "x_global": np.zeros(0, dtype=np.float64),
                "y_global": np.zeros(0, dtype=np.float64),
                "n_samples": 0, "nan_pct": 100.0,
            }
            continue
        cell = signal_prob[y1c:y2c, x1c:x2c]
        line_local = extract_lines_from_cell(cell)         # (W_cell,) em coords da cell
        # Y global = local + y1c. NaN preservado.
        line_global = line_local.astype(np.float64) + float(y1c)
        x_global = np.arange(line_global.shape[0], dtype=np.float64) + float(x1c)
        valid = ~np.isnan(line_global)
        nan_pct = 100.0 * (1.0 - valid.sum() / max(line_global.shape[0], 1))
        out[name] = {
            "x_global": x_global,
            "y_global": line_global,
            "n_samples": int(line_global.shape[0]),
            "nan_pct": float(nan_pct),
        }
    return out


def _render_overlay(
    image_bgr: np.ndarray,
    lines_pixel: dict[str, dict],
    title: str,
    out_path: Path,
    blend_white: float = 0.40,
) -> None:
    """Renderiza a imagem undistorted com as linhas extraídas em
    vermelho por cima (em coords de pixel)."""
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Clarear: blend com branco
    img_blend = (1.0 - blend_white) * img_rgb + blend_white * 1.0
    img_blend = np.clip(img_blend, 0.0, 1.0)

    # figsize proporcional ao tamanho original
    fig_w = max(16.0, w / 200.0)
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Y cresce pra baixo, igual à imagem
    ax.set_xticks([])
    ax.set_yticks([])

    for name, info in lines_pixel.items():
        x = info["x_global"]
        y = info["y_global"]
        if x.size == 0:
            continue
        ax.plot(x, y, color="red", linewidth=1.5, alpha=0.85, zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    if not UNDIST_PATH.exists():
        print(f"ERRO: undistorted nao encontrada: {UNDIST_PATH}", file=sys.stderr)
        return 1
    if not MASK_PATH.exists():
        print(f"ERRO: mascara gabarito nao encontrada: {MASK_PATH}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(f" OVERLAY Stenhede UNet vs Mascara Gabarito -- {_IMG_STEM}")
    print("=" * 78)

    img = cv2.imread(str(UNDIST_PATH))
    mask_raw = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("ERRO: falha ao ler undistorted", file=sys.stderr); return 3
    if mask_raw is None:
        print("ERRO: falha ao ler mascara", file=sys.stderr); return 4
    print(f"  img shape  : {img.shape}")
    print(f"  mask shape : {mask_raw.shape}")
    if mask_raw.shape[:2] != img.shape[:2]:
        mask_raw = cv2.resize(
            mask_raw, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    cal = _calibrate_normalized(img)
    fs = float(cal["sampling_rate_hz"])
    print(
        f"\n  calibracao: px/mm={cal['px_per_mm']:.3f} fs={fs:.1f}Hz "
        f"uv/px={cal['uv_per_pixel']:.2f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============== TESTE A — Stenhede U-Net ==============
    print("\n[A] Stenhede UNet -> SignalExtractor")
    t0 = time.perf_counter()
    try:
        prob_unet = extract_signal_probabilities(img)
    except Exception as e:
        print(f"[A][ERRO] extract_signal_probabilities: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 5
    print(f"    UNet: {(time.perf_counter()-t0)*1000:.0f} ms, "
          f"prob range=[{prob_unet.min():.3f}, {prob_unet.max():.3f}]")

    try:
        sep_a = separate_and_extract(
            mask=None, normalized_image=img, calibration=cal,
            signal_prob=prob_unet,
        )
    except Exception as e:
        print(f"[A][ERRO] separate_and_extract: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 6
    lines_a = _extract_lines_in_pixel_coords(prob_unet, sep_a["leads"])

    # ============== TESTE B — Mascara Gabarito ==============
    print("\n[B] Mascara gabarito -> SignalExtractor")
    prob_gab = (mask_raw.astype(np.float32) / 255.0).astype(np.float32)
    print(
        f"    prob_gab px_ativos>0.5: {int(np.sum(prob_gab > 0.5))}"
    )
    try:
        sep_b = separate_and_extract(
            mask=None, normalized_image=img, calibration=cal,
            signal_prob=prob_gab,
        )
    except Exception as e:
        print(f"[B][ERRO] separate_and_extract: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 7
    lines_b = _extract_lines_in_pixel_coords(prob_gab, sep_b["leads"])

    # ============== Renders ==============
    out_a = OUT_DIR / f"{_IMG_STEM}_overlay_stenhede_unet.png"
    out_b = OUT_DIR / f"{_IMG_STEM}_overlay_gabarito_mask.png"
    print(f"\n[Render] {out_a.name}")
    _render_overlay(img, lines_a,
                    f"{_IMG_STEM} -- Overlay: Stenhede U-Net (linha vermelha)",
                    out_a)
    print(f"[Render] {out_b.name}")
    _render_overlay(img, lines_b,
                    f"{_IMG_STEM} -- Overlay: Mascara Gabarito (linha vermelha)",
                    out_b)

    # ============== Tabela ==============
    LEAD_PRINT = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6", "II_rhythm"]
    print("\n=== TABELA COMPARATIVA ===")
    print(
        f"{'Lead':<10}{'NaN% UNet':>11}{'NaN% Gab':>11}"
        f"{'Samples UNet':>15}{'Samples Gab':>15}"
    )
    print("-" * 62)
    for name in LEAD_PRINT:
        a = lines_a.get(name)
        b = lines_b.get(name)
        if a is None or b is None:
            print(f"{name:<10}  (ausente)")
            continue
        print(
            f"{name:<10}{a['nan_pct']:>10.1f}%{b['nan_pct']:>10.1f}%"
            f"{a['n_samples']:>15d}{b['n_samples']:>15d}"
        )
    print(f"\nSalvos em: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
