"""
Roda extract_signals_stenhede e renderiza o overlay (raw_lines em pixel
sobre a foto undistorted) para TODAS as imagens .png da pasta
"ECGs Undistorted".

A U-Net Stenhede + LeadIdentifier ficam cached entre chamadas (lru_cache),
então a 2ª imagem em diante é mais rápida que a primeira.

Saída: IMG_<stem>_overlay_stenhede_full.png em
  modal_functions/pipeline/digitize/_visualizations/

Uso:
    python -m modal_functions.pipeline.test_stenhede_full_batch
"""

from __future__ import annotations

import logging
import re
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


def _safe_stem(p: Path) -> str:
    """Stem 'safe' para nomes de arquivo (substitui espaços/parens)."""
    s = p.stem
    s = re.sub(r"[\s().]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _calibrate(img: np.ndarray) -> dict | None:
    """Mantido pra compat — não é usado no caminho 100% Stenhede (Versão A)."""
    digitizer = ECGDigitizer(use_mock=False)
    _, kps = digitizer.dotter(img)
    if len(kps) < 4:
        _, kps = digitizer.dotter_mock(img)
    if len(kps) < 4:
        return None
    grid_matrix, _ = digitizer.gridder(kps, img.shape[:2])
    try:
        return calibrate(grid_matrix=grid_matrix, normalized_image=img)
    except Exception:
        return None


def _render_overlay(
    image_bgr: np.ndarray, raw_lines_pixel: np.ndarray,
    title: str, out_path: Path, blend_white: float = 0.40,
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
    if raw_lines_pixel.ndim == 2 and raw_lines_pixel.size > 0:
        x = np.arange(raw_lines_pixel.shape[1])
        for i in range(raw_lines_pixel.shape[0]):
            ax.plot(x, raw_lines_pixel[i], color="red", lw=1.5,
                    alpha=0.85, zorder=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _process_one(image_path: Path) -> tuple[str, str]:
    """Pipeline 100% Stenhede (Versão A): U-Net + PixelSizeFinder +
    SignalExtractor + LeadIdentifier. Sem cropper, sem o nosso calibrator."""
    img = cv2.imread(str(image_path))
    if img is None:
        return "ERRO", f"falha ao ler ({image_path.name})"

    try:
        result = extract_signals_stenhede(
            image_bgr=img,
            use_cropper=False,
            use_internal_pixel_size=True,
        )
    except Exception as e:
        return "FAIL_STENHEDE", f"{type(e).__name__}: {e}"

    raw_lines = result["raw_lines_pixel"]
    n_lines = result.get("n_lines_detected", 0)
    layout = result["match"].get("layout", "?")
    cost = result["match"].get("cost", float("nan"))
    pxmm = result.get("avg_pixel_per_mm", float("nan"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_stem(image_path)
    out_path = OUT_DIR / f"{safe}_overlay_stenhede_full.png"
    title = (
        f"{image_path.name} -- 100% Stenhede "
        f"(layout={layout}, cost={cost:.2f}, lines={n_lines}, "
        f"px/mm={pxmm:.2f})"
    )
    try:
        _render_overlay(img, raw_lines, title, out_path)
    except Exception as e:
        return "FAIL_RENDER", f"{type(e).__name__}: {e}"

    return "OK", (
        f"layout={layout} cost={cost:.2f} lines={n_lines} "
        f"px/mm={pxmm:.2f} file={out_path.name}"
    )


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")

    if not UNDIST_DIR.is_dir():
        print(f"ERRO: pasta nao encontrada: {UNDIST_DIR}", file=sys.stderr)
        return 1

    images = sorted(UNDIST_DIR.glob("*.png"))
    print("=" * 78)
    print(f" Stenhede overlay BATCH em {len(images)} imagens")
    print("=" * 78)
    print(f"Pasta: {UNDIST_DIR}")
    print(f"Saida: {OUT_DIR.resolve()}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, str, float]] = []
    t_start = time.perf_counter()

    for i, img_path in enumerate(images, 1):
        print(f"[{i:2d}/{len(images)}] {img_path.name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            status, msg = _process_one(img_path)
        except Exception as e:
            status, msg = "EXCEPTION", f"{type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
        results.append((img_path.name, status, msg, dt))
        prefix = "[OK]" if status == "OK" else f"[{status}]"
        print(f"  {prefix} ({dt:.1f}s) {msg}")

    total_s = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"RESUMO  ({total_s:.0f}s total)")
    print("=" * 78)
    n_ok = sum(1 for _, s, _, _ in results if s == "OK")
    print(f"  OK : {n_ok}/{len(results)}")
    n_fail = len(results) - n_ok
    if n_fail:
        print(f"  Falhas:")
        for name, status, msg, _dt in results:
            if status != "OK":
                print(f"    [{status}] {name}: {msg}")

    print(f"\nPNGs em: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
