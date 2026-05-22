"""
Cria um ECG digital do zero:
  - Canvas branco
  - Grid gerado matematicamente (px_per_mm do cache) — alinhado com o
    original (detecta fase via cross-correlacao da projecao grid_prob)
  - SOMENTE o tracado do canal 2 (signal_prob > threshold) em preto

Output:
  ~/Desktop/Projeto ECG/resultados_teste_v1/IMG_1407/ecg_digital_v2.png
  ~/Desktop/Projeto ECG/resultados_teste_v1/IMG_1407/ecg_digital_v2_vs_original.png
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("digital_ecg_v2")

CACHE_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407"
    r"\_cache_pre_signal_extractor.pkl"
)
OUTPUT_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407\ecg_digital_v2.png"
)
OUTPUT_COMPARISON = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407"
    r"\ecg_digital_v2_vs_original.png"
)

TRACE_THRESHOLD = 0.05

# Cores estilo ECG papel
COLOR_BG = (255, 255, 255)         # branco
COLOR_GRID_MINOR = (255, 200, 210) # rosa claro (1mm)
COLOR_GRID_MAJOR = (250, 130, 140) # rosa medio (5mm)
COLOR_TRACE = (0, 0, 0)            # preto


def _detect_phase(projection: np.ndarray, period: float) -> int:
    """Detecta fase (offset) de um grid periodico via cross-correlacao
    com um pente sintetico."""
    L = len(projection)
    period_int = int(round(period))
    if period_int < 2 or L < 3 * period_int:
        return 0
    # Pente: 1 nos multiplos de period, 0 caso contrario
    comb = np.zeros(L, dtype=np.float32)
    for i in range(0, L, period_int):
        comb[i] = 1.0
    # Normaliza projection
    proj = projection - projection.mean()
    proj_max = float(np.abs(proj).max())
    if proj_max > 0:
        proj = proj / proj_max
    # Cross-correlate
    corr = correlate(proj, comb, mode="full")
    # Lag central
    center = len(corr) // 2
    # Procura o pico no range de fases possiveis (0..period)
    best_lag = 0
    best_val = -float("inf")
    for lag in range(-period_int, period_int + 1):
        v = float(corr[center + lag])
        if v > best_val:
            best_val = v
            best_lag = lag
    return int(best_lag) % period_int


def _draw_grid_lines(canvas: np.ndarray, px_per_mm: float,
                      offset_y: int, offset_x: int) -> None:
    """Desenha grid 1mm (rosa claro) + 5mm (rosa medio) no canvas in-place."""
    H, W, _ = canvas.shape
    # MINOR grid (1mm)
    period_minor = px_per_mm
    # Linhas horizontais
    y = offset_y
    while y < H:
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = COLOR_GRID_MINOR
        y += period_minor
    y = offset_y - period_minor
    while y >= 0:
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = COLOR_GRID_MINOR
        y -= period_minor
    # Linhas verticais
    x = offset_x
    while x < W:
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = COLOR_GRID_MINOR
        x += period_minor
    x = offset_x - period_minor
    while x >= 0:
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = COLOR_GRID_MINOR
        x -= period_minor

    # MAJOR grid (5mm) — mais escuro, sobrescreve o minor
    period_major = 5 * px_per_mm
    y = offset_y
    while y < H:
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = COLOR_GRID_MAJOR
        y += period_major
    y = offset_y - period_major
    while y >= 0:
        yi = int(round(y))
        if 0 <= yi < H:
            canvas[yi, :] = COLOR_GRID_MAJOR
        y -= period_major
    x = offset_x
    while x < W:
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = COLOR_GRID_MAJOR
        x += period_major
    x = offset_x - period_major
    while x >= 0:
        xi = int(round(x))
        if 0 <= xi < W:
            canvas[:, xi] = COLOR_GRID_MAJOR
        x -= period_major


def main() -> int:
    if not CACHE_PATH.is_file():
        logger.error("Cache nao existe")
        return 1
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)

    signal_prob = cache["signal_prob"]
    grid_prob = cache["grid_prob"]
    normalized = cache["normalized"]
    px_size = cache["pixel_size"]
    px_per_mm = float(px_size["avg_pixel_per_mm"])

    H, W = signal_prob.shape
    logger.info("Shape %dx%d, px_per_mm=%.3f, 5mm=%.1fpx",
                H, W, px_per_mm, 5 * px_per_mm)

    # Detecta fase do grid major (5mm) usando cross-correlacao das projecoes
    proj_y = grid_prob.sum(axis=1)  # (H,) projecao horizontal -> picos onde tem linha horizontal
    proj_x = grid_prob.sum(axis=0)  # (W,) projecao vertical -> picos onde tem linha vertical
    offset_y = _detect_phase(proj_y, 5 * px_per_mm)
    offset_x = _detect_phase(proj_x, 5 * px_per_mm)
    logger.info("Fase detectada do grid major: offset_y=%d, offset_x=%d",
                offset_y, offset_x)

    # 1. Canvas branco
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # 2. Grid
    _draw_grid_lines(canvas, px_per_mm, offset_y, offset_x)
    logger.info("Grid desenhado")

    # 3. Tracado: canal 2 binarizado, em preto
    trace_mask = signal_prob > TRACE_THRESHOLD
    canvas[trace_mask] = COLOR_TRACE
    logger.info("Tracado plotado: %d pixels", int(trace_mask.sum()))

    # Salva
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    logger.info("Salvo: %s", OUTPUT_PATH)

    # Comparacao lado a lado
    fig, axes = plt.subplots(2, 1, figsize=(20, 22), dpi=110)
    axes[0].imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("ECG ORIGINAL (foto undistorted)",
                       fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(canvas)
    axes[1].set_title(
        f"ECG DIGITAL v2 (grid sintetico {px_per_mm:.2f}px/mm, "
        f"fase y={offset_y} x={offset_x} | so tracado UNet canal 2)",
        fontsize=14, fontweight="bold",
    )
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(str(OUTPUT_COMPARISON), bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("Comparacao: %s", OUTPUT_COMPARISON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
