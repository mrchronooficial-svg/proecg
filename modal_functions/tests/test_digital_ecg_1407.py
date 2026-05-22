"""
Cria uma versao DIGITAL/LIMPA do ECG IMG_1407 usando:
  - canal 0 (grid)   → linhas vermelhas do grid
  - canal 1 (text)   → texto preto (labels: I, II, III, aVR, ..., V6 + cabecalho)
  - canal 2 (signal) → traçado preto do ECG
  - canal 3 (bg)     → fundo branco

Output:
  ~/Desktop/Projeto ECG/resultados_teste_v1/IMG_1407/ecg_digital.png
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

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("digital_ecg")

CACHE_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407"
    r"\_cache_pre_signal_extractor.pkl"
)
OUTPUT_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407\ecg_digital.png"
)
OUTPUT_COMPARISON_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407\ecg_digital_vs_original.png"
)


def main() -> int:
    if not CACHE_PATH.is_file():
        logger.error("Cache nao existe: %s", CACHE_PATH)
        return 1

    logger.info("Carregando cache do IMG_1407...")
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)

    signal_prob = cache["signal_prob"]
    grid_prob = cache["grid_prob"]
    text_prob = cache["text_prob"]
    bg_prob = cache["bg_prob"]
    normalized = cache["normalized"]

    H, W = signal_prob.shape
    logger.info("Shape: %d x %d", H, W)

    # Canvas RGB branco
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # === Grid: vermelho claro (cor padrao ECG papel rosa) ===
    GRID_THRESHOLD = 0.05  # baixo pra pegar o grid fraco
    grid_mask = grid_prob > GRID_THRESHOLD
    # Intensidade gradual baseado na prob (mais escuro = mais confiante)
    grid_intensity = np.clip(grid_prob / max(grid_prob.max(), 1e-6), 0, 1)
    # Cor: rosa-salmao tipico de papel ECG
    grid_R = 240
    grid_G = 180
    grid_B = 200
    # Aplica grid no canvas
    alpha = (grid_intensity * 0.8)[..., np.newaxis]
    grid_color = np.array([grid_R, grid_G, grid_B], dtype=np.float32)
    canvas_f = canvas.astype(np.float32)
    grid_pixels = grid_mask[..., np.newaxis] * alpha
    canvas_f = canvas_f * (1 - grid_pixels) + grid_color * grid_pixels
    canvas = canvas_f.astype(np.uint8)
    logger.info("Grid plotado: %d pixels", int(grid_mask.sum()))

    # === Texto: preto ===
    TEXT_THRESHOLD = 0.3
    text_mask = text_prob > TEXT_THRESHOLD
    # Aplica texto preto onde houver mask
    canvas[text_mask] = [0, 0, 0]
    logger.info("Texto plotado: %d pixels", int(text_mask.sum()))

    # === Traçado ECG: preto bold ===
    TRACE_THRESHOLD = 0.05
    trace_mask = signal_prob > TRACE_THRESHOLD
    # Intensidade gradual baseado na prob — pixels mais confiantes ficam mais escuros
    trace_intensity = np.clip(signal_prob / max(signal_prob.max(), 1e-6), 0, 1)
    # Mistura preto com canvas atual
    canvas_f = canvas.astype(np.float32)
    alpha_t = (trace_intensity * 1.0)[..., np.newaxis]
    trace_pixels = trace_mask[..., np.newaxis] * alpha_t
    canvas_f = canvas_f * (1 - trace_pixels) + 0.0 * trace_pixels  # preto = 0
    canvas = canvas_f.astype(np.uint8)
    logger.info("Tracado plotado: %d pixels", int(trace_mask.sum()))

    # === Salva imagem digital ===
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Converte RGB -> BGR pra cv2
    cv2.imwrite(str(OUTPUT_PATH), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    logger.info("Salvo: %s", OUTPUT_PATH)

    # === Imagem comparativa: ECG original vs Digital ===
    fig, axes = plt.subplots(2, 1, figsize=(20, 22), dpi=110)
    axes[0].imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("ECG ORIGINAL (foto do papel, undistorted)",
                       fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(canvas)
    axes[1].set_title("ECG DIGITAL (reconstrucao apenas com canais da UNet)",
                       fontsize=14, fontweight="bold")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(str(OUTPUT_COMPARISON_PATH), bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("Comparacao salva: %s", OUTPUT_COMPARISON_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
