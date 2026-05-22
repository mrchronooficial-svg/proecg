"""
v7: trace preenche 820x370 com aspect preservado por BANDA.

Algoritmo:
  1. Detecta numero de bandas (rows) no mask via row-projection + find_peaks
  2. Canvas 900x450 com grid 232/245
  3. Divide area util em N strips horizontais de altura igual (370/N)
  4. Cada banda do mask vira uma strip, preserva aspect dentro dela
  5. Compoe tudo

Resultado: outputs sempre preenchem 820x370 (igual ao treino), independente
do layout (12x1, 3x4, 6x2).
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecg_training_format_v7")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 900
CANVAS_H = 450
MARGIN = 40
USABLE_W = CANVAS_W - 2 * MARGIN  # 820
USABLE_H = CANVAS_H - 2 * MARGIN  # 370
TRACE_THRESHOLD = 0.05
BBOX_THRESHOLD = 0.5
ALPHA_BOOST = 1.5


def _detect_bands(signal_prob):
    """Detecta bandas (rows) via row-projection. Cada banda eh SIMETRICA
    em torno do peak — primeira e ultima banda recebem mesma altura que
    seus vizinhos, evitando "espaco vazio" acima/abaixo."""
    binary = signal_prob > BBOX_THRESHOLD
    row_count = binary.sum(axis=1).astype(float)
    if row_count.sum() == 0:
        return []
    smoothed = gaussian_filter1d(row_count, sigma=10)
    peaks, _ = find_peaks(smoothed, distance=50, prominence=smoothed.max() * 0.1)
    if peaks.size == 0:
        return []

    # Valleys reais entre peaks consecutivos
    inner_valleys = []
    for i in range(len(peaks) - 1):
        segment = smoothed[peaks[i]:peaks[i + 1]]
        local_min = int(peaks[i] + np.argmin(segment))
        inner_valleys.append(local_min)

    L = len(smoothed)
    bands = []
    # Primeira/ultima banda recebem 20% extra de buffer pra capturar QRS spike
    # que pode extrapolar a distancia simetrica
    BUFFER_FACTOR = 1.2
    for i, peak in enumerate(peaks):
        # Boundary superior
        if i == 0:
            if inner_valleys:
                half_dist = int((inner_valleys[0] - peak) * BUFFER_FACTOR)
                y0 = max(0, peak - half_dist)
            else:
                y0 = 0
        else:
            y0 = inner_valleys[i - 1]
        # Boundary inferior
        if i == len(peaks) - 1:
            if inner_valleys:
                half_dist = int((peak - inner_valleys[-1]) * BUFFER_FACTOR)
                y1 = min(L, peak + half_dist)
            else:
                y1 = L
        else:
            y1 = inner_valleys[i]
        bands.append((y0, y1))
    return bands


def _render_band_alpha(signal_prob_band, target_w, target_h, signal_max):
    """Renderiza uma banda com alpha gradient + boost + INTER_AREA resize."""
    h, w = signal_prob_band.shape
    canvas = np.full((h, w), 255.0, dtype=np.float32)
    trace_mask = signal_prob_band > TRACE_THRESHOLD
    trace_intensity = np.clip(
        signal_prob_band / max(signal_max, 1e-6) * ALPHA_BOOST, 0, 1,
    )
    alpha_t = trace_intensity * trace_mask.astype(np.float32)
    canvas = canvas * (1.0 - alpha_t)
    canvas_u8 = canvas.astype(np.uint8)
    # STRETCH resize pra (target_w, target_h) — preserva aspect dentro da banda
    # (i.e., aspect ratio do conteudo VS strip target nao e exatamente igual,
    # mas a banda enche a strip)
    return cv2.resize(canvas_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_v7(cache):
    signal_prob = cache["signal_prob"]
    H, W = signal_prob.shape

    # 1. Detecta bandas
    bands = _detect_bands(signal_prob)
    if not bands:
        return np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8), 0, []
    n_bands = len(bands)
    logger.info("Bandas detectadas: %d", n_bands)

    # 2. Bbox horizontal de TODAS as bandas juntas (pra cropar lado a lado se necessario)
    binary = signal_prob > BBOX_THRESHOLD
    cols_with = np.any(binary, axis=0)
    x_min = int(np.argmax(cols_with))
    x_max = W - int(np.argmax(cols_with[::-1]))

    # 3. Canvas branco 900x450
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)

    # 4. Calcula strip por banda — altura igual pra todas
    strip_h = USABLE_H // n_bands
    # px_per_mm: assume ~250mm de largura total (10s × 25mm/s)
    px_per_mm_target = USABLE_W / 250.0
    # Strip height em mm: assume cada banda = altura natural ~10-25mm dependendo do layout
    # mas pra uniformidade visual, todas strips ficam com mesma altura

    # 5. Grid no canvas (px_per_mm uniforme, X e Y)
    p_min = px_per_mm_target
    p_maj = 5 * px_per_mm_target

    def draw_lines(canvas, period, color, axis):
        H_, W_ = canvas.shape
        if axis == "y":
            offset = MARGIN
            v = offset
            while v < H_:
                vi = int(round(v))
                if 0 <= vi < H_:
                    canvas[vi, :] = np.minimum(canvas[vi, :], color)
                v += period
            v = offset - period
            while v >= 0:
                vi = int(round(v))
                if 0 <= vi < H_:
                    canvas[vi, :] = np.minimum(canvas[vi, :], color)
                v -= period
        else:
            offset = MARGIN
            v = offset
            while v < W_:
                vi = int(round(v))
                if 0 <= vi < W_:
                    canvas[:, vi] = np.minimum(canvas[:, vi], color)
                v += period
            v = offset - period
            while v >= 0:
                vi = int(round(v))
                if 0 <= vi < W_:
                    canvas[:, vi] = np.minimum(canvas[:, vi], color)
                v -= period

    draw_lines(canvas, p_min, 245, "y")
    draw_lines(canvas, p_min, 245, "x")
    draw_lines(canvas, p_maj, 232, "y")
    draw_lines(canvas, p_maj, 232, "x")

    # 6. Pra cada banda: crop sinal, renderiza com alpha, posiciona na strip
    signal_max = float(signal_prob.max())
    for i, (y0, y1) in enumerate(bands):
        # Crop sinal da banda — bbox horizontal total
        sp_band = signal_prob[y0:y1, x_min:x_max]
        # Strip target dimensions
        strip_y0 = MARGIN + i * strip_h
        strip_y1 = strip_y0 + strip_h
        # Render
        rendered = _render_band_alpha(
            sp_band, USABLE_W, strip_h, signal_max,
        )
        # Compose com minimum
        canvas[strip_y0:strip_y1, MARGIN:MARGIN + USABLE_W] = np.minimum(
            canvas[strip_y0:strip_y1, MARGIN:MARGIN + USABLE_W],
            rendered,
        )

    return canvas, n_bands, bands


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.warning("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        img, n_bands, bands = make_v7(cache)
        out_path = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out_path), img)
        logger.info("Salvo: %s (900x450, %d bandas)", out_path.name, n_bands)
    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
