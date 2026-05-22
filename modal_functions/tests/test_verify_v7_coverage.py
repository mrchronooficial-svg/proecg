"""
Verifica se alguma parte da mascara do canal 2 ficou de fora do render v7.

Pra cada ECG:
  1. Carrega mascara (signal_prob > 0.05)
  2. Detecta bandas (mesma logica do v7)
  3. Identifica pixels INSIDE bandas vs OUTSIDE
  4. Reporta coverage
  5. Salva imagem de overlay: mascara em verde, missed em vermelho
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
logger = logging.getLogger("verify_v7")

ECG_LIST = [
    "IMG_1407",
    "IMG_1316",
    "0a8c7db0-e31a-4e64-a15e-f7d29c1f7661",
]
RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
OUTPUT_DIR = RESULTS_ROOT / "ecg_training_format_v7" / "verificacao_coverage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACE_THRESHOLD = 0.05
BBOX_THRESHOLD = 0.5


def _detect_bands(signal_prob):
    binary = signal_prob > BBOX_THRESHOLD
    row_count = binary.sum(axis=1).astype(float)
    if row_count.sum() == 0:
        return []
    smoothed = gaussian_filter1d(row_count, sigma=10)
    peaks, _ = find_peaks(smoothed, distance=50, prominence=smoothed.max() * 0.1)
    if peaks.size == 0:
        return []
    valleys = [0]
    for i in range(len(peaks) - 1):
        segment = smoothed[peaks[i]:peaks[i + 1]]
        local_min = int(peaks[i] + np.argmin(segment))
        valleys.append(local_min)
    valleys.append(len(smoothed))
    bands = []
    for i in range(len(peaks)):
        bands.append((valleys[i], valleys[i + 1]))
    return bands


def main() -> int:
    for stem in ECG_LIST:
        cache_path = RESULTS_ROOT / stem / "_cache_pre_signal_extractor.pkl"
        if not cache_path.is_file():
            logger.warning("Cache nao existe pra %s — pulando", stem)
            continue
        logger.info("===== %s =====", stem)
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        signal_prob = cache["signal_prob"]
        normalized = cache["normalized"]
        H, W = signal_prob.shape

        # Mascara do tracado (threshold 0.05 = TRACE_THRESHOLD do v7)
        trace_mask = signal_prob > TRACE_THRESHOLD
        n_trace = int(trace_mask.sum())

        # Bandas (mesma logica do v7)
        bands = _detect_bands(signal_prob)
        n_bands = len(bands)
        logger.info("%d bandas detectadas", n_bands)

        # Bbox horizontal (mesmo crop do v7)
        binary_strong = signal_prob > BBOX_THRESHOLD
        cols_with = np.any(binary_strong, axis=0)
        if cols_with.any():
            x_min = int(np.argmax(cols_with))
            x_max = W - int(np.argmax(cols_with[::-1]))
        else:
            x_min, x_max = 0, W

        # Determina quais pixels caem DENTRO de alguma banda + dentro do x range
        inside_render = np.zeros_like(trace_mask)
        for y0, y1 in bands:
            inside_render[y0:y1, x_min:x_max] = True

        # Trace pixels que vao pro render vs missed
        rendered = trace_mask & inside_render
        missed = trace_mask & ~inside_render
        n_rendered = int(rendered.sum())
        n_missed = int(missed.sum())
        pct_rendered = 100.0 * n_rendered / max(n_trace, 1)
        pct_missed = 100.0 * n_missed / max(n_trace, 1)

        # Categoriza missed:
        # - Acima da primeira banda (y < bands[0][0])
        # - Abaixo da ultima banda (y > bands[-1][1])
        # - Nas valleys (entre bandas)
        # - Fora do x range (x < x_min ou x >= x_max)
        if bands:
            above_first = np.zeros_like(trace_mask)
            above_first[:bands[0][0], :] = True
            below_last = np.zeros_like(trace_mask)
            below_last[bands[-1][1]:, :] = True
            in_valleys = np.zeros_like(trace_mask)
            for i in range(len(bands) - 1):
                in_valleys[bands[i][1]:bands[i + 1][0], :] = True
            outside_x = np.zeros_like(trace_mask)
            outside_x[:, :x_min] = True
            outside_x[:, x_max:] = True

            n_above = int((trace_mask & above_first).sum())
            n_below = int((trace_mask & below_last).sum())
            n_valley = int((trace_mask & in_valleys).sum())
            n_outx = int((trace_mask & outside_x).sum())
        else:
            n_above = n_below = n_valley = n_outx = 0

        logger.info(
            "Mascara total: %d | Rendered: %d (%.2f%%) | Missed: %d (%.2f%%)",
            n_trace, n_rendered, pct_rendered, n_missed, pct_missed,
        )
        logger.info(
            "  Acima primeira banda: %d | Abaixo ultima: %d | Em valleys: %d | "
            "Fora do x range: %d",
            n_above, n_below, n_valley, n_outx,
        )

        # Imagem de overlay: undistorted + render area verde + missed pixels vermelhos
        overlay = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB).copy()
        # Bandas: highlight verde semi-transparente
        for y0, y1 in bands:
            overlay[y0:y1, x_min:x_max, 1] = np.minimum(
                255, overlay[y0:y1, x_min:x_max, 1].astype(int) + 30,
            )
        # Mascara renderizada: ponto verde forte
        ys_r, xs_r = np.where(rendered)
        # Subamostra pra plot ser leve
        stride = max(1, n_rendered // 5000)
        overlay_pil = overlay.copy()
        overlay_pil[ys_r[::stride], xs_r[::stride]] = [0, 220, 0]
        # Pixels missed: vermelho forte
        ys_m, xs_m = np.where(missed)
        overlay_pil[ys_m, xs_m] = [255, 0, 0]

        # Adiciona texto resumo no topo
        text = f"{stem}: rendered={n_rendered} ({pct_rendered:.1f}%), MISSED={n_missed} ({pct_missed:.2f}%)"
        cv2.putText(overlay_pil, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        text2 = f"acima:{n_above} abaixo:{n_below} valleys:{n_valley} fora_x:{n_outx}"
        cv2.putText(overlay_pil, text2, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        out_path = OUTPUT_DIR / f"{stem}_coverage.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay_pil, cv2.COLOR_RGB2BGR))
        logger.info("Salvo: %s", out_path)

    logger.info("Tudo concluido. Outputs em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
