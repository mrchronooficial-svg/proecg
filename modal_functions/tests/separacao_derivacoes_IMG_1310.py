"""
Separacao de derivacoes ECG combinando OCR (texto) + mascara de sinal.

Etapas:
  1. OCR no normalized -> labels (lead names) + calibracao (mm/s, mm/mV) + formato
  2. Band detection no canal 2 -> 4 bandas (3 rows + rhythm)
  3. Detec bar vertical (Sobel + morfologia) por banda -> chunk boundaries
  4. Fallback: equal split (W/cols) se bar detection falhar
  5. Crop canal 2 por derivacao (margem vertical 30%)
  6. Viterbi por mascara individual
  7. Convert px -> mV/segundos, save .npy/.csv, plota

Layout padrao 3x4+1 (confirmado pelo OCR "12 derivs; posicionamento padrao"):
  Row 0: I    aVR  V1  V4
  Row 1: II   aVL  V2  V5
  Row 2: III  aVF  V3  V6
  Row 3: II (rhythm)
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

from pipeline.digitize.viterbi_extractor import extrair_sinal_viterbi  # noqa: E402

# ----- Paths -----
BASE = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1310_pipeline_completo")
NORMALIZED_PATH = BASE / "00_normalized.png"
MASK_PATH = BASE / "05_canal_2_signal_PB.png"
OUTPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1310_separacao_ocr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Constants -----
LEAD_LAYOUT_3x4 = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
ALL_LEADS = [lead for row in LEAD_LAYOUT_3x4 for lead in row]
RHYTHM_LABEL = "II_rhythm"
VALID_LEAD_PATTERN = {"I", "II", "III", "aVR", "aVL", "aVF",
                      "V1", "V2", "V3", "V4", "V5", "V6"}
# Variantes que o OCR pode produzir
LEAD_ALIASES = {
    "I": "I", "II": "II", "III": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",
    "VI": "V1", "V|": "V1",  # OCR confusions comuns
}

VERTICAL_MARGIN_FRAC = 0.40  # 40% da altura da banda como margem


# =====================================================================
# Etapa 1: OCR
# =====================================================================

def normalize_lead_token(text: str) -> str | None:
    """Mapeia texto OCR -> lead canonico se for um lead conhecido."""
    t = text.strip().replace(" ", "")
    # Limpa caracteres parecidos com letras
    t = t.replace("|", "I").replace("l", "I").replace("0", "O")
    if t in LEAD_ALIASES:
        return LEAD_ALIASES[t]
    # Tenta uppercase
    if t.upper() in LEAD_ALIASES:
        return LEAD_ALIASES[t.upper()]
    return None


def run_ocr_on_strips(reader, img: np.ndarray, band_tops: list[int],
                      strip_height: int = 60, upscale: int = 2) -> list[dict]:
    """OCR em strips horizontais ao redor do topo de cada banda.
    Retorna lista de deteccoes {text, lead, cx, cy, conf, band_idx}."""
    H, W = img.shape[:2]
    detections = []
    for b_idx, y_top in enumerate(band_tops):
        y0 = max(0, y_top - 15)
        y1 = min(H, y_top + strip_height)
        strip = img[y0:y1, :]
        if upscale > 1:
            strip = cv2.resize(strip, (W * upscale, (y1 - y0) * upscale),
                               interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(strip, paragraph=False,
                                  width_ths=0.2, height_ths=0.5)
        for bbox, text, conf in results:
            lead = normalize_lead_token(text)
            if lead is None:
                continue
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            cx = int(np.mean(xs)) // upscale
            cy = (int(np.mean(ys)) // upscale) + y0
            detections.append({
                "text": text, "lead": lead, "cx": cx, "cy": cy,
                "conf": float(conf), "band_idx": b_idx,
            })
    return detections


# =====================================================================
# Etapa 2: Band detection (canal 2)
# =====================================================================

def detect_bands(binary_mask, sigma=10.0, distance=50, prominence_factor=0.1,
                 buffer_factor=1.2):
    h = binary_mask.shape[0]
    row_count = binary_mask.sum(axis=1).astype(float)
    if row_count.sum() == 0:
        return []
    smoothed = gaussian_filter1d(row_count, sigma=sigma)
    peaks, _ = find_peaks(smoothed, distance=distance,
                          prominence=smoothed.max() * prominence_factor)
    if peaks.size == 0:
        return []
    inner_valleys = []
    for i in range(len(peaks) - 1):
        seg = smoothed[peaks[i]:peaks[i + 1]]
        inner_valleys.append(int(peaks[i] + np.argmin(seg)))
    bands = []
    for i, peak in enumerate(peaks):
        if i == 0:
            y0 = max(0, peak - int((inner_valleys[0] - peak) * buffer_factor)) if inner_valleys else 0
        else:
            y0 = inner_valleys[i - 1]
        if i == len(peaks) - 1:
            y1 = min(h, peak + int((peak - inner_valleys[-1]) * buffer_factor)) if inner_valleys else h
        else:
            y1 = inner_valleys[i]
        bands.append((y0, y1))
    return bands


# =====================================================================
# Etapa 3: Calibracao e parsing de texto OCR completo
# =====================================================================

def parse_calibration(full_ocr_results: list[tuple]) -> dict:
    """Le texto OCR completo, extrai velocidade e ganhos. Por seguranca, sempre
    parsea texto + numero adjacente (mesma posicao Y, X proximo) — nao um regex
    pluripotente que pode pegar lixo distante."""
    cal = {"speed_mm_s": 25.0, "gain_limb_mm_mV": 10.0,
           "gain_chest_mm_mV": None, "format": "3x4+1"}
    # Reune em (cx, cy, text)
    items = []
    for bbox, text, _conf in full_ocr_results:
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        items.append((int(np.mean(xs)), int(np.mean(ys)), text.strip()))

    def find_numeric_neighbor(keyword: str) -> float | None:
        for i, (cx, cy, txt) in enumerate(items):
            if keyword.lower() not in txt.lower():
                continue
            # busca numero mais proximo na MESMA linha (cy +-15 px) e a direita
            best = None; best_dx = float("inf")
            for j, (cx2, cy2, txt2) in enumerate(items):
                if j == i or abs(cy2 - cy) > 25:
                    continue
                if cx2 <= cx:  # a direita
                    continue
                # tenta extrair digitos
                m = re.search(r"\d+", txt2)
                if not m:
                    continue
                # ignora "100", "Hz" etc que estao muito longe (>500 px)
                dx = cx2 - cx
                if dx > 500:
                    continue
                if dx < best_dx:
                    best_dx = dx; best = float(m.group(0))
            return best
        return None

    v = find_numeric_neighbor("Veloc")
    if v is not None: cal["speed_mm_s"] = v
    v = find_numeric_neighbor("Membr")
    if v is not None: cal["gain_limb_mm_mV"] = v
    v = find_numeric_neighbor("Torax") or find_numeric_neighbor("Tórax")
    if v is not None: cal["gain_chest_mm_mV"] = v
    # Se chest nao detectado, assume igual a limb (default clinico)
    if cal["gain_chest_mm_mV"] is None:
        cal["gain_chest_mm_mV"] = cal["gain_limb_mm_mV"]
    text_combined = " ".join(t for _, _, t in items)
    if re.search(r"12\s+deriv", text_combined, re.IGNORECASE):
        cal["format"] = "3x4+1"
    return cal


# =====================================================================
# Etapa 4: Build leads structure
# =====================================================================

def find_trace_bounds_per_band(binary: np.ndarray, y0: int, y1: int,
                                px_per_mm: float = 14.0,
                                cal_pulse_max_mm: float = 12.0,
                                sustained_window: int = 50,
                                sustained_min_cols: int = 25,
                                col_min_pixels: int = 3) -> tuple[int, int]:
    """Acha (x_start, x_end) do tracado REAL na banda, pulando pulso de cal.
    - col_min_pixels: minimo de pixels por coluna pra contar como 'tem traco'
    - sustained: trace real = >= sustained_min_cols cols ativas em janela sustained_window
    - cal_pulse_max_mm: regiao no inicio com largura ate 12mm e' considerada
      candidata a pulso de calibracao e pulada"""
    band = binary[y0:y1, :]
    col_sums = band.sum(axis=0)
    W = band.shape[1]
    active = col_sums >= col_min_pixels

    # Rolling count de cols ativas
    kernel = np.ones(sustained_window, dtype=np.int32)
    active_int = active.astype(np.int32)
    rolling = np.convolve(active_int, kernel, mode="same")
    sustained = rolling >= sustained_min_cols

    if not sustained.any():
        return 0, W
    # x_start = primeira posicao sustained, pulando pulso de calibracao
    sustained_idx = np.where(sustained)[0]
    cal_pulse_max_px = int(cal_pulse_max_mm * px_per_mm)
    # Se primeiro sustained estiver dentro dos primeiros cal_pulse_max_px,
    # pular: pula esse bloco e busca proximo gap + sustained
    candidate_start = int(sustained_idx[0])
    if candidate_start < cal_pulse_max_px:
        # encontra fim desse bloco sustained
        for i in range(candidate_start, W):
            if not sustained[i]:
                # achou gap apos pulso; busca proximo sustained
                gap_end = i
                for j in range(gap_end, W):
                    if sustained[j]:
                        candidate_start = j
                        break
                break
    x_start = candidate_start
    # x_end = ultima posicao sustained
    x_end = int(sustained_idx[-1]) + sustained_window // 2
    x_end = min(W, x_end)
    return x_start, x_end


def build_leads_structure(bands: list[tuple], W: int, mask: np.ndarray,
                          ocr_dets: list[dict],
                          px_per_mm: float = 14.0) -> list[dict]:
    """Monta lista de derivacoes. Detecta x_start/x_end REAL por banda
    pulando pulso de calibracao, e usa max(starts) / min(ends) das 3 bandas
    de leads pra ter range consistente."""
    leads = []
    binary = mask < 128

    # Bounds por banda (so as 3 lead rows; rhythm e' separado)
    band_bounds = []
    for i, (y0, y1) in enumerate(bands[:3]):
        x_start, x_end = find_trace_bounds_per_band(binary, y0, y1, px_per_mm)
        band_bounds.append((x_start, x_end))
        print(f"  band {i}: x_start_real={x_start} x_end_real={x_end}")

    # Range consistente: max dos starts, min dos ends
    if band_bounds:
        x_start_common = max(b[0] for b in band_bounds)
        x_end_common = min(b[1] for b in band_bounds)
    else:
        x_start_common, x_end_common = 0, W
    print(f"  RANGE COMUM: x[{x_start_common}..{x_end_common}] "
          f"({x_end_common - x_start_common} px)")

    chunk_w = (x_end_common - x_start_common) // 4

    # Agrupa OCR por banda
    ocr_by_band: dict[int, list[dict]] = {}
    for d in ocr_dets:
        ocr_by_band.setdefault(d["band_idx"], []).append(d)

    for row_idx, row_leads in enumerate(LEAD_LAYOUT_3x4):
        if row_idx >= len(bands):
            break
        y0, y1 = bands[row_idx]
        y_center = (y0 + y1) // 2
        ocr_for_row = sorted(ocr_by_band.get(row_idx, []), key=lambda d: d["cx"])
        ocr_for_row = [d for d in ocr_for_row if d["lead"] in row_leads]

        for col_idx, lead in enumerate(row_leads):
            x_start = x_start_common + col_idx * chunk_w
            x_end = (x_start_common + (col_idx + 1) * chunk_w
                     if col_idx < 3 else x_end_common)
            matched = [d for d in ocr_for_row if d["lead"] == lead]
            if matched:
                cx_label = matched[0]["cx"]
                idx_in_ocr = ocr_for_row.index(matched[0])
                if idx_in_ocr + 1 < len(ocr_for_row):
                    x_end = ocr_for_row[idx_in_ocr + 1]["cx"]
                else:
                    x_end = x_end_common
                x_start = max(cx_label, x_start_common)
            leads.append({
                "name": lead, "row": row_idx,
                "x_start": int(x_start), "x_end": int(x_end),
                "y_center": int(y_center), "y0": int(y0), "y1": int(y1),
                "anchor": "ocr" if matched else "split_real",
            })

    # Rhythm: bounds independentes (toda largura ate fim do tracado)
    if len(bands) >= 4:
        y0, y1 = bands[3]
        x_rs, x_re = find_trace_bounds_per_band(binary, y0, y1, px_per_mm)
        leads.append({
            "name": RHYTHM_LABEL, "row": 3,
            "x_start": int(x_rs), "x_end": int(x_re),
            "y_center": (y0 + y1) // 2, "y0": int(y0), "y1": int(y1),
            "anchor": "rhythm",
        })

    return leads


# =====================================================================
# Etapa 5: Cropping e Viterbi
# =====================================================================

def crop_lead_mask(mask: np.ndarray, lead: dict) -> np.ndarray:
    """Recorta mascara pra uma derivacao com margem vertical."""
    band_h = lead["y1"] - lead["y0"]
    margin = int(band_h * VERTICAL_MARGIN_FRAC)
    y_top = max(0, lead["y_center"] - margin)
    y_bot = min(mask.shape[0], lead["y_center"] + margin)
    return mask[y_top:y_bot, lead["x_start"]:lead["x_end"]]


def visualize_bounding_boxes(mask: np.ndarray, leads: list[dict],
                              out_path: Path) -> None:
    """Sobrepoe bboxes coloridos na mascara pra inspecao visual."""
    img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colors = [(255, 80, 80), (80, 200, 80), (80, 80, 255), (200, 200, 0),
              (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
              (0, 128, 128), (200, 100, 100), (100, 200, 100), (100, 100, 200),
              (50, 50, 50)]
    for i, lead in enumerate(leads):
        c = colors[i % len(colors)]
        band_h = lead["y1"] - lead["y0"]
        margin = int(band_h * VERTICAL_MARGIN_FRAC)
        y_top = max(0, lead["y_center"] - margin)
        y_bot = min(mask.shape[0], lead["y_center"] + margin)
        cv2.rectangle(img, (lead["x_start"], y_top),
                      (lead["x_end"], y_bot), c, 2)
        cv2.putText(img, f"{lead['name']} [{lead['anchor']}]",
                    (lead["x_start"] + 6, y_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


# =====================================================================
# Etapa 6: Output
# =====================================================================

def plot_12_leads_grid(signals: dict, leads: list[dict], cal: dict,
                       px_per_mm: float, out_path: Path) -> None:
    """Plot 3x4+1 padrao com escala em mV e segundos."""
    fig = plt.figure(figsize=(16, 11), dpi=110)
    fig.suptitle(
        f"ECG digitalizado — IMG_1310 | "
        f"vel={cal['speed_mm_s']:.0f} mm/s, "
        f"gain={cal['gain_limb_mm_mV']:.0f} mm/mV | "
        f"px/mm={px_per_mm:.2f}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    # 3 rows x 4 cols + rhythm
    for row_idx, row_leads in enumerate(LEAD_LAYOUT_3x4):
        for col_idx, lead_name in enumerate(row_leads):
            ax = plt.subplot2grid((4, 4), (row_idx, col_idx))
            sig_px = signals.get(lead_name)
            if sig_px is None or len(sig_px) == 0:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(lead_name, fontsize=10)
                continue
            # Convert
            mv = sig_px / (cal["gain_limb_mm_mV"] * px_per_mm)
            t = np.arange(len(sig_px)) / (cal["speed_mm_s"] * px_per_mm)
            ax.plot(t, mv, color="#1f78b4", linewidth=0.7)
            ax.axhline(0, color="#aaa", linewidth=0.3, linestyle="--")
            ax.set_title(lead_name, fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3, linewidth=0.4)
            ax.set_xlabel("t (s)", fontsize=8)
            ax.set_ylabel("mV", fontsize=8)
            ax.tick_params(labelsize=7)
    # Rhythm
    ax = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    sig = signals.get(RHYTHM_LABEL)
    if sig is not None:
        mv = sig / (cal["gain_limb_mm_mV"] * px_per_mm)
        t = np.arange(len(sig)) / (cal["speed_mm_s"] * px_per_mm)
        ax.plot(t, mv, color="#1f78b4", linewidth=0.6)
        ax.axhline(0, color="#aaa", linewidth=0.3, linestyle="--")
        ax.set_title("II (rhythm strip)", fontsize=10, fontweight="bold")
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("mV", fontsize=8)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def plot_comparison(mask: np.ndarray, signals_px: dict, leads: list[dict],
                     out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 10), dpi=100)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(mask, cmap="gray", aspect="auto")
    for lead in leads:
        band_h = lead["y1"] - lead["y0"]
        margin = int(band_h * VERTICAL_MARGIN_FRAC)
        y_top = max(0, lead["y_center"] - margin)
        y_bot = min(mask.shape[0], lead["y_center"] + margin)
        ax1.add_patch(plt.Rectangle(
            (lead["x_start"], y_top), lead["x_end"] - lead["x_start"],
            y_bot - y_top, fill=False, edgecolor="red", linewidth=1, alpha=0.6))
        ax1.text(lead["x_start"] + 5, y_top + 15, lead["name"],
                 color="red", fontsize=8, fontweight="bold")
    ax1.set_title("Mascara canal 2 + bboxes das derivacoes", fontsize=12)
    ax1.axis("off")

    # Painel inferior: sinais sobrepostos com offset
    ax2 = fig.add_subplot(2, 1, 2)
    offset = 0
    for lead in leads:
        sig = signals_px.get(lead["name"])
        if sig is None:
            continue
        x = lead["x_start"] + np.arange(len(sig))
        ax2.plot(x, sig + offset, linewidth=0.5)
        ax2.text(lead["x_start"] - 5, offset, lead["name"], fontsize=8,
                 ha="right", va="center")
        offset += 50
    ax2.set_title("Sinais extraidos (com offset visual entre derivacoes)",
                  fontsize=12)
    ax2.set_xlabel("coluna (x px)")
    ax2.set_ylabel("Y (px) + offset")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=100, facecolor="white")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    print("=" * 70)
    print("SEPARACAO DE DERIVACOES — IMG_1310")
    print("=" * 70)

    # Load
    norm = cv2.imread(str(NORMALIZED_PATH))
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if norm is None or mask is None:
        print("ERRO: nao consegui ler imagens")
        return 1
    H, W = mask.shape
    print(f"Normalized: {norm.shape[1]}x{norm.shape[0]} | "
          f"Mask: {W}x{H}")

    # px/mm de calibracao do Stenhede (do log do pipeline)
    PX_PER_MM = 14.0  # do PixelSizeFinder, valor avg
    print(f"px/mm (do Stenhede): {PX_PER_MM}")

    # ----- Etapa 1: OCR full image pra calibracao + formato -----
    print("\n--- OCR para calibracao ---")
    import easyocr  # noqa
    reader = easyocr.Reader(["en"], gpu=False)
    t0 = time.perf_counter()
    full_ocr = reader.readtext(norm, paragraph=False)
    print(f"OCR full image: {len(full_ocr)} deteccoes em {time.perf_counter()-t0:.1f}s")
    cal = parse_calibration(full_ocr)
    print(f"Calibracao: {cal}")

    # ----- Etapa 2: Band detection -----
    print("\n--- Band detection ---")
    binary = mask < 128
    bands = detect_bands(binary)
    print(f"Bandas detectadas: {len(bands)}")
    for i, (y0, y1) in enumerate(bands):
        print(f"  banda {i}: y[{y0}:{y1}] altura={y1-y0}")
    if len(bands) < 4:
        print(f"AVISO: esperava 4 bandas, achei {len(bands)}")

    # ----- Etapa 3: OCR focado em strips de label (topo de cada banda) -----
    print("\n--- OCR por strip de banda ---")
    band_tops = [b[0] for b in bands[:3]]  # lead rows only (not rhythm)
    ocr_dets = run_ocr_on_strips(reader, norm, band_tops,
                                 strip_height=70, upscale=2)
    print(f"Labels de derivacao encontrados: {len(ocr_dets)}")
    for d in ocr_dets:
        print(f"  band {d['band_idx']} | {d['lead']:4s} @ "
              f"({d['cx']:4d},{d['cy']:4d}) conf={d['conf']:.2f} "
              f"text={d['text']!r}")

    # ----- Etapa 4: Build leads -----
    print("\n--- Build leads structure (com deteccao de bounds reais) ---")
    leads = build_leads_structure(bands, W, mask, ocr_dets, px_per_mm=PX_PER_MM)
    for L in leads:
        print(f"  {L['name']:11s} row={L['row']} "
              f"x[{L['x_start']:4d}..{L['x_end']:4d}] "
              f"y_center={L['y_center']} [{L['anchor']}]")

    # ----- Etapa 5: Crop + visualization -----
    visualize_bounding_boxes(mask, leads, OUTPUT_DIR / "01_bboxes_overlay.png")

    # ----- Etapa 6: Viterbi por lead -----
    print("\n--- Viterbi por derivacao ---")
    signals_px: dict[str, np.ndarray] = {}
    signals_mv: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}
    for lead in leads:
        crop = crop_lead_mask(mask, lead)
        sig_px = extrair_sinal_viterbi(crop, invert=True)
        signals_px[lead["name"]] = sig_px
        # Convert pra mV
        gain = (cal["gain_chest_mm_mV"] if lead["name"].startswith("V")
                else cal["gain_limb_mm_mV"])
        sig_mv = sig_px / (gain * PX_PER_MM)
        signals_mv[lead["name"]] = sig_mv
        binary_crop = crop < 128
        cols_with = int(np.any(binary_crop, axis=0).sum())
        stats[lead["name"]] = {
            "n": len(sig_px), "valid": int((~np.isnan(sig_px)).sum()),
            "nan": int(np.isnan(sig_px).sum()),
            "gaps_raw": crop.shape[1] - cols_with,
            "min_mv": float(np.nanmin(sig_mv)),
            "max_mv": float(np.nanmax(sig_mv)),
            "duration_s": len(sig_px) / (cal["speed_mm_s"] * PX_PER_MM),
            "gain_mm_mV": gain,
        }
        print(f"  {lead['name']:11s}: n={len(sig_px):4d} "
              f"valid={stats[lead['name']]['valid']:4d} "
              f"NaN={stats[lead['name']]['nan']:3d} "
              f"gaps_raw={stats[lead['name']]['gaps_raw']:4d} "
              f"range=[{stats[lead['name']]['min_mv']:+.2f},"
              f"{stats[lead['name']]['max_mv']:+.2f}] mV "
              f"dur={stats[lead['name']]['duration_s']:.2f}s")

    # ----- Etapa 7: Output -----
    print("\n--- Output ---")
    # .npy individual (px)
    for name, sig in signals_px.items():
        np.save(OUTPUT_DIR / f"sig_px_{name}.npy", sig)
    # .npy individual (mV)
    for name, sig in signals_mv.items():
        np.save(OUTPUT_DIR / f"sig_mv_{name}.npy", sig)

    # Matriz (12, N) com 12 leads no formato CNN — usa duracao do menor
    n_min = min(len(signals_mv[L]) for L in ALL_LEADS if L in signals_mv)
    matrix = np.stack([signals_mv[L][:n_min] for L in ALL_LEADS])
    np.save(OUTPUT_DIR / "ecg_12_leads_mv.npy", matrix)

    # CSV
    csv_lines = ["t_s," + ",".join(ALL_LEADS) + "," + RHYTHM_LABEL]
    n_rhythm = len(signals_mv.get(RHYTHM_LABEL, np.array([])))
    n_csv = max(n_min, n_rhythm)
    for i in range(n_csv):
        t_s = i / (cal["speed_mm_s"] * PX_PER_MM)
        vals = []
        for L in ALL_LEADS:
            s = signals_mv.get(L)
            v = s[i] if s is not None and i < len(s) else np.nan
            vals.append(f"{v:.4f}" if not np.isnan(v) else "")
        r = signals_mv.get(RHYTHM_LABEL)
        rv = r[i] if r is not None and i < len(r) else np.nan
        vals.append(f"{rv:.4f}" if not np.isnan(rv) else "")
        csv_lines.append(f"{t_s:.4f}," + ",".join(vals))
    (OUTPUT_DIR / "ecg_signals.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # Plots
    plot_12_leads_grid(signals_px, leads, cal, PX_PER_MM,
                       OUTPUT_DIR / "02_ecg_digitalizado.png")
    plot_comparison(mask, signals_px, leads,
                    OUTPUT_DIR / "03_comparacao_mascara_vs_sinal.png")

    # Sumario txt
    sum_lines = [
        "Separacao de derivacoes IMG_1310",
        "=" * 60,
        f"Formato: {cal['format']}",
        f"Velocidade: {cal['speed_mm_s']} mm/s",
        f"Ganho membros: {cal['gain_limb_mm_mV']} mm/mV",
        f"Ganho precordiais: {cal['gain_chest_mm_mV']} mm/mV",
        f"px/mm: {PX_PER_MM}",
        f"OCR labels encontrados: {len(ocr_dets)}",
        "",
        "Derivacoes:",
    ]
    for L in leads:
        s = stats[L["name"]]
        sum_lines.append(
            f"  {L['name']:11s} row={L['row']} x[{L['x_start']}..{L['x_end']}] "
            f"y={L['y_center']} anchor={L['anchor']} | "
            f"n={s['n']} valid={s['valid']} gaps_raw={s['gaps_raw']} | "
            f"range [{s['min_mv']:+.2f},{s['max_mv']:+.2f}] mV | "
            f"dur={s['duration_s']:.2f}s"
        )
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(sum_lines), encoding="utf-8")
    print("\n".join(sum_lines))
    print(f"\nOutputs em: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
