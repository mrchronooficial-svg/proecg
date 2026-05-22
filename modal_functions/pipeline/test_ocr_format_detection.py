"""
Teste isolado: detecção de formato do ECG (3x4, 6x2, 12x1) por OCR
dos labels de derivação (I, II, III, aVR/L/F, V1-V6).

NÃO integra com o pipeline. Apenas testa a lógica.

Fluxo:
  1. Preprocess + Dotter + Undistort (ECGDigitizer) → imagem normalizada
  2. EasyOCR na imagem normalizada → bboxes + textos
  3. Filtra resultados que casam com nomes de derivação (com fuzzy matching
     pros misreads típicos: I↔1↔l, V1↔Vi, AVR↔avr↔aVR, etc.)
  4. Agrupa por Y (mesma "linha do ECG") e dentro de cada linha por X
     (colunas dentro da linha)
  5. Determina formato pela contagem: rows × cols + (rhythm extra)
  6. Salva overlay com círculos coloridos, linhas horizontais entre labels
     da mesma row, e header "FORMATO DETECTADO: ..."
  7. Printa relatório no terminal

Uso:
  python modal_functions/pipeline/test_ocr_format_detection.py [img1.jpg img2.jpg ...]

Sem args: roda na lista default (IMG_1461, IMG_1310, IMG_1400, IMG_1383
em ~/Desktop/Projeto ECG/ECGs Reais3/).
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))
sys.path.insert(0, str(MODAL_ROOT.parent))

from pipeline.digitize.ecg_digitizer import ECGDigitizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("test_ocr_format")

HOME = Path.home()
OUT_DIR = HOME / "Desktop" / "Projeto ECG" / "teste_formato"

DEFAULT_IMAGES = [
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1461.jpg",
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1310.jpg",
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1400.jpg",
    HOME / "Desktop" / "Projeto ECG" / "ECGs Reais3" / "IMG_1383.jpg",
]

CANONICAL_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Cores BGR por row do ECG
ROW_COLORS_BGR = [
    (80, 80, 255),    # 0 vermelho
    (80, 200, 80),    # 1 verde
    (255, 80, 80),    # 2 azul
    (0, 200, 220),    # 3 amarelo
    (200, 100, 200),  # 4 magenta
    (200, 200, 100),  # 5 ciano
    (180, 130, 30),   # 6 marrom
    (130, 0, 130),    # 7 roxo
    (0, 130, 130),    # 8 oliva
    (180, 180, 180),  # 9 cinza
    (50, 200, 200),   # 10
    (200, 50, 200),   # 11
]


# ---------------------------------------------------------------------------
# Normalização do texto OCR -> nome canônico
# ---------------------------------------------------------------------------

def normalize_lead_text(raw: str) -> str | None:
    """Tenta reconhecer o texto OCR como nome de derivação. Retorna o
    nome canônico (I, II, III, aVR, aVL, aVF, V1..V6) ou None.

    Trata misreads comuns:
      I, l, 1, |        -> I
      II, 11, ll, ii    -> II
      III, 111, lll     -> III
      AVR/avr/oVR/vR    -> aVR
      AVL/avl/oVL/vL    -> aVL
      AVF/avf/oVF/vF    -> aVF
      V1/Vi/v1/V|       -> V1
      V5/Vs             -> V5
      V6/Vg/Vb          -> V6
    """
    s = raw.strip().lower()
    # Mantém só alfanuméricos (joga fora pontuação, espaços)
    s = re.sub(r"[^a-z0-9]", "", s)
    if not s:
        return None

    # Match direto (já no formato canônico)
    direct_map = {
        "i": "I", "ii": "II", "iii": "III",
        "avr": "aVR", "avl": "aVL", "avf": "aVF",
        "v1": "V1", "v2": "V2", "v3": "V3",
        "v4": "V4", "v5": "V5", "v6": "V6",
    }
    if s in direct_map:
        return direct_map[s]

    # aVR/aVL/aVF — primeira letra pode ser 'a' ou 'o' (OCR confunde a com o),
    # ou pode faltar. Última letra é r/l/f.
    if len(s) in (2, 3):
        if len(s) == 3 and s[0] in "ao" and s[1] == "v" and s[2] in "rlf":
            return {"r": "aVR", "l": "aVL", "f": "aVF"}[s[2]]
        if len(s) == 2 and s[0] == "v" and s[1] in "rlf":
            return {"r": "aVR", "l": "aVL", "f": "aVF"}[s[1]]

    # V1-V6 com possível misread do dígito
    if len(s) == 2 and s[0] == "v":
        digit_map = {
            "1": "1", "i": "1", "l": "1", "!": "1", "|": "1",
            "2": "2", "z": "2",
            "3": "3",
            "4": "4", "a": "4",
            "5": "5", "s": "5",
            "6": "6", "g": "6", "b": "6",
        }
        if s[1] in digit_map:
            return f"V{digit_map[s[1]]}"

    # I, II, III — sequência de 1-3 chars do conjunto {i, l, 1, !, |}
    if len(s) <= 3 and re.fullmatch(r"[il1!|]+", s):
        return {1: "I", 2: "II", 3: "III"}[len(s)]

    return None


# ---------------------------------------------------------------------------
# Preprocess (mesma lógica de test_lead_identifier_bboxes.py)
# ---------------------------------------------------------------------------

def load_image_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is not None:
        return img
    import pillow_heif  # type: ignore
    from PIL import Image

    pillow_heif.register_heif_opener()
    pil = Image.open(str(path)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def normalize_image(img_path: Path) -> np.ndarray:
    img_bgr = load_image_any(img_path)
    h, w = img_bgr.shape[:2]
    if h > w * 1.2:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    digitizer = ECGDigitizer(use_mock=False)
    cropped, _ = digitizer.preprocess(img_bgr)
    _, keypoints = digitizer.dotter(cropped)
    if len(keypoints) == 0:
        _, keypoints = digitizer.dotter_mock(cropped)
    grid_matrix, grid_info = digitizer.gridder(keypoints, cropped.shape[:2])
    if len(keypoints) >= 100:
        return digitizer.undistort(
            cropped, grid_matrix, grid_info["px_per_mm"],
        )
    return cropped.copy()


# ---------------------------------------------------------------------------
# OCR + filtro de labels
# ---------------------------------------------------------------------------

def run_ocr(normalized_bgr: np.ndarray) -> list[tuple[list, str, float]]:
    import easyocr  # type: ignore
    reader = easyocr.Reader(["en"], gpu=False)
    return reader.readtext(normalized_bgr, paragraph=False)


def extract_lead_labels(
    ocr_results: list[tuple[list, str, float]],
    min_confidence: float = 0.3,
) -> list[dict]:
    """Filtra resultados OCR pra ficar só com labels de derivação.
    Retorna lista de dicts com {name, x, y, w, h, confidence, raw_text, bbox}.
    Em caso de duplicatas no mesmo lugar, mantém a com maior confidence.
    """
    candidates: list[dict] = []
    for bbox, text, conf in ocr_results:
        if conf < min_confidence:
            continue
        name = normalize_lead_text(text)
        if name is None:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_center = float(np.mean(xs))
        y_center = float(np.mean(ys))
        w = float(max(xs) - min(xs))
        h = float(max(ys) - min(ys))
        candidates.append({
            "name": name,
            "x": x_center, "y": y_center,
            "w": w, "h": h,
            "confidence": float(conf),
            "raw_text": text,
            "bbox": bbox,
        })

    # Dedup: 2 candidatos do MESMO nome dentro de ~50px = manter o melhor
    # (OCR pode ler "II" duas vezes na mesma região).
    deduped: list[dict] = []
    for c in sorted(candidates, key=lambda x: -x["confidence"]):
        dup = False
        for d in deduped:
            if d["name"] == c["name"]:
                if abs(d["x"] - c["x"]) < 60 and abs(d["y"] - c["y"]) < 60:
                    dup = True
                    break
        if not dup:
            deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Clustering espacial
# ---------------------------------------------------------------------------

def cluster_1d(values: list[float], tol: float) -> list[list[int]]:
    """Agrupa indices por proximidade em values (1D). Retorna lista de
    listas de indices, onde cada lista é um cluster.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = []
    current = [order[0]]
    for idx in order[1:]:
        if abs(values[idx] - values[current[-1]]) <= tol:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)
    return clusters


def group_into_rows_and_cols(
    labels: list[dict], image_h: int, image_w: int,
) -> tuple[list[list[dict]], list[int]]:
    """Agrupa labels em rows (Y) e dentro de cada row conta colunas (X).
    Retorna (rows_sorted_top_to_bottom, cols_per_row).
    """
    if not labels:
        return [], []

    # Tolerância Y: a altura típica de uma "linha" do ECG em 3x4 é ~H/4.
    # Usa 4% da altura como tolerância de agrupamento (~80px em imagens 2000px).
    tol_y = max(20.0, image_h * 0.04)
    tol_x = max(30.0, image_w * 0.04)

    ys = [lb["y"] for lb in labels]
    row_clusters = cluster_1d(ys, tol_y)

    rows: list[list[dict]] = []
    for cluster_idx_list in row_clusters:
        row = [labels[i] for i in cluster_idx_list]
        row.sort(key=lambda lb: lb["x"])
        rows.append(row)
    # Ordena rows top-to-bottom pelo Y médio
    rows.sort(key=lambda r: float(np.mean([lb["y"] for lb in r])))

    cols_per_row: list[int] = []
    for row in rows:
        xs = [lb["x"] for lb in row]
        col_clusters = cluster_1d(xs, tol_x)
        cols_per_row.append(len(col_clusters))
    return rows, cols_per_row


def detect_format(
    rows: list[list[dict]], cols_per_row: list[int],
) -> tuple[str, int, int, int]:
    """Decide o formato a partir dos clusters. Retorna (format_str,
    n_main_rows, n_cols, n_rhythm).
    """
    if not cols_per_row:
        return "unknown", 0, 0, 0

    most_common_cols, _ = Counter(cols_per_row).most_common(1)[0]
    n_main_rows = sum(1 for c in cols_per_row if c == most_common_cols)
    n_rhythm = len(cols_per_row) - n_main_rows
    fmt = f"{n_main_rows}x{most_common_cols}"
    if n_rhythm > 0:
        fmt += f"+{n_rhythm}"
    return fmt, n_main_rows, int(most_common_cols), n_rhythm


# ---------------------------------------------------------------------------
# Overlay visual
# ---------------------------------------------------------------------------

def draw_overlay(
    normalized_bgr: np.ndarray,
    labels_by_row: list[list[dict]],
    fmt: str,
    n_rows: int,
    n_cols: int,
    n_rhythm: int,
    all_labels: list[dict],
) -> np.ndarray:
    img = normalized_bgr.copy()
    H, W = img.shape[:2]

    # Faixa superior com o formato detectado
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, 80), (255, 255, 255), -1)
    img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
    title = (
        f"FORMATO DETECTADO: {fmt}    "
        f"({n_rows} rows x {n_cols} cols"
        + (f" + {n_rhythm} rhythm)" if n_rhythm > 0 else ")")
        + f"    | Labels: {len(all_labels)}/12"
    )
    cv2.putText(
        img, title, (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA,
    )

    # Linha horizontal por row + círculos coloridos + labels
    for row_idx, row in enumerate(labels_by_row):
        color = ROW_COLORS_BGR[row_idx % len(ROW_COLORS_BGR)]
        ys_row = [int(lb["y"]) for lb in row]
        y_med = int(np.median(ys_row))
        # Linha horizontal cobrindo o range X da row
        x_min = min(int(lb["x"]) for lb in row)
        x_max = max(int(lb["x"]) for lb in row)
        cv2.line(img, (x_min - 30, y_med), (x_max + 30, y_med), color, 1)
        for lb in row:
            cx, cy = int(lb["x"]), int(lb["y"])
            cv2.circle(img, (cx, cy), 18, color, 3)
            label_txt = f"{lb['name']} ({lb['confidence']:.2f})"
            cv2.putText(
                img, label_txt, (cx + 22, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

    return img


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def print_report(
    stem: str,
    labels: list[dict],
    rows: list[list[dict]],
    cols_per_row: list[int],
    fmt: str,
    n_rows: int,
    n_cols: int,
    n_rhythm: int,
) -> None:
    print(f"\n=== {stem} ===")
    found_set = sorted({lb["name"] for lb in labels})
    missing = sorted(set(CANONICAL_LEADS) - set(found_set))

    found_str = ", ".join(
        f"{lb['name']}({lb['confidence']:.2f})"
        for lb in sorted(labels, key=lambda x: (x["y"], x["x"]))
    )
    print(f"  Labels encontrados ({len(labels)}): {found_str}")
    if missing:
        print(f"  Faltando: {missing}")
    print("  Posições:")
    for i, (row, n_cols_row) in enumerate(zip(rows, cols_per_row), 1):
        y_med = int(np.median([lb["y"] for lb in row]))
        per_lead = ", ".join(
            f"{lb['name']}(x={int(lb['x'])})" for lb in row
        )
        print(
            f"    Linha {i} (y={y_med:>4}): {per_lead} -> {n_cols_row} cols"
        )
    rhythm_str = f" + {n_rhythm} rhythm" if n_rhythm > 0 else ""
    print(f"\n  Rows: {n_rows} | Cols: {n_cols}{rhythm_str}")
    print(f"  FORMATO: {fmt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_image(img_path: Path) -> None:
    print(f"\n>> Processando {img_path.name}...")
    t0 = time.perf_counter()
    normalized = normalize_image(img_path)
    H, W = normalized.shape[:2]
    logger.info("Normalized: %dx%d", W, H)

    t_ocr = time.perf_counter()
    ocr_results = run_ocr(normalized)
    logger.info(
        "OCR: %d detecções em %.1fs", len(ocr_results),
        time.perf_counter() - t_ocr,
    )

    labels = extract_lead_labels(ocr_results)
    logger.info("Labels de derivação reconhecidos: %d", len(labels))

    rows, cols_per_row = group_into_rows_and_cols(labels, H, W)
    fmt, n_main_rows, n_cols, n_rhythm = detect_format(rows, cols_per_row)

    print_report(
        img_path.stem, labels, rows, cols_per_row,
        fmt, n_main_rows, n_cols, n_rhythm,
    )

    overlay = draw_overlay(
        normalized, rows, fmt, n_main_rows, n_cols, n_rhythm, labels,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{img_path.stem}_ocr_format.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"  -> {out_path}  ({time.perf_counter() - t0:.1f}s)")


def main(argv: list[str]) -> int:
    if argv:
        images = [Path(a).expanduser() for a in argv]
    else:
        images = DEFAULT_IMAGES
    print(f"Saída: {OUT_DIR}")
    for img_path in images:
        if not img_path.is_file():
            print(f"[SKIP] não encontrado: {img_path}")
            continue
        try:
            process_image(img_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERRO: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
