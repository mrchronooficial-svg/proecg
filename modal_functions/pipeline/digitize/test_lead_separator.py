"""
Teste do Lead Separator
=======================

Para cada máscara de Leader em "Leader Masks Teste":
  1. Carrega a imagem normalizada correspondente
  2. Roda o pipeline (preprocess → dotter → gridder → calibrator) só pra
     obter o calibration dict (px_per_mm, uv_per_pixel, etc.)
  3. Roda o lead_separator com a máscara anotada
  4. Salva uma visualização: imagem com cada derivação pintada + plot dos sinais

Output em: Leader Masks Teste/_visualizations/

Uso:
    python -m modal_functions.pipeline.digitize.test_lead_separator
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # backend headless
import matplotlib.pyplot as plt
import numpy as np

from .calibrator import calibrate
from .ecg_digitizer import ECGDigitizer
from .lead_separator import LAYOUT_3x4, separate_and_extract

NORMALIZED_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Normalizados Leader")
MASKS_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\Leader Masks Teste")
OUT_DIR = MASKS_DIR / "_visualizations"


# Cores distintas pra 12 derivações + DII longo (BGR pra OpenCV)
LEAD_COLORS = {
    "I":         (255, 102, 102),
    "aVR":       (102, 178, 255),
    "V1":        (102, 255, 102),
    "V4":        (255, 178, 102),
    "II":        (178, 102, 255),
    "aVL":       (102, 255, 255),
    "V2":        (255, 102, 178),
    "V5":        (178, 255, 102),
    "III":       (255, 255, 102),
    "aVF":       (102, 102, 255),
    "V3":        (102, 255, 178),
    "V6":        (255, 102, 255),
    "II_rhythm": (200, 200, 200),
}


def _calibrate_image(img: np.ndarray) -> dict:
    """Roda preprocess → dotter → gridder → calibrate na imagem normalizada
    (que já é o output do undistort no pipeline real)."""
    digitizer = ECGDigitizer(use_mock=False)

    # Já normalizada — pula preprocess, mas o digitizer espera imagem crua...
    # Como já está normalizada, executamos dotter direto e depois calibrate.
    grid_mask, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        grid_mask, keypoints = digitizer.dotter_mock(img)

    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _draw_lead_overlay(
    image: np.ndarray, mask: np.ndarray, separation: dict
) -> np.ndarray:
    """Pinta cada derivação com sua cor (zona de ownership baseline-midpoint).

    Cada lead "possui" pixels [upper_limit_y, lower_limit_y] em sua coluna,
    onde os limites são midpoints entre baselines vizinhas. QRS altos que
    invadem fora da banda de pico ficam coloridos com a cor da derivação dona.
    """
    out = image.copy().astype(np.uint8)
    h, w = out.shape[:2]

    # Pinta os pixels da máscara dentro do bbox de ownership de cada lead
    # (faixa retangular limitada por midpoints entre baselines).
    for name, info in separation["leads"].items():
        color = LEAD_COLORS.get(name, (200, 200, 200))
        x1, y1, x2, y2 = info["bbox"]
        x1 = max(0, x1); x2 = min(w, x2)
        y1 = max(0, y1); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        sub_mask = mask[y1:y2, x1:x2] > 0
        sub_out = out[y1:y2, x1:x2]
        sub_out[sub_mask] = color

    # Linhas horizontais nos midpoints (= fronteiras de ownership)
    seen_y: set[int] = set()
    for info in separation["leads"].values():
        for y in (info["bbox"][1], info["bbox"][3]):
            if y in seen_y:
                continue
            seen_y.add(y)
            if 0 < y < h - 1:
                cv2.line(out, (0, y), (w, y), (0, 0, 0), 1)

    # Linhas verticais (separadores das 4 colunas) — só onde há banda
    column_bboxes = separation.get("column_bboxes", [])
    band_y_min = h
    band_y_max = 0
    for band in separation.get("lead_bands", []):
        band_y_min = min(band_y_min, band[0])
        band_y_max = max(band_y_max, band[1])
    if separation.get("rhythm_band"):
        band_y_max = max(band_y_max, separation["rhythm_band"][1])
    for x1_col, x2_col in column_bboxes:
        cv2.line(out, (x1_col, band_y_min), (x1_col, band_y_max), (40, 40, 40), 1)
    if column_bboxes:
        cv2.line(
            out, (column_bboxes[-1][1], band_y_min),
            (column_bboxes[-1][1], band_y_max), (40, 40, 40), 1,
        )

    # Baseline tracejada por banda (linha horizontal cinza)
    for name, by in separation.get("baselines", {}).items():
        info = separation["leads"][name]
        x1, y1, x2, y2 = info["bbox"]
        for x in range(x1, x2, 8):
            cv2.line(out, (x, int(by)), (min(x + 4, x2), int(by)), (80, 80, 80), 1)

    # Labels nos cantos
    for name, info in separation["leads"].items():
        color = LEAD_COLORS.get(name, (200, 200, 200))
        x1, y1, x2, y2 = info["bbox"]
        cv2.putText(
            out, name, (x1 + 4, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    return out


def _plot_signals(separation: dict, calibration: dict, ax) -> None:
    """Plota todos os sinais empilhados (estilo Holter)."""
    leads = separation["leads"]
    if not leads:
        ax.text(0.5, 0.5, "Sem sinais", ha="center", va="center")
        return

    # Ordem de exibição
    order = []
    for row in LAYOUT_3x4:
        for name in row:
            if name in leads:
                order.append(name)
    if "II_rhythm" in leads:
        order.append("II_rhythm")

    sr = float(calibration["sampling_rate_hz"])

    # Espaçamento vertical entre traçados (em µV)
    offset_uv = 1500.0
    for i, name in enumerate(order):
        sig = leads[name]["signal_uv"]
        t = np.arange(len(sig)) / sr if sr > 0 else np.arange(len(sig))
        y_offset = -i * offset_uv
        color_bgr = LEAD_COLORS.get(name, (128, 128, 128))
        # BGR → RGB normalizado
        c = (color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0)
        ax.plot(t, sig + y_offset, color=c, linewidth=0.8)
        ax.text(
            -0.3, y_offset, name, fontsize=9, va="center", ha="right",
            color=c, fontweight="bold",
        )

    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("amplitude (uV)")
    ax.set_title(f"Sinais extraidos ({separation['layout']}, {len(order)} derivacoes)")
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])


def _process_one(
    image_path: Path, mask_path: Path, out_path: Path
) -> dict | None:
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None
    if img.shape[:2] != mask.shape[:2]:
        # Redimensiona máscara pra casar
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # 1. Calibração (do pipeline)
    try:
        cal = _calibrate_image(img)
    except Exception as e:
        print(f"  ERRO calibrando: {type(e).__name__}: {e}")
        return None

    # 2. Lead separator
    try:
        sep = separate_and_extract(mask, img, cal)
    except Exception as e:
        print(f"  ERRO separando: {type(e).__name__}: {e}")
        return None

    # 3. Visualização
    overlay = _draw_lead_overlay(img, mask, sep)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_img.set_title(
        f"{image_path.name}  -  {sep['layout']}  -  "
        f"{len(sep['leads'])} derivacoes  -  px/mm={cal['px_per_mm']:.2f}"
    )
    ax_img.axis("off")

    ax_sig = fig.add_subplot(gs[0, 1])
    _plot_signals(sep, cal, ax_sig)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": image_path.name,
        "layout": sep["layout"],
        "n_leads": len(sep["leads"]),
        "px_per_mm": cal["px_per_mm"],
        "sampling_rate_hz": cal["sampling_rate_hz"],
        "duration_s": (
            next(iter(sep["leads"].values()))["duration_s"]
            if sep["leads"] else 0.0
        ),
        "warnings": sep["warnings"],
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")

    if not MASKS_DIR.exists():
        print(f"ERRO: {MASKS_DIR} nao existe", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Foco nas imagens pedidas pra validar fronteiras + baseline filter + smoothing
    target_names = {"IMG_1275.png", "IMG_1279.png", "IMG_1303.png"}
    masks = [
        p for p in sorted(MASKS_DIR.iterdir())
        if p.is_file() and p.suffix.lower() == ".png" and p.name in target_names
    ]
    if not masks:
        print(f"ERRO: nenhuma das mascaras alvo em {MASKS_DIR}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 78}")
    print(f"Mascaras: {len(masks)}  |  Output: {OUT_DIR}")
    print(f"{'=' * 78}\n")

    summary = []
    for mask_path in masks:
        image_path = NORMALIZED_DIR / mask_path.name
        if not image_path.exists():
            print(f"  SKIP {mask_path.name}: imagem normalizada nao encontrada")
            continue

        out_path = OUT_DIR / f"{mask_path.stem}_viz.png"
        print(f"  -> {mask_path.name}")
        result = _process_one(image_path, mask_path, out_path)
        if result is not None:
            summary.append(result)
            warns = ", ".join(result["warnings"]) if result["warnings"] else "—"
            print(
                f"     layout={result['layout']}  leads={result['n_leads']}  "
                f"px/mm={result['px_per_mm']:.2f}  "
                f"sr={result['sampling_rate_hz']:.0f}Hz  dur={result['duration_s']:.1f}s"
            )
            if result["warnings"]:
                for w in result["warnings"]:
                    print(f"     warn: {w}")

    print(f"\n{'=' * 78}")
    print(f"RESUMO  |  {len(summary)} processadas")
    print(f"{'=' * 78}")
    print(f"  {'Imagem':50s}  {'Layout':>7s}  {'Leads':>5s}  {'px/mm':>6s}  {'Dur(s)':>7s}")
    print(f"  {'-' * 50}  {'-' * 7}  {'-' * 5}  {'-' * 6}  {'-' * 7}")
    for r in summary:
        name = r["image"][:50]
        print(
            f"  {name:50s}  {r['layout']:>7s}  {r['n_leads']:>5d}  "
            f"{r['px_per_mm']:>6.2f}  {r['duration_s']:>7.2f}"
        )
    print(f"\nVisualizacoes salvas em: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
