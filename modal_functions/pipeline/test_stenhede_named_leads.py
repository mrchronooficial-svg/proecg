"""
Visualização do reconhecimento das derivações pelo LeadIdentifier
do Stenhede.

Roda em IMG_1405, IMG_1303 e IMG_1387 e renderiza overlay com cada
derivação pintada de uma cor diferente, com rótulo no início do chunk.

Saídas em modal_functions/pipeline/digitize/_visualizations/:
  IMG_1405_named_leads.png
  IMG_1303_named_leads.png
  IMG_1387_named_leads.png
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.stenhede_adapter import (
    LEAD_CHANNEL_ORDER,
    extract_signals_stenhede,
    _VENDOR_LAYOUTS_YAML,
)

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
TARGETS = ["IMG_1405", "IMG_1303", "IMG_1387"]


# Cores fixas por lead — uma cor por derivação canônica
LEAD_COLORS: dict[str, str] = {
    "I":   "#e6194B",  # vermelho
    "II":  "#3cb44b",  # verde
    "III": "#ffe119",  # amarelo
    "aVR": "#4363d8",  # azul
    "aVL": "#f58231",  # laranja
    "aVF": "#911eb4",  # roxo
    "V1":  "#42d4f4",  # ciano
    "V2":  "#f032e6",  # magenta
    "V3":  "#bfef45",  # lima
    "V4":  "#fabed4",  # rosa
    "V5":  "#469990",  # teal
    "V6":  "#9A6324",  # marrom
    "RHY": "#000000",  # rhythm strip (preto)
    "X":   "#808080",  # placeholder (cinza)
}


def _load_layouts() -> dict:
    with open(_VENDOR_LAYOUTS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_LAYOUTS = _load_layouts()


def _normalize_leads_def(layout_def: dict) -> tuple[list[list[str]], int, int, list[str]]:
    """Devolve (matriz NxM de nomes de lead, rows, cols, rhythm_leads).
    Suporta formato linha-por-linha (lista de listas) e linha-única (lista
    plana). Strip de "-" para sinais negativos (ex: "-aVR" -> "aVR").
    """
    rows = int(layout_def["layout"]["rows"])
    cols = int(layout_def["layout"]["cols"])
    leads = layout_def["leads"]
    rhythm_leads = list(layout_def.get("rhythm_leads", []))

    if isinstance(leads[0], list):
        matrix = [[str(name).lstrip("-") for name in row] for row in leads]
    elif len(leads) == rows * cols:
        matrix = []
        for r in range(rows):
            matrix.append(
                [str(leads[r * cols + c]).lstrip("-") for c in range(cols)]
            )
    else:
        matrix = [[str(leads[r]).lstrip("-")] for r in range(rows)]
    return matrix, rows, cols, rhythm_leads


def _render_named_leads(
    image_bgr: np.ndarray,
    raw_lines_padded: np.ndarray,
    layout_name: str,
    out_path: Path,
    title: str,
    blend_white: float = 0.45,
) -> None:
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = (1 - blend_white) * img_rgb + blend_white
    img_blend = np.clip(img_blend, 0, 1)
    fig_w = max(20.0, w / 200.0); fig_h = fig_w * (h / w) + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])

    layout_def = _LAYOUTS.get(layout_name)
    if layout_def is None:
        logging.warning("Layout %s não encontrado em layouts_all.yml", layout_name)
        ax.set_title(f"{title}\n[layout {layout_name} desconhecido]",
                     fontsize=12, fontweight="bold", color="red")
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    matrix, rows, cols, rhythm_leads = _normalize_leads_def(layout_def)
    n_lines = raw_lines_padded.shape[0] if raw_lines_padded.ndim == 2 else 0
    if n_lines == 0:
        ax.set_title(f"{title}\n[0 linhas extraídas]",
                     fontsize=12, fontweight="bold", color="red")
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    # Largura útil = largura da imagem
    chunk_w = w / cols
    leads_used: list[str] = []

    # Para cada row do layout, divide em N=cols chunks; cada chunk pinta
    # com a cor do lead daquela posição
    for row_idx in range(min(rows, n_lines)):
        line = raw_lines_padded[row_idx]
        for col_idx in range(cols):
            if row_idx >= len(matrix) or col_idx >= len(matrix[row_idx]):
                continue
            name = matrix[row_idx][col_idx]
            color = LEAD_COLORS.get(name, "#999999")
            x_start = int(round(col_idx * chunk_w))
            x_end = int(round((col_idx + 1) * chunk_w)) if col_idx < cols - 1 else w

            xs = np.arange(x_start, x_end)
            ys = line[x_start:x_end]
            ax.plot(xs, ys, color=color, lw=1.6, alpha=0.9, zorder=3)

            # Label do lead no canto superior-esquerdo do chunk
            valid = ~np.isnan(ys)
            if valid.any():
                first_x = xs[np.argmax(valid)]
                # Y fica acima do traçado, com pequeno offset
                first_y = ys[np.argmax(valid)]
                label_y = max(20, first_y - 35)
                ax.text(
                    first_x + 8, label_y, name,
                    fontsize=12, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.9),
                    zorder=5,
                )
            if name not in leads_used:
                leads_used.append(name)

    # Rhythm strip — última linha do raw_lines (full width)
    n_rhythm = len(rhythm_leads)
    rhythm_color = LEAD_COLORS["RHY"]
    if n_rhythm > 0 and n_lines > rows:
        # Plot full-width rhythm em preto
        line = raw_lines_padded[rows]  # primeira linha de rhythm
        xs = np.arange(w)
        ax.plot(xs, line, color=rhythm_color, lw=1.6, alpha=0.9, zorder=3)
        valid = ~np.isnan(line)
        if valid.any():
            first_x = int(np.argmax(valid))
            first_y = line[first_x]
            label_y = max(20, first_y - 35)
            ax.text(
                first_x + 8, label_y, "Rhythm strip",
                fontsize=12, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=rhythm_color,
                          ec="black", alpha=0.95),
                zorder=5,
            )
        leads_used.append("Rhythm")

    # Legenda
    legend_handles = []
    for name in matrix[0] if matrix else []:
        legend_handles.append(
            mpatches.Patch(color=LEAD_COLORS.get(name, "#999"), label=name)
        )
    # Adiciona resto do layout
    seen = set([h.get_label() for h in legend_handles])
    for row in matrix:
        for name in row:
            if name not in seen:
                legend_handles.append(
                    mpatches.Patch(color=LEAD_COLORS.get(name, "#999"), label=name)
                )
                seen.add(name)
    if n_rhythm > 0 and n_lines > rows:
        legend_handles.append(
            mpatches.Patch(color=rhythm_color, label="Rhythm strip")
        )
    ax.legend(handles=legend_handles, loc="upper right", ncol=4,
              fontsize=10, framealpha=0.95)

    ax.set_title(title, fontsize=13, fontweight="bold")
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
    print(f"  {stem}.png  shape=(H={h}, W={w})")
    print("=" * 78)

    t0 = time.perf_counter()
    try:
        result = extract_signals_stenhede(
            image_bgr=img, use_cropper=False, use_internal_pixel_size=True,
        )
    except Exception as e:
        print(f"[ERRO] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 3
    dt = time.perf_counter() - t0

    layout = result["match"].get("layout", "?")
    cost = result["match"].get("cost", float("nan"))
    n_lines = result["n_lines_detected"]
    pxmm = result["avg_pixel_per_mm"]
    print(
        f"  layout={layout} cost={cost:.2f} linhas={n_lines} "
        f"px/mm={pxmm:.2f} ({dt:.1f}s)"
    )

    raw = result["raw_lines_pixel"]
    print(f"  raw_lines_pixel.shape={raw.shape}")

    out_path = OUT_DIR / f"{stem}_named_leads.png"
    title = (
        f"{stem} — Reconhecimento Stenhede LeadIdentifier "
        f"(layout={layout}, cost={cost:.2f})"
    )
    _render_named_leads(img, raw, layout, out_path, title)
    print(f"  [Render] {out_path.name}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(" Visualização: nomes das derivações detectadas pelo LeadIdentifier")
    print("=" * 78)
    print(f" Imagens: {TARGETS}")
    rc = 0
    for stem in TARGETS:
        rc = _process(stem) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
