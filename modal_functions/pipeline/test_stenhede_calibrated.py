"""
Teste do canonicalize calibrado em IMG_1387 e IMG_1303.

Saídas:
  IMG_1387_overlay_calibrated.png
  IMG_1303_overlay_calibrated.png

E imprime tabela comparativa: x_start antigo (W//cols) vs novo (calibrado)
vs ideal (centro do chunk visualmente correto).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.stenhede_adapter import (
    LEAD_CHANNEL_ORDER,
    _SECONDS_PER_CELL_BY_COLS,
    extract_signals_stenhede,
    _VENDOR_LAYOUTS_YAML,
)
import yaml

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
TARGETS = ["IMG_1387", "IMG_1303"]


LEAD_COLORS: dict[str, str] = {
    "I":   "#e6194B", "II":  "#3cb44b", "III": "#ffe119",
    "aVR": "#4363d8", "aVL": "#f58231", "aVF": "#911eb4",
    "V1":  "#42d4f4", "V2":  "#f032e6", "V3":  "#bfef45",
    "V4":  "#fabed4", "V5":  "#469990", "V6":  "#9A6324",
    "RHY": "#000000",
}


def _load_layouts() -> dict:
    with open(_VENDOR_LAYOUTS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _render_named(
    img: np.ndarray, raw_lines: np.ndarray, x_offset: int,
    chunk_px: int, layout_name: str, layouts: dict,
    title: str, out_path: Path, blend: float = 0.45,
) -> None:
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = (1 - blend) * img_rgb + blend
    img_blend = np.clip(img_blend, 0, 1)
    fig_w = max(20.0, w / 200.0); fig_h = fig_w * (h / w) + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])

    layout_def = layouts.get(layout_name)
    if layout_def is None:
        ax.set_title(f"{title}\n[layout {layout_name} desconhecido]",
                     fontsize=11, color="red")
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    leads_def = layout_def["leads"]
    cols = int(layout_def["layout"]["cols"])
    rhythm_leads = layout_def.get("rhythm_leads", [])
    is_matrix = isinstance(leads_def[0], list) if leads_def else False

    rows_iter = leads_def if is_matrix else [leads_def]
    n_lines = raw_lines.shape[0]
    seen_in_legend: set = set()

    for row_idx, row_leads in enumerate(rows_iter):
        if row_idx >= n_lines:
            break
        if not isinstance(row_leads, list):
            row_leads = [row_leads]
        line_full = raw_lines[row_idx]   # (lines_w,) em pixel-Y
        line_w = line_full.shape[0]
        for col_idx, lead_name_raw in enumerate(row_leads):
            name = str(lead_name_raw).lstrip("-")
            color = LEAD_COLORS.get(name, "#999999")
            x_local_start = col_idx * chunk_px
            x_local_end = x_local_start + chunk_px
            if x_local_start >= line_w:
                continue
            actual_end = min(x_local_end, line_w)
            chunk = line_full[x_local_start:actual_end]
            xs = np.arange(actual_end - x_local_start) + x_offset + x_local_start
            ax.plot(xs, chunk, color=color, lw=1.6, alpha=0.9, zorder=3)
            valid = ~np.isnan(chunk)
            if valid.any():
                fx = xs[np.argmax(valid)]
                fy = chunk[np.argmax(valid)]
                ax.text(
                    fx + 8, max(20, fy - 35), name,
                    fontsize=12, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.95),
                    zorder=5,
                )
            seen_in_legend.add(name)

    # Rhythm strip — full width (cols * chunk_px), em preto
    n_main = len(rows_iter)
    if len(rhythm_leads) > 0 and n_lines > n_main:
        line = raw_lines[n_main]
        full_local = cols * chunk_px
        actual_end = min(full_local, line.shape[0])
        chunk = line[:actual_end]
        xs = np.arange(actual_end) + x_offset
        ax.plot(xs, chunk, color=LEAD_COLORS["RHY"], lw=1.6, alpha=0.9, zorder=3)
        valid = ~np.isnan(chunk)
        if valid.any():
            fx = xs[np.argmax(valid)]
            fy = chunk[np.argmax(valid)]
            ax.text(
                fx + 8, max(20, fy - 35), "Rhythm strip",
                fontsize=12, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=LEAD_COLORS["RHY"],
                          ec="black", alpha=0.95),
                zorder=5,
            )

    legend_handles = [
        mpatches.Patch(color=LEAD_COLORS.get(n, "#999"), label=n)
        for n in sorted(seen_in_legend)
    ]
    if len(rhythm_leads) > 0 and n_lines > n_main:
        legend_handles.append(
            mpatches.Patch(color=LEAD_COLORS["RHY"], label="Rhythm")
        )
    ax.legend(handles=legend_handles, loc="upper right", ncol=4,
              fontsize=10, framealpha=0.95)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _process(stem: str, layouts_yaml: dict) -> None:
    img_path = UNDIST_DIR / f"{stem}.png"
    if not img_path.exists():
        print(f"[ERRO] {img_path} nao existe")
        return
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] falha ao ler {img_path}")
        return
    h, w = img.shape[:2]
    print(f"\n{'=' * 78}")
    print(f"  {stem}.png  shape=(H={h}, W={w})")
    print("=" * 78)

    try:
        result = extract_signals_stenhede(
            image_bgr=img, use_cropper=False, use_internal_pixel_size=True,
        )
    except Exception as e:
        print(f"[ERRO Stenhede] {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return

    layout = result["match"].get("layout")
    cost = result["match"].get("cost", float("nan"))
    pxmm = result["avg_pixel_per_mm"]
    x_offset = result["raw_lines_x_offset"]
    fs = result["sampling_rate_hz"]
    print(
        f"layout={layout} cost={cost:.2f} px/mm={pxmm:.2f} "
        f"x_offset={x_offset} fs={fs:.1f}Hz"
    )

    canonical = result["canonical_lines_uv"]   # (12, chunk_px)
    chunk_px_calib = canonical.shape[1]
    print(
        f"canonical (calibrado): shape={canonical.shape}  "
        f"chunk_px={chunk_px_calib}  "
        f"({chunk_px_calib / pxmm:.1f} mm = "
        f"{chunk_px_calib / pxmm / 25.0:.2f} s)"
    )

    # Reconstroi raw_lines_pixel original em sistema da imagem (not padded)
    # extract_signals_stenhede só retorna padded — vamos derivar raw lines
    # daqui novamente pelo mesmo path interno
    raw_padded = result["raw_lines_pixel"]  # (N, W) com NaN nas margens
    # raw_lines bruto = raw_padded[:, x_offset:x_offset+lines_w]
    # Pra render, usamos raw_padded direto SLICED por chunk
    n_lines = raw_padded.shape[0]
    # Reconstruir raw_lines (N, lines_w) removendo NaN padding
    valid_per_col = (~np.isnan(raw_padded)).any(axis=0)
    if valid_per_col.any():
        first = int(np.argmax(valid_per_col))
        last = len(valid_per_col) - int(np.argmax(valid_per_col[::-1])) - 1
        lines_w = last - first + 1
        # Sanity: first ≈ x_offset
        raw_lines = raw_padded[:, first:last + 1]
        actual_x_offset = first
    else:
        raw_lines = raw_padded
        actual_x_offset = 0
        lines_w = raw_padded.shape[1]
    print(f"raw_lines reconstruído: shape={raw_lines.shape}, "
          f"actual_x_offset={actual_x_offset}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stem}_overlay_calibrated.png"
    title = (
        f"{stem} — Canonicalize CALIBRADO "
        f"(layout={layout}, chunk={chunk_px_calib}px = "
        f"{chunk_px_calib / pxmm:.0f}mm)"
    )
    _render_named(
        img, raw_lines, actual_x_offset, chunk_px_calib,
        layout or "standard_3x4_with_r1", layouts_yaml, title, out_path,
    )
    print(f"[Render] {out_path.name}")

    # ===== Tabela comparativa =====
    layout_def = layouts_yaml.get(layout)
    if layout_def is None:
        print(f"[AVISO] layout {layout} não encontrado em yaml — sem tabela")
        return
    cols = int(layout_def["layout"]["cols"])
    leads_def = layout_def["leads"]

    chunk_old = w // cols  # antigo (rígido)
    print(f"\nTabela comparativa de fronteiras X:")
    print(f"  {'Lead':<6}{'col':>4}{'x_start_old':>13}"
          f"{'x_start_new':>13}{'x_center_old':>14}{'x_center_new':>14}")
    print("  " + "-" * 60)
    is_matrix = isinstance(leads_def[0], list)
    rows_iter = leads_def if is_matrix else [leads_def]
    for row_idx, row_leads in enumerate(rows_iter):
        if not isinstance(row_leads, list):
            row_leads = [row_leads]
        for c_idx, lead_raw in enumerate(row_leads):
            name = str(lead_raw).lstrip("-")
            x_start_old = c_idx * chunk_old
            x_start_new = actual_x_offset + c_idx * chunk_px_calib
            x_center_old = x_start_old + chunk_old / 2
            x_center_new = x_start_new + chunk_px_calib / 2
            print(
                f"  {name:<6}{c_idx:>4}{x_start_old:>13d}{x_start_new:>13d}"
                f"{x_center_old:>14.0f}{x_center_new:>14.0f}"
            )


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(name)s] %(message)s")
    layouts_yaml = _load_layouts()
    print("=" * 78)
    print(" Canonicalize CALIBRADO — IMG_1387 e IMG_1303")
    print("=" * 78)
    for stem in TARGETS:
        _process(stem, layouts_yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
