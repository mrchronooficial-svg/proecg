"""Diagnostico do pico R perdido em IMG_1387.

Saidas em _visualizations/:
  IMG_1387_perfil_vertical_picoR.png   (perfil vertical em X* +- 5)
  IMG_1387_islands_picoR.png           (heatmap + ilhas + linhas extraidas)
  IMG_1387_overlay_extract_max.png     (extracao por argmax_y)

Imprime no stdout: coluna escolhida, ilhas no recorte (bbox, label, sum, classify),
e tabela de Y-line (mean ponderada) vs Y-line (argmax) por coluna.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import label as sk_label

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.stenhede_adapter import (
    _DEFAULT_MAX_DIM,
    _SIGNAL_CLASS,
    _ensure_vendor_on_path,
    get_unet,
)

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
STEM = "IMG_1387"
# ROI agora cobre o ULTIMO pico R em V5/V6 (col 4 do layout 3x4, ultimas
# colunas da imagem). 3500..3950 deixa o argmax achar o pico mais à direita.
ROI_X = (3500, 3950)


def _signal_prob_pair(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(softmax_raw, sparse_processed) shape (H, W) float32."""
    h_orig, w_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (image_rgb - image_rgb.min()) / max(image_rgb.max() - image_rgb.min(), 1e-8)
    max_side = max(h_orig, w_orig)
    if max_side > _DEFAULT_MAX_DIM:
        scale = _DEFAULT_MAX_DIM / float(max_side)
        new_h = int(round(h_orig * scale))
        new_w = int(round(w_orig * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        sp_raw = probs[0, _SIGNAL_CLASS].clone()
        sp_sparse = sp_raw.clone()
        sp_sparse = sp_sparse - sp_sparse.mean()
        sp_sparse = torch.clamp(sp_sparse, min=0)
        sp_sparse = sp_sparse / (sp_sparse.max() + 1e-9)
    raw_np = sp_raw.cpu().numpy().astype(np.float32)
    spr_np = sp_sparse.cpu().numpy().astype(np.float32)
    if raw_np.shape != (h_orig, w_orig):
        raw_np = cv2.resize(raw_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    if spr_np.shape != (h_orig, w_orig):
        spr_np = cv2.resize(spr_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return raw_np, spr_np


def _find_peak_column(sp: np.ndarray, x_lo: int, x_hi: int) -> int:
    """Coluna X* mais à DIREITA na ROI com signal_prob > 0.5 e que seja
    maximo local da projecao por coluna (evita pegar borda de papel)."""
    H, W = sp.shape
    x_hi = min(x_hi, W)
    sl = sp[:, x_lo:x_hi]
    col_max = sl.max(axis=0)  # max prob em cada coluna da ROI
    # Quero o maximo local mais à direita em torno dos picos R
    threshold = 0.5
    above = col_max > threshold
    if not above.any():
        # fallback: argmax global
        return x_lo + int(col_max.argmax())
    # rightmost run of "above"; pega o argmax dentro dessa run
    idxs = np.where(above)[0]
    # encontra clusters separados por gaps de pelo menos 30 colunas
    gaps = np.where(np.diff(idxs) > 30)[0]
    if gaps.size == 0:
        cluster = idxs
    else:
        cluster = idxs[gaps[-1] + 1:]
    rel = cluster[col_max[cluster].argmax()]
    return x_lo + int(rel)


def _plot_vertical_profile(sp_sparse: np.ndarray, sp_raw: np.ndarray,
                           x_star: int, out_path: Path) -> None:
    H, _ = sp_sparse.shape
    cols = list(range(x_star - 5, x_star + 6))
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), facecolor="white")
    ax_top, ax_bot = axes

    y_axis = np.arange(H)
    cmap = plt.get_cmap("tab20")
    for k, c in enumerate(cols):
        if 0 <= c < sp_sparse.shape[1]:
            color = cmap(k % 20)
            ax_top.plot(sp_sparse[:, c], y_axis,
                        color=color, lw=0.9, alpha=0.85,
                        label=f"x={c}{'*' if c == x_star else ''}")
            ax_bot.plot(sp_raw[:, c], y_axis,
                        color=color, lw=0.9, alpha=0.85,
                        label=f"x={c}{'*' if c == x_star else ''}")
    ax_top.invert_yaxis()
    ax_bot.invert_yaxis()
    ax_top.set_title(f"COM process_sparse_prob (x={x_star} +- 5)",
                     fontsize=12, fontweight="bold")
    ax_bot.set_title("SEM process_sparse_prob (softmax puro)",
                     fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("signal_prob")
        ax.set_ylabel("Y (row)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _diag_islands(sp_sparse: np.ndarray, x_star: int,
                  ext_threshold_sum: float = 10.0,
                  ext_label_thresh: float = 0.1) -> list[dict]:
    """Replica _label_regions do SignalExtractor mas com debug por ilha.
    Devolve lista de ilhas que contem alguma coluna em [x_star-5, x_star+5].
    Cada item: {label, ymin, ymax, xmin, xmax, sum_prob, n_pixels,
                line_mean (y por col), line_argmax}.
    """
    fmap = sp_sparse.astype(np.float32)
    H, W = fmap.shape
    bin_mask = (fmap > ext_label_thresh).astype(np.int32)
    lab_initial = sk_label(bin_mask, connectivity=1)
    # Filtra por threshold_sum (igual _label_regions)
    keep = np.zeros_like(lab_initial)
    counts = {}
    for lid in np.unique(lab_initial):
        if lid == 0:
            continue
        s = float(fmap[lab_initial == lid].sum())
        counts[lid] = s
        if s >= ext_threshold_sum:
            keep[lab_initial == lid] = lid
    # relabel
    final_lab = sk_label(keep > 0, connectivity=1)

    # Acha labels que tem pixels em colunas [x_star-5, x_star+5]
    col_lo = max(0, x_star - 5)
    col_hi = min(W, x_star + 6)
    col_slice = final_lab[:, col_lo:col_hi]
    interesting = sorted(set(int(v) for v in np.unique(col_slice) if v != 0))

    pos = np.arange(H, dtype=np.float64).reshape(-1, 1)
    diags = []
    for lid in interesting:
        m = final_lab == lid
        ys, xs = np.where(m)
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
        s = float(fmap[m].sum())
        n = int(m.sum())

        # _extract_line_from_region (mean ponderada)
        masked = np.where(m, fmap, 0.0)
        denom = masked.sum(axis=0, keepdims=True).clip(min=1e-6)
        masked_n = masked / denom
        line_mean = (masked_n * pos).sum(axis=0)
        line_mean[line_mean < 1.0 / H] = np.nan

        # versao MAX (argmax_y dentro da mascara)
        line_max = np.full(W, np.nan, dtype=np.float64)
        for col in range(W):
            col_m = m[:, col]
            if not col_m.any():
                continue
            vals = fmap[:, col] * col_m
            line_max[col] = int(np.argmax(vals))

        # _classify_line: line precisa cair na mask em >= 95% das colunas com mask
        cols_any = m.any(0)
        line_int = np.clip(np.nan_to_num(line_mean, nan=0).astype(int), 0, H - 1)
        in_mask = m[line_int, np.arange(W)]
        in_mask_pct = float(in_mask[cols_any].mean()) if cols_any.any() else 0.0

        diags.append({
            "label": lid,
            "ymin": ymin, "ymax": ymax,
            "xmin": xmin, "xmax": xmax,
            "sum_prob": s, "n_pixels": n,
            "in_mask_pct": in_mask_pct,
            "would_classify_as_good":
                in_mask_pct >= 0.95,
            "line_mean": line_mean,
            "line_max": line_max,
        })
    return diags, final_lab


def _plot_islands(img_bgr: np.ndarray, sp_sparse: np.ndarray,
                  x_star: int, diags: list[dict], final_lab: np.ndarray,
                  out_path: Path) -> None:
    """Recorte focado em ROI: heatmap + ilhas (bordas coloridas) + line_mean (vermelho)
    + line_max (verde). Limita Y ao bbox unido das ilhas, +20px margem."""
    if not diags:
        return
    H, W = sp_sparse.shape
    y0 = max(0, min(d["ymin"] for d in diags) - 20)
    y1 = min(H, max(d["ymax"] for d in diags) + 20)
    x0 = max(0, x_star - 80)
    x1 = min(W, x_star + 80)

    img_crop = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sp_crop = sp_sparse[y0:y1, x0:x1]
    lab_crop = final_lab[y0:y1, x0:x1]

    blend = 0.5 * img_crop + 0.5
    blend = np.clip(blend, 0, 1)

    fig_h = max(6.0, (y1 - y0) / 90.0)
    fig_w = max(8.0, (x1 - x0) / 90.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(blend, interpolation="nearest",
              extent=(x0, x1, y1, y0))
    # heatmap em cima
    ax.imshow(np.where(sp_crop > 0.1, sp_crop, np.nan),
              cmap="YlOrRd", alpha=0.55,
              extent=(x0, x1, y1, y0), interpolation="nearest")

    cmap = plt.get_cmap("tab10")
    for k, d in enumerate(diags):
        # bbox da ilha
        color = cmap(k % 10)
        rect = plt.Rectangle((d["xmin"], d["ymin"]),
                             d["xmax"] - d["xmin"],
                             d["ymax"] - d["ymin"],
                             fill=False, edgecolor=color, lw=1.5)
        ax.add_patch(rect)
        ax.text(d["xmin"], d["ymin"] - 3,
                f"L{d['label']} sum={d['sum_prob']:.1f} "
                f"n={d['n_pixels']}  mask_ok={d['in_mask_pct']*100:.0f}%",
                fontsize=8, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

        x_axis = np.arange(W)
        ax.plot(x_axis, d["line_mean"], color="red", lw=1.5,
                alpha=0.9, label=f"L{d['label']} mean" if k == 0 else None)
        ax.plot(x_axis, d["line_max"], color="lime", lw=1.5, ls="--",
                alpha=0.9, label=f"L{d['label']} argmax" if k == 0 else None)

    ax.axvline(x_star, color="cyan", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title(f"{STEM}  ilhas em x={x_star}  "
                 f"label_thresh=0.1 threshold_sum=10.0  "
                 f"vermelho=mean ponderada, verde=argmax",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _run_extractor_max(signal_prob: np.ndarray) -> tuple[np.ndarray, int]:
    """Roda SignalExtractor com defaults, mas substitui _extract_line_from_region
    por argmax_y dentro da mascara. Resto (label/threshold_sum/classify/merge)
    eh igual."""
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore

    H, W = signal_prob.shape
    extractor = SignalExtractor()

    def _extract_line_max(fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        H_, W_ = fmap.shape
        masked = torch.where(mask, fmap, torch.zeros_like(fmap))
        # argmax por coluna; col sem mask -> NaN
        any_in_col = mask.any(dim=0)
        argmax_y = masked.argmax(dim=0).float()
        line = torch.where(any_in_col, argmax_y, torch.full_like(argmax_y, float("nan")))
        return line

    extractor._extract_line_from_region = _extract_line_max  # type: ignore[assignment]

    fmap = torch.from_numpy(np.ascontiguousarray(signal_prob)).float()
    lines_list = extractor._iterative_extraction(fmap.clone())
    extractor.num_peaks = extractor._autodetect_num_peaks(fmap)
    lines_list = [
        ln for ln in lines_list
        if (~torch.isnan(ln)).sum() > extractor.min_line_width
    ]
    if len(lines_list) == 0:
        return np.full((0, W), np.nan, dtype=np.float64), 0
    lines_full = torch.stack(lines_list, dim=0)
    chk = lines_full.clone()
    chk[chk == 0] = float("nan")
    valid_cols = chk.nan_to_num(0.0).abs().sum(0) > 0
    nonzero = torch.nonzero(valid_cols, as_tuple=True)[0]
    offset = int(nonzero[0].item()) if nonzero.numel() > 0 else 0
    merged_list, _ = extractor.match_and_merge_lines(lines_full)
    if len(merged_list) == 0:
        return np.full((0, W), np.nan, dtype=np.float64), offset
    merged = torch.stack(merged_list, dim=0).cpu().numpy().astype(np.float64)
    n_lines, lw = merged.shape
    padded = np.full((n_lines, W), np.nan, dtype=np.float64)
    end = min(offset + lw, W)
    padded[:, offset:end] = merged[:, :end - offset]
    return padded, offset


def _render_overlay(img: np.ndarray, raw_lines: np.ndarray, title: str,
                    out_path: Path) -> None:
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = 0.6 * img_rgb + 0.4
    img_blend = np.clip(img_blend, 0, 1)
    fig_w = max(20.0, w / 200.0); fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    if raw_lines.ndim == 2 and raw_lines.size > 0:
        x = np.arange(raw_lines.shape[1])
        for i in range(raw_lines.shape[0]):
            ax.plot(x, raw_lines[i], color="red", lw=1.5,
                    alpha=0.85, zorder=3)
    ax.axvspan(ROI_X[0], ROI_X[1], color="yellow", alpha=0.18, zorder=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _nanpct(raw_lines: np.ndarray) -> float:
    if raw_lines.size == 0:
        return 100.0
    valid = ~np.isnan(raw_lines)
    any_per_col = valid.any(axis=0)
    if not any_per_col.any():
        return 100.0
    first = int(np.argmax(any_per_col))
    last = len(any_per_col) - int(np.argmax(any_per_col[::-1])) - 1
    trim = raw_lines[:, first:last + 1]
    return float(np.isnan(trim).mean() * 100.0)


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(UNDIST_DIR / f"{STEM}.png"))
    if img is None:
        print("[ERRO] nao encontrei IMG_1387"); return 1

    H, W = img.shape[:2]
    print(f"--- {STEM}  H={H} W={W}  ROI x={ROI_X[0]}:{ROI_X[1]} ---\n")

    print("Computing signal_prob (raw + sparse)...")
    t0 = time.perf_counter()
    sp_raw, sp_sparse = _signal_prob_pair(img)
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s")

    # Passo 1 — coluna X*
    x_star = _find_peak_column(sp_sparse, *ROI_X)
    print(f"\n[Passo 1] X* (col com maior signal_prob na ROI) = {x_star}")
    print(f"  signal_prob[:, {x_star}].max() = {sp_sparse[:, x_star].max():.4f} (sparse), "
          f"{sp_raw[:, x_star].max():.4f} (raw)")
    out_perfil = OUT_DIR / f"{STEM}_perfil_vertical_picoR_v56.png"
    _plot_vertical_profile(sp_sparse, sp_raw, x_star, out_perfil)
    print(f"  [Render] {out_perfil.name}")

    # Passo 2 — ilhas e linhas
    print(f"\n[Passo 2] _label_regions na regiao do pico R (label_thresh=0.1, threshold_sum=10):")
    diags, final_lab = _diag_islands(sp_sparse, x_star)
    print(f"  N ilhas que tocam x={x_star}+-5: {len(diags)}")
    for k, d in enumerate(diags):
        print(f"  ilha #{k} L={d['label']}: bbox=(y={d['ymin']}-{d['ymax']}, "
              f"x={d['xmin']}-{d['xmax']})  sum_prob={d['sum_prob']:.2f}  "
              f"n_pix={d['n_pixels']}  classify_in_mask%={d['in_mask_pct']*100:.1f}  "
              f"good={d['would_classify_as_good']}")
        # Tabela mean vs argmax na coluna X*
        if d["xmin"] <= x_star <= d["xmax"]:
            ym = d["line_mean"][x_star]
            ya = d["line_max"][x_star]
            print(f"    em col x={x_star}: mean_y={ym:.1f}  argmax_y={ya:.1f}  "
                  f"diff={ym - ya:+.1f}")

    out_islands = OUT_DIR / f"{STEM}_islands_picoR_v56.png"
    _plot_islands(img, sp_sparse, x_star, diags, final_lab, out_islands)
    print(f"  [Render] {out_islands.name}")

    # Passo 3 — extracao por MAX
    print("\n[Passo 3] Extracao com argmax_y (substitui _extract_line_from_region):")
    t0 = time.perf_counter()
    rl_max, _ = _run_extractor_max(sp_sparse)
    n_max = int(rl_max.shape[0])
    nan_max = _nanpct(rl_max)
    print(f"  n_lines={n_max}  NaN%={nan_max:.1f}%  t={(time.perf_counter()-t0):.1f}s")
    out_max = OUT_DIR / f"{STEM}_overlay_extract_max_v56.png"
    _render_overlay(img, rl_max,
                    f"{STEM} - extract por argmax_y (sparse_proc, defaults)  "
                    f"n_lines={n_max}  NaN%={nan_max:.1f}%",
                    out_max)
    print(f"  [Render] {out_max.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
