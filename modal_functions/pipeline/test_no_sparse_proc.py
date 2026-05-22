"""COM/SEM process_sparse_prob no signal_prob da U-Net (IMG_1387).

Testes:
  1. COM process_sparse_prob (path atual em produção)
  2. SEM process_sparse_prob (softmax puro do canal signal)
  3. SEM process_sparse_prob + label_thresh=0.03 + threshold_sum=3.0

Em todos: threshold_sum=5.0 (exceto teste 3) e demais params em default.
Saidas em _visualizations/:
  IMG_1387_with_sparse_proc.png
  IMG_1387_without_sparse_proc.png
  IMG_1387_without_sparse_proc_sensitive.png

E imprime, na regiao x=950:1050, max signal_prob com/sem sparse_proc + diff.
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

# regiao para diagnosticar o pico R perdido
ROI_X = (950, 1050)


def _signal_prob_raw_and_sparse(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (raw_softmax, sparse_processed) ambos shape (H, W) float32.
    raw_softmax = softmax channel signal direto (sem clamp/mean/normalize).
    sparse_processed = mesma coisa apos process_sparse_prob (path atual)."""
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
    sparse_np = sp_sparse.cpu().numpy().astype(np.float32)
    if raw_np.shape != (h_orig, w_orig):
        raw_np = cv2.resize(raw_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    if sparse_np.shape != (h_orig, w_orig):
        sparse_np = cv2.resize(sparse_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return raw_np, sparse_np


def _run_extractor(signal_prob: np.ndarray, **kwargs) -> tuple[np.ndarray, int, int]:
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore

    H, W = signal_prob.shape
    extractor = SignalExtractor(**kwargs)
    fmap = torch.from_numpy(np.ascontiguousarray(signal_prob)).float()
    lines_list = extractor._iterative_extraction(fmap.clone())
    extractor.num_peaks = extractor._autodetect_num_peaks(fmap)
    lines_list = [
        ln for ln in lines_list
        if (~torch.isnan(ln)).sum() > extractor.min_line_width
    ]
    if len(lines_list) == 0:
        return np.full((0, W), np.nan, dtype=np.float64), 0, 0
    lines_full = torch.stack(lines_list, dim=0)
    chk = lines_full.clone()
    chk[chk == 0] = float("nan")
    valid_cols = chk.nan_to_num(0.0).abs().sum(0) > 0
    nonzero = torch.nonzero(valid_cols, as_tuple=True)[0]
    offset = int(nonzero[0].item()) if nonzero.numel() > 0 else 0
    merged_list, _ = extractor.match_and_merge_lines(lines_full)
    if len(merged_list) == 0:
        return np.full((0, W), np.nan, dtype=np.float64), offset, 0
    merged = torch.stack(merged_list, dim=0).cpu().numpy().astype(np.float64)
    n_lines, lw = merged.shape
    padded = np.full((n_lines, W), np.nan, dtype=np.float64)
    end = min(offset + lw, W)
    padded[:, offset:end] = merged[:, :end - offset]
    return padded, offset, n_lines


def _render(img: np.ndarray, raw_lines: np.ndarray, title: str,
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
    # marca a ROI vertical para deixar visivel
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


def _roi_stats(arr: np.ndarray, x_lo: int, x_hi: int) -> dict:
    H, W = arr.shape
    x_hi = min(x_hi, W)
    if x_hi <= x_lo:
        return {"max_global": 0.0, "argmax_y": -1, "max_per_quarter": []}
    sl = arr[:, x_lo:x_hi]
    g_max = float(sl.max())
    argmax_idx = int(sl.argmax())
    y_at_max = argmax_idx // sl.shape[1]
    # 4 quartis de altura -> max em cada faixa (aproxima as 4 rows do 3x4+1)
    quarters = []
    for k in range(4):
        y0 = k * (H // 4)
        y1 = (k + 1) * (H // 4) if k < 3 else H
        quarters.append(float(sl[y0:y1, :].max()))
    return {"max_global": g_max, "argmax_y": y_at_max, "max_per_quarter": quarters}


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_path = UNDIST_DIR / f"{STEM}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] {img_path}"); return 1

    print(f"--- {STEM}  shape={img.shape}  ROI x={ROI_X[0]}:{ROI_X[1]} ---")
    print("computing signal_prob (raw softmax + sparse_proc)...")
    t0 = time.perf_counter()
    sp_raw, sp_sparse = _signal_prob_raw_and_sparse(img)
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s "
          f"raw[max={sp_raw.max():.3f} mean={sp_raw.mean():.4f}] "
          f"sparse[max={sp_sparse.max():.3f} mean={sp_sparse.mean():.4f}]")

    # Diagnostico ROI
    stats_sparse = _roi_stats(sp_sparse, *ROI_X)
    stats_raw = _roi_stats(sp_raw, *ROI_X)
    print("\n[ROI x=950:1050] max signal_prob:")
    print(f"  COM process_sparse_prob: max={stats_sparse['max_global']:.4f}  "
          f"y_argmax={stats_sparse['argmax_y']}  "
          f"per-row-quarter={['%.4f' % v for v in stats_sparse['max_per_quarter']]}")
    print(f"  SEM process_sparse_prob: max={stats_raw['max_global']:.4f}  "
          f"y_argmax={stats_raw['argmax_y']}  "
          f"per-row-quarter={['%.4f' % v for v in stats_raw['max_per_quarter']]}")
    print(f"  diff (SEM - COM): {stats_raw['max_global'] - stats_sparse['max_global']:.4f}")

    # Teste 1: COM process_sparse_prob, threshold_sum=5
    print("\n[Teste 1] COM process_sparse_prob, threshold_sum=5.0")
    t0 = time.perf_counter()
    rl1, _, n1 = _run_extractor(sp_sparse, threshold_sum=5.0)
    nan1 = _nanpct(rl1)
    print(f"  n_lines={n1}  NaN%={nan1:.1f}%  t={(time.perf_counter()-t0):.1f}s")
    out1 = OUT_DIR / f"{STEM}_with_sparse_proc.png"
    _render(img, rl1,
            f"{STEM} - COM process_sparse_prob, threshold_sum=5.0  "
            f"n_lines={n1}  NaN%={nan1:.1f}%",
            out1)
    print(f"  [Render] {out1.name}")

    # Teste 2: SEM process_sparse_prob, threshold_sum=5
    print("\n[Teste 2] SEM process_sparse_prob (softmax puro), threshold_sum=5.0")
    t0 = time.perf_counter()
    rl2, _, n2 = _run_extractor(sp_raw, threshold_sum=5.0)
    nan2 = _nanpct(rl2)
    print(f"  n_lines={n2}  NaN%={nan2:.1f}%  t={(time.perf_counter()-t0):.1f}s")
    out2 = OUT_DIR / f"{STEM}_without_sparse_proc.png"
    _render(img, rl2,
            f"{STEM} - SEM process_sparse_prob, threshold_sum=5.0  "
            f"n_lines={n2}  NaN%={nan2:.1f}%",
            out2)
    print(f"  [Render] {out2.name}")

    # Teste 3: SEM process_sparse_prob + label_thresh=0.03 + threshold_sum=3.0
    print("\n[Teste 3] SEM process_sparse_prob + label_thresh=0.03 + threshold_sum=3.0")
    t0 = time.perf_counter()
    rl3, _, n3 = _run_extractor(sp_raw, label_thresh=0.03, threshold_sum=3.0)
    nan3 = _nanpct(rl3)
    print(f"  n_lines={n3}  NaN%={nan3:.1f}%  t={(time.perf_counter()-t0):.1f}s")
    out3 = OUT_DIR / f"{STEM}_without_sparse_proc_sensitive.png"
    _render(img, rl3,
            f"{STEM} - SEM sparse_proc, label_thresh=0.03, threshold_sum=3.0  "
            f"n_lines={n3}  NaN%={nan3:.1f}%",
            out3)
    print(f"  [Render] {out3.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
