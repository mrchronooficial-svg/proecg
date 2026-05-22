"""Sweep de threshold_sum em IMG_1387 (defaults exceto threshold_sum)."""

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
    _pad_lines_to_image_width,
    get_unet,
)

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
STEM = "IMG_1387"

DEFAULTS = dict(
    label_thresh=0.1,
    candidate_span=10,
    max_iterations=4,
    min_line_width=30,
)
THSUM_VALUES = [1.0]


def _signal_prob(image_bgr: np.ndarray) -> np.ndarray:
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
        sp = probs[0, _SIGNAL_CLASS]
        sp = sp - sp.mean()
        sp = torch.clamp(sp, min=0)
        sp = sp / (sp.max() + 1e-9)
    sp_np = sp.cpu().numpy().astype(np.float32)
    if sp_np.shape != (h_orig, w_orig):
        sp_np = cv2.resize(sp_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return sp_np


def _run_extractor(signal_prob: np.ndarray, **kwargs):
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
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_path = UNDIST_DIR / f"{STEM}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] {img_path}"); return 1

    print(f"--- {STEM} sweep threshold_sum ---")
    print("  pré-computando signal_prob (U-Net)...")
    t0 = time.perf_counter()
    sp = _signal_prob(img)
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s")

    for v in THSUM_VALUES:
        kwargs = dict(DEFAULTS)
        kwargs["threshold_sum"] = v
        t0 = time.perf_counter()
        try:
            raw_lines, x_off, n_lines = _run_extractor(sp, **kwargs)
        except Exception as e:
            print(f"  threshold_sum={v}: ERRO {type(e).__name__}: {e}")
            continue
        dt = time.perf_counter() - t0

        valid = ~np.isnan(raw_lines)
        any_per_col = valid.any(axis=0)
        if any_per_col.any():
            first = int(np.argmax(any_per_col))
            last = len(any_per_col) - int(np.argmax(any_per_col[::-1])) - 1
            trim = raw_lines[:, first:last + 1]
            nan_pct = float(np.isnan(trim).mean() * 100.0)
        else:
            nan_pct = 100.0

        print(f"  threshold_sum={v}: n_lines={n_lines} "
              f"NaN%={nan_pct:.1f}% tempo={dt:.1f}s")

        v_label = str(int(v)) if float(v).is_integer() else str(v)
        out_path = OUT_DIR / f"{STEM}_thsum_{v_label}.png"
        _render(
            img, raw_lines,
            f"{STEM} — threshold_sum={v} (default outros)  "
            f"n_lines={n_lines}  NaN%={nan_pct:.1f}%  t={dt:.1f}s",
            out_path,
        )
        print(f"    [Render] {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
