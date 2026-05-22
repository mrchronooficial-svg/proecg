"""3 versoes de _extract_line_from_region: mean / argmax / hybrid.

Roda em IMG_1387 e IMG_1303 (undistorted).
NAO altera codigo do vendor — substitui o metodo via monkey-patch
APENAS dentro deste teste.

Saidas em _visualizations/:
  IMG_1387_extract_mean.png      IMG_1303_extract_mean.png
  IMG_1387_extract_argmax.png    IMG_1303_extract_argmax.png
  IMG_1387_extract_hybrid.png    IMG_1303_extract_hybrid.png
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
TARGETS = ["IMG_1387", "IMG_1303"]


def _signal_prob(image_bgr: np.ndarray) -> np.ndarray:
    """signal_prob com process_sparse_prob (default em producao)."""
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
        sp = probs[0, _SIGNAL_CLASS].clone()
        sp = sp - sp.mean()
        sp = torch.clamp(sp, min=0)
        sp = sp / (sp.max() + 1e-9)
    sp_np = sp.cpu().numpy().astype(np.float32)
    if sp_np.shape != (h_orig, w_orig):
        sp_np = cv2.resize(sp_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return sp_np


# =====================================================================
# 3 extratores (substitutos de _extract_line_from_region)
# =====================================================================

def _extract_mean(fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Original do Stenhede: media ponderada Y por coluna."""
    H, W = fmap.shape
    pos = torch.arange(H).view(-1, 1).float()
    masked = torch.where(mask, fmap, torch.zeros_like(fmap))
    masked = masked / masked.sum(0, keepdim=True).clamp(min=1e-6)
    line = (masked * pos).sum(0)
    line[line < 1.0 / H] = float("nan")
    return line


def _extract_argmax(fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax_y dentro da mascara: segue o pixel de maior prob por coluna."""
    H, W = fmap.shape
    masked = torch.where(mask, fmap, torch.zeros_like(fmap))
    any_in_col = mask.any(dim=0)
    argmax_y = masked.argmax(dim=0).float()
    return torch.where(any_in_col, argmax_y,
                       torch.full_like(argmax_y, float("nan")))


def _extract_hybrid(fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hybrid: argmax onde range vertical da mascara na coluna > 30 px,
    senao media ponderada."""
    H, W = fmap.shape
    pos = torch.arange(H).view(-1, 1).float()
    masked = torch.where(mask, fmap, torch.zeros_like(fmap))

    # mean ponderada (sem nanificar ainda)
    masked_n = masked / masked.sum(0, keepdim=True).clamp(min=1e-6)
    line_mean = (masked_n * pos).sum(0)

    # argmax
    line_argmax = masked.argmax(dim=0).float()

    # range vertical da mascara em cada coluna
    pos_full = torch.arange(H).view(-1, 1).float().expand(H, W)
    pos_in_mask = torch.where(mask, pos_full, torch.full_like(pos_full, float("nan")))
    y_min = torch.nan_to_num(pos_in_mask, nan=float("inf")).min(dim=0).values
    y_max = torch.nan_to_num(pos_in_mask, nan=float("-inf")).max(dim=0).values
    y_range = y_max - y_min
    use_argmax = y_range > 30.0
    any_in_col = mask.any(dim=0)

    line = torch.where(use_argmax, line_argmax, line_mean)
    line = torch.where(any_in_col, line,
                       torch.full_like(line, float("nan")))
    # ainda aplica o threshold de "linha muito pequena = nan" do original
    too_small = line_mean < 1.0 / H
    line = torch.where(too_small & ~use_argmax,
                       torch.full_like(line, float("nan")), line)
    return line


# =====================================================================
# Pipeline executor com extrator customizado (mantem resto do Stenhede)
# =====================================================================

def _run_extractor_with(
    signal_prob: np.ndarray,
    extract_fn,
) -> tuple[np.ndarray, int, int]:
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore

    H, W = signal_prob.shape
    extractor = SignalExtractor()  # defaults
    extractor._extract_line_from_region = extract_fn  # type: ignore[assignment]

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


def _process(stem: str) -> None:
    img_path = UNDIST_DIR / f"{stem}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] {img_path}"); return

    print(f"\n=== {stem}  shape={img.shape} ===")
    print("computando signal_prob...")
    t0 = time.perf_counter()
    sp = _signal_prob(img)
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s")

    versions = [
        ("mean", _extract_mean, "media ponderada (original)"),
        ("argmax", _extract_argmax, "argmax_y por coluna"),
        ("hybrid", _extract_hybrid, "hybrid (argmax se range>30 senao mean)"),
    ]
    for tag, fn, desc in versions:
        t0 = time.perf_counter()
        rl, _x_off, n = _run_extractor_with(sp, fn)
        nan = _nanpct(rl)
        dt = time.perf_counter() - t0
        out_path = OUT_DIR / f"{stem}_extract_{tag}.png"
        _render(img, rl,
                f"{stem}  extract={tag} ({desc})  "
                f"n_lines={n}  NaN%={nan:.1f}%  t={dt:.1f}s",
                out_path)
        print(f"  [{tag:7s}] n_lines={n}  NaN%={nan:5.1f}%  t={dt:5.1f}s  "
              f"-> {out_path.name}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in TARGETS:
        _process(stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
