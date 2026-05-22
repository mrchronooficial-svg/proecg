"""Sweep alpha do process_sparse_prob em IMG_1387.

Original (alpha=1):  (x - x.mean()).clamp(0) / (x.max() + 1e-8)
Generalizado:        (x - alpha * x.mean()).clamp(0) / (x.max() + 1e-8)

Alpha = 0 -> sem subtracao da media (mantem softmax bruto, so normaliza pelo max).
Alpha = 1 -> equivalente ao process_sparse_prob original.

Rodar SignalExtractor com defaults do Stenhede.
Para cada alpha imprime:
  - NaN% medio (trim de bordas)
  - n_lines detectadas
  - max signal_prob na ROI x=950:1050 (regiao do pico R problematico)

Saidas em _visualizations/:
  IMG_1387_alpha_100.png
  IMG_1387_alpha_075.png
  IMG_1387_alpha_050.png
  IMG_1387_alpha_025.png
  IMG_1387_alpha_000.png
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

ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]
ROI_X = (950, 1050)


def _signal_prob_alpha(image_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """Roda U-Net e devolve signal_prob com (x - alpha*x.mean()).clamp(0)/x.max()."""
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
        # generalizacao do process_sparse_prob
        sp = sp - float(alpha) * sp.mean()
        sp = torch.clamp(sp, min=0)
        sp = sp / (sp.max() + 1e-9)
    sp_np = sp.cpu().numpy().astype(np.float32)
    if sp_np.shape != (h_orig, w_orig):
        sp_np = cv2.resize(sp_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return sp_np


def _run_extractor(signal_prob: np.ndarray) -> tuple[np.ndarray, int]:
    """Defaults do Stenhede: threshold_sum=10, label_thresh=0.1, etc."""
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore

    H, W = signal_prob.shape
    extractor = SignalExtractor()  # tudo em default
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
    ax.axvspan(ROI_X[0], ROI_X[1], color="yellow", alpha=0.18, zorder=1)
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

    print(f"--- {STEM}  shape={img.shape}  ROI x={ROI_X[0]}:{ROI_X[1]} ---")
    print(f"{'alpha':>6} {'n_lines':>8} {'NaN%':>7} "
          f"{'max_prob_ROI':>13} {'tempo':>7}")
    print("-" * 55)

    rows = []
    for alpha in ALPHAS:
        t0 = time.perf_counter()
        sp = _signal_prob_alpha(img, alpha=alpha)
        roi_max = float(sp[:, ROI_X[0]:ROI_X[1]].max())
        rl, _x_off = _run_extractor(sp)
        n = int(rl.shape[0])
        nan = _nanpct(rl)
        dt = time.perf_counter() - t0
        rows.append({"alpha": alpha, "n": n, "nan": nan,
                     "roi_max": roi_max, "dt": dt})
        print(f"{alpha:6.2f} {n:>8d} {nan:6.1f}% {roi_max:13.4f} "
              f"{dt:6.1f}s")
        a_label = f"{int(round(alpha * 100)):03d}"
        out_path = OUT_DIR / f"{STEM}_alpha_{a_label}.png"
        _render(img, rl,
                f"{STEM}  alpha={alpha:.2f} (process_sparse_prob)  "
                f"n_lines={n}  NaN%={nan:.1f}%  ROI_max={roi_max:.3f}",
                out_path)
        print(f"        [Render] {out_path.name}")

    # tabela final consolidada
    print("\nResumo:")
    print(f"{'alpha':>6} {'n_lines':>8} {'NaN%':>7} {'max_prob_ROI':>13}")
    for r in rows:
        print(f"{r['alpha']:6.2f} {r['n']:>8d} {r['nan']:6.1f}% "
              f"{r['roi_max']:13.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
