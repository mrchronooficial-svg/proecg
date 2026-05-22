"""Pos-processamento de R peaks DEPOIS do SignalExtractor (mean).

NAO altera codigo do Stenhede. Roda a extracao mean normal, e depois
aplica correcao por argmax onde:
  delta = |argmax_y - line_mean_y| > delta_thresh
  E prob[argmax_y, x] > prob_thresh
Por fim, suaviza com filtro 3-pontos (box).

3 combos:
  conservador: delta=15, prob=0.30
  moderado:    delta=10, prob=0.20
  agressivo:   delta= 8, prob=0.15

Saidas em _visualizations/:
  IMG_XXXX_extract_mean.png  (sem correcao)
  IMG_XXXX_extract_mean_corr_conservador_15_030.png
  IMG_XXXX_extract_mean_corr_moderado_10_020.png
  IMG_XXXX_extract_mean_corr_agressivo_8_015.png
  IMG_XXXX_extract_mean_corrected.png  (copia do melhor)
"""

from __future__ import annotations

import logging
import shutil
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
TARGETS = ["IMG_1303", "IMG_1387"]

# Combos (rotulo, delta, prob)
COMBOS = [
    ("conservador", 15, 0.30),
    ("moderado",    10, 0.20),
    ("agressivo",    8, 0.15),
]

# Limite de seguranca: nao corrigir se argmax estiver a mais que isso
# da linha_mean — protege contra row-jumping em ilhas multi-row
# (caso patologico V5/V6 em IMG_1387, ilha gigante L4).
# Cell de 3x4 layout tem ~333 px de altura; uso 150 px como teto razoavel.
SEARCH_RADIUS_PX = 150


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
        sp = probs[0, _SIGNAL_CLASS].clone()
        sp = sp - sp.mean()
        sp = torch.clamp(sp, min=0)
        sp = sp / (sp.max() + 1e-9)
    sp_np = sp.cpu().numpy().astype(np.float32)
    if sp_np.shape != (h_orig, w_orig):
        sp_np = cv2.resize(sp_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return sp_np


def _run_extractor_mean(signal_prob: np.ndarray) -> tuple[np.ndarray, int]:
    """Roda SignalExtractor com defaults (mean ponderada)."""
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore

    H, W = signal_prob.shape
    extractor = SignalExtractor()
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


# ---------------------------------------------------------------------
# Pos-processamento (camada nossa)
# ---------------------------------------------------------------------

def _postprocess_r_peaks_line(
    line_mean: np.ndarray, signal_prob: np.ndarray,
    delta_thresh: float = 15.0, prob_thresh: float = 0.3,
    search_radius: int = SEARCH_RADIUS_PX,
) -> tuple[np.ndarray, int]:
    """Corrige R peaks achatados pela media ponderada — aplicado por linha.

    Para cada coluna X com line_mean[x] valido, busca o argmax de
    signal_prob[:, x] dentro de [line_mean[x] - search_radius, line_mean[x] + search_radius]
    (proteçao contra row-jumping em ilhas multi-row). Se delta > delta_thresh
    E prob no argmax > prob_thresh, substitui line_mean[x] por argmax_y.

    Returns:
      (line_corrected, n_correcoes)
    """
    H, W = signal_prob.shape
    line_corr = line_mean.copy()
    n_corr = 0
    for x in range(W):
        y0 = line_mean[x]
        if np.isnan(y0):
            continue
        y0i = int(y0)
        lo = max(0, y0i - search_radius)
        hi = min(H, y0i + search_radius + 1)
        col = signal_prob[lo:hi, x]
        if col.size == 0:
            continue
        local_argmax = int(col.argmax())
        y_max = lo + local_argmax
        prob_at_max = float(col[local_argmax])
        delta = abs(y_max - y0)
        if delta > delta_thresh and prob_at_max > prob_thresh:
            line_corr[x] = float(y_max)
            n_corr += 1
    return line_corr, n_corr


def _smooth_3pt(line: np.ndarray) -> np.ndarray:
    """Filtro de 3 pontos (box) preservando NaN."""
    n = len(line)
    out = line.copy()
    for i in range(1, n - 1):
        a, b, c = line[i - 1], line[i], line[i + 1]
        if np.isnan(b):
            continue
        vals = [v for v in (a, b, c) if not np.isnan(v)]
        out[i] = float(np.mean(vals))
    return out


def _postprocess_lines(
    raw_lines: np.ndarray, signal_prob: np.ndarray,
    delta_thresh: float, prob_thresh: float,
) -> tuple[np.ndarray, int]:
    """Aplica _postprocess_r_peaks_line em cada linha + smooth_3pt."""
    if raw_lines.size == 0:
        return raw_lines, 0
    out = raw_lines.copy()
    total_corr = 0
    for i in range(raw_lines.shape[0]):
        corrected, n = _postprocess_r_peaks_line(
            raw_lines[i], signal_prob,
            delta_thresh=delta_thresh, prob_thresh=prob_thresh,
        )
        smoothed = _smooth_3pt(corrected)
        out[i] = smoothed
        total_corr += n
    return out, total_corr


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


def _process(stem: str) -> dict:
    img_path = UNDIST_DIR / f"{stem}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] {img_path}"); return {}

    print(f"\n=== {stem}  shape={img.shape} ===")
    print("computing signal_prob...")
    t0 = time.perf_counter()
    sp = _signal_prob(img)
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s")

    # Mean original
    t0 = time.perf_counter()
    rl_mean, _x_off = _run_extractor_mean(sp)
    n_mean = int(rl_mean.shape[0])
    nan_mean = _nanpct(rl_mean)
    dt_mean = time.perf_counter() - t0
    print(f"  [mean] n_lines={n_mean}  NaN%={nan_mean:.1f}%  t={dt_mean:.1f}s")
    out_mean = OUT_DIR / f"{stem}_extract_mean.png"
    _render(img, rl_mean,
            f"{stem}  mean ponderada (sem correcao)  "
            f"n_lines={n_mean}  NaN%={nan_mean:.1f}%",
            out_mean)
    print(f"    -> {out_mean.name}")

    results = {}
    for tag, dthr, pthr in COMBOS:
        t0 = time.perf_counter()
        rl_corr, n_corr = _postprocess_lines(rl_mean, sp,
                                             delta_thresh=dthr, prob_thresh=pthr)
        nan_corr = _nanpct(rl_corr)
        dt = time.perf_counter() - t0
        print(f"  [{tag:12s} delta={dthr:>2} prob={pthr:.2f}]  "
              f"n_correcoes={n_corr:>5d}  NaN%={nan_corr:.1f}%  t={dt:.1f}s")
        out_path = OUT_DIR / (
            f"{stem}_extract_mean_corr_{tag}_{dthr}_"
            f"{int(pthr*100):03d}.png"
        )
        _render(img, rl_corr,
                f"{stem}  mean+corr {tag} (delta>{dthr}, prob>{pthr})  "
                f"n_corr={n_corr}  NaN%={nan_corr:.1f}%",
                out_path)
        print(f"    -> {out_path.name}")
        results[tag] = {"path": out_path, "n_corr": n_corr,
                        "nan_pct": nan_corr, "rl": rl_corr}

    return {"mean_path": out_mean, "results": results, "n_lines": n_mean}


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for stem in TARGETS:
        summary[stem] = _process(stem)
    print("\n" + "=" * 60)
    print("Resumo:")
    for stem, d in summary.items():
        print(f"\n{stem}:")
        for tag, r in d.get("results", {}).items():
            print(f"  {tag:12s}  n_corr={r['n_corr']:>5d}  NaN%={r['nan_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
