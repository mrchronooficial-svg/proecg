"""
Debug da extração que está perdendo QRS em algumas derivações.

Passos:
  1. Heatmap de signal_prob sobreposto na imagem (mostra onde a U-Net
     está/não está vendo traçado)
  2. Testes A/B/C com parâmetros do SignalExtractor mais sensíveis
  3. Tabela comparativa de NaN%, linhas detectadas, tempo
  4. min/max/mean do signal_prob ANTES e DEPOIS do process_sparse_prob

Saídas em modal_functions/pipeline/digitize/_visualizations/:
  IMG_1303_heatmap_signal_prob.png
  IMG_1387_heatmap_signal_prob.png
  IMG_1387_overlay_params_A.png
  IMG_1387_overlay_params_B.png
  IMG_1387_overlay_params_C.png
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
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
    _ensure_vendor_on_path,
    _SIGNAL_CLASS,
    _UNET_KWARGS,
    get_unet,
    get_unet_feature_maps,
)

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")


def _signal_prob_raw_and_processed(
    image_bgr: np.ndarray, max_dim: int = _DEFAULT_MAX_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (signal_prob_raw, signal_prob_processed) ambos `(H, W)`
    em coords da imagem original.

    raw = softmax direto (canal signal) — escala [0, 1] mas tipicamente
          mass concentrada em valores baixos
    processed = depois do `process_sparse_prob`: subtrai média, clamp 0,
                normaliza pelo max → realça regiões acima da média
    """
    h_orig, w_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (image_rgb - image_rgb.min()) / max(image_rgb.max() - image_rgb.min(), 1e-8)

    max_side = max(h_orig, w_orig)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_h, new_w = int(round(h_orig * scale)), int(round(w_orig * scale))
        img_in = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_in = img

    tensor = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0).float()
    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        sp = probs[0, _SIGNAL_CLASS]              # (H, W) raw após softmax
        sp_raw = sp.cpu().numpy().astype(np.float32)
        # process_sparse_prob
        sp2 = sp - sp.mean()
        sp2 = torch.clamp(sp2, min=0)
        sp2 = sp2 / (sp2.max() + 1e-9)
        sp_proc = sp2.cpu().numpy().astype(np.float32)

    if sp_raw.shape != (h_orig, w_orig):
        sp_raw = cv2.resize(sp_raw, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    if sp_proc.shape != (h_orig, w_orig):
        sp_proc = cv2.resize(sp_proc, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return sp_raw, sp_proc


def _save_heatmap(
    image_bgr: np.ndarray, signal_prob: np.ndarray,
    title: str, out_path: Path, alpha: float = 0.6,
) -> None:
    """Imagem original clareada com heatmap de signal_prob por cima."""
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = 0.4 * img_rgb + 0.6  # clarear com 60% branco
    img_blend = np.clip(img_blend, 0, 1)

    # Colormap "hot" do signal_prob
    sp_norm = signal_prob.copy()
    if sp_norm.max() > 0:
        sp_norm = sp_norm / sp_norm.max()
    heat_rgba = cm.get_cmap("hot")(sp_norm)   # (H, W, 4)
    heat_rgb = heat_rgba[..., :3].astype(np.float32)

    # Blend: heatmap só onde signal_prob > 0
    sp_mask = (sp_norm > 0.01)[..., None].astype(np.float32)
    final = img_blend * (1 - alpha * sp_mask) + heat_rgb * (alpha * sp_mask)
    final = np.clip(final, 0, 1)

    fig_w = max(16.0, w / 200.0); fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(final, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _signal_extractor_with_kwargs(signal_prob: np.ndarray, **kwargs):
    """Roda o SignalExtractor com kwargs customizados (e detecta x_offset).
    Devolve (raw_lines_padded_to_W, x_offset, n_lines)."""
    _ensure_vendor_on_path()
    from src.model.signal_extractor import SignalExtractor  # type: ignore

    H, W = signal_prob.shape
    extractor = SignalExtractor(**kwargs)
    fmap = torch.from_numpy(np.ascontiguousarray(signal_prob)).float()

    # Pré-trim lines (full-width) só pra calcular o offset
    lines_list = extractor._iterative_extraction(fmap.clone())
    extractor.num_peaks = extractor._autodetect_num_peaks(fmap)
    lines_list = [
        ln for ln in lines_list
        if (~torch.isnan(ln)).sum() > extractor.min_line_width
    ]
    if len(lines_list) == 0:
        return np.full((0, W), np.nan), 0, 0

    lines_full = torch.stack(lines_list, dim=0)
    chk = lines_full.clone()
    chk[chk == 0] = float("nan")
    valid_cols = chk.nan_to_num(0.0).abs().sum(0) > 0
    nonzero = torch.nonzero(valid_cols, as_tuple=True)[0]
    offset = int(nonzero[0].item()) if nonzero.numel() > 0 else 0

    merged_list, _ = extractor.match_and_merge_lines(lines_full)
    if len(merged_list) == 0:
        return np.full((0, W), np.nan), offset, 0
    merged = torch.stack(merged_list, dim=0).cpu().numpy().astype(np.float64)
    n_lines, lw = merged.shape
    padded = np.full((n_lines, W), np.nan, dtype=np.float64)
    end = min(offset + lw, W)
    padded[:, offset:end] = merged[:, :end - offset]
    return padded, offset, n_lines


def _render_overlay(
    image_bgr: np.ndarray, raw_lines: np.ndarray,
    title: str, out_path: Path,
    blend: float = 0.4,
) -> None:
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = (1 - blend) * img_rgb + blend
    img_blend = np.clip(img_blend, 0, 1)
    fig_w = max(16.0, w / 200.0); fig_h = fig_w * (h / w)
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


def _process_heatmap(stem: str) -> None:
    img_path = UNDIST_DIR / f"{stem}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] {img_path}")
        return
    print(f"\n--- {stem} (heatmap signal_prob) ---")
    sp_raw, sp_proc = _signal_prob_raw_and_processed(img)
    print(
        f"  signal_prob RAW (após softmax):       "
        f"min={sp_raw.min():.4f} max={sp_raw.max():.4f} mean={sp_raw.mean():.4f}"
    )
    print(
        f"  signal_prob PROCESSED (process_sparse): "
        f"min={sp_proc.min():.4f} max={sp_proc.max():.4f} mean={sp_proc.mean():.4f}"
    )
    print(
        f"  pixels com prob_raw > 0.10:  {int((sp_raw > 0.10).sum())}"
    )
    print(
        f"  pixels com prob_raw > 0.05:  {int((sp_raw > 0.05).sum())}"
    )
    print(
        f"  pixels com prob_raw > 0.02:  {int((sp_raw > 0.02).sum())}"
    )
    print(
        f"  pixels com prob_proc > 0.10: {int((sp_proc > 0.10).sum())}"
    )
    out = OUT_DIR / f"{stem}_heatmap_signal_prob.png"
    _save_heatmap(
        img, sp_proc,
        f"{stem} — signal_prob (processed) sobre imagem (vermelho=alta prob)",
        out,
    )
    print(f"  [Render] {out.name}")


def _process_param_tests(stem: str) -> None:
    img_path = UNDIST_DIR / f"{stem}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] {img_path}")
        return
    print(f"\n--- {stem} (testes A/B/C de SignalExtractor) ---")

    # Pré-computa signal_prob (uma vez só)
    print("  pré-computando signal_prob (U-Net Stenhede)...")
    t0 = time.perf_counter()
    sp_raw, sp_proc = _signal_prob_raw_and_processed(img)
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s")

    tests = [
        ("A", dict(label_thresh=0.1, threshold_sum=10.0, candidate_span=10,
                   max_iterations=4, min_line_width=30)),
        ("B", dict(label_thresh=0.05, threshold_sum=5.0, candidate_span=15,
                   max_iterations=6, min_line_width=20)),
        ("C", dict(label_thresh=0.03, threshold_sum=3.0, candidate_span=20,
                   max_iterations=8, min_line_width=15)),
    ]

    results = []
    for name, kw in tests:
        print(f"\n  Teste {name}: {kw}")
        t0 = time.perf_counter()
        try:
            raw_lines, offset, n_lines = _signal_extractor_with_kwargs(sp_proc, **kw)
        except Exception as e:
            print(f"    [ERRO] {type(e).__name__}: {e}")
            results.append((name, kw, 0, float("inf"), float("nan"), 0))
            continue
        dt = time.perf_counter() - t0
        # NaN % médio das 12 cells "ideais" (mas raw_lines tem só N_rows linhas;
        # para análise simples: NaN% nas N linhas detectadas)
        if n_lines > 0:
            nan_pct = float(np.isnan(raw_lines).mean() * 100.0)
            valid_cols_per_line = (~np.isnan(raw_lines)).sum(axis=1).tolist()
        else:
            nan_pct = 100.0
            valid_cols_per_line = []
        print(
            f"    n_lines={n_lines}  NaN%={nan_pct:.1f}%  offset={offset}  "
            f"tempo={dt:.1f}s"
        )
        if valid_cols_per_line:
            print(f"    samples válidos por linha: {valid_cols_per_line}")
        out_path = OUT_DIR / f"{stem}_overlay_params_{name}.png"
        _render_overlay(
            img, raw_lines,
            f"{stem} — Teste {name} ({kw['label_thresh']}, {kw['threshold_sum']}, "
            f"span={kw['candidate_span']}, iter={kw['max_iterations']}, "
            f"min_w={kw['min_line_width']}) -- {n_lines} linhas",
            out_path,
        )
        print(f"    [Render] {out_path.name}")
        results.append((name, kw, n_lines, nan_pct, dt, offset))

    # Tabela
    print(f"\n  {'='*70}")
    print(f"  TABELA RESUMO — {stem}")
    print(f"  {'='*70}")
    print(f"  {'Param':<18}{'A':>10}{'B':>10}{'C':>10}")
    keys = ["label_thresh", "threshold_sum", "candidate_span", "max_iterations", "min_line_width"]
    for k in keys:
        vals = [r[1].get(k) for r in results]
        print(f"  {k:<18}{vals[0]!s:>10}{vals[1]!s:>10}{vals[2]!s:>10}")
    print(f"  {'n_lines':<18}{results[0][2]:>10}{results[1][2]:>10}{results[2][2]:>10}")
    print(f"  {'NaN%':<18}"
          f"{results[0][3]:>9.1f}%{results[1][3]:>9.1f}%{results[2][3]:>9.1f}%")
    print(f"  {'tempo (s)':<18}{results[0][4]:>10.1f}{results[1][4]:>10.1f}{results[2][4]:>10.1f}")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(" Debug da extração — heatmap signal_prob + testes de parâmetros")
    print("=" * 78)

    # Passo 1: heatmap em IMG_1303 e IMG_1387
    for stem in ["IMG_1303", "IMG_1387"]:
        _process_heatmap(stem)

    # Passo 2 + 3: testes A/B/C em IMG_1387
    _process_param_tests("IMG_1387")
    return 0


if __name__ == "__main__":
    sys.exit(main())
