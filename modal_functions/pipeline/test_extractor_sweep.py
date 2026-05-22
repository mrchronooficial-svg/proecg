"""
Análise de sensibilidade do SignalExtractor.

Para cada imagem (IMG_1279, IMG_1303, IMG_1387) e cada parâmetro:
  • varia UM parâmetro por vez (resto fica no default do Stenhede)
  • mede NaN% (na região trimmed, sem contar margens vazias) e n_lines
  • renderiza overlay para o melhor valor de cada parâmetro
  • após todos os sweeps, roda combinação ótima e renderiza
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
    _ensure_vendor_on_path,
    _SIGNAL_CLASS,
    get_unet,
)

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
TARGETS = ["IMG_1279", "IMG_1303", "IMG_1387"]

DEFAULTS = {
    "label_thresh": 0.1,
    "threshold_sum": 10.0,
    "candidate_span": 10,
    "max_iterations": 4,
    "min_line_width": 30,
}

SWEEPS = {
    "label_thresh":   [0.01, 0.03, 0.05, 0.1, 0.2],
    "threshold_sum":  [1.0, 3.0, 5.0, 10.0, 20.0],
    "candidate_span": [5, 10, 15, 20, 30],
    "max_iterations": [2, 4, 6, 8],
    "min_line_width": [10, 15, 20, 30, 50],
}

TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# U-Net signal_prob (cache por imagem)
# ---------------------------------------------------------------------------

_signal_prob_cache: dict[str, np.ndarray] = {}


def _get_signal_prob(image_bgr: np.ndarray, key: str) -> np.ndarray:
    if key in _signal_prob_cache:
        return _signal_prob_cache[key]
    h_orig, w_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (image_rgb - image_rgb.min()) / max(image_rgb.max() - image_rgb.min(), 1e-8)
    max_side = max(h_orig, w_orig)
    if max_side > _DEFAULT_MAX_DIM:
        scale = _DEFAULT_MAX_DIM / float(max_side)
        new_h, new_w = int(round(h_orig * scale)), int(round(w_orig * scale))
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
    _signal_prob_cache[key] = sp_np
    return sp_np


# ---------------------------------------------------------------------------
# SignalExtractor com kwargs customizados
# ---------------------------------------------------------------------------

def _run_extractor(
    signal_prob: np.ndarray, **kwargs
) -> tuple[np.ndarray, int, int, float]:
    """Devolve (raw_lines_padded, x_offset, n_lines, nan_pct_trimmed_region).

    nan_pct é calculado SÓ na região trimmed (sem contar margens vazias)."""
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
        return np.full((0, W), np.nan, dtype=np.float64), 0, 0, 100.0

    lines_full = torch.stack(lines_list, dim=0)
    chk = lines_full.clone()
    chk[chk == 0] = float("nan")
    valid_cols = chk.nan_to_num(0.0).abs().sum(0) > 0
    nonzero = torch.nonzero(valid_cols, as_tuple=True)[0]
    offset = int(nonzero[0].item()) if nonzero.numel() > 0 else 0

    merged_list, _ = extractor.match_and_merge_lines(lines_full)
    if len(merged_list) == 0:
        return np.full((0, W), np.nan, dtype=np.float64), offset, 0, 100.0

    merged = torch.stack(merged_list, dim=0).cpu().numpy().astype(np.float64)
    n_lines, lw = merged.shape
    padded = np.full((n_lines, W), np.nan, dtype=np.float64)
    end = min(offset + lw, W)
    padded[:, offset:end] = merged[:, :end - offset]

    # NaN% na região trimmed
    valid_cols_total = (~np.isnan(padded)).any(axis=0)
    if not valid_cols_total.any():
        nan_pct = 100.0
    else:
        first = int(np.argmax(valid_cols_total))
        last = len(valid_cols_total) - int(np.argmax(valid_cols_total[::-1])) - 1
        trimmed = padded[:, first:last + 1]
        nan_pct = float(np.isnan(trimmed).mean() * 100.0)

    return padded, offset, n_lines, nan_pct


def _render_overlay(
    img: np.ndarray, raw_lines: np.ndarray, title: str, out_path: Path,
    blend: float = 0.4,
) -> None:
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-image sweep
# ---------------------------------------------------------------------------

def _sweep_image(
    stem: str,
) -> tuple[list[dict], dict[str, dict]]:
    """Devolve (all_results, baseline_result_per_image).

    all_results: [{stem, param, value, n_lines, nan_pct, time, skipped}, ...]
    baseline_result_per_image[stem]: {raw_lines, n_lines, nan_pct, time}
    """
    img_path = UNDIST_DIR / f"{stem}.png"
    if not img_path.exists():
        print(f"[ERRO] {img_path}")
        return [], {}
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] falha ao ler {img_path}")
        return [], {}

    print(f"\n{'=' * 78}")
    print(f"  IMAGE: {stem}.png  shape={img.shape}")
    print(f"{'=' * 78}")

    # Pré-computa signal_prob
    print("  pré-computando signal_prob (U-Net)...")
    t0 = time.perf_counter()
    sp = _get_signal_prob(img, stem)
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s")

    # Baseline run
    print(f"\n  Baseline (defaults): {DEFAULTS}")
    t0 = time.perf_counter()
    raw_lines_base, x_off_base, n_lines_base, nan_base = _run_extractor(sp, **DEFAULTS)
    dt_base = time.perf_counter() - t0
    print(f"    n_lines={n_lines_base}  NaN%={nan_base:.1f}%  tempo={dt_base:.1f}s")

    # Render baseline overlay
    out_baseline = OUT_DIR / f"{stem}_baseline.png"
    _render_overlay(
        img, raw_lines_base,
        f"{stem} — baseline (defaults Stenhede)",
        out_baseline,
    )
    print(f"    [Render] {out_baseline.name}")

    all_results: list[dict] = [
        {"stem": stem, "param": "BASELINE", "value": "default",
         "n_lines": n_lines_base, "nan_pct": nan_base, "time": dt_base,
         "skipped": False},
    ]

    # Sweeps por parâmetro
    best_per_param: dict[str, dict] = {}
    for param_name, values in SWEEPS.items():
        print(f"\n  --- Sweep: {param_name} ---")
        param_results = []
        for v in values:
            kwargs = DEFAULTS.copy()
            kwargs[param_name] = v
            t0 = time.perf_counter()
            try:
                raw_lines, x_off, n_lines, nan_pct = _run_extractor(sp, **kwargs)
                dt = time.perf_counter() - t0
                skipped = dt > TIMEOUT_S
                if skipped:
                    print(f"    {param_name}={v}: SKIP (>{TIMEOUT_S}s, {dt:.1f}s)")
                    raw_lines = None
                else:
                    print(f"    {param_name}={v}: n_lines={n_lines} "
                          f"NaN%={nan_pct:.1f}% tempo={dt:.1f}s")
            except Exception as e:
                dt = time.perf_counter() - t0
                print(f"    {param_name}={v}: ERRO {type(e).__name__}: {e}")
                n_lines = 0
                nan_pct = float("nan")
                skipped = True
                raw_lines = None
            entry = {
                "stem": stem, "param": param_name, "value": v,
                "n_lines": n_lines, "nan_pct": nan_pct, "time": dt,
                "skipped": skipped,
            }
            all_results.append(entry)
            param_results.append({**entry, "raw_lines": raw_lines})

        # Encontra o melhor (menor NaN%, ignorando skipped)
        valid = [
            r for r in param_results
            if not r["skipped"] and not np.isnan(r["nan_pct"])
        ]
        if not valid:
            continue
        best = min(valid, key=lambda r: (r["nan_pct"], r["time"]))
        best_per_param[param_name] = {
            "value": best["value"], "nan_pct": best["nan_pct"],
            "time": best["time"], "raw_lines": best["raw_lines"],
        }
        # Render best
        v = best["value"]
        out_best = OUT_DIR / f"{stem}_best_{param_name}.png"
        _render_overlay(
            img, best["raw_lines"],
            f"{stem} — best {param_name}={v} "
            f"(NaN%={best['nan_pct']:.1f}, t={best['time']:.1f}s)",
            out_best,
        )
        print(f"    -> melhor: {param_name}={v} (NaN%={best['nan_pct']:.1f}%) "
              f"[Render] {out_best.name}")

    return all_results, best_per_param


# ---------------------------------------------------------------------------
# Combinação ótima
# ---------------------------------------------------------------------------

def _run_optimized(stem: str, optimal: dict) -> None:
    img_path = UNDIST_DIR / f"{stem}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        return
    sp = _get_signal_prob(img, stem)
    print(f"\n  Otimizado em {stem}: {optimal}")
    t0 = time.perf_counter()
    try:
        raw_lines, x_off, n_lines, nan_pct = _run_extractor(sp, **optimal)
        dt = time.perf_counter() - t0
        print(f"    n_lines={n_lines}  NaN%={nan_pct:.1f}%  tempo={dt:.1f}s")
    except Exception as e:
        print(f"    [ERRO] {type(e).__name__}: {e}")
        return
    out = OUT_DIR / f"{stem}_overlay_optimized.png"
    _render_overlay(
        img, raw_lines,
        f"{stem} — Combinação ótima (NaN%={nan_pct:.1f}, t={dt:.1f}s)",
        out,
    )
    print(f"    [Render] {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" Sweep de sensibilidade do SignalExtractor")
    print("=" * 78)

    all_results: list[dict] = []
    best_by_image: dict[str, dict[str, dict]] = {}
    for stem in TARGETS:
        results, best_pp = _sweep_image(stem)
        all_results.extend(results)
        best_by_image[stem] = best_pp

    # Tabela 1: todos os resultados
    print(f"\n{'=' * 92}")
    print(" TABELA 1 — Todos os resultados")
    print(f"{'=' * 92}")
    print(f"  {'Imagem':<10}{'Parâmetro':<18}{'Valor':>10}"
          f"{'NaN%':>10}{'n_lines':>10}{'tempo':>10}")
    print("  " + "-" * 70)
    for r in all_results:
        skip_marker = " (SKIP)" if r["skipped"] else ""
        nan_s = f"{r['nan_pct']:.1f}" if not np.isnan(r['nan_pct']) else "-"
        print(f"  {r['stem']:<10}{r['param']:<18}{str(r['value']):>10}"
              f"{nan_s:>10}{r['n_lines']:>10}{r['time']:>10.1f}{skip_marker}")

    # Tabela 2: melhor por imagem por parâmetro
    print(f"\n{'=' * 92}")
    print(" TABELA 2 — Melhor valor por (imagem × parâmetro)")
    print(f"{'=' * 92}")
    print(f"  {'Imagem':<10}{'Parâmetro':<18}{'Default':>10}{'Best':>10}"
          f"{'NaN% best':>12}")
    print("  " + "-" * 60)
    for stem, best_pp in best_by_image.items():
        for pname, info in best_pp.items():
            print(f"  {stem:<10}{pname:<18}{str(DEFAULTS[pname]):>10}"
                  f"{str(info['value']):>10}{info['nan_pct']:>11.1f}%")

    # Tabela 3: recomendação geral (consenso entre as 3 imagens)
    # Pegar o valor que MAIS aparece como "best" nas 3 imagens; em empate,
    # menor NaN% médio
    print(f"\n{'=' * 92}")
    print(" TABELA 3 — Recomendação consensual (entre as 3 imagens)")
    print(f"{'=' * 92}")
    print(f"  {'Parâmetro':<18}{'Default':>10}{'Recomendado':>15}"
          f"{'Δ NaN%':>10}{'Risco':<35}")
    print("  " + "-" * 80)

    risks = {
        "label_thresh": "menor = mais ruído tinta capturado",
        "threshold_sum": "menor = aceita ilhas pequenas (false positives)",
        "candidate_span": "maior = pula com mais facilidade de derivação",
        "max_iterations": "maior = só custo de tempo",
        "min_line_width": "menor = aceita fragmentos ruidosos",
    }

    optimal_combo: dict = dict(DEFAULTS)
    for pname in SWEEPS.keys():
        # Coleta "best value" de cada imagem
        votes = []
        nan_per_value: dict = {}
        for stem in TARGETS:
            if pname in best_by_image.get(stem, {}):
                v = best_by_image[stem][pname]["value"]
                votes.append(v)
                nan_per_value.setdefault(v, []).append(
                    best_by_image[stem][pname]["nan_pct"]
                )
        if not votes:
            continue
        # Frequência
        from collections import Counter
        freq = Counter(votes)
        max_count = max(freq.values())
        candidates = [v for v, c in freq.items() if c == max_count]
        # Em empate, menor NaN% médio
        if len(candidates) > 1:
            best_v = min(
                candidates,
                key=lambda v: float(np.mean(nan_per_value[v])),
            )
        else:
            best_v = candidates[0]
        # Δ NaN% médio (best vs baseline)
        nan_best = float(np.mean(nan_per_value[best_v]))
        # Baseline (parametro = default em SWEEPS) — pega das tabelas
        baseline_nans: list[float] = []
        for r in all_results:
            if r["param"] == pname and r["value"] == DEFAULTS[pname]:
                if not np.isnan(r["nan_pct"]):
                    baseline_nans.append(r["nan_pct"])
        nan_default = float(np.mean(baseline_nans)) if baseline_nans else float("nan")
        delta = nan_best - nan_default

        risk = risks.get(pname, "")
        print(f"  {pname:<18}{str(DEFAULTS[pname]):>10}{str(best_v):>15}"
              f"{delta:>+9.1f}%  {risk}")
        optimal_combo[pname] = best_v

    # Tabela 4: combinação final
    print(f"\n{'=' * 60}")
    print(" TABELA 4 — Combinação final recomendada")
    print(f"{'=' * 60}")
    for k, v in optimal_combo.items():
        print(f"  {k:<18}{str(v):>15}")

    # Roda combinação ótima nas 3 imagens
    print(f"\n{'=' * 78}")
    print(" Rodando combinação ÓTIMA nas 3 imagens")
    print(f"{'=' * 78}")
    for stem in TARGETS:
        _run_optimized(stem, optimal_combo)

    print(f"\nPNGs salvos em: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
