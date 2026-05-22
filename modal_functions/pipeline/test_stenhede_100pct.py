"""
Teste 100% Stenhede — Versões A (sem cropper) e B (com cropper).
================================================================

Usa o pipeline completo do Stenhede APÓS a imagem undistorted:
  • U-Net 4 canais (Stenhede)
  • (Versão B apenas) cropper.apply_perspective com cantos da imagem
  • PixelSizeFinder (Stenhede) → avg_px/mm
  • SignalExtractor (Stenhede) com x_offset detectado
  • LeadIdentifier (Stenhede) → 12 leads em µV

Compara `avg_pixel_per_mm` do PixelSizeFinder com o do nosso calibrator
(Dotter+Gridder) e renderiza overlay (linha vermelha sobre a foto).

Saídas em modal_functions/pipeline/digitize/_visualizations/:
  IMG_1378_overlay_100pct_stenhede_A.png
  IMG_1378_overlay_100pct_stenhede_B.png
  IMG_1303_overlay_100pct_stenhede_A.png
  IMG_1303_overlay_100pct_stenhede_B.png
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

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.stenhede_adapter import extract_signals_stenhede

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
TARGETS = ["IMG_1378", "IMG_1303"]


def _our_calibrator_px_per_mm(img: np.ndarray) -> float | None:
    """px/mm pelo nosso Dotter+Gridder+Calibrator (referência)."""
    try:
        digitizer = ECGDigitizer(use_mock=False)
        _, kps = digitizer.dotter(img)
        if len(kps) < 4:
            _, kps = digitizer.dotter_mock(img)
        grid_matrix, _ = digitizer.gridder(kps, img.shape[:2])
        cal = calibrate(grid_matrix=grid_matrix, normalized_image=img)
        return float(cal["px_per_mm"])
    except Exception as e:
        print(f"  [calibrator nosso] FALHOU: {type(e).__name__}: {e}")
        return None


def _check_alignment(image_bgr: np.ndarray, raw_padded: np.ndarray,
                     search_px: int = 30) -> tuple[int, float]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    valid_mask = ~np.isnan(raw_padded)
    dxs = np.arange(-search_px, search_px + 1)
    scores = []
    for dx in dxs:
        sc = 0.0
        for li in range(raw_padded.shape[0]):
            row_v = valid_mask[li]
            if row_v.sum() < 50:
                continue
            xs = np.where(row_v)[0]
            xs_s = xs + dx
            m = (xs_s >= 0) & (xs_s < w)
            xs_s = xs_s[m]; xs_use = xs[m]
            ys = raw_padded[li, xs_use]
            yi = np.clip(np.round(ys).astype(int), 0, h - 1)
            sc += float((255.0 - gray[yi, xs_s]).sum())
        scores.append(sc)
    sa = np.array(scores)
    best_i = int(np.argmax(sa))
    return int(dxs[best_i]), float(
        sa[best_i] / max(sa[search_px], 1.0) - 1.0,
    ) * 100.0


def _render(image_bgr: np.ndarray, raw_padded: np.ndarray,
            title: str, out_path: Path, blend: float = 0.40) -> None:
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (1 - blend) * img_rgb + blend
    img = np.clip(img, 0, 1)
    fig_w = max(16.0, w / 200.0); fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    if raw_padded.ndim == 2 and raw_padded.size > 0:
        x = np.arange(raw_padded.shape[1])
        for i in range(raw_padded.shape[0]):
            ax.plot(x, raw_padded[i], color=(0.85, 0, 0),
                    lw=1.5, alpha=0.85, zorder=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _process(stem: str, summary: list) -> None:
    img_path = UNDIST_DIR / f"{stem}.png"
    if not img_path.exists():
        print(f"\n[ERRO] {img_path} nao existe")
        return
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"\n[ERRO] falha ao ler {img_path}")
        return
    h, w = img.shape[:2]
    print(f"\n{'=' * 78}\n  {stem}.png  shape=(H={h}, W={w})\n{'=' * 78}")

    # Referência: px/mm do nosso calibrator (Dotter + Gridder)
    pxmm_nosso = _our_calibrator_px_per_mm(img)
    print(f"  px/mm pelo NOSSO calibrator (Dotter+Gridder): "
          f"{pxmm_nosso:.3f}" if pxmm_nosso else "  px/mm nosso: FALHOU")

    for version, use_cropper in [("A", False), ("B", True)]:
        print(f"\n  --- Versão {version} "
              f"(cropper={'ON' if use_cropper else 'OFF'}) ---")
        t0 = time.perf_counter()
        try:
            res = extract_signals_stenhede(
                image_bgr=img, px_per_mm=None,
                use_cropper=use_cropper,
                use_internal_pixel_size=True,
            )
        except Exception as e:
            print(f"  [ERRO] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
        dt = time.perf_counter() - t0

        avg_pxmm = res["avg_pixel_per_mm"]
        x_off = res["raw_lines_x_offset"]
        layout = res["match"].get("layout")
        cost = res["match"].get("cost", float("nan"))
        n_lines = res["n_lines_detected"]

        diff_pct = (
            (avg_pxmm - pxmm_nosso) / pxmm_nosso * 100.0
            if pxmm_nosso else float("nan")
        )
        print(f"    avg_px/mm Stenhede: {avg_pxmm:.3f} "
              f"(nosso: {pxmm_nosso:.3f}, diff: {diff_pct:+.1f}%)"
              if pxmm_nosso else f"    avg_px/mm Stenhede: {avg_pxmm:.3f}")
        print(f"    layout={layout} cost={cost:.2f} linhas={n_lines} "
              f"x_offset={x_off} ({dt:.1f}s)")

        raw_padded = res["raw_lines_pixel"]
        if n_lines > 0:
            best_dx, impr = _check_alignment(img, raw_padded)
            ok = abs(best_dx) <= 3
            print(f"    Alinhamento residual: dx={best_dx:+d} px "
                  f"(melhora {impr:.1f}%) {'OK' if ok else 'DRIFT'}")
        else:
            best_dx = 0; ok = False

        out_path = OUT_DIR / f"{stem}_overlay_100pct_stenhede_{version}.png"
        _render(img, raw_padded,
                f"{stem} -- Versão {version} "
                f"(cropper={'ON' if use_cropper else 'OFF'}, "
                f"avg_px/mm={avg_pxmm:.2f})",
                out_path)
        print(f"    [Render] {out_path.name}")

        summary.append({
            "img": stem, "version": version,
            "use_cropper": use_cropper,
            "pxmm_nosso": pxmm_nosso,
            "pxmm_stenhede": avg_pxmm,
            "diff_pct": diff_pct,
            "best_dx": best_dx, "aligned_ok": ok,
            "layout": layout, "cost": cost, "n_lines": n_lines,
        })


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(name)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(" TESTE 100% Stenhede — Versão A (sem cropper) vs B (com cropper)")
    print("=" * 78)
    summary: list = []
    for stem in TARGETS:
        _process(stem, summary)

    # Tabela final
    print(f"\n{'=' * 92}")
    print(" RESUMO COMPARATIVO")
    print(f"{'=' * 92}")
    print(f"{'IMG':<10}{'Ver':>5}{'Crop':>6}"
          f"{'pxmm_nosso':>13}{'pxmm_stnhd':>13}{'diff%':>9}"
          f"{'dx_px':>7}{'OK?':>5}{'layout':>22}{'cost':>7}")
    print("-" * 92)
    for r in summary:
        print(
            f"{r['img']:<10}{r['version']:>5}{'ON' if r['use_cropper'] else 'OFF':>6}"
            f"{r['pxmm_nosso']:>13.3f}{r['pxmm_stenhede']:>13.3f}"
            f"{r['diff_pct']:>8.1f}%"
            f"{r['best_dx']:>+7d}{('SIM' if r['aligned_ok'] else 'NAO'):>5}"
            f"{r['layout']:>22}{r['cost']:>7.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
