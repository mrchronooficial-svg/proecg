"""
Diagnóstico de overlay + delineação NeuroKit2 sobre IMG_1303.
=============================================================

PARTE 1: diagnostica e corrige o desalinhamento do overlay.
PARTE 2: delineação NK2 (P, QRS, T + intervalos PR/QRS/QT/RR) em DII
         desenhada sobre a imagem.
PARTE 3: overlay com ondas coloridas em TODOS os leads.

Saídas:
  modal_functions/pipeline/digitize/_visualizations/
    IMG_1303_overlay_corrigido.png
    IMG_1303_DII_delineacao.png
    IMG_1303_overlay_ondas_coloridas.png

Uso:
    python -m modal_functions.pipeline.test_stenhede_delineate
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

# Garantir UTF-8 no stdout (Windows cp1252 não aceita "->" / "—" / ...)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import torch

from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.stenhede_adapter import (
    LEAD_CHANNEL_ORDER,
    _get_signal_extractor,
    _ensure_vendor_on_path,
    extract_signals_stenhede,
    get_unet,
    get_unet_feature_maps,
)

UNDIST_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted\IMG_1303.png")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calibrate_normalized(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    _, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        _, keypoints = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _run_unet_with_diagnostics(
    image_bgr: np.ndarray, max_dim: int = 3000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Roda a U-Net retornando (signal_prob_at_resized, signal_prob_at_orig, info)."""
    h_orig, w_orig = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (image_rgb - image_rgb.min()) / max(image_rgb.max() - image_rgb.min(), 1e-8)

    max_side = max(h_orig, w_orig)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_h, new_w = int(round(h_orig * scale)), int(round(w_orig * scale))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        new_h, new_w = h_orig, w_orig
        img_resized = img

    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
    model = get_unet()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        signal_at_resized = probs[0, 2].cpu().numpy().astype(np.float32)
    # process_sparse_prob
    sp = signal_at_resized
    sp = sp - sp.mean(); sp = np.clip(sp, 0, None); sp = sp / max(sp.max(), 1e-9)
    signal_at_resized = sp

    # Resize back to original
    signal_at_orig = cv2.resize(
        signal_at_resized, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR,
    )
    info = {
        "h_orig": h_orig, "w_orig": w_orig,
        "h_feat": new_h, "w_feat": new_w,
        "scale": scale,
    }
    return signal_at_resized, signal_at_orig, info


def _signal_extractor_lines(prob_map: np.ndarray) -> np.ndarray:
    """Roda SignalExtractor numa prob (H, W) e devolve linhas (N, W) em pixel-Y."""
    extractor = _get_signal_extractor()
    fmap = torch.from_numpy(np.ascontiguousarray(prob_map)).float()
    lines = extractor(fmap)
    return lines.cpu().numpy().astype(np.float64)


def _render_overlay(
    image_bgr: np.ndarray,
    lines: np.ndarray,
    title: str,
    out_path: Path,
    blend_white: float = 0.40,
    color: tuple = (0.85, 0, 0),
    lw: float = 1.4,
) -> None:
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = (1 - blend_white) * img_rgb + blend_white
    img_blend = np.clip(img_blend, 0, 1)
    fig_w = max(16.0, w / 200.0); fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    if lines.ndim == 2 and lines.size > 0:
        x = np.arange(lines.shape[1])
        for i in range(lines.shape[0]):
            ax.plot(x, lines[i], color=color, lw=lw, alpha=0.85, zorder=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PARTE 1 — diagnóstico + overlay corrigido
# ---------------------------------------------------------------------------

def part1_diagnose_and_fix(image_bgr: np.ndarray) -> dict:
    print("\n" + "=" * 78)
    print(" PARTE 1 — Diagnóstico do overlay")
    print("=" * 78)

    h_orig, w_orig = image_bgr.shape[:2]
    print(f"\nDimensões da imagem undistorted: H={h_orig}, W={w_orig}")

    # Roda U-Net retornando o feature map nas DUAS resoluções
    sp_resized, sp_orig, info = _run_unet_with_diagnostics(image_bgr)
    print(f"Feature map U-Net (resized): H={info['h_feat']}, W={info['w_feat']}")
    print(f"Feature map (resize-back original): H={sp_orig.shape[0]}, W={sp_orig.shape[1]}")
    print(f"Scale factor (orig -> feat): {info['scale']:.4f}")

    # SignalExtractor nos dois — pra comparar
    print("\n[A] SignalExtractor no feature map RESIZED (resolução U-Net nativa):")
    lines_resized = _signal_extractor_lines(sp_resized)
    print(f"    shape: {lines_resized.shape}, "
          f"Y range: [{np.nanmin(lines_resized):.1f}, {np.nanmax(lines_resized):.1f}] "
          f"(esperado: 0..{info['h_feat']})")

    print("\n[B] SignalExtractor no feature map RESIZED-BACK (resolução original):")
    lines_orig = _signal_extractor_lines(sp_orig)
    print(f"    shape: {lines_orig.shape}, "
          f"Y range: [{np.nanmin(lines_orig):.1f}, {np.nanmax(lines_orig):.1f}] "
          f"(esperado: 0..{h_orig})")

    if lines_orig.shape[0] > 0:
        print(f"\nPrimeiros 20 valores Y da linha 0 (resize-back):")
        first20 = lines_orig[0, :20]
        print(f"    {[f'{y:.1f}' for y in first20.tolist()]}")
    if lines_resized.shape[0] > 0:
        print(f"Primeiros 20 valores Y da linha 0 (resized native):")
        first20 = lines_resized[0, :20]
        print(f"    {[f'{y:.1f}' for y in first20.tolist()]}")

    # Verifica alinhamento sampling a darkness da imagem original nos pontos
    print("\n[Validação] Em que posição cada linha está mais escura na imagem original?")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _check_darkness_offset(lines_local: np.ndarray, label: str) -> None:
        """Para cada linha, varre Y_offset em [-15, +15] e mede a soma de
        intensidade-inversa nas posições. O offset que maximiza darkness =
        offset correto."""
        if lines_local.shape[0] == 0:
            return
        # Mapeia Y do feature map -> Y da imagem original
        # Linhas em coords da imagem original (após scale-back)
        # Para o feature map RESIZED, multiplicar Y por (h_orig / h_feat)
        if label == "resized_native":
            lines_in_orig = lines_local * (h_orig / info["h_feat"])
        else:
            lines_in_orig = lines_local
        for li in range(lines_in_orig.shape[0]):
            y_pred = lines_in_orig[li]
            valid = ~np.isnan(y_pred)
            if valid.sum() < 100:
                continue
            xs = np.arange(len(y_pred))[valid]
            ys = y_pred[valid]
            offsets = np.arange(-15, 16)
            scores = []
            for off in offsets:
                ys_off = np.clip(ys + off, 0, h_orig - 1).astype(int)
                xs_clip = np.clip(xs, 0, w_orig - 1).astype(int)
                # darkness = 255 - intensidade
                pixels = gray[ys_off, xs_clip]
                scores.append(float((255.0 - pixels).sum()))
            best_off = int(offsets[int(np.argmax(scores))])
            ratio_at_zero = scores[len(offsets) // 2] / max(scores), 1
            print(
                f"  [{label}] Linha {li}: "
                f"melhor Y_offset = {best_off:+d} px "
                f"(score 0={scores[len(offsets)//2]:.0f}, "
                f"score best={max(scores):.0f}, "
                f"melhora={(max(scores)/max(scores[len(offsets)//2],1)-1)*100:.1f}%)"
            )

    _check_darkness_offset(lines_resized, "resized_native")
    _check_darkness_offset(lines_orig, "resize_back_orig")

    # Salva overlay com a versão escolhida (resize-back, que é o que
    # extract_signals_stenhede já usa)
    out_path = OUT_DIR / "IMG_1303_overlay_corrigido.png"
    _render_overlay(
        image_bgr, lines_orig,
        "IMG_1303 — Overlay corrigido (SignalExtractor sobre feature map em resolução original)",
        out_path,
    )
    print(f"\n[Render] {out_path.name}")

    return {
        "lines_orig": lines_orig,
        "lines_resized": lines_resized,
        "info": info,
        "signal_prob_orig": sp_orig,
    }


# ---------------------------------------------------------------------------
# PARTE 2 — Delineação NeuroKit2 em DII com brackets de intervalos
# ---------------------------------------------------------------------------

WAVE_COLORS = {
    "P": "#1f77b4",       # azul
    "QRS": "#2ca02c",     # verde
    "T": "#ff7f0e",       # laranja
    "baseline": "#7f7f7f",  # cinza
}


def _delineate_signal(sig_uv: np.ndarray, fs: float) -> dict | None:
    """Roda NK2 ecg_clean -> ecg_peaks -> ecg_delineate(method='dwt')."""
    if sig_uv is None or sig_uv.size == 0:
        return None
    valid = ~np.isnan(sig_uv)
    if valid.sum() < int(2 * fs):  # menos de 2 s
        return None
    # NK2 funciona com mV/qualquer escala — vamos usar mV pra estabilidade
    sig_mv = np.where(valid, sig_uv / 1000.0, 0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ecg_clean(sig_mv, sampling_rate=int(fs))
            _, rpeaks_d = nk.ecg_peaks(cleaned, sampling_rate=int(fs))
            rpeaks = rpeaks_d.get("ECG_R_Peaks", [])
            if len(rpeaks) < 2:
                return None
            _, waves = nk.ecg_delineate(
                cleaned, rpeaks, sampling_rate=int(fs), method="dwt",
            )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "NK delineate falhou: %s: %s", type(e).__name__, e,
        )
        return None

    return {
        "cleaned": cleaned,
        "rpeaks": list(rpeaks),
        "waves": waves,  # dict ECG_*: list de indices
        "valid_mask": valid,
    }


def _segment_color_per_sample(
    n: int, waves: dict, rpeaks: list,
) -> np.ndarray:
    """Devolve array (n,) com a cor de cada sample: P, QRS, T, baseline.

    Convenção:
      [P_Onsets..P_Offsets] = "P"
      [R_Onsets..R_Offsets] = "QRS"
      [T_Onsets..T_Offsets] = "T"
      caso contrário        = "baseline"
    """
    seg = np.full(n, "baseline", dtype=object)

    def _fill(starts_key: str, ends_key: str, label: str) -> None:
        starts = waves.get(starts_key, [])
        ends = waves.get(ends_key, [])
        for s, e in zip(starts, ends):
            if s is None or e is None:
                continue
            try:
                si = int(s); ei = int(e)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(si) and np.isfinite(ei)):
                continue
            if si < 0 or ei < 0 or ei <= si:
                continue
            si = max(0, min(n - 1, si))
            ei = max(0, min(n, ei + 1))
            seg[si:ei] = label

    _fill("ECG_P_Onsets", "ECG_P_Offsets", "P")
    _fill("ECG_R_Onsets", "ECG_R_Offsets", "QRS")
    _fill("ECG_T_Onsets", "ECG_T_Offsets", "T")
    return seg


def part2_dii_delineation(
    full_result: dict,
    image_bgr: np.ndarray,
    fs: float,
) -> None:
    print("\n" + "=" * 78)
    print(" PARTE 2 — Delineação NK2 em DII")
    print("=" * 78)
    sig_dii = full_result["signals"].get("II")
    if sig_dii is None or sig_dii.size == 0:
        print("[ERRO] DII não disponível no output Stenhede")
        return
    print(f"DII: {len(sig_dii)} samples, NaN={np.isnan(sig_dii).mean()*100:.1f}%, "
          f"range=[{np.nanmin(sig_dii):+.0f}, {np.nanmax(sig_dii):+.0f}] uV")

    deli = _delineate_signal(sig_dii, fs)
    if deli is None:
        print("[ERRO] NK delineate falhou em DII — pulando")
        return
    rpeaks = deli["rpeaks"]
    waves = deli["waves"]
    print(f"R-peaks: {len(rpeaks)} | "
          f"P-onsets: {sum(1 for x in waves.get('ECG_P_Onsets', []) if x is not None and not np.isnan(x))} "
          f"| T-offsets: {sum(1 for x in waves.get('ECG_T_Offsets', []) if x is not None and not np.isnan(x))}")

    # Plot
    n = len(sig_dii)
    t = np.arange(n) / fs   # segundos
    sig_mv = sig_dii / 1000.0  # mV

    # Pega 3 batimentos centrais pra zoom
    if len(rpeaks) >= 3:
        center_r = rpeaks[len(rpeaks) // 2]
        rr_med = int(np.median(np.diff(rpeaks))) if len(rpeaks) >= 2 else int(0.8 * fs)
        x_start = max(0, int(center_r - 1.4 * rr_med))
        x_end = min(n, int(center_r + 1.6 * rr_med))
    else:
        x_start, x_end = 0, n

    fig, ax = plt.subplots(figsize=(20, 7), facecolor="white")
    seg_colors = _segment_color_per_sample(n, waves, rpeaks)

    # Plota segmentos coloridos
    for label, color in WAVE_COLORS.items():
        mask = seg_colors == label
        if not np.any(mask):
            continue
        # split em runs
        ys = np.where(mask, sig_mv, np.nan)
        ax.plot(t, ys, color=color,
                lw=2.0 if label != "baseline" else 1.0,
                label=label if label != "baseline" else None,
                zorder=3)

    # Picos
    def _scatter(idx_list, color, marker, size, label_pt):
        xs, ys = [], []
        for i in idx_list:
            if i is None: continue
            try:
                ii = int(i)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ii) or ii < 0 or ii >= n:
                continue
            xs.append(t[ii]); ys.append(sig_mv[ii])
        if xs:
            ax.scatter(xs, ys, color=color, marker=marker, s=size,
                       zorder=5, edgecolors="black", linewidths=0.4,
                       label=label_pt)

    _scatter(waves.get("ECG_P_Peaks", []), "#1f77b4", "o", 60, "P peak")
    _scatter(waves.get("ECG_Q_Peaks", []), "#1a5d1a", "o", 60, "Q peak")
    _scatter(rpeaks, "red", "o", 100, "R peak")
    _scatter(waves.get("ECG_S_Peaks", []), "#7be07b", "o", 60, "S peak")
    _scatter(waves.get("ECG_T_Peaks", []), "#ff7f0e", "o", 60, "T peak")
    _scatter(waves.get("ECG_P_Onsets", []), "#1f77b4", "^", 50, "P on")
    _scatter(waves.get("ECG_P_Offsets", []), "#1f77b4", "v", 50, "P off")
    _scatter(waves.get("ECG_R_Onsets", []), "#2ca02c", "^", 50, "QRS on")
    _scatter(waves.get("ECG_R_Offsets", []), "#2ca02c", "v", 50, "QRS off (J)")
    _scatter(waves.get("ECG_T_Offsets", []), "#ff7f0e", "v", 50, "T off")

    # Brackets de intervalos abaixo do traçado
    y_min = float(np.nanmin(sig_mv[x_start:x_end]) if x_end > x_start else np.nanmin(sig_mv))
    y_max = float(np.nanmax(sig_mv[x_start:x_end]) if x_end > x_start else np.nanmax(sig_mv))
    y_range = max(0.5, y_max - y_min)
    bracket_y_pr = y_min - 0.20 * y_range
    bracket_y_qrs = y_min - 0.34 * y_range
    bracket_y_qt = y_min - 0.48 * y_range
    bracket_y_rr = y_min - 0.62 * y_range
    bracket_y_st = y_min - 0.07 * y_range  # ST entre J e T-onset, próximo da curva

    def _bracket(x1: float, x2: float, y: float, label: str, color: str,
                 below: bool = True) -> None:
        if not (np.isfinite(x1) and np.isfinite(x2)) or x2 <= x1:
            return
        # linha horizontal
        ax.plot([x1, x2], [y, y], color=color, lw=1.5, zorder=4)
        # ticks verticais
        tick_h = y_range * 0.025
        tick_dir = -1 if below else 1
        ax.plot([x1, x1], [y, y + tick_dir * tick_h], color=color, lw=1.5, zorder=4)
        ax.plot([x2, x2], [y, y + tick_dir * tick_h], color=color, lw=1.5, zorder=4)
        # texto
        ax.text((x1 + x2) / 2, y - tick_h * 1.5, label,
                ha="center", va="top" if below else "bottom",
                fontsize=8, color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    # Para cada batimento dentro da zoom window, desenha brackets
    p_ons = [i for i in waves.get("ECG_P_Onsets", []) if i is not None and np.isfinite(i)]
    qrs_ons = [i for i in waves.get("ECG_R_Onsets", []) if i is not None and np.isfinite(i)]
    qrs_offs = [i for i in waves.get("ECG_R_Offsets", []) if i is not None and np.isfinite(i)]
    t_offs = [i for i in waves.get("ECG_T_Offsets", []) if i is not None and np.isfinite(i)]
    t_ons = [i for i in waves.get("ECG_T_Onsets", []) if i is not None and np.isfinite(i)]

    for i_beat, r in enumerate(rpeaks):
        if r < x_start or r > x_end:
            continue
        # Encontra o P-onset/QRS-onset/QRS-offset/T-offset associados a este r
        # (mais próximos antes/depois)
        def _nearest(vals, target, before=True):
            for v in vals:
                vi = int(v)
                if before and vi <= target:
                    candidates = [int(x) for x in vals if int(x) <= target]
                    if candidates:
                        return max(candidates)
                if not before and vi >= target:
                    candidates = [int(x) for x in vals if int(x) >= target]
                    if candidates:
                        return min(candidates)
            return None
        p_on = _nearest(p_ons, r, before=True)
        q_on = _nearest(qrs_ons, r, before=True)
        q_off = _nearest(qrs_offs, r, before=False)
        t_off = _nearest(t_offs, r, before=False)
        t_on = _nearest(t_ons, r, before=False)

        # PR
        if p_on is not None and q_on is not None:
            pr_ms = (q_on - p_on) / fs * 1000
            _bracket(t[p_on], t[q_on], bracket_y_pr,
                     f"PR: {pr_ms:.0f}ms", "#1f77b4")
        # QRS
        if q_on is not None and q_off is not None:
            qrs_ms = (q_off - q_on) / fs * 1000
            _bracket(t[q_on], t[q_off], bracket_y_qrs,
                     f"QRS: {qrs_ms:.0f}ms", "#2ca02c")
        # QT
        if q_on is not None and t_off is not None:
            qt_ms = (t_off - q_on) / fs * 1000
            _bracket(t[q_on], t[t_off], bracket_y_qt,
                     f"QT: {qt_ms:.0f}ms", "#9467bd")
        # ST
        if q_off is not None and t_on is not None:
            _bracket(t[q_off], t[t_on], bracket_y_st,
                     "ST", "#ff7f0e", below=False)
        # RR
        if i_beat + 1 < len(rpeaks) and rpeaks[i_beat + 1] <= x_end:
            r_next = rpeaks[i_beat + 1]
            rr_ms = (r_next - r) / fs * 1000
            hr = 60000 / max(rr_ms, 1)
            _bracket(t[r], t[r_next], bracket_y_rr,
                     f"RR: {rr_ms:.0f}ms (FC: {hr:.0f} bpm)", "red")

    # Eixos / legenda
    ax.set_xlim(t[x_start], t[max(x_start + 1, x_end - 1)])
    y_extent = bracket_y_rr - 0.05 * y_range
    ax.set_ylim(y_extent, y_max + 0.05 * y_range)
    ax.set_xlabel("tempo (s)", fontsize=10)
    ax.set_ylabel("mV", fontsize=10)
    ax.grid(True, which="major", color="lightcoral", alpha=0.3)
    ax.set_title(
        "ProECG — IMG_1303 — DII — Delineação NeuroKit2",
        fontsize=13, fontweight="bold",
    )

    # Legenda customizada
    legend_handles = [
        mpatches.Patch(color="#1f77b4", label="Onda P (azul)"),
        mpatches.Patch(color="#2ca02c", label="Complexo QRS (verde)"),
        mpatches.Patch(color="#ff7f0e", label="Onda T (laranja)"),
        mpatches.Patch(color="#7f7f7f", label="Baseline (cinza)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=10, label="R peak"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9)

    out_path = OUT_DIR / "IMG_1303_DII_delineacao.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Render] {out_path.name}")


# ---------------------------------------------------------------------------
# PARTE 3 — Overlay completo com ondas coloridas em todos os leads
# ---------------------------------------------------------------------------

def _pad_lines_to_full_width(
    raw_lines: np.ndarray, signal_prob: np.ndarray, w_orig: int,
) -> tuple[np.ndarray, int]:
    """SignalExtractor.preprocess_lines TRIMA as colunas inválidas dos
    extremos. Reconstrói (N, W_orig) com NaN antes/depois usando
    `signal_prob` pra detectar onde começa o conteúdo.

    Returns:
        (padded_lines, x_offset) — `padded_lines[:, x_offset:x_offset+L]`
        contém os dados da linha; resto é NaN.
    """
    n, lw = raw_lines.shape
    if lw >= w_orig:
        return raw_lines[:, :w_orig], 0

    # Heurística: detectar primeira coluna do feature map com massa de prob
    col_active = signal_prob.sum(axis=0)
    thr = max(col_active.max() * 0.02, 1e-3)
    valid = col_active > thr
    if valid.any():
        first = int(np.argmax(valid))
    else:
        first = 0
    # Garante que cabe
    first = max(0, min(first, w_orig - lw))

    padded = np.full((n, w_orig), np.nan, dtype=raw_lines.dtype)
    padded[:, first : first + lw] = raw_lines
    return padded, first


def part3_full_overlay_colored(
    full_result: dict,
    image_bgr: np.ndarray,
    fs: float,
    raw_lines_pixel: np.ndarray,
    signal_prob: np.ndarray,
) -> None:
    print("\n" + "=" * 78)
    print(" PARTE 3 — Overlay colorido por onda em todos os leads")
    print("=" * 78)

    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_blend = 0.6 * img_rgb + 0.4
    img_blend = np.clip(img_blend, 0, 1)

    # PAD raw_lines_pixel pra largura original (SignalExtractor trima)
    padded_lines, x_offset = _pad_lines_to_full_width(
        raw_lines_pixel, signal_prob, w,
    )
    print(f"raw_lines_pixel: shape={raw_lines_pixel.shape} -> padded={padded_lines.shape} "
          f"(x_offset={x_offset})")

    fig_w = max(20.0, w / 200.0); fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.imshow(img_blend, interpolation="nearest")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])

    color_map = {
        "P": "#1f77b4",
        "QRS": "#2ca02c",
        "T": "#ff7f0e",
        "baseline": "#cc0000",
    }
    LAYOUT = [
        ["I",  "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III","aVF", "V3", "V6"],
    ]

    n_lines = padded_lines.shape[0]
    if n_lines == 0:
        print("[ERRO] Nenhuma linha extraída — pulando")
        return

    rhythm_present = n_lines >= 4
    n_main_rows = min(3, n_lines if not rhythm_present else n_lines - 1)
    failed_leads: list[str] = []

    def _try_delineate(sig_mv: np.ndarray) -> tuple[list, dict] | None:
        """Wrapper que retorna None se NK falhar ou faltar dados."""
        valid = ~np.isnan(sig_mv) & ~np.isinf(sig_mv)
        if valid.sum() < int(2.0 * fs):
            return None
        sig_clean = np.where(valid, sig_mv, 0.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cleaned = nk.ecg_clean(sig_clean, sampling_rate=int(fs))
                _, rd = nk.ecg_peaks(cleaned, sampling_rate=int(fs))
                rpeaks = list(rd.get("ECG_R_Peaks", []))
                if len(rpeaks) < 1:
                    return None
                _, waves = nk.ecg_delineate(
                    cleaned, rpeaks, sampling_rate=int(fs), method="dwt",
                )
            return rpeaks, waves
        except Exception:
            return None

    def _plot_segments_on_line(
        line_y: np.ndarray, x_start: int, seg_colors: np.ndarray,
        rpeaks_local: list,
    ) -> None:
        """Plota a parte da `line_y` entre x_start e x_start+len(seg_colors)
        colorida por seg_colors."""
        n_seg = len(seg_colors)
        x_end = min(x_start + n_seg, line_y.shape[0])
        if x_end <= x_start:
            return
        n_eff = x_end - x_start
        xs = np.arange(x_start, x_end)
        ys = line_y[x_start:x_end]
        seg = seg_colors[:n_eff]
        for label, color in color_map.items():
            mask = seg == label
            if not np.any(mask):
                continue
            ys_seg = np.where(mask, ys, np.nan)
            ax.plot(
                xs, ys_seg, color=color,
                lw=1.6 if label != "baseline" else 1.1,
                alpha=0.92, zorder=3,
            )
        for rp in rpeaks_local:
            try:
                rp_i = int(rp)
            except (TypeError, ValueError):
                continue
            if 0 <= rp_i < n_eff and not np.isnan(line_y[x_start + rp_i]):
                ax.scatter(
                    [x_start + rp_i], [line_y[x_start + rp_i]],
                    color="red", s=30, zorder=5,
                    edgecolors="black", linewidths=0.4,
                )

    # Para cada cell, divide a row em 4 chunks. O canonical signal pra
    # esse lead já está em [chunk_x_start, chunk_x_end) — usamos ele pra
    # determinar P/QRS/T. O Y desenhado vem da padded_lines do row.
    chunk_w = w // 4
    for row_idx in range(n_main_rows):
        line = padded_lines[row_idx]
        names = LAYOUT[row_idx]
        for c, name in enumerate(names):
            x_start = c * chunk_w
            x_end = (c + 1) * chunk_w if c < 3 else w
            sig_uv = full_result["signals"].get(name)
            if sig_uv is None or x_end > len(sig_uv):
                # Ainda assim plota em cinza pra visualizar
                xs_c = np.arange(x_start, x_end)
                line_c = line[x_start:x_end]
                ax.plot(xs_c, line_c, color="gray", lw=0.8, alpha=0.6)
                failed_leads.append(f"{name}(sem sinal)")
                continue
            seg_canon = sig_uv[x_start:x_end] / 1000.0  # mV
            res = _try_delineate(seg_canon)
            if res is None:
                xs_c = np.arange(x_start, x_end)
                line_c = line[x_start:x_end]
                ax.plot(xs_c, line_c, color="gray", lw=0.8, alpha=0.6)
                failed_leads.append(f"{name}(NK falhou)")
                continue
            rpeaks_local, waves_local = res
            seg_colors = _segment_color_per_sample(
                len(seg_canon), waves_local, rpeaks_local,
            )
            _plot_segments_on_line(line, x_start, seg_colors, rpeaks_local)

    # Rhythm strip
    if rhythm_present and n_lines >= 4:
        line_r = padded_lines[-1]
        rhy_uv = full_result["signals"].get(
            "II_rhythm", full_result["signals"].get("II"),
        )
        if rhy_uv is None:
            xs_full = np.arange(w)
            ax.plot(xs_full, line_r, color="gray", lw=0.8, alpha=0.6)
            failed_leads.append("II_rhythm(sem sinal)")
        else:
            n_uv = min(len(rhy_uv), w)
            res = _try_delineate(rhy_uv[:n_uv] / 1000.0)
            if res is None:
                xs_full = np.arange(w)
                ax.plot(xs_full, line_r, color="gray", lw=0.8, alpha=0.6)
                failed_leads.append("II_rhythm(NK falhou)")
            else:
                rpeaks_r, waves_r = res
                seg_colors_r = _segment_color_per_sample(
                    n_uv, waves_r, rpeaks_r,
                )
                _plot_segments_on_line(line_r, 0, seg_colors_r, rpeaks_r)

    # Legenda
    legend_handles = [
        mpatches.Patch(color="#1f77b4", label="Onda P"),
        mpatches.Patch(color="#2ca02c", label="QRS"),
        mpatches.Patch(color="#ff7f0e", label="Onda T"),
        mpatches.Patch(color="#cc0000", label="Baseline (vermelho)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=8, label="R peak"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10,
              framealpha=0.9)
    ax.set_title(
        "ProECG — IMG_1303 — Overlay com ondas coloridas (12 leads + rhythm)",
        fontsize=13, fontweight="bold",
    )

    out_path = OUT_DIR / "IMG_1303_overlay_ondas_coloridas.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Render] {out_path.name}")
    if failed_leads:
        print(f"\n[Aviso] Leads com falha de delineação (plotados em cinza):")
        for f in failed_leads:
            print(f"  - {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s",
    )
    if not UNDIST_PATH.exists():
        print(f"ERRO: {UNDIST_PATH} nao encontrada", file=sys.stderr)
        return 1

    img = cv2.imread(str(UNDIST_PATH))
    if img is None:
        print("ERRO: falha ao ler imagem", file=sys.stderr)
        return 2
    print(f"Imagem: {UNDIST_PATH} shape={img.shape}")

    cal = _calibrate_normalized(img)
    print(f"Calibracao: px/mm={cal['px_per_mm']:.3f} fs={cal['sampling_rate_hz']:.1f}Hz")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # PARTE 1
    diag = part1_diagnose_and_fix(img)

    # Roda extract_signals_stenhede pra obter signals em µV (canonical)
    print("\n[Stenhede] extract_signals_stenhede ...")
    t0 = time.perf_counter()
    full_result = extract_signals_stenhede(
        image_bgr=img, px_per_mm=float(cal["px_per_mm"]),
    )
    print(f"  pronto em {(time.perf_counter()-t0):.1f}s, "
          f"fs={full_result['sampling_rate_hz']:.1f}Hz, "
          f"layout={full_result['match'].get('layout')}")

    fs = float(full_result["sampling_rate_hz"])

    # PARTE 2
    part2_dii_delineation(full_result, img, fs)

    # PARTE 3
    part3_full_overlay_colored(
        full_result, img, fs, full_result["raw_lines_pixel"],
        signal_prob=diag["signal_prob_orig"],
    )

    print(f"\n[OK] Saídas em {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
