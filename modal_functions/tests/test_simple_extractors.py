"""
Teste de 4 extractors SIMPLES no IMG_1407 (cache pre-SignalExtractor).

Cada metodo gera 5 imagens em subfolder dedicado:
  benchmark_extractors/
    skeleton/        — skimage.skeletonize
    thinning/        — skimage.thin
    borda_superior/  — primeiro pixel non-zero por coluna
    media_bordas/    — (primeiro + ultimo) / 2 por coluna

REGRA INVIOLAVEL VALIDADA: imagem 04 (sinal sobre mascara) mostra pontos
fora da mascara como AMARELO. Devem ser ZERO.
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize, thin

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("test_simple")

CACHE_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407"
    r"\_cache_pre_signal_extractor.pkl"
)
OUTPUT_ROOT = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\benchmark_extractors"
)
THRESHOLD = 0.1


# =====================================================================
# Os 4 metodos de extracao
# =====================================================================

def extract_skeleton(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Esqueletoniza e le Y por coluna."""
    t = {}
    t0 = time.perf_counter()
    skel = skeletonize(mask > 0)
    t["skeletonize_ms"] = (time.perf_counter() - t0) * 1000

    H, W = mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    t0 = time.perf_counter()
    for x in range(W):
        ys = np.where(skel[:, x])[0]
        if ys.size == 0:
            continue
        elif ys.size == 1:
            signal[x] = float(ys[0])
        else:
            # Multiplos Y: pega o mais proximo do anterior valido
            if x > 0 and not np.isnan(signal[x - 1]):
                signal[x] = float(ys[np.argmin(np.abs(ys - signal[x - 1]))])
            else:
                signal[x] = float(ys[ys.size // 2])
    t["read_ms"] = (time.perf_counter() - t0) * 1000
    return signal, skel, t


def extract_thinning(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Thinning morfologico (mais preserva topologia que skeletonize)."""
    t = {}
    t0 = time.perf_counter()
    thinned = thin(mask > 0)
    t["thinning_ms"] = (time.perf_counter() - t0) * 1000

    H, W = mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    t0 = time.perf_counter()
    for x in range(W):
        ys = np.where(thinned[:, x])[0]
        if ys.size == 0:
            continue
        elif ys.size == 1:
            signal[x] = float(ys[0])
        else:
            if x > 0 and not np.isnan(signal[x - 1]):
                signal[x] = float(ys[np.argmin(np.abs(ys - signal[x - 1]))])
            else:
                signal[x] = float(ys[ys.size // 2])
    t["read_ms"] = (time.perf_counter() - t0) * 1000
    return signal, thinned, t


def extract_borda_superior(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Primeiro pixel non-zero por coluna (topo da faixa)."""
    t = {}
    H, W = mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    overlay = np.zeros_like(mask, dtype=bool)
    t0 = time.perf_counter()
    for x in range(W):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size > 0:
            signal[x] = float(ys[0])
            overlay[int(ys[0]), x] = True
    t["read_ms"] = (time.perf_counter() - t0) * 1000
    return signal, overlay, t


def extract_media_bordas(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """(Primeiro + Ultimo) / 2 por coluna = ponto medio da espessura."""
    t = {}
    H, W = mask.shape
    signal = np.full(W, np.nan, dtype=np.float32)
    overlay = np.zeros_like(mask, dtype=bool)
    t0 = time.perf_counter()
    for x in range(W):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size > 0:
            mid = int(round((ys[0] + ys[-1]) / 2.0))
            signal[x] = float(mid)
            if 0 <= mid < H:
                overlay[mid, x] = True
    t["read_ms"] = (time.perf_counter() - t0) * 1000
    return signal, overlay, t


# =====================================================================
# Helpers de visualizacao
# =====================================================================

def _validate_signal_in_mask(
    signal: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Retorna (signal_corrigido, in_count, out_count). Pontos fora vao pra NaN."""
    H, W = mask.shape
    out = signal.copy()
    in_count = 0
    out_count = 0
    for x in range(W):
        if np.isnan(out[x]):
            continue
        y = int(round(float(out[x])))
        if 0 <= y < H and mask[y, x]:
            in_count += 1
        else:
            out_count += 1
            out[x] = np.nan
    return out, in_count, out_count


def _save_img1_mascara_original(signal_prob, out_dir):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(signal_prob, cmap="hot", vmin=0, vmax=1)
    ax.set_title("1. Mascara original (canal 2 UNet)", fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.savefig(str(out_dir / "01_mascara_original.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def _save_img2_binarizada(mask, threshold, out_dir):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(mask, cmap="gray_r")
    ax.set_title(f"2. Binarizada (threshold = {threshold})",
                  fontsize=13, fontweight="bold")
    ax.axis("off")
    H, W = mask.shape
    fig.text(0.5, 0.02,
              f"Pixels na mascara: {int(mask.sum())} ({100.0*mask.sum()/(H*W):.2f}%)",
              ha="center", fontsize=10)
    fig.savefig(str(out_dir / "02_mascara_binarizada.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def _save_img3_processed(processed, normalized, method_name, out_dir):
    """Mostra a mascara processada (esqueleto/borda/etc) sobre a imagem."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB), alpha=0.55)
    # Overlay processed em verde
    ys, xs = np.where(processed)
    ax.scatter(xs, ys, s=0.4, c="#00cc44", alpha=0.95)
    ax.set_title(f"3. {method_name} ({int(processed.sum())} pixels)",
                  fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.savefig(str(out_dir / "03_processed.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def _save_img4_validation(signal, mask, signal_prob, out_dir):
    """Sinal verde sobre heatmap vermelho. Pontos fora em amarelo."""
    H, W = mask.shape
    in_x, in_y, out_x, out_y = [], [], [], []
    for x in range(W):
        if np.isnan(signal[x]):
            continue
        y = int(round(float(signal[x])))
        if 0 <= y < H and mask[y, x]:
            in_x.append(x); in_y.append(y)
        else:
            out_x.append(x); out_y.append(y)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(signal_prob, cmap="Reds", alpha=0.65)
    if in_x:
        ax.scatter(in_x, in_y, s=1.0, c="#00cc44",
                    label=f"DENTRO ({len(in_x)})")
    if out_x:
        ax.scatter(out_x, out_y, s=20, c="#ffff00", marker="x", linewidths=2,
                    label=f"FORA ({len(out_x)})")
    total = len(in_x) + len(out_x)
    status = "OK (todos dentro)" if len(out_x) == 0 else f"ERRO: {len(out_x)} fora"
    color = "#006622" if len(out_x) == 0 else "#cc0000"
    ax.set_title(f"4. VALIDACAO: sinal dentro da mascara — {status}",
                  fontsize=13, fontweight="bold", color=color)
    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")
    fig.text(0.5, 0.02,
              f"Dentro: {len(in_x)} | Fora: {len(out_x)} | Total: {total}",
              ha="center", fontsize=10, color=color, fontweight="bold")
    fig.savefig(str(out_dir / "04_sinal_sobre_mascara.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    return len(in_x), len(out_x)


def _save_img5_vs_stenhede(signal, normalized, stenhede_lines, out_dir, method_name):
    H, W, _ = normalized.shape
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB), alpha=0.55)
    # Stenhede em vermelho tracejado
    if stenhede_lines is not None:
        for i in range(stenhede_lines.shape[0]):
            ln = stenhede_lines[i].copy()
            ln[ln == 0] = np.nan
            ax.plot(np.arange(ln.shape[0]), ln, color="#cc0000",
                     linewidth=0.7, alpha=0.85, linestyle="--")
    # Metodo em verde solido
    ax.plot(np.arange(len(signal)), signal, color="#00aa44", linewidth=0.9, alpha=0.95)
    ax.set_title(f"5. Comparacao: Stenhede (vermelho) vs {method_name} (verde)",
                  fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.savefig(str(out_dir / "05_vs_stenhede.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def _save_metrics(method_name, signal, mask, in_count, out_count, timings, out_dir):
    H, W = mask.shape
    coverage = int((~np.isnan(signal)).sum())
    nan = W - coverage
    total_ms = sum(timings.values())
    text = [
        f"{method_name.upper()} — Resultado",
        "=" * 40,
        f"Mascara: IMG_1407 canal 2",
        f"Threshold: {THRESHOLD}",
        f"Image shape: {H} x {W}",
        f"Pixels na mascara: {int(mask.sum())}",
        "",
        "Sinal extraido:",
        f"  Colunas com sinal: {coverage} / {W} ({100.0*coverage/W:.1f}%)",
        f"  Colunas NaN: {nan}",
        "",
        "VALIDACAO:",
        f"  Pixels DENTRO: {in_count}",
        f"  Pixels FORA da mascara: {out_count}  ({'OK' if out_count == 0 else 'ERRO!'})",
        "",
        "Tempo de execucao:",
    ]
    for k, v in timings.items():
        text.append(f"  {k}: {v:.1f} ms")
    text.append(f"  TOTAL: {total_ms:.1f} ms")
    (out_dir / "metrics.txt").write_text("\n".join(text), encoding="utf-8")


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    if not CACHE_PATH.is_file():
        logger.error("Cache nao existe: %s", CACHE_PATH)
        return 1

    logger.info("Carregando cache...")
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    signal_prob: np.ndarray = cache["signal_prob"]
    normalized: np.ndarray = cache["normalized"]
    H, W = signal_prob.shape

    # Binariza uma vez (mesma mask pra todos os metodos)
    mask = (signal_prob > THRESHOLD).astype(np.uint8)
    logger.info("Mascara binarizada: %d pixels (%.2f%%)",
                int(mask.sum()), 100.0 * mask.sum() / (H * W))

    # Roda Stenhede ORIGINAL uma vez (mesma comparacao pra todos)
    logger.info("Rodando Stenhede ORIGINAL pra comparacao...")
    try:
        from pipeline.digitize.stenhede_adapter import _ensure_vendor_on_path
        _ensure_vendor_on_path()
        from src.model.signal_extractor import SignalExtractor  # type: ignore
        import torch
        sx = SignalExtractor()
        fmap = torch.from_numpy(np.ascontiguousarray(signal_prob)).float()
        st_lines = sx(fmap)
        stenhede_lines = (
            st_lines.cpu().numpy() if st_lines.shape[0] > 0 else None
        )
    except Exception as e:
        logger.warning("Stenhede falhou: %s", e)
        stenhede_lines = None

    methods = [
        ("skeleton", extract_skeleton),
        ("thinning", extract_thinning),
        ("borda_superior", extract_borda_superior),
        ("media_bordas", extract_media_bordas),
    ]

    for method_name, method_fn in methods:
        logger.info("=== Metodo: %s ===", method_name)
        out_dir = OUTPUT_ROOT / method_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1, 2 sao iguais pra todos os metodos
        _save_img1_mascara_original(signal_prob, out_dir)
        _save_img2_binarizada(mask, THRESHOLD, out_dir)

        # Roda o extractor
        t0 = time.perf_counter()
        signal, processed, timings = method_fn(mask)
        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        # Validacao
        signal_clean, in_count, out_count = _validate_signal_in_mask(signal, mask)

        # 3, 4, 5
        _save_img3_processed(processed, normalized, method_name, out_dir)
        _save_img4_validation(signal_clean, mask, signal_prob, out_dir)
        _save_img5_vs_stenhede(signal_clean, normalized, stenhede_lines, out_dir, method_name)

        # Metricas
        _save_metrics(method_name, signal_clean, mask, in_count, out_count, timings, out_dir)

        logger.info(
            "%s: cobertura=%.1f%% (%d cols) | in=%d, fora=%d | %.0f ms",
            method_name,
            100.0 * int((~np.isnan(signal_clean)).sum()) / W,
            int((~np.isnan(signal_clean)).sum()),
            in_count, out_count,
            timings["total_ms"],
        )

    logger.info("Concluido. Outputs em %s/", OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
