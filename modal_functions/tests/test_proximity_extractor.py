"""
Teste do proximity_extractor no IMG_1407 (usa cache do canal 2 da UNet).

Gera 7 imagens + metricas em:
  ~/Desktop/Projeto ECG/resultados_teste_v1/benchmark_extractors/

REGRA INVIOLAVEL VALIDADA: imagem 06 mostra cada ponto do sinal sobre a
mascara. Pontos FORA da mascara aparecem em VERMELHO (devem ser ZERO).
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

from pipeline.digitize.proximity_extractor import extract_signal_proximity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("test_proximity")

CACHE_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1407"
    r"\_cache_pre_signal_extractor.pkl"
)
OUTPUT_DIR = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\benchmark_extractors"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.1


def _save_text_metrics(
    result: dict,
    threshold: float,
    H: int,
    W: int,
    binary_pixels: int,
    out_path: Path,
) -> None:
    coverage_pts = int((~np.isnan(result["signal"])).sum())
    nan_pts = W - coverage_pts
    cov_pct = 100.0 * coverage_pts / W
    text = [
        "PROXIMITY EXTRACTOR — Metricas",
        "=" * 50,
        "Input: IMG_1407, canal 2 UNet",
        f"Threshold: {threshold}",
        f"Image shape: {H} x {W}",
        f"Pixels de tracado na mascara: {binary_pixels} "
        f"({100.0*binary_pixels/(H*W):.2f}% da imagem)",
        "",
        "Caminhada:",
        f"  Passos totais: {result['steps']}",
        f"  Colunas com multipla visita: {result['multi_y_columns']}",
        f"  Tempo de execucao: {result['time_s']:.1f}s",
        "",
        "Sinal extraido:",
        f"  Colunas com sinal: {coverage_pts} / {W}",
        f"  Colunas NaN: {nan_pts}",
        f"  Cobertura: {cov_pct:.1f}%",
        "",
        "VALIDACAO MASCARA:",
        f"  Pixels do sinal DENTRO da mascara: {result['in_mask']}",
        f"  Pixels do sinal FORA da mascara: {result['out_of_mask']} "
        f"{'❌ ERRO!' if result['out_of_mask'] > 0 else '✅ OK (deve ser zero)'}",
    ]
    out_path.write_text("\n".join(text), encoding="utf-8")
    logger.info("Metricas salvas em %s", out_path)


def main() -> int:
    if not CACHE_PATH.is_file():
        logger.error("Cache nao existe em %s", CACHE_PATH)
        return 1

    logger.info("Carregando cache...")
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    signal_prob: np.ndarray = cache["signal_prob"]
    normalized: np.ndarray = cache["normalized"]
    H, W = signal_prob.shape
    logger.info("signal_prob: %s, normalized: %s", signal_prob.shape, normalized.shape)

    # ============================================================
    # IMAGEM 01: Mascara de input (heatmap canal 2)
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    im = ax.imshow(signal_prob, cmap="hot", vmin=0, vmax=1)
    ax.set_title(
        "01. Subetapa 1: Mascara de input (canal 2 UNet)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.025, label="prob")
    fig.savefig(OUTPUT_DIR / "01_mascara_input.png", bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("01_mascara_input.png salvo")

    # ============================================================
    # IMAGEM 02: Mascara binarizada
    # ============================================================
    binary = signal_prob > THRESHOLD
    n_trace = int(binary.sum())
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(binary, cmap="gray_r")
    ax.set_title(
        f"02. Subetapa 2: Mascara binarizada (threshold = {THRESHOLD})",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    fig.text(0.5, 0.02,
             f"Total de pixels de tracado: {n_trace} "
             f"({100.0*n_trace/(H*W):.2f}% da imagem)",
             ha="center", fontsize=10)
    fig.savefig(OUTPUT_DIR / "02_mascara_binarizada.png", bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("02_mascara_binarizada.png salvo")

    # ============================================================
    # Roda o extractor (gera path)
    # ============================================================
    logger.info("Rodando proximity extractor...")
    result = extract_signal_proximity(signal_prob, threshold=THRESHOLD,
                                       start_side="left")
    path = result["path"]
    signal_prox = result["signal"]

    if not path:
        logger.error("Path vazio")
        return 1

    # ============================================================
    # IMAGEM 03: Ponto de partida
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(binary, cmap="gray_r")
    start_y, start_x = path[0]
    ax.add_patch(mpatches.Circle((start_x, start_y), radius=30,
                                  facecolor="#00cc44", edgecolor="black",
                                  linewidth=2, alpha=0.8))
    ax.set_title(
        "03. Subetapa 3: Ponto de partida da caminhada (lado esquerdo, Y mediano)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    fig.text(0.5, 0.02, f"Inicio: (x={start_x}, y={start_y})",
             ha="center", fontsize=10)
    fig.savefig(OUTPUT_DIR / "03_ponto_partida.png", bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("03_ponto_partida.png salvo")

    # ============================================================
    # IMAGEM 04: Caminhada completa (cor por SEGMENTO = derivacao)
    # ============================================================
    segments = result.get("segments", [path])
    lines = result.get("lines", [])
    n_segments = len(segments)
    n_lines = len(lines)

    # Paleta distinta pra ate 16 segmentos
    palette = plt.get_cmap("tab20", max(n_segments, 13))

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB), alpha=0.55)
    for si, seg in enumerate(segments):
        if len(seg) < 10:
            continue
        seg_arr = np.array(seg)  # (N, 2) [y, x]
        stride = max(1, len(seg_arr) // 2000)
        sub = seg_arr[::stride]
        ax.scatter(sub[:, 1], sub[:, 0], c=[palette(si % palette.N)],
                   s=1.5, alpha=0.9, label=f"Seg {si+1} ({len(seg)} pts)")

    ax.set_title(
        f"04. Caminhada por proximidade — {n_segments} segmentos "
        f"(= candidatos a derivacao)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    if n_segments <= 16:
        ax.legend(loc="lower right", fontsize=8, ncol=2, markerscale=4)
    fig.text(0.5, 0.02,
             f"Total: {len(path)} passos | {n_segments} segmentos | "
             f"{n_lines} linhas validas (>= 30 cols)",
             ha="center", fontsize=10)
    fig.savefig(OUTPUT_DIR / "04_caminhada_completa.png", bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("04_caminhada_completa.png salvo (%d segmentos)", n_segments)

    # ============================================================
    # IMAGEM 05: Sinal extraido — uma linha por segmento, sobre normalized
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB), alpha=0.55)
    xs_all = np.arange(W)
    for li, ln in enumerate(lines):
        color = palette(li % palette.N)
        ax.plot(xs_all, ln, color=color, linewidth=0.9, alpha=0.95,
                label=f"Linha {li+1}")
    ax.set_title(
        f"05. Sinal extraido — {n_lines} linhas (1 por derivacao)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    if n_lines <= 16:
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    total_pts = sum(int((~np.isnan(ln)).sum()) for ln in lines)
    fig.text(0.5, 0.02,
             f"Total pontos validos: {total_pts}  |  "
             f"Linhas: {n_lines}  |  Multi-Y cols (somados): {result['multi_y_columns']}",
             ha="center", fontsize=10)
    fig.savefig(OUTPUT_DIR / "05_sinal_extraido.png", bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("05_sinal_extraido.png salvo")

    # ============================================================
    # IMAGEM 06: VALIDACAO — sinal sobre mascara (verde = dentro, vermelho = fora)
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    # Fundo: heatmap em vermelho/laranja
    ax.imshow(signal_prob, cmap="Reds", alpha=0.65)

    # Sinal: pontos verdes (dentro mask) ou vermelhos (fora) — agregado de todas as linhas
    inside_xs, inside_ys = [], []
    outside_xs, outside_ys = [], []
    for ln in lines:
        for x in range(W):
            if np.isnan(ln[x]):
                continue
            y = int(round(float(ln[x])))
            if 0 <= y < H and binary[y, x]:
                inside_xs.append(x)
                inside_ys.append(y)
            else:
                outside_xs.append(x)
                outside_ys.append(y)

    if inside_xs:
        ax.scatter(inside_xs, inside_ys, s=1.0, c="#00cc44", alpha=0.95,
                   label=f"DENTRO ({len(inside_xs)})")
    if outside_xs:
        ax.scatter(outside_xs, outside_ys, s=15, c="#ff0000", marker="x",
                   linewidths=2, label=f"⚠️ FORA ({len(outside_xs)})")

    inside_n = len(inside_xs)
    outside_n = len(outside_xs)
    total_n = inside_n + outside_n
    title_status = (
        f"✅ TODOS dentro da mascara"
        if outside_n == 0
        else f"❌ {outside_n} pontos FORA da mascara"
    )
    ax.set_title(
        f"06. VALIDACAO: Sinal extraido sobre mascara de calor — {title_status}",
        fontsize=13, fontweight="bold",
        color="#006622" if outside_n == 0 else "#cc0000",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")
    fig.text(0.5, 0.02,
             f"Pixels do sinal dentro: {inside_n}/{total_n} "
             f"({100.0*inside_n/max(total_n,1):.1f}%)  |  "
             f"FORA: {outside_n} (DEVE SER ZERO)",
             ha="center", fontsize=10,
             color="#006622" if outside_n == 0 else "#cc0000",
             fontweight="bold")
    fig.savefig(OUTPUT_DIR / "06_sinal_sobre_mascara.png", bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("06_sinal_sobre_mascara.png salvo (dentro=%d, fora=%d)",
                inside_n, outside_n)

    # ============================================================
    # IMAGEM 07: Comparacao com Stenhede ORIGINAL
    # ============================================================
    logger.info("Rodando Stenhede ORIGINAL pra comparacao...")
    try:
        from pipeline.digitize.stenhede_adapter import _ensure_vendor_on_path
        _ensure_vendor_on_path()
        from src.model.signal_extractor import SignalExtractor  # type: ignore
        import torch

        stenhede = SignalExtractor()
        fmap = torch.from_numpy(np.ascontiguousarray(signal_prob)).float()
        stenhede_lines = stenhede(fmap)
        stenhede_lines_np = (
            stenhede_lines.cpu().numpy() if stenhede_lines.shape[0] > 0 else None
        )
    except Exception as e:
        logger.warning("Stenhede falhou: %s", e)
        stenhede_lines_np = None

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB), alpha=0.55)

    # Stenhede (vermelho tracejado)
    if stenhede_lines_np is not None:
        for i in range(stenhede_lines_np.shape[0]):
            ln = stenhede_lines_np[i].copy()
            ln[ln == 0] = np.nan
            xs_s = np.arange(ln.shape[0])
            ax.plot(xs_s, ln, color="#cc0000", linewidth=0.7,
                    alpha=0.85, linestyle="--")
        n_st_lines = stenhede_lines_np.shape[0]
    else:
        n_st_lines = 0

    # Proximity (verde solido) — TODAS as linhas
    for ln in lines:
        ax.plot(np.arange(W), ln, color="#00aa44", linewidth=1.0, alpha=0.95)

    ax.set_title(
        "07. Comparacao: Stenhede ORIGINAL (vermelho tracejado) vs Proximidade (verde)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    total_prox_pts = sum(int((~np.isnan(ln)).sum()) for ln in lines)
    fig.text(0.5, 0.02,
             f"Stenhede: {n_st_lines} linhas  |  "
             f"Proximidade: {n_lines} linhas ({total_prox_pts} pts, "
             f"{result['steps']} passos, {result['time_s']:.1f}s)",
             ha="center", fontsize=10)
    fig.savefig(OUTPUT_DIR / "07_comparacao_stenhede.png", bbox_inches="tight",
                dpi=110, facecolor="white")
    plt.close(fig)
    logger.info("07_comparacao_stenhede.png salvo")

    # ============================================================
    # Metricas
    # ============================================================
    _save_text_metrics(
        result, THRESHOLD, H, W, n_trace,
        OUTPUT_DIR / "proximity_metrics.txt",
    )

    logger.info("Concluido. Imagens em %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
