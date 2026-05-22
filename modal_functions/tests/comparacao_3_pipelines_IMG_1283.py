"""
Compara 3 pipelines de digitalizacao de ECG na IMG_1283.

Pipeline 1: nnU-Net (ECG-Digitiser) + Viterbi (Fortune/Rahimi)  -> BLOQUEADO
Pipeline 2: ECG-Digitiser completo                              -> BLOQUEADO
Pipeline 3: Stenhede + bandas + colunas + Viterbi               -> OK

Motivo bloqueio P1+P2: pesos pre-treinados do nnU-Net em
https://github.com/felixkrones/ECG-Digitiser usam Git-LFS, e a cota LFS
do repo esta esgotada ("This repository exceeded its LFS budget"). Sem
fonte alternativa documentada (zenodo/huggingface).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

from pipeline.digitize.viterbi_extractor import extrair_sinal_viterbi  # noqa: E402

# ----- Paths -----
MASK_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1283_pipeline_completo\05_canal_2_signal_PB.png")
OUTPUT_BASE = Path(r"C:\Users\rafae\Desktop\Projeto ECG\IMG_1283_comparacao_pipelines")
OUT_P1 = OUTPUT_BASE / "pipeline_1_nnunet_viterbi"
OUT_P2 = OUTPUT_BASE / "pipeline_2_ecgdigitiser_full"
OUT_P3 = OUTPUT_BASE / "pipeline_3_stenhede_viterbi"
for d in (OUT_P1, OUT_P2, OUT_P3):
    d.mkdir(parents=True, exist_ok=True)

LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
ALL_LEADS = [lead for row in LEAD_LAYOUT for lead in row]
RHYTHM_LABEL = "II_rhythm"

PX_PER_MM_ORIG = 13.0


def detect_bands(binary_mask, sigma=10.0, distance=50, prominence_factor=0.1,
                 buffer_factor=1.2):
    h = binary_mask.shape[0]
    row_count = binary_mask.sum(axis=1).astype(float)
    if row_count.sum() == 0:
        return []
    smoothed = gaussian_filter1d(row_count, sigma=sigma)
    peaks, _ = find_peaks(smoothed, distance=distance,
                          prominence=smoothed.max() * prominence_factor)
    if peaks.size == 0:
        return []
    inner_valleys = []
    for i in range(len(peaks) - 1):
        seg = smoothed[peaks[i]:peaks[i + 1]]
        inner_valleys.append(int(peaks[i] + np.argmin(seg)))
    bands = []
    for i, peak in enumerate(peaks):
        if i == 0:
            y0 = max(0, peak - int((inner_valleys[0] - peak) * buffer_factor)) if inner_valleys else 0
        else:
            y0 = inner_valleys[i - 1]
        if i == len(peaks) - 1:
            y1 = min(h, peak + int((peak - inner_valleys[-1]) * buffer_factor)) if inner_valleys else h
        else:
            y1 = inner_valleys[i]
        bands.append((y0, y1))
    return bands


# =====================================================================
# Pipeline 1 + 2: BLOQUEADOS
# =====================================================================

def write_pipeline_1_2_failures():
    error_msg_lfs = (
        "PIPELINE BLOQUEADO\n\n"
        "Dependencia: nnU-Net pre-treinado de https://github.com/felixkrones/ECG-Digitiser\n\n"
        "Tentativa: clonar repo + 'git lfs pull' / 'git lfs fetch --all'\n"
        "Erro retornado pelo servidor GitHub:\n\n"
        "    batch response: This repository exceeded its LFS budget.\n"
        "    The account responsible for the budget should increase it to restore access.\n\n"
        "Resultado: os arquivos models/M1/.../checkpoint_final.pth (e M3) ficaram\n"
        "como pointers LFS vazios (smudge filter falhou).\n\n"
        "Busca por fonte alternativa (Zenodo, HuggingFace, FigShare, release\n"
        "GitHub direto): nada documentado no README do projeto nem no paper\n"
        "(arXiv 2410.14185). Sem checkpoint -> nnU-Net nao pode rodar inferencia\n"
        "-> pipeline_1 (mascaras por derivacao) e pipeline_2 (vetorizacao deles)\n"
        "ambos ficam bloqueados.\n\n"
        "Conforme instrucao do usuario, esses pipelines foram documentados como\n"
        "falha e a comparacao final foi gerada usando apenas o pipeline_3.\n"
    )
    (OUT_P1 / "ERROR.txt").write_text(error_msg_lfs, encoding="utf-8")
    (OUT_P2 / "ERROR.txt").write_text(error_msg_lfs, encoding="utf-8")
    print(f"P1+P2: ERROR.txt salvo em {OUT_P1} e {OUT_P2}")


# =====================================================================
# Pipeline 3: Stenhede + Bandas + Viterbi por derivacao
# =====================================================================

def run_pipeline_3() -> dict[str, np.ndarray]:
    """Roda P3 e salva .npy + plot. Retorna dict lead_name -> sinal."""
    print(f"\n--- Pipeline 3: Stenhede + Viterbi ---")
    print(f"Carregando: {MASK_PATH}")
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(MASK_PATH)
    H, W = mask.shape
    print(f"Mascara: {W}x{H}")
    binary = mask < 128
    bands = detect_bands(binary)
    print(f"Bandas detectadas: {len(bands)}")
    if len(bands) < 4:
        print(f"AVISO: detectei {len(bands)} bandas (esperava 4)")
    chunk_px = W // 4
    print(f"Chunk px: {chunk_px}")

    signals: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}

    # 12 leads (bandas 1-3, 4 chunks cada)
    for row_idx, (band, row_leads) in enumerate(zip(bands[:3], LEAD_LAYOUT)):
        y0, y1 = band
        band_mask = mask[y0:y1, :]
        for col_idx, lead_name in enumerate(row_leads):
            x_start = col_idx * chunk_px
            x_end = (col_idx + 1) * chunk_px if col_idx < 3 else W
            chunk_mask = band_mask[:, x_start:x_end]
            sig = extrair_sinal_viterbi(chunk_mask, invert=True)
            signals[lead_name] = sig
            stats[lead_name] = {
                "n_total": len(sig), "n_valid": int((~np.isnan(sig)).sum()),
                "n_nan": int(np.isnan(sig).sum()),
                "min": float(np.nanmin(sig)) if not np.all(np.isnan(sig)) else float("nan"),
                "max": float(np.nanmax(sig)) if not np.all(np.isnan(sig)) else float("nan"),
            }

    # Rhythm
    if len(bands) >= 4:
        y0, y1 = bands[3]
        rhythm_mask = mask[y0:y1, :]
        rhythm_sig = extrair_sinal_viterbi(rhythm_mask, invert=True)
        signals[RHYTHM_LABEL] = rhythm_sig
        stats[RHYTHM_LABEL] = {
            "n_total": len(rhythm_sig),
            "n_valid": int((~np.isnan(rhythm_sig)).sum()),
            "n_nan": int(np.isnan(rhythm_sig).sum()),
            "min": float(np.nanmin(rhythm_sig)),
            "max": float(np.nanmax(rhythm_sig)),
        }

    # Salva .npy
    for name, sig in signals.items():
        np.save(OUT_P3 / f"{name}.npy", sig)

    # Plot 12 subplots + rhythm
    plot_signals = [(lead, signals[lead]) for lead in ALL_LEADS if lead in signals]
    if RHYTHM_LABEL in signals:
        plot_signals.append((RHYTHM_LABEL, signals[RHYTHM_LABEL]))
    n = len(plot_signals)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.4 * n), dpi=100, sharex=False)
    fig.suptitle("Pipeline 3 — Stenhede + Viterbi (por derivacao)",
                 fontsize=14, fontweight="bold", y=0.995)
    for ax, (name, sig) in zip(axes, plot_signals):
        ax.plot(sig, color="#1f78b4", linewidth=0.7)
        ax.axhline(0, color="#aaa", linewidth=0.3, linestyle="--")
        s = stats[name]
        ax.set_title(
            f"{name}: n={s['n_total']} | valid={s['n_valid']} | "
            f"NaN={s['n_nan']} | range [{s['min']:.1f}, {s['max']:.1f}]",
            fontsize=9, loc="left",
        )
        ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("coluna (x)")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(str(OUT_P3 / "pipeline_3_signals.png"),
                bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)

    # Sumario
    sum_lines = ["Pipeline 3 — sumario\n", "=" * 50, ""]
    for name, s in stats.items():
        sum_lines.append(
            f"{name:12s}: n={s['n_total']:5d} | valid={s['n_valid']:5d} | "
            f"NaN={s['n_nan']:5d} | range [{s['min']:7.1f}, {s['max']:7.1f}]"
        )
    (OUT_P3 / "summary.txt").write_text("\n".join(sum_lines), encoding="utf-8")

    print(f"P3: {len(signals)} sinais salvos em {OUT_P3}")
    for name, s in stats.items():
        print(f"  {name}: n={s['n_total']} valid={s['n_valid']} "
              f"NaN={s['n_nan']} range [{s['min']:.1f}, {s['max']:.1f}]")
    return signals


# =====================================================================
# Comparacao final
# =====================================================================

def gerar_comparacao(signals_p3: dict[str, np.ndarray]):
    """Painel 12 linhas (derivacoes) x 3 colunas (P1/P2/P3).
    P1 e P2 mostram texto indicando que falharam."""
    fig, axes = plt.subplots(12, 3, figsize=(18, 24), dpi=100)
    fig.suptitle(
        "Comparacao 3 pipelines — IMG_1283\n"
        "P1 (nnU-Net+Viterbi) e P2 (ECG-Digitiser) bloqueados: "
        "pesos LFS indisponiveis",
        fontsize=14, fontweight="bold", y=0.995,
    )
    col_titles = ["P1: nnU-Net + Viterbi", "P2: ECG-Digitiser full",
                  "P3: Stenhede + Viterbi"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=18)

    for row_idx, lead in enumerate(ALL_LEADS):
        # P1
        ax = axes[row_idx, 0]
        ax.text(0.5, 0.5, "FAILED\nnnU-Net weights\nLFS quota exceeded",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#cc0000",
                bbox=dict(facecolor="#fff0f0", edgecolor="#cc0000"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(lead, fontsize=10, rotation=0, ha="right", va="center",
                      fontweight="bold")
        # P2
        ax = axes[row_idx, 1]
        ax.text(0.5, 0.5, "FAILED\nnnU-Net weights\nLFS quota exceeded",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#cc0000",
                bbox=dict(facecolor="#fff0f0", edgecolor="#cc0000"))
        ax.set_xticks([]); ax.set_yticks([])
        # P3
        ax = axes[row_idx, 2]
        sig = signals_p3.get(lead)
        if sig is not None:
            ax.plot(sig, color="#1f78b4", linewidth=0.7)
            ax.axhline(0, color="#aaa", linewidth=0.3, linestyle="--")
            mn = float(np.nanmin(sig)); mx = float(np.nanmax(sig))
            ax.set_title(f"range [{mn:.0f}, {mx:.0f}]", fontsize=8, loc="left")
        else:
            ax.text(0.5, 0.5, "(sem sinal)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888")
        ax.grid(alpha=0.3)
        ax.set_xticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out = OUTPUT_BASE / "comparacao_pipelines.png"
    plt.savefig(str(out), bbox_inches="tight", dpi=100, facecolor="white")
    plt.close(fig)
    print(f"\nComparacao final salva: {out}")


def main() -> int:
    print("=" * 70)
    print("COMPARACAO DE 3 PIPELINES — IMG_1283")
    print("=" * 70)
    write_pipeline_1_2_failures()
    signals_p3 = run_pipeline_3()
    gerar_comparacao(signals_p3)
    print("\nConcluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
