"""
Teste full pipeline ProECG v5b: foto -> 12 sinais -> medições -> regras -> CNN 24 classes -> laudo.

Roda pipeline_completo_v1 (digitização) e depois roda measure/rules/classify/report
em cima do .npy gerado, salvando visualizações de cada etapa e imprimindo o relatório
completo no stdout.

Uso:
    python -m tests.test_full_v5b [<imagem.jpg>] [<out_dir>]

Se nenhum argumento é passado, busca uma foto-padrão (IMG_1283.jpg, IMG_1310.jpg, IMG_1473.jpg)
em locais conhecidos no Desktop.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

# Força stdout/stderr em UTF-8 para emojis funcionarem no Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
PROJECT_ROOT = MODAL_ROOT.parent
sys.path.insert(0, str(MODAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imagens candidatas
# ---------------------------------------------------------------------------
CANDIDATE_IMAGES = [
    Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3\IMG_1283.jpg"),
    Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3\IMG_1310.jpg"),
    Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3\IMG_1473.jpg"),
    Path(r"C:\Users\rafae\Desktop\EcgPró\IMG_1283.HEIC"),
    Path(r"C:\Users\rafae\Desktop\EcgPró\IMG_1310.HEIC"),
    Path(r"C:\Users\rafae\Desktop\EcgPró\IMG_1473.HEIC"),
]


def find_test_image() -> Path | None:
    for p in CANDIDATE_IMAGES:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Fallback: sinal sintético quando a digitalização falha
# ---------------------------------------------------------------------------

def make_mock_signal_13x4096() -> np.ndarray:
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10.24, 4096)
    # Batimentos a ~75 bpm (1.25 Hz)
    base = 0.4 * np.sin(2 * np.pi * 1.25 * t)
    # Spike QRS ~80ms em cada batimento
    spikes = np.zeros_like(t)
    for beat in range(int(10.24 * 1.25)):
        center = beat / 1.25
        idx = (np.abs(t - center)).argmin()
        if idx + 4 < len(spikes):
            spikes[idx : idx + 4] = 1.2
    sig = base + spikes + 0.03 * rng.normal(size=4096)
    out = np.zeros((13, 4096), dtype=np.float32)
    for i in range(13):
        out[i] = sig * (0.4 + 0.5 * (i / 12)) + 0.02 * rng.normal(size=4096)
    return out


# ---------------------------------------------------------------------------
# Visualizações extras
# ---------------------------------------------------------------------------

def save_measurements_plot(
    lead_ii: np.ndarray,
    fs: int,
    measurements: dict,
    out_path: Path,
) -> None:
    """Plota o lead II com R-peaks marcados + medições no canto."""
    fig, ax = plt.subplots(figsize=(14, 4), dpi=110)
    t = np.arange(len(lead_ii)) / fs
    ax.plot(t, lead_ii, color="#1f78b4", linewidth=0.8)

    # Detecta R-peaks pra marcar visualmente
    try:
        sig_uv = lead_ii * 1000.0  # mV -> µV
        height = max(50.0, float(np.percentile(np.abs(sig_uv), 99)) * 0.5)
        peaks, _ = find_peaks(
            sig_uv, height=height, distance=int(0.3 * fs),
            prominence=max(40.0, float(np.max(np.abs(sig_uv))) * 0.4),
        )
        if len(peaks):
            ax.plot(peaks / fs, lead_ii[peaks], "rv", markersize=8,
                    label=f"R-peaks ({len(peaks)})")
    except Exception:
        pass

    ax.axhline(0, color="#aaa", linewidth=0.3, linestyle="--")
    ax.set_xlabel("t (s)"); ax.set_ylabel("mV"); ax.grid(alpha=0.3)
    ax.set_title("Lead II — R-peaks detectados", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    # Painel de medições
    hr = measurements.get("heart_rate")
    if isinstance(hr, dict):
        hr_val = hr.get("mean_bpm")
    else:
        hr_val = hr
    iv = measurements.get("intervals", {})
    pr = iv.get("pr_ms") if isinstance(iv, dict) else measurements.get("pr_interval")
    qrs = iv.get("qrs_ms") if isinstance(iv, dict) else measurements.get("qrs_duration")
    qt = iv.get("qt_ms") if isinstance(iv, dict) else measurements.get("qt_interval")
    qtc = iv.get("qtc_ms") if isinstance(iv, dict) else measurements.get("qtc_bazett")
    ax_obj = measurements.get("axis", {})
    if isinstance(ax_obj, dict):
        axis_val = ax_obj.get("degrees")
    else:
        axis_val = ax_obj

    text = "\n".join([
        f"FC:   {hr_val:.0f} bpm" if hr_val is not None else "FC:   —",
        f"Eixo: {axis_val:+.0f}°" if axis_val is not None else "Eixo: —",
        f"PR:   {pr:.0f} ms" if pr is not None else "PR:   —",
        f"QRS:  {qrs:.0f} ms" if qrs is not None else "QRS:  —",
        f"QT:   {qt:.0f} ms" if qt is not None else "QT:   —",
        f"QTc:  {qtc:.0f} ms" if qtc is not None else "QTc:  —",
    ])
    ax.text(
        0.01, 0.98, text, transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#666", alpha=0.9, pad=6),
    )
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def save_cnn_scores_plot(
    all_scores: dict[str, float],
    youden_thresholds: dict[str, float],
    out_path: Path,
) -> None:
    """Barra horizontal dos 24 scores da CNN, vermelha quando passa Youden."""
    classes = list(all_scores.keys())
    scores = [all_scores[c] for c in classes]
    thresholds = [youden_thresholds.get(c, 0.5) for c in classes]
    passed = [s >= t for s, t in zip(scores, thresholds)]
    colors = ["#d33" if p else "#999" for p in passed]

    fig, ax = plt.subplots(figsize=(11, 9), dpi=110)
    y = np.arange(len(classes))
    ax.barh(y, scores, color=colors, edgecolor="black", linewidth=0.4)

    # Linha do threshold por classe
    for i, t in enumerate(thresholds):
        ax.plot([t, t], [i - 0.4, i + 0.4], color="#28a", linewidth=2.0)

    # Labels dos scores
    for i, (s, p) in enumerate(zip(scores, passed)):
        ax.text(s + 0.01, i, f"{s:.3f}", va="center", fontsize=8,
                color="#d33" if p else "#666",
                fontweight="bold" if p else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probabilidade (sigmoid)")
    ax.set_title(
        "CNN v5b — 24 classes (vermelho = passou Youden threshold)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(0.5, color="#bbb", linestyle="--", linewidth=0.7, label="0.5")

    n_passed = sum(passed)
    ax.text(
        0.99, 0.02,
        f"{n_passed}/24 acima do limiar",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#666", pad=4),
    )

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


def save_laudo_image(report_text: str, out_path: Path) -> None:
    """Renderiza o texto do laudo como imagem (monospace, A4-ish)."""
    lines = report_text.splitlines() or ["(laudo vazio)"]
    n_lines = len(lines)
    height = max(6, 0.22 * n_lines + 1.0)
    fig, ax = plt.subplots(figsize=(11, height), dpi=110)
    ax.axis("off")
    ax.text(
        0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", family="monospace", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_measurements(m: dict) -> None:
    _hr("MEDIÇÕES")
    hr = m.get("heart_rate")
    if isinstance(hr, dict):
        print(f"  FC mean:   {hr.get('mean_bpm')} bpm")
        print(f"  FC min:    {hr.get('min_bpm')} bpm")
        print(f"  FC max:    {hr.get('max_bpm')} bpm")
        print(f"  Regular:   {hr.get('regular')}")
    else:
        print(f"  FC: {hr}")
    ax = m.get("axis", {})
    if isinstance(ax, dict):
        print(f"  Eixo:      {ax.get('degrees')}° ({ax.get('classification')})")
    iv = m.get("intervals", {})
    if isinstance(iv, dict):
        print(f"  PR:        {iv.get('pr_ms')} ms")
        print(f"  QRS:       {iv.get('qrs_ms')} ms")
        print(f"  QT:        {iv.get('qt_ms')} ms")
        print(f"  QTc:       {iv.get('qtc_ms')} ms (Bazett)")
    pw = m.get("p_wave", {})
    if isinstance(pw, dict):
        print(f"  P present: {pw.get('present')}")
    quality = m.get("quality", {})
    if isinstance(quality, dict):
        print(f"  Quality:   {quality.get('overall')} "
              f"(p_conf={quality.get('p_wave_confidence')}, "
              f"t_conf={quality.get('t_wave_confidence')})")
    warnings_ = m.get("warnings", [])
    if warnings_:
        print("  Warnings:")
        for w in warnings_:
            print(f"    - {w}")


def print_cnn(all_scores: dict, findings: list[dict],
              youden: dict) -> None:
    _hr("CNN v5b — 24 CLASSES")
    print(f"  {'classe':<8} {'score':>8}  {'youden':>8}  passou?")
    for cn, sc in all_scores.items():
        t = youden.get(cn, 0.5)
        passed = "✓" if sc >= t else " "
        print(f"  {cn:<8} {sc:>8.4f}  {t:>8.4f}    {passed}")

    print(f"\n  → {len(findings)} achado(s) acima do Youden threshold:")
    for f in findings:
        flag = "🚩" if f.get("is_red_flag") else "  "
        print(f"    {flag} [{f['code']:<6}] {f['description']} "
              f"(score={f.get('score')}, thr={f.get('threshold')})")


def print_rules(rule_findings: list[dict]) -> None:
    _hr("REGRAS CLÍNICAS")
    if not rule_findings:
        print("  (nenhum achado por regras)")
        return
    for f in rule_findings:
        leads = f.get("leads_affected") or []
        leads_str = f" [{', '.join(leads)}]" if leads else ""
        print(f"  - [{f['code']}] {f['description']}{leads_str}")


def print_frontend(frontend: dict) -> None:
    _hr("FRONTEND JSON (preview)")
    preview = {
        "severity": frontend.get("severity"),
        "red_flags": frontend.get("red_flags"),
        "diagnoses": frontend.get("diagnoses"),
        "warnings": frontend.get("warnings"),
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))


def print_report_text(report: dict) -> None:
    _hr("LAUDO (texto)")
    print(report.get("report_text", "(vazio)"))


def list_outputs(out_dir: Path) -> None:
    _hr(f"ARQUIVOS GERADOS EM {out_dir}")
    files = sorted(out_dir.iterdir())
    if not files:
        print("  (vazio)")
        return
    for p in files:
        if p.is_file():
            sz = p.stat().st_size
            unit, val = (("KB", sz / 1024) if sz < 1_000_000
                         else ("MB", sz / 1_048_576))
            print(f"  {p.name:<40} {val:>8.1f} {unit}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    img_arg = Path(argv[1]) if len(argv) >= 2 else None
    out_arg = Path(argv[2]) if len(argv) >= 3 else None

    img_path = img_arg if (img_arg and img_arg.exists()) else find_test_image()
    if img_path is None:
        print("ERRO: nenhuma imagem de teste encontrada.")
        print("Procurei em:")
        for c in CANDIDATE_IMAGES:
            print(f"  - {c}")
        print("Passe um caminho como primeiro argumento:")
        print(f"  python -m tests.test_full_v5b <imagem.jpg>")
        return 1

    out_dir = out_arg or (HERE / "_v5b_full_output" / img_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[IMG] Imagem: {img_path}")
    print(f"[OUT] Output: {out_dir}")

    # ----- 1. Pipeline de digitalização -----
    _hr("ETAPA 1 — pipeline_completo_v1 (digitalização)")
    digit_ok = False
    try:
        from pipeline.pipeline_completo_v1 import run_pipeline
        rc = run_pipeline(img_path, out_dir)
        if rc == 0:
            digit_ok = True
            print("  [OK] Digitalizacao OK")
        else:
            print(f"  [ERR] run_pipeline retornou {rc}")
    except Exception:
        print("  [ERR] Excecao na digitalizacao:")
        traceback.print_exc()

    # ----- 2. Carregar sinal (do .npy ou mock) -----
    _hr("ETAPA 2 — carregando sinal (13, 4096)")
    ecg_13ch: np.ndarray | None = None
    npy_13 = out_dir / "ecg_13_leads_mv.npy"
    npy_12 = out_dir / "ecg_12_leads_mv.npy"
    try:
        if npy_13.exists():
            arr = np.load(npy_13)
            print(f"  ✓ Carregado {npy_13.name} shape={arr.shape}")
            ecg_13ch = arr.astype(np.float32)
        elif npy_12.exists():
            arr = np.load(npy_12)
            print(f"  ⚠ {npy_12.name} sem rhythm strip — duplicando lead II "
                  f"como canal 13")
            full = np.zeros((13, arr.shape[1]), dtype=np.float32)
            full[:12] = arr
            full[12] = arr[1]
            ecg_13ch = full
        else:
            raise FileNotFoundError("Nenhum .npy gerado pela digitalização.")
    except Exception:
        print("  ✗ Falha ao carregar — usando MOCK signal:")
        traceback.print_exc()
        ecg_13ch = make_mock_signal_13x4096()

    fs = 400  # pipeline_completo_v1 reamostra pra 400Hz

    # ----- 3. Medições -----
    _hr("ETAPA 3 — measure_ecg")
    measurements: dict[str, Any] = {}
    try:
        from pipeline.measure import measure_ecg
        measurements = measure_ecg(ecg_13ch[:12], fs=fs)
        print("  ✓ Medições OK")
    except Exception:
        print("  ✗ measure_ecg falhou:")
        traceback.print_exc()
        measurements = {
            "heart_rate": {"mean_bpm": None},
            "axis": {"degrees": None, "classification": "indeterminado"},
            "intervals": {"pr_ms": None, "qrs_ms": None, "qt_ms": None, "qtc_ms": None},
            "p_wave": {"present": False},
            "quality": {"overall": "poor"},
            "warnings": ["measure_ecg falhou — usando placeholders"],
        }

    # ----- 4. Regras clínicas -----
    _hr("ETAPA 4 — apply_clinical_rules")
    rule_findings: list[dict] = []
    try:
        from pipeline.rules import apply_clinical_rules
        rule_findings = apply_clinical_rules(measurements)
        print(f"  ✓ {len(rule_findings)} achado(s) por regras")
    except Exception:
        print("  ✗ rules falhou:")
        traceback.print_exc()

    # ----- 5. CNN v5b -----
    _hr("ETAPA 5 — classify_ecg v5b")
    cnn_findings: list[dict] = []
    cnn_all_scores: dict[str, float] = {}
    youden_thresholds: dict[str, float] = {}
    cnn_ok = False
    try:
        from pipeline.classify import (
            classify_ecg, classify_ecg_full, YOUDEN_THRESHOLDS, CLASS_NAMES,
        )
        youden_thresholds = dict(YOUDEN_THRESHOLDS)
        cnn_findings = classify_ecg(ecg_13ch)
        cnn_all_scores = classify_ecg_full(ecg_13ch)
        cnn_ok = True
        print(f"  ✓ CNN rodou — {len(cnn_findings)} achado(s) acima do Youden")
    except FileNotFoundError as e:
        print(f"  ✗ Modelo CNN v5b ausente: {e}")
        # Mock para visualização não quebrar
        try:
            from pipeline.classify import CLASS_NAMES, YOUDEN_THRESHOLDS
            youden_thresholds = dict(YOUDEN_THRESHOLDS)
            cnn_all_scores = {c: 0.0 for c in CLASS_NAMES}
        except Exception:
            cnn_all_scores = {}
    except Exception:
        print("  ✗ classify_ecg falhou:")
        traceback.print_exc()

    # ----- 6. Report -----
    _hr("ETAPA 6 — generate_report")
    report: dict[str, Any] = {"report_text": "", "findings": [], "diagnoses": []}
    frontend: dict[str, Any] = {}
    try:
        from pipeline.report import generate_report, generate_frontend_report
        report = generate_report(measurements, rule_findings, cnn_findings)
        frontend = generate_frontend_report(measurements, rule_findings, cnn_findings)
        print("  ✓ Laudo gerado")
    except Exception:
        print("  ✗ report falhou:")
        traceback.print_exc()

    # ----- 7. Visualizações extras -----
    _hr("ETAPA 7 — visualizações extras")
    try:
        save_measurements_plot(
            ecg_13ch[1], fs, measurements, out_dir / "10_medicoes.png",
        )
        print("  ✓ 10_medicoes.png")
    except Exception:
        print("  ✗ 10_medicoes.png falhou:")
        traceback.print_exc()

    try:
        if cnn_all_scores:
            save_cnn_scores_plot(
                cnn_all_scores, youden_thresholds, out_dir / "11_cnn_scores.png",
            )
            print("  ✓ 11_cnn_scores.png")
        else:
            print("  — pulando 11_cnn_scores.png (sem scores)")
    except Exception:
        print("  ✗ 11_cnn_scores.png falhou:")
        traceback.print_exc()

    try:
        if report.get("report_text"):
            save_laudo_image(report["report_text"], out_dir / "12_laudo.png")
            print("  ✓ 12_laudo.png")
        else:
            print("  — pulando 12_laudo.png (laudo vazio)")
    except Exception:
        print("  ✗ 12_laudo.png falhou:")
        traceback.print_exc()

    # ----- 8. Print final -----
    print_measurements(measurements)
    print_rules(rule_findings)
    if cnn_all_scores:
        print_cnn(cnn_all_scores, cnn_findings, youden_thresholds)
    else:
        _hr("CNN v5b — INDISPONÍVEL")
        print("  Modelo best_ecg_model_v5b_13ch_mv.pth não encontrado.")
    print_report_text(report)
    if frontend:
        print_frontend(frontend)
    list_outputs(out_dir)

    print("\n[OK] Teste concluido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
