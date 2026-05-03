"""
Pipeline completo focado em V2 -- IMG_1279
==========================================

Roda calibracao -> extracao -> measure_ecg -> regras clinicas -> CNN ->
laudo. Depois imprime SO o que diz respeito a V2:
  - sinal V2 (estatisticas)
  - medicoes V2 (baseline, ST, T, Q, QRS amplitudes, R/S)
  - achados das regras clinicas que incluem V2
  - top 5 classes da CNN
  - trecho do laudo final que menciona V2

Uso:
    python -m modal_functions.pipeline.test_v2_pipeline
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from .classify import CLASS_DESCRIPTIONS, classify_ecg, classify_ecg_full
from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.lead_separator import _build_grid_mask, separate_and_extract
from .measure import (
    LEAD_NAMES,
    UV_PER_MM,
    _baseline_uv,
    _detect_p_peaks,
    _detect_qrs_onsets_offsets,
    _detect_r_peaks,
    _detect_t_peaks_offsets,
    _strip_nans,
    measure_ecg,
)
from .report import generate_frontend_report
from .rules import apply_clinical_rules

NORMALIZED_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Normalizados Leader")
MASKS_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\Leader Masks Teste")
TARGET = "IMG_1279.png"
# Lead alvo via argumento CLI (default V2). Aliases DI/DII/DIII suportados.
_LEAD_ALIASES = {"DI": "I", "DII": "II", "DIII": "III"}
_arg = sys.argv[1] if len(sys.argv) > 1 else "V2"
LEAD = _LEAD_ALIASES.get(_arg, _arg)


def _calibrate_normalized(img: np.ndarray):
    digitizer = ECGDigitizer(use_mock=False)
    _, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        _, keypoints = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    cal = calibrate(grid_matrix=grid_matrix, normalized_image=img)
    return cal, grid_matrix


def _signals_dict_to_array(leads: dict[str, dict]) -> np.ndarray:
    """Converte dict {lead: {signal_uv}} em array (12, N) em mV pra CNN."""
    max_n = max(
        (info["signal_uv"].shape[0] for info in leads.values()), default=1
    )
    out = np.zeros((12, max_n), dtype=np.float64)
    for i, name in enumerate(LEAD_NAMES):
        if name in leads:
            sig = leads[name]["signal_uv"]
            sig = np.where(np.isnan(sig), 0.0, sig)
            n = min(len(sig), max_n)
            out[i, :n] = sig[:n] / 1000.0  # uV -> mV
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s"
    )

    img_path = NORMALIZED_DIR / TARGET
    mask_path = MASKS_DIR / TARGET
    if not img_path.exists() or not mask_path.exists():
        print(f"ERRO: faltam arquivos pra {TARGET}", file=sys.stderr)
        return 1

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"ERRO ao carregar {TARGET}", file=sys.stderr)
        return 1
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    print("=" * 78)
    print(f" PIPELINE COMPLETO -- {TARGET} (foco em {LEAD})")
    print("=" * 78)

    # --- 1. Calibracao + grid mask ---
    cal, grid_matrix = _calibrate_normalized(img)
    grid_bin = _build_grid_mask(grid_matrix, img.shape[:2])
    print(
        f"\n[1] Calibracao:"
        f"\n    px_per_mm        = {cal['px_per_mm']:.3f}"
        f"\n    sampling_rate_hz = {cal['sampling_rate_hz']:.1f} Hz"
        f"\n    uv_per_pixel     = {cal['uv_per_pixel']:.3f} uV/px"
        f"\n    gain             = {cal['gain_mm_per_mV']:.0f} mm/mV"
    )

    # --- 2. Lead separation + extracao ---
    sep = separate_and_extract(mask, img, cal, grid_mask=grid_bin)
    print(
        f"\n[2] Lead separator: layout={sep['layout']}, "
        f"{len(sep['leads'])} derivacoes extraidas"
    )

    leads_signals = {
        n: i["signal_uv"]
        for n, i in sep["leads"].items() if n != "II_rhythm"
    }
    if "II" not in leads_signals and "II_rhythm" in sep["leads"]:
        leads_signals["II"] = sep["leads"]["II_rhythm"]["signal_uv"]

    if LEAD not in leads_signals:
        print(f"ERRO: {LEAD} nao foi extraido", file=sys.stderr)
        return 1

    sig_v2 = _strip_nans(leads_signals[LEAD])
    fs = int(round(cal["sampling_rate_hz"]))
    duration_s = len(sig_v2) / fs

    print(
        f"\n[3] Sinal {LEAD}:"
        f"\n    samples          = {len(sig_v2)}"
        f"\n    duracao          = {duration_s:.2f} s"
        f"\n    range            = [{sig_v2.min():+.1f}, {sig_v2.max():+.1f}] uV"
        f"\n    range em mm      = [{sig_v2.min()/UV_PER_MM:+.2f}, {sig_v2.max()/UV_PER_MM:+.2f}] mm"
        f"\n    mediana          = {np.median(sig_v2):+.1f} uV"
    )

    # --- Detecao por lead (R, P, QRS, T, baseline) em V2 ---
    rp = _detect_r_peaks(sig_v2, fs)
    if len(rp) < 2:
        rp = _detect_r_peaks(sig_v2, fs, bipolar=True)
    pp, _ = _detect_p_peaks(sig_v2, rp, fs)
    qon, qoff = _detect_qrs_onsets_offsets(sig_v2, rp, fs)
    tp, toff = _detect_t_peaks_offsets(sig_v2, rp, qoff, fs)
    bl_v2 = _baseline_uv(sig_v2, toff, pp, rp, fs)

    # ST por batimento em V2
    j_offset = int(round(0.060 * fs))
    st_uv_per_beat = []
    for jp in qoff:
        jp = int(jp)
        if jp < 0:
            continue
        idx = jp + j_offset
        if idx >= len(sig_v2):
            continue
        st_uv_per_beat.append(float(sig_v2[idx]) - bl_v2)
    st_med_uv = float(np.median(st_uv_per_beat)) if st_uv_per_beat else float("nan")

    print(
        f"\n[4] Detecao em {LEAD} (per-lead):"
        f"\n    R-peaks (idx)        = {rp.tolist()}"
        f"\n    R-peaks (ms)         = {[round(int(r)/fs*1000, 1) for r in rp]}"
        f"\n    P-peaks detectados   = {sum(1 for p in pp if p >= 0)}/{len(rp)}"
        f"\n    QRS onsets/offsets   = "
        f"{[(int(a), int(b)) for a, b in zip(qon, qoff)]}"
        f"\n    T peaks/offsets      = "
        f"{[(int(a), int(b)) for a, b in zip(tp, toff)]}"
        f"\n    Baseline (TP segs)   = {bl_v2:+.1f} uV ({bl_v2/UV_PER_MM:+.2f} mm)"
        f"\n    ST per beat (uV)     = {[round(x, 1) for x in st_uv_per_beat]}"
        f"\n    ST mediano (uV)      = {st_med_uv:+.1f} uV"
        f"\n    ST mediano (mm)      = {st_med_uv/UV_PER_MM:+.2f} mm"
    )

    # --- 5. measure_ecg (todos os leads, mas filtramos V2) ---
    measurements = measure_ecg(leads_signals, fs=fs, calibration=cal)

    iv = measurements["intervals"]
    hr = measurements["heart_rate"]
    ax = measurements["axis"]
    st_v2 = measurements["st_segment"].get(LEAD, {})
    t_v2 = measurements["t_wave"].get(LEAD, {})
    q_v2 = measurements["q_wave"].get(LEAD, {})
    qrs_a_v2 = measurements["qrs_amplitudes"].get(LEAD, {})
    rs = measurements["r_s_ratio"]

    print(
        f"\n[5] measure_ecg (globais):"
        f"\n    HR mean/min/max      = {hr['mean_bpm']}/{hr['min_bpm']}/{hr['max_bpm']} bpm  Reg={hr['regular']}"
        f"\n    Eixo                 = {ax['degrees']}deg ({ax['classification']})"
        f"\n    PR/QRS/QT/QTc        = {iv['pr_ms']}/{iv['qrs_ms']}/{iv['qt_ms']}/{iv['qtc_ms']} ms"
        f"\n    Ritmo                = {measurements.get('p_wave', {}).get('present')} (P presente?)"
    )

    print(
        f"\n[6] measure_ecg ({LEAD} especifico):"
        f"\n    ST                   = {st_v2.get('value_mm')} mm "
        f"({st_v2.get('value_uv')} uV) -> {st_v2.get('class')}"
        f"\n    T-wave polaridade    = {t_v2.get('polarity')}  amp={t_v2.get('amplitude_uv')} uV"
        f"\n    Q-wave patologica    = {q_v2.get('pathological')}  "
        f"dur={q_v2.get('duration_ms')}ms  amp={q_v2.get('amplitude_uv')} uV"
        f"\n    QRS R amplitude      = {qrs_a_v2.get('r_mm')} mm ({qrs_a_v2.get('r_uv')} uV)"
        f"\n    QRS S amplitude      = {qrs_a_v2.get('s_mm')} mm ({qrs_a_v2.get('s_uv')} uV)"
        f"\n    QRS net (signed)     = {qrs_a_v2.get('net_uv')} uV"
        f"\n    R/S ratio (V1/V6)    = {rs.get('V1')} / {rs.get('V6')}"
    )

    # --- 7. Regras clinicas ---
    rule_findings = apply_clinical_rules(measurements)
    v2_rules = [
        f for f in rule_findings
        if (LEAD in (f.get("leads_affected") or []))
        or LEAD in (f.get("description") or "")
    ]
    print(f"\n[7] Regras clinicas que tocam {LEAD} ({len(v2_rules)}):")
    for f in v2_rules:
        affected = ", ".join(f.get("leads_affected") or [])
        print(f"    [{f['code']}] {f['description']}  ({affected})")
    print(f"\n    Outras regras (nao-{LEAD}): {len(rule_findings) - len(v2_rules)}")
    for f in rule_findings:
        if f in v2_rules:
            continue
        print(f"      [{f['code']}] {f['description']}")

    # --- 8. CNN ---
    signal_12 = _signals_dict_to_array(sep["leads"])
    try:
        cnn_findings = classify_ecg(signal_12, fs_in=fs)
        cnn_probs = classify_ecg_full(signal_12, fs_in=fs)
    except Exception as e:
        print(f"\n[8] CNN falhou: {type(e).__name__}: {e}")
        cnn_findings, cnn_probs = [], {}

    if cnn_probs:
        top = sorted(cnn_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"\n[8] CNN top-5:")
        for code, p in top:
            desc = CLASS_DESCRIPTIONS.get(code, code)
            mark = " (>=0.5)" if p >= 0.5 else ""
            print(f"    {p:.2f}{mark}  [{code}] {desc}")

    # --- 9. Laudo final ---
    out = generate_frontend_report(measurements, rule_findings, cnn_findings)
    print(f"\n[9] LAUDO FINAL:")
    print(f"    Severidade : {out['severity'].upper()}")
    if out["red_flags"]:
        print(f"    Red flags  : {' | '.join(out['red_flags'])}")
    print(f"    Diagnosticos: {len(out['diagnoses'])}")
    for d in out["diagnoses"]:
        sev = d.get("severity", "low")
        print(
            f"      [{sev}] [{d['code']}] (conf={d['confidence']:.2f}, src={d['source']}) "
            f"{d['name']}"
        )
    print(f"\n--- texto do laudo ---")
    print(out["text_report"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
