"""
test_full_report.py — Roda pipeline completo e mostra report JSON.
Uso: python scripts/test_full_report.py caminho/ecg.jpg
"""

import json
import os
import sys
import time

_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _modal_dir)

import numpy as np
from PIL import Image

from pipeline.digitize import digitize_ecg

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../docs/ECG_teste.jpeg"
    img = Image.open(path)
    print(f"Imagem: {path} ({img.size[0]}x{img.size[1]} {img.mode})")
    print()

    # 1. Rodar digitize_ecg
    print("=" * 60)
    print("DIGITIZE_ECG")
    print("=" * 60)
    t0 = time.perf_counter()
    signal = digitize_ecg(img)
    dt = time.perf_counter() - t0
    print(f"Tempo: {dt:.2f}s")
    print(f"Shape: {signal.shape}, dtype: {signal.dtype}, NaN: {np.any(np.isnan(signal))}")
    print()

    print(f"{'Lead':>5s}  {'Min':>9s}  {'Max':>9s}  {'Std':>8s}  Ativo")
    print("-" * 55)
    for i, name in enumerate(LEAD_NAMES):
        row = signal[i]
        active = np.any(row != 0)
        print(f"{name:>5s}  {row.min():9.4f}  {row.max():9.4f}  {row.std():8.4f}  {'sim' if active else '---'}")
    print()

    # 2. Rodar pipeline completo (measure + rules + classify + report)
    print("=" * 60)
    print("ORCHESTRATOR — FULL PIPELINE")
    print("=" * 60)

    from pipeline.measure import measure_ecg
    from pipeline.rules import apply_clinical_rules
    from pipeline.classify import classify_ecg
    from pipeline.report import generate_report

    measurements = measure_ecg(signal, fs=500)
    print("\nMeasurements:")
    for k, v in measurements.items():
        print(f"  {k}: {v}")

    rule_findings = apply_clinical_rules(measurements)
    print(f"\nRule findings ({len(rule_findings)}):")
    for f in rule_findings:
        print(f"  [{f.get('code')}] {f.get('description')} (source={f.get('source')})")

    cnn_findings = classify_ecg(signal)
    print(f"\nCNN findings ({len(cnn_findings)}):")
    for f in cnn_findings:
        print(f"  [{f.get('code')}] {f.get('description')} (score={f.get('score', 'N/A')})")

    report = generate_report(measurements, rule_findings, cnn_findings)
    print(f"\nDiagnoses ({len(report.get('diagnoses', []))}):")
    for d in report.get("diagnoses", []):
        print(f"  [{d.get('code')}] {d.get('description')}")

    print("\n" + "=" * 60)
    print("REPORT TEXT:")
    print("=" * 60)
    print(report.get("report_text", "(vazio)"))

    # 3. JSON completo
    print("\n" + "=" * 60)
    print("JSON OUTPUT (contrato):")
    print("=" * 60)

    import math

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if (math.isnan(val) or math.isinf(val)) else val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    output = sanitize({
        "success": True,
        "measurements": measurements,
        "findings": [{k: v for k, v in f.items() if k != "score"} for f in report.get("findings", [])],
        "diagnoses": report.get("diagnoses", []),
        "report_text": report.get("report_text", ""),
    })

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
