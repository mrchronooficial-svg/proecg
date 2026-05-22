"""
Roda extract_signals_stenhede + measure_ecg + apply_clinical_rules +
classify_ecg_full em IMG_1303, IMG_1378 e IMG_1405 e imprime tudo
formatado.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from .classify import CLASS_DESCRIPTIONS, CLASS_NAMES, classify_ecg_full
from .digitize.stenhede_adapter import (
    LEAD_CHANNEL_ORDER,
    extract_signals_stenhede,
)
from .measure import measure_ecg
from .rules import apply_clinical_rules

UNDIST_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Undistorted")
# Permite filtrar via CLI: python -m ... IMG_1405
_argv = sys.argv[1:]
TARGETS = _argv if _argv else ["IMG_1303", "IMG_1378", "IMG_1405"]


def _signals_dict_to_array_mv(
    signals: dict[str, np.ndarray], target_n: int,
) -> np.ndarray:
    """Converte dict {lead_name: array em uV} em np.ndarray (12, N) em mV
    pra classify_ecg (que espera mV)."""
    out = np.zeros((12, target_n), dtype=np.float64)
    for i, name in enumerate(LEAD_CHANNEL_ORDER):
        if name not in signals:
            continue
        sig = signals[name]
        valid = ~np.isnan(sig)
        if valid.sum() < 2:
            continue
        clean = np.where(valid, sig, 0.0)
        n = min(len(clean), target_n)
        out[i, :n] = clean[:n] / 1000.0  # uV -> mV
    return out


def _process(stem: str) -> int:
    img_path = UNDIST_DIR / f"{stem}.png"
    if not img_path.exists():
        print(f"\n=== {stem} ===\n[ERRO] {img_path} nao existe")
        return 1
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"\n=== {stem} ===\n[ERRO] falha ao ler {img_path}")
        return 2

    print(f"\n=== {stem} ===")

    # 1. Stenhede
    try:
        result = extract_signals_stenhede(
            image_bgr=img, use_cropper=False, use_internal_pixel_size=True,
        )
    except Exception as e:
        print(f"[ERRO Stenhede] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 3
    fs = float(result["sampling_rate_hz"])
    signals = result["signals"]

    # 2. measure_ecg
    try:
        m = measure_ecg(signals, fs=int(round(fs)))
    except Exception as e:
        print(f"[ERRO measure] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 4

    hr = (m.get("heart_rate") or {}).get("mean_bpm")
    iv = m.get("intervals") or {}
    pr = iv.get("pr_ms"); qrs = iv.get("qrs_ms")
    qt = iv.get("qt_ms"); qtc = iv.get("qtc_ms")

    def _f(v):
        return f"{v:.0f}" if isinstance(v, (int, float)) and v is not None else "-"

    print(
        f"Medições: FC={_f(hr)}, PR={_f(pr)}, QRS={_f(qrs)}, "
        f"QT={_f(qt)}, QTc={_f(qtc)}"
    )

    # 3. apply_clinical_rules
    try:
        rule_findings = apply_clinical_rules(m)
    except Exception as e:
        print(f"[ERRO rules] {type(e).__name__}: {e}")
        rule_findings = []
    if rule_findings:
        print(f"Regras: ({len(rule_findings)} achados)")
        for f in rule_findings:
            affected = ", ".join(f.get("leads_affected") or [])
            extra = f"  [{affected}]" if affected else ""
            print(f"  • [{f['code']}] {f['description']}{extra}")
    else:
        print("Regras: (sem achados)")

    # 4. classify_ecg_full
    target_n = 5000  # 10s @ 500Hz é o esperado pela CNN
    sig_array = _signals_dict_to_array_mv(signals, target_n)
    try:
        probs = classify_ecg_full(sig_array, fs_in=int(round(fs)))
    except Exception as e:
        print(f"[ERRO CNN] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 5

    print("CNN (todas as 13 probabilidades):")
    for cn in CLASS_NAMES:
        p = probs.get(cn, 0.0)
        marker = "  >0.5" if p >= 0.5 else ""
        print(f"  {cn:<14} {p:.3f}{marker}")

    above = [(cn, probs[cn]) for cn in CLASS_NAMES if probs.get(cn, 0.0) >= 0.5]
    above.sort(key=lambda x: x[1], reverse=True)
    if above:
        labels = [
            f"{cn} ({p:.2f}) — {CLASS_DESCRIPTIONS.get(cn, cn)}"
            for cn, p in above
        ]
        print(f"CNN (acima de 0.5):")
        for s in labels:
            print(f"  • {s}")
    else:
        print("CNN (acima de 0.5): (nenhum)")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    print("=" * 78)
    print(" Stenhede + measure + rules + CNN — IMG_1303, IMG_1378, IMG_1405")
    print("=" * 78)
    rc = 0
    for stem in TARGETS:
        rc = _process(stem) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
