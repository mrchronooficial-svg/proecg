"""
Teste do measure_ecg em sinais reais
====================================

Pipeline:
  1. Carrega máscara binária + imagem normalizada (Leader Masks Teste)
  2. Roda calibrador → px_per_mm, sampling_rate, uv_per_pixel
  3. Roda lead_separator → 12 sinais em µV
  4. Roda measure_ecg → medições completas
  5. Imprime FC, eixo, intervalos, ST por derivação

Uso:
    python -m modal_functions.pipeline.test_measure
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.format_detector import detect_layout
from .digitize.lead_separator import separate_and_extract
from .measure import measure_ecg

NORMALIZED_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Normalizados Leader")
MASKS_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\Leader Masks Teste")
TARGETS = ["IMG_1275.png", "IMG_1279.png", "IMG_1303.png"]


def _calibrate_normalized(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    grid_mask, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        grid_mask, keypoints = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _process_one(image_path: Path, mask_path: Path) -> None:
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"  ERRO ao carregar {image_path.name}")
        return
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    print(f"\n{'=' * 78}")
    print(f"IMAGEM: {image_path.name}  ({img.shape[1]}x{img.shape[0]})")
    print(f"{'=' * 78}")

    # 1. Calibração
    cal = _calibrate_normalized(img)
    print(
        f"  Calibração: px/mm={cal['px_per_mm']:.2f}  "
        f"sr={cal['sampling_rate_hz']:.0f}Hz  "
        f"uV/px={cal['uv_per_pixel']:.2f}  "
        f"ganho={cal['gain_mm_per_mV']:.0f} mm/mV"
    )

    # 2. Lead separation
    sep = separate_and_extract(mask, img, cal)
    print(
        f"  Layout: {sep['layout']}  ({len(sep['leads'])} derivações)"
    )

    # 3. Construir dict de sinais (excluir o II_rhythm pra medições padrão)
    leads_signals = {
        name: info["signal_uv"]
        for name, info in sep["leads"].items()
        if name != "II_rhythm"
    }
    # Se faltar derivação importante (II), tenta usar o rhythm
    if "II" not in leads_signals and "II_rhythm" in sep["leads"]:
        leads_signals["II"] = sep["leads"]["II_rhythm"]["signal_uv"]

    # 4. Medições
    fs = int(round(cal["sampling_rate_hz"]))
    try:
        m = measure_ecg(leads_signals, fs=fs, calibration=cal)
    except Exception as e:
        print(f"  ERRO measure_ecg: {type(e).__name__}: {e}")
        return

    # 5. Resultados
    hr = m["heart_rate"]
    print(f"\n  --- Frequência Cardíaca ---")
    print(
        f"    Mean={hr['mean_bpm']} bpm  "
        f"Min={hr['min_bpm']} bpm  "
        f"Max={hr['max_bpm']} bpm  "
        f"Regular={hr['regular']}"
    )
    if hr["rr_intervals_ms"]:
        rr_str = ", ".join(f"{x:.0f}" for x in hr["rr_intervals_ms"][:8])
        print(f"    RR (ms): [{rr_str}{'...' if len(hr['rr_intervals_ms']) > 8 else ''}]")

    ax = m["axis"]
    print(f"\n  --- Eixo Elétrico ---")
    print(f"    Degrees={ax['degrees']}°  Class={ax['classification']}")

    iv = m["intervals"]
    print(f"\n  --- Intervalos ---")
    print(
        f"    PR={iv['pr_ms']}ms  QRS={iv['qrs_ms']}ms  "
        f"QT={iv['qt_ms']}ms  QTc={iv['qtc_ms']}ms  "
        f"PP={iv['pp_ms']}ms"
    )

    pw = m["p_wave"]
    print(f"\n  --- Onda P ---")
    print(
        f"    Present={pw['present']}  "
        f"Duration={pw['duration_ms']}ms  "
        f"Amplitude={pw['amplitude_uv']}µV  "
        f"Bifásica V1={pw['bifasic_v1']}"
    )

    print(f"\n  --- Segmento ST (por derivação) ---")
    st = m["st_segment"]
    for name in ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]:
        if name in st:
            d = st[name]
            mm = d["value_mm"]
            cls = d["class"]
            mm_str = f"{mm:+.2f}" if mm is not None else "  N/A"
            print(f"    {name:4s}: {mm_str} mm  ({cls})")

    print(f"\n  --- Onda T (polaridade) ---")
    tw = m["t_wave"]
    polarities = "  ".join(
        f"{n}:{tw[n]['polarity'][:3]}"
        for n in ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]
        if n in tw
    )
    print(f"    {polarities}")

    print(f"\n  --- Onda Q patológica ---")
    qw = m["q_wave"]
    path_leads = [n for n in qw if qw[n].get("pathological")]
    if path_leads:
        print(f"    Detectada em: {', '.join(path_leads)}")
    else:
        print(f"    Nenhuma derivação com Q patológica")

    rs = m["r_s_ratio"]
    print(f"\n  --- Relação R/S ---")
    print(f"    V1={rs['V1']}  V6={rs['V6']}")

    q = m["quality"]
    print(f"\n  --- Qualidade ---")
    print(
        f"    P_conf={q['p_wave_confidence']}  "
        f"T_conf={q['t_wave_confidence']}  "
        f"Overall={q['overall']}"
    )

    if m["warnings"]:
        print(f"\n  --- Warnings ---")
        for w in m["warnings"]:
            print(f"    • {w}")


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s"
    )

    if not MASKS_DIR.exists() or not NORMALIZED_DIR.exists():
        print("ERRO: pastas não encontradas", file=sys.stderr)
        return 1

    for name in TARGETS:
        img_path = NORMALIZED_DIR / name
        mask_path = MASKS_DIR / name
        if not img_path.exists() or not mask_path.exists():
            print(f"AVISO: faltam arquivos pra {name}", file=sys.stderr)
            continue
        _process_one(img_path, mask_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
