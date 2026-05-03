"""
Teste end-to-end completo: extração → medições → regras → CNN → laudo + viz
============================================================================

Pipeline completo nas 3 imagens (IMG_1275, IMG_1279, IMG_1303):
  1. Calibrador (px_per_mm, fs)
  2. Lead separator → 12 sinais µV
  3. measure_ecg → medições estruturadas
  4. apply_clinical_rules → achados por regra
  5. classify_ecg_full → probabilidades CNN
  6. generate_frontend_report → laudo SBC + JSON estruturado
  7. Visualização: retângulo VERMELHO/AMARELO por achado/diagnóstico

Uso:
    python -m modal_functions.pipeline.test_full_pipeline
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .classify import CLASS_DESCRIPTIONS, classify_ecg, classify_ecg_full
from .digitize.calibrator import calibrate
from .digitize.ecg_digitizer import ECGDigitizer
from .digitize.lead_separator import LAYOUT_3x4, separate_and_extract
from .measure import LEAD_NAMES, measure_ecg
from .report import generate_frontend_report
from .rules import apply_clinical_rules

NORMALIZED_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Normalizados Leader")
MASKS_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\Leader Masks Teste")
OUT_DIR = Path("modal_functions/pipeline/digitize/_visualizations")
TARGETS = ["IMG_1275.png", "IMG_1279.png", "IMG_1303.png"]

# Mapeamento de severidade pra cor + thickness do retângulo
SEVERITY_STYLE = {
    "critical": {"color": (0, 0, 255), "thickness": 3},  # vermelho grosso
    "high":     {"color": (0, 0, 255), "thickness": 3},
    "moderate": {"color": (0, 200, 255), "thickness": 2},  # amarelo
    "low":      None,  # não marca
}


def _calibrate_normalized(img: np.ndarray) -> dict:
    digitizer = ECGDigitizer(use_mock=False)
    grid_mask, keypoints = digitizer.dotter(img)
    if len(keypoints) < 4:
        grid_mask, keypoints = digitizer.dotter_mock(img)
    grid_matrix, _ = digitizer.gridder(keypoints, img.shape[:2])
    return calibrate(grid_matrix=grid_matrix, normalized_image=img)


def _signals_dict_to_array(leads: dict[str, dict]) -> np.ndarray:
    """Converte dict {lead_name: {signal_uv}} em array (12, N) em mV."""
    out = np.zeros((12, 1), dtype=np.float64)
    max_n = max(
        (info["signal_uv"].shape[0] for info in leads.values()),
        default=1,
    )
    out = np.zeros((12, max_n), dtype=np.float64)
    for i, name in enumerate(LEAD_NAMES):
        if name in leads:
            sig = leads[name]["signal_uv"]
            sig = np.where(np.isnan(sig), 0.0, sig)  # NaN → 0
            n = min(len(sig), max_n)
            # uV → mV
            out[i, :n] = sig[:n] / 1000.0
    return out


def _resolve_lead_bbox(
    name: str, separation: dict
) -> tuple[int, int, int, int] | None:
    """Bbox de um lead na imagem normalizada (x1, y1, x2, y2)."""
    leads = separation.get("leads", {})
    if name in leads:
        return tuple(leads[name]["bbox"])
    return None


def _draw_findings_overlay(
    image: np.ndarray,
    separation: dict,
    measurements: dict,
    diagnoses: list[dict],
    red_flags: list[str],
) -> np.ndarray:
    """Pinta retângulos vermelhos/amarelos sobre as derivações afetadas."""
    out = image.copy()
    h, w = out.shape[:2]

    # Acumula label por bbox (para concatenar quando há múltiplos achados na mesma derivação)
    labels_by_lead: dict[str, list[tuple[str, str]]] = {}

    # ST por derivação a partir de measurements
    st = (measurements.get("st_segment") or {})
    for lead_new, info in st.items():
        cls = info.get("class")
        if cls in ("supra", "infra"):
            sev = "critical" if cls == "supra" else "moderate"
            mm = info.get("value_mm")
            mm_str = f"{mm:+.1f}mm" if mm is not None else ""
            label = f"{'Supra' if cls == 'supra' else 'Infra'} ST {mm_str}".strip()
            labels_by_lead.setdefault(lead_new, []).append((sev, label))

    # Q patológica por derivação
    qw = (measurements.get("q_wave") or {})
    for lead_new, info in qw.items():
        if info.get("pathological"):
            dur = info.get("duration_ms")
            label = f"Q patológica {dur:.0f}ms" if dur else "Q patológica"
            labels_by_lead.setdefault(lead_new, []).append(("moderate", label))

    # QRS alargado: anota uma vez no DII
    intervals = measurements.get("intervals") or {}
    qrs = intervals.get("qrs_ms")
    if qrs is not None and qrs >= 120.0:
        sev = "critical" if qrs > 160.0 else "moderate"
        labels_by_lead.setdefault("II", []).append((sev, f"QRS alargado {qrs:.0f}ms"))

    # PR longo
    pr = intervals.get("pr_ms")
    if pr is not None and pr > 200.0:
        labels_by_lead.setdefault("II", []).append(("moderate", f"PR longo {pr:.0f}ms"))

    # Ondas T anormais (negativas onde deveria ser positiva)
    tw = (measurements.get("t_wave") or {})
    expected_negative = {"aVR"}
    for lead_new, info in tw.items():
        polarity = info.get("polarity")
        if polarity == "negative" and lead_new not in expected_negative:
            labels_by_lead.setdefault(lead_new, []).append(
                ("moderate", "T invertida")
            )

    # Sobrecarga ventricular esquerda — marca em V1, V5, V6
    has_lvh = any(d.get("code") == "LVH_SOKOLOW" for d in diagnoses)
    if has_lvh:
        for ld in ("V1", "V5", "V6"):
            labels_by_lead.setdefault(ld, []).append(
                ("moderate", "SVE (Sokolow)")
            )

    # SVD — V1
    if any(d.get("code") == "RVH" for d in diagnoses):
        labels_by_lead.setdefault("V1", []).append(("moderate", "SVD R/S>1"))

    # SAE — V1
    if any(d.get("code") == "LAE" for d in diagnoses):
        labels_by_lead.setdefault("V1", []).append(("moderate", "SAE"))

    # Desenhar
    for lead_new, items in labels_by_lead.items():
        bbox = _resolve_lead_bbox(lead_new, separation)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Severidade max do bbox = a mais grave
        sev_priority = {"critical": 3, "high": 2, "moderate": 1, "low": 0}
        max_sev = max(items, key=lambda kv: sev_priority.get(kv[0], 0))[0]
        style = SEVERITY_STYLE.get(max_sev)
        if style is None:
            continue

        # Retângulo + overlay translúcido
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), style["color"], -1)
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), style["color"], style["thickness"])

        # Labels (empilhados sobre o bbox)
        text_y = max(20, y1 - 8)
        for sev_lvl, label in items:
            color = SEVERITY_STYLE[sev_lvl]["color"]
            cv2.putText(
                out, f"{lead_new}: {label}",
                (x1 + 4, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            text_y += 22

    # Cabeçalho com red flags / severidade
    if red_flags:
        rf_label = "RED FLAGS: " + " | ".join(red_flags)
        cv2.rectangle(out, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(
            out, rf_label, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 80, 255), 2,
        )

    return out


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

    print(f"\n{'#' * 80}")
    print(f"# {image_path.name}")
    print(f"{'#' * 80}")

    # 1. Calibração + 2. Separação
    cal = _calibrate_normalized(img)
    sep = separate_and_extract(mask, img, cal)
    leads_signals = {
        n: i["signal_uv"]
        for n, i in sep["leads"].items() if n != "II_rhythm"
    }
    if "II" not in leads_signals and "II_rhythm" in sep["leads"]:
        leads_signals["II"] = sep["leads"]["II_rhythm"]["signal_uv"]
    fs = int(round(cal["sampling_rate_hz"]))

    # 3. Medições
    measurements = measure_ecg(leads_signals, fs=fs, calibration=cal)

    # 4. Regras clínicas
    rule_findings = apply_clinical_rules(measurements)

    # 5. CNN — converte sinais pra array (12, N) em mV
    signal_12lead_mv = _signals_dict_to_array(sep["leads"])
    try:
        cnn_findings = classify_ecg(signal_12lead_mv, fs_in=fs)
        cnn_probs = classify_ecg_full(signal_12lead_mv, fs_in=fs)
    except Exception as e:
        print(f"  AVISO CNN falhou: {type(e).__name__}: {e}")
        cnn_findings, cnn_probs = [], {}

    # 6. Laudo final (frontend JSON)
    out = generate_frontend_report(measurements, rule_findings, cnn_findings)

    # ---- Output stdout ----
    sev_icon = {"critical": "🚨", "moderate": "⚠️", "normal": "✅"}.get(
        out["severity"], ""
    )
    print(f"\nSEVERIDADE: {sev_icon} {out['severity'].upper()}")

    if out["red_flags"]:
        print(f"\n--- 🚨 RED FLAGS ({len(out['red_flags'])}) ---")
        for rf in out["red_flags"]:
            print(f"  • {rf}")

    if cnn_probs:
        # Top 5 da CNN
        top = sorted(cnn_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"\n--- CNN top-5 ---")
        for code, p in top:
            desc = CLASS_DESCRIPTIONS.get(code, code)
            mark = " ✓" if p >= 0.5 else ""
            print(f"  {p:.2f}{mark}  [{code}] {desc}")

    print(f"\n--- ACHADOS COMBINADOS ({len(out['_findings_raw'])}) ---")
    for f in out["_findings_raw"]:
        src = f.get("source", "?")
        conf = f.get("confidence", 1.0)
        affected = ", ".join(f.get("leads_affected") or [])
        suffix = f"  [{affected}]" if affected else ""
        print(f"  • [{f['code']}] ({src}, conf={conf:.2f}) {f['description']}{suffix}")

    print(f"\n--- DIAGNÓSTICOS ({len(out['diagnoses'])}) ---")
    for d in out["diagnoses"]:
        sev = d.get("severity", "low")
        marker = {"critical": "🚨", "moderate": "⚠️ ", "low": "  "}.get(sev, "  ")
        print(
            f"  {marker} [{d['code']}] (conf={d['confidence']:.2f}, src={d['source']}, sev={sev}) "
            f"{d['name']}"
        )

    if out["warnings"]:
        print(f"\n--- WARNINGS ---")
        for w in out["warnings"]:
            print(f"  • {w}")

    print(f"\n--- LAUDO TEXTO ---")
    print(out["text_report"])

    # ---- Visualização anotada ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated = _draw_findings_overlay(
        img, sep, measurements, out["diagnoses"], out["red_flags"]
    )
    out_path = OUT_DIR / f"{image_path.stem}_report_viz.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n  Visualização salva em: {out_path}")


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    if not MASKS_DIR.exists() or not NORMALIZED_DIR.exists():
        print("ERRO: pastas não encontradas", file=sys.stderr)
        return 1

    for name in TARGETS:
        ip = NORMALIZED_DIR / name
        mp = MASKS_DIR / name
        if not ip.exists() or not mp.exists():
            print(f"AVISO: faltam arquivos pra {name}", file=sys.stderr)
            continue
        _process_one(ip, mp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
