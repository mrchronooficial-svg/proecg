"""
Roda o pipeline COMPLETO (digitalizacao + medicoes + regras + CNN + laudo)
no IMG_1473 e imprime o laudo final.
"""

import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

from pipeline.orchestrator import analyze_from_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_laudo")

IMAGE_PATH = r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3\IMG_1473.jpg"
OUTPUT_PATH = Path(
    r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1\IMG_1473\laudo.txt"
)


def main() -> int:
    logger.info("Rodando pipeline completo no %s", IMAGE_PATH)
    t0 = time.perf_counter()
    result = analyze_from_file(IMAGE_PATH, use_mock=False)
    elapsed = time.perf_counter() - t0
    logger.info("Pipeline concluido em %.1fs", elapsed)

    if not result.get("success"):
        logger.error("Pipeline falhou: %s", result.get("error"))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 1

    # Imprime laudo no console + salva em arquivo
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 70)
    lines.append(f"LAUDO ECG — IMG_1473")
    lines.append("=" * 70)
    lines.append("")
    lines.append("MEDIDAS:")
    m = result.get("measurements", {})
    for key, label in [
        ("heart_rate", "FC"),
        ("axis", "Eixo"),
        ("pr_interval", "PR"),
        ("qrs_duration", "QRS"),
        ("qt_interval", "QT"),
        ("qtc_bazett", "QTc (Bazett)"),
        ("rhythm", "Ritmo"),
        ("p_wave_present", "Onda P"),
    ]:
        if key in m:
            unit = m.get(f"{key.split('_')[0]}_unit", "")
            lines.append(f"  {label}: {m.get(key)} {unit}")
    st = m.get("st_segment", {})
    if st:
        lines.append(f"  Segmento ST: {st}")
    lines.append("")
    lines.append("ACHADOS:")
    for f in result.get("findings", []):
        lines.append(f"  - {f.get('label', f.get('code', '?'))}")
    lines.append("")
    lines.append("DIAGNOSTICOS:")
    for d in result.get("diagnoses", []):
        lines.append(f"  - {d.get('label', d.get('code', '?'))}")
    rf = result.get("red_flags", [])
    if rf:
        lines.append("")
        lines.append("RED FLAGS (urgencia):")
        for r in rf:
            lines.append(f"  - {r}")
    lines.append("")
    lines.append("SEVERIDADE: " + str(result.get("severity", "n/a")))
    lines.append("")
    lines.append("METADATA:")
    md = result.get("metadata", {})
    for k, v in md.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("LAUDO COMPLETO:")
    lines.append("-" * 70)
    lines.append(result.get("report_text", "(vazio)"))
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"Processing time: {result.get('processing_time_ms', 0)} ms")

    text = "\n".join(lines)
    print(text)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    logger.info("Laudo salvo em %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
