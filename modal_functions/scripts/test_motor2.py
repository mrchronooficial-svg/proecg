"""
test_motor2.py — Testa Motor 2 (Claude Vision) com foto real de ECG.
Uso: python scripts/test_motor2.py [caminho/ecg.jpg]
"""

import json
import os
import sys
import time

# Carregar .env.local para ANTHROPIC_API_KEY
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for env_file in [".env.local", ".env", "apps/web/.env"]:
    env_path = os.path.join(_project_root, env_file)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _modal_dir)

from PIL import Image


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../docs/ECG_teste.jpeg"

    if not os.path.exists(path):
        print(f"Arquivo nao encontrado: {path}")
        sys.exit(1)

    # Verificar API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERRO: ANTHROPIC_API_KEY nao definida")
        print("Defina em .env.local ou como variavel de ambiente")
        sys.exit(1)
    print(f"API Key: {api_key[:12]}...{api_key[-4:]}")

    img = Image.open(path)
    print(f"Imagem: {path} ({img.size[0]}x{img.size[1]} {img.mode})")
    print()

    # Rodar Motor 2
    print("=" * 70)
    print("MOTOR 2 — Claude Vision API")
    print("=" * 70)

    from pipeline.motor2 import motor2_analyze

    t0 = time.perf_counter()
    result = motor2_analyze(img)
    elapsed = time.perf_counter() - t0

    print(f"\nTempo: {elapsed:.1f}s")
    print(f"Engine: {result.get('engine', '?')}")
    print(f"Success: {result.get('success', '?')}")

    if result.get("error"):
        print(f"ERRO: {result['error']}")
        if result.get("raw_response"):
            print(f"Raw: {result['raw_response'][:300]}")
        sys.exit(1)

    tokens = result.get("api_tokens", {})
    if tokens:
        print(f"Tokens: input={tokens.get('input', '?')}, output={tokens.get('output', '?')}")

    # Technical params
    tp = result.get("technical_params", {})
    if tp:
        print(f"\nTechnical: speed={tp.get('paper_speed_mm_s')} mm/s, "
              f"gain={tp.get('gain_mm_mv')} mm/mV, "
              f"layout={tp.get('layout')}, "
              f"quality={tp.get('quality')}, "
              f"orientation={tp.get('orientation')}")

    # Measurements
    m = result.get("measurements", {})
    print(f"\n{'='*70}")
    print("MEDICOES:")
    print(f"{'='*70}")
    print(f"  FC:      {m.get('heart_rate')} {m.get('heart_rate_unit', 'bpm')}")
    print(f"  Ritmo:   {m.get('rhythm')} ({m.get('rhythm_detail', '')})")
    print(f"  Eixo:    {m.get('axis')}° ({m.get('axis_classification', '')})")
    print(f"  PR:      {m.get('pr_interval')} {m.get('pr_unit', 'ms')}")
    print(f"  QRS:     {m.get('qrs_duration')} {m.get('qrs_unit', 'ms')}")
    print(f"  QT:      {m.get('qt_interval')} {m.get('qt_unit', 'ms')}")
    print(f"  QTc:     {m.get('qtc_bazett')} {m.get('qtc_unit', 'ms')}")

    # Findings
    findings = result.get("findings", [])
    print(f"\n{'='*70}")
    print(f"ACHADOS ({len(findings)}):")
    print(f"{'='*70}")
    for f in findings:
        leads = ", ".join(f.get("leads_affected", []))
        sev = f.get("severity", "")
        print(f"  [{sev:>10s}] {f.get('code', '?')}")
        print(f"             {f.get('description', '')}")
        if leads:
            print(f"             Derivacoes: {leads}")
        print()

    # Diagnoses
    diags = result.get("diagnoses", [])
    print(f"{'='*70}")
    print(f"DIAGNOSTICOS ({len(diags)}):")
    print(f"{'='*70}")
    for d in diags:
        urg = d.get("urgency", "")
        cat = d.get("category", "")
        print(f"  [{urg:>11s}] [{cat}] {d.get('code', '?')}")
        print(f"              {d.get('description', '')}")
        print()

    # Report text
    print(f"{'='*70}")
    print("LAUDO:")
    print(f"{'='*70}")
    print(result.get("report_text", "(vazio)"))

    # JSON completo
    print(f"\n{'='*70}")
    print("JSON COMPLETO:")
    print(f"{'='*70}")
    # Remover raw_response e api_tokens do output limpo
    clean = {k: v for k, v in result.items() if k not in ("raw_response", "api_tokens")}
    print(json.dumps(clean, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
