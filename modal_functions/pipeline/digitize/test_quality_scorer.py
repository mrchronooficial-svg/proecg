"""
Teste do Quality Scorer
=======================

Roda em 10 imagens da pasta "ECGs Reais3" (mostra como o scorer trata
fotos reais variadas).

Uso:
    cd proecg
    python -m modal_functions.pipeline.digitize.test_quality_scorer
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from .quality_scorer import score_quality

ECGS_REAIS3_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Reais3")
N_IMAGES = 10
RANDOM_SEED = 42


def _print_result(label: str, img: np.ndarray, result: dict) -> None:
    h, w = img.shape[:2]
    decision = "ACEITA " if result["accept"] else "REJEITA"
    print(f"\n{'-' * 78}")
    print(f"  {decision}  |  {label}  ({w}x{h})")
    print(f"{'-' * 78}")
    print(f"    score:    {result['score']:.3f}")
    if result["rejection_reason"]:
        print(f"    razão:    {result['rejection_reason']}")
    if result["warning"]:
        print(f"    warning:  {result['warning']}")

    print(f"    details:")
    d = result["details"]
    if "resolution" in d:
        r = d["resolution"]
        print(
            f"      resolução       pass={r['pass']!s:5s}  "
            f"longer={r['longer_side']}px"
        )
    if "ecg_detected" in d:
        e = d["ecg_detected"]
        print(
            f"      ecg detectado    pass={e['pass']!s:5s}  "
            f"score={e['periodic_score']:.3f}  "
            f"(h={e['h_score']:.3f}, v={e['v_score']:.3f})"
        )
    if "focus" in d:
        f = d["focus"]
        print(
            f"      foco             pass={f['pass']!s:5s}  "
            f"laplacian_var={f['laplacian_var']:.1f}"
        )
    if "exposure" in d:
        ex = d["exposure"]
        print(
            f"      exposição        pass={ex['pass']!s:5s}  "
            f"brilho={ex['mean_brightness']:.1f}"
        )
    if "aspect_ratio" in d:
        a = d["aspect_ratio"]
        print(
            f"      proporção        pass={a['pass']!s:5s}  "
            f"ratio={a['ratio']:.2f}"
        )


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s"
    )

    if not ECGS_REAIS3_DIR.exists():
        print(f"ERRO: pasta não encontrada: {ECGS_REAIS3_DIR}", file=sys.stderr)
        return 1

    # Pega todas as imagens .jpg/.JPG/.jpeg/.png da pasta
    all_imgs = sorted(
        p for p in ECGS_REAIS3_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not all_imgs:
        print(f"ERRO: nenhuma imagem em {ECGS_REAIS3_DIR}", file=sys.stderr)
        return 1

    # Seleção determinística de N_IMAGES (com seed)
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(all_imgs, k=min(N_IMAGES, len(all_imgs)))

    print(f"\n{'=' * 78}")
    print(f"Pasta: {ECGS_REAIS3_DIR}")
    print(f"Total de imagens disponíveis: {len(all_imgs)}")
    print(f"Amostra (seed={RANDOM_SEED}): {len(sample)} imagens")
    print(f"{'=' * 78}")

    scenarios: list[tuple[str, np.ndarray]] = []
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            print(f"AVISO: não abriu {p.name}", file=sys.stderr)
            continue
        scenarios.append((p.name, img))

    # ---------- Executa ----------
    print(f"\n{'=' * 78}")
    print(f"QUALITY SCORER — TESTE EM {len(scenarios)} IMAGENS REAIS")
    print(f"{'=' * 78}")

    results: list[tuple[str, dict]] = []
    for label, img in scenarios:
        try:
            r = score_quality(img)
        except Exception as e:
            r = {
                "accept": False,
                "score": 0.0,
                "rejection_reason": f"ERRO {type(e).__name__}: {e}",
                "warning": None,
                "details": {},
            }
        _print_result(label, img, r)
        results.append((label, r))

    # ---------- Resumo ----------
    print(f"\n{'=' * 78}")
    print("RESUMO")
    print(f"{'=' * 78}")
    print(f"  {'Cenário':50s}  {'Score':>6s}  Decisão")
    print(f"  {'-' * 50}  {'-' * 6}  {'-' * 30}")
    for label, r in results:
        decision = "ACEITA" if r["accept"] else "REJEITA"
        if r["accept"] and r["warning"]:
            decision += " (com warning)"
        # Trunca label longo
        short = label if len(label) <= 50 else label[:47] + "..."
        print(f"  {short:50s}  {r['score']:>6.3f}  {decision}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
