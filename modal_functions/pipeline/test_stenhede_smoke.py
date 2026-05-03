"""
Smoke-test do pipeline Stenhede em IMG_1279.
=============================================

Valida o caminho completo de digitização usando o U-Net + SignalExtractor
do Stenhede (vendorizado em modal_functions/vendor/open_ecg_digitizer/):

  1. ECGDigitizer.run(image_path) — pipeline completo (NOVO caminho)
  2. Imprime shape + range de cada lead
  3. Conta NaN ratio por lead
  4. Verifica que dict tem 12 leads + II_rhythm + (compat) II_long

Uso:
    python -m modal_functions.pipeline.test_stenhede_smoke
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from .digitize.ecg_digitizer import ECGDigitizer

IMG_PATH = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Normalizados Leader\IMG_1279.png")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    if not IMG_PATH.exists():
        print(f"ERRO: imagem não encontrada: {IMG_PATH}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f" SMOKE TEST -- Stenhede pipeline em {IMG_PATH.name}")
    print("=" * 78)

    digitizer = ECGDigitizer(use_mock=False)
    t0 = time.perf_counter()
    try:
        result = digitizer.run(str(IMG_PATH))
    except Exception as e:
        print(f"\nFALHA no digitizer.run: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2
    total_ms = (time.perf_counter() - t0) * 1000.0

    signals = result["signals"]
    sr = result["sampling_rate"]
    pxmm = result["px_per_mm"]
    qf = result["quality_flags"]

    print(f"\n[OK] digitizer.run finalizou em {total_ms:.0f} ms")
    print(f"\nResumo:")
    print(f"   sampling_rate     = {sr} Hz")
    print(f"   px_per_mm         = {pxmm:.3f}")
    print(f"   grid_shape        = {result['grid_shape']}")
    print(f"   segmenter         = {qf.get('segmenter')}")
    print(f"   lead_pixels       = {qf.get('lead_pixels')}")
    print(f"   leads_extracted   = {qf.get('leads_extracted')}")
    print(f"   undistortion      = {qf.get('undistortion')}")

    print(f"\nKeys em signals: {sorted(signals.keys())}")

    expected_main = {"I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"}
    missing = sorted(expected_main - set(signals.keys()))
    extras = sorted(set(signals.keys()) - expected_main)
    print(f"\nLeads faltando = {missing if missing else 'nenhum'}")
    print(f"Leads extras (rhythm/aliases) = {extras}")

    print(f"\n{'Lead':<10}{'shape':>10}{'nan_ratio':>12}"
          f"{'min_uV':>12}{'max_uV':>12}{'min_mm':>10}{'max_mm':>10}")
    print("-" * 76)
    LEAD_ORDER_PRINT = ["I", "II", "III", "aVR", "aVL", "aVF",
                        "V1", "V2", "V3", "V4", "V5", "V6", "II_rhythm"]
    for name in LEAD_ORDER_PRINT:
        if name not in signals:
            print(f"{name:<10}{'(ausente)':>10}")
            continue
        s = signals[name]
        n = s.shape[0]
        valid = ~np.isnan(s)
        nan_ratio = 1.0 - valid.sum() / max(n, 1)
        if valid.any():
            mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
        else:
            mn, mx = float("nan"), float("nan")
        print(
            f"{name:<10}{n:>10d}{nan_ratio:>12.2%}"
            f"{mn:>12.1f}{mx:>12.1f}"
            f"{mn / 100.0:>10.2f}{mx / 100.0:>10.2f}"
        )

    # Health-check resumido
    n_leads_with_data = sum(
        1 for n in expected_main
        if n in signals and np.any(~np.isnan(signals[n]))
        and (np.nanmax(signals[n]) - np.nanmin(signals[n])) > 50.0
    )
    print(f"\n[Health] Leads com sinal não-trivial (range > 0.5mm): "
          f"{n_leads_with_data}/{len(expected_main)}")

    if missing:
        print(f"[FALHA] Leads obrigatórios faltando: {missing}")
        return 3
    if n_leads_with_data == 0:
        print("[FALHA] Nenhum lead tem sinal não-trivial — pipeline degenerado")
        return 4
    print("\n[PASS] Smoke test concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
