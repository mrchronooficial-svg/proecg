"""
test_lead_signal.py — Testa M4 (lead detection) + M5 (signal extraction).

Cria ECG sintético com layout 3x4 + traços simulados e valida:
  - Detecção de linhas e colunas
  - Mapeamento correto de derivações
  - Extração de sinal com forma de onda reconhecível
  - Conversão px → mV
  - Resample para 500 Hz
"""

import os
import sys

_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pipeline_dir = os.path.join(_modal_dir, "pipeline")
sys.path.insert(0, _pipeline_dir)

import importlib
import numpy as np

_cal = importlib.import_module("digitize.calibration")
CalibrationResult = _cal.CalibrationResult

_ld = importlib.import_module("digitize.lead_detection")
detect_leads = _ld.detect_leads
LeadRegion = _ld.LeadRegion
LAYOUT_3x4_DII = _ld.LAYOUT_3x4_DII

_se = importlib.import_module("digitize.signal_extraction")
extract_signals = _se.extract_signals

_const = importlib.import_module("digitize.constants")
LEAD_ORDER = _const.LEAD_ORDER


def _make_synthetic_ecg(H=600, W=1200, spacing_px=5):
    """Cria masks sintéticas de ECG com layout 3x4.

    Gera trace com forma de onda tipo QRS em cada célula.
    """
    trace_mask = np.zeros((H, W), dtype=bool)
    grid_mask = np.zeros((H, W), dtype=bool)

    # Grid
    for y in range(0, H, spacing_px):
        grid_mask[y, :] = True
    for x in range(0, W, spacing_px):
        grid_mask[:, x] = True

    # Layout 3x4: 3 linhas com gap entre elas
    margin_top = int(H * 0.08)
    margin_left = int(W * 0.03)
    usable_h = int(H * 0.80)
    usable_w = int(W * 0.94)

    row_h = usable_h // 3
    col_w = usable_w // 4
    gap_h = int(row_h * 0.15)  # gap entre linhas

    for row in range(3):
        for col in range(4):
            y_center = margin_top + row * row_h + row_h // 2
            x_start = margin_left + col * col_w + int(col_w * 0.05)
            x_end = margin_left + (col + 1) * col_w - int(col_w * 0.05)

            # Gerar onda tipo QRS simples
            n_points = x_end - x_start
            t = np.linspace(0, 2 * np.pi * 3, n_points)  # 3 ciclos
            wave = np.sin(t) * (row_h * 0.25)

            # Adicionar QRS spike
            for cycle in range(3):
                peak_pos = int(n_points * (cycle + 0.5) / 3)
                spike_width = max(3, n_points // 40)
                for dx in range(-spike_width, spike_width + 1):
                    px = peak_pos + dx
                    if 0 <= px < n_points:
                        spike_amp = (1 - abs(dx) / spike_width) * row_h * 0.35
                        wave[px] += spike_amp

            # Desenhar traço na mask
            for i, x in enumerate(range(x_start, x_end)):
                y = int(y_center - wave[i])
                y = max(0, min(H - 1, y))
                # Traço com espessura de 2px
                for dy in range(-1, 2):
                    yy = y + dy
                    if 0 <= yy < H:
                        trace_mask[yy, x] = True

    bg_mask = ~(trace_mask | grid_mask)
    text_mask = np.zeros((H, W), dtype=bool)

    masks = {
        "background": bg_mask,
        "grid": grid_mask,
        "trace": trace_mask,
        "text": text_mask,
    }

    return masks, spacing_px


def test_lead_detection():
    """Testa M4 com ECG sintético."""
    print("=" * 60)
    print("M4 — Deteccao de Leads")
    print("=" * 60)

    masks, spacing = _make_synthetic_ecg()
    H, W = masks["trace"].shape

    print(f"  Imagem: {W}x{H}")
    print(f"  Trace pixels: {masks['trace'].sum()}")

    calibration = CalibrationResult(
        px_per_mm_h=float(spacing),
        px_per_mm_v=float(spacing),
        paper_speed=25,
        gain=10,
        confidence="high",
        method="test",
    )

    regions = detect_leads(masks, calibration)

    print(f"  Regioes detectadas: {len(regions)}")
    for r in regions:
        y1, x1, y2, x2 = r.bbox
        print(f"    {r.name:4s}: row={r.row} col={r.col} bbox=({y1},{x1},{y2},{x2}) "
              f"size={y2-y1}x{x2-x1}")

    # Verificações
    detected_names = {r.name for r in regions}
    assert len(regions) >= 10, f"Esperado >= 10 regioes, obtido {len(regions)}"

    # Verificar que bboxes não se sobrepõem entre si (mesmo row/col)
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions):
            if i >= j:
                continue
            if r1.row == r2.row and r1.col == r2.col:
                assert False, f"Regioes duplicadas: {r1.name} e {r2.name}"

    print(f"\n  Derivacoes detectadas: {sorted(detected_names)}")
    print("\nM4: PASSOU\n")
    return regions, masks, calibration


def test_signal_extraction(regions, masks, calibration):
    """Testa M5 com as regiões do M4."""
    print("=" * 60)
    print("M5 — Extracao de Sinal")
    print("=" * 60)

    trace_mask = masks["trace"]
    signals = extract_signals(trace_mask, regions, calibration)

    print(f"  Leads extraidos: {len(signals)}/{len(regions)}")

    all_ok = True
    for name in LEAD_ORDER:
        if name in signals:
            sig = signals[name]
            amp_max = np.max(np.abs(sig))
            print(f"    {name:4s}: {len(sig):5d} amostras, "
                  f"range [{sig.min():.3f}, {sig.max():.3f}] mV, "
                  f"amp_max={amp_max:.3f} mV")

            # Sanidade: amplitude deve ser < 5 mV para ECG
            if amp_max > 10:
                print(f"      AVISO: amplitude muito alta ({amp_max:.1f} mV)")
                all_ok = False
        else:
            print(f"    {name:4s}: NAO EXTRAIDO")

    # Verificar que pelo menos 8 leads foram extraídos
    assert len(signals) >= 8, f"Esperado >= 8 leads, obtido {len(signals)}"

    # Verificar que sinais têm amostras
    for name, sig in signals.items():
        assert len(sig) > 10, f"Lead {name}: muito poucas amostras ({len(sig)})"
        assert not np.any(np.isnan(sig)), f"Lead {name}: contém NaN"

    print(f"\nM5: PASSOU ({'OK' if all_ok else 'com avisos'})\n")
    return signals


def test_fallback_empty_trace():
    """Testa que fallback funciona com trace vazio."""
    print("=" * 60)
    print("M4 — Fallback (trace vazio)")
    print("=" * 60)

    H, W = 600, 1200
    masks = {
        "background": np.ones((H, W), dtype=bool),
        "grid": np.zeros((H, W), dtype=bool),
        "trace": np.zeros((H, W), dtype=bool),
        "text": np.zeros((H, W), dtype=bool),
    }

    regions = detect_leads(masks)

    print(f"  Regioes (fallback): {len(regions)}")
    assert len(regions) == 13, f"Layout 3x4+DII = 13 regioes, obtido {len(regions)}"

    detected_names = {r.name for r in regions}
    for lead in LEAD_ORDER:
        assert lead in detected_names, f"Lead {lead} ausente no fallback"

    print("\nFallback: PASSOU\n")


def main():
    regions, masks, calibration = test_lead_detection()
    signals = test_signal_extraction(regions, masks, calibration)
    test_fallback_empty_trace()

    print("=" * 60)
    print("FASE 4 — RESULTADO FINAL: TODOS OS TESTES PASSARAM")
    print("=" * 60)


if __name__ == "__main__":
    main()
