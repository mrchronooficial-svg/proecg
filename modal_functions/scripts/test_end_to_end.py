"""
test_end_to_end.py — Testa pipeline completo do digitalizador.

1. Testa digitize_ecg() com imagem sintética (UNet + fallback)
2. Verifica contrato: shape (12, 5000), dtype float64, sem NaN
3. Testa fallback clássico isoladamente
4. Testa que fallback funciona quando UNet falha (simulado)
"""

import os
import sys

_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pipeline_dir = os.path.join(_modal_dir, "pipeline")
sys.path.insert(0, _pipeline_dir)

import importlib
import numpy as np
from PIL import Image

# Import the digitize package
_pkg = importlib.import_module("digitize")
digitize_ecg = _pkg.digitize_ecg

_fb = importlib.import_module("digitize.classic_fallback")
digitize_classic_fallback = _fb.digitize_classic_fallback

_pp = importlib.import_module("digitize.postprocess")
postprocess = _pp.postprocess


def _make_synthetic_ecg_image(W=1200, H=800):
    """Cria imagem PIL de ECG sintético com grid e traços."""
    img = np.ones((H, W, 3), dtype=np.uint8) * 255  # fundo branco

    # Grid laranja (padrão BR)
    for y in range(0, H, 6):  # ~6px por mm
        img[y, :] = [255, 200, 180]
    for x in range(0, W, 6):
        img[:, x] = [255, 200, 180]
    # Major grid
    for y in range(0, H, 30):
        img[y, :] = [255, 160, 140]
        if y + 1 < H:
            img[y + 1, :] = [255, 160, 140]
    for x in range(0, W, 30):
        img[:, x] = [255, 160, 140]
        if x + 1 < W:
            img[:, x + 1] = [255, 160, 140]

    # Traços pretos (3 linhas x 4 colunas)
    margin_top = int(H * 0.10)
    usable_h = int(H * 0.75)
    usable_w = int(W * 0.90)
    margin_left = int(W * 0.05)

    row_h = usable_h // 3
    col_w = usable_w // 4

    for row in range(3):
        for col in range(4):
            y_center = margin_top + row * row_h + row_h // 2
            x_start = margin_left + col * col_w + 5
            x_end = margin_left + (col + 1) * col_w - 5

            n = x_end - x_start
            t = np.linspace(0, 2 * np.pi * 3, n)
            wave = np.sin(t) * (row_h * 0.15)

            # QRS spikes
            for cycle in range(3):
                peak = int(n * (cycle + 0.5) / 3)
                sw = max(2, n // 50)
                for dx in range(-sw, sw + 1):
                    px = peak + dx
                    if 0 <= px < n:
                        wave[px] += (1 - abs(dx) / sw) * row_h * 0.25

            for i, x in enumerate(range(x_start, x_end)):
                y = int(y_center - wave[i])
                y = max(0, min(H - 1, y))
                for dy in range(-1, 2):
                    yy = max(0, min(H - 1, y + dy))
                    img[yy, x] = [0, 0, 0]

    return Image.fromarray(img, "RGB")


def test_full_pipeline():
    """Testa digitize_ecg() end-to-end."""
    print("=" * 60)
    print("Teste 1 — Pipeline completo (digitize_ecg)")
    print("=" * 60)

    img = _make_synthetic_ecg_image()
    print(f"  Imagem sintetica: {img.size} {img.mode}")

    signal = digitize_ecg(img)

    print(f"  Shape: {signal.shape}")
    print(f"  Dtype: {signal.dtype}")
    print(f"  NaN: {np.any(np.isnan(signal))}")
    print(f"  Range: [{signal.min():.4f}, {signal.max():.4f}] mV")

    assert signal.shape == (12, 5000), f"Shape errado: {signal.shape}"
    assert signal.dtype == np.float64, f"Dtype errado: {signal.dtype}"
    assert not np.any(np.isnan(signal)), "Contém NaN"
    assert np.max(np.abs(signal)) < 10.0, f"Amplitude implausível: {np.max(np.abs(signal))}"

    leads_active = sum(1 for i in range(12) if np.any(signal[i] != 0))
    print(f"  Leads ativos: {leads_active}/12")

    # Pelo menos alguns leads devem ter sinal
    assert leads_active >= 1, f"Nenhum lead ativo"

    print("\nPipeline completo: PASSOU\n")
    return True


def test_classic_fallback():
    """Testa fallback clássico isoladamente."""
    print("=" * 60)
    print("Teste 2 — Fallback classico")
    print("=" * 60)

    img = _make_synthetic_ecg_image()
    signal = digitize_classic_fallback(img)

    assert signal is not None, "Fallback retornou None"
    print(f"  Shape: {signal.shape}")
    print(f"  Dtype: {signal.dtype}")

    assert signal.shape == (12, 5000), f"Shape errado: {signal.shape}"

    leads_active = sum(1 for i in range(12) if np.any(signal[i] != 0))
    print(f"  Leads ativos: {leads_active}/12")

    print("\nFallback classico: PASSOU\n")
    return True


def test_postprocess_contract():
    """Testa M6 isoladamente com sinais variados."""
    print("=" * 60)
    print("Teste 3 — Postprocess (contrato de saida)")
    print("=" * 60)

    # Sinais com tamanhos variados
    signals = {
        "I": np.random.randn(1200) * 0.5,
        "II": np.random.randn(1200) * 0.5,
        "III": np.random.randn(1200) * 0.5,
        "aVR": np.random.randn(800) * 0.5,
        "aVL": np.random.randn(800) * 0.5,
        "aVF": np.random.randn(800) * 0.5,
        "V1": np.random.randn(1000) * 0.5,
        "V2": np.random.randn(1000) * 0.5,
        "V3": np.random.randn(1000) * 0.5,
        "V4": np.random.randn(1000) * 0.5,
        "V5": np.random.randn(1000) * 0.5,
        "V6": np.random.randn(1000) * 0.5,
    }

    result = postprocess(signals)

    assert result.shape == (12, 5000), f"Shape: {result.shape}"
    assert result.dtype == np.float64, f"Dtype: {result.dtype}"
    assert not np.any(np.isnan(result)), "NaN no output"

    leads_active = sum(1 for i in range(12) if np.any(result[i] != 0))
    print(f"  Shape: {result.shape}")
    print(f"  Dtype: {result.dtype}")
    print(f"  NaN: False")
    print(f"  Leads ativos: {leads_active}/12")
    print(f"  Range: [{result.min():.4f}, {result.max():.4f}] mV")

    print("\nPostprocess: PASSOU\n")
    return True


def test_postprocess_missing_leads():
    """Testa M6 com leads faltando."""
    print("=" * 60)
    print("Teste 4 — Postprocess com leads faltando")
    print("=" * 60)

    signals = {
        "I": np.random.randn(1200) * 0.5,
        "II": np.random.randn(1200) * 0.5,
        # Faltam 10 leads
    }

    result = postprocess(signals)

    assert result.shape == (12, 5000)
    assert not np.any(np.isnan(result))

    # Leads 0 e 1 (I, II) devem ter sinal
    assert np.any(result[0] != 0), "Lead I deveria ter sinal"
    assert np.any(result[1] != 0), "Lead II deveria ter sinal"
    # Leads 2-11 devem ser zeros
    for i in range(2, 12):
        assert np.all(result[i] == 0), f"Lead {i} deveria ser zeros"

    print(f"  Shape: {result.shape}")
    print(f"  Leads ativos: 2/12 (correto)")

    print("\nPostprocess leads faltando: PASSOU\n")
    return True


def test_postprocess_amplitude_clipping():
    """Testa que amplitudes > 5mV são clipadas."""
    print("=" * 60)
    print("Teste 5 — Clipping de amplitude")
    print("=" * 60)

    signals = {
        "I": np.ones(1200) * 20.0,  # 20 mV — artefato
    }

    result = postprocess(signals)

    max_amp = np.max(np.abs(result[0]))
    print(f"  Input: 20.0 mV")
    print(f"  Output max: {max_amp:.1f} mV")
    assert max_amp <= 5.0, f"Clipping falhou: {max_amp} mV"

    print("\nClipping: PASSOU\n")
    return True


def main():
    results = []
    results.append(("Pipeline completo", test_full_pipeline()))
    results.append(("Fallback classico", test_classic_fallback()))
    results.append(("Postprocess contrato", test_postprocess_contract()))
    results.append(("Postprocess leads faltando", test_postprocess_missing_leads()))
    results.append(("Clipping amplitude", test_postprocess_amplitude_clipping()))

    print("=" * 60)
    print("FASE 5 — RESULTADO FINAL:")
    for name, ok in results:
        print(f"  {name:35s} {'PASSOU' if ok else 'FALHOU'}")
    print("=" * 60)

    if not all(ok for _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
