"""
test_segmentation.py — Testa M0 (preprocess) + M1 (segmentation).
Roda: python scripts/test_segmentation.py [caminho_foto_ecg.jpg]

Sem argumento: usa imagem sintética para validar shapes e funcionamento.
Com argumento: segmenta a foto real e salva as masks como PNGs.
"""

import os
import sys

_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add pipeline's parent so 'pipeline.digitize' resolves as a proper package
# but avoid triggering pipeline/__init__.py by importing the subpackage directly
_pipeline_dir = os.path.join(_modal_dir, "pipeline")
sys.path.insert(0, _pipeline_dir)

import numpy as np
from PIL import Image

# Import the digitize subpackage modules via importlib to skip pipeline/__init__.py
import importlib
_digitize_pkg = importlib.import_module("digitize.preprocess")
preprocess = _digitize_pkg.preprocess

_seg_mod = importlib.import_module("digitize.segmentation")
segment = _seg_mod.segment


def test_preprocess():
    """Testa M0 com diferentes inputs."""
    print("=" * 60)
    print("M0 — Pré-processamento")
    print("=" * 60)

    # Teste 1: imagem RGB normal
    img = Image.new("RGB", (4000, 3000), color=(255, 200, 200))
    result = preprocess(img)
    assert result.dtype == np.float32, f"dtype errado: {result.dtype}"
    assert result.min() >= 0.0 and result.max() <= 1.0, "range fora de [0, 1]"
    assert max(result.shape[:2]) <= 2048, f"resize falhou: {result.shape}"
    assert result.shape[2] == 3, "deve ter 3 canais"
    print(f"  RGB 4000x3000 -> {result.shape} OK")

    # Teste 2: imagem grayscale
    img_gray = Image.new("L", (800, 600), color=128)
    result2 = preprocess(img_gray)
    assert result2.shape[2] == 3, "grayscale deve ser convertido para RGB"
    print(f"  Grayscale 800x600 -> {result2.shape} OK")

    # Teste 3: imagem pequena (sem resize)
    img_small = Image.new("RGB", (512, 512), color=(0, 0, 0))
    result3 = preprocess(img_small)
    assert result3.shape == (512, 512, 3), f"pequena nao deve redimensionar: {result3.shape}"
    print(f"  Small 512x512 -> {result3.shape} OK")

    # Teste 4: RGBA
    img_rgba = Image.new("RGBA", (1000, 800), color=(255, 0, 0, 128))
    result4 = preprocess(img_rgba)
    assert result4.shape[2] == 3, "RGBA deve ser convertido para RGB"
    print(f"  RGBA 1000x800 -> {result4.shape} OK")

    print("\nM0: TODOS OS TESTES PASSARAM\n")


def test_segmentation_synthetic():
    """Testa M1 com imagem sintética."""
    print("=" * 60)
    print("M1 — Segmentacao (imagem sintetica)")
    print("=" * 60)

    # Criar imagem sintetica simples (256x256)
    img = np.random.rand(256, 256, 3).astype(np.float32)

    masks = segment(img)
    if masks is None:
        print("  AVISO: Modelo nao carregou (pesos nao encontrados)")
        print("  Isso e esperado se os pesos nao estiverem no caminho correto")
        return False

    # Verificar que retornou 4 masks
    expected_keys = {"background", "grid", "trace", "text"}
    assert set(masks.keys()) == expected_keys, f"keys erradas: {masks.keys()}"

    for name, mask in masks.items():
        assert mask.shape == (256, 256), f"mask '{name}' shape errado: {mask.shape}"
        assert mask.dtype == bool, f"mask '{name}' dtype errado: {mask.dtype}"
        print(f"  Mask '{name}': shape {mask.shape}, {mask.sum()} pixels ({mask.sum()/mask.size*100:.1f}%)")

    # Verificar que cada pixel pertence a exatamente uma classe
    total = masks["background"].astype(int) + masks["grid"].astype(int) + \
            masks["trace"].astype(int) + masks["text"].astype(int)
    assert np.all(total == 1), "Cada pixel deve pertencer a exatamente uma classe"
    print("  Particao exclusiva: OK (cada pixel em exatamente 1 classe)")

    print("\nM1 sintetico: TODOS OS TESTES PASSARAM\n")
    return True


def test_segmentation_real(image_path: str):
    """Testa M1 com foto real de ECG e salva masks como PNG."""
    print("=" * 60)
    print(f"M1 — Segmentacao (foto real: {os.path.basename(image_path)})")
    print("=" * 60)

    img_pil = Image.open(image_path)
    print(f"  Original: {img_pil.size} {img_pil.mode}")

    # M0: preprocess
    img_np = preprocess(img_pil)
    print(f"  Preprocessed: {img_np.shape}")

    # M1: segment
    masks = segment(img_np)
    if masks is None:
        print("  ERRO: Modelo nao carregou")
        return

    # Salvar masks como PNGs
    output_dir = os.path.join(_modal_dir, "scripts", "test_output")
    os.makedirs(output_dir, exist_ok=True)

    for name, mask in masks.items():
        pct = mask.sum() / mask.size * 100
        print(f"  Mask '{name}': {mask.sum()} px ({pct:.1f}%)")
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        path = os.path.join(output_dir, f"mask_{name}.png")
        mask_img.save(path)
        print(f"    Salvo: {path}")

    # Salvar overlay colorido
    H, W = img_np.shape[:2]
    overlay = (img_np * 255).astype(np.uint8).copy()
    # Grid = azul, Trace = vermelho, Text = verde
    colors = {"grid": [0, 0, 200], "trace": [200, 0, 0], "text": [0, 200, 0]}
    for name, color in colors.items():
        mask = masks[name]
        for c in range(3):
            overlay[:, :, c] = np.where(mask, color[c], overlay[:, :, c])

    overlay_img = Image.fromarray(overlay)
    overlay_path = os.path.join(output_dir, "overlay_segmentation.png")
    overlay_img.save(overlay_path)
    print(f"\n  Overlay salvo: {overlay_path}")
    print("\nInspecione visualmente as masks para validar a segmentacao.")


def main():
    # Sempre rodar testes de preprocess
    test_preprocess()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Arquivo nao encontrado: {image_path}")
            sys.exit(1)
        test_segmentation_real(image_path)
    else:
        ok = test_segmentation_synthetic()
        if ok:
            print("Para testar com foto real:")
            print("  python scripts/test_segmentation.py caminho/para/ecg.jpg")


if __name__ == "__main__":
    main()
