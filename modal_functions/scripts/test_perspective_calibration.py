"""
test_perspective_calibration.py — Testa M2 (perspectiva) + M3 (calibração).

Testa com:
  1. Grid sintético perfeito (valida autocorrelação)
  2. Grid sintético rotacionado (valida correção de perspectiva)
  3. Fallback (sem grid)
"""

import os
import sys

_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pipeline_dir = os.path.join(_modal_dir, "pipeline")
sys.path.insert(0, _pipeline_dir)

import importlib
import numpy as np

_perspective = importlib.import_module("digitize.perspective")
correct_perspective = _perspective.correct_perspective

_calibration = importlib.import_module("digitize.calibration")
calibrate = _calibration.calibrate
CalibrationResult = _calibration.CalibrationResult


def _make_grid_mask(H: int, W: int, spacing_px: int) -> np.ndarray:
    """Cria mask de grid sintético com espaçamento conhecido."""
    mask = np.zeros((H, W), dtype=bool)
    # Linhas horizontais a cada spacing_px pixels
    for y in range(0, H, spacing_px):
        if y < H:
            mask[y, :] = True
            # Linhas do major grid (a cada 5 minor) são mais grossas
            if y % (spacing_px * 5) == 0 and y + 1 < H:
                mask[y + 1, :] = True
    # Linhas verticais a cada spacing_px pixels
    for x in range(0, W, spacing_px):
        if x < W:
            mask[:, x] = True
            if x % (spacing_px * 5) == 0 and x + 1 < W:
                mask[:, x + 1] = True
    return mask


def _rotate_mask_np(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotaciona mask usando OpenCV."""
    import cv2
    H, W = mask.shape
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(mask.astype(np.uint8), M, (W, H),
                              flags=cv2.INTER_NEAREST)
    return rotated.astype(bool)


def test_calibration_autocorrelation():
    """Testa calibração com grid sintético perfeito."""
    print("=" * 60)
    print("M3 — Calibracao: Autocorrelacao com grid sintetico")
    print("=" * 60)

    H, W = 512, 1024
    KNOWN_SPACING = 6  # 6 pixels por mm

    grid_mask = _make_grid_mask(H, W, KNOWN_SPACING)
    trace_mask = np.zeros((H, W), dtype=bool)
    text_mask = np.zeros((H, W), dtype=bool)
    bg_mask = ~(grid_mask | trace_mask | text_mask)

    masks = {
        "background": bg_mask,
        "grid": grid_mask,
        "trace": trace_mask,
        "text": text_mask,
    }
    image = np.random.rand(H, W, 3).astype(np.float32)

    result = calibrate(masks, image)

    print(f"  Espacamento real: {KNOWN_SPACING} px/mm")
    print(f"  Detectado px/mm_h: {result.px_per_mm_h:.2f}")
    print(f"  Detectado px/mm_v: {result.px_per_mm_v:.2f}")
    print(f"  Metodo: {result.method}")
    print(f"  Confianca: {result.confidence}")

    error_h = abs(result.px_per_mm_h - KNOWN_SPACING) / KNOWN_SPACING * 100
    error_v = abs(result.px_per_mm_v - KNOWN_SPACING) / KNOWN_SPACING * 100
    print(f"  Erro h: {error_h:.1f}%")
    print(f"  Erro v: {error_v:.1f}%")

    if result.method == "autocorrelation" and error_h < 20 and error_v < 20:
        print("\nAutocorrelacao: PASSOU\n")
        return True
    else:
        print(f"\nAutocorrelacao: FALHOU (metodo={result.method}, erro_h={error_h:.1f}%, erro_v={error_v:.1f}%)\n")
        return False


def test_perspective_correction():
    """Testa correção de perspectiva com grid rotacionado."""
    print("=" * 60)
    print("M2 — Correcao de perspectiva: grid rotacionado")
    print("=" * 60)

    H, W = 512, 1024
    SPACING = 6
    ROTATION = 3.5  # graus

    # Criar grid reto e depois rotacionar
    grid_straight = _make_grid_mask(H, W, SPACING)
    grid_rotated = _rotate_mask_np(grid_straight, ROTATION)

    image = np.random.rand(H, W, 3).astype(np.float32)
    masks = {
        "background": ~grid_rotated,
        "grid": grid_rotated,
        "trace": np.zeros((H, W), dtype=bool),
        "text": np.zeros((H, W), dtype=bool),
    }

    print(f"  Rotacao aplicada: {ROTATION} graus")
    print(f"  Grid pixels antes: {grid_rotated.sum()}")

    image_corr, masks_corr = correct_perspective(image, masks)

    # Verificar que as masks corrigidas têm shapes corretos
    for name, mask in masks_corr.items():
        assert mask.shape == (H, W), f"Shape errado para {name}: {mask.shape}"

    print(f"  Grid pixels depois: {masks_corr['grid'].sum()}")
    print(f"  Imagem corrigida shape: {image_corr.shape}")
    print("\nPerspectiva: PASSOU (sem crash, shapes corretos)\n")
    return True


def test_fallback():
    """Testa que fallback funciona quando não há grid."""
    print("=" * 60)
    print("M3 — Calibracao: Fallback (sem grid)")
    print("=" * 60)

    H, W = 512, 1024
    masks = {
        "background": np.ones((H, W), dtype=bool),
        "grid": np.zeros((H, W), dtype=bool),
        "trace": np.zeros((H, W), dtype=bool),
        "text": np.zeros((H, W), dtype=bool),
    }
    image = np.random.rand(H, W, 3).astype(np.float32)

    result = calibrate(masks, image)

    print(f"  Metodo: {result.method}")
    print(f"  px/mm_h: {result.px_per_mm_h:.2f}")
    print(f"  px/mm_v: {result.px_per_mm_v:.2f}")
    print(f"  Velocidade: {result.paper_speed} mm/s")
    print(f"  Ganho: {result.gain} mm/mV")
    print(f"  Confianca: {result.confidence}")

    assert result.method == "fallback", f"Esperado fallback, obtido {result.method}"
    assert result.confidence == "low"
    assert result.paper_speed == 25
    assert result.gain == 10
    assert result.px_per_mm_h > 0
    print("\nFallback: PASSOU\n")
    return True


def test_calibration_different_spacings():
    """Testa autocorrelação com diferentes espaçamentos."""
    print("=" * 60)
    print("M3 — Calibracao: Diferentes espacamentos de grid")
    print("=" * 60)

    H, W = 512, 1024
    all_ok = True

    for spacing in [3, 4, 5, 6, 8, 10]:
        grid_mask = _make_grid_mask(H, W, spacing)
        masks = {
            "background": ~grid_mask,
            "grid": grid_mask,
            "trace": np.zeros((H, W), dtype=bool),
            "text": np.zeros((H, W), dtype=bool),
        }
        image = np.random.rand(H, W, 3).astype(np.float32)
        result = calibrate(masks, image)

        error = abs(result.px_per_mm_h - spacing) / spacing * 100
        status = "OK" if result.method == "autocorrelation" and error < 20 else "FALHOU"
        if status == "FALHOU":
            all_ok = False
        print(f"  spacing={spacing}px: detectado={result.px_per_mm_h:.1f} erro={error:.0f}% metodo={result.method} [{status}]")

    print(f"\nDiferentes espacamentos: {'PASSOU' if all_ok else 'PARCIAL'}\n")
    return all_ok


def main():
    ok1 = test_calibration_autocorrelation()
    ok2 = test_perspective_correction()
    ok3 = test_fallback()
    ok4 = test_calibration_different_spacings()

    print("=" * 60)
    print("RESULTADO FINAL:")
    print(f"  Autocorrelacao grid:        {'PASSOU' if ok1 else 'FALHOU'}")
    print(f"  Correcao perspectiva:       {'PASSOU' if ok2 else 'FALHOU'}")
    print(f"  Fallback:                   {'PASSOU' if ok3 else 'FALHOU'}")
    print(f"  Diferentes espacamentos:    {'PASSOU' if ok4 else 'FALHOU'}")
    print("=" * 60)

    if not all([ok1, ok2, ok3, ok4]):
        sys.exit(1)


if __name__ == "__main__":
    main()
