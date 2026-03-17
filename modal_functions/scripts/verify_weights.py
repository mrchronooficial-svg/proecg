"""
verify_weights.py — Verifica que os pesos UNet carregam corretamente.
Roda: python scripts/verify_weights.py (de dentro de modal_functions/)
"""

import os
import sys

# Add digitize package to path directly (avoids pulling in all pipeline deps)
_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_modal_dir, "pipeline", "digitize"))

import torch

from constants import (
    LEAD_UNET_ENCODER_WIDTHS,
    LEAD_UNET_IN_CHANNELS,
    LEAD_UNET_NUM_CLASSES,
    LEAD_UNET_WEIGHTS_FILENAME,
    UNET_ENCODER_WIDTHS,
    UNET_IN_CHANNELS,
    UNET_NUM_CLASSES,
    UNET_WEIGHTS_FILENAME,
)
from unet import ResidualUNet, load_weights

WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "digitizer",
)


def verify_model(name, in_channels, num_classes, encoder_widths, weights_filename):
    print(f"\n{'='*60}")
    print(f"Verificando: {name}")
    print(f"  in_channels={in_channels}, num_classes={num_classes}")
    print(f"  encoder_widths={encoder_widths}")
    print(f"{'='*60}")

    weights_path = os.path.join(WEIGHTS_DIR, weights_filename)
    if not os.path.exists(weights_path):
        print(f"❌ Arquivo não encontrado: {weights_path}")
        return False

    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"  Arquivo: {weights_path} ({file_size_mb:.1f} MB)")

    try:
        model = ResidualUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_widths=encoder_widths,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Modelo instanciado: {total_params:,} parâmetros")

        model = load_weights(model, weights_path, device="cpu")
        print("  Pesos carregados com strict=True ✓")

        # Forward pass com input dummy
        test_h, test_w = 256, 256
        dummy = torch.randn(1, in_channels, test_h, test_w)
        with torch.no_grad():
            output = model(dummy)
            output_np = output.detach().cpu().numpy()

        expected_shape = (1, num_classes, test_h, test_w)
        if output_np.shape == expected_shape:
            print(f"  Forward pass OK: input {dummy.shape} → output {output_np.shape} ✓")
            print(f"  Output range: [{output_np.min():.4f}, {output_np.max():.4f}]")
            print(f"\n✅ {name}: SUCESSO")
            return True
        else:
            print(f"  ❌ Shape incorreto: esperado {expected_shape}, obtido {output_np.shape}")
            return False

    except Exception as e:
        print(f"  ❌ Erro: {e}")
        return False


def main():
    print("ProECG — Verificação de Pesos UNet")
    print("=" * 60)

    ok_main = verify_model(
        name="UNet Principal (segmentação)",
        in_channels=UNET_IN_CHANNELS,
        num_classes=UNET_NUM_CLASSES,
        encoder_widths=UNET_ENCODER_WIDTHS,
        weights_filename=UNET_WEIGHTS_FILENAME,
    )

    ok_lead = verify_model(
        name="Lead Name UNet",
        in_channels=LEAD_UNET_IN_CHANNELS,
        num_classes=LEAD_UNET_NUM_CLASSES,
        encoder_widths=LEAD_UNET_ENCODER_WIDTHS,
        weights_filename=LEAD_UNET_WEIGHTS_FILENAME,
    )

    print("\n" + "=" * 60)
    print("RESULTADO FINAL:")
    print(f"  UNet Principal: {'✅' if ok_main else '❌'}")
    print(f"  Lead Name UNet: {'✅' if ok_lead else '❌'}")
    print("=" * 60)

    if not (ok_main and ok_lead):
        sys.exit(1)


if __name__ == "__main__":
    main()
