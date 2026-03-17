"""
ProECG Digitizer — Arquitetura ResidualUNet
Verificada contra pesos reais do Ahus-AIM (Open-ECG-Digitizer).

Architecture derived from direct inspection of weight tensor shapes.
Key findings vs the paper description:
  - Decoder j uses features[N-2-j] as skip (skips the last encoder level)
  - Decoder residual skip operates on the concatenated input, not decoded output
  - Decoder output width = encoder_widths[N-2-j] (width of the skip feature)
  - final_conv has NO bias; input channels = encoder_widths[0]
"""

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """2x (Conv3x3 -> InstanceNorm -> LeakyReLU)."""
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        ),
        nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        ),
    )


class ResidualUNet(nn.Module):
    """
    Residual U-Net compatible with Ahus-AIM pretrained weights.

    Encoder (N levels, 0..N-1):
      residual = encoder_skips[i](x)          # Conv1x1, no bias
      x = encoders[i](x) + residual           # conv_block + skip
      features[i] = x
      x = encoder_downscaling[i](x)           # StridedConv 2x2, no bias

    Decoder (N-1 levels, 0..N-2):
      feat = features[N-2-j]                  # skip connection (skips features[N-1])
      x = upsample(x) -> cat(x, feat)
      combined = cat(x, feat)
      residual = decoder_skips[j](combined)   # Conv1x1 on concatenated input
      x = decoders[j](combined) + residual

    Final:
      x = upsample(x) to original size
      x = final_conv(x)                       # Conv3x3, NO bias
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        encoder_widths: Sequence[int],
    ):
        super().__init__()
        N = len(encoder_widths)
        widths = list(encoder_widths)

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.encoder_skips = nn.ModuleList()
        self.encoder_downscaling = nn.ModuleList()

        for i in range(N):
            ch_in = in_channels if i == 0 else widths[i - 1]
            ch_out = widths[i]
            self.encoders.append(_conv_block(ch_in, ch_out))
            self.encoder_skips.append(
                nn.Sequential(nn.Conv2d(ch_in, ch_out, 1, bias=False))
            )
            self.encoder_downscaling.append(
                nn.Conv2d(ch_out, ch_out, 2, stride=2, bias=False)
            )

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        self.decoder_skips = nn.ModuleList()

        prev_ch = widths[-1]  # output of last encoder downscaling
        for j in range(N - 1):
            feat_level = N - 2 - j  # skip connection from this encoder level
            feat_ch = widths[feat_level]
            ch_in = prev_ch + feat_ch  # concatenation of upsampled + skip
            ch_out = feat_ch  # output matches the skip feature width
            self.decoders.append(_conv_block(ch_in, ch_out))
            self.decoder_skips.append(
                nn.Sequential(nn.Conv2d(ch_in, ch_out, 1, bias=False))
            )
            prev_ch = ch_out

        # --- Final conv ---
        # Last decoder outputs encoder_widths[0] channels
        self.final_conv = nn.Conv2d(widths[0], num_classes, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[2:]
        features = []

        # Encoder
        for enc, skip, down in zip(
            self.encoders, self.encoder_skips, self.encoder_downscaling
        ):
            residual = skip(x)
            x = enc(x) + residual
            features.append(x)
            x = down(x)

        # Decoder
        N = len(features)
        for j, (dec, dskip) in enumerate(zip(self.decoders, self.decoder_skips)):
            feat_level = N - 2 - j
            feat = features[feat_level]
            x = F.interpolate(
                x, size=feat.shape[2:], mode="bilinear", align_corners=False
            )
            combined = torch.cat([x, feat], dim=1)
            residual = dskip(combined)
            x = dec(combined) + residual

        # Final upsample to original resolution
        x = F.interpolate(
            x, size=original_size, mode="bilinear", align_corners=False
        )
        x = self.final_conv(x)
        return x


def load_weights(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load pretrained weights, stripping _orig_mod. prefix from torch.compile()."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    logger.info("Loaded weights from %s (%d parameters)", path, len(cleaned))
    return model
