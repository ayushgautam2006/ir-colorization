"""
PatchGAN discriminator for Pix2Pix. Classifies whether NxN patches
of the IR+RGB pair are real or generated.
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4, features=64):  # 1 (IR) + 3 (RGB) = 4
        super().__init__()

        def block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, features, normalize=False),   # 256 -> 128
            block(features, features * 2),                    # 128 -> 64
            block(features * 2, features * 4),                # 64 -> 32
            block(features * 4, features * 8, stride=1),       # 32 -> 31 (patch-level)
            nn.Conv2d(features * 8, 1, 4, stride=1, padding=1)  # -> 30x30 patch predictions
        )

    def forward(self, ir, rgb):
        x = torch.cat([ir, rgb], dim=1)  # concat along channel dim
        return self.model(x)


if __name__ == "__main__":
    disc = PatchGANDiscriminator()
    ir = torch.randn(2, 1, 256, 256)
    rgb = torch.randn(2, 3, 256, 256)
    out = disc(ir, rgb)
    print(f"IR shape: {ir.shape}, RGB shape: {rgb.shape}")
    print(f"Discriminator output shape: {out.shape}")  # should be [2, 1, 30, 30]