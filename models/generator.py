"""
U-Net based generator for Pix2Pix IR -> RGB colorization.
Takes a 1-channel IR image, outputs a 3-channel RGB image.
"""

import torch
import torch.nn as nn


def down_block(in_ch, out_ch, normalize=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_ch, out_ch, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64):
        super().__init__()

        # Encoder (downsampling)
        self.down1 = down_block(in_channels, features, normalize=False)   # 256 -> 128
        self.down2 = down_block(features, features * 2)                  # 128 -> 64
        self.down3 = down_block(features * 2, features * 4)              # 64 -> 32
        self.down4 = down_block(features * 4, features * 8)              # 32 -> 16
        self.down5 = down_block(features * 8, features * 8)              # 16 -> 8
        self.down6 = down_block(features * 8, features * 8)              # 8 -> 4
        self.down7 = down_block(features * 8, features * 8)              # 4 -> 2
        self.bottleneck = down_block(features * 8, features * 8, normalize=False)         # 2 -> 1

        # Decoder (upsampling) with skip connections
        self.up1 = up_block(features * 8, features * 8, dropout=True)        # 1 -> 2
        self.up2 = up_block(features * 16, features * 8, dropout=True)       # 2 -> 4
        self.up3 = up_block(features * 16, features * 8, dropout=True)       # 4 -> 8
        self.up4 = up_block(features * 16, features * 8)                     # 8 -> 16
        self.up5 = up_block(features * 16, features * 4)                     # 16 -> 32
        self.up6 = up_block(features * 8, features * 2)                      # 32 -> 64
        self.up7 = up_block(features * 4, features)                          # 64 -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        return self.final(torch.cat([u7, d1], dim=1))


if __name__ == "__main__":
    model = UNetGenerator()
    x = torch.randn(2, 1, 256, 256)  # batch of 2 IR images
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # should be [2, 3, 256, 256]