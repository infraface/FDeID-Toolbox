"""
3D AutoEncoder for video de-identification.
DeID-rPPG: Rethinking the tradeoff between utility and privacy in video-based remote PPG.
"""

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    3D AutoEncoder for video-based face de-identification.

    Input: (B, 3, T, H, W) - Video tensor with T frames of HxW RGB images
    Output: (B, 3, T, H, W) - De-identified video tensor in range [-1, 1]

    Architecture:
    - Encoder: 3D Conv blocks with avg pooling
    - Skip connections for preserving spatial details
    - Decoder: 3D Conv blocks with upsampling
    - Tanh activation for [-1, 1] output range
    """

    def __init__(self, frames=128):
        super(AutoEncoder, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 32, [3, 3, 3], stride=1, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(inplace=True),
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(inplace=True),
        )

        self.ConvUpBlock3 = nn.Sequential(
            nn.Conv3d(64, 32, [3, 3, 3], stride=1, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(inplace=True),
        )

        self.ConvUpBlock1 = nn.Sequential(
            nn.Conv3d(32, 3, [3, 3, 3], stride=1, padding=1, padding_mode="replicate"),
        )

        self.Maxpool = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.upnnSpa = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, T, H, W) in range [-1, 1]

        Returns:
            De-identified tensor (B, 3, T, H, W) in range [-1, 1]
        """
        x = self.ConvBlock1(x)  # (B, 32, T, H, W)
        x1 = x.clone()  # Skip connection 1
        x = self.Maxpool(x)  # (B, 32, T, H/2, W/2)

        x = self.ConvBlock3(x)  # (B, 64, T, H/2, W/2)
        x2 = x.clone()  # Skip connection 2
        x = self.Maxpool(x)  # (B, 64, T, H/4, W/4)

        x = self.ConvBlock5(x)  # (B, 64, T, H/4, W/4)

        x = self.upnnSpa(x) + x2  # (B, 64, T, H/2, W/2)
        x = self.ConvUpBlock3(x)  # (B, 32, T, H/2, W/2)

        x = self.upnnSpa(x) + x1  # (B, 32, T, H, W)
        x = self.ConvUpBlock1(x)  # (B, 3, T, H, W)
        x = self.tanh(x)

        return x
