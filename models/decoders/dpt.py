from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.heads import SegmentationHead


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DPTSingleLayerDecoder(nn.Module):
    """Single-layer DPT-style decoder over a Galileo spatial token grid."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        decoder_channels: int = 256,
        decoder_blocks: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            *[ResidualConvBlock(decoder_channels, dropout=dropout) for _ in range(decoder_blocks)]
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.head = SegmentationHead(decoder_channels, num_classes)

    def forward(self, features: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.projection(features)
        x = self.refine(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.smooth(x)
        return self.head(x)
