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


class DPTMultiLayerDecoder(nn.Module):
    """DPT-style decoder that fuses several Galileo transformer hidden layers."""

    expects_feature_pyramid = True

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_layers: int,
        decoder_channels: int = 256,
        decoder_blocks: int = 3,
        fusion_blocks: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("DPTMultiLayerDecoder needs at least two hidden layers.")

        self.num_layers = int(num_layers)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, decoder_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.GELU(),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.fusion = nn.ModuleList(
            [
                nn.Sequential(
                    *[ResidualConvBlock(decoder_channels, dropout=dropout) for _ in range(fusion_blocks)]
                )
                for _ in range(self.num_layers - 1)
            ]
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

    def forward(
        self,
        features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} feature maps, got {len(features)}.")

        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        x = projected[-1]
        for feature, fusion_block in zip(reversed(projected[:-1]), reversed(self.fusion)):
            if x.shape[-2:] != feature.shape[-2:]:
                x = F.interpolate(x, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            x = fusion_block(x + feature)

        x = self.refine(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.smooth(x)
        return self.head(x)
