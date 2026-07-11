from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.heads import SegmentationHead


class PyramidPoolingModule(nn.Module):
    """Pool the deepest feature map at several spatial context scales."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_channels: int,
        pool_scales: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        scales = tuple(int(scale) for scale in pool_scales)
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError(f"pool_scales must contain positive integers, got {scales}")
        if branch_channels <= 0:
            raise ValueError(f"branch_channels must be positive, got {branch_channels}")

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(1, branch_channels),
                    nn.GELU(),
                )
                for scale in scales
            ]
        )
        merged_channels = in_channels + len(scales) * branch_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]
        pooled = [x]
        for branch in self.branches:
            context = branch(x)
            context = F.interpolate(
                context,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            pooled.append(context)
        return self.bottleneck(torch.cat(pooled, dim=1))


class UPerNetDecoder(nn.Module):
    """UPerNet-style PPM and FPN decoder over Galileo hidden-layer grids."""

    expects_feature_pyramid = True

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_layers: int,
        decoder_channels: int = 256,
        ppm_channels: int = 64,
        ppm_scales: Sequence[int] = (1, 2, 3, 6),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("UPerNetDecoder needs at least two hidden layers.")

        self.num_layers = int(num_layers)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, decoder_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.GELU(),
                )
                for _ in range(self.num_layers - 1)
            ]
        )
        self.ppm = PyramidPoolingModule(
            in_channels=in_channels,
            out_channels=decoder_channels,
            branch_channels=ppm_channels,
            pool_scales=ppm_scales,
            dropout=dropout,
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        decoder_channels,
                        decoder_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(decoder_channels),
                    nn.GELU(),
                )
                for _ in range(self.num_layers - 1)
            ]
        )
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                self.num_layers * decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
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

        laterals = [
            projection(feature)
            for projection, feature in zip(self.lateral_convs, features[:-1])
        ]
        laterals.append(self.ppm(features[-1]))

        for level in range(self.num_layers - 1, 0, -1):
            top_down = laterals[level]
            if top_down.shape[-2:] != laterals[level - 1].shape[-2:]:
                top_down = F.interpolate(
                    top_down,
                    size=laterals[level - 1].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            laterals[level - 1] = laterals[level - 1] + top_down

        fpn_outputs = [
            fpn_conv(lateral)
            for fpn_conv, lateral in zip(self.fpn_convs, laterals[:-1])
        ]
        fpn_outputs.append(laterals[-1])
        fusion_size = fpn_outputs[0].shape[-2:]
        fpn_outputs = [
            output
            if output.shape[-2:] == fusion_size
            else F.interpolate(
                output,
                size=fusion_size,
                mode="bilinear",
                align_corners=False,
            )
            for output in fpn_outputs
        ]

        x = self.fpn_bottleneck(torch.cat(fpn_outputs, dim=1))
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.smooth(x)
        return self.head(x)
