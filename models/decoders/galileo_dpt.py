from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, preferred: int = 32) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvUnit(nn.Module):
    """Pre-activation residual unit used by the DPT fusion path."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GalileoReassemble(nn.Module):
    """Project one Galileo hidden grid and place it at a DPT pyramid scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale_factor = float(scale_factor)
        groups = _group_count(out_channels)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        if self.scale_factor == 4.0:
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=4,
                bias=False,
            )
        elif self.scale_factor == 2.0:
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
            )
        elif self.scale_factor == 1.0:
            self.resample = nn.Identity()
        elif self.scale_factor == 0.5:
            self.resample = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        else:
            raise ValueError(
                "GalileoReassemble supports scale factors 4, 2, 1, and 0.5, "
                f"got {self.scale_factor}."
            )
        self.refine = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, feature: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected Galileo feature [B, D, H, W], got {tuple(feature.shape)}")
        x = self.resample(self.projection(feature))
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.refine(x)

    def forward_native(self, feature: torch.Tensor) -> torch.Tensor:
        """Project and refine without discarding the native Galileo grid."""

        if feature.ndim != 4:
            raise ValueError(f"Expected Galileo feature [B, D, H, W], got {tuple(feature.shape)}")
        return self.refine(self.projection(feature))


class FeatureFusionBlock(nn.Module):
    """Fuse a deep path with one lateral DPT feature at the lateral scale."""

    def __init__(self, channels: int, blocks: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("FeatureFusionBlock needs at least one residual block.")
        self.lateral_refine = ResidualConvUnit(channels, dropout=dropout)
        self.output_refine = nn.Sequential(
            *[ResidualConvUnit(channels, dropout=dropout) for _ in range(blocks)]
        )

    def forward(self, deep: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        if deep.shape[-2:] != lateral.shape[-2:]:
            deep = F.interpolate(
                deep,
                size=lateral.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.output_refine(deep + self.lateral_refine(lateral))


class GalileoDPTDecoder(nn.Module):
    """DPT decoder adapted to four early-fused Galileo hidden-layer grids."""

    expects_feature_pyramid = True
    scale_factors = (4.0, 2.0, 1.0, 0.5)

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_layers: int = 4,
        decoder_channels: int = 256,
        fusion_blocks: int = 2,
        head_channels: int = 128,
        dropout: float = 0.1,
        preserve_native_deep_skip: bool = True,
    ) -> None:
        super().__init__()
        if num_layers != 4:
            raise ValueError("GalileoDPTDecoder requires exactly four shallow-to-deep inputs.")
        if decoder_channels < 1 or head_channels < 1:
            raise ValueError("decoder_channels and head_channels must be positive.")
        if fusion_blocks < 1:
            raise ValueError("fusion_blocks must be at least one.")

        self.num_layers = int(num_layers)
        self.preserve_native_deep_skip = bool(preserve_native_deep_skip)
        self.reassemble = nn.ModuleList(
            [
                GalileoReassemble(
                    in_channels=in_channels,
                    out_channels=decoder_channels,
                    scale_factor=scale_factor,
                    dropout=dropout,
                )
                for scale_factor in self.scale_factors
            ]
        )
        self.deep_refine = nn.Sequential(
            *[
                ResidualConvUnit(decoder_channels, dropout=dropout)
                for _ in range(fusion_blocks)
            ]
        )
        self.fusions = nn.ModuleList(
            [
                FeatureFusionBlock(
                    decoder_channels,
                    blocks=fusion_blocks,
                    dropout=dropout,
                )
                for _ in range(num_layers - 1)
            ]
        )
        head_groups = _group_count(head_channels)
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(head_groups, head_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(head_channels, num_classes, kernel_size=1),
        )

    @staticmethod
    def pyramid_sizes(target_size: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        height, width = target_size
        if height < 8 or width < 8:
            raise ValueError(f"DPT target size must be at least 8x8, got {target_size}.")
        return tuple(
            (max(1, height // (2**level)), max(1, width // (2**level)))
            for level in range(4)
        )

    def forward(
        self,
        features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != self.num_layers:
            raise ValueError(f"Expected four Galileo hidden layers, got {len(features)}.")
        batch_shape = features[0].shape
        if any(feature.shape != batch_shape for feature in features):
            shapes = [tuple(feature.shape) for feature in features]
            raise ValueError(f"All Galileo DPT inputs must share one grid shape, got {shapes}.")

        sizes = self.pyramid_sizes(target_size)
        pyramid = [
            adapter(feature, size)
            for adapter, feature, size in zip(self.reassemble, features, sizes)
        ]

        if self.preserve_native_deep_skip:
            deep_native = self.reassemble[-1].forward_native(features[-1])
            if deep_native.shape[-2:] != pyramid[-2].shape[-2:]:
                deep_native = F.interpolate(
                    deep_native,
                    size=pyramid[-2].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            pyramid[-2] = pyramid[-2] + deep_native

        x = self.deep_refine(pyramid[-1])
        for lateral, fusion in zip(reversed(pyramid[:-1]), reversed(self.fusions)):
            x = fusion(x, lateral)
        logits = self.head(x)
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits
