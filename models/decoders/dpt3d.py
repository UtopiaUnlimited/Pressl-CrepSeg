from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, preferred: int = 32) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _resize_spatial(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, channels, timesteps, height, width = x.shape
    if (height, width) == size:
        return x
    frames = x.permute(0, 2, 1, 3, 4).reshape(batch * timesteps, channels, height, width)
    frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
    return frames.reshape(batch, timesteps, channels, *size).permute(0, 2, 1, 3, 4)


class ChannelLayerNorm3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_probability)
        return x * mask / keep_probability


class FeedForward3D(nn.Module):
    def __init__(self, channels: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden_channels = channels * int(expansion)
        self.norm = ChannelLayerNorm3D(channels)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class TemporalAttention3D(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm3D(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, timesteps, height, width = x.shape
        normalized = self.norm(x)
        sequence = normalized.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width,
            timesteps,
            channels,
        )
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        return attended.reshape(batch, height, width, timesteps, channels).permute(0, 4, 3, 1, 2)


class SpatialWindowAttention3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        shifted: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = self.window_size // 2 if shifted else 0
        self.norm = ChannelLayerNorm3D(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, timesteps, height, width = x.shape
        frames = self.norm(x).permute(0, 2, 1, 3, 4).reshape(
            batch * timesteps,
            channels,
            height,
            width,
        )

        window = min(self.window_size, height, width)
        shift = min(self.shift_size, max(0, window - 1))
        pad_left = shift
        pad_top = shift
        pad_right = (window - (width + pad_left) % window) % window
        pad_bottom = (window - (height + pad_top) % window) % window
        frames = F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom))

        valid = torch.ones(
            (batch * timesteps, 1, height, width),
            dtype=torch.bool,
            device=x.device,
        )
        valid = F.pad(valid, (pad_left, pad_right, pad_top, pad_bottom), value=False)
        padded_height, padded_width = frames.shape[-2:]

        windows = frames.permute(0, 2, 3, 1).reshape(
            batch * timesteps,
            padded_height // window,
            window,
            padded_width // window,
            window,
            channels,
        ).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)
        valid_windows = valid.permute(0, 2, 3, 1).reshape(
            batch * timesteps,
            padded_height // window,
            window,
            padded_width // window,
            window,
        ).permute(0, 1, 3, 2, 4).reshape(-1, window * window)

        attended, _ = self.attention(
            windows,
            windows,
            windows,
            key_padding_mask=~valid_windows,
            need_weights=False,
        )
        frames = attended.reshape(
            batch * timesteps,
            padded_height // window,
            padded_width // window,
            window,
            window,
            channels,
        ).permute(0, 1, 3, 2, 4, 5).reshape(
            batch * timesteps,
            padded_height,
            padded_width,
            channels,
        )
        frames = frames[:, pad_top : pad_top + height, pad_left : pad_left + width]
        return frames.reshape(batch, timesteps, height, width, channels).permute(0, 4, 1, 2, 3)


class LocalSpaceTimeConv(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.net = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class DividedSpaceTimeBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        shifted: bool,
        mlp_expansion: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.temporal = TemporalAttention3D(channels, num_heads, dropout)
        self.spatial = SpatialWindowAttention3D(
            channels,
            num_heads,
            window_size,
            shifted,
            dropout,
        )
        self.local = LocalSpaceTimeConv(channels, dropout)
        self.feed_forward = FeedForward3D(channels, mlp_expansion, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.temporal(x))
        x = x + self.drop_path(self.spatial(x))
        x = x + self.drop_path(self.local(x))
        return x + self.drop_path(self.feed_forward(x))


class GlobalSpaceTimeBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_expansion: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm3D(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = FeedForward3D(channels, mlp_expansion, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, timesteps, height, width = x.shape
        sequence = self.norm(x).permute(0, 2, 3, 4, 1).reshape(
            batch,
            timesteps * height * width,
            channels,
        )
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        attended = attended.reshape(batch, timesteps, height, width, channels).permute(0, 4, 1, 2, 3)
        x = x + self.drop_path(attended)
        return x + self.drop_path(self.feed_forward(x))


class Reassemble3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.refine(_resize_spatial(self.projection(x), size))

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Project and refine while retaining the native spatial token grid."""

        return self.refine(self.projection(x))


class GatedCrossScaleFusion3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.gate = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, deep: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        deep = _resize_spatial(deep, lateral.shape[-2:])
        joined = torch.cat((deep, lateral), dim=1)
        gate = torch.sigmoid(self.gate(joined))
        return gate * lateral + (1.0 - gate) * deep + self.projection(joined)


class TemporalQueryPool(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, channels))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, timesteps, height, width = x.shape
        sequence = x.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width,
            timesteps,
            channels,
        )
        sequence = self.norm(sequence)
        query = sequence.mean(dim=1, keepdim=True) + self.query
        pooled, _ = self.attention(query, sequence, sequence, need_weights=False)
        pooled = self.output_norm(query + pooled).squeeze(1)
        return pooled.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class ThreeDAwareDPTDecoder(nn.Module):
    """Late-fusion DPT over Galileo's contextualized per-month hidden grids."""

    expects_temporal_feature_pyramid = True

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_layers: int = 4,
        decoder_channels: int = 256,
        num_heads: int = 8,
        spatial_window: int = 8,
        global_3d_blocks: int = 4,
        fusion_blocks_per_stage: int = 2,
        mlp_expansion: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        temporal_pool_heads: int = 8,
        num_months: int = 12,
        preserve_native_deep_skip: bool = True,
        phenology_prior: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_layers != 4:
            raise ValueError("ThreeDAwareDPTDecoder currently requires four Galileo hidden layers.")
        if decoder_channels % num_heads:
            raise ValueError("decoder_channels must be divisible by num_heads.")
        if decoder_channels % temporal_pool_heads:
            raise ValueError("decoder_channels must be divisible by temporal_pool_heads.")

        self.num_layers = int(num_layers)
        self.preserve_native_deep_skip = bool(preserve_native_deep_skip)
        self.phenology_prior = phenology_prior
        self.month_embedding = nn.Embedding(int(num_months), decoder_channels)
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, decoder_channels))
        nn.init.trunc_normal_(self.layer_embedding, std=0.02)
        self.reassemble = nn.ModuleList(
            [Reassemble3D(in_channels, decoder_channels) for _ in range(num_layers)]
        )

        self.global_blocks = nn.ModuleList(
            [
                GlobalSpaceTimeBlock(
                    decoder_channels,
                    num_heads,
                    mlp_expansion,
                    dropout,
                    drop_path,
                )
                for _ in range(global_3d_blocks)
            ]
        )
        self.fusions = nn.ModuleList(
            [GatedCrossScaleFusion3D(decoder_channels) for _ in range(num_layers - 1)]
        )
        self.stage_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        DividedSpaceTimeBlock(
                            decoder_channels,
                            num_heads,
                            spatial_window,
                            shifted=block_index % 2 == 1,
                            mlp_expansion=mlp_expansion,
                            dropout=dropout,
                            drop_path=drop_path,
                        )
                        for block_index in range(fusion_blocks_per_stage)
                    ]
                )
                for _ in range(num_layers)
            ]
        )
        self.temporal_pool = TemporalQueryPool(
            decoder_channels,
            temporal_pool_heads,
            dropout,
        )
        groups = _group_count(decoder_channels)
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        months: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} temporal feature maps, got {len(features)}.")
        if months.ndim != 2:
            raise ValueError(f"Expected months [B, T], got {tuple(months.shape)}")

        batch, timesteps = months.shape
        month_embedding = self.month_embedding(months).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        prior_context = None
        if self.phenology_prior is not None:
            prior_context = self.phenology_prior(months).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        scales = [
            (
                max(1, math.ceil(target_size[0] / (2**layer_index))),
                max(1, math.ceil(target_size[1] / (2**layer_index))),
            )
            for layer_index in range(self.num_layers)
        ]

        pyramid = []
        deep_native = None
        for layer_index, (feature, reassemble, size) in enumerate(
            zip(features, self.reassemble, scales)
        ):
            if feature.ndim != 5:
                raise ValueError(
                    f"Expected layer {layer_index} feature [B, T, D, H, W], "
                    f"got {tuple(feature.shape)}"
                )
            if feature.shape[:2] != (batch, timesteps):
                raise ValueError("Temporal feature and month shapes do not match.")
            feature = feature.permute(0, 2, 1, 3, 4)
            native_feature = feature
            feature = reassemble(feature, size)
            feature = feature + month_embedding
            feature = feature + self.layer_embedding[layer_index].view(1, -1, 1, 1, 1)
            if prior_context is not None:
                # Inject the shared class-month context before any temporal
                # attention or cross-layer fusion; layer identity stays in the
                # existing layer embedding above.
                feature = feature + prior_context
            pyramid.append(feature)
            if self.preserve_native_deep_skip and layer_index == self.num_layers - 1:
                deep_native = reassemble.forward_native(native_feature)
                deep_native = deep_native + self.layer_embedding[layer_index].view(
                    1, -1, 1, 1, 1
                )

        if self.preserve_native_deep_skip:
            if deep_native is None:
                raise RuntimeError("The native deepest temporal feature was not constructed.")
            pyramid[-2] = pyramid[-2] + _resize_spatial(
                deep_native,
                pyramid[-2].shape[-2:],
            )

        x = pyramid[-1]
        for block in self.global_blocks:
            x = block(x)
        x = self.stage_blocks[-1](x)

        for layer_index in range(self.num_layers - 2, -1, -1):
            x = self.fusions[layer_index](x, pyramid[layer_index])
            x = self.stage_blocks[layer_index](x)

        x = self.temporal_pool(x)
        x = self.head(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x
