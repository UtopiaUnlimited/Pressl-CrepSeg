from __future__ import annotations

import torch
from torch import nn


TEMPORAL_READOUT_DECODER_BASES = {
    "temporal_readout_single_layer_dpt": "single_layer_dpt",
    "temporal_single_layer_dpt": "single_layer_dpt",
    "temporal_readout_multi_layer_dpt": "multi_layer_dpt",
    "temporal_multi_layer_dpt": "multi_layer_dpt",
    "temporal_readout_upernet": "upernet",
    "temporal_upernet": "upernet",
    "temporal_readout_galileo_dpt": "galileo_dpt",
    "temporal_galileo_dpt": "galileo_dpt",
}


class MonthAwareTemporalReadout(nn.Module):
    """Learn a spatially varying readout over contextualized monthly features."""

    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_months: int = 12,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels < 1 or num_layers < 1 or num_months < 1:
            raise ValueError("channels, num_layers, and num_months must be positive.")

        hidden_channels = int(hidden_channels or max(32, channels // 4))
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive.")

        self.channels = int(channels)
        self.num_layers = int(num_layers)
        self.num_months = int(num_months)
        self.feature_norm = nn.LayerNorm(self.channels)
        self.month_embedding = nn.Embedding(self.num_months, self.channels)
        self.layer_embedding = nn.Embedding(self.num_layers, self.channels)
        self.scorer = nn.Sequential(
            nn.Linear(self.channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        nn.init.trunc_normal_(self.month_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.layer_embedding.weight, std=0.02)
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        months: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(
                "Expected temporal features [B, T, D, H, W], "
                f"got {tuple(features.shape)}"
            )
        if months.ndim != 2:
            raise ValueError(f"Expected months [B, T], got {tuple(months.shape)}")
        if features.shape[:2] != months.shape:
            raise ValueError(
                "Temporal feature and month shapes do not match: "
                f"{tuple(features.shape[:2])} vs {tuple(months.shape)}"
            )
        if features.shape[2] != self.channels:
            raise ValueError(
                f"Expected {self.channels} feature channels, got {features.shape[2]}."
            )
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError(
                f"layer_index must be in [0, {self.num_layers - 1}], got {layer_index}."
            )
        if months.numel() and (
            int(months.min()) < 0 or int(months.max()) >= self.num_months
        ):
            raise ValueError(
                f"Month indices must be in [0, {self.num_months - 1}]."
            )

        sequence = features.permute(0, 1, 3, 4, 2)
        month_context = self.month_embedding(months).unsqueeze(2).unsqueeze(2)
        layer_context = self.layer_embedding.weight[layer_index].view(1, 1, 1, 1, -1)
        score_input = self.feature_norm(sequence) + month_context + layer_context
        scores = self.scorer(score_input).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        fused = (weights * sequence).sum(dim=1)
        return fused.permute(0, 3, 1, 2).contiguous()


class TemporalReadoutDecoder(nn.Module):
    """Apply one learned temporal readout before an existing 2D decoder."""

    expects_temporal_feature_pyramid = True

    def __init__(
        self,
        spatial_decoder: nn.Module,
        in_channels: int,
        num_layers: int,
        num_months: int = 12,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial_decoder = spatial_decoder
        self.num_layers = int(num_layers)
        self.uses_feature_pyramid = bool(
            getattr(self.spatial_decoder, "expects_feature_pyramid", False)
        )
        self.temporal_layer_indices = (
            tuple(range(self.num_layers))
            if self.uses_feature_pyramid
            else (self.num_layers - 1,)
        )
        self.readout = MonthAwareTemporalReadout(
            channels=in_channels,
            num_layers=num_layers,
            num_months=num_months,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def forward(
        self,
        features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        months: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) == self.num_layers:
            selected_features = tuple(
                features[layer_index] for layer_index in self.temporal_layer_indices
            )
        elif len(features) == len(self.temporal_layer_indices):
            selected_features = tuple(features)
        else:
            raise ValueError(
                f"Expected {len(self.temporal_layer_indices)} selected feature maps "
                f"or all {self.num_layers} feature maps, "
                f"got {len(features)}."
            )

        if self.uses_feature_pyramid:
            spatial_features = tuple(
                self.readout(feature, months, layer_index)
                for layer_index, feature in zip(
                    self.temporal_layer_indices,
                    selected_features,
                )
            )
            return self.spatial_decoder(spatial_features, target_size=target_size)

        final_feature = self.readout(
            selected_features[0],
            months,
            self.temporal_layer_indices[0],
        )
        return self.spatial_decoder(final_feature, target_size=target_size)
