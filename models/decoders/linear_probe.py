from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GalileoLinearProbeDecoder(nn.Module):
    """Official Galileo segmentation probe: one linear patch-to-pixels map."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        output_patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.output_patch_size = int(output_patch_size)
        if self.output_patch_size < 1:
            raise ValueError("output_patch_size must be at least 1.")
        self.probe = nn.Linear(
            int(in_channels),
            self.num_classes * self.output_patch_size**2,
        )

    def forward(self, features: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected features [B, D, H, W], got {tuple(features.shape)}")
        batch, _, grid_h, grid_w = features.shape
        patch = self.output_patch_size
        logits = self.probe(features.permute(0, 2, 3, 1))
        logits = logits.reshape(
            batch,
            grid_h,
            grid_w,
            self.num_classes,
            patch,
            patch,
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            batch,
            self.num_classes,
            grid_h * patch,
            grid_w * patch,
        )
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=True,
            )
        return logits
