from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        valid = torch.ones_like(target, dtype=torch.bool)
        safe_target = target
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            safe_target = target.masked_fill(~valid, 0)

        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=probs.dtype)
        valid = valid.unsqueeze(1)
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1 - dice.mean()
