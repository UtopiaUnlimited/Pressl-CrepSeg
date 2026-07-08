from __future__ import annotations

import torch
from torch import nn

from .ce import cross_entropy_loss
from .dice import DiceLoss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = ignore_index
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = logits.new_tensor(0.0)
        if self.ce_weight:
            loss = loss + self.ce_weight * cross_entropy_loss(logits, target, self.ignore_index)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss


def build_loss(config: dict) -> nn.Module:
    data_cfg = config["data"]
    loss_cfg = config.get("loss", {})
    return CombinedLoss(
        num_classes=int(data_cfg["num_classes"]),
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        ignore_index=loss_cfg.get("ignore_index"),
    )
