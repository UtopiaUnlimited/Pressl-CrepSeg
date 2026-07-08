from __future__ import annotations

import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int | None = None) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)

    def reset(self) -> None:
        self.matrix.zero_()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        if self.ignore_index is not None:
            keep = target != self.ignore_index
            pred = pred[keep]
            target = target[keep]
        keep = (target >= 0) & (target < self.num_classes)
        pred = pred[keep]
        target = target[keep]
        index = target * self.num_classes + pred
        counts = torch.bincount(index, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def to(self, device: torch.device | str) -> "ConfusionMatrix":
        self.matrix = self.matrix.to(device)
        return self
