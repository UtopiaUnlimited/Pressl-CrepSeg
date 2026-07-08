from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None = None,
) -> torch.Tensor:
    if ignore_index is None:
        return F.cross_entropy(logits, target)
    return F.cross_entropy(logits, target, ignore_index=ignore_index)
