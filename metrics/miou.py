from __future__ import annotations

import torch


def mean_iou(confusion_matrix: torch.Tensor, eps: float = 1e-6) -> tuple[float, torch.Tensor]:
    matrix = confusion_matrix.to(dtype=torch.float32)
    intersection = torch.diag(matrix)
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - intersection
    valid = union > 0
    iou = torch.zeros_like(intersection)
    iou[valid] = intersection[valid] / (union[valid] + eps)
    if valid.any():
        return float(iou[valid].mean().item()), iou
    return 0.0, iou
