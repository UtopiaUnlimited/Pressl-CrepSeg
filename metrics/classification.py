from __future__ import annotations

import torch


def pixel_accuracy(confusion_matrix: torch.Tensor) -> float:
    """Return global pixel accuracy from a semantic-segmentation confusion matrix."""

    matrix = confusion_matrix.to(dtype=torch.float64)
    total = matrix.sum()
    if total <= 0:
        return 0.0
    return float((torch.diag(matrix).sum() / total).item())


def macro_f1(
    confusion_matrix: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[float, torch.Tensor]:
    """Return macro F1 and per-class F1 for classes present in target or prediction."""

    matrix = confusion_matrix.to(dtype=torch.float64)
    true_positive = torch.diag(matrix)
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    valid = denominator > 0
    per_class_f1 = torch.zeros_like(true_positive)
    per_class_f1[valid] = 2.0 * true_positive[valid] / (denominator[valid] + eps)
    if valid.any():
        return float(per_class_f1[valid].mean().item()), per_class_f1.to(torch.float32)
    return 0.0, per_class_f1.to(torch.float32)
