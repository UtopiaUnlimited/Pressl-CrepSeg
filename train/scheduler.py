from __future__ import annotations

import math

import torch


def build_scheduler(
    config: dict,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_cfg = config.get("scheduler", {})
    name = scheduler_cfg.get("name", "none").lower()
    if name in {"none", "null"}:
        return None
    if name == "cosine":
        epochs = int(config.get("train", {}).get("epochs", 1))
        t_max = max(1, epochs * max(1, steps_per_epoch))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
    if name == "warmup_cosine":
        epochs = int(config.get("train", {}).get("epochs", 1))
        total_steps = max(1, epochs * max(1, steps_per_epoch))
        warmup_epochs = float(scheduler_cfg.get("warmup_epochs", epochs * 0.1))
        warmup_steps = max(1, int(warmup_epochs * max(1, steps_per_epoch)))
        max_lr = float(optimizer.param_groups[0]["lr"])
        min_lr = float(scheduler_cfg.get("min_lr", 1e-5))
        min_ratio = min_lr / max_lr

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    raise ValueError(f"Unsupported scheduler: {name}")
