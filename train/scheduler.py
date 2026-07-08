from __future__ import annotations

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
    raise ValueError(f"Unsupported scheduler: {name}")
