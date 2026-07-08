from __future__ import annotations

import torch


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = config.get("optimizer", {})
    name = optim_cfg.get("name", "adamw").lower()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. The decoder/head should be trainable.")
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(optim_cfg.get("lr", 3e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")
