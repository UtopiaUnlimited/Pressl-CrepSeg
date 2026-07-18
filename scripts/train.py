from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_pastis_dataset, pastis_collate_fn  # noqa: E402
from losses import build_loss  # noqa: E402
from models import build_model  # noqa: E402
from train import Trainer, build_optimizer, build_scheduler  # noqa: E402
from utils import (  # noqa: E402
    apply_phenology_overlay,
    apply_prior_injection_overlay,
    load_config,
    merge_cli_overrides,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_dpt.yaml")
    parser.add_argument("--phenology-config", default=None)
    parser.add_argument("--prior-config", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_loader(config: dict, split: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    dataset = build_pastis_dataset(data_cfg, split)
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=pastis_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = apply_phenology_overlay(load_config(args.config), args.phenology_config)
    config = apply_prior_injection_overlay(config, args.prior_config)
    config = merge_cli_overrides(config, args)
    seed_everything(int(config.get("seed", 42)))

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    train_loader = build_loader(config, "train", shuffle=True)
    val_loader = build_loader(config, "val", shuffle=False)

    model = build_model(config)
    criterion = build_loss(config).to(device)
    optimizer = build_optimizer(config, model)
    accumulation_steps = int(config["train"].get("gradient_accumulation_steps", 1))
    scheduler = build_scheduler(
        config,
        optimizer,
        steps_per_epoch=math.ceil(len(train_loader) / accumulation_steps),
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=int(config["data"]["num_classes"]),
        amp=bool(config["train"].get("amp", False)),
        amp_dtype=str(config["train"].get("amp_dtype", "float16")),
        log_dir=config["train"].get("log_dir", "logs"),
        checkpoint_dir=config["train"].get("checkpoint_dir", "checkpoints"),
        ignore_index=config.get("loss", {}).get("ignore_index"),
        save_best=bool(config["train"].get("save_best", True)),
        gradient_accumulation_steps=accumulation_steps,
        max_grad_norm=config["train"].get("max_grad_norm"),
        save_trainable_only=bool(config["train"].get("save_trainable_only", False)),
        save_last=bool(config["train"].get("save_last", False)),
        early_stopping=config["train"].get("early_stopping"),
    )
    trainer.fit(
        epochs=int(config["train"]["epochs"]),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )


if __name__ == "__main__":
    main()
