from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CachedFeatureDataset, cached_feature_collate_fn  # noqa: E402
from losses import build_loss  # noqa: E402
from models import build_cached_feature_model  # noqa: E402
from train import Trainer, build_optimizer, build_scheduler  # noqa: E402
from utils import feature_cache_dir, load_config, merge_cli_overrides, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_dpt.yaml")
    parser.add_argument("--train-cache-dir", default=None)
    parser.add_argument("--val-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_loader(cache_dir: str, config: dict, shuffle: bool) -> DataLoader:
    decoder_name = str(config.get("model", {}).get("decoder", "")).lower()
    load_features_by_layer = decoder_name not in {"linear_probe", "linear", "lp"}
    dataset = CachedFeatureDataset(
        cache_dir,
        load_features_by_layer=load_features_by_layer,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=cached_feature_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = merge_cli_overrides(load_config(args.config), args)
    seed_everything(int(config.get("seed", 42)))

    train_cache_dir = args.train_cache_dir or feature_cache_dir(config, "train")
    val_cache_dir = args.val_cache_dir or feature_cache_dir(config, "val")
    train_loader = build_loader(train_cache_dir, config, shuffle=True)
    val_loader = build_loader(val_cache_dir, config, shuffle=False)

    first_batch = next(iter(train_loader))
    if "features_by_layer" in first_batch:
        in_channels = int(first_batch["features_by_layer"].shape[2])
        num_layers = int(first_batch["features_by_layer"].shape[1])
    else:
        in_channels = int(first_batch["features"].shape[1])
        num_layers = None
    model = build_cached_feature_model(config, in_channels=in_channels, num_layers=num_layers)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
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
        log_dir=config["train"].get("log_dir", "logs") + "_cached",
        checkpoint_dir=config["train"].get("checkpoint_dir", "checkpoints") + "_cached",
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
