from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CachedFeatureDataset, cached_feature_collate_fn  # noqa: E402
from losses import build_loss  # noqa: E402
from metrics import ConfusionMatrix, mean_iou  # noqa: E402
from models import build_cached_feature_model  # noqa: E402
from utils import feature_cache_dir, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_dpt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_loader(cache_dir: str, config: dict, batch_size: int | None) -> DataLoader:
    dataset = CachedFeatureDataset(cache_dir)
    return DataLoader(
        dataset,
        batch_size=int(batch_size or config["data"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=cached_feature_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cache_dir = args.cache_dir or feature_cache_dir(config, args.split)
    loader = build_loader(cache_dir, config, args.batch_size)

    first_batch = next(iter(loader))
    if "features_by_layer" in first_batch:
        in_channels = int(first_batch["features_by_layer"].shape[2])
        num_layers = int(first_batch["features_by_layer"].shape[1])
    else:
        in_channels = int(first_batch["features"].shape[1])
        num_layers = None
    model = build_cached_feature_model(config, in_channels=in_channels, num_layers=num_layers)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    criterion = build_loss(config).to(device)
    confusion = ConfusionMatrix(
        num_classes=int(config["data"]["num_classes"]),
        ignore_index=config.get("loss", {}).get("ignore_index"),
    )

    total_loss = 0.0
    batch_count = 0
    for batch in tqdm(loader, desc=f"cached {args.split}"):
        target = batch["target"].to(device, non_blocking=True)
        batch["target"] = target
        logits = model(batch)
        total_loss += float(criterion(logits, target).item())
        batch_count += 1
        confusion.update(logits, target)

    miou, per_class_iou = mean_iou(confusion.matrix)
    print(f"{args.split}_loss={total_loss / max(1, batch_count):.5f}")
    print(f"{args.split}_miou={miou:.5f}")
    print("per_class_iou=" + ",".join(f"{value:.5f}" for value in per_class_iou.tolist()))


if __name__ == "__main__":
    main()
