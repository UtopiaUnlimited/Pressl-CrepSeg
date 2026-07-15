from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    CachedFeatureDataset,
    PASTISDataset,
    build_pastis_dataset,
    cached_feature_collate_fn,
)
from losses import build_loss  # noqa: E402
from metrics import ConfusionMatrix, macro_f1, mean_iou, pixel_accuracy  # noqa: E402
from models import (  # noqa: E402
    build_cached_feature_model,
    cached_decoder_uses_feature_pyramid,
    cached_decoder_uses_temporal_features,
)
from scripts.visualize_predictions import make_rgb_composite, render_triptych  # noqa: E402
from utils import (  # noqa: E402
    apply_cache_overrides,
    apply_phenology_overlay,
    feature_cache_dir,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_dpt.yaml")
    parser.add_argument("--phenology-config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-format", choices=["spatial_v1", "temporal_v2"], default=None)
    parser.add_argument("--temporal-dtype", choices=["float16", "float32"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output"),
        help="Directory for every RGB / ground-truth / prediction triptych.",
    )
    parser.add_argument("--panel-size", type=int, default=384)
    return parser.parse_args()


def sample_index_by_id(dataset: PASTISDataset) -> dict[str, int]:
    index_by_id: dict[str, int] = {}
    for record_index, record in enumerate(dataset.records):
        for tile_id in range(dataset.tiles_per_record):
            tile_row, tile_col = divmod(tile_id, dataset.tiles_per_side)
            sample_id = (
                f"{record.patch_id}_y{tile_row * dataset.tile_size}"
                f"_x{tile_col * dataset.tile_size}"
            )
            index_by_id[sample_id] = record_index * dataset.tiles_per_record + tile_id
    return index_by_id


def build_loader(cache_dir: str, config: dict, batch_size: int | None) -> DataLoader:
    load_temporal = cached_decoder_uses_temporal_features(config)
    dataset = CachedFeatureDataset(
        cache_dir,
        load_features_by_layer=cached_decoder_uses_feature_pyramid(config),
        load_temporal_features_by_layer=load_temporal,
    )
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
    config = apply_phenology_overlay(load_config(args.config), args.phenology_config)
    config = apply_cache_overrides(config, args.cache_format, args.temporal_dtype)
    cache_dir = args.cache_dir or feature_cache_dir(config, args.split)
    loader = build_loader(cache_dir, config, args.batch_size)

    first_batch = next(iter(loader))
    if "temporal_features_by_layer" in first_batch:
        in_channels = int(first_batch["temporal_features_by_layer"].shape[3])
        num_layers = int(first_batch["temporal_features_by_layer"].shape[1])
    elif "features_by_layer" in first_batch:
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
    raw_dataset = build_pastis_dataset(config["data"], args.split)
    raw_index_by_id = sample_index_by_id(raw_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for batch in tqdm(loader, desc=f"cached {args.split}"):
        target = batch["target"].to(device, non_blocking=True)
        batch["target"] = target
        logits = model(batch)
        total_loss += float(criterion(logits, target).item())
        batch_count += 1
        confusion.update(logits, target)

        predictions = logits.argmax(dim=1).detach().cpu().numpy()
        targets = target.detach().cpu().numpy()
        for sample_index, sample_id in enumerate(batch["sample_id"]):
            if sample_id not in raw_index_by_id:
                raise KeyError(
                    f"Cached sample is not present in raw {args.split} split: {sample_id}"
                )
            raw_sample = raw_dataset[raw_index_by_id[sample_id]]
            comparison = render_triptych(
                rgb=make_rgb_composite(raw_sample, config),
                target=targets[sample_index],
                prediction=predictions[sample_index],
                sample_id=sample_id,
                fold=int(raw_sample["fold"]),
                panel_size=args.panel_size,
            )
            comparison.save(output_dir / f"{args.split}_{sample_id}.png")
            saved_count += 1

    miou, per_class_iou = mean_iou(confusion.matrix)
    accuracy = pixel_accuracy(confusion.matrix)
    f1, per_class_f1 = macro_f1(confusion.matrix)
    print(f"{args.split}_loss={total_loss / max(1, batch_count):.5f}")
    print(f"{args.split}_miou={miou:.5f}")
    print(f"{args.split}_acc={accuracy:.5f}")
    print(f"{args.split}_f1={f1:.5f}")
    print(
        "per_class_iou="
        + ",".join(f"{value:.5f}" for value in per_class_iou.tolist())
    )
    print(
        "per_class_f1="
        + ",".join(f"{value:.5f}" for value in per_class_f1.tolist())
    )
    print(f"saved_predictions={saved_count} output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
