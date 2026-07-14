from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_pastis_dataset, pastis_collate_fn  # noqa: E402
from models.encoders import GalileoHFEncoder  # noqa: E402
from utils import apply_cache_overrides, feature_cache_dir, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_dpt.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--cache-format",
        choices=["spatial_v1", "temporal_v2"],
        default=None,
    )
    parser.add_argument("--temporal-dtype", choices=["float16", "float32"], default=None)
    parser.add_argument(
        "--save-hidden-state",
        action="store_true",
        help=(
            "Also save full Galileo token hidden_state. This is large and not needed "
            "for cached decoder training."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cache_overrides(config, args.cache_format, args.temporal_dtype)
    cache_cfg = config["cache"]
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]

    cache_format = str(cache_cfg.get("format", "spatial_v1")).lower()
    if cache_format not in {"spatial_v1", "temporal_v2"}:
        raise ValueError(f"Unsupported cache format: {cache_format}")
    dtype_map = {"float16": np.float16, "float32": np.float32}
    temporal_dtype_name = str(cache_cfg.get("temporal_dtype", "float16")).lower()
    if temporal_dtype_name not in dtype_map:
        raise ValueError("Temporal cache dtype must be float16 or float32.")
    temporal_dtype = dtype_map[temporal_dtype_name]
    preserve_temporal = cache_format == "temporal_v2"
    if preserve_temporal and not encoder_cfg.get("hidden_layers"):
        raise ValueError("temporal_v2 cache requires encoder.hidden_layers.")
    if preserve_temporal and args.save_hidden_state:
        raise ValueError("temporal_v2 stores only temporal features; do not use --save-hidden-state.")

    output_dir = Path(args.output_dir or feature_cache_dir(config, args.split))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_pastis_dataset(data_cfg, args.split)
    batch_size = int(args.batch_size or data_cfg.get("batch_size", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pastis_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    encoder = GalileoHFEncoder(
        checkpoint=encoder_cfg["checkpoint"],
        patch_size=encoder_cfg.get("patch_size", 8),
        freeze=True,
        normalize=encoder_cfg.get("normalize", True),
        local_files_only=encoder_cfg.get("local_files_only", True),
        spatial_token_strategy=encoder_cfg.get("spatial_token_strategy", "auto"),
        hidden_layers=encoder_cfg.get("hidden_layers"),
        hidden_size=encoder_cfg.get("hidden_size"),
        preserve_temporal_features=preserve_temporal,
    ).to(device)
    encoder.eval()

    saved_samples = 0
    for batch in tqdm(loader, desc=f"cache {args.split}"):
        if args.max_samples is not None and saved_samples >= args.max_samples:
            break
        encoded = encoder(batch["samples"])
        cached_temporal_features_by_layer = None
        if preserve_temporal:
            if not encoded.temporal_features_by_layer:
                raise RuntimeError("Galileo did not return temporal hidden-layer features.")
            if encoded.temporal_features.numel() == 0:
                raise RuntimeError("Galileo did not return final temporal features.")
            temporal_pyramid = list(encoded.temporal_features_by_layer)
            temporal_pyramid[-1] = encoded.temporal_features
            cached_temporal_features_by_layer = np.stack(
                [
                    feature.detach().cpu().numpy()
                    for feature in temporal_pyramid
                ],
                axis=1,
            ).astype(temporal_dtype, copy=False)

        for sample_index, sample in enumerate(batch["samples"]):
            if args.max_samples is not None and saved_samples >= args.max_samples:
                break

            patch_id = sample["patch_id"]
            sample_id = sample["sample_id"]
            payload = {
                "patch_id": np.asarray(patch_id),
                "sample_id": np.asarray(sample_id),
                "tile_id": np.asarray(sample["tile_id"]),
                "tile_y": np.asarray(sample["tile_y"]),
                "tile_x": np.asarray(sample["tile_x"]),
                "fold": np.asarray(sample["fold"]),
                "dates": sample["dates"].numpy(),
                "months": sample["months"].numpy(),
                "aggregation_counts": sample["aggregation_counts"].numpy(),
                "target": batch["target"][sample_index].numpy(),
                "hidden_layers": np.asarray(encoder_cfg.get("hidden_layers") or [], dtype=np.int64),
                "encoder_name": encoder_cfg["name"],
                "encoder_checkpoint": encoder_cfg["checkpoint"],
                "cache_format": np.asarray(cache_format),
                "patch_size": np.asarray(encoder_cfg["patch_size"]),
                "selected_timesteps": np.asarray(data_cfg["selected_timesteps"]),
                "temporal_aggregation": np.asarray(
                    str(data_cfg.get("temporal_aggregation", "uniform"))
                ),
                "tile_size": np.asarray(data_cfg.get("tile_size", data_cfg.get("image_size", 128))),
                "normalization": np.asarray(str(data_cfg.get("normalization", "none"))),
            }
            if not preserve_temporal:
                payload["features"] = encoded.features[sample_index].detach().cpu().numpy()
            if "selected_indices" in sample:
                payload["selected_indices"] = sample["selected_indices"].numpy()
            if args.save_hidden_state:
                payload["hidden_state"] = encoded.hidden_state[sample_index].detach().cpu().numpy()
            if not preserve_temporal and encoded.features_by_layer:
                payload["features_by_layer"] = np.stack(
                    [
                        feature[sample_index].detach().cpu().numpy()
                        for feature in encoded.features_by_layer
                    ],
                    axis=0,
                )
            if cached_temporal_features_by_layer is not None:
                payload["temporal_dtype"] = np.asarray(temporal_dtype_name)
                feature_sources = [
                    f"block_{layer}" for layer in (encoder_cfg.get("hidden_layers") or [])
                ]
                feature_sources[-1] = "encoder_final"
                payload["temporal_feature_sources"] = np.asarray(feature_sources)
                payload["temporal_features_by_layer"] = (
                    cached_temporal_features_by_layer[sample_index]
                )
            np.savez_compressed(output_dir / f"{sample_id}.npz", **payload)
            saved_samples += 1


if __name__ == "__main__":
    main()
