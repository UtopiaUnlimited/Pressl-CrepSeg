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
from utils import feature_cache_dir, load_config  # noqa: E402


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
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]

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
    ).to(device)
    encoder.eval()

    saved_samples = 0
    for batch in tqdm(loader, desc=f"cache {args.split}"):
        if args.max_samples is not None and saved_samples >= args.max_samples:
            break
        encoded = encoder(batch["samples"])
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
                "features": encoded.features[sample_index].detach().cpu().numpy(),
                "hidden_layers": np.asarray(encoder_cfg.get("hidden_layers") or [], dtype=np.int64),
                "encoder_name": encoder_cfg["name"],
                "encoder_checkpoint": encoder_cfg["checkpoint"],
                "patch_size": np.asarray(encoder_cfg["patch_size"]),
                "selected_timesteps": np.asarray(data_cfg["selected_timesteps"]),
                "temporal_aggregation": np.asarray(
                    str(data_cfg.get("temporal_aggregation", "uniform"))
                ),
                "tile_size": np.asarray(data_cfg.get("tile_size", data_cfg.get("image_size", 128))),
                "normalization": np.asarray(str(data_cfg.get("normalization", "none"))),
            }
            if "selected_indices" in sample:
                payload["selected_indices"] = sample["selected_indices"].numpy()
            if args.save_hidden_state:
                payload["hidden_state"] = encoded.hidden_state[sample_index].detach().cpu().numpy()
            if encoded.features_by_layer:
                payload["features_by_layer"] = np.stack(
                    [
                        feature[sample_index].detach().cpu().numpy()
                        for feature in encoded.features_by_layer
                    ],
                    axis=0,
                )
            np.savez_compressed(output_dir / f"{sample_id}.npz", **payload)
            saved_samples += 1


if __name__ == "__main__":
    main()
