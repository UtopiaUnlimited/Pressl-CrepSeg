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

from data import PASTISDataset, pastis_collate_fn  # noqa: E402
from models.encoders import GalileoHFEncoder  # noqa: E402
from utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_dpt.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]

    output_dir = Path(
        args.output_dir
        or f"data/cache/{encoder_cfg['name']}/t{data_cfg['selected_timesteps']}_patch{encoder_cfg['patch_size']}_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PASTISDataset(
        root=data_cfg["root"],
        folds=data_cfg[f"{args.split}_folds"],
        selected_timesteps=data_cfg["selected_timesteps"],
        target_channel=data_cfg.get("target_channel", 0),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pastis_collate_fn)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    encoder = GalileoHFEncoder(
        checkpoint=encoder_cfg["checkpoint"],
        patch_size=encoder_cfg.get("patch_size", 8),
        freeze=True,
        normalize=encoder_cfg.get("normalize", True),
        local_files_only=encoder_cfg.get("local_files_only", True),
        spatial_token_strategy=encoder_cfg.get("spatial_token_strategy", "auto"),
        hidden_size=encoder_cfg.get("hidden_size"),
    ).to(device)
    encoder.eval()

    for batch_index, batch in enumerate(tqdm(loader, desc=f"cache {args.split}"), start=1):
        if args.max_samples is not None and batch_index > args.max_samples:
            break
        encoded = encoder(batch["samples"])
        sample = batch["samples"][0]
        patch_id = sample["patch_id"]
        target = batch["target"][0].numpy()
        np.savez_compressed(
            output_dir / f"{patch_id}.npz",
            patch_id=np.asarray(patch_id),
            fold=np.asarray(sample["fold"]),
            dates=sample["dates"].numpy(),
            selected_indices=sample["selected_indices"].numpy(),
            months=sample["months"].numpy(),
            target=target,
            features=encoded.features.detach().cpu().numpy(),
            hidden_state=encoded.hidden_state.detach().cpu().numpy(),
            encoder_name=encoder_cfg["name"],
            encoder_checkpoint=encoder_cfg["checkpoint"],
            patch_size=np.asarray(encoder_cfg["patch_size"]),
            selected_timesteps=np.asarray(data_cfg["selected_timesteps"]),
            normalization=np.asarray(str(encoder_cfg.get("normalize", True))),
        )


if __name__ == "__main__":
    main()
