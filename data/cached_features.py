from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Dataset backed by Galileo feature cache .npz files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Missing feature cache directory: {self.root}")
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz cache files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
            if features.ndim == 4 and features.shape[0] == 1:
                features = features[0]
            if features.ndim != 3:
                raise ValueError(f"Expected cached features [D, H, W], got {features.shape} in {path}")

            features_by_layer = None
            if "features_by_layer" in data:
                features_by_layer = data["features_by_layer"].astype(np.float32, copy=False)
                if features_by_layer.ndim == 5 and features_by_layer.shape[1] == 1:
                    features_by_layer = features_by_layer[:, 0]
                if features_by_layer.ndim != 4:
                    raise ValueError(
                        f"Expected cached features_by_layer [L, D, H, W], "
                        f"got {features_by_layer.shape} in {path}"
                    )

            target = data["target"].astype(np.int64, copy=False)
            if target.ndim != 2:
                raise ValueError(f"Expected cached target [H, W], got {target.shape} in {path}")

            item = {
                "features": torch.from_numpy(features),
                "target": torch.from_numpy(target),
                "patch_id": int(data["patch_id"]),
                "fold": int(data["fold"]),
                "cache_path": str(path),
            }
            if features_by_layer is not None:
                item["features_by_layer"] = torch.from_numpy(features_by_layer)
            return item


def cached_feature_collate_fn(batch: list[dict]) -> dict:
    collated = {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "patch_id": [item["patch_id"] for item in batch],
        "fold": torch.tensor([item["fold"] for item in batch], dtype=torch.long),
        "cache_path": [item["cache_path"] for item in batch],
    }
    if all("features_by_layer" in item for item in batch):
        collated["features_by_layer"] = torch.stack(
            [item["features_by_layer"] for item in batch],
            dim=0,
        )
    return collated
