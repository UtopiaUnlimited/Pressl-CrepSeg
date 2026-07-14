from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Dataset backed by Galileo feature cache .npz files."""

    def __init__(
        self,
        root: str | Path,
        load_features_by_layer: bool = True,
        load_temporal_features_by_layer: bool = False,
    ) -> None:
        self.root = Path(root)
        self.load_features_by_layer = bool(load_features_by_layer)
        self.load_temporal_features_by_layer = bool(load_temporal_features_by_layer)
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
            needs_temporal = (
                self.load_temporal_features_by_layer
                or "features" not in data
                or (self.load_features_by_layer and "features_by_layer" not in data)
            )
            temporal_features = None
            if needs_temporal:
                if "temporal_features_by_layer" not in data:
                    if self.load_temporal_features_by_layer:
                        raise ValueError(
                            f"Cache {path} has no temporal_features_by_layer. "
                            "Use the legacy cache only with schemes one to four, or "
                            "generate a temporal_v2 cache for scheme five."
                        )
                    raise ValueError(
                        f"Cache {path} has neither the requested spatial features nor "
                        "temporal_features_by_layer."
                    )
                temporal_features = data["temporal_features_by_layer"]
                if temporal_features.ndim != 5:
                    raise ValueError(
                        "Expected cached temporal_features_by_layer [L, T, D, H, W], "
                        f"got {temporal_features.shape} in {path}"
                    )

            if "features" in data:
                features = data["features"].astype(np.float32, copy=False)
                if features.ndim == 4 and features.shape[0] == 1:
                    features = features[0]
                if features.ndim != 3:
                    raise ValueError(
                        f"Expected cached features [D, H, W], got {features.shape} in {path}"
                    )
            else:
                if temporal_features is None:
                    raise RuntimeError("Temporal cache was not loaded.")
                features = temporal_features[-1].mean(axis=0, dtype=np.float32)

            features_by_layer = None
            if self.load_features_by_layer and "features_by_layer" in data:
                features_by_layer = data["features_by_layer"].astype(np.float32, copy=False)
                if features_by_layer.ndim == 5 and features_by_layer.shape[1] == 1:
                    features_by_layer = features_by_layer[:, 0]
                if features_by_layer.ndim != 4:
                    raise ValueError(
                        f"Expected cached features_by_layer [L, D, H, W], "
                        f"got {features_by_layer.shape} in {path}"
                    )
            elif self.load_features_by_layer:
                if temporal_features is None:
                    raise RuntimeError("Temporal cache was not loaded.")
                features_by_layer = temporal_features.mean(axis=1, dtype=np.float32)

            temporal_features_by_layer = None
            if self.load_temporal_features_by_layer:
                if temporal_features is None:
                    raise RuntimeError("Temporal cache was not loaded.")
                temporal_features_by_layer = temporal_features.astype(
                    np.float32,
                    copy=False,
                )

            months = None
            if "months" in data:
                months = data["months"].astype(np.int64, copy=False)
                if months.ndim != 1:
                    raise ValueError(f"Expected cached months [T], got {months.shape} in {path}")
            if temporal_features is not None:
                if months is None:
                    raise ValueError(f"Temporal cache {path} is missing months.")
                if temporal_features.shape[1] != months.shape[0]:
                    raise ValueError(
                        "Temporal feature length does not match months: "
                        f"{temporal_features.shape[1]} vs {months.shape[0]} in {path}"
                    )

            target = data["target"].astype(np.int64, copy=False)
            if target.ndim != 2:
                raise ValueError(f"Expected cached target [H, W], got {target.shape} in {path}")

            item = {
                "features": torch.from_numpy(features),
                "target": torch.from_numpy(target),
                "patch_id": int(data["patch_id"]),
                "sample_id": str(data["sample_id"]) if "sample_id" in data else path.stem,
                "tile_id": int(data["tile_id"]) if "tile_id" in data else 0,
                "tile_y": int(data["tile_y"]) if "tile_y" in data else 0,
                "tile_x": int(data["tile_x"]) if "tile_x" in data else 0,
                "fold": int(data["fold"]),
                "cache_format": str(data["cache_format"])
                if "cache_format" in data
                else "spatial_v1",
                "cache_path": str(path),
            }
            if features_by_layer is not None:
                item["features_by_layer"] = torch.from_numpy(features_by_layer)
            if temporal_features_by_layer is not None:
                item["temporal_features_by_layer"] = torch.from_numpy(
                    temporal_features_by_layer
                )
            if months is not None:
                item["months"] = torch.from_numpy(months)
            return item


def cached_feature_collate_fn(batch: list[dict]) -> dict:
    collated = {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "patch_id": [item["patch_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "tile_id": torch.tensor([item["tile_id"] for item in batch], dtype=torch.long),
        "tile_y": torch.tensor([item["tile_y"] for item in batch], dtype=torch.long),
        "tile_x": torch.tensor([item["tile_x"] for item in batch], dtype=torch.long),
        "fold": torch.tensor([item["fold"] for item in batch], dtype=torch.long),
        "cache_format": [item["cache_format"] for item in batch],
        "cache_path": [item["cache_path"] for item in batch],
    }
    if all("features_by_layer" in item for item in batch):
        collated["features_by_layer"] = torch.stack(
            [item["features_by_layer"] for item in batch],
            dim=0,
        )
    if all("temporal_features_by_layer" in item for item in batch):
        collated["temporal_features_by_layer"] = torch.stack(
            [item["temporal_features_by_layer"] for item in batch],
            dim=0,
        )
    if all("months" in item for item in batch):
        collated["months"] = torch.stack([item["months"] for item in batch], dim=0)
    return collated
