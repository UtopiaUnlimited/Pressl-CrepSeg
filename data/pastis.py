from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PastisRecord:
    patch_id: int
    fold: int
    dates: tuple[int, ...]


def uniform_sample_indices(length: int, max_timesteps: int | None) -> np.ndarray:
    if length <= 0:
        raise ValueError("A PASTIS sample must contain at least one timestep.")
    if max_timesteps is None or length <= max_timesteps:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, max_timesteps, dtype=np.int64)


def _date_to_month_index(date_yyyymmdd: int) -> int:
    month = (int(date_yyyymmdd) // 100) % 100
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid PASTIS date: {date_yyyymmdd}")
    return month - 1


def _sorted_dates(raw_dates: dict[str, int] | list[int]) -> tuple[int, ...]:
    if isinstance(raw_dates, dict):
        return tuple(int(raw_dates[key]) for key in sorted(raw_dates, key=lambda value: int(value)))
    return tuple(int(value) for value in raw_dates)


class PASTISDataset(Dataset):
    """PASTIS Sentinel-2 semantic segmentation dataset.

    S2 files are loaded as [T, 10, H, W] and target channel 0 is used as the
    semantic mask with labels 0..19. Spatial resolution is never resized.
    """

    def __init__(
        self,
        root: str | Path = "data/PASTIS",
        folds: Iterable[int] = (3,),
        selected_timesteps: int | None = 24,
        target_channel: int = 0,
    ) -> None:
        self.root = Path(root)
        self.folds = {int(fold) for fold in folds}
        self.selected_timesteps = selected_timesteps
        self.target_channel = target_channel

        metadata_path = self.root / "metadata.geojson"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing PASTIS metadata: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        records: list[PastisRecord] = []
        for feature in metadata.get("features", []):
            props = feature["properties"]
            fold = int(props["Fold"])
            if fold not in self.folds:
                continue
            records.append(
                PastisRecord(
                    patch_id=int(props["ID_PATCH"]),
                    fold=fold,
                    dates=_sorted_dates(props["dates-S2"]),
                )
            )

        self.records = sorted(records, key=lambda record: record.patch_id)
        if not self.records:
            raise ValueError(f"No PASTIS samples found for folds {sorted(self.folds)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        s2_path = self.root / "DATA_S2" / f"S2_{record.patch_id}.npy"
        target_path = self.root / "ANNOTATIONS" / f"TARGET_{record.patch_id}.npy"

        if not s2_path.exists():
            raise FileNotFoundError(f"Missing S2 file: {s2_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target file: {target_path}")

        s2 = np.load(s2_path)
        if s2.ndim != 4 or s2.shape[1] != 10:
            raise ValueError(f"Expected S2 shape [T, 10, H, W], got {s2.shape} for {s2_path}")

        selected_indices = uniform_sample_indices(s2.shape[0], self.selected_timesteps)
        s2 = s2[selected_indices].astype(np.float32, copy=False)
        selected_dates = tuple(record.dates[int(i)] for i in selected_indices)
        months = np.asarray([_date_to_month_index(date) for date in selected_dates], dtype=np.int64)

        target = np.load(target_path)
        if target.ndim == 3:
            target = target[self.target_channel]
        if target.ndim != 2:
            raise ValueError(f"Expected target shape [H, W] or [C, H, W], got {target.shape}")

        return {
            "s2": torch.from_numpy(s2),
            "months": torch.from_numpy(months),
            "target": torch.from_numpy(target.astype(np.int64, copy=False)),
            "dates": torch.tensor(selected_dates, dtype=torch.int64),
            "selected_indices": torch.from_numpy(selected_indices),
            "patch_id": record.patch_id,
            "fold": record.fold,
            "image_size": tuple(int(value) for value in target.shape[-2:]),
        }
