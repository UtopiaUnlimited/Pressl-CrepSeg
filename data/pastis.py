from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


# Galileo's official evaluation calls these the "OURS" Sentinel-2 statistics.
# PASTIS supplies B2, B3, B4, B5, B6, B7, B8, B8A, B11, and B12 in this order.
GALILEO_S2_MEAN = np.asarray(
    [
        1395.3408730676722,
        1338.4026921784578,
        1343.09883810357,
        1543.8607982512297,
        2186.2022069512263,
        2525.0932853316694,
        2410.3377187373408,
        2750.2854646886753,
        2234.911100061487,
        1474.5311266077113,
    ],
    dtype=np.float32,
)
GALILEO_S2_STD = np.asarray(
    [
        917.7041440370853,
        913.2988423581528,
        1092.678723527555,
        1047.2206083460424,
        1048.0101611156767,
        1143.6903026819996,
        1098.979177731649,
        1204.472755085893,
        1145.9774063078878,
        980.2429840007796,
    ],
    dtype=np.float32,
)


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


def _date_parts(date_yyyymmdd: int) -> tuple[int, int]:
    date = int(date_yyyymmdd)
    year = date // 10000
    month = (date // 100) % 100
    if year <= 0 or not 1 <= month <= 12:
        raise ValueError(f"Invalid PASTIS date: {date_yyyymmdd}")
    return year, month


def _add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    month_index = year * 12 + (month - 1) + int(offset)
    return month_index // 12, month_index % 12 + 1


def aggregate_monthly_s2(
    s2: np.ndarray,
    dates: tuple[int, ...] | list[int],
    num_timesteps: int = 12,
    start_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate PASTIS acquisitions into a complete monthly crop-year sequence.

    PASTIS spans partial edge months from September 2018 through October 2019.
    The Galileo benchmark keeps the 12 complete interior months. A missing
    monthly composite is linearly interpolated from its nearest available
    neighbours so every sample has the official fixed T=12 shape.
    """

    if s2.ndim != 4:
        raise ValueError(f"Expected S2 [T, C, H, W], got {s2.shape}")
    if len(dates) != s2.shape[0]:
        raise ValueError(f"Got {len(dates)} dates for {s2.shape[0]} S2 acquisitions")
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")

    date_parts = [_date_parts(date) for date in dates]
    first_year, first_month = min(date_parts)
    start_year, start_month = _add_months(first_year, first_month, start_offset)
    target_months = [
        _add_months(start_year, start_month, offset) for offset in range(num_timesteps)
    ]

    composites: list[np.ndarray | None] = []
    source_counts = np.zeros(num_timesteps, dtype=np.int64)
    for output_index, target_month in enumerate(target_months):
        indices = [index for index, date_part in enumerate(date_parts) if date_part == target_month]
        source_counts[output_index] = len(indices)
        if indices:
            composites.append(s2[indices].mean(axis=0, dtype=np.float32))
        else:
            composites.append(None)

    available = [index for index, composite in enumerate(composites) if composite is not None]
    if not available:
        raise ValueError("No PASTIS acquisitions fall inside the requested monthly window")

    for index, composite in enumerate(composites):
        if composite is not None:
            continue
        left = max((value for value in available if value < index), default=None)
        right = min((value for value in available if value > index), default=None)
        if left is None:
            if right is None or composites[right] is None:
                raise RuntimeError("Could not interpolate missing monthly composite")
            composites[index] = composites[right]
        elif right is None:
            if composites[left] is None:
                raise RuntimeError("Could not interpolate missing monthly composite")
            composites[index] = composites[left]
        else:
            left_composite = composites[left]
            right_composite = composites[right]
            if left_composite is None or right_composite is None:
                raise RuntimeError("Could not interpolate missing monthly composite")
            weight = (index - left) / (right - left)
            composites[index] = left_composite * (1.0 - weight) + right_composite * weight

    monthly_s2 = np.stack(composites, axis=0).astype(np.float32, copy=False)
    month_indices = np.asarray([month - 1 for _, month in target_months], dtype=np.int64)
    representative_dates = np.asarray(
        [year * 10000 + month * 100 + 1 for year, month in target_months],
        dtype=np.int64,
    )
    return monthly_s2, month_indices, representative_dates, source_counts


def normalize_s2_for_galileo(
    s2: np.ndarray,
    std_multiplier: float = 2.0,
) -> np.ndarray:
    """Apply Galileo's official unclipped input scaling for linear probing."""

    if s2.ndim != 4 or s2.shape[1] != len(GALILEO_S2_MEAN):
        raise ValueError(f"Expected S2 [T, 10, H, W], got {s2.shape}")
    if std_multiplier <= 0:
        raise ValueError(f"std_multiplier must be positive, got {std_multiplier}")

    means = GALILEO_S2_MEAN[None, :, None, None]
    stds = GALILEO_S2_STD[None, :, None, None] * float(std_multiplier)
    minimum = means - stds
    maximum = means + stds
    return ((s2.astype(np.float32, copy=False) - minimum) / (maximum - minimum)).astype(
        np.float32,
        copy=False,
    )


def remap_void_label(
    target: np.ndarray,
    void_label: int | None = 19,
    ignore_index: int = -1,
) -> np.ndarray:
    target = target.astype(np.int64, copy=False)
    if void_label is None:
        return target
    target = target.copy()
    target[target == int(void_label)] = int(ignore_index)
    return target


def _sorted_dates(raw_dates: dict[str, int] | list[int]) -> tuple[int, ...]:
    if isinstance(raw_dates, dict):
        return tuple(int(raw_dates[key]) for key in sorted(raw_dates, key=lambda value: int(value)))
    return tuple(int(value) for value in raw_dates)


class PASTISDataset(Dataset):
    """Paper-aligned PASTIS Sentinel-2 semantic segmentation dataset."""

    def __init__(
        self,
        root: str | Path = "data/PASTIS",
        folds: Iterable[int] = (3,),
        selected_timesteps: int | None = 12,
        target_channel: int = 0,
        temporal_aggregation: str = "monthly",
        monthly_start_offset: int = 1,
        source_image_size: int = 128,
        tile_size: int | None = 64,
        void_label: int | None = 19,
        ignore_index: int = -1,
        normalization: str = "galileo_norm_no_clip",
        normalization_std_multiplier: float = 2.0,
    ) -> None:
        self.root = Path(root)
        self.folds = {int(fold) for fold in folds}
        self.selected_timesteps = selected_timesteps
        self.target_channel = int(target_channel)
        self.temporal_aggregation = str(temporal_aggregation).lower()
        self.monthly_start_offset = int(monthly_start_offset)
        self.source_image_size = int(source_image_size)
        self.tile_size = int(tile_size or source_image_size)
        self.void_label = void_label
        self.ignore_index = int(ignore_index)
        self.normalization = str(normalization).lower()
        self.normalization_std_multiplier = float(normalization_std_multiplier)

        if self.temporal_aggregation not in {"monthly", "uniform"}:
            raise ValueError(
                "temporal_aggregation must be 'monthly' or 'uniform', "
                f"got {self.temporal_aggregation}"
            )
        if self.temporal_aggregation == "monthly" and self.selected_timesteps is None:
            raise ValueError("Monthly aggregation requires selected_timesteps")
        if self.source_image_size % self.tile_size:
            raise ValueError(
                f"source_image_size={self.source_image_size} is not divisible by tile_size={self.tile_size}"
            )
        if self.normalization not in {"galileo_norm_no_clip", "none"}:
            raise ValueError(
                "normalization must be 'galileo_norm_no_clip' or 'none', "
                f"got {self.normalization}"
            )

        self.tiles_per_side = self.source_image_size // self.tile_size
        self.tiles_per_record = self.tiles_per_side**2

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
        return len(self.records) * self.tiles_per_record

    def __getitem__(self, index: int) -> dict:
        record_index, tile_id = divmod(index, self.tiles_per_record)
        record = self.records[record_index]
        s2_path = self.root / "DATA_S2" / f"S2_{record.patch_id}.npy"
        target_path = self.root / "ANNOTATIONS" / f"TARGET_{record.patch_id}.npy"

        if not s2_path.exists():
            raise FileNotFoundError(f"Missing S2 file: {s2_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target file: {target_path}")

        s2 = np.load(s2_path)
        if s2.ndim != 4 or s2.shape[1] != 10:
            raise ValueError(f"Expected S2 shape [T, 10, H, W], got {s2.shape} for {s2_path}")
        if tuple(s2.shape[-2:]) != (self.source_image_size, self.source_image_size):
            raise ValueError(
                f"Expected PASTIS spatial size {self.source_image_size}, got {s2.shape[-2:]}"
            )

        selected_indices: np.ndarray | None = None
        if self.temporal_aggregation == "monthly":
            s2, months, dates, aggregation_counts = aggregate_monthly_s2(
                s2,
                record.dates,
                num_timesteps=int(self.selected_timesteps),
                start_offset=self.monthly_start_offset,
            )
        else:
            selected_indices = uniform_sample_indices(s2.shape[0], self.selected_timesteps)
            s2 = s2[selected_indices].astype(np.float32, copy=False)
            dates = np.asarray([record.dates[int(i)] for i in selected_indices], dtype=np.int64)
            months = np.asarray([_date_parts(date)[1] - 1 for date in dates], dtype=np.int64)
            aggregation_counts = np.ones(len(selected_indices), dtype=np.int64)

        if self.normalization == "galileo_norm_no_clip":
            s2 = normalize_s2_for_galileo(s2, self.normalization_std_multiplier)
        else:
            s2 = s2.astype(np.float32, copy=False)

        target = np.load(target_path)
        if target.ndim == 3:
            target = target[self.target_channel]
        if target.ndim != 2:
            raise ValueError(f"Expected target shape [H, W] or [C, H, W], got {target.shape}")
        target = remap_void_label(target, self.void_label, self.ignore_index)

        tile_row, tile_col = divmod(tile_id, self.tiles_per_side)
        y0 = tile_row * self.tile_size
        x0 = tile_col * self.tile_size
        y1 = y0 + self.tile_size
        x1 = x0 + self.tile_size
        s2 = s2[:, :, y0:y1, x0:x1]
        target = target[y0:y1, x0:x1]
        sample_id = f"{record.patch_id}_y{y0}_x{x0}"

        sample = {
            "s2": torch.from_numpy(s2),
            "months": torch.from_numpy(months),
            "target": torch.from_numpy(target),
            "dates": torch.from_numpy(dates),
            "aggregation_counts": torch.from_numpy(aggregation_counts),
            "patch_id": record.patch_id,
            "sample_id": sample_id,
            "tile_id": tile_id,
            "tile_y": y0,
            "tile_x": x0,
            "fold": record.fold,
            "image_size": tuple(int(value) for value in target.shape[-2:]),
        }
        if selected_indices is not None:
            sample["selected_indices"] = torch.from_numpy(selected_indices)
        return sample


def build_pastis_dataset(data_config: dict, split: str) -> PASTISDataset:
    return PASTISDataset(
        root=data_config["root"],
        folds=data_config[f"{split}_folds"],
        selected_timesteps=data_config.get("selected_timesteps", 12),
        target_channel=data_config.get("target_channel", 0),
        temporal_aggregation=data_config.get("temporal_aggregation", "monthly"),
        monthly_start_offset=data_config.get("monthly_start_offset", 1),
        source_image_size=data_config.get("source_image_size", 128),
        tile_size=data_config.get("tile_size", 64),
        void_label=data_config.get("void_label", 19),
        ignore_index=data_config.get("ignore_index", -1),
        normalization=data_config.get("normalization", "galileo_norm_no_clip"),
        normalization_std_multiplier=data_config.get("normalization_std_multiplier", 2.0),
    )
