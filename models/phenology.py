from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


MONTH_COLUMNS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)


def load_phenology_prior(
    path: str | Path,
    num_classes: int,
) -> torch.Tensor:
    """Load a class-by-calendar-month soft prior from a validated CSV."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing phenology prior CSV: {source}")

    table = np.zeros((int(num_classes), len(MONTH_COLUMNS)), dtype=np.float32)
    seen: set[int] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Phenology prior CSV has no header: {source}")
        required = {"class_id", *MONTH_COLUMNS}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Phenology prior CSV is missing columns {missing}: {source}"
            )

        for row in reader:
            try:
                class_id = int(row["class_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid class_id in phenology prior: {row}") from exc
            if not 0 <= class_id < int(num_classes):
                raise ValueError(
                    f"Phenology prior class_id={class_id} outside [0, {num_classes})"
                )
            if class_id in seen:
                raise ValueError(f"Duplicate phenology prior class_id={class_id}")
            seen.add(class_id)

            values = []
            for month in MONTH_COLUMNS:
                raw_value = row.get(month, "")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid value for class_id={class_id}, month={month}: {raw_value!r}"
                    ) from exc
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Phenology prior values must be finite and in [0,1], "
                        f"got class_id={class_id}, month={month}, value={value}"
                    )
                values.append(value)
            table[class_id] = values

    missing_classes = sorted(set(range(int(num_classes))) - seen)
    if missing_classes:
        raise ValueError(
            f"Phenology prior is missing class ids {missing_classes}: {source}"
        )
    return torch.from_numpy(table)


class PhenologyPriorAdapter(nn.Module):
    """Project class-month priors into a temporal decoder feature context.

    The same class-by-month table is shared by every encoder layer. The layer
    identity remains represented by the decoder's existing layer embedding.
    """

    def __init__(
        self,
        prior_table: torch.Tensor,
        decoder_channels: int,
        hidden_dim: int = 128,
        strength: float = 0.1,
    ) -> None:
        super().__init__()
        if prior_table.ndim != 2 or prior_table.shape[1] != len(MONTH_COLUMNS):
            raise ValueError(
                "Expected phenology prior [num_classes, 12], "
                f"got {tuple(prior_table.shape)}"
            )
        if not math.isfinite(float(strength)) or float(strength) < 0.0:
            raise ValueError(f"Phenology prior strength must be non-negative, got {strength}")

        self.register_buffer("prior_table", prior_table.float().contiguous())
        self.strength = float(strength)
        self.projector = nn.Sequential(
            nn.Linear(prior_table.shape[0], int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(decoder_channels)),
            nn.LayerNorm(int(decoder_channels)),
        )

    def forward(self, months: torch.Tensor) -> torch.Tensor:
        if months.ndim != 2:
            raise ValueError(f"Expected months [B,T], got {tuple(months.shape)}")
        months = months.long()
        if torch.any((months < 0) | (months >= len(MONTH_COLUMNS))):
            raise ValueError("Month indices must be calendar values in [0, 11].")

        # [K, B, T] -> [B, T, K]. The table is indexed by natural month,
        # while the cache stores the actual calendar month for each timestep.
        prior = self.prior_table[:, months].permute(1, 2, 0).contiguous()
        return self.projector(prior) * self.strength


def build_phenology_prior(
    config: dict,
    decoder_channels: int,
) -> PhenologyPriorAdapter | None:
    phenology_cfg = config.get("phenology", {}) or {}
    if not bool(phenology_cfg.get("enabled", False)):
        return None

    path = phenology_cfg.get("path")
    if not path:
        raise ValueError("phenology.enabled=true requires phenology.path")
    num_classes = int(config["data"]["num_classes"])
    table = load_phenology_prior(path, num_classes=num_classes)
    return PhenologyPriorAdapter(
        prior_table=table,
        decoder_channels=int(decoder_channels),
        hidden_dim=int(phenology_cfg.get("hidden_dim", 128)),
        strength=float(phenology_cfg.get("strength", 0.1)),
    )
