from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .prior_injection import PriorBatch, PriorTokenEncoder, StructuredPriorEncoder


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

DEFAULT_CONFIDENCE_MAPPING = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
}


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


def load_phenology_confidence(
    path: str | Path,
    num_classes: int,
    confidence_mapping: dict[str, float] | None = None,
    default_confidence: float = 1.0,
) -> torch.Tensor:
    """Load one auditable confidence value per class and repeat over months."""

    if not math.isfinite(float(default_confidence)) or not 0.0 <= float(
        default_confidence
    ) <= 1.0:
        raise ValueError("default_confidence must be finite and in [0,1].")
    mapping = {
        str(key).strip().lower(): float(value)
        for key, value in (confidence_mapping or DEFAULT_CONFIDENCE_MAPPING).items()
    }
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in mapping.values()
    ):
        raise ValueError("confidence mapping values must be finite and in [0,1].")

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing phenology prior CSV: {source}")
    confidence = np.full(
        (int(num_classes), len(MONTH_COLUMNS)),
        float(default_confidence),
        dtype=np.float32,
    )
    seen: set[int] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "class_id" not in reader.fieldnames:
            raise ValueError(f"Phenology prior CSV needs class_id: {source}")
        has_confidence = "confidence" in reader.fieldnames
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

            raw_value = str(row.get("confidence", "") if has_confidence else "").strip()
            if not raw_value:
                value = float(default_confidence)
            elif raw_value.lower() in mapping:
                value = mapping[raw_value.lower()]
            else:
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Unknown confidence {raw_value!r} for class_id={class_id}"
                    ) from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Confidence must be in [0,1], got class_id={class_id}, value={value}"
                )
            confidence[class_id, :] = value

    missing_classes = sorted(set(range(int(num_classes))) - seen)
    if missing_classes:
        raise ValueError(
            f"Phenology prior is missing class ids {missing_classes}: {source}"
        )
    return torch.from_numpy(confidence)


class PhenologyPriorAdapter(nn.Module):
    """Project class-month priors into a Galileo temporal feature context.

    The output is added to frozen Galileo features before they enter a
    time-preserving decoder. The same class-by-month table is deliberately
    shared by every selected encoder layer; decoder-specific layer handling
    remains inside the decoder.
    """

    def __init__(
        self,
        prior_table: torch.Tensor,
        feature_channels: int,
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
            nn.Linear(int(hidden_dim), int(feature_channels)),
            nn.LayerNorm(int(feature_channels)),
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


def inject_temporal_phenology_prior(
    features: tuple[torch.Tensor, ...] | list[torch.Tensor],
    months: torch.Tensor,
    prior_adapter: PhenologyPriorAdapter | None,
) -> tuple[torch.Tensor, ...]:
    """Add one shared calendar-conditioned residual to each temporal layer.

    Both inputs and outputs use ``[B, T, D, H, W]``. Keeping this operation
    at the cached/Galileo feature boundary makes it independent of the
    downstream decoder architecture.
    """

    features = tuple(features)
    if prior_adapter is None:
        return features
    if not features:
        raise ValueError("Cannot inject a phenology prior into an empty feature tuple.")

    context = prior_adapter(months)
    if context.ndim != 3:
        raise ValueError(f"Expected prior context [B, T, D], got {tuple(context.shape)}")
    context = context.unsqueeze(-1).unsqueeze(-1)

    injected = []
    for feature in features:
        if feature.ndim != 5:
            raise ValueError(
                "Expected temporal features [B, T, D, H, W], "
                f"got {tuple(feature.shape)}"
            )
        if feature.shape[:3] != context.shape[:3]:
            raise ValueError(
                "Temporal feature and phenology context shapes do not match: "
                f"{tuple(feature.shape[:3])} vs {tuple(context.shape[:3])}"
            )
        injected.append(feature + context)
    return tuple(injected)


def build_phenology_prior(
    config: dict,
    feature_channels: int,
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
        feature_channels=int(feature_channels),
        hidden_dim=int(phenology_cfg.get("hidden_dim", 128)),
        strength=float(phenology_cfg.get("strength", 0.1)),
    )


class PhenologyPriorTokenEncoder(PriorTokenEncoder):
    """Convert a class-by-month table into structured heterogeneous tokens."""

    NUMERIC_TYPE_ID = 0

    def __init__(
        self,
        prior_table: torch.Tensor,
        token_dim: int = 128,
        hidden_dim: int = 128,
        time_frequencies: int = 4,
        default_confidence: float = 1.0,
        confidence_table: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if prior_table.ndim != 2:
            raise ValueError(
                f"Expected prior_table [entities,months], got {tuple(prior_table.shape)}"
            )
        num_entities, num_months = prior_table.shape
        if num_entities < 1 or num_months < 1:
            raise ValueError("prior_table needs at least one entity and month.")
        if not math.isfinite(float(default_confidence)) or not 0.0 <= float(
            default_confidence
        ) <= 1.0:
            raise ValueError("default_confidence must be finite and in [0,1].")

        prior_table = prior_table.float()
        inferred_mask = torch.isfinite(prior_table)
        if valid_mask is None:
            valid_mask = inferred_mask
        else:
            if tuple(valid_mask.shape) != tuple(prior_table.shape):
                raise ValueError("valid_mask must match prior_table.")
            valid_mask = valid_mask.bool() & inferred_mask
        safe_table = torch.where(valid_mask, prior_table, torch.zeros_like(prior_table))
        valid_values = safe_table[valid_mask]
        if valid_values.numel() and torch.any(
            (valid_values < 0.0) | (valid_values > 1.0)
        ):
            raise ValueError("valid phenology prior values must be in [0,1].")

        if confidence_table is None:
            confidence_table = torch.full_like(
                safe_table,
                float(default_confidence),
            )
        else:
            if tuple(confidence_table.shape) != tuple(prior_table.shape):
                raise ValueError("confidence_table must match prior_table.")
            confidence_table = confidence_table.float()
        if not torch.isfinite(confidence_table).all() or torch.any(
            (confidence_table < 0.0) | (confidence_table > 1.0)
        ):
            raise ValueError("confidence_table must be finite and in [0,1].")
        confidence_table = torch.where(
            valid_mask,
            confidence_table,
            torch.zeros_like(confidence_table),
        )

        entity_ids = torch.arange(num_entities, dtype=torch.long).view(-1, 1).expand(
            num_entities, num_months
        )
        month_ids = torch.arange(num_months, dtype=torch.long).view(1, -1).expand(
            num_entities, num_months
        )

        self.num_entities = int(num_entities)
        self.num_months = int(num_months)
        self.num_tokens = self.num_entities * self.num_months
        self.register_buffer(
            "numeric_values",
            safe_table.reshape(1, self.num_tokens, 1).contiguous(),
        )
        self.register_buffer(
            "token_mask",
            valid_mask.reshape(1, self.num_tokens).contiguous(),
        )
        self.register_buffer(
            "token_confidence",
            confidence_table.reshape(1, self.num_tokens).contiguous(),
        )
        self.register_buffer(
            "type_ids",
            torch.full(
                (1, self.num_tokens),
                self.NUMERIC_TYPE_ID,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "entity_ids",
            entity_ids.reshape(1, self.num_tokens).contiguous(),
        )
        self.register_buffer(
            "time_values",
            (
                month_ids.float().reshape(1, self.num_tokens)
                / float(self.num_months)
            ).contiguous(),
        )
        self.encoder = StructuredPriorEncoder(
            numeric_dim=1,
            token_dim=int(token_dim),
            hidden_dim=int(hidden_dim),
            num_types=1,
            num_entities=self.num_entities,
            time_frequencies=int(time_frequencies),
            dropout=float(dropout),
        )

    def forward(
        self,
        batch_size: int,
        batch: dict | None = None,
    ) -> PriorBatch:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return self.encoder(
            numeric_values=self.numeric_values.expand(batch_size, -1, -1),
            mask=self.token_mask.expand(batch_size, -1),
            confidence=self.token_confidence.expand(batch_size, -1),
            type_ids=self.type_ids.expand(batch_size, -1),
            entity_ids=self.entity_ids.expand(batch_size, -1),
            time_values=self.time_values.expand(batch_size, -1),
        )


def build_phenology_token_encoder_from_source(
    source_cfg: dict,
    *,
    num_classes: int,
    token_dim: int,
    hidden_dim: int,
    time_frequencies: int,
    dropout: float,
) -> PhenologyPriorTokenEncoder:
    """Build the M1 adapter from one source mapping.

    Kept separate from the config wrapper so multi-source CA-HPI can reuse
    exactly the same audited class-month table behaviour as the original M1.
    """
    if not isinstance(source_cfg, dict):
        raise ValueError("Phenology source must be a mapping.")
    source_kind = str(source_cfg.get("kind", "phenology_table")).lower()
    if source_kind not in {"phenology_table", "class_month_table"}:
        raise ValueError(f"Unsupported prior source kind: {source_kind}")
    path = source_cfg.get("path")
    if not path:
        raise ValueError("Phenology prior source.path is required.")

    table = load_phenology_prior(path, num_classes=num_classes)
    confidence_mapping = source_cfg.get("confidence_mapping")
    if confidence_mapping is not None and not isinstance(confidence_mapping, dict):
        raise ValueError("Phenology source.confidence_mapping must be a mapping.")
    confidence_table = load_phenology_confidence(
        path,
        num_classes=num_classes,
        confidence_mapping=confidence_mapping,
        default_confidence=float(source_cfg.get("default_confidence", 1.0)),
    )
    return PhenologyPriorTokenEncoder(
        prior_table=table,
        token_dim=int(token_dim),
        hidden_dim=int(hidden_dim),
        time_frequencies=int(time_frequencies),
        confidence_table=confidence_table,
        default_confidence=float(source_cfg.get("default_confidence", 1.0)),
        dropout=float(dropout),
    )


def build_phenology_token_encoder(config: dict) -> PhenologyPriorTokenEncoder | None:
    """Compatibility wrapper for the original one-source M1 configuration."""
    prior_cfg = config.get("prior_injection", {}) or {}
    if not bool(prior_cfg.get("enabled", False)):
        return None
    source_cfg = prior_cfg.get("source", {}) or {}
    encoder_cfg = prior_cfg.get("encoder", {}) or {}
    return build_phenology_token_encoder_from_source(
        source_cfg,
        num_classes=int(config["data"]["num_classes"]),
        token_dim=int(prior_cfg.get("token_dim", 128)),
        hidden_dim=int(encoder_cfg.get("hidden_dim", 128)),
        time_frequencies=int(encoder_cfg.get("time_frequencies", 4)),
        dropout=float(encoder_cfg.get("dropout", 0.0)),
    )
