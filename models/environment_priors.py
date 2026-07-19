from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from .prior_injection import PriorBatch, PriorTokenEncoder, StructuredPriorEncoder


CLIMATE_FEATURES = ("t2m_c", "tp_mm", "ssrd_mj_m2", "swvl1")
SOIL_FEATURES = (
    "ph",
    "soc_gkg",
    "clay_pct",
    "sand_pct",
    "cec_cmolkg",
    "nitrogen_gkg",
)
SOIL_DEPTHS = ("0-5", "5-15", "15-30")
GEOGRAPHY_FEATURES = ("lon", "lat")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Environmental prior table does not exist: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Environmental prior table has no header: {source}")
        return list(reader)


def _require_columns(
    rows: list[dict[str, str]],
    required: Sequence[str],
    source: str | Path,
) -> None:
    if not rows:
        raise ValueError(f"Environmental prior table is empty: {source}")
    columns = set(rows[0])
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(
            f"Environmental prior table {source} is missing columns: {missing}"
        )


def _parse_patch_id(value: object, source: str | Path, row_number: int) -> int:
    try:
        patch_id = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid patch_id in {source} row {row_number}: {value!r}"
        ) from exc
    if patch_id < 0:
        raise ValueError(f"patch_id must be non-negative in {source} row {row_number}.")
    return patch_id


def _parse_float(value: object, label: str, source: str | Path, row_number: int) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {label} in {source} row {row_number}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite in {source} row {row_number}.")
    return parsed


def _parse_bool(value: object, label: str, source: str | Path, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(
        f"Invalid {label} in {source} row {row_number}: {value!r}; use true/false."
    )


def _load_standardization(
    path: str | Path,
    features: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Environmental prior statistics do not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Environmental prior statistics must be a JSON object: {source}")
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, dict) or not isinstance(std, dict):
        raise ValueError(f"Environmental statistics need 'mean' and 'std' mappings: {source}")
    missing = [feature for feature in features if feature not in mean or feature not in std]
    if missing:
        raise ValueError(f"Environmental statistics miss fields {missing}: {source}")
    means = torch.tensor([float(mean[feature]) for feature in features], dtype=torch.float32)
    stds = torch.tensor([float(std[feature]) for feature in features], dtype=torch.float32)
    if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or torch.any(stds <= 0):
        raise ValueError(f"Environmental statistics need finite positive standard deviations: {source}")
    return means, stds


def _batch_patch_ids(batch_size: int, batch: dict | None) -> list[int]:
    if batch is None:
        raise ValueError("Patch-keyed environmental priors require the training batch.")
    raw_patch_ids: object | None = batch.get("patch_id")
    if raw_patch_ids is None:
        samples = batch.get("samples")
        if isinstance(samples, (list, tuple)):
            raw_patch_ids = [sample.get("patch_id") for sample in samples]
    if raw_patch_ids is None:
        raise ValueError("Environmental prior batch has no patch_id information.")
    if isinstance(raw_patch_ids, torch.Tensor):
        values = raw_patch_ids.detach().cpu().reshape(-1).tolist()
    elif isinstance(raw_patch_ids, (list, tuple)):
        values = list(raw_patch_ids)
    else:
        values = [raw_patch_ids]
    if len(values) != int(batch_size):
        raise ValueError(
            "Environmental prior patch_id count does not match batch size: "
            f"{len(values)} != {batch_size}."
        )
    return [int(value) for value in values]


class _PatchTablePriorEncoder(PriorTokenEncoder):
    """Shared patch-id lookup for frozen, external environmental tables."""

    def __init__(
        self,
        patch_ids: Sequence[int],
        numeric_values: torch.Tensor,
        token_mask: torch.Tensor,
        token_confidence: torch.Tensor,
        type_ids: torch.Tensor,
        structured_encoder: StructuredPriorEncoder,
        entity_ids: torch.Tensor | None = None,
        time_values: torch.Tensor | None = None,
        allow_missing_patch: bool = False,
    ) -> None:
        super().__init__()
        if len(patch_ids) != numeric_values.shape[0]:
            raise ValueError("patch_ids must align with the environmental table rows.")
        if len(set(int(value) for value in patch_ids)) != len(patch_ids):
            raise ValueError("Environmental prior patch_ids must be unique after grouping.")
        if numeric_values.ndim != 3:
            raise ValueError("Environmental prior values must be [patch, token, feature].")
        expected = numeric_values.shape[:2]
        if tuple(token_mask.shape) != expected or tuple(token_confidence.shape) != expected:
            raise ValueError("Environmental mask/confidence must match [patch, token].")
        if tuple(type_ids.shape) != expected:
            raise ValueError("Environmental type_ids must match [patch, token].")
        if entity_ids is not None and tuple(entity_ids.shape) != expected:
            raise ValueError("Environmental entity_ids must match [patch, token].")
        if time_values is not None and tuple(time_values.shape[:2]) != expected:
            raise ValueError("Environmental time_values must start with [patch, token].")

        self.patch_row = {int(patch_id): index for index, patch_id in enumerate(patch_ids)}
        self.allow_missing_patch = bool(allow_missing_patch)
        self.register_buffer("numeric_values", numeric_values.float().contiguous())
        self.register_buffer("token_mask", token_mask.bool().contiguous())
        self.register_buffer("token_confidence", token_confidence.float().contiguous())
        self.register_buffer("type_ids", type_ids.long().contiguous())
        self.register_buffer(
            "entity_ids",
            None if entity_ids is None else entity_ids.long().contiguous(),
        )
        self.register_buffer(
            "time_values",
            None if time_values is None else time_values.float().contiguous(),
        )
        self.encoder = structured_encoder

    def _lookup_rows(self, patch_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        row_values = [self.patch_row.get(int(patch_id), -1) for patch_id in patch_ids]
        missing = [patch_id for patch_id, row in zip(patch_ids, row_values) if row < 0]
        if missing and not self.allow_missing_patch:
            examples = ", ".join(str(value) for value in missing[:5])
            raise KeyError(
                "Environmental prior table has no rows for patch_id(s): "
                f"{examples}. Prepare the table for all PASTIS folds or set "
                "allow_missing_patch=true deliberately."
            )
        known = torch.tensor(
            [row >= 0 for row in row_values],
            dtype=torch.bool,
            device=self.numeric_values.device,
        )
        safe_rows = torch.tensor(
            [max(row, 0) for row in row_values],
            dtype=torch.long,
            device=self.numeric_values.device,
        )
        return safe_rows, known

    def forward(self, batch_size: int, batch: dict | None = None) -> PriorBatch:
        patch_ids = _batch_patch_ids(int(batch_size), batch)
        rows, known = self._lookup_rows(patch_ids)
        mask = self.token_mask.index_select(0, rows) & known.unsqueeze(1)
        confidence = self.token_confidence.index_select(0, rows) * mask.to(torch.float32)
        kwargs: dict[str, torch.Tensor] = {
            "numeric_values": self.numeric_values.index_select(0, rows),
            "mask": mask,
            "confidence": confidence,
            "type_ids": self.type_ids.index_select(0, rows),
        }
        if self.entity_ids is not None:
            kwargs["entity_ids"] = self.entity_ids.index_select(0, rows)
        if self.time_values is not None:
            kwargs["time_values"] = self.time_values.index_select(0, rows)
        return self.encoder(**kwargs)


class PatchNumericPriorEncoder(_PatchTablePriorEncoder):
    """Encode one static numeric context token for each patch.

    This is the generic adapter for patch-keyed metadata that is available at
    inference time, such as geographic coordinates, topography, or other
    frozen environmental summaries.  Feature normalization must be supplied
    by a statistics file fitted on the training folds only.
    """

    def __init__(
        self,
        table_path: str | Path,
        stats_path: str | Path,
        features: Sequence[str],
        token_dim: int = 128,
        hidden_dim: int = 128,
        time_frequencies: int = 4,
        dropout: float = 0.0,
        patch_id_column: str = "patch_id",
        valid_column: str = "valid",
        confidence_column: str = "confidence",
        allow_missing_patch: bool = False,
    ) -> None:
        features = tuple(str(feature) for feature in features)
        if not features:
            raise ValueError("Patch numeric prior needs at least one feature.")
        rows = _read_csv_rows(table_path)
        _require_columns(rows, [patch_id_column, *features], table_path)
        means, stds = _load_standardization(stats_path, features)
        has_valid = valid_column in rows[0]
        has_confidence = confidence_column in rows[0]

        grouped: dict[int, tuple[list[float], bool, float]] = {}
        for row_number, row in enumerate(rows, start=2):
            patch_id = _parse_patch_id(row[patch_id_column], table_path, row_number)
            if patch_id in grouped:
                raise ValueError(f"Duplicate patch_id in {table_path}: {patch_id}")
            valid = (
                _parse_bool(row[valid_column], valid_column, table_path, row_number)
                if has_valid
                else True
            )
            confidence = (
                _parse_float(row[confidence_column], confidence_column, table_path, row_number)
                if has_confidence
                else 1.0
            )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Patch numeric confidence must be in [0,1] in "
                    f"{table_path} row {row_number}."
                )
            if valid:
                values = [
                    _parse_float(row[feature], feature, table_path, row_number)
                    for feature in features
                ]
            else:
                values = [0.0] * len(features)
                confidence = 0.0
            grouped[patch_id] = (values, valid, confidence)

        patch_ids = sorted(grouped)
        values = torch.zeros((len(patch_ids), 1, len(features)), dtype=torch.float32)
        mask = torch.zeros((len(patch_ids), 1), dtype=torch.bool)
        confidence = torch.zeros((len(patch_ids), 1), dtype=torch.float32)
        for patch_index, patch_id in enumerate(patch_ids):
            raw_values, valid, row_confidence = grouped[patch_id]
            values[patch_index, 0] = torch.tensor(raw_values)
            mask[patch_index, 0] = valid
            confidence[patch_index, 0] = row_confidence
        values = (values - means.view(1, 1, -1)) / stds.view(1, 1, -1)
        values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
        type_ids = torch.zeros((len(patch_ids), 1), dtype=torch.long)
        super().__init__(
            patch_ids=patch_ids,
            numeric_values=values,
            token_mask=mask,
            token_confidence=confidence,
            type_ids=type_ids,
            structured_encoder=StructuredPriorEncoder(
                numeric_dim=len(features),
                token_dim=int(token_dim),
                hidden_dim=int(hidden_dim),
                num_types=1,
                time_frequencies=int(time_frequencies),
                dropout=float(dropout),
            ),
            allow_missing_patch=allow_missing_patch,
        )
        self.features = features


class PatchClimatePriorEncoder(_PatchTablePriorEncoder):
    """Encode one 12-month ERA5-Land context sequence per PASTIS patch."""

    def __init__(
        self,
        table_path: str | Path,
        stats_path: str | Path,
        features: Sequence[str] = CLIMATE_FEATURES,
        token_dim: int = 128,
        hidden_dim: int = 128,
        time_frequencies: int = 4,
        dropout: float = 0.0,
        patch_id_column: str = "patch_id",
        month_column: str = "month",
        valid_column: str = "valid",
        confidence_column: str = "confidence",
        allow_missing_patch: bool = False,
    ) -> None:
        features = tuple(str(feature) for feature in features)
        if not features:
            raise ValueError("Climate prior needs at least one numeric feature.")
        rows = _read_csv_rows(table_path)
        _require_columns(rows, [patch_id_column, month_column, *features], table_path)
        means, stds = _load_standardization(stats_path, features)
        has_valid = valid_column in rows[0]
        has_confidence = confidence_column in rows[0]

        grouped: dict[int, dict[int, tuple[list[float], bool, float]]] = {}
        for row_number, row in enumerate(rows, start=2):
            patch_id = _parse_patch_id(row[patch_id_column], table_path, row_number)
            try:
                month = int(str(row[month_column]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid month in {table_path} row {row_number}: {row[month_column]!r}"
                ) from exc
            if month not in range(1, 13):
                raise ValueError(f"Climate month must be 1..12 in {table_path} row {row_number}.")
            if month in grouped.setdefault(patch_id, {}):
                raise ValueError(f"Duplicate patch_id/month in {table_path}: {patch_id}/{month}")
            valid = (
                _parse_bool(row[valid_column], valid_column, table_path, row_number)
                if has_valid
                else True
            )
            confidence = (
                _parse_float(row[confidence_column], confidence_column, table_path, row_number)
                if has_confidence
                else 1.0
            )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Climate confidence must be in [0,1] in {table_path} row {row_number}."
                )
            if valid:
                values = [
                    _parse_float(row[feature], feature, table_path, row_number)
                    for feature in features
                ]
            else:
                values = [0.0] * len(features)
                confidence = 0.0
            grouped[patch_id][month] = (values, valid, confidence)

        patch_ids = sorted(grouped)
        values = torch.zeros((len(patch_ids), 12, len(features)), dtype=torch.float32)
        mask = torch.zeros((len(patch_ids), 12), dtype=torch.bool)
        confidence = torch.zeros((len(patch_ids), 12), dtype=torch.float32)
        for patch_index, patch_id in enumerate(patch_ids):
            for month, (raw_values, valid, row_confidence) in grouped[patch_id].items():
                token_index = month - 1
                values[patch_index, token_index] = torch.tensor(raw_values)
                mask[patch_index, token_index] = valid
                confidence[patch_index, token_index] = row_confidence
        values = (values - means.view(1, 1, -1)) / stds.view(1, 1, -1)
        values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
        type_ids = torch.zeros((len(patch_ids), 12), dtype=torch.long)
        time_values = torch.arange(12, dtype=torch.float32).view(1, 12).expand(len(patch_ids), -1) / 12.0
        super().__init__(
            patch_ids=patch_ids,
            numeric_values=values,
            token_mask=mask,
            token_confidence=confidence,
            type_ids=type_ids,
            time_values=time_values,
            structured_encoder=StructuredPriorEncoder(
                numeric_dim=len(features),
                token_dim=int(token_dim),
                hidden_dim=int(hidden_dim),
                num_types=1,
                time_frequencies=int(time_frequencies),
                dropout=float(dropout),
            ),
            allow_missing_patch=allow_missing_patch,
        )
        self.features = features


class PatchSoilPriorEncoder(_PatchTablePriorEncoder):
    """Encode three frozen SoilGrids depth contexts per PASTIS patch."""

    def __init__(
        self,
        table_path: str | Path,
        stats_path: str | Path,
        features: Sequence[str] = SOIL_FEATURES,
        depths: Sequence[str] = SOIL_DEPTHS,
        token_dim: int = 128,
        hidden_dim: int = 128,
        time_frequencies: int = 4,
        dropout: float = 0.0,
        patch_id_column: str = "patch_id",
        depth_column: str = "depth_cm",
        valid_column: str = "valid",
        confidence_column: str = "confidence",
        allow_missing_patch: bool = False,
    ) -> None:
        features = tuple(str(feature) for feature in features)
        depths = tuple(str(depth) for depth in depths)
        if not features or not depths:
            raise ValueError("Soil prior needs at least one feature and one depth.")
        if len(set(depths)) != len(depths):
            raise ValueError("Soil prior depth labels must be unique.")
        rows = _read_csv_rows(table_path)
        _require_columns(rows, [patch_id_column, depth_column, *features], table_path)
        means, stds = _load_standardization(stats_path, features)
        has_valid = valid_column in rows[0]
        has_confidence = confidence_column in rows[0]
        depth_to_index = {depth: index for index, depth in enumerate(depths)}

        grouped: dict[int, dict[str, tuple[list[float], bool, float]]] = {}
        for row_number, row in enumerate(rows, start=2):
            patch_id = _parse_patch_id(row[patch_id_column], table_path, row_number)
            depth = str(row[depth_column]).strip()
            if depth not in depth_to_index:
                raise ValueError(
                    f"Unsupported soil depth {depth!r} in {table_path} row {row_number}; "
                    f"expected one of {list(depths)}."
                )
            if depth in grouped.setdefault(patch_id, {}):
                raise ValueError(f"Duplicate patch_id/depth in {table_path}: {patch_id}/{depth}")
            valid = (
                _parse_bool(row[valid_column], valid_column, table_path, row_number)
                if has_valid
                else True
            )
            confidence = (
                _parse_float(row[confidence_column], confidence_column, table_path, row_number)
                if has_confidence
                else 1.0
            )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Soil confidence must be in [0,1] in {table_path} row {row_number}."
                )
            if valid:
                values = [
                    _parse_float(row[feature], feature, table_path, row_number)
                    for feature in features
                ]
            else:
                values = [0.0] * len(features)
                confidence = 0.0
            grouped[patch_id][depth] = (values, valid, confidence)

        patch_ids = sorted(grouped)
        values = torch.zeros((len(patch_ids), len(depths), len(features)), dtype=torch.float32)
        mask = torch.zeros((len(patch_ids), len(depths)), dtype=torch.bool)
        confidence = torch.zeros((len(patch_ids), len(depths)), dtype=torch.float32)
        for patch_index, patch_id in enumerate(patch_ids):
            for depth, (raw_values, valid, row_confidence) in grouped[patch_id].items():
                token_index = depth_to_index[depth]
                values[patch_index, token_index] = torch.tensor(raw_values)
                mask[patch_index, token_index] = valid
                confidence[patch_index, token_index] = row_confidence
        values = (values - means.view(1, 1, -1)) / stds.view(1, 1, -1)
        values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
        type_ids = torch.zeros((len(patch_ids), len(depths)), dtype=torch.long)
        entity_ids = torch.arange(len(depths), dtype=torch.long).view(1, -1).expand(len(patch_ids), -1)
        super().__init__(
            patch_ids=patch_ids,
            numeric_values=values,
            token_mask=mask,
            token_confidence=confidence,
            type_ids=type_ids,
            entity_ids=entity_ids,
            structured_encoder=StructuredPriorEncoder(
                numeric_dim=len(features),
                token_dim=int(token_dim),
                hidden_dim=int(hidden_dim),
                num_types=1,
                num_entities=len(depths),
                time_frequencies=int(time_frequencies),
                dropout=float(dropout),
            ),
            allow_missing_patch=allow_missing_patch,
        )
        self.features = features
        self.depths = depths
