from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .environment_priors import (
    CLIMATE_FEATURES,
    GEOGRAPHY_FEATURES,
    SOIL_DEPTHS,
    SOIL_FEATURES,
    PatchClimatePriorEncoder,
    PatchNumericPriorEncoder,
    PatchSoilPriorEncoder,
)
from .phenology import build_phenology_token_encoder_from_source
from .prior_injection import PriorBatch, PriorTokenEncoder


class CompositePriorTokenEncoder(PriorTokenEncoder):
    """Concatenate independently encoded prior sources into one CA-HPI set."""

    def __init__(
        self,
        encoders: Sequence[PriorTokenEncoder],
        source_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("Composite prior encoder needs at least one source.")
        self.encoders = nn.ModuleList(encoders)
        if source_names is None:
            source_names = [f"source_{index}" for index in range(len(encoders))]
        if len(source_names) != len(encoders):
            raise ValueError("source_names must align with prior encoders.")
        self.source_names = tuple(str(name) for name in source_names)
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError("CA-HPI prior source names must be unique.")

    def forward(self, batch_size: int, batch: dict | None = None) -> PriorBatch:
        source_batches = [
            encoder(batch_size=batch_size, batch=batch) for encoder in self.encoders
        ]
        token_dims = {item.tokens.shape[-1] for item in source_batches}
        if len(token_dims) != 1:
            raise ValueError("All prior sources must use the same token_dim.")
        device = source_batches[0].tokens.device
        for item in source_batches[1:]:
            if item.tokens.device != device:
                raise RuntimeError("All prior sources must be on the same device.")
        type_ids = [
            torch.full_like(item.type_ids, source_index)
            for source_index, item in enumerate(source_batches)
        ]
        return PriorBatch(
            tokens=torch.cat([item.tokens for item in source_batches], dim=1),
            mask=torch.cat([item.mask for item in source_batches], dim=1),
            confidence=torch.cat([item.confidence for item in source_batches], dim=1),
            type_ids=torch.cat(type_ids, dim=1),
            source_names=self.source_names,
        )


def _source_mapping_list(prior_cfg: dict) -> list[dict]:
    has_source = "source" in prior_cfg and prior_cfg.get("source") is not None
    has_sources = "sources" in prior_cfg and prior_cfg.get("sources") is not None
    if has_source and has_sources:
        raise ValueError(
            "prior_injection may define either source or sources, not both."
        )
    if has_source:
        source = prior_cfg["source"]
        if not isinstance(source, dict):
            raise ValueError("prior_injection.source must be a mapping.")
        return [source]
    if has_sources:
        sources = prior_cfg["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError(
                "prior_injection.sources must be a non-empty list of mappings."
            )
        if not all(isinstance(source, dict) for source in sources):
            raise ValueError("Every prior_injection.sources item must be a mapping.")
        return list(sources)
    raise ValueError("prior_injection needs a source or sources definition.")


def build_prior_token_encoder(config: dict) -> PriorTokenEncoder | None:
    """Build M1/M2/M3 adapters while retaining the legacy single-source schema."""

    prior_cfg = config.get("prior_injection", {}) or {}
    if not bool(prior_cfg.get("enabled", False)):
        return None
    method = str(prior_cfg.get("method", "ca_hpi")).lower()
    supported_methods = {
        "ca_hpi",
        "cahpi",
        "content_aware",
        "source_aware_spatial_film",
        "source_aware_film",
        "spatial_film",
        "sa_sfilm",
    }
    if method not in supported_methods:
        raise ValueError(f"Unsupported prior_injection.method: {method}")
    encoder_cfg = prior_cfg.get("encoder", {}) or {}
    if not isinstance(encoder_cfg, dict):
        raise ValueError("prior_injection.encoder must be a mapping.")
    token_dim = int(prior_cfg.get("token_dim", 128))
    hidden_dim = int(encoder_cfg.get("hidden_dim", 128))
    time_frequencies = int(encoder_cfg.get("time_frequencies", 4))
    dropout = float(encoder_cfg.get("dropout", 0.0))
    num_classes = int(config["data"]["num_classes"])

    encoders: list[PriorTokenEncoder] = []
    source_names: list[str] = []
    for source_index, source_cfg in enumerate(_source_mapping_list(prior_cfg)):
        source_kind = str(source_cfg.get("kind", "phenology_table")).lower()
        source_names.append(str(source_cfg.get("name", f"source_{source_index}")))
        if source_kind in {"phenology_table", "class_month_table"}:
            encoders.append(
                build_phenology_token_encoder_from_source(
                    source_cfg,
                    num_classes=num_classes,
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    time_frequencies=time_frequencies,
                    dropout=dropout,
                )
            )
        elif source_kind in {"climate_table", "era5_land"}:
            path = source_cfg.get("path")
            stats_path = source_cfg.get("stats_path")
            if not path or not stats_path:
                raise ValueError("Climate prior source needs path and stats_path.")
            encoders.append(
                PatchClimatePriorEncoder(
                    table_path=path,
                    stats_path=stats_path,
                    features=source_cfg.get("features", CLIMATE_FEATURES),
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    time_frequencies=time_frequencies,
                    dropout=dropout,
                    patch_id_column=str(source_cfg.get("patch_id_column", "patch_id")),
                    month_column=str(source_cfg.get("month_column", "month")),
                    valid_column=str(source_cfg.get("valid_column", "valid")),
                    confidence_column=str(
                        source_cfg.get("confidence_column", "confidence")
                    ),
                    allow_missing_patch=bool(
                        source_cfg.get("allow_missing_patch", False)
                    ),
                )
            )
        elif source_kind in {"soil_table", "soilgrids"}:
            path = source_cfg.get("path")
            stats_path = source_cfg.get("stats_path")
            if not path or not stats_path:
                raise ValueError("Soil prior source needs path and stats_path.")
            encoders.append(
                PatchSoilPriorEncoder(
                    table_path=path,
                    stats_path=stats_path,
                    features=source_cfg.get("features", SOIL_FEATURES),
                    depths=source_cfg.get("depths", SOIL_DEPTHS),
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    time_frequencies=time_frequencies,
                    dropout=dropout,
                    patch_id_column=str(source_cfg.get("patch_id_column", "patch_id")),
                    depth_column=str(source_cfg.get("depth_column", "depth_cm")),
                    valid_column=str(source_cfg.get("valid_column", "valid")),
                    confidence_column=str(
                        source_cfg.get("confidence_column", "confidence")
                    ),
                    allow_missing_patch=bool(
                        source_cfg.get("allow_missing_patch", False)
                    ),
                )
            )
        elif source_kind in {
            "patch_numeric_table",
            "geography_table",
            "location_table",
        }:
            path = source_cfg.get("path")
            stats_path = source_cfg.get("stats_path")
            if not path or not stats_path:
                raise ValueError(
                    "Patch numeric prior source needs path and stats_path."
                )
            encoders.append(
                PatchNumericPriorEncoder(
                    table_path=path,
                    stats_path=stats_path,
                    features=source_cfg.get("features", GEOGRAPHY_FEATURES),
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    time_frequencies=time_frequencies,
                    dropout=dropout,
                    patch_id_column=str(source_cfg.get("patch_id_column", "patch_id")),
                    valid_column=str(source_cfg.get("valid_column", "valid")),
                    confidence_column=str(
                        source_cfg.get("confidence_column", "confidence")
                    ),
                    allow_missing_patch=bool(
                        source_cfg.get("allow_missing_patch", False)
                    ),
                )
            )
        else:
            raise ValueError(f"Unsupported CA-HPI prior source kind: {source_kind}")
    return CompositePriorTokenEncoder(encoders, source_names=source_names)
