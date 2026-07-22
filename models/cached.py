from __future__ import annotations

import torch
from torch import nn

from models.decoders import (
    DPTMultiLayerDecoder,
    DPTSingleLayerDecoder,
    GalileoDPTDecoder,
    GalileoLinearProbeDecoder,
    TEMPORAL_READOUT_DECODER_BASES,
    ThreeDAwareDPTDecoder,
    TemporalReadoutDecoder,
    UPerNetDecoder,
)
from models.phenology import (
    PhenologyPriorAdapter,
    build_phenology_prior,
    inject_temporal_phenology_prior,
)
from models.prior_injection import (
    PriorTokenEncoder,
    TemporalFeaturePyramidPriorInjection,
    build_temporal_prior_injection,
    prior_injection_enabled,
)
from models.prior_sources import build_prior_token_encoder


THREE_D_CACHED_DECODER_NAMES = {
    "3d_aware_dpt",
    "3d-aware-dpt",
    "three_d_aware_dpt",
}
TEMPORAL_CACHED_DECODER_NAMES = (
    THREE_D_CACHED_DECODER_NAMES | set(TEMPORAL_READOUT_DECODER_BASES)
)
FEATURE_PYRAMID_CACHED_DECODER_NAMES = {
    "multi_layer_dpt",
    "multilayer_dpt",
    "galileo_dpt",
    "galileo_adapted_dpt",
    "multiscale_dpt",
    "upernet",
    "upernet_style",
    "upernet-style",
}


def cached_decoder_uses_temporal_features(config: dict) -> bool:
    decoder_name = str(config.get("model", {}).get("decoder", "")).lower()
    return decoder_name in TEMPORAL_CACHED_DECODER_NAMES


def cached_decoder_uses_feature_pyramid(config: dict) -> bool:
    decoder_name = str(config.get("model", {}).get("decoder", "")).lower()
    return decoder_name in FEATURE_PYRAMID_CACHED_DECODER_NAMES


class CachedFeatureSegmentation(nn.Module):
    """Train a decoder/head from cached Galileo feature maps."""

    def __init__(
        self,
        decoder: nn.Module,
        temporal_phenology_prior: PhenologyPriorAdapter | None = None,
        prior_token_encoder: PriorTokenEncoder | None = None,
        pre_decoder_prior_injection: TemporalFeaturePyramidPriorInjection | None = None,
    ) -> None:
        super().__init__()
        if (prior_token_encoder is None) != (pre_decoder_prior_injection is None):
            raise ValueError(
                "prior_token_encoder and pre_decoder_prior_injection must be enabled together."
            )
        if temporal_phenology_prior is not None and prior_token_encoder is not None:
            raise ValueError("Legacy phenology and CA-HPI cannot be enabled together.")
        self.decoder = decoder
        self.temporal_phenology_prior = temporal_phenology_prior
        self.prior_token_encoder = prior_token_encoder
        self.pre_decoder_prior_injection = pre_decoder_prior_injection

    def forward(self, batch: dict) -> torch.Tensor:
        device = next(self.decoder.parameters()).device
        target = batch.get("target")
        if target is None:
            raise ValueError("CachedFeatureSegmentation needs batch['target'] for output size.")
        target_size = tuple(int(value) for value in target.shape[-2:])

        if getattr(self.decoder, "expects_temporal_feature_pyramid", False):
            temporal_features = batch.get("temporal_features_by_layer")
            months = batch.get("months")
            if temporal_features is None or months is None:
                raise ValueError(
                    "This decoder needs a temporal_v2 cache with "
                    "temporal_features_by_layer and months."
                )
            months = months.to(device, non_blocking=True)
            layer_indices = getattr(
                self.decoder,
                "temporal_layer_indices",
                tuple(range(temporal_features.shape[1])),
            )
            features = tuple(
                temporal_features[:, layer_index].to(device, non_blocking=True)
                for layer_index in layer_indices
            )
            features = inject_temporal_phenology_prior(
                features,
                months,
                self.temporal_phenology_prior,
            )
            if self.prior_token_encoder is not None:
                prior = self.prior_token_encoder(
                    batch_size=features[0].shape[0],
                    batch=batch,
                )
                features = self.pre_decoder_prior_injection(
                    features,
                    prior,
                    layer_indices=tuple(int(index) for index in layer_indices),
                )
            return self.decoder(features, months=months, target_size=target_size)

        if getattr(self.decoder, "expects_feature_pyramid", False):
            features_by_layer = batch.get("features_by_layer")
            if features_by_layer is None:
                raise ValueError("This cached decoder needs caches produced with encoder.hidden_layers.")
            features_by_layer = features_by_layer.to(device, non_blocking=True)
            features = tuple(
                features_by_layer[:, layer_index]
                for layer_index in range(features_by_layer.shape[1])
            )
            return self.decoder(features, target_size=target_size)

        features = batch["features"].to(device, non_blocking=True)
        return self.decoder(features, target_size=target_size)


def build_cached_feature_model(
    config: dict,
    in_channels: int,
    num_layers: int | None = None,
) -> CachedFeatureSegmentation:
    data_cfg = config["data"]
    encoder_cfg = config.get("encoder", {})
    model_cfg = config["model"]
    decoder_name = str(model_cfg.get("decoder", "single_layer_dpt")).lower()
    spatial_decoder_name = TEMPORAL_READOUT_DECODER_BASES.get(
        decoder_name,
        decoder_name,
    )
    phenology_enabled = bool((config.get("phenology", {}) or {}).get("enabled", False))
    heterogeneous_prior_enabled = prior_injection_enabled(config)
    if phenology_enabled and heterogeneous_prior_enabled:
        raise ValueError("Legacy phenology and prior_injection are mutually exclusive.")
    if phenology_enabled and decoder_name not in TEMPORAL_CACHED_DECODER_NAMES:
        raise ValueError(
            "Phenology prior injection requires a decoder that preserves the temporal dimension."
        )
    if heterogeneous_prior_enabled and decoder_name not in TEMPORAL_CACHED_DECODER_NAMES:
        raise ValueError(
            "CA-HPI requires a decoder that consumes temporal feature pyramids."
        )
    if spatial_decoder_name in {"linear_probe", "linear", "lp"}:
        decoder = GalileoLinearProbeDecoder(
            in_channels=int(in_channels),
            num_classes=int(data_cfg["num_classes"]),
            output_patch_size=int(model_cfg.get("output_patch_size", 4)),
        )
    elif spatial_decoder_name in {"single_layer_dpt", "single", "dpt"}:
        decoder = DPTSingleLayerDecoder(
            in_channels=int(in_channels),
            num_classes=int(data_cfg["num_classes"]),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            decoder_blocks=int(model_cfg.get("decoder_blocks", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif spatial_decoder_name in {"multi_layer_dpt", "multilayer_dpt"}:
        hidden_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        decoder = DPTMultiLayerDecoder(
            in_channels=int(in_channels),
            num_classes=int(data_cfg["num_classes"]),
            num_layers=int(num_layers or len(hidden_layers)),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            decoder_blocks=int(model_cfg.get("decoder_blocks", 3)),
            fusion_blocks=int(model_cfg.get("fusion_blocks", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif spatial_decoder_name in {
        "galileo_dpt",
        "galileo_adapted_dpt",
        "multiscale_dpt",
    }:
        hidden_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        decoder = GalileoDPTDecoder(
            in_channels=int(in_channels),
            num_classes=int(data_cfg["num_classes"]),
            num_layers=int(num_layers or len(hidden_layers)),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            fusion_blocks=int(model_cfg.get("fusion_blocks", 2)),
            head_channels=int(model_cfg.get("head_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            preserve_native_deep_skip=bool(
                model_cfg.get("preserve_native_deep_skip", True)
            ),
        )
    elif spatial_decoder_name in {"upernet", "upernet_style", "upernet-style"}:
        hidden_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        decoder = UPerNetDecoder(
            in_channels=int(in_channels),
            num_classes=int(data_cfg["num_classes"]),
            num_layers=int(num_layers or len(hidden_layers)),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            ppm_channels=int(model_cfg.get("ppm_channels", 64)),
            ppm_scales=tuple(model_cfg.get("ppm_scales", (1, 2, 3, 6))),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif decoder_name in THREE_D_CACHED_DECODER_NAMES:
        hidden_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        decoder = ThreeDAwareDPTDecoder(
            in_channels=int(in_channels),
            num_classes=int(data_cfg["num_classes"]),
            num_layers=int(num_layers or len(hidden_layers)),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            spatial_window=int(model_cfg.get("spatial_window", 8)),
            global_3d_blocks=int(model_cfg.get("global_3d_blocks", 4)),
            fusion_blocks_per_stage=int(model_cfg.get("fusion_blocks_per_stage", 2)),
            mlp_expansion=int(model_cfg.get("mlp_expansion", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            drop_path=float(model_cfg.get("drop_path", 0.1)),
            temporal_pool_heads=int(model_cfg.get("temporal_pool_heads", 8)),
            num_months=int(model_cfg.get("num_months", 12)),
            preserve_native_deep_skip=bool(
                model_cfg.get("preserve_native_deep_skip", True)
            ),
        )
    else:
        raise ValueError(f"Unsupported decoder: {decoder_name}")

    if decoder_name in TEMPORAL_READOUT_DECODER_BASES:
        readout_cfg = model_cfg.get("temporal_readout", {})
        configured_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        temporal_num_layers = int(num_layers or len(configured_layers))
        decoder = TemporalReadoutDecoder(
            spatial_decoder=decoder,
            in_channels=int(in_channels),
            num_layers=temporal_num_layers,
            num_months=int(readout_cfg.get("num_months", 12)),
            hidden_channels=readout_cfg.get("hidden_channels"),
            dropout=float(readout_cfg.get("dropout", 0.0)),
        )

    configured_num_layers = int(
        num_layers or len(tuple(encoder_cfg.get("hidden_layers") or ()))
    )
    return CachedFeatureSegmentation(
        decoder=decoder,
        temporal_phenology_prior=build_phenology_prior(
            config,
            feature_channels=int(in_channels),
        ),
        prior_token_encoder=build_prior_token_encoder(config),
        pre_decoder_prior_injection=build_temporal_prior_injection(
            config,
            feature_channels=int(in_channels),
            num_layers=configured_num_layers,
        ),
    )
