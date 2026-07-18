from __future__ import annotations

import torch
from torch import nn

from models.decoders import (
    DPTMultiLayerDecoder,
    DPTSingleLayerDecoder,
    GalileoDPTDecoder,
    ThreeDAwareDPTDecoder,
    UPerNetDecoder,
)
from models.encoders import GalileoHFEncoder
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


class GalileoDPTSegmentation(nn.Module):
    def __init__(
        self,
        encoder: GalileoHFEncoder,
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
        self.encoder = encoder
        self.decoder = decoder
        self.temporal_phenology_prior = temporal_phenology_prior
        self.prior_token_encoder = prior_token_encoder
        self.pre_decoder_prior_injection = pre_decoder_prior_injection

    def forward(self, batch: dict) -> torch.Tensor:
        encoded = self.encoder(batch["samples"])
        target = batch.get("target")
        if target is not None:
            target_size = tuple(int(value) for value in target.shape[-2:])
        else:
            target_size = batch["samples"][0]["image_size"]
        if getattr(self.decoder, "expects_temporal_feature_pyramid", False):
            if not encoded.temporal_features_by_layer:
                raise ValueError("This decoder needs temporal Galileo hidden-layer features.")
            months = torch.stack([sample["months"] for sample in batch["samples"]], dim=0)
            months = months.to(encoded.temporal_features_by_layer[0].device, non_blocking=True)
            features = inject_temporal_phenology_prior(
                encoded.temporal_features_by_layer,
                months,
                self.temporal_phenology_prior,
            )
            if self.prior_token_encoder is not None:
                prior = self.prior_token_encoder(
                    batch_size=features[0].shape[0],
                    batch=batch,
                )
                features = self.pre_decoder_prior_injection(features, prior)
            return self.decoder(
                features,
                months=months,
                target_size=target_size,
            )
        if getattr(self.decoder, "expects_feature_pyramid", False):
            if not encoded.features_by_layer:
                raise ValueError("This decoder needs encoder.hidden_layers to be configured.")
            return self.decoder(encoded.features_by_layer, target_size=target_size)
        return self.decoder(encoded.features, target_size=target_size)


def build_model(config: dict) -> GalileoDPTSegmentation:
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]
    model_cfg = config["model"]
    decoder_name = str(model_cfg.get("decoder", "single_layer_dpt")).lower()
    temporal_decoder = decoder_name in {
        "3d_aware_dpt",
        "3d-aware-dpt",
        "three_d_aware_dpt",
    }
    phenology_enabled = bool((config.get("phenology", {}) or {}).get("enabled", False))
    heterogeneous_prior_enabled = prior_injection_enabled(config)
    if phenology_enabled and heterogeneous_prior_enabled:
        raise ValueError("Legacy phenology and prior_injection are mutually exclusive.")
    if phenology_enabled and not temporal_decoder:
        raise ValueError(
            "Phenology prior injection requires a decoder that preserves the temporal dimension."
        )
    if heterogeneous_prior_enabled and not temporal_decoder:
        raise ValueError(
            "CA-HPI requires a decoder that consumes temporal feature pyramids."
        )
    if temporal_decoder and not bool(encoder_cfg.get("freeze", True)):
        raise ValueError("3D-Aware DPT experiments require the Galileo encoder to stay frozen.")

    encoder = GalileoHFEncoder(
        checkpoint=encoder_cfg["checkpoint"],
        patch_size=encoder_cfg.get("patch_size", 8),
        freeze=encoder_cfg.get("freeze", True),
        normalize=encoder_cfg.get("normalize", True),
        local_files_only=encoder_cfg.get("local_files_only", True),
        output_hidden_states=encoder_cfg.get("output_hidden_states", False),
        spatial_token_strategy=encoder_cfg.get("spatial_token_strategy", "auto"),
        hidden_layers=encoder_cfg.get("hidden_layers"),
        hidden_size=encoder_cfg.get("hidden_size"),
        preserve_temporal_features=bool(
            encoder_cfg.get("preserve_temporal_features", temporal_decoder)
        ),
    )
    hidden_size = encoder.hidden_size or encoder_cfg.get("hidden_size")
    if hidden_size is None:
        raise ValueError("Could not infer Galileo hidden size. Set encoder.hidden_size in the config.")

    if decoder_name in {"single_layer_dpt", "single", "dpt"}:
        decoder = DPTSingleLayerDecoder(
            in_channels=int(hidden_size),
            num_classes=int(data_cfg["num_classes"]),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            decoder_blocks=int(model_cfg.get("decoder_blocks", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif decoder_name in {"multi_layer_dpt", "multilayer_dpt"}:
        hidden_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        decoder = DPTMultiLayerDecoder(
            in_channels=int(hidden_size),
            num_classes=int(data_cfg["num_classes"]),
            num_layers=len(hidden_layers),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            decoder_blocks=int(model_cfg.get("decoder_blocks", 3)),
            fusion_blocks=int(model_cfg.get("fusion_blocks", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif decoder_name in {
        "galileo_dpt",
        "galileo_adapted_dpt",
        "multiscale_dpt",
    }:
        hidden_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        decoder = GalileoDPTDecoder(
            in_channels=int(hidden_size),
            num_classes=int(data_cfg["num_classes"]),
            num_layers=len(hidden_layers),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            fusion_blocks=int(model_cfg.get("fusion_blocks", 2)),
            head_channels=int(model_cfg.get("head_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            preserve_native_deep_skip=bool(
                model_cfg.get("preserve_native_deep_skip", True)
            ),
        )
    elif decoder_name in {"upernet", "upernet_style", "upernet-style"}:
        hidden_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        decoder = UPerNetDecoder(
            in_channels=int(hidden_size),
            num_classes=int(data_cfg["num_classes"]),
            num_layers=len(hidden_layers),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            ppm_channels=int(model_cfg.get("ppm_channels", 64)),
            ppm_scales=tuple(model_cfg.get("ppm_scales", (1, 2, 3, 6))),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif temporal_decoder:
        hidden_layers = tuple(encoder_cfg.get("hidden_layers") or ())
        decoder = ThreeDAwareDPTDecoder(
            in_channels=int(hidden_size),
            num_classes=int(data_cfg["num_classes"]),
            num_layers=len(hidden_layers),
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
    return GalileoDPTSegmentation(
        encoder=encoder,
        decoder=decoder,
        temporal_phenology_prior=build_phenology_prior(
            config,
            feature_channels=int(hidden_size),
        ),
        prior_token_encoder=build_prior_token_encoder(config),
        pre_decoder_prior_injection=build_temporal_prior_injection(
            config,
            feature_channels=int(hidden_size),
            num_layers=len(tuple(encoder_cfg.get("hidden_layers") or ())),
        ),
    )
