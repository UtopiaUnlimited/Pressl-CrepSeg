from __future__ import annotations

import torch
from torch import nn

from models.decoders import DPTMultiLayerDecoder, DPTSingleLayerDecoder, UPerNetDecoder


class CachedFeatureSegmentation(nn.Module):
    """Train the decoder/head from cached Galileo spatial feature maps."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, batch: dict) -> torch.Tensor:
        device = next(self.decoder.parameters()).device
        target = batch.get("target")
        if target is None:
            raise ValueError("CachedFeatureSegmentation needs batch['target'] for output size.")
        target_size = tuple(int(value) for value in target.shape[-2:])

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
    if decoder_name in {"single_layer_dpt", "single", "dpt"}:
        decoder = DPTSingleLayerDecoder(
            in_channels=int(in_channels),
            num_classes=int(data_cfg["num_classes"]),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            decoder_blocks=int(model_cfg.get("decoder_blocks", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif decoder_name in {"multiscale_dpt", "multi_layer_dpt", "multilayer_dpt"}:
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
    elif decoder_name in {"upernet", "upernet_style", "upernet-style"}:
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
    else:
        raise ValueError(f"Unsupported decoder: {decoder_name}")
    return CachedFeatureSegmentation(decoder=decoder)
