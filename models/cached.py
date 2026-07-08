from __future__ import annotations

import torch
from torch import nn

from models.decoders import DPTSingleLayerDecoder


class CachedFeatureSegmentation(nn.Module):
    """Train the decoder/head from cached Galileo spatial feature maps."""

    def __init__(self, decoder: DPTSingleLayerDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, batch: dict) -> torch.Tensor:
        device = next(self.decoder.parameters()).device
        features = batch["features"].to(device, non_blocking=True)
        target = batch.get("target")
        if target is None:
            raise ValueError("CachedFeatureSegmentation needs batch['target'] for output size.")
        target_size = tuple(int(value) for value in target.shape[-2:])
        return self.decoder(features, target_size=target_size)


def build_cached_feature_model(config: dict, in_channels: int) -> CachedFeatureSegmentation:
    data_cfg = config["data"]
    model_cfg = config["model"]
    decoder = DPTSingleLayerDecoder(
        in_channels=int(in_channels),
        num_classes=int(data_cfg["num_classes"]),
        decoder_channels=int(model_cfg.get("decoder_channels", 256)),
        decoder_blocks=int(model_cfg.get("decoder_blocks", 3)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    return CachedFeatureSegmentation(decoder=decoder)
