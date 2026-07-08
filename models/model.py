from __future__ import annotations

import torch
from torch import nn

from models.decoders import DPTSingleLayerDecoder
from models.encoders import GalileoHFEncoder


class GalileoDPTSegmentation(nn.Module):
    def __init__(self, encoder: GalileoHFEncoder, decoder: DPTSingleLayerDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch: dict) -> torch.Tensor:
        encoded = self.encoder(batch["samples"])
        target = batch.get("target")
        if target is not None:
            target_size = tuple(int(value) for value in target.shape[-2:])
        else:
            target_size = batch["samples"][0]["image_size"]
        return self.decoder(encoded.features, target_size=target_size)


def build_model(config: dict) -> GalileoDPTSegmentation:
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]
    model_cfg = config["model"]

    encoder = GalileoHFEncoder(
        checkpoint=encoder_cfg["checkpoint"],
        patch_size=encoder_cfg.get("patch_size", 8),
        freeze=encoder_cfg.get("freeze", True),
        normalize=encoder_cfg.get("normalize", True),
        local_files_only=encoder_cfg.get("local_files_only", True),
        output_hidden_states=encoder_cfg.get("output_hidden_states", False),
        spatial_token_strategy=encoder_cfg.get("spatial_token_strategy", "auto"),
        hidden_size=encoder_cfg.get("hidden_size"),
    )
    hidden_size = encoder.hidden_size or encoder_cfg.get("hidden_size")
    if hidden_size is None:
        raise ValueError("Could not infer Galileo hidden size. Set encoder.hidden_size in the config.")

    decoder = DPTSingleLayerDecoder(
        in_channels=int(hidden_size),
        num_classes=int(data_cfg["num_classes"]),
        decoder_channels=int(model_cfg.get("decoder_channels", 256)),
        decoder_blocks=int(model_cfg.get("decoder_blocks", 3)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    return GalileoDPTSegmentation(encoder=encoder, decoder=decoder)
