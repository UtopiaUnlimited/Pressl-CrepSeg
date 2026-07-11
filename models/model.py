from __future__ import annotations

import torch
from torch import nn

from models.decoders import DPTMultiLayerDecoder, DPTSingleLayerDecoder, UPerNetDecoder
from models.encoders import GalileoHFEncoder


class GalileoDPTSegmentation(nn.Module):
    def __init__(self, encoder: GalileoHFEncoder, decoder: nn.Module) -> None:
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
        if getattr(self.decoder, "expects_feature_pyramid", False):
            if not encoded.features_by_layer:
                raise ValueError("This decoder needs encoder.hidden_layers to be configured.")
            return self.decoder(encoded.features_by_layer, target_size=target_size)
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
        hidden_layers=encoder_cfg.get("hidden_layers"),
        hidden_size=encoder_cfg.get("hidden_size"),
    )
    hidden_size = encoder.hidden_size or encoder_cfg.get("hidden_size")
    if hidden_size is None:
        raise ValueError("Could not infer Galileo hidden size. Set encoder.hidden_size in the config.")

    decoder_name = str(model_cfg.get("decoder", "single_layer_dpt")).lower()
    if decoder_name in {"single_layer_dpt", "single", "dpt"}:
        decoder = DPTSingleLayerDecoder(
            in_channels=int(hidden_size),
            num_classes=int(data_cfg["num_classes"]),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            decoder_blocks=int(model_cfg.get("decoder_blocks", 3)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif decoder_name in {"multiscale_dpt", "multi_layer_dpt", "multilayer_dpt"}:
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
    else:
        raise ValueError(f"Unsupported decoder: {decoder_name}")
    return GalileoDPTSegmentation(encoder=encoder, decoder=decoder)
