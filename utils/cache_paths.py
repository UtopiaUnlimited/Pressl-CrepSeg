from __future__ import annotations


def feature_cache_dir(config: dict, split: str) -> str:
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]
    hidden_layers = encoder_cfg.get("hidden_layers") or []

    temporal_aggregation = str(data_cfg.get("temporal_aggregation", "uniform")).lower()
    timesteps = data_cfg.get("selected_timesteps")
    if temporal_aggregation == "monthly":
        temporal_suffix = f"monthly{timesteps}"
    else:
        temporal_suffix = f"t{timesteps}"

    tile_size = data_cfg.get("tile_size")
    tile_suffix = f"_tile{tile_size}" if tile_size else ""
    layer_suffix = ""
    if hidden_layers:
        layer_suffix = "_hl" + "-".join(str(layer) for layer in hidden_layers)

    return (
        f"data/cache/{encoder_cfg['name']}/"
        f"{temporal_suffix}{tile_suffix}_patch{encoder_cfg['patch_size']}"
        f"{layer_suffix}_{split}"
    )
