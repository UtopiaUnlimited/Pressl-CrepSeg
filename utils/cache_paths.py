from __future__ import annotations


def apply_cache_overrides(
    config: dict,
    cache_format: str | None = None,
    temporal_dtype: str | None = None,
) -> dict:
    cache_cfg = dict(config.get("cache", {}))
    if cache_format is not None:
        cache_cfg["format"] = cache_format
    if temporal_dtype is not None:
        cache_cfg["temporal_dtype"] = temporal_dtype
    config["cache"] = cache_cfg
    return config


def feature_cache_dir(config: dict, split: str) -> str:
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]
    cache_cfg = config.get("cache", {})
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

    cache_suffix = ""
    cache_format = str(cache_cfg.get("format", "spatial_v1")).lower()
    if cache_format == "temporal_v2":
        temporal_dtype = str(cache_cfg.get("temporal_dtype", "float16")).lower()
        dtype_aliases = {"float16": "fp16", "float32": "fp32"}
        if temporal_dtype not in dtype_aliases:
            raise ValueError("Temporal cache dtype must be float16 or float32.")
        cache_suffix = f"_temporal-v2_t{dtype_aliases[temporal_dtype]}"
    elif cache_format != "spatial_v1":
        raise ValueError(f"Unsupported cache format: {cache_format}")

    return (
        f"data/cache/{encoder_cfg['name']}/"
        f"{temporal_suffix}{tile_suffix}_patch{encoder_cfg['patch_size']}"
        f"{layer_suffix}{cache_suffix}_{split}"
    )
