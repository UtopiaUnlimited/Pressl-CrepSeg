from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_phenology_overlay(
    config: dict[str, Any],
    overlay_path: str | Path | None,
) -> dict[str, Any]:
    """Attach one reusable phenology specification to a decoder config.

    The base config owns the decoder and training protocol. The overlay owns
    only the prior table and a run-name suffix, so the same prior can be used
    with any decoder that consumes time-preserving Galileo features.
    """

    if overlay_path is None:
        return config

    source = Path(overlay_path)
    overlay = load_config(source)
    if not isinstance(overlay, dict):
        raise ValueError(f"Phenology overlay must be a mapping: {source}")
    unexpected = set(overlay) - {"phenology", "run_suffix"}
    if unexpected:
        raise ValueError(
            "Phenology overlay may only define 'phenology' and 'run_suffix', "
            f"got {sorted(unexpected)} in {source}"
        )
    phenology = overlay.get("phenology")
    if not isinstance(phenology, dict):
        raise ValueError(f"Phenology overlay needs a 'phenology' mapping: {source}")

    merged = copy.deepcopy(config)
    merged["phenology"] = copy.deepcopy(phenology)

    suffix = str(overlay.get("run_suffix") or source.stem)
    train_cfg = merged.get("train")
    if not isinstance(train_cfg, dict):
        raise ValueError("Base config needs a train mapping for a phenology overlay.")
    for key in ("log_dir", "checkpoint_dir"):
        value = train_cfg.get(key)
        if value:
            train_cfg[key] = f"{value}_phenology_{suffix}"
    return merged


def merge_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    if getattr(args, "batch_size", None) is not None:
        config["data"]["batch_size"] = args.batch_size
    if getattr(args, "epochs", None) is not None:
        config["train"]["epochs"] = args.epochs
    if getattr(args, "data_root", None) is not None:
        config["data"]["root"] = args.data_root
    if getattr(args, "encoder_checkpoint", None) is not None:
        config["encoder"]["checkpoint"] = args.encoder_checkpoint
    if getattr(args, "no_amp", False):
        config["train"]["amp"] = False
    return config
