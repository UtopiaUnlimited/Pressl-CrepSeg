from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
