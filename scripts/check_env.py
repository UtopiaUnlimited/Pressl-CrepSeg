from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import PASTISDataset  # noqa: E402
from models import build_model  # noqa: E402
from utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_dpt.yaml")
    parser.add_argument("--try-model", action="store_true")
    return parser.parse_args()


def check_gitignore() -> None:
    gitignore = ROOT / ".gitignore"
    required = ["data/PASTIS/", "pretrained/", "logs/", "checkpoints/", "__pycache__/", "*.pyc"]
    if not gitignore.exists():
        print("gitignore=missing")
        return
    text = gitignore.read_text(encoding="utf-8")
    missing = [item for item in required if item not in text]
    print("gitignore=ok" if not missing else f"gitignore=missing {missing}")


def check_dataset(config: dict) -> None:
    data_cfg = config["data"]
    dataset = PASTISDataset(
        root=data_cfg["root"],
        folds=data_cfg["train_folds"],
        selected_timesteps=data_cfg["selected_timesteps"],
        target_channel=data_cfg.get("target_channel", 0),
    )
    sample = dataset[0]
    target = sample["target"].numpy()
    print(f"dataset_train_samples={len(dataset)}")
    print(f"s2_shape={tuple(sample['s2'].shape)} dtype={sample['s2'].dtype}")
    print(f"months_minmax={int(sample['months'].min())},{int(sample['months'].max())}")
    print(f"target_shape={tuple(sample['target'].shape)} labels={int(target.min())}..{int(target.max())}")
    if int(target.max()) >= int(data_cfg["num_classes"]):
        raise ValueError("Semantic target labels exceed configured num_classes.")


def check_pretrained(config: dict) -> None:
    checkpoint = ROOT / config["encoder"]["checkpoint"]
    expected = ["config.json", "modeling_galileo.py", "processing_galileo.py"]
    existing = [name for name in expected if (checkpoint / name).exists()]
    weights = list(checkpoint.glob("*.safetensors")) + list(checkpoint.glob("*.bin"))
    print(f"encoder_checkpoint={checkpoint}")
    print(f"encoder_files={existing}")
    print(f"encoder_weights={len(weights)}")
    if not checkpoint.exists():
        print("encoder_status=missing_directory")
    elif len(existing) < len(expected) or not weights:
        print("encoder_status=incomplete")
    else:
        print("encoder_status=ok")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"numpy={np.__version__}")
    check_gitignore()
    check_dataset(config)
    check_pretrained(config)
    if args.try_model:
        model = build_model(config)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        print(f"model_trainable={trainable}")
        print(f"model_frozen={frozen}")


if __name__ == "__main__":
    main()
