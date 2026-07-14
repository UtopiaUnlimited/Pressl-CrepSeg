from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CachedFeatureDataset, cached_feature_collate_fn  # noqa: E402
from losses import build_loss  # noqa: E402
from metrics import ConfusionMatrix, macro_f1, mean_iou, pixel_accuracy  # noqa: E402
from models import build_cached_feature_model  # noqa: E402
from train import Trainer, build_optimizer, build_scheduler  # noqa: E402
from utils import feature_cache_dir, load_config, seed_everything  # noqa: E402


# Paper Appendix C.1: {1, 3, 4, 5} x 10^{-4, -3, -2, -1}.
PAPER_LEARNING_RATES = tuple(
    multiplier * 10.0**exponent
    for exponent in (-4, -3, -2, -1)
    for multiplier in (1, 3, 4, 5)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_linear_probe.yaml")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=PAPER_LEARNING_RATES)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/linear_probe_sweep/results.json",
    )
    return parser.parse_args()


def build_loader(
    cache_dir: str,
    config: dict,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = CachedFeatureDataset(cache_dir, load_features_by_layer=False)
    return DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 128)),
        shuffle=shuffle,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=cached_feature_collate_fn,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def lr_slug(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("+", "").replace("-", "m")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int | None,
    amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None,
) -> tuple[float, float, float, float]:
    model.eval()
    confusion = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    batch_count = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="linear probe test"), start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        target = batch["target"].to(device, non_blocking=True)
        batch["target"] = target
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(batch)
            loss = criterion(logits, target)
        total_loss += float(loss.item())
        batch_count += 1
        confusion.update(logits, target)
    miou, _ = mean_iou(confusion.matrix)
    accuracy = pixel_accuracy(confusion.matrix)
    f1, _ = macro_f1(confusion.matrix)
    return total_loss / max(1, batch_count), miou, accuracy, f1


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    if args.epochs is not None:
        base_config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        base_config["data"]["batch_size"] = args.batch_size
    if str(base_config.get("model", {}).get("decoder", "")).lower() != "linear_probe":
        raise ValueError("The sweep config must set model.decoder=linear_probe.")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_cache = feature_cache_dir(base_config, "train")
    val_cache = feature_cache_dir(base_config, "val")
    test_cache = feature_cache_dir(base_config, "test")
    output_path = Path(args.output)
    base_seed = int(base_config.get("seed", 42))
    run_results: list[dict] = []

    for run_id in range(args.runs):
        seed = base_seed + run_id
        candidates: list[dict] = []
        for learning_rate in args.learning_rates:
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            config["optimizer"]["lr"] = float(learning_rate)
            config["train"]["save_best"] = False
            config["train"]["save_last"] = True
            config["train"]["early_stopping"] = {"enabled": False}
            slug = lr_slug(float(learning_rate))
            config["train"]["log_dir"] = f"logs/galileo_linear_probe_sweep/run_{run_id:02d}/lr_{slug}"
            config["train"]["checkpoint_dir"] = (
                f"checkpoints/galileo_linear_probe_sweep/run_{run_id:02d}/lr_{slug}"
            )

            seed_everything(seed)
            train_loader = build_loader(train_cache, config, shuffle=True, seed=seed)
            val_loader = build_loader(val_cache, config, shuffle=False, seed=seed)
            first_batch = next(iter(train_loader))
            in_channels = int(first_batch["features"].shape[1])
            model = build_cached_feature_model(config, in_channels=in_channels)
            criterion = build_loss(config).to(device)
            optimizer = build_optimizer(config, model)
            accumulation = int(config["train"].get("gradient_accumulation_steps", 1))
            scheduler = build_scheduler(
                config,
                optimizer,
                steps_per_epoch=math.ceil(len(train_loader) / accumulation),
            )
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_classes=int(config["data"]["num_classes"]),
                amp=bool(config["train"].get("amp", False)),
                amp_dtype=str(config["train"].get("amp_dtype", "float16")),
                log_dir=config["train"]["log_dir"],
                checkpoint_dir=config["train"]["checkpoint_dir"],
                ignore_index=config.get("loss", {}).get("ignore_index"),
                save_best=False,
                save_last=True,
            )
            summary = trainer.fit(
                epochs=int(config["train"]["epochs"]),
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
            )
            candidate = {
                "learning_rate": float(learning_rate),
                "val_loss": float(summary["last_val_loss"]),
                "val_miou": float(summary["last_val_miou"]),
                "val_acc": float(summary["last_val_acc"]),
                "val_f1": float(summary["last_val_f1"]),
                "checkpoint": str(Path(config["train"]["checkpoint_dir"]) / "last.pt"),
            }
            candidates.append(candidate)
            print(f"run={run_id} candidate={candidate}")

        selected = max(candidates, key=lambda item: item["val_miou"])
        selected_config = copy.deepcopy(base_config)
        test_loader = build_loader(test_cache, selected_config, shuffle=False, seed=seed)
        first_batch = next(iter(test_loader))
        model = build_cached_feature_model(
            selected_config,
            in_channels=int(first_batch["features"].shape[1]),
        ).to(device)
        checkpoint = torch.load(selected["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        criterion = build_loss(selected_config).to(device)
        amp_dtype = (
            torch.bfloat16
            if str(selected_config["train"].get("amp_dtype", "float16")) == "bfloat16"
            else torch.float16
        )
        test_loss, test_miou, test_acc, test_f1 = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=int(selected_config["data"]["num_classes"]),
            ignore_index=selected_config.get("loss", {}).get("ignore_index"),
            amp=bool(selected_config["train"].get("amp", False)),
            amp_dtype=amp_dtype,
            max_batches=args.max_test_batches,
        )
        run_result = {
            "run_id": run_id,
            "seed": seed,
            "selected_learning_rate": selected["learning_rate"],
            "val_miou": selected["val_miou"],
            "test_loss": test_loss,
            "test_miou": test_miou,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "checkpoint": selected["checkpoint"],
            "candidates": candidates,
        }
        run_results.append(run_result)
        write_results(output_path, {"runs": run_results})
        print(f"selected run={run_id}: {run_result}")

    test_scores = [float(result["test_miou"]) for result in run_results]
    test_acc_scores = [float(result["test_acc"]) for result in run_results]
    test_f1_scores = [float(result["test_f1"]) for result in run_results]
    payload = {
        "protocol": {
            "epochs": int(base_config["train"]["epochs"]),
            "learning_rates": [float(value) for value in args.learning_rates],
            "selection": "highest final fold4 val_mIoU per run",
            "test": "fold5, evaluated once after LR selection",
        },
        "test_miou_mean": statistics.mean(test_scores),
        "test_miou_population_std": statistics.pstdev(test_scores),
        "test_acc_mean": statistics.mean(test_acc_scores),
        "test_acc_population_std": statistics.pstdev(test_acc_scores),
        "test_f1_mean": statistics.mean(test_f1_scores),
        "test_f1_population_std": statistics.pstdev(test_f1_scores),
        "runs": run_results,
    }
    write_results(output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
