from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_pastis_dataset, pastis_collate_fn  # noqa: E402
from losses import build_loss  # noqa: E402
from metrics import ConfusionMatrix, macro_f1, mean_iou, pixel_accuracy  # noqa: E402
from models import build_model  # noqa: E402
from scripts.visualize_predictions import make_rgb_composite, render_triptych  # noqa: E402
from utils import apply_phenology_overlay, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/galileo_dpt.yaml")
    parser.add_argument("--phenology-config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output"),
        help="Directory for every RGB / ground-truth / prediction triptych.",
    )
    parser.add_argument("--panel-size", type=int, default=384)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = apply_phenology_overlay(load_config(args.config), args.phenology_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_cfg = config["data"]
    dataset = build_pastis_dataset(data_cfg, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=pastis_collate_fn,
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(
        checkpoint["model"],
        strict=not bool(checkpoint.get("trainable_only", False)),
    )
    model.eval()

    criterion = build_loss(config).to(device)
    confusion = ConfusionMatrix(
        num_classes=int(data_cfg["num_classes"]),
        ignore_index=config.get("loss", {}).get("ignore_index"),
    )
    total_loss = 0.0
    batch_count = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for batch in tqdm(loader, desc=args.split):
        target = batch["target"].to(device)
        batch["target"] = target
        logits = model(batch)
        total_loss += float(criterion(logits, target).item())
        batch_count += 1
        confusion.update(logits, target)

        predictions = logits.argmax(dim=1).detach().cpu().numpy()
        targets = target.detach().cpu().numpy()
        for sample_index, sample in enumerate(batch["samples"]):
            sample_id = str(sample["sample_id"])
            comparison = render_triptych(
                rgb=make_rgb_composite(sample, config),
                target=targets[sample_index],
                prediction=predictions[sample_index],
                sample_id=sample_id,
                fold=int(sample["fold"]),
                panel_size=args.panel_size,
            )
            comparison.save(output_dir / f"{args.split}_{sample_id}.png")
            saved_count += 1

    miou, per_class_iou = mean_iou(confusion.matrix)
    accuracy = pixel_accuracy(confusion.matrix)
    f1, per_class_f1 = macro_f1(confusion.matrix)
    print(f"{args.split}_loss={total_loss / max(1, batch_count):.5f}")
    print(f"{args.split}_miou={miou:.5f}")
    print(f"{args.split}_acc={accuracy:.5f}")
    print(f"{args.split}_f1={f1:.5f}")
    print(
        "per_class_iou="
        + ",".join(f"{value:.5f}" for value in per_class_iou.tolist())
    )
    print(
        "per_class_f1="
        + ",".join(f"{value:.5f}" for value in per_class_f1.tolist())
    )
    print(f"saved_predictions={saved_count} output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
