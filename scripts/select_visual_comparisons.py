from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    CachedFeatureDataset,
    PASTISDataset,
    build_pastis_dataset,
    cached_feature_collate_fn,
)
from models import build_cached_feature_model  # noqa: E402
from scripts.visualize_predictions import (  # noqa: E402
    PASTIS_PALETTE,
    colorize_mask,
    make_rgb_composite,
    render_triptych,
    sample_metrics,
)
from utils import feature_cache_dir, load_config  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    config_path: Path
    checkpoint_path: Path


DEFAULT_MODELS = (
    ModelSpec(
        "scheme1",
        "最终层卷积",
        ROOT / "configs/galileo_single_layer_dpt_shared.yaml",
        ROOT
        / (
            "checkpoints/galileo_single_layer_dpt_shared_paper_input_bs16_rerun_"
            "seed42_cached/best_val_miou.pt"
        ),
    ),
    ModelSpec(
        "scheme2",
        "多层同尺度融合",
        ROOT / "configs/galileo_multi_layer_dpt_shared.yaml",
        ROOT
        / (
            "checkpoints/galileo_multi_layer_dpt_shared_paper_input_bs16_rerun_"
            "seed42_cached/best_val_miou.pt"
        ),
    ),
    ModelSpec(
        "scheme3",
        "UPerNet",
        ROOT / "configs/galileo_upernet_shared.yaml",
        ROOT
        / "checkpoints/galileo_upernet_shared_paper_input_bs16_cached/best_val_miou.pt",
    ),
    ModelSpec(
        "scheme4",
        "Adapted DPT",
        ROOT / "configs/galileo_adapted_dpt_shared.yaml",
        ROOT
        / (
            "checkpoints/galileo_adapted_dpt_native_skip_paper_input_bs16_seed42_"
            "cached/best_val_miou.pt"
        ),
    ),
)


PASTIS_CLASS_NAMES_ZH = (
    "非农业背景",
    "草地/牧草地",
    "软质冬小麦",
    "玉米",
    "冬大麦",
    "冬油菜",
    "春大麦",
    "向日葵",
    "葡萄园",
    "甜菜",
    "冬小黑麦",
    "冬硬粒小麦",
    "水果、蔬菜和花卉",
    "马铃薯",
    "豆科饲料作物",
    "大豆",
    "果园",
    "混合谷物",
    "高粱",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select moderately above-average samples and compare the four "
            "legacy mean(T) decoders."
        )
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--target-percentiles",
        nargs="+",
        type=float,
        default=[0.60, 0.68, 0.76],
        help="Desired combined percentile for each selected sample.",
    )
    parser.add_argument(
        "--max-percentile",
        type=float,
        default=0.82,
        help="Exclude samples above this combined metric percentile.",
    )
    parser.add_argument("--panel-size", type=int, default=260)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output/four_scheme_test_comparison"),
    )
    parser.add_argument("--scheme1-checkpoint", default=None)
    parser.add_argument("--scheme2-checkpoint", default=None)
    parser.add_argument("--scheme3-checkpoint", default=None)
    parser.add_argument("--scheme4-checkpoint", default=None)
    return parser.parse_args()


def model_specs(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    specs = []
    for index, default in enumerate(DEFAULT_MODELS, start=1):
        override = getattr(args, f"scheme{index}_checkpoint")
        checkpoint = Path(override).expanduser() if override else default.checkpoint_path
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        specs.append(
            ModelSpec(
                default.key,
                default.label,
                default.config_path,
                checkpoint.resolve(),
            )
        )
    return tuple(specs)


def build_loader(cache_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    dataset = CachedFeatureDataset(cache_dir, load_features_by_layer=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cached_feature_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def checkpoint_metadata(checkpoint: dict) -> dict[str, float | int | None]:
    return {
        key: checkpoint.get(key)
        for key in ("epoch", "val_loss", "val_miou", "val_acc", "val_f1")
    }


@torch.no_grad()
def infer_model(
    spec: ModelSpec,
    loader: DataLoader,
    first_batch: dict,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], dict]:
    config = load_config(spec.config_path)
    in_channels = int(first_batch["features_by_layer"].shape[2])
    num_layers = int(first_batch["features_by_layer"].shape[1])
    model = build_cached_feature_model(config, in_channels, num_layers)
    try:
        checkpoint = torch.load(
            spec.checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(spec.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for batch in tqdm(loader, desc=spec.label):
        target = batch["target"]
        logits = model(batch)
        batch_predictions = logits.argmax(dim=1).detach().cpu().numpy()
        batch_targets = target.numpy()
        for index, sample_id in enumerate(batch["sample_id"]):
            prediction = batch_predictions[index].astype(np.int16, copy=False)
            accuracy, miou, f1 = sample_metrics(batch_targets[index], prediction)
            predictions[sample_id] = prediction
            metrics[sample_id] = {"miou": miou, "f1": f1, "acc": accuracy}

    metadata = checkpoint_metadata(checkpoint)
    del model, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, metrics, metadata


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    finite_indices = np.flatnonzero(np.isfinite(values))
    ranks = np.full(len(values), np.nan, dtype=np.float64)
    if len(finite_indices) <= 1:
        ranks[finite_indices] = 0.5
        return ranks
    order = finite_indices[np.argsort(values[finite_indices], kind="stable")]
    ranks[order] = np.arange(len(order), dtype=np.float64) / (len(order) - 1)
    return ranks


def select_samples(
    sample_ids: list[str],
    model_metrics: dict[str, dict[str, dict[str, float]]],
    targets: dict[str, np.ndarray],
    count: int,
    target_percentiles: list[float],
    max_percentile: float,
) -> tuple[list[str], dict[str, dict], dict[str, float]]:
    metric_names = ("miou", "f1", "acc")
    model_keys = tuple(model_metrics)
    aggregate = np.asarray(
        [
            [
                np.mean(
                    [model_metrics[key][sample_id][metric] for key in model_keys]
                )
                for metric in metric_names
            ]
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )
    dataset_means = np.nanmean(aggregate, axis=0)
    metric_ranks = np.column_stack(
        [percentile_ranks(aggregate[:, index]) for index in range(len(metric_names))]
    )
    finite_rank_counts = np.isfinite(metric_ranks).sum(axis=1)
    combined_percentile = np.divide(
        np.nansum(metric_ranks, axis=1),
        finite_rank_counts,
        out=np.full(len(metric_ranks), np.nan, dtype=np.float64),
        where=finite_rank_counts > 0,
    )
    eligible = np.all(np.isfinite(aggregate), axis=1)
    eligible &= np.all(aggregate > dataset_means[None, :], axis=1)
    eligible &= combined_percentile <= max_percentile

    candidate_indices = np.flatnonzero(eligible).tolist()
    if len(candidate_indices) < count:
        raise RuntimeError(
            f"Only {len(candidate_indices)} samples satisfy the selection rule; "
            "increase --max-percentile."
        )

    desired = list(target_percentiles)
    while len(desired) < count:
        desired.append(desired[-1] if desired else 0.68)
    desired = desired[:count]

    selected_indices: list[int] = []
    selected_patches: set[str] = set()
    selected_class_sets: list[set[int]] = []
    for target_percentile in desired:
        scored = []
        for index in candidate_indices:
            if index in selected_indices:
                continue
            sample_id = sample_ids[index]
            patch_id = sample_id.split("_y", maxsplit=1)[0]
            classes = set(
                int(value)
                for value in np.unique(targets[sample_id])
                if 0 <= int(value) < 19
            )
            duplicate_patch_penalty = 1.0 if patch_id in selected_patches else 0.0
            similarity = max(
                (
                    len(classes & previous) / max(1, len(classes | previous))
                    for previous in selected_class_sets
                ),
                default=0.0,
            )
            richness_bonus = min(len(classes), 5) * 0.008
            score = (
                abs(combined_percentile[index] - target_percentile)
                + duplicate_patch_penalty
                + 0.08 * similarity
                - richness_bonus
            )
            scored.append((score, index, patch_id, classes))
        _, selected_index, patch_id, classes = min(scored, key=lambda item: item[0])
        selected_indices.append(selected_index)
        selected_patches.add(patch_id)
        selected_class_sets.append(classes)

    details = {}
    for index in selected_indices:
        sample_id = sample_ids[index]
        details[sample_id] = {
            "four_model_mean": {
                metric: float(aggregate[index, metric_index])
                for metric_index, metric in enumerate(metric_names)
            },
            "metric_percentiles": {
                metric: float(metric_ranks[index, metric_index])
                for metric_index, metric in enumerate(metric_names)
            },
            "combined_percentile": float(combined_percentile[index]),
            "target_classes": sorted(
                int(value)
                for value in np.unique(targets[sample_id])
                if 0 <= int(value) < 19
            ),
            "models": {
                key: model_metrics[key][sample_id] for key in model_keys
            },
        }
    means = {
        metric: float(dataset_means[index])
        for index, metric in enumerate(metric_names)
    }
    return [sample_ids[index] for index in selected_indices], details, means


def raw_index_by_id(dataset: PASTISDataset) -> dict[str, int]:
    mapping = {}
    for index in range(len(dataset)):
        record_index, tile_id = divmod(index, dataset.tiles_per_record)
        record = dataset.records[record_index]
        tile_row, tile_col = divmod(tile_id, dataset.tiles_per_side)
        sample_id = (
            f"{record.patch_id}_y{tile_row * dataset.tile_size}"
            f"_x{tile_col * dataset.tile_size}"
        )
        mapping[sample_id] = index
    return mapping


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "msyhbd.ttc" if bold else "msyh.ttc",
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
            if bold
            else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        ),
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_montage(
    selected_ids: list[str],
    raw_samples: dict[str, dict],
    predictions: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, dict[str, float]]],
    model_specs: tuple[ModelSpec, ...],
    config: dict,
    panel_size: int,
) -> Image.Image:
    labels = ("S2 RGB", "Ground truth") + tuple(spec.label for spec in model_specs)
    columns = len(labels)
    gap = 14
    margin = 24
    header_height = 58
    row_title_height = 58
    metric_height = 48
    row_height = row_title_height + panel_size + metric_height
    legend_columns = 5
    legend_rows = math.ceil(len(PASTIS_CLASS_NAMES_ZH) / legend_columns)
    legend_height = 54 + legend_rows * 34
    width = 2 * margin + columns * panel_size + (columns - 1) * gap
    height = header_height + len(selected_ids) * row_height + legend_height + margin
    canvas = Image.new("RGB", (width, height), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22, bold=True)
    row_font = _font(18, bold=True)
    metric_font = _font(15)

    for column, label in enumerate(labels):
        x = margin + column * (panel_size + gap)
        bounds = draw.textbbox((0, 0), label, font=title_font)
        draw.text(
            (x + (panel_size - (bounds[2] - bounds[0])) / 2, 18),
            label,
            fill=(28, 35, 48),
            font=title_font,
        )

    for row, sample_id in enumerate(selected_ids):
        sample = raw_samples[sample_id]
        target = sample["target"].cpu().numpy().astype(np.int64, copy=False)
        y0 = header_height + row * row_height
        draw.text((margin, y0 + 10), sample_id, fill=(28, 35, 48), font=row_font)
        image_y = y0 + row_title_height
        arrays = [make_rgb_composite(sample, config), colorize_mask(target)]
        arrays.extend(colorize_mask(predictions[spec.key][sample_id]) for spec in model_specs)
        for column, array in enumerate(arrays):
            x = margin + column * (panel_size + gap)
            resampling = Image.Resampling.BILINEAR if column == 0 else Image.Resampling.NEAREST
            panel = Image.fromarray(array).resize(
                (panel_size, panel_size), resample=resampling
            )
            canvas.paste(panel, (x, image_y))
            if column >= 2:
                spec = model_specs[column - 2]
                values = metrics[spec.key][sample_id]
                text = (
                    f"mIoU {values['miou']:.3f}  F1 {values['f1']:.3f}\n"
                    f"Acc {values['acc']:.3f}"
                )
                draw.multiline_text(
                    (x + 4, image_y + panel_size + 5),
                    text,
                    fill=(42, 49, 61),
                    font=metric_font,
                    spacing=2,
                )

    legend_y = header_height + len(selected_ids) * row_height + 8
    draw.line(
        (margin, legend_y, width - margin, legend_y),
        fill=(195, 201, 211),
        width=2,
    )
    draw.text(
        (margin, legend_y + 12),
        "PASTIS 类别图例",
        fill=(28, 35, 48),
        font=row_font,
    )
    item_width = (width - 2 * margin) // legend_columns
    for class_id, class_name in enumerate(PASTIS_CLASS_NAMES_ZH):
        legend_row, legend_column = divmod(class_id, legend_columns)
        x = margin + legend_column * item_width
        y = legend_y + 50 + legend_row * 34
        color = tuple(int(value) for value in PASTIS_PALETTE[class_id])
        draw.rectangle((x, y + 2, x + 20, y + 22), fill=color, outline=(80, 80, 80))
        draw.text(
            (x + 28, y),
            f"{class_id}  {class_name}",
            fill=(42, 49, 61),
            font=metric_font,
        )
    return canvas


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.5 < args.max_percentile < 1.0:
        raise ValueError("--max-percentile must be between 0.5 and 1.0")

    specs = model_specs(args)
    missing = [str(spec.checkpoint_path) for spec in specs if not spec.checkpoint_path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

    base_config = load_config(specs[0].config_path)
    cache_dir = Path(args.cache_dir or feature_cache_dir(base_config, args.split))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    loader = build_loader(cache_dir, args.batch_size, args.num_workers)
    first_batch = next(iter(loader))
    targets = {
        item["sample_id"]: item["target"].numpy()
        for item in tqdm(loader.dataset, desc="targets")
    }
    sample_ids = list(targets)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    checkpoint_info = {}
    for spec in specs:
        predictions, metrics, metadata = infer_model(spec, loader, first_batch, device)
        all_predictions[spec.key] = predictions
        all_metrics[spec.key] = metrics
        checkpoint_info[spec.key] = {
            "label": spec.label,
            "config": str(spec.config_path.relative_to(ROOT)),
            "checkpoint": str(spec.checkpoint_path.relative_to(ROOT)),
            **metadata,
        }

    selected_ids, details, dataset_means = select_samples(
        sample_ids,
        all_metrics,
        targets,
        args.count,
        args.target_percentiles,
        args.max_percentile,
    )

    raw_dataset = build_pastis_dataset(base_config["data"], args.split)
    index_by_id = raw_index_by_id(raw_dataset)
    raw_samples = {sample_id: raw_dataset[index_by_id[sample_id]] for sample_id in selected_ids}
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in selected_ids:
        sample = raw_samples[sample_id]
        target = sample["target"].cpu().numpy().astype(np.int64, copy=False)
        rgb = make_rgb_composite(sample, base_config)
        for spec in specs:
            comparison = render_triptych(
                rgb,
                target,
                all_predictions[spec.key][sample_id],
                sample_id,
                int(sample["fold"]),
                args.panel_size,
            )
            comparison.save(output_dir / f"{sample_id}_{spec.key}.png")

    montage = render_montage(
        selected_ids,
        raw_samples,
        all_predictions,
        all_metrics,
        specs,
        base_config,
        args.panel_size,
    )
    montage_path = output_dir / "four_scheme_comparison.png"
    montage.save(montage_path)

    summary = {
        "selection_rule": {
            "split": args.split,
            "description": (
                "All three four-model mean metrics exceed their split mean; combined "
                "metric percentile is capped to avoid top-performing cherry-picks."
            ),
            "dataset_four_model_mean": dataset_means,
            "target_percentiles": args.target_percentiles,
            "max_percentile": args.max_percentile,
        },
        "checkpoints": checkpoint_info,
        "selected_sample_ids": selected_ids,
        "samples": details,
    }
    summary_path = output_dir / "selection_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("dataset_four_model_mean=" + json.dumps(dataset_means, ensure_ascii=False))
    for sample_id in selected_ids:
        values = details[sample_id]["four_model_mean"]
        percentile = details[sample_id]["combined_percentile"]
        print(
            f"selected={sample_id} mean_miou={values['miou']:.5f} "
            f"mean_f1={values['f1']:.5f} mean_acc={values['acc']:.5f} "
            f"combined_percentile={percentile:.3f}"
        )
    print(f"montage={montage_path.resolve()}")
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
