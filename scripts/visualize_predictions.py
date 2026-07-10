from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import PASTISDataset, build_pastis_dataset, pastis_collate_fn  # noqa: E402
from data.pastis import GALILEO_S2_MEAN, GALILEO_S2_STD  # noqa: E402
from models import build_model  # noqa: E402
from utils import load_config  # noqa: E402


# A fixed categorical palette keeps ground truth and prediction directly comparable.
PASTIS_PALETTE = np.asarray(
    [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213),
        (140, 86, 75),
        (196, 156, 148),
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
    ],
    dtype=np.uint8,
)
VOID_COLOR = np.asarray((48, 48, 48), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PASTIS RGB composites, labels, and model predictions."
    )
    parser.add_argument(
        "--config",
        default="configs/galileo_single_layer_dpt_shared.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/galileo_single_layer_dpt_shared_"
            "paper_input_bs16_cached/best.pt"
        ),
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional explicit IDs such as 10002_y0_x64; overrides automatic selection.",
    )
    parser.add_argument("--output-dir", default="outputs/test_predictions")
    parser.add_argument("--device", default=None)
    parser.add_argument("--panel-size", type=int, default=384)
    return parser.parse_args()


def _sample_id(dataset: PASTISDataset, index: int) -> str:
    record_index, tile_id = divmod(index, dataset.tiles_per_record)
    record = dataset.records[record_index]
    tile_row, tile_col = divmod(tile_id, dataset.tiles_per_side)
    return (
        f"{record.patch_id}_y{tile_row * dataset.tile_size}"
        f"_x{tile_col * dataset.tile_size}"
    )


def _indices_for_sample_ids(
    dataset: PASTISDataset,
    sample_ids: list[str],
) -> list[int]:
    index_by_id = {_sample_id(dataset, index): index for index in range(len(dataset))}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Sample IDs are not in this split: {missing}")
    return [index_by_id[sample_id] for sample_id in sample_ids]


def _target_selection_score(target: np.ndarray, void_label: int | None) -> float:
    valid = np.ones(target.shape, dtype=bool)
    if void_label is not None:
        valid &= target != int(void_label)
    valid &= target >= 0
    if not valid.any():
        return float("-inf")

    labels, counts = np.unique(target[valid], return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    if len(labels) > 1:
        entropy /= math.log(len(labels))
    substantial_classes = int((counts >= max(8, int(valid.sum() * 0.01))).sum())
    valid_fraction = float(valid.mean())
    return 2.0 * entropy + 0.25 * substantial_classes + valid_fraction


def select_representative_indices(dataset: PASTISDataset, count: int) -> list[int]:
    """Pick diverse labels from separate portions of the split without using predictions."""

    if count <= 0:
        raise ValueError("num_samples must be positive")
    count = min(count, len(dataset))
    boundaries = np.linspace(0, len(dataset), count + 1, dtype=np.int64)
    selected: list[int] = []

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        best_index = int(start)
        best_score = float("-inf")
        loaded_record_index = -1
        full_target: np.ndarray | None = None

        for index in range(int(start), int(stop)):
            record_index, tile_id = divmod(index, dataset.tiles_per_record)
            if record_index != loaded_record_index:
                record = dataset.records[record_index]
                target_path = dataset.root / "ANNOTATIONS" / f"TARGET_{record.patch_id}.npy"
                full_target = np.load(target_path)
                if full_target.ndim == 3:
                    full_target = full_target[dataset.target_channel]
                loaded_record_index = record_index

            if full_target is None:
                continue
            tile_row, tile_col = divmod(tile_id, dataset.tiles_per_side)
            y0 = tile_row * dataset.tile_size
            x0 = tile_col * dataset.tile_size
            tile_target = full_target[
                y0 : y0 + dataset.tile_size,
                x0 : x0 + dataset.tile_size,
            ]
            score = _target_selection_score(tile_target, dataset.void_label)
            if score > best_score:
                best_index = index
                best_score = score

        selected.append(best_index)
    return selected


def _denormalize_s2(sample: dict, config: dict) -> np.ndarray:
    s2 = sample["s2"].cpu().numpy().astype(np.float32, copy=False)
    data_cfg = config["data"]
    normalization = str(data_cfg.get("normalization", "none")).lower()
    if normalization == "none":
        return s2
    if normalization != "galileo_norm_no_clip":
        raise ValueError(f"Unsupported visualization normalization: {normalization}")

    multiplier = float(data_cfg.get("normalization_std_multiplier", 2.0))
    means = GALILEO_S2_MEAN[None, :, None, None]
    stds = GALILEO_S2_STD[None, :, None, None] * multiplier
    return s2 * (2.0 * stds) + (means - stds)


def make_rgb_composite(sample: dict, config: dict) -> np.ndarray:
    """Create a contrast-stretched temporal median from PASTIS B4/B3/B2."""

    raw_s2 = _denormalize_s2(sample, config)
    rgb = np.median(raw_s2[:, [2, 1, 0]], axis=0).transpose(1, 2, 0)
    finite_values = rgb[np.isfinite(rgb)]
    low, high = np.percentile(finite_values, [2.0, 98.0])
    if high <= low:
        output = np.zeros_like(rgb, dtype=np.float32)
    else:
        # One shared range preserves the relative B4/B3/B2 balance.
        output = np.clip((rgb - low) / (high - low), 0.0, 1.0)
    output = np.power(output, 0.9)
    return np.rint(output * 255.0).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    colored = np.broadcast_to(VOID_COLOR, (*mask.shape, 3)).copy()
    valid = (mask >= 0) & (mask < len(PASTIS_PALETTE))
    colored[valid] = PASTIS_PALETTE[mask[valid]]
    return colored


def sample_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    valid = (target >= 0) & (target < len(PASTIS_PALETTE))
    if not valid.any():
        return float("nan"), float("nan")
    target_valid = target[valid]
    prediction_valid = prediction[valid]
    accuracy = float((target_valid == prediction_valid).mean())

    ious = []
    classes = np.union1d(np.unique(target_valid), np.unique(prediction_valid))
    for class_id in classes:
        target_class = target_valid == class_id
        prediction_class = prediction_valid == class_id
        union = np.logical_or(target_class, prediction_class).sum()
        if union:
            ious.append(np.logical_and(target_class, prediction_class).sum() / union)
    return accuracy, float(np.mean(ious)) if ious else float("nan")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    x_center: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.text((x_center - width // 2, y), text, font=font, fill=fill)


def render_triptych(
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    sample_id: str,
    fold: int,
    panel_size: int,
) -> Image.Image:
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")

    margin = 20
    gap = 18
    image_y = 68
    footer_height = 54
    width = 2 * margin + 3 * panel_size + 2 * gap
    height = image_y + panel_size + footer_height
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    meta_font = _font(17)

    accuracy, miou = sample_metrics(target, prediction)
    draw.text(
        (margin, 9),
        f"{sample_id} | fold {fold} | pixel accuracy {accuracy:.3f} | sample mIoU {miou:.3f}",
        font=meta_font,
        fill=(25, 25, 25),
    )

    panels = (
        ("S2 RGB median composite", rgb, Image.Resampling.BILINEAR),
        ("Ground truth", colorize_mask(target), Image.Resampling.NEAREST),
        ("Prediction", colorize_mask(prediction), Image.Resampling.NEAREST),
    )
    for panel_index, (label, array, resampling) in enumerate(panels):
        x = margin + panel_index * (panel_size + gap)
        _draw_centered(
            draw,
            x + panel_size // 2,
            39,
            label,
            title_font,
            (25, 25, 25),
        )
        panel = Image.fromarray(array).resize((panel_size, panel_size), resample=resampling)
        canvas.paste(panel, (x, image_y))

    present_classes = np.union1d(
        np.unique(target[target >= 0]),
        np.unique(prediction[prediction >= 0]),
    )
    legend_y = image_y + panel_size + 17
    draw.text((margin, legend_y), "Class IDs:", font=meta_font, fill=(25, 25, 25))
    legend_x = margin + 82
    for class_id in present_classes:
        class_id = int(class_id)
        if not 0 <= class_id < len(PASTIS_PALETTE):
            continue
        color = tuple(int(value) for value in PASTIS_PALETTE[class_id])
        draw.rectangle((legend_x, legend_y + 2, legend_x + 14, legend_y + 16), fill=color)
        draw.text((legend_x + 18, legend_y), str(class_id), font=meta_font, fill=(25, 25, 25))
        legend_x += 45
    return canvas


def _load_decoder_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_encoder_missing = [
        key for key in incompatible.missing_keys if not key.startswith("encoder.")
    ]
    if unexpected or non_encoder_missing:
        raise RuntimeError(
            "Checkpoint does not match the configured decoder: "
            f"unexpected={unexpected}, missing={non_encoder_missing}"
        )
    return checkpoint


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    dataset = build_pastis_dataset(config["data"], args.split)
    if args.sample_ids:
        selected_indices = _indices_for_sample_ids(dataset, args.sample_ids)
    else:
        selected_indices = select_representative_indices(dataset, args.num_samples)

    inference_config = copy.deepcopy(config)
    decoder_name = str(inference_config["model"].get("decoder", "single_layer_dpt")).lower()
    if decoder_name in {"single_layer_dpt", "single", "dpt"}:
        inference_config["encoder"]["hidden_layers"] = []

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(inference_config)
    checkpoint = _load_decoder_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Image.Image] = []
    print(
        f"checkpoint={checkpoint_path} epoch={checkpoint.get('epoch', 'unknown')} "
        f"device={device}"
    )

    for index in selected_indices:
        sample = dataset[index]
        batch = pastis_collate_fn([sample])
        logits = model(batch)
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64, copy=False)
        target = sample["target"].cpu().numpy().astype(np.int64, copy=False)
        rgb = make_rgb_composite(sample, config)
        sample_id = str(sample["sample_id"])
        comparison = render_triptych(
            rgb=rgb,
            target=target,
            prediction=prediction,
            sample_id=sample_id,
            fold=int(sample["fold"]),
            panel_size=args.panel_size,
        )
        output_path = output_dir / f"{args.split}_{sample_id}.png"
        comparison.save(output_path)
        rendered.append(comparison)
        accuracy, miou = sample_metrics(target, prediction)
        print(
            f"sample={sample_id} pixel_accuracy={accuracy:.5f} "
            f"sample_miou={miou:.5f} output={output_path}"
        )

    if rendered:
        gap = 16
        overview = Image.new(
            "RGB",
            (
                max(image.width for image in rendered),
                sum(image.height for image in rendered) + gap * (len(rendered) - 1),
            ),
            (225, 225, 225),
        )
        y = 0
        for image in rendered:
            overview.paste(image, (0, y))
            y += image.height + gap
        overview_path = output_dir / f"{args.split}_overview.png"
        overview.save(overview_path)
        print(f"overview={overview_path}")


if __name__ == "__main__":
    main()
