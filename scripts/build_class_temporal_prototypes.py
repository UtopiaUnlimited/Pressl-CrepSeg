from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import apply_cache_overrides, feature_cache_dir, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fold-safe class-month prototype archives from temporal_v2 caches."
    )
    parser.add_argument("--config", default="configs/galileo_3d_aware_dpt.yaml")
    parser.add_argument("--train-cache-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="data/priors/class_temporal_prototypes",
    )
    parser.add_argument(
        "--prototypes-per-group",
        nargs="+",
        type=int,
        default=[1, 4],
        help="Build one archive for every requested K, normally 1 and 4.",
    )
    parser.add_argument("--feature-layer-index", type=int, default=-1)
    parser.add_argument("--min-tokens-per-prototype", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-format", choices=["temporal_v2"], default="temporal_v2")
    parser.add_argument("--temporal-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def _resolve_layer_index(requested_index: int, num_layers: int) -> int:
    index = requested_index if requested_index >= 0 else num_layers + requested_index
    if index < 0 or index >= num_layers:
        raise ValueError(f"feature_layer_index={requested_index} is invalid for {num_layers} layers.")
    return index


def _downsample_target_majority(
    target: np.ndarray,
    height: int,
    width: int,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    if target.ndim != 2:
        raise ValueError(f"Expected target [H,W], got {target.shape}.")
    source_height, source_width = target.shape
    if source_height % height or source_width % width:
        raise ValueError(
            "Target resolution must be an integer multiple of the Galileo feature grid: "
            f"target={target.shape}, feature={(height, width)}."
        )
    row_scale = source_height // height
    col_scale = source_width // width
    reduced = np.full((height, width), ignore_index, dtype=np.int64)
    for row in range(height):
        for col in range(width):
            values = target[
                row * row_scale : (row + 1) * row_scale,
                col * col_scale : (col + 1) * col_scale,
            ].reshape(-1)
            valid = values[(values >= 0) & (values < num_classes)]
            if valid.size:
                reduced[row, col] = int(np.bincount(valid, minlength=num_classes).argmax())
    return reduced


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def _update_online_centers(
    centers: np.ndarray,
    counts: np.ndarray,
    vectors: np.ndarray,
) -> None:
    """One shuffled-pass mini-batch k-means update without retaining pixels."""

    if vectors.size == 0:
        return
    remaining = vectors
    missing = np.flatnonzero(counts == 0)
    if missing.size:
        take = min(int(missing.size), int(remaining.shape[0]))
        centers[missing[:take]] = remaining[:take]
        counts[missing[:take]] = 1
        remaining = remaining[take:]
    active = np.flatnonzero(counts > 0)
    if remaining.size == 0 or active.size == 0:
        return
    active_centers = centers[active]
    active_centers = active_centers / np.clip(
        np.linalg.norm(active_centers, axis=1, keepdims=True), 1e-8, None
    )
    assignment = active[(remaining @ active_centers.T).argmax(axis=1)]
    for prototype_index in np.unique(assignment):
        chosen = remaining[assignment == prototype_index]
        previous_count = int(counts[prototype_index])
        updated_count = previous_count + int(chosen.shape[0])
        centers[prototype_index] = (
            centers[prototype_index] * previous_count + chosen.sum(axis=0)
        ) / max(1, updated_count)
        counts[prototype_index] = updated_count


def _prototype_output_path(output_dir: Path, prototypes_per_group: int) -> Path:
    return output_dir / (
        "pastis_fold123_final_layer_"
        f"class_temporal_prototypes_k{prototypes_per_group}_online_v1.npz"
    )


def build_prototype_archives(
    cache_dir: str | Path,
    output_dir: str | Path,
    num_classes: int,
    train_folds: tuple[int, ...],
    ignore_index: int,
    prototypes_per_group: tuple[int, ...],
    feature_layer_index: int,
    min_tokens_per_prototype: int,
    seed: int,
) -> list[Path]:
    if min_tokens_per_prototype < 1:
        raise ValueError("min_tokens_per_prototype must be positive.")
    requested_counts = tuple(sorted({int(value) for value in prototypes_per_group}))
    if not requested_counts or any(value < 1 for value in requested_counts):
        raise ValueError("prototypes_per_group must contain positive integers.")
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No cache files found in {cache_dir}.")

    with np.load(files[0], allow_pickle=False) as first:
        if "temporal_features_by_layer" not in first:
            raise ValueError("Class-temporal prototypes require temporal_v2 cache files.")
        shape = first["temporal_features_by_layer"].shape
    if len(shape) != 5:
        raise ValueError(f"Expected [L,T,D,H,W] temporal features, got {shape}.")
    num_layers, timesteps, channels, height, width = (int(value) for value in shape)
    layer_index = _resolve_layer_index(feature_layer_index, num_layers)

    states: dict[int, dict[str, np.ndarray]] = {}
    for count in requested_counts:
        states[count] = {
            "centers": np.zeros(
                (3, num_classes, timesteps, count, channels), dtype=np.float32
            ),
            "counts": np.zeros((3, num_classes, timesteps, count), dtype=np.int64),
        }

    rng = np.random.default_rng(int(seed))
    shuffled_files = [files[index] for index in rng.permutation(len(files))]
    used_files = 0
    skipped_files = 0
    group_file_counts = [0, 0]
    for file_path in tqdm(shuffled_files, desc="build class-temporal prototypes"):
        with np.load(file_path, allow_pickle=False) as sample:
            fold = int(sample["fold"])
            if fold not in train_folds:
                skipped_files += 1
                continue
            temporal = sample["temporal_features_by_layer"]
            if temporal.shape != shape:
                raise ValueError(
                    f"Temporal cache shape changed in {file_path}: {temporal.shape} vs {shape}."
                )
            months = sample["months"].astype(np.int64, copy=False)
            if months.shape != (timesteps,):
                raise ValueError(
                    f"Expected {timesteps} month indices in {file_path}, got {months.shape}."
                )
            if months.size and (months.min() < 0 or months.max() >= timesteps):
                raise ValueError(f"Invalid calendar month values in {file_path}: {months}.")
            target = sample["target"].astype(np.int64, copy=False)
            target_low = _downsample_target_majority(
                target, height, width, num_classes, ignore_index
            ).reshape(-1)
            features = temporal[layer_index]
            patch_group = int(sample["patch_id"]) % 2
            group_file_counts[patch_group] += 1
            used_files += 1

            valid_labels = np.unique(target_low[(target_low >= 0) & (target_low < num_classes)])
            for time_index, month in enumerate(months.tolist()):
                vectors = features[time_index].transpose(1, 2, 0).reshape(-1, channels)
                for class_index in valid_labels.tolist():
                    class_vectors = _normalize_rows(vectors[target_low == class_index])
                    for bank_index in (patch_group, 2):
                        for count, state in states.items():
                            _update_online_centers(
                                state["centers"][bank_index, class_index, month],
                                state["counts"][bank_index, class_index, month],
                                class_vectors,
                            )

    if used_files == 0:
        raise ValueError(f"No cache files in {cache_dir} belonged to train folds {train_folds}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for count, state in states.items():
        centers = state["centers"]
        counts = state["counts"]
        valid = counts >= int(min_tokens_per_prototype)
        confidence = np.clip(
            counts.astype(np.float32) / float(min_tokens_per_prototype), 0.0, 1.0
        )
        centers = centers / np.clip(
            np.linalg.norm(centers, axis=-1, keepdims=True), 1e-8, None
        )
        path = _prototype_output_path(output_dir, count)
        metadata = {
            "format": "pastis_class_temporal_prototype_online_v1",
            "bank_names": ["partition_a", "partition_b", "all_train"],
            "partition_modulus": 2,
            "train_folds": list(train_folds),
            "feature_layer_index": layer_index,
            "timesteps": timesteps,
            "num_classes": num_classes,
            "prototypes_per_group": count,
            "channels": channels,
            "min_tokens_per_prototype": int(min_tokens_per_prototype),
            "seed": int(seed),
            "cache_dir": str(cache_dir),
            "used_files": used_files,
            "skipped_files": skipped_files,
            "partition_file_counts": group_file_counts,
            "feature_normalization": "l2",
        }
        np.savez_compressed(
            path,
            prototypes=centers.astype(np.float32, copy=False),
            mask=valid,
            confidence=confidence,
            counts=counts,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=True, sort_keys=True)),
        )
        outputs.append(path)
    return outputs


def main() -> None:
    args = parse_args()
    config = apply_cache_overrides(
        load_config(args.config),
        cache_format=args.cache_format,
        temporal_dtype=args.temporal_dtype,
    )
    cache_dir = args.train_cache_dir or feature_cache_dir(config, "train")
    paths = build_prototype_archives(
        cache_dir=cache_dir,
        output_dir=args.output_dir,
        num_classes=int(config["data"]["num_classes"]),
        train_folds=tuple(int(value) for value in config["data"]["train_folds"]),
        ignore_index=int(config.get("loss", {}).get("ignore_index", -1)),
        prototypes_per_group=tuple(args.prototypes_per_group),
        feature_layer_index=int(args.feature_layer_index),
        min_tokens_per_prototype=int(args.min_tokens_per_prototype),
        seed=int(args.seed),
    )
    for path in paths:
        print(f"prototype_archive={path.resolve()}")


if __name__ == "__main__":
    main()
