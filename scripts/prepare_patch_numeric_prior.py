from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-fold-only normalization statistics for a patch-keyed "
            "numeric CA-HPI source."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--patch-id-column", default="patch_id")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--valid-column", default="valid")
    parser.add_argument("--train-folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--source-name", default="patch_numeric_table")
    return parser.parse_args()


def prepare_patch_numeric_stats(
    input_path: str | Path,
    output_path: str | Path,
    features: list[str] | tuple[str, ...],
    train_folds: set[int],
    patch_id_column: str = "patch_id",
    fold_column: str = "fold",
    valid_column: str = "valid",
    source_name: str = "patch_numeric_table",
) -> dict[str, object]:
    source = Path(input_path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Patch numeric table has no header: {source}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Patch numeric table is empty: {source}")

    features = tuple(str(feature) for feature in features)
    if not features:
        raise ValueError("At least one numeric feature is required.")
    required = [patch_id_column, fold_column, *features]
    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ValueError(f"Patch numeric table {source} is missing columns: {missing}")

    has_valid = valid_column in rows[0]
    seen_patch_ids: set[int] = set()
    values_by_feature: dict[str, list[float]] = {feature: [] for feature in features}
    for row_number, row in enumerate(rows, start=2):
        try:
            patch_id = int(str(row[patch_id_column]).strip())
            fold = int(str(row[fold_column]).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid patch/fold identifier in {source} row {row_number}."
            ) from exc
        if patch_id in seen_patch_ids:
            raise ValueError(f"Duplicate patch_id in {source}: {patch_id}")
        seen_patch_ids.add(patch_id)
        valid = True
        if has_valid:
            normalized = str(row[valid_column]).strip().lower()
            if normalized in {"0", "false", "no", "n", ""}:
                valid = False
            elif normalized not in {"1", "true", "yes", "y"}:
                raise ValueError(
                    f"Invalid {valid_column} in {source} row {row_number}: "
                    f"{row[valid_column]!r}"
                )
        if not valid or fold not in train_folds:
            continue
        for feature in features:
            try:
                value = float(str(row[feature]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {feature} in {source} row {row_number}: {row[feature]!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"{feature} must be finite in {source} row {row_number}."
                )
            values_by_feature[feature].append(value)

    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    counts: dict[str, int] = {}
    zero_variance: list[str] = []
    for feature, values in values_by_feature.items():
        if not values:
            raise ValueError(
                f"No valid train-fold values for patch numeric feature {feature}."
            )
        mean[feature] = statistics.fmean(values)
        feature_std = statistics.pstdev(values)
        if feature_std <= 1e-12:
            feature_std = 1.0
            zero_variance.append(feature)
        std[feature] = feature_std
        counts[feature] = len(values)

    payload: dict[str, object] = {
        "source": source_name,
        "feature_names": list(features),
        "train_folds": sorted(train_folds),
        "mean": mean,
        "std": std,
        "train_value_count": counts,
        "zero_variance_features_standardized_with_std_1": zero_variance,
        "table": str(source).replace("\\", "/"),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def main() -> None:
    args = parse_args()
    payload = prepare_patch_numeric_stats(
        input_path=args.input,
        output_path=args.output,
        features=args.features,
        train_folds={int(fold) for fold in args.train_folds},
        patch_id_column=args.patch_id_column,
        fold_column=args.fold_column,
        valid_column=args.valid_column,
        source_name=args.source_name,
    )
    print(
        f"wrote {args.output} for {payload['feature_names']} using "
        f"train folds {payload['train_folds']}"
    )


if __name__ == "__main__":
    main()
