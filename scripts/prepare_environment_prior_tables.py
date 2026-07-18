from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence


CLIMATE_FEATURES = ("t2m_c", "tp_mm", "ssrd_mj_m2", "swvl1")
SOIL_FEATURES = (
    "ph",
    "soc_gkg",
    "clay_pct",
    "sand_pct",
    "cec_cmolkg",
    "nitrogen_gkg",
)
SOIL_DEPTHS = ("0-5", "5-15", "15-30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and freeze externally extracted ERA5-Land/SoilGrids tables for CA-HPI. "
            "Input values must already use the units documented in docs/M2_M3_CLIMATE_SOIL_PRIOR_CATALOG.md."
        )
    )
    parser.add_argument("--metadata", default="data/PASTIS/metadata.geojson")
    parser.add_argument("--train-folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--allow-incomplete-patches", action="store_true")
    parser.add_argument("--source-version", default="external_frozen_v1")
    parser.add_argument("--climate-input")
    parser.add_argument("--climate-output")
    parser.add_argument("--climate-stats-output")
    parser.add_argument("--soil-input")
    parser.add_argument("--soil-output")
    parser.add_argument("--soil-stats-output")
    return parser.parse_args()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header in {source}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {source}")
    return rows


def _parse_bool(value: object, row_label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"{row_label}: valid must use true/false, got {value!r}")


def _float(value: object, label: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a numeric value, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label}: expected a finite numeric value")
    return parsed


def _metadata_folds(path: str | Path) -> dict[int, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    result: dict[int, int] = {}
    for feature in metadata.get("features", []):
        properties = feature.get("properties") or {}
        patch_id = int(properties["ID_PATCH"])
        if patch_id in result:
            raise ValueError(f"Duplicate ID_PATCH in metadata: {patch_id}")
        result[patch_id] = int(properties["Fold"])
    if not result:
        raise ValueError(f"No PASTIS records found in {path}")
    return result


def _require_columns(rows: list[dict[str, str]], columns: Iterable[str], source: str) -> None:
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _stats(
    records: Sequence[dict[str, object]],
    features: Sequence[str],
    patch_folds: dict[int, int],
    train_folds: set[int],
    source_name: str,
) -> dict[str, object]:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    counts: dict[str, int] = {}
    zero_variance: list[str] = []
    for feature in features:
        values = [
            float(record[feature])
            for record in records
            if bool(record["valid"])
            and patch_folds[int(record["patch_id"])] in train_folds
        ]
        if not values:
            raise ValueError(
                f"No valid train-fold values for {source_name} feature {feature}."
            )
        mean[feature] = statistics.fmean(values)
        raw_std = statistics.pstdev(values)
        if raw_std <= 1e-12:
            raw_std = 1.0
            zero_variance.append(feature)
        std[feature] = raw_std
        counts[feature] = len(values)
    return {
        "source": source_name,
        "feature_names": list(features),
        "train_folds": sorted(train_folds),
        "mean": mean,
        "std": std,
        "train_value_count": counts,
        "zero_variance_features_standardized_with_std_1": zero_variance,
    }


def _write_csv(path: str | Path, records: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: str | Path, value: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_climate(
    input_path: str,
    output_path: str,
    stats_path: str,
    patch_folds: dict[int, int],
    train_folds: set[int],
    allow_incomplete: bool,
    source_version: str,
) -> None:
    rows = _read_csv(input_path)
    _require_columns(rows, ["patch_id", "month", *CLIMATE_FEATURES], input_path)
    seen: set[tuple[int, int]] = set()
    records: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        patch_id = int(row["patch_id"])
        if patch_id not in patch_folds:
            raise ValueError(f"{input_path} row {row_number}: unknown PASTIS patch_id {patch_id}")
        month = int(row["month"])
        if month not in range(1, 13):
            raise ValueError(f"{input_path} row {row_number}: month must be 1..12")
        key = (patch_id, month)
        if key in seen:
            raise ValueError(f"{input_path}: duplicate patch_id/month {key}")
        seen.add(key)
        valid = _parse_bool(row.get("valid", "true"), f"{input_path} row {row_number}")
        confidence = _float(row.get("confidence", "1"), f"{input_path} row {row_number} confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{input_path} row {row_number}: confidence must be in [0,1]")
        record: dict[str, object] = {
            "patch_id": patch_id,
            "month": month,
            "valid": valid,
            "confidence": confidence if valid else 0.0,
            "source_version": row.get("source_version") or source_version,
        }
        for feature in CLIMATE_FEATURES:
            record[feature] = _float(row[feature], f"{input_path} row {row_number} {feature}") if valid else 0.0
        records.append(record)
    expected = {(patch_id, month) for patch_id in patch_folds for month in range(1, 13)}
    missing = expected - seen
    if missing and not allow_incomplete:
        raise ValueError(
            f"{input_path} misses {len(missing)} patch-month records. Use --allow-incomplete-patches "
            "only when missing tokens are intentional."
        )
    records.sort(key=lambda item: (int(item["patch_id"]), int(item["month"])))
    _write_csv(
        output_path,
        records,
        ["patch_id", "month", *CLIMATE_FEATURES, "valid", "confidence", "source_version"],
    )
    stats = _stats(records, CLIMATE_FEATURES, patch_folds, train_folds, "ERA5-Land")
    stats.update({"table": str(output_path), "missing_patch_month_records": len(missing)})
    _write_json(stats_path, stats)
    print(f"Climate: {len(records)} rows -> {output_path}; stats -> {stats_path}")


def prepare_soil(
    input_path: str,
    output_path: str,
    stats_path: str,
    patch_folds: dict[int, int],
    train_folds: set[int],
    allow_incomplete: bool,
    source_version: str,
) -> None:
    rows = _read_csv(input_path)
    _require_columns(rows, ["patch_id", "depth_cm", *SOIL_FEATURES], input_path)
    has_confidence = "confidence" in rows[0]
    has_uncertainty = all(
        f"{feature}_q05" in rows[0] and f"{feature}_q95" in rows[0]
        for feature in SOIL_FEATURES
    )
    if not has_confidence and not has_uncertainty:
        raise ValueError(
            f"{input_path} needs confidence or each {{feature}}_q05/{{feature}}_q95 pair."
        )
    seen: set[tuple[int, str]] = set()
    records: list[dict[str, object]] = []
    spreads: list[list[float] | None] = []
    for row_number, row in enumerate(rows, start=2):
        patch_id = int(row["patch_id"])
        if patch_id not in patch_folds:
            raise ValueError(f"{input_path} row {row_number}: unknown PASTIS patch_id {patch_id}")
        depth = str(row["depth_cm"]).strip()
        if depth not in SOIL_DEPTHS:
            raise ValueError(f"{input_path} row {row_number}: unsupported depth {depth!r}")
        key = (patch_id, depth)
        if key in seen:
            raise ValueError(f"{input_path}: duplicate patch_id/depth {key}")
        seen.add(key)
        valid = _parse_bool(row.get("valid", "true"), f"{input_path} row {row_number}")
        record: dict[str, object] = {
            "patch_id": patch_id,
            "depth_cm": depth,
            "valid": valid,
            "confidence": 0.0,
            "source_version": row.get("source_version") or source_version,
        }
        for feature in SOIL_FEATURES:
            record[feature] = _float(row[feature], f"{input_path} row {row_number} {feature}") if valid else 0.0
        if has_confidence:
            confidence = _float(row["confidence"], f"{input_path} row {row_number} confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{input_path} row {row_number}: confidence must be in [0,1]")
            record["confidence"] = confidence if valid else 0.0
            spreads.append(None)
        else:
            spread = [
                _float(row[f"{feature}_q95"], f"{input_path} row {row_number} {feature}_q95")
                - _float(row[f"{feature}_q05"], f"{input_path} row {row_number} {feature}_q05")
                for feature in SOIL_FEATURES
            ]
            if any(value < 0 for value in spread):
                raise ValueError(f"{input_path} row {row_number}: q95 must be >= q05")
            spreads.append(spread if valid else [0.0] * len(SOIL_FEATURES))
        records.append(record)
    stats = _stats(records, SOIL_FEATURES, patch_folds, train_folds, "SoilGrids")
    if not has_confidence:
        std = stats["std"]
        for record, spread in zip(records, spreads):
            if not bool(record["valid"]):
                continue
            assert spread is not None
            normalized_width = statistics.fmean(
                width / float(std[feature])
                for feature, width in zip(SOIL_FEATURES, spread)
            )
            record["confidence"] = 1.0 / (1.0 + normalized_width)
    expected = {(patch_id, depth) for patch_id in patch_folds for depth in SOIL_DEPTHS}
    missing = expected - seen
    if missing and not allow_incomplete:
        raise ValueError(
            f"{input_path} misses {len(missing)} patch-depth records. Use --allow-incomplete-patches "
            "only when missing tokens are intentional."
        )
    records.sort(key=lambda item: (int(item["patch_id"]), SOIL_DEPTHS.index(str(item["depth_cm"]))))
    _write_csv(
        output_path,
        records,
        ["patch_id", "depth_cm", *SOIL_FEATURES, "valid", "confidence", "source_version"],
    )
    stats.update(
        {
            "table": str(output_path),
            "missing_patch_depth_records": len(missing),
            "confidence": (
                "provided_by_input" if has_confidence else "1 / (1 + mean(IQR90 / train_std))"
            ),
        }
    )
    _write_json(stats_path, stats)
    print(f"Soil: {len(records)} rows -> {output_path}; stats -> {stats_path}")


def main() -> None:
    args = parse_args()
    climate_args = (args.climate_input, args.climate_output, args.climate_stats_output)
    soil_args = (args.soil_input, args.soil_output, args.soil_stats_output)
    if not any(climate_args) and not any(soil_args):
        raise ValueError("Provide a complete --climate-* group and/or a complete --soil-* group.")
    if any(climate_args) and not all(climate_args):
        raise ValueError("Climate preparation needs --climate-input, --climate-output, --climate-stats-output.")
    if any(soil_args) and not all(soil_args):
        raise ValueError("Soil preparation needs --soil-input, --soil-output, --soil-stats-output.")
    patch_folds = _metadata_folds(args.metadata)
    train_folds = set(args.train_folds)
    if args.climate_input:
        prepare_climate(
            args.climate_input,
            args.climate_output,
            args.climate_stats_output,
            patch_folds,
            train_folds,
            args.allow_incomplete_patches,
            args.source_version,
        )
    if args.soil_input:
        prepare_soil(
            args.soil_input,
            args.soil_output,
            args.soil_stats_output,
            patch_folds,
            train_folds,
            args.allow_incomplete_patches,
            args.source_version,
        )


if __name__ == "__main__":
    main()
