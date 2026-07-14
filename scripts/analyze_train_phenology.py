from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.pastis import (  # noqa: E402
    PASTIS_CLASS_NAMES,
    PASTIS_VOID_LABEL,
    aggregate_monthly_s2,
)


MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
RED_BAND_INDEX = 2  # B4 in PASTIS order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
NIR_BAND_INDEX = 6  # B8.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only PASTIS class-month NDVI statistics and a data-driven "
            "soft prior candidate."
        )
    )
    parser.add_argument("--root", default="data/PASTIS")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output-dir", default="data/priors")
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def read_records(root: Path, folds: set[int]) -> list[tuple[int, int, tuple[int, ...]]]:
    with (root / "metadata.geojson").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    records: list[tuple[int, int, tuple[int, ...]]] = []
    for feature in metadata.get("features", []):
        props = feature["properties"]
        fold = int(props["Fold"])
        if fold not in folds:
            continue
        raw_dates = props["dates-S2"]
        if isinstance(raw_dates, dict):
            dates = tuple(int(raw_dates[key]) for key in sorted(raw_dates, key=lambda value: int(value)))
        else:
            dates = tuple(int(value) for value in raw_dates)
        records.append((int(props["ID_PATCH"]), fold, dates))
    return sorted(records, key=lambda value: value[0])


def write_stats(
    path: Path,
    sums: np.ndarray,
    squared_sums: np.ndarray,
    counts: np.ndarray,
    patch_counts: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_id",
                "official_name",
                "calendar_month",
                "mean_ndvi",
                "std_ndvi",
                "pixel_count",
                "patch_count",
            ]
        )
        for class_id, class_name in enumerate(PASTIS_CLASS_NAMES):
            for month_index, month_name in enumerate(MONTH_NAMES):
                count = int(counts[class_id, month_index])
                if count == 0:
                    mean = float("nan")
                    std = float("nan")
                else:
                    mean = float(sums[class_id, month_index] / count)
                    variance = squared_sums[class_id, month_index] / count - mean**2
                    std = float(np.sqrt(max(variance, 0.0)))
                writer.writerow(
                    [
                        class_id,
                        class_name,
                        month_name,
                        f"{mean:.8f}" if np.isfinite(mean) else "",
                        f"{std:.8f}" if np.isfinite(std) else "",
                        count,
                        int(patch_counts[class_id, month_index]),
                    ]
                )


def write_prior(path: Path, sums: np.ndarray, counts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_id",
                "official_name",
                "prior_status",
                "source_ref",
                *MONTH_NAMES,
                "notes",
            ]
        )
        for class_id, class_name in enumerate(PASTIS_CLASS_NAMES):
            means = np.full(len(MONTH_NAMES), np.nan, dtype=np.float64)
            supported = counts[class_id] > 0
            means[supported] = sums[class_id, supported] / counts[class_id, supported]

            if not np.any(supported):
                prior = np.ones(len(MONTH_NAMES), dtype=np.float64)
                status = "no_support"
            elif np.nanmax(means[supported]) - np.nanmin(means[supported]) < 1e-8:
                prior = np.ones(len(MONTH_NAMES), dtype=np.float64)
                prior[supported] = 1.0
                status = "flat_data"
            else:
                low = np.nanmin(means[supported])
                high = np.nanmax(means[supported])
                prior = np.ones(len(MONTH_NAMES), dtype=np.float64)
                prior[supported] = 0.1 + 0.9 * (means[supported] - low) / (high - low)
                status = "train_ndvi_minmax"

            if class_id == PASTIS_VOID_LABEL:
                status = "excluded_void"
                prior[:] = np.nan

            values = [f"{value:.6f}" if np.isfinite(value) else "" for value in prior]
            writer.writerow(
                [
                    class_id,
                    class_name,
                    status,
                    "train_folds_only:1,2,3",
                    *values,
                    "NDVI mean min-max scaled per class; this is an analysis candidate, not a probability.",
                ]
            )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    folds = {int(value) for value in args.folds}
    records = read_records(root, folds)
    if args.max_patches is not None:
        records = records[: int(args.max_patches)]
    if not records:
        raise RuntimeError(f"No PASTIS records found for folds {sorted(folds)}")

    class_count = len(PASTIS_CLASS_NAMES)
    month_count = len(MONTH_NAMES)
    sums = np.zeros((class_count, month_count), dtype=np.float64)
    squared_sums = np.zeros((class_count, month_count), dtype=np.float64)
    counts = np.zeros((class_count, month_count), dtype=np.int64)
    patch_counts = np.zeros((class_count, month_count), dtype=np.int64)

    for record_index, (patch_id, fold, dates) in enumerate(records, start=1):
        s2 = np.load(root / "DATA_S2" / f"S2_{patch_id}.npy")
        target = np.load(root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy")
        if target.ndim == 3:
            target = target[0]
        if target.ndim != 2:
            raise ValueError(f"Expected target [H,W] or [C,H,W], got {target.shape} for {patch_id}")

        monthly_s2, month_indices, _, _ = aggregate_monthly_s2(s2, dates, num_timesteps=12, start_offset=1)
        red = monthly_s2[:, RED_BAND_INDEX].astype(np.float64)
        nir = monthly_s2[:, NIR_BAND_INDEX].astype(np.float64)
        denominator = nir + red
        ndvi = np.divide(nir - red, denominator, out=np.zeros_like(nir), where=np.abs(denominator) > 1e-6)

        valid_target = target != PASTIS_VOID_LABEL
        for class_id in range(class_count):
            class_mask = (target == class_id) & valid_target
            if not np.any(class_mask):
                continue
            for timestep, calendar_month in enumerate(month_indices):
                values = ndvi[timestep][class_mask]
                month_index = int(calendar_month)
                sums[class_id, month_index] += values.sum()
                squared_sums[class_id, month_index] += np.square(values).sum()
                counts[class_id, month_index] += values.size
                patch_counts[class_id, month_index] += 1

        if record_index == 1 or record_index % max(int(args.progress_every), 1) == 0 or record_index == len(records):
            print(f"processed {record_index}/{len(records)} patches (last patch={patch_id}, fold={fold})", flush=True)

    write_stats(output_dir / "pastis_train_ndvi_stats.csv", sums, squared_sums, counts, patch_counts)
    write_prior(output_dir / "pastis_data_prior_draft.csv", sums, counts)
    with (output_dir / "pastis_train_phenology_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "root": str(root),
                "folds": sorted(folds),
                "patch_count": len(records),
                "aggregation": "aggregate_monthly_s2(start_offset=1, num_timesteps=12)",
                "ndvi_bands": {"red": "B4", "nir": "B8"},
                "void_label": PASTIS_VOID_LABEL,
                "warning": "This is train-only descriptive analysis; it is not a calibrated probability prior.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"wrote statistics and prior candidates to {output_dir}")


if __name__ == "__main__":
    main()
