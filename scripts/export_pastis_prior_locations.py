from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PASTIS patch centres for offline ERA5-Land/SoilGrids extraction."
    )
    parser.add_argument("--metadata", default="data/PASTIS/metadata.geojson")
    parser.add_argument("--output", default="data/priors/pastis_patch_locations_v1.csv")
    parser.add_argument(
        "--target-crs",
        default="EPSG:4326",
        help="Coordinate reference system expected by the external sampler.",
    )
    return parser.parse_args()


def _points(coordinates: object) -> Iterator[tuple[float, float]]:
    if isinstance(coordinates, (list, tuple)):
        if len(coordinates) >= 2 and all(
            isinstance(value, (int, float)) for value in coordinates[:2]
        ):
            yield float(coordinates[0]), float(coordinates[1])
            return
        for item in coordinates:
            yield from _points(item)


def _source_crs(metadata: dict) -> str:
    crs = metadata.get("crs") or {}
    properties = crs.get("properties") if isinstance(crs, dict) else None
    name = properties.get("name") if isinstance(properties, dict) else None
    if not name:
        raise ValueError("metadata.geojson has no declared CRS; pass a GeoJSON with CRS metadata.")
    if "EPSG::" in str(name):
        return f"EPSG:{str(name).rsplit('EPSG::', 1)[1]}"
    return str(name)


def export_locations(
    metadata_path: str | Path,
    output_path: str | Path,
    target_crs: str = "EPSG:4326",
) -> int:
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    features = metadata.get("features")
    if not isinstance(features, list):
        raise ValueError("metadata.geojson must be a GeoJSON FeatureCollection.")

    source_crs = _source_crs(metadata)
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "Exporting longitude/latitude needs pyproj. Install it in the active environment."
        ) from exc
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    records: list[dict[str, object]] = []
    seen_patch_ids: set[int] = set()
    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        patch_id = int(properties["ID_PATCH"])
        if patch_id in seen_patch_ids:
            raise ValueError(f"Duplicate ID_PATCH in metadata: {patch_id}")
        seen_patch_ids.add(patch_id)
        points = list(_points(geometry.get("coordinates")))
        if not points:
            raise ValueError(f"Patch {patch_id} has no usable geometry coordinates.")
        x_values, y_values = zip(*points)
        center_x = (min(x_values) + max(x_values)) / 2.0
        center_y = (min(y_values) + max(y_values)) / 2.0
        lon, lat = transformer.transform(center_x, center_y)
        records.append(
            {
                "patch_id": patch_id,
                "fold": int(properties["Fold"]),
                "tile": str(properties.get("TILE", "")),
                "x_epsg2154": f"{center_x:.6f}",
                "y_epsg2154": f"{center_y:.6f}",
                "lon": f"{lon:.8f}",
                "lat": f"{lat:.8f}",
            }
        )
    records.sort(key=lambda item: int(item["patch_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main() -> None:
    args = parse_args()
    count = export_locations(args.metadata, args.output, args.target_crs)
    print(f"Wrote {count} patch locations to {args.output}")
    print(f"Transformed metadata CRS -> {args.target_crs}")


if __name__ == "__main__":
    main()
