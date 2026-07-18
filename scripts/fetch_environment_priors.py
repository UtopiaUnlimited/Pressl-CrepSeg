from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_pastis_prior_locations import export_locations  # noqa: E402
from scripts.prepare_environment_prior_tables import (  # noqa: E402
    CLIMATE_FEATURES,
    SOIL_DEPTHS,
    SOIL_FEATURES,
    _metadata_folds,
    prepare_climate,
    prepare_soil,
)


ERA5_DATASET = "reanalysis-era5-land-monthly-means"
ERA5_VARIABLES = (
    "2m_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "volumetric_soil_water_layer_1",
)
ERA5_CROP_MONTHS = (
    (2018, 10),
    (2018, 11),
    (2018, 12),
    (2019, 1),
    (2019, 2),
    (2019, 3),
    (2019, 4),
    (2019, 5),
    (2019, 6),
    (2019, 7),
    (2019, 8),
    (2019, 9),
)
SOIL_SOURCE_FIELDS = {
    "ph": ("phh2o", 10.0),
    "soc_gkg": ("soc", 10.0),
    "clay_pct": ("clay", 10.0),
    "sand_pct": ("sand", 10.0),
    "cec_cmolkg": ("cec", 10.0),
    "nitrogen_gkg": ("nitrogen", 100.0),
}
SOIL_QUANTILES = {"q05": "Q0.05", "q50": "Q0.5", "q95": "Q0.95"}


@dataclass(frozen=True)
class PatchLocation:
    patch_id: int
    tile: str
    lon: float
    lat: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fully download, sample, validate, and freeze M2 ERA5-Land plus "
            "M3 SoilGrids priors. ERA5-Land requires a personal CDS API token."
        )
    )
    parser.add_argument("--metadata", default="data/PASTIS/metadata.geojson")
    parser.add_argument("--locations", default="data/priors/pastis_patch_locations_v1.csv")
    parser.add_argument("--raw-dir", default="data/priors/raw/environment_v1")
    parser.add_argument("--climate-raw", default="data/priors/era5_land_pastis_cropyear_raw.csv")
    parser.add_argument("--soil-raw", default="data/priors/soilgrids_pastis_surface_raw.csv")
    parser.add_argument("--climate-output", default="data/priors/era5_land_pastis_cropyear_v1.csv")
    parser.add_argument(
        "--climate-stats-output",
        default="data/priors/era5_land_pastis_cropyear_v1_stats.json",
    )
    parser.add_argument("--soil-output", default="data/priors/soilgrids_pastis_surface_v1.csv")
    parser.add_argument(
        "--soil-stats-output",
        default="data/priors/soilgrids_pastis_surface_v1_stats.json",
    )
    parser.add_argument("--source-version", default="m2_m3_auto_center_v1")
    parser.add_argument("--skip-era5", action="store_true")
    parser.add_argument("--skip-soil", action="store_true")
    parser.add_argument("--skip-freeze", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Request external NetCDF/GeoTIFF files again instead of reusing raw downloads.",
    )
    parser.add_argument("--soil-timeout", type=int, default=120)
    parser.add_argument("--soil-retries", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_locations(path: str | Path) -> list[PatchLocation]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patch_id", "tile", "lon", "lat"}
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise ValueError(f"Location table {source} needs columns {sorted(required)}.")
        locations = [
            PatchLocation(
                patch_id=int(row["patch_id"]),
                tile=str(row["tile"]),
                lon=float(row["lon"]),
                lat=float(row["lat"]),
            )
            for row in reader
        ]
    if not locations:
        raise ValueError(f"Location table is empty: {source}")
    if len({location.patch_id for location in locations}) != len(locations):
        raise ValueError(f"Location table has duplicate patch_id: {source}")
    return locations


def _ensure_locations(metadata: str | Path, locations: str | Path, overwrite: bool) -> list[PatchLocation]:
    destination = Path(locations)
    if overwrite or not destination.is_file():
        count = export_locations(metadata, destination)
        print(f"Location export: {count} patches -> {destination}")
    return _read_locations(destination)


def _import_cdsapi():
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "Missing cdsapi. Install project requirements, then configure your personal CDS token."
        ) from exc
    credentials = Path.home() / ".cdsapirc"
    if not credentials.is_file():
        raise FileNotFoundError(
            "ERA5-Land download needs your CDS credentials at ~/.cdsapirc. Create it with:\n"
            "url: https://cds.climate.copernicus.eu/api\n"
            "key: <your-personal-access-token>\n"
            "Also log into CDS once and accept the ERA5-Land dataset terms."
        )
    return cdsapi


def _era5_area(locations: Iterable[PatchLocation]) -> list[float]:
    longitudes = [location.lon for location in locations]
    latitudes = [location.lat for location in locations]
    padding = 0.15
    return [
        min(90.0, max(latitudes) + padding),
        max(-180.0, min(longitudes) - padding),
        max(-90.0, min(latitudes) - padding),
        min(180.0, max(longitudes) + padding),
    ]


def _download_era5(locations: list[PatchLocation], destination: Path, redownload: bool) -> dict:
    if destination.is_file() and not redownload:
        print(f"Reuse existing ERA5-Land NetCDF: {destination}")
        return {"reused": True}
    cdsapi = _import_cdsapi()
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": list(ERA5_VARIABLES),
        "year": ["2018", "2019"],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": _era5_area(locations),
    }
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print("Requesting ERA5-Land monthly means from CDS. This can wait in the CDS queue.")
    try:
        cdsapi.Client().retrieve(ERA5_DATASET, request, str(temporary))
    except Exception as exc:
        if "required licences not accepted" in str(exc).lower():
            raise PermissionError(
                "CDS API token is valid, but ERA5-Land terms are not accepted. "
                "Log into CDS and accept the licences at: "
                "https://cds.climate.copernicus.eu/datasets/"
                "reanalysis-era5-land-monthly-means?tab=download#manage-licences"
            ) from exc
        raise
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("CDS reported success but did not create a non-empty NetCDF file.")
    temporary.replace(destination)
    return {"reused": False, "request": request}


def _require_xarray():
    try:
        import xarray
    except ImportError as exc:
        raise ImportError(
            "Missing xarray/netCDF4. Install project requirements before reading ERA5-Land NetCDF."
        ) from exc
    return xarray


def _find_name(available: Iterable[str], candidates: Iterable[str], label: str) -> str:
    available_set = set(available)
    for candidate in candidates:
        if candidate in available_set:
            return candidate
    raise KeyError(f"Could not find {label}; expected one of {list(candidates)}, got {sorted(available_set)}")


def _scalar(value: object) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else float("nan")


def _nearest_valid_era5_cells(
    dataset,
    *,
    locations: list[PatchLocation],
    latitude_name: str,
    longitude_name: str,
    time_name: str,
    variable_names: dict[str, str],
) -> tuple[dict[int, tuple[int, int]], dict[int, float]]:
    """Choose one complete land grid cell for every patch over the crop year.

    ERA5-Land has no values over water. A bare nearest-neighbour lookup can
    therefore select a coastal water cell for an otherwise valid PASTIS patch.
    We instead choose the nearest grid cell whose four variables are finite
    for all 12 crop-year months, keeping the chosen cell fixed per patch.
    """
    import numpy as np

    time_keys = [f"{year}-{month:02d}-01" for year, month in ERA5_CROP_MONTHS]
    crop_year = dataset.sel({time_name: time_keys})
    valid_grid = np.ones(
        (dataset.sizes[latitude_name], dataset.sizes[longitude_name]),
        dtype=bool,
    )
    for variable_name in variable_names.values():
        values = crop_year[variable_name].transpose(
            time_name,
            latitude_name,
            longitude_name,
        ).values
        valid_grid &= np.isfinite(values).all(axis=0)
    if not valid_grid.any():
        raise ValueError("ERA5-Land crop-year request contains no complete valid land cells.")

    latitudes = dataset[latitude_name].values.astype(float)
    longitudes = dataset[longitude_name].values.astype(float)
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    indices: dict[int, tuple[int, int]] = {}
    distances: dict[int, float] = {}
    for location in locations:
        longitude_scale = math.cos(math.radians(location.lat))
        squared_distance = (latitude_grid - location.lat) ** 2 + (
            (longitude_grid - location.lon) * longitude_scale
        ) ** 2
        squared_distance = np.where(valid_grid, squared_distance, np.inf)
        row, column = np.unravel_index(np.argmin(squared_distance), squared_distance.shape)
        distance = math.sqrt(float(squared_distance[row, column]))
        if not math.isfinite(distance) or distance > 0.25:
            raise ValueError(
                "No complete ERA5-Land grid cell within 0.25 degrees for PASTIS patch "
                f"{location.patch_id}; nearest valid distance is {distance:.3f} degrees."
            )
        indices[location.patch_id] = (int(row), int(column))
        distances[location.patch_id] = distance
    return indices, distances


def _extract_era5(
    locations: list[PatchLocation],
    netcdf_path: Path,
    output_path: Path,
    manifest_path: Path,
    source_version: str,
    overwrite: bool,
) -> None:
    if output_path.is_file() and not overwrite:
        print(f"Reuse existing raw ERA5-Land table: {output_path}")
        return
    xarray = _require_xarray()
    dataset = xarray.open_dataset(netcdf_path)
    try:
        latitude_name = _find_name(dataset.coords, ("latitude", "lat"), "latitude coordinate")
        longitude_name = _find_name(dataset.coords, ("longitude", "lon"), "longitude coordinate")
        time_name = _find_name(dataset.coords, ("valid_time", "time"), "time coordinate")
        variable_names = {
            "t2m_c": _find_name(dataset.data_vars, ("t2m", "2m_temperature"), "2m temperature"),
            "tp_mm": _find_name(dataset.data_vars, ("tp", "total_precipitation"), "total precipitation"),
            "ssrd_mj_m2": _find_name(
                dataset.data_vars,
                ("ssrd", "surface_solar_radiation_downwards"),
                "surface solar radiation downwards",
            ),
            "swvl1": _find_name(
                dataset.data_vars,
                ("swvl1", "volumetric_soil_water_layer_1"),
                "volumetric soil water layer 1",
            ),
        }
        cell_indices, cell_distances = _nearest_valid_era5_cells(
            dataset,
            locations=locations,
            latitude_name=latitude_name,
            longitude_name=longitude_name,
            time_name=time_name,
            variable_names=variable_names,
        )
        records: list[dict[str, object]] = []
        for location in locations:
            latitude_index, longitude_index = cell_indices[location.patch_id]
            point = dataset.isel(
                {
                    latitude_name: latitude_index,
                    longitude_name: longitude_index,
                }
            )
            era5_lat = _scalar(point[latitude_name].values)
            era5_lon = _scalar(point[longitude_name].values)
            for year, month in ERA5_CROP_MONTHS:
                time_key = f"{year}-{month:02d}-01"
                cell = point.sel({time_name: time_key})
                days = calendar.monthrange(year, month)[1]
                t2m = _scalar(cell[variable_names["t2m_c"]].values)
                tp = _scalar(cell[variable_names["tp_mm"]].values)
                ssrd = _scalar(cell[variable_names["ssrd_mj_m2"]].values)
                swvl1 = _scalar(cell[variable_names["swvl1"]].values)
                valid = all(math.isfinite(value) for value in (t2m, tp, ssrd, swvl1))
                records.append(
                    {
                        "patch_id": location.patch_id,
                        "year": year,
                        "month": month,
                        "lon": f"{location.lon:.8f}",
                        "lat": f"{location.lat:.8f}",
                        "era5_lon": f"{era5_lon:.8f}",
                        "era5_lat": f"{era5_lat:.8f}",
                        "t2m_c": t2m - 273.15 if valid else 0.0,
                        # CDS monthly averaged reanalysis stores accumulated fields as mean daily totals.
                        "tp_mm": tp * 1000.0 * days if valid else 0.0,
                        "ssrd_mj_m2": ssrd / 1_000_000.0 * days if valid else 0.0,
                        "swvl1": swvl1 if valid else 0.0,
                        "valid": str(valid).lower(),
                        "confidence": 1.0 if valid else 0.0,
                        "source_version": source_version,
                    }
                )
    finally:
        dataset.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patch_id",
                "year",
                "month",
                "lon",
                "lat",
                "era5_lon",
                "era5_lat",
                *CLIMATE_FEATURES,
                "valid",
                "confidence",
                "source_version",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    _write_json(
        manifest_path,
        {
            "dataset": ERA5_DATASET,
            "doi": "10.24381/cds.68d2bb30",
            "netcdf": str(netcdf_path),
            "netcdf_sha256": _sha256(netcdf_path),
            "output": str(output_path),
            "crop_months": [f"{year}-{month:02d}" for year, month in ERA5_CROP_MONTHS],
            "variables": variable_names,
            "conversions": {
                "t2m_c": "Kelvin - 273.15",
                "tp_mm": "monthly mean daily total (m) * 1000 * days_in_month",
                "ssrd_mj_m2": "monthly mean daily total (J/m2) / 1e6 * days_in_month",
                "swvl1": "volumetric soil water content; unchanged",
            },
            "record_count": len(records),
            "spatial_sampling": {
                "method": "nearest complete ERA5-Land crop-year land cell",
                "max_distance_degrees": max(cell_distances.values()),
                "distance_limit_degrees": 0.25,
            },
        },
    )
    print(f"ERA5-Land sample table: {len(records)} rows -> {output_path}")


def _require_soil_dependencies():
    try:
        import numpy
        import requests
        from pyproj import Transformer
        from rasterio.io import MemoryFile
    except ImportError as exc:
        raise ImportError(
            "SoilGrids automation needs requests, pyproj, rasterio, and numpy. "
            "Install project requirements before running it."
        ) from exc
    return numpy, requests, Transformer, MemoryFile


def _group_by_tile(locations: Iterable[PatchLocation]) -> dict[str, list[PatchLocation]]:
    result: dict[str, list[PatchLocation]] = {}
    for location in locations:
        result.setdefault(location.tile, []).append(location)
    return result


def _soil_bbox(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    x_values, y_values = zip(*points)
    padding_meters = 500.0
    return (
        min(x_values) - padding_meters,
        min(y_values) - padding_meters,
        max(x_values) + padding_meters,
        max(y_values) + padding_meters,
    )


def _soil_wcs_request(
    session,
    *,
    property_name: str,
    depth: str,
    quantile: str,
    bbox: tuple[float, float, float, float],
    destination: Path,
    timeout: int,
    retries: int,
    overwrite: bool,
) -> None:
    if destination.is_file() and not overwrite:
        return
    min_x, min_y, max_x, max_y = bbox
    params = [
        ("map", f"/map/{property_name}.map"),
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", f"{property_name}_{depth}cm_{quantile}"),
        ("FORMAT", "GEOTIFF_INT16"),
        ("SUBSET", f"X({min_x:.3f},{max_x:.3f})"),
        ("SUBSET", f"Y({min_y:.3f},{max_y:.3f})"),
        ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/152160"),
        ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/152160"),
    ]
    url = "https://maps.isric.org/mapserv"
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "tiff" not in content_type and not response.content.startswith((b"II*", b"MM\x00*")):
                snippet = response.text[:300].replace("\n", " ")
                raise RuntimeError(f"SoilGrids WCS returned non-GeoTIFF content: {snippet}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(response.content)
            temporary.replace(destination)
            return
        except Exception as exc:  # Network/WCS failures are retryable and leave no final artifact.
            error = exc
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(
        f"SoilGrids WCS failed after {retries} attempts for {property_name}/{depth}/{quantile}."
    ) from error


def _sample_geotiff(path: Path, points: list[tuple[float, float]]) -> list[float | None]:
    numpy, _, _, MemoryFile = _require_soil_dependencies()
    with MemoryFile(path.read_bytes()) as memory_file:
        with memory_file.open() as dataset:
            result: list[float | None] = []
            for value in dataset.sample(points, indexes=1, masked=True):
                scalar = value[0]
                if numpy.ma.is_masked(scalar) or not numpy.isfinite(scalar):
                    result.append(None)
                else:
                    result.append(float(scalar))
            return result


def _download_and_extract_soil(
    locations: list[PatchLocation],
    raw_dir: Path,
    output_path: Path,
    manifest_path: Path,
    source_version: str,
    timeout: int,
    retries: int,
    overwrite_output: bool,
    redownload: bool,
) -> None:
    if output_path.is_file() and not overwrite_output:
        print(f"Reuse existing raw SoilGrids table: {output_path}")
        return
    _, requests, Transformer, _ = _require_soil_dependencies()
    # SoilGrids WCS calls its native Homolosine grid EPSG:152160, a pseudo code.
    # PROJ/GDAL expose the identical projection locally as ESRI:54052.
    transformer = Transformer.from_crs("EPSG:4326", "ESRI:54052", always_xy=True)
    locations_by_tile = _group_by_tile(locations)
    samples: dict[tuple[int, str, str, str], float | None] = {}
    session = requests.Session()
    try:
        for tile, group in sorted(locations_by_tile.items()):
            points = [transformer.transform(location.lon, location.lat) for location in group]
            bbox = _soil_bbox(points)
            for model_field, (source_field, _) in SOIL_SOURCE_FIELDS.items():
                del model_field
                for depth in SOIL_DEPTHS:
                    for quantile_key, quantile_name in SOIL_QUANTILES.items():
                        destination = raw_dir / "soilgrids" / tile / f"{source_field}_{depth}_{quantile_key}.tif"
                        _soil_wcs_request(
                            session,
                            property_name=source_field,
                            depth=depth,
                            quantile=quantile_name,
                            bbox=bbox,
                            destination=destination,
                            timeout=timeout,
                            retries=retries,
                            overwrite=redownload,
                        )
                        values = _sample_geotiff(destination, points)
                        for location, value in zip(group, values):
                            samples[(location.patch_id, source_field, depth, quantile_key)] = value
    finally:
        session.close()

    records: list[dict[str, object]] = []
    for location in locations:
        for depth in SOIL_DEPTHS:
            values: dict[str, float] = {}
            valid = True
            for output_field, (source_field, scale) in SOIL_SOURCE_FIELDS.items():
                central = samples.get((location.patch_id, source_field, depth, "q50"))
                q05 = samples.get((location.patch_id, source_field, depth, "q05"))
                q95 = samples.get((location.patch_id, source_field, depth, "q95"))
                if central is None or q05 is None or q95 is None:
                    valid = False
                    values[output_field] = 0.0
                    values[f"{output_field}_q05"] = 0.0
                    values[f"{output_field}_q95"] = 0.0
                else:
                    values[output_field] = central / scale
                    values[f"{output_field}_q05"] = q05 / scale
                    values[f"{output_field}_q95"] = q95 / scale
            record: dict[str, object] = {
                "patch_id": location.patch_id,
                "depth_cm": depth,
                **values,
                "valid": str(valid).lower(),
                "source_version": source_version,
            }
            records.append(record)
    records.sort(key=lambda item: (int(item["patch_id"]), SOIL_DEPTHS.index(str(item["depth_cm"]))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["patch_id", "depth_cm", *SOIL_FEATURES]
    fields.extend(f"{field}_q05" for field in SOIL_FEATURES)
    fields.extend(f"{field}_q95" for field in SOIL_FEATURES)
    fields.extend(("valid", "source_version"))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    tiffs = sorted((raw_dir / "soilgrids").rglob("*.tif"))
    _write_json(
        manifest_path,
        {
            "dataset": "SoilGrids WCS",
            "output": str(output_path),
            "properties": {
                output_field: {"source": source_field, "conversion_divisor": scale}
                for output_field, (source_field, scale) in SOIL_SOURCE_FIELDS.items()
            },
            "depths_cm": list(SOIL_DEPTHS),
            "quantiles": SOIL_QUANTILES,
            "spatial_sampling": "nearest SoilGrids 250 m WCS pixel at PASTIS patch centre",
            "downloaded_geotiffs": [
                {"path": str(path), "sha256": _sha256(path)} for path in tiffs
            ],
            "record_count": len(records),
        },
    )
    print(f"SoilGrids sample table: {len(records)} rows -> {output_path}")


def main() -> None:
    args = parse_args()
    locations = _ensure_locations(args.metadata, args.locations, args.overwrite)
    raw_dir = Path(args.raw_dir)
    climate_raw = Path(args.climate_raw)
    soil_raw = Path(args.soil_raw)
    if not args.skip_era5:
        era5_netcdf = raw_dir / "era5_land_pastis_cropyear_v1.nc"
        _download_era5(locations, era5_netcdf, args.redownload)
        _extract_era5(
            locations,
            era5_netcdf,
            climate_raw,
            raw_dir / "era5_land_manifest.json",
            args.source_version,
            args.overwrite,
        )
    if not args.skip_soil:
        _download_and_extract_soil(
            locations,
            raw_dir,
            soil_raw,
            raw_dir / "soilgrids_manifest.json",
            args.source_version,
            timeout=int(args.soil_timeout),
            retries=int(args.soil_retries),
            overwrite_output=args.overwrite,
            redownload=args.redownload,
        )
    if args.skip_freeze:
        return
    if not climate_raw.is_file() or not soil_raw.is_file():
        raise FileNotFoundError(
            "Freezing needs both raw tables. Omit --skip-era5/--skip-soil or provide existing raw tables."
        )
    patch_folds = _metadata_folds(args.metadata)
    prepare_climate(
        str(climate_raw),
        args.climate_output,
        args.climate_stats_output,
        patch_folds,
        {1, 2, 3},
        allow_incomplete=False,
        source_version=args.source_version,
    )
    prepare_soil(
        str(soil_raw),
        args.soil_output,
        args.soil_stats_output,
        patch_folds,
        {1, 2, 3},
        allow_incomplete=False,
        source_version=args.source_version,
    )
    print("M2/M3 environmental priors are ready for the CA-HPI training overlays.")


if __name__ == "__main__":
    main()
