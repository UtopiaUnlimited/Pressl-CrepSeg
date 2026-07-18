from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MetricSpec:
    tensorboard_tag: str
    title: str
    y_label: str
    unit_interval: bool = False


METRICS = {
    "train_loss": MetricSpec("loss/train", "Train Loss", "Loss"),
    "val_loss": MetricSpec("loss/val", "Validation Loss", "Loss"),
    "val_miou": MetricSpec(
        "metrics/val_miou", "Validation mIoU", "mIoU", unit_interval=True
    ),
    "val_acc": MetricSpec(
        "metrics/val_acc", "Validation Accuracy", "Accuracy", unit_interval=True
    ),
    "val_f1": MetricSpec(
        "metrics/val_f1", "Validation Macro F1", "Macro F1", unit_interval=True
    ),
    "lr": MetricSpec("lr", "Learning Rate", "Learning rate"),
}


SCHEME_RUNS = {
    "1": "galileo_single_layer_dpt_shared_paper_input_bs16_rerun_seed42_cached",
    "2": "galileo_multi_layer_dpt_shared_paper_input_bs16_rerun_seed42_cached",
    "3": "galileo_upernet_shared_paper_input_bs16_cached",
    "4": "galileo_adapted_dpt_native_skip_paper_input_bs16_seed42_cached",
    "5": "galileo_3d_aware_dpt_late_fusion_seed42",
}


SCHEME_LABELS = {
    "1": "Scheme 1 - Final-layer baseline",
    "2": "Scheme 2 - Multi-layer baseline",
    "3": "Scheme 3 - UPerNet",
    "4": "Scheme 4 - Adapted DPT",
    "5": "Scheme 5 - 3D-Aware DPT",
}


COLORS = (
    (36, 99, 161),
    (230, 85, 13),
    (0, 145, 114),
    (204, 121, 167),
    (213, 94, 0),
    (86, 180, 233),
    (240, 228, 66),
    (0, 158, 115),
    (204, 121, 167),
    (0, 114, 178),
)


@dataclass
class RunData:
    label: str
    path: Path
    values: dict[str, list[tuple[int, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot selected epoch metrics from TensorBoard events or "
            "training_history.json files."
        )
    )
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    parser.add_argument(
        "--runs",
        nargs="+",
        help=(
            "Runs to compare. Use scheme aliases 1..5, a directory name under "
            "--logs-root, a full path, or LABEL=RUN."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRICS),
        default=["train_loss", "val_loss", "val_miou"],
    )
    parser.add_argument("--min-epoch", type=int, default=1)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Trailing moving-average window. Use 1 for raw epoch values.",
    )
    parser.add_argument(
        "--y-limit",
        action="append",
        default=[],
        metavar="METRIC=MIN:MAX",
        help="Override one metric's y-axis, for example val_miou=0.3:0.7.",
    )
    parser.add_argument(
        "--auto-y",
        action="store_true",
        help="Auto-scale mIoU, accuracy and F1 instead of fixing them to [0, 1].",
    )
    parser.add_argument("--columns", type=int, choices=[1, 2], default=1)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--panel-height", type=int, default=680)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "selected_training_curves.png"),
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List usable run directories and exit.",
    )
    return parser.parse_args()


def discover_runs(logs_root: Path) -> list[Path]:
    if not logs_root.exists():
        return []
    discovered = []
    for path in sorted(logs_root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        has_history = (path / "training_history.json").is_file()
        has_events = any(path.glob("events.out.tfevents.*"))
        if has_history or has_events:
            discovered.append(path)
    return discovered


def resolve_run(spec: str, logs_root: Path) -> tuple[str, Path]:
    explicit_label = None
    run_spec = spec
    if "=" in spec:
        explicit_label, run_spec = spec.split("=", maxsplit=1)
        if not explicit_label or not run_spec:
            raise ValueError(f"Invalid LABEL=RUN value: {spec}")

    alias = run_spec.lower()
    if alias.startswith("scheme"):
        alias = alias.removeprefix("scheme")
    elif alias.startswith("s") and alias[1:] in SCHEME_RUNS:
        alias = alias[1:]

    if alias in SCHEME_RUNS:
        path = logs_root / SCHEME_RUNS[alias]
        label = explicit_label or SCHEME_LABELS[alias]
    else:
        candidate = Path(run_spec).expanduser()
        path = candidate if candidate.is_absolute() else logs_root / candidate
        label = explicit_label or path.name

    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {path}. Use --list-runs to inspect logs."
        )
    return label, path


def load_run(
    label: str,
    path: Path,
    metrics: Iterable[str],
    min_epoch: int,
    max_epoch: int | None,
    smoothing_window: int,
) -> RunData:
    requested = tuple(metrics)
    values = _load_tensorboard(path, requested)
    history_values = _load_history(path, requested)
    for metric in requested:
        if not values.get(metric):
            values[metric] = history_values.get(metric, [])

        filtered = [
            (epoch, value)
            for epoch, value in values.get(metric, [])
            if epoch >= min_epoch and (max_epoch is None or epoch <= max_epoch)
        ]
        values[metric] = _moving_average(filtered, smoothing_window)
    return RunData(label=label, path=path, values=values)


def _load_tensorboard(
    run_dir: Path, metrics: Iterable[str]
) -> dict[str, list[tuple[int, float]]]:
    event_files = tuple(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {}

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", ()))
    loaded: dict[str, list[tuple[int, float]]] = {}
    for metric in metrics:
        tag = METRICS[metric].tensorboard_tag
        if tag not in available_tags:
            continue
        latest_by_epoch: dict[int, tuple[float, float]] = {}
        for event in accumulator.Scalars(tag):
            epoch = int(event.step)
            previous = latest_by_epoch.get(epoch)
            if previous is None or event.wall_time >= previous[0]:
                latest_by_epoch[epoch] = (event.wall_time, float(event.value))
        loaded[metric] = [
            (epoch, latest_by_epoch[epoch][1]) for epoch in sorted(latest_by_epoch)
        ]
    return loaded


def _load_history(
    run_dir: Path, metrics: Iterable[str]
) -> dict[str, list[tuple[int, float]]]:
    history_path = run_dir / "training_history.json"
    if not history_path.is_file():
        return {}
    with history_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    loaded: dict[str, list[tuple[int, float]]] = {metric: [] for metric in metrics}
    for row in rows:
        epoch = int(row["epoch"])
        for metric in metrics:
            if metric in row:
                loaded[metric].append((epoch, float(row[metric])))
    return loaded


def _moving_average(
    points: list[tuple[int, float]], window: int
) -> list[tuple[int, float]]:
    if window <= 1:
        return points
    smoothed = []
    for index, (epoch, _) in enumerate(points):
        start = max(0, index - window + 1)
        finite = [value for _, value in points[start : index + 1] if math.isfinite(value)]
        value = sum(finite) / len(finite) if finite else float("nan")
        smoothed.append((epoch, value))
    return smoothed


def parse_y_limits(values: Iterable[str]) -> dict[str, tuple[float, float]]:
    parsed = {}
    for value in values:
        try:
            metric, limits = value.split("=", maxsplit=1)
            lower_text, upper_text = limits.split(":", maxsplit=1)
            lower, upper = float(lower_text), float(upper_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid --y-limit '{value}'; expected METRIC=MIN:MAX."
            ) from error
        if metric not in METRICS:
            raise ValueError(f"Unknown metric in --y-limit: {metric}")
        if not lower < upper:
            raise ValueError(f"Y-axis minimum must be below maximum: {value}")
        parsed[metric] = (lower, upper)
    return parsed


def render(
    runs: list[RunData],
    metrics: list[str],
    output_path: Path,
    width: int,
    panel_height: int,
    columns: int,
    title: str | None,
    y_limits: dict[str, tuple[float, float]],
    auto_y: bool,
) -> None:
    panel_width = width // columns
    title_height = 90 if title else 20
    rows = math.ceil(len(metrics) / columns)
    canvas = Image.new(
        "RGB", (width, title_height + rows * panel_height), (244, 246, 249)
    )
    if title:
        draw = ImageDraw.Draw(canvas)
        font = _font(34, bold=True)
        bounds = draw.textbbox((0, 0), title, font=font)
        draw.text(
            ((width - (bounds[2] - bounds[0])) / 2, 24),
            title,
            fill=(25, 32, 45),
            font=font,
        )

    for index, metric in enumerate(metrics):
        row, column = divmod(index, columns)
        panel = _render_metric_panel(
            runs,
            metric,
            panel_width - 20,
            panel_height - 20,
            y_limits.get(metric),
            auto_y,
        )
        canvas.paste(panel, (column * panel_width + 10, title_height + row * panel_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _render_metric_panel(
    runs: list[RunData],
    metric: str,
    width: int,
    height: int,
    y_limit: tuple[float, float] | None,
    auto_y: bool,
) -> Image.Image:
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    title_font = _font(29, bold=True)
    axis_font = _font(20)
    tick_font = _font(17)
    legend_font = _font(17)
    spec = METRICS[metric]

    all_points = [
        point for run in runs for point in run.values.get(metric, []) if math.isfinite(point[1])
    ]
    if not all_points:
        draw.text(
            (40, 40),
            f"{spec.title}: no data",
            fill=(120, 30, 30),
            font=title_font,
        )
        return panel

    draw.text((38, 22), spec.title, fill=(25, 32, 45), font=title_font)
    legend_rows = _draw_legend(draw, runs, width, y=72, font=legend_font)
    left, right = 105, width - 40
    top = 112 + legend_rows * 30
    bottom = height - 78
    plot_width, plot_height = right - left, bottom - top

    x_min = min(epoch for epoch, _ in all_points)
    x_max = max(epoch for epoch, _ in all_points)
    if x_min == x_max:
        x_max = x_min + 1

    values = [value for _, value in all_points]
    y_min, y_max = _axis_limits(values, spec, y_limit, auto_y)

    for tick in range(6):
        ratio = tick / 5
        y = bottom - round(ratio * plot_height)
        value = y_min + ratio * (y_max - y_min)
        draw.line((left, y, right, y), fill=(224, 228, 234), width=1)
        label = _format_tick(value, y_max - y_min)
        bounds = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (left - 12 - (bounds[2] - bounds[0]), y - 10),
            label,
            fill=(68, 75, 88),
            font=tick_font,
        )

    draw.line((left, top, left, bottom), fill=(45, 52, 64), width=2)
    draw.line((left, bottom, right, bottom), fill=(45, 52, 64), width=2)

    for epoch in _tick_values(x_min, x_max, max_ticks=10):
        x = _scale(epoch, x_min, x_max, left, right)
        draw.line((x, bottom, x, bottom + 7), fill=(45, 52, 64), width=2)
        label = str(epoch)
        bounds = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (x - (bounds[2] - bounds[0]) / 2, bottom + 12),
            label,
            fill=(68, 75, 88),
            font=tick_font,
        )

    for run_index, run in enumerate(runs):
        points = []
        for epoch, value in run.values.get(metric, []):
            if not math.isfinite(value):
                continue
            x = _scale(epoch, x_min, x_max, left, right)
            y = _scale(value, y_min, y_max, bottom, top)
            points.append((x, y))
        color = COLORS[run_index % len(COLORS)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4, joint="curve")
        if points:
            x, y = points[-1]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white")

    draw.text((17, top - 2), spec.y_label, fill=(45, 52, 64), font=axis_font)
    x_label = "Epoch"
    bounds = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(
        ((width - (bounds[2] - bounds[0])) / 2, height - 34),
        x_label,
        fill=(45, 52, 64),
        font=axis_font,
    )
    return panel


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    runs: list[RunData],
    width: int,
    y: int,
    font: ImageFont.ImageFont,
) -> int:
    x = 40
    row = 0
    for index, run in enumerate(runs):
        label = run.label
        bounds = draw.textbbox((0, 0), label, font=font)
        item_width = 38 + (bounds[2] - bounds[0]) + 24
        if x + item_width > width - 30 and x > 40:
            row += 1
            x = 40
        item_y = y + row * 30
        color = COLORS[index % len(COLORS)]
        draw.line((x, item_y + 10, x + 27, item_y + 10), fill=color, width=5)
        draw.text((x + 36, item_y), label, fill=(55, 62, 74), font=font)
        x += item_width
    return row + 1


def _axis_limits(
    values: list[float],
    spec: MetricSpec,
    override: tuple[float, float] | None,
    auto_y: bool,
) -> tuple[float, float]:
    if override is not None:
        return override
    if spec.unit_interval and not auto_y:
        return 0.0, 1.0

    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.1, 0.05)
    else:
        padding = (maximum - minimum) * 0.08
    lower = minimum - padding
    upper = maximum + padding
    if not spec.unit_interval and minimum >= 0:
        lower = max(0.0, lower)
    if spec.unit_interval:
        lower, upper = max(0.0, lower), min(1.0, upper)
    if math.isclose(lower, upper):
        upper = lower + 1e-6
    return lower, upper


def _scale(value: float, minimum: float, maximum: float, start: int, end: int) -> int:
    ratio = (value - minimum) / max(maximum - minimum, 1e-12)
    ratio = min(1.0, max(0.0, ratio))
    return start + round(ratio * (end - start))


def _tick_values(minimum: int, maximum: int, max_ticks: int) -> list[int]:
    if maximum - minimum + 1 <= max_ticks:
        return list(range(minimum, maximum + 1))
    values = {
        round(minimum + tick * (maximum - minimum) / (max_ticks - 1))
        for tick in range(max_ticks)
    }
    return sorted(values)


def _format_tick(value: float, span: float) -> str:
    if span < 0.01:
        return f"{value:.4f}"
    if span < 0.1:
        return f"{value:.3f}"
    if span <= 2.0:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
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


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root).expanduser().resolve()
    if args.list_runs:
        runs = discover_runs(logs_root)
        if not runs:
            print(f"No usable runs found under {logs_root}")
            return
        print(f"Available runs under {logs_root}:")
        for run in runs:
            print(f"  {run.name}")
        return

    if not args.runs:
        raise SystemExit("--runs is required unless --list-runs is used.")
    if args.min_epoch < 1:
        raise SystemExit("--min-epoch must be at least 1.")
    if args.max_epoch is not None and args.max_epoch < args.min_epoch:
        raise SystemExit("--max-epoch must be greater than or equal to --min-epoch.")
    if args.smoothing_window < 1:
        raise SystemExit("--smoothing-window must be at least 1.")
    if args.width < 800 or args.panel_height < 450:
        raise SystemExit("Use --width >= 800 and --panel-height >= 450.")

    y_limits = parse_y_limits(args.y_limit)
    run_data = []
    for run_spec in args.runs:
        label, path = resolve_run(run_spec, logs_root)
        loaded = load_run(
            label,
            path,
            args.metrics,
            args.min_epoch,
            args.max_epoch,
            args.smoothing_window,
        )
        run_data.append(loaded)
        counts = ", ".join(
            f"{metric}={len(loaded.values.get(metric, []))}" for metric in args.metrics
        )
        print(f"loaded {label}: {counts}")

    output_path = Path(args.output).expanduser().resolve()
    render(
        run_data,
        args.metrics,
        output_path,
        args.width,
        args.panel_height,
        args.columns,
        args.title,
        y_limits,
        args.auto_y,
    )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
