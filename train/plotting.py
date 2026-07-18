from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HISTORY_FIELDS = ("epoch", "train_loss", "val_loss", "val_miou", "val_acc", "val_f1")
PLOT_SPECS = (
    ("train_loss", "Train Loss", "Loss", None),
    ("val_loss", "Validation Loss", "Loss", None),
    ("val_miou", "Validation mIoU", "mIoU", (0.0, 1.0)),
    ("val_acc", "Validation Accuracy", "Accuracy", (0.0, 1.0)),
    ("val_f1", "Validation Macro F1", "Macro F1", (0.0, 1.0)),
)


def save_training_history(history: list[dict[str, float]], output_dir: str | Path) -> None:
    """Persist raw epoch metrics and render unsmoothed plots without TensorBoard."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2, allow_nan=False)

    with (output_path / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    if not history:
        return

    rendered: list[Image.Image] = []
    for key, title, y_label, y_limits in PLOT_SPECS:
        chart = _render_plot(history, key, title, y_label, y_limits)
        chart.save(output_path / f"{key}.png")
        rendered.append(chart)
    _render_overview(rendered).save(output_path / "training_curves.png")


def save_prior_diagnostics_history(
    history: list[dict[str, float | int]],
    output_dir: str | Path,
) -> None:
    """Persist dynamic CA-HPI scalar diagnostics without changing base curves."""

    if not history:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "prior_diagnostics_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2, allow_nan=False)

    metric_fields = sorted(
        {
            key
            for row in history
            for key in row
            if key != "epoch"
        }
    )
    with (output_path / "prior_diagnostics_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", *metric_fields))
        writer.writeheader()
        writer.writerows(history)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_names = (
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_plot(
    history: list[dict[str, float]],
    key: str,
    title: str,
    y_label: str,
    y_limits: tuple[float, float] | None,
    width: int = 1200,
    height: int = 700,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(32, bold=True)
    axis_font = _font(21)
    tick_font = _font(18)

    left, right, top, bottom = 112, width - 42, 82, height - 86
    plot_width = right - left
    plot_height = bottom - top
    epochs = [int(row["epoch"]) for row in history]
    values = [float(row[key]) for row in history]

    finite_values = [value for value in values if math.isfinite(value)]
    if y_limits is None:
        y_min = 0.0
        upper = max(finite_values, default=1.0)
        y_max = max(upper * 1.08, 1e-6)
    else:
        y_min, y_max = y_limits

    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (title_box[2] - title_box[0])) / 2, 22),
        title,
        fill=(25, 25, 25),
        font=title_font,
    )

    for tick_index in range(6):
        ratio = tick_index / 5
        y = bottom - round(ratio * plot_height)
        value = y_min + ratio * (y_max - y_min)
        draw.line((left, y, right, y), fill=(222, 226, 232), width=1)
        label = f"{value:.2f}" if y_max <= 1.1 else f"{value:.3g}"
        label_box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (left - 14 - (label_box[2] - label_box[0]), y - 10),
            label,
            fill=(65, 65, 65),
            font=tick_font,
        )

    draw.line((left, top, left, bottom), fill=(35, 35, 35), width=2)
    draw.line((left, bottom, right, bottom), fill=(35, 35, 35), width=2)

    tick_indices = _x_tick_indices(len(epochs), max_ticks=10)
    for index in tick_indices:
        x = _x_position(index, len(epochs), left, plot_width)
        draw.line((x, bottom, x, bottom + 7), fill=(35, 35, 35), width=2)
        label = str(epochs[index])
        label_box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (x - (label_box[2] - label_box[0]) / 2, bottom + 12),
            label,
            fill=(65, 65, 65),
            font=tick_font,
        )

    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        ratio = min(1.0, max(0.0, (value - y_min) / max(y_max - y_min, 1e-12)))
        points.append(
            (
                _x_position(index, len(values), left, plot_width),
                bottom - round(ratio * plot_height),
            )
        )
    if len(points) >= 2:
        draw.line(points, fill=(31, 119, 180), width=4, joint="curve")
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(31, 119, 180), outline="white")

    x_label = "Epoch"
    x_label_box = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(
        ((width - (x_label_box[2] - x_label_box[0])) / 2, height - 36),
        x_label,
        fill=(35, 35, 35),
        font=axis_font,
    )
    draw.text((16, top - 4), y_label, fill=(35, 35, 35), font=axis_font)
    return canvas


def _x_position(index: int, count: int, left: int, plot_width: int) -> int:
    if count <= 1:
        return left + plot_width // 2
    return left + round(index * plot_width / (count - 1))


def _x_tick_indices(count: int, max_ticks: int) -> list[int]:
    if count <= max_ticks:
        return list(range(count))
    indices = {
        round(tick * (count - 1) / (max_ticks - 1)) for tick in range(max_ticks)
    }
    return sorted(indices)


def _render_overview(charts: list[Image.Image]) -> Image.Image:
    panel_width, panel_height = 800, 466
    gap = 18
    canvas = Image.new(
        "RGB",
        (3 * panel_width + 4 * gap, 2 * panel_height + 3 * gap),
        (235, 238, 242),
    )
    for index, chart in enumerate(charts):
        row, column = divmod(index, 3)
        panel = chart.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        canvas.paste(
            panel,
            (gap + column * (panel_width + gap), gap + row * (panel_height + gap)),
        )

    draw = ImageDraw.Draw(canvas)
    x = gap + 2 * (panel_width + gap)
    y = gap + panel_height + gap
    draw.rounded_rectangle(
        (x, y, x + panel_width, y + panel_height),
        radius=14,
        fill="white",
        outline=(205, 210, 218),
        width=2,
    )
    draw.text(
        (x + 58, y + 105),
        "Raw epoch values\n(no smoothing)",
        fill=(25, 25, 25),
        font=_font(34, bold=True),
        spacing=10,
    )
    draw.text(
        (x + 58, y + 250),
        "mIoU / Accuracy / F1\ny-axis fixed to [0, 1]",
        fill=(55, 55, 55),
        font=_font(27),
        spacing=8,
    )
    return canvas
