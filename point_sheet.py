#!/usr/bin/env python3
"""Render a small contact sheet for the curated deep-zoom catalogue."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import visualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make a labelled preview sheet of a formula's bundled points."
    )
    parser.add_argument(
        "--formula",
        choices=visualizer.FORMULA_CHOICES,
        default="mandelbrot",
        help="formula catalogue to preview",
    )
    parser.add_argument("--output", type=Path, default=Path("deep-zoom-points.png"))
    parser.add_argument("--zoom-log", type=float, default=8.0, help="preview log10 zoom")
    parser.add_argument("--width", type=int, default=240, help="tile width")
    parser.add_argument("--height", type=int, default=180, help="tile height")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--palette",
        choices=visualizer.PALETTE_CHOICES,
        default="aurora",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        not math.isfinite(args.zoom_log)
        or args.zoom_log < 0.0
        or args.zoom_log > visualizer.MAX_LOG10_ZOOM
        or args.width < 16
        or args.height < 16
        or args.columns < 1
        or args.threads < 0
    ):
        raise SystemExit(
            "zoom-log must be finite and non-negative; dimensions, columns, and threads "
            "must be positive/non-negative"
        )
    try:
        args.threads = visualizer._validate_thread_count(args.threads, "threads")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.formula != "mandelbrot" and args.zoom_log > 300.0:
        raise SystemExit("alternate formula point sheets support zooms only through 1e300")
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - dependency error
        raise SystemExit("Pillow is required for the point sheet") from exc

    points = visualizer.FORMULA_POINT_CATALOGUES[args.formula]
    label_height = 30
    rows = (len(points) + args.columns - 1) // args.columns
    output_path = args.output.expanduser()
    try:
        visualizer._reject_final_symlink(output_path, "output")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output_path = visualizer._absolute_path(output_path)
    if output_path.exists() and output_path.is_dir():
        raise SystemExit(f"output path is a directory: {output_path}")
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise SystemExit(f"output parent is not a directory: {output_path.parent}")
    try:
        visualizer._validate_dimensions(
            args.columns * args.width,
            rows * (args.height + label_height),
            "point sheet",
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    sheet = Image.new(
        "RGB",
        (args.columns * args.width, rows * (args.height + label_height)),
        (12, 12, 18),
    )
    draw = ImageDraw.Draw(sheet)
    native_library = visualizer._get_native_library()
    for index, point in enumerate(points):
        max_iter = visualizer.max_iterations(args.zoom_log, 384, 500, 20000)
        reference = None
        if args.formula == "mandelbrot" and native_library is not None and args.zoom_log >= 12.0:
            native_library, reference = visualizer._create_native_reference(
                point.x,
                point.y,
                max_iter,
                args.zoom_log,
                3,
                args.zoom_log,
            )
        try:
            julia_constant = point.julia_c or visualizer.DEFAULT_JULIA_C
            field = visualizer.render_fractal(
                args.width,
                args.height,
                args.zoom_log,
                point.x,
                point.y,
                max_iter,
                "auto",
                args.threads,
                reference,
                formula=args.formula,
                julia_constant=julia_constant,
            )
        finally:
            if reference is not None:
                native_library.fractal_destroy_reference(reference)
        rgb = visualizer._colourise_view(
            field,
            max_iter,
            0.0,
            0.6,
            0.7,
            native_library,
            args.threads,
            args.palette,
            0.5,
        )
        x = (index % args.columns) * args.width
        y = (index // args.columns) * (args.height + label_height)
        sheet.paste(Image.fromarray(rgb, mode="RGB"), (x, y))
        draw.rectangle((x, y + args.height, x + args.width, y + args.height + label_height), fill=(12, 12, 18))
        draw.text((x + 5, y + args.height + 7), point.slug, fill=(235, 235, 245))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = visualizer._reserved_temporary_sibling(output_path, "writing")
    try:
        sheet.save(temporary_output)
        os.replace(temporary_output, output_path)
    finally:
        if temporary_output.exists():
            try:
                temporary_output.unlink()
            except OSError:
                pass
    print(f"Point sheet ready: {output_path}")


if __name__ == "__main__":
    main()
