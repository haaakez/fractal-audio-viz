#!/usr/bin/env python3
"""Render a small contact sheet for the curated deep-zoom catalogue."""

from __future__ import annotations

import argparse
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
        choices=("aurora", "fire", "ocean", "neon", "sunset", "mono"),
        default="aurora",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.zoom_log < 0.0 or args.width < 16 or args.height < 16 or args.columns < 1:
        raise SystemExit("zoom-log must be non-negative; dimensions and columns must be positive")
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - dependency error
        raise SystemExit("Pillow is required for the point sheet") from exc

    points = visualizer.FORMULA_POINT_CATALOGUES[args.formula]
    label_height = 30
    rows = (len(points) + args.columns - 1) // args.columns
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Point sheet ready: {args.output}")


if __name__ == "__main__":
    main()
