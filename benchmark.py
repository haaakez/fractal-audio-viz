#!/usr/bin/env python3
"""Measure fractal field throughput without decoding audio or encoding video."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import visualizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--zoom", default="1e100")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--renderer", choices=("auto", "native", "python"), default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height, args.iterations, args.repeat) <= 0:
        raise SystemExit("width, height, iterations, and repeat must be positive")

    log_zoom = visualizer._zoom_log(args.zoom)
    native_reference = None
    native_library = None
    if args.renderer != "python" and log_zoom >= 12.0:
        native_library, native_reference = visualizer._create_native_reference(
            visualizer.DEFAULT_X_CENTER,
            visualizer.DEFAULT_Y_CENTER,
            args.iterations,
            log_zoom,
            3,
            6.0,
        )
    timings = []
    try:
        for _ in range(args.repeat):
            started = time.perf_counter()
            field = visualizer.render_fractal(
                args.width,
                args.height,
                log_zoom,
                visualizer.DEFAULT_X_CENTER,
                visualizer.DEFAULT_Y_CENTER,
                args.iterations,
                args.renderer,
                args.threads,
                native_reference,
                3,
                256,
            )
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
        result = {
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
            "zoom": args.zoom,
            "iterations": args.iterations,
            "repeat": args.repeat,
            "seconds": timings,
            "best_seconds": min(timings),
            "pixels_per_second": args.width * args.height / min(timings),
            "finite_fraction": float((field == field).mean()),
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"{args.width}x{args.height} at {args.zoom}: "
                f"best {result['best_seconds']:.3f}s, "
                f"{result['pixels_per_second']:.1f} pixels/s"
            )
    finally:
        if native_reference is not None and native_library is not None:
            native_library.fractal_destroy_reference(native_reference)


if __name__ == "__main__":
    main()
