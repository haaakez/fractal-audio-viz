#!/usr/bin/env python3
"""Measure fractal field throughput without decoding audio or encoding video."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import visualizer


def main() -> None:
    np = visualizer._require_numpy()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--stage",
        choices=("field", "compositor"),
        default="field",
        help="benchmark deep field generation or the native atlas output pass",
    )
    parser.add_argument("--zoom", default="1e100")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--x-center", default=visualizer.DEFAULT_X_CENTER)
    parser.add_argument("--y-center", default=visualizer.DEFAULT_Y_CENTER)
    parser.add_argument("--renderer", choices=("auto", "native", "python"), default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--series-order", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--series-block", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height, args.iterations, args.repeat) <= 0:
        raise SystemExit("width, height, iterations, and repeat must be positive")
    if args.threads < 0:
        raise SystemExit("threads cannot be negative")
    if not 2 <= args.series_block <= 4096:
        raise SystemExit("series-block must be between 2 and 4096")

    log_zoom = visualizer._zoom_log(args.zoom)
    if args.stage == "compositor":
        if args.renderer == "python":
            raise SystemExit("the compositor benchmark requires --renderer native or auto")
        library = visualizer._get_native_library()
        if library is None or not hasattr(library, "fractal_atlas_colourise"):
            raise SystemExit("the native atlas compositor is unavailable; run make first")
        parent = np.random.default_rng(1234).random(
            (args.height, args.width), dtype=np.float32
        ) * float(args.iterations)
        child = np.random.default_rng(5678).random(
            (args.height, args.width), dtype=np.float32
        ) * float(args.iterations)
        output = np.empty((args.height, args.width, 3), dtype=np.uint8)
        call_args = (
            parent.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            args.width,
            args.height,
            args.iterations,
            child.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            args.width,
            args.height,
            args.iterations,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            args.width,
            args.height,
            1.5,
            0.65,
            args.iterations,
            0.3,
            0.4,
            0.7,
            0.5,
            args.threads,
        )
        library.fractal_atlas_colourise(*call_args)
        timings = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            status = library.fractal_atlas_colourise(*call_args)
            if status != 0:
                message = library.fractal_last_error() or b"unknown native compositor error"
                raise SystemExit(message.decode("utf-8", errors="replace"))
            timings.append(time.perf_counter() - started)
        best = min(timings)
        result = {
            "stage": args.stage,
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
            "repeat": args.repeat,
            "seconds": timings,
            "best_seconds": best,
            "pixels_per_second": args.width * args.height / best,
            "rgb_megabytes_per_second": args.width * args.height * 3 / best / 1.0e6,
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"atlas compositor {args.width}x{args.height}: "
                f"best {best:.3f}s, {result['pixels_per_second']:.1f} pixels/s"
            )
        return

    native_reference = None
    native_library = None
    if args.renderer != "python" and log_zoom >= 12.0:
        native_library, native_reference = visualizer._create_native_reference(
            args.x_center,
            args.y_center,
            args.iterations,
            log_zoom,
            args.series_order,
            12.0,
        )
    timings = []
    try:
        for _ in range(args.repeat):
            started = time.perf_counter()
            field = visualizer.render_fractal(
                args.width,
                args.height,
                log_zoom,
                args.x_center,
                args.y_center,
                args.iterations,
                args.renderer,
                args.threads,
                native_reference,
                args.series_order,
                args.series_block,
            )
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
        result = {
            "stage": args.stage,
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
            "zoom": args.zoom,
            "iterations": args.iterations,
            "x_center": args.x_center,
            "y_center": args.y_center,
            "repeat": args.repeat,
            "seconds": timings,
            "best_seconds": min(timings),
            "pixels_per_second": args.width * args.height / min(timings),
            "finite_fraction": float(np.isfinite(field).mean()),
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
