#!/usr/bin/env python3
"""Measure fractal field throughput without decoding audio or encoding video."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path
from typing import Any

import visualizer


def _atlas_sweep(args: argparse.Namespace, np: Any, log_zoom: float) -> None:
    """Render every nested atlas source once, without video or cache I/O."""

    if args.renderer == "python" and log_zoom >= 300.0:
        raise SystemExit("the Python renderer cannot sweep zooms beyond 1e300")
    if args.renderer != "python" and log_zoom >= 12.0:
        native_library = visualizer._get_native_library()
        if native_library is None:
            raise SystemExit("the native renderer is unavailable; run make first")
    else:
        native_library = visualizer._get_native_library() if args.renderer == "auto" else None

    zooms = np.asarray([0.0, log_zoom], dtype=np.float64)
    origin, step, level_count = visualizer._atlas_geometry(zooms, args.keyframe_factor)
    reference = None
    reference_seconds = 0.0
    stats_supported = False
    if native_library is not None and log_zoom >= 12.0:
        reference_iter = visualizer.max_iterations(
            log_zoom,
            args.iteration_base,
            args.iterations_per_decade,
            args.iteration_cap,
        )
        reference_started = time.perf_counter()
        reference_log_zoom = (
            args.reference_zoom_log
            if args.reference_zoom_log is not None
            else log_zoom
        )
        native_library, reference = visualizer._create_native_reference(
            args.x_center,
            args.y_center,
            reference_iter,
            reference_log_zoom,
            args.series_order,
            12.0,
        )
        reference_seconds = time.perf_counter() - reference_started
        stats_supported = (
            args.stats
            and not args.local_references
            and visualizer._native_set_stats_enabled(native_library, True)
        )

    records = []
    try:
        for level in range(level_count + 1):
            tile_log = origin + step * float(level)
            tile_end = min(log_zoom, tile_log + step)
            tile_iter = visualizer.max_iterations(
                tile_end,
                args.iteration_base,
                args.iterations_per_decade,
                args.iteration_cap,
            )
            started = time.perf_counter()
            if args.local_references:
                field = visualizer._atlas_local_reference_field(
                    render_width=args.width,
                    render_height=args.height,
                    log10_zoom=tile_log,
                    x_center=args.x_center,
                    y_center=args.y_center,
                    max_iter=tile_iter,
                    series_order=args.series_order,
                    series_block=args.series_block,
                    renderer=args.renderer,
                    native_threads=args.threads,
                    native_library=native_library,
                )
            else:
                field = None
            if field is None:
                field = visualizer.render_fractal(
                    args.width,
                    args.height,
                    tile_log,
                    args.x_center,
                    args.y_center,
                    tile_iter,
                    args.renderer,
                    args.threads,
                    reference,
                    args.series_order,
                    args.series_block,
                )
            elapsed = time.perf_counter() - started
            record = {
                "level": level,
                "zoom_log10": tile_log,
                "iterations": tile_iter,
                "seconds": elapsed,
                "pixels_per_second": args.width * args.height / max(elapsed, 1.0e-12),
                "finite_fraction": float(np.isfinite(field).mean()),
            }
            if stats_supported:
                record["stats"] = visualizer._native_get_stats(native_library)
            records.append(record)
            del field
            if args.verbose:
                print(
                    f"level {level:4d}/{level_count}: 10^{tile_log:8.3f}, "
                    f"{tile_iter:6d} iterations, {elapsed:8.3f}s",
                    flush=True,
                )
    finally:
        if stats_supported:
            visualizer._native_set_stats_enabled(native_library, False)
        if reference is not None and native_library is not None:
            native_library.fractal_destroy_reference(reference)

    timings = np.asarray([record["seconds"] for record in records], dtype=np.float64)
    result = {
        "stage": "atlas",
        "renderer": args.renderer,
        "width": args.width,
        "height": args.height,
        "max_zoom": args.zoom,
        "max_zoom_log10": log_zoom,
        "local_references": bool(args.local_references),
        "keyframe_factor": args.keyframe_factor,
        "levels": len(records),
        "origin_log10": origin,
        "step_log10": step,
        "reference_seconds": reference_seconds,
        "reference_zoom_log10": (
            args.reference_zoom_log
            if args.reference_zoom_log is not None
            else log_zoom
        ),
        "total_seconds": float(np.sum(timings)),
        "mean_seconds": float(np.mean(timings)),
        "median_seconds": float(np.median(timings)),
        "p95_seconds": float(np.percentile(timings, 95.0)),
        "worst_seconds": float(np.max(timings)),
        "records": records,
    }
    stats_records = [
        record["stats"] for record in records
        if isinstance(record.get("stats"), dict)
    ]
    if stats_records:
        result["stats_totals"] = {
            name: int(sum(int(stats.get(name, 0)) for stats in stats_records))
            for name in visualizer.NATIVE_STATS_FIELDS
        }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    slowest = sorted(records, key=lambda record: record["seconds"], reverse=True)[:5]
    print(
        f"atlas sweep {args.width}x{args.height} to 10^{log_zoom:.6f}: "
        f"{len(records)} levels, total {result['total_seconds']:.3f}s, "
        f"median {result['median_seconds']:.3f}s, "
        f"p95 {result['p95_seconds']:.3f}s, "
        f"worst {result['worst_seconds']:.3f}s; "
        f"reference setup {reference_seconds:.3f}s"
    )
    print("slowest levels:")
    for record in slowest:
        print(
            f"  {record['level']:4d}: 10^{record['zoom_log10']:.3f}, "
            f"{record['iterations']} iterations, {record['seconds']:.3f}s"
        )
    if stats_records:
        totals = result["stats_totals"]
        print(
            "stats totals: "
            f"BLA {totals['bla_blocks']}, "
            f"exact {totals['exact_steps']}, "
            f"double-tail pixels {totals['double_tail_pixels']}, "
            f"cycle-inside {totals['cycle_inside']}"
        )


def main() -> None:
    np = visualizer._require_numpy()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--stage",
        choices=("field", "compositor", "atlas"),
        default="field",
        help="benchmark one field, the native atlas output pass, or every atlas level",
    )
    parser.add_argument("--zoom", default="1e100")
    parser.add_argument(
        "--zoom-log",
        type=float,
        default=None,
        help="exact base-10 log10 zoom; useful for benchmarking one atlas level",
    )
    parser.add_argument(
        "--reference-zoom-log",
        type=float,
        default=None,
        help="reference precision zoom for a field probe; defaults to the rendered zoom",
    )
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--x-center", default=visualizer.DEFAULT_X_CENTER)
    parser.add_argument("--y-center", default=visualizer.DEFAULT_Y_CENTER)
    parser.add_argument("--renderer", choices=("auto", "native", "python"), default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--series-order", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--series-block", type=int, default=256)
    parser.add_argument(
        "--local-references",
        action="store_true",
        help=(
            "use the production adaptive secondary-reference path for "
            "deep fields; useful for atlas bottleneck measurements"
        ),
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--keyframe-factor",
        type=float,
        default=2.0,
        help="zoom factor between atlas levels for --stage atlas",
    )
    parser.add_argument("--iteration-base", type=int, default=384)
    parser.add_argument("--iterations-per-decade", type=int, default=500)
    parser.add_argument("--iteration-cap", type=int, default=100000)
    parser.add_argument(
        "--stats",
        action="store_true",
        help="collect native BLA/fallback counters when supported",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every level during an atlas sweep",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height, args.iterations, args.repeat) <= 0:
        raise SystemExit("width, height, iterations, and repeat must be positive")
    if args.threads < 0:
        raise SystemExit("threads cannot be negative")
    if not 2 <= args.series_block <= 4096:
        raise SystemExit("series-block must be between 2 and 4096")
    if not args.keyframe_factor > 1.0:
        raise SystemExit("keyframe-factor must be greater than 1")
    if min(args.iteration_base, args.iterations_per_decade, args.iteration_cap) <= 0:
        raise SystemExit("iteration settings must be positive")
    if args.iteration_cap < args.iteration_base:
        raise SystemExit("iteration-cap must be at least iteration-base")

    if args.zoom_log is not None:
        if not np.isfinite(args.zoom_log) or args.zoom_log < 0.0:
            raise SystemExit("zoom-log must be a finite non-negative number")
        log_zoom = float(args.zoom_log)
    else:
        log_zoom = visualizer._zoom_log(args.zoom)
    if args.reference_zoom_log is not None:
        if not np.isfinite(args.reference_zoom_log) or args.reference_zoom_log < log_zoom:
            raise SystemExit("reference-zoom-log must be finite and at least zoom-log")
    if args.stage == "atlas":
        _atlas_sweep(args, np, log_zoom)
        return
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
    reference_seconds = 0.0
    reference_log_zoom = None
    stats_supported = False
    if args.renderer != "python" and log_zoom >= 12.0:
        reference_started = time.perf_counter()
        reference_log_zoom = (
            args.reference_zoom_log
            if args.reference_zoom_log is not None
            else log_zoom
        )
        reference_iterations = max(
            args.iterations,
            visualizer.max_iterations(
                reference_log_zoom,
                args.iteration_base,
                args.iterations_per_decade,
                args.iteration_cap,
            ),
        )
        native_library, native_reference = visualizer._create_native_reference(
            args.x_center,
            args.y_center,
            reference_iterations,
            reference_log_zoom,
            args.series_order,
            12.0,
        )
        reference_seconds = time.perf_counter() - reference_started
        stats_supported = (
            args.stats
            and visualizer._native_set_stats_enabled(native_library, True)
        )
    timings = []
    try:
        for _ in range(args.repeat):
            started = time.perf_counter()
            if args.local_references:
                field = visualizer._atlas_local_reference_field(
                    render_width=args.width,
                    render_height=args.height,
                    log10_zoom=log_zoom,
                    x_center=args.x_center,
                    y_center=args.y_center,
                    max_iter=args.iterations,
                    series_order=args.series_order,
                    series_block=args.series_block,
                    renderer=args.renderer,
                    native_threads=args.threads,
                    native_library=native_library,
                )
            else:
                field = None
            if field is None:
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
            "zoom_log10": log_zoom,
            "local_references": bool(args.local_references),
            "iterations": args.iterations,
            "x_center": args.x_center,
            "y_center": args.y_center,
            "repeat": args.repeat,
            "reference_seconds": reference_seconds,
            "reference_zoom_log10": reference_log_zoom,
            "seconds": timings,
            "best_seconds": min(timings),
            "best_with_reference_seconds": reference_seconds + min(timings),
            "pixels_per_second": args.width * args.height / min(timings),
            "finite_fraction": float(np.isfinite(field).mean()),
        }
        if stats_supported:
            result["stats"] = visualizer._native_get_stats(native_library)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"{args.width}x{args.height} at 10^{log_zoom:.6f}: "
                f"best {result['best_seconds']:.3f}s, "
                f"{result['pixels_per_second']:.1f} pixels/s; "
                f"reference setup {reference_seconds:.3f}s, "
                f"one-shot total {result['best_with_reference_seconds']:.3f}s"
            )
    finally:
        if stats_supported and native_library is not None:
            visualizer._native_set_stats_enabled(native_library, False)
        if native_reference is not None and native_library is not None:
            native_library.fractal_destroy_reference(native_reference)


if __name__ == "__main__":
    main()
