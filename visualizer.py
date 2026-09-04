#!/usr/bin/env python3
"""Render a music-reactive Mandelbrot zoom from one audio file.

The renderer is deliberately split into two different rates:

* A high-resolution fractal image is rendered only at zoom keyframes.
* Video frames between keyframes are high-quality crops of that image.

This is considerably cheaper than calculating a small Mandelbrot image for
every video frame, and it is the same useful idea as a Kalles Fraktaler zoom
sequence: spend time rendering a good source image, then animate that image.

For deep views the image renderer uses perturbation theory plus a hierarchical
bilinear approximation (BLA).  The selected centre is kept as a decimal
string, a high-precision reference orbit is calculated once, and each pixel
is represented as a small offset from it.  The per-pixel path uses scaled
mantissa/exponent arithmetic, so a 100+ digit zoom does not turn the pixel
delta into zero when it is converted to a Python ``float``.

Audio separation uses Demucs automatically when it is installed.  When
reliable stems are unavailable, the default keeps the full song driving both
controls; an explicit spectral mode is available for frequency-band proxies.
The default logarithmic zoom velocity is slightly negative between strong
instrumental events, creating a small musical pullback; the final frame still
lands exactly on the requested maximum zoom.

Required packages:
    numpy, librosa, Pillow

Native build:
    nix-shell
    make

Optional package:
    mpmath (the standard-library decimal module is used if it is absent)
    demucs (automatic vocal/instrumental source separation)

Example:
    python visualizer.py song.mp3 --output fractal_viz.mp4
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from collections import deque
import datetime as _datetime
import hashlib
import heapq
import importlib.util
import json
import math
import os
import operator
import queue
import random
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from deep_zoom_points import (
    BURNING_SHIP_POINTS,
    DEEP_ZOOM_POINTS,
    FORMULA_POINT_CATALOGUES,
    FORMULA_POINTS_BY_SLUG,
    JULIA_POINTS,
    TRICORN_POINTS,
    DeepZoomPoint,
)
from profiles import (
    CANONICAL_PROFILE_CHOICES,
    DEFAULT_PROFILE,
    FAST_PROFILE_CHOICES,
    NEAR_LOSSLESS_CRF,
    PROFILE_ALIASES,
    PROFILE_CHOICES,
    PROFILE_DEFAULTS,
    PROFILE_DESCRIPTIONS,
    SOURCE_MODE_CHOICES,
    UPSCALED_SOURCE_SCALE,
)


# Do this before importing numerical libraries.  It helps BLAS-backed NumPy
# without forcing an unnecessarily large number of worker threads.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 3))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")


DEFAULT_AUDIO = "song.mp3"
DEFAULT_OUTPUT = "fractal_viz.mp4"
# Use a catalogue centre with enough exported digits for the default e100
# quality profile. The old bundled centre stopped at 129 fractional places,
# which made a no-argument quality render fail its own precision guard.
_DEFAULT_MANDELBROT_POINT = next(
    point for point in DEEP_ZOOM_POINTS if point.slug == "oldwooddish"
)
DEFAULT_X_CENTER = _DEFAULT_MANDELBROT_POINT.x
DEFAULT_Y_CENTER = _DEFAULT_MANDELBROT_POINT.y

# Fields are expensive, so they can be cached; their cache identity must
# change when native numerical behaviour changes.  This prevents a new
# renderer from silently reusing an old, under-resolved .npy keyframe.
KEYFRAME_CACHE_SCHEMA = "raw-field-series-v12-kalles-interior"
AUDIO_CACHE_SCHEMA = "audio-controls-v10-onset-sync"
ATLAS_CACHE_SCHEMA = "nested-raw-atlas-v17-overscanned-tiles"

FORMULA_CHOICES = (
    "mandelbrot",
    "julia",
    "burning-ship",
    "tricorn",
)
VIDEO_PRESET_CHOICES = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
)
PALETTE_CHOICES = (
    "aurora",
    "fire",
    "ocean",
    "neon",
    "sunset",
    "mono",
    "midnight",
    "ember-night",
    "terminal",
    "kalles-default",
)
KALLES_DEFAULT_PALETTE_STOPS = (
    (255, 255, 255),
    (128, 0, 64),
    (160, 0, 0),
    (192, 128, 0),
    (64, 128, 0),
    (0, 255, 255),
    (64, 128, 255),
    (0, 0, 255),
)


@dataclass(frozen=True)
class KfpPalette:
    """Portable Kalles Fraktaler palette plus its colour-pipeline settings."""

    stops: tuple[tuple[int, int, int], ...]
    iter_div: float = 1.0
    color_offset: float = 0.0
    ratio: float = 360.0
    color_method: int = 0
    smooth_method: int = 0
    smooth: bool = True
    flat: bool = False
    inverse_transition: bool = False
    phase_color_strength: float = 0.0
    multi_color: bool = False
    blend_multi_color: bool = False
    multi_colors: tuple[tuple[int, int, int], ...] = ()
    power: float = 2.0
    slopes: bool = False
    slope_power: float = 50.0
    slope_ratio: float = 20.0
    slope_angle: float = 45.0
    differences: int = 3
    interior_color: tuple[int, int, int] = (0, 0, 0)


# These are the settings written by Kalles Fraktaler's own default palette.
# Keeping them with the imported key colours is important: a KFP is more than
# a gradient, and its short iteration cycle plus slope pass is what produces
# the characteristic layered, high-contrast deep-zoom appearance.
KALLES_DEFAULT_KFP = KfpPalette(
    stops=KALLES_DEFAULT_PALETTE_STOPS,
    iter_div=0.01,
    ratio=360.0,
    color_method=7,
    smooth_method=0,
    smooth=True,
    power=2.0,
    slopes=True,
    slope_power=50.0,
    slope_ratio=20.0,
    slope_angle=45.0,
    differences=3,
    interior_color=(0, 0, 0),
)


# Ordinary palettes intentionally reuse Aurora's three flowing waves.  These
# are three real RGB accents, rather than one hue rotation applied to Aurora;
# the waves are composited through all three accents so each palette retains a
# distinct colour language while keeping the fast native colour path.
BUILTIN_AURORA_ACCENTS = {
    "fire": (
        (255, 20, 0),
        (255, 150, 0),
        (255, 245, 170),
    ),
    "ocean": (
        (0, 55, 255),
        (0, 220, 255),
        (160, 255, 235),
    ),
    "neon": (
        (255, 0, 160),
        (85, 0, 255),
        (0, 255, 170),
    ),
    "sunset": (
        (255, 20, 115),
        (255, 105, 0),
        (255, 240, 75),
    ),
    "mono": (
        (32, 32, 32),
        (170, 170, 170),
        (255, 255, 255),
    ),
    # These themes deliberately keep the exterior near-black while retaining
    # three distinct Aurora wave accents. Their white interiors are applied
    # separately, so the set remains legible instead of blending into the
    # dark exterior.
    "midnight": (
        (2, 4, 12),
        (12, 48, 96),
        (90, 180, 255),
    ),
    "ember-night": (
        (8, 0, 2),
        (68, 4, 8),
        (255, 105, 25),
    ),
    "terminal": (
        (0, 8, 0),
        (0, 90, 20),
        (170, 255, 170),
    ),
}
# Keep the non-liquid palettes punchy at their source instead of trying to
# recover contrast after audio effects and integer quantisation.  The first
# and last stops deliberately reach near-black and near-white so fractal
# boundaries remain readable on both bright and dark desktop themes.
# Kalles' stops are kept verbatim: they are the portable representation of the
# bundled .kfp file and must continue to match that file's source colours.
BUILTIN_PALETTE_STOPS = {
    "fire": (
        (0, 0, 0),
        (48, 0, 12),
        (180, 0, 0),
        (255, 40, 0),
        (255, 180, 0),
        (255, 255, 190),
        (255, 255, 255),
    ),
    "ocean": (
        (0, 0, 10),
        (0, 12, 70),
        (0, 70, 190),
        (0, 205, 255),
        (40, 255, 220),
        (210, 255, 255),
        (255, 255, 255),
    ),
    "neon": (
        (0, 0, 20),
        (75, 0, 140),
        (255, 0, 150),
        (0, 180, 255),
        (0, 255, 200),
        (220, 255, 0),
        (255, 255, 255),
    ),
    "sunset": (
        (5, 0, 20),
        (80, 0, 80),
        (210, 0, 55),
        (255, 45, 0),
        (255, 170, 0),
        (255, 240, 100),
        (255, 255, 230),
    ),
    "mono": (
        (0, 0, 0),
        (8, 8, 8),
        (48, 48, 48),
        (192, 192, 192),
        (248, 248, 248),
        (255, 255, 255),
    ),
    "midnight": (
        (1, 2, 8),
        (3, 12, 36),
        (12, 55, 120),
        (70, 170, 255),
        (232, 250, 255),
    ),
    "ember-night": (
        (4, 0, 1),
        (35, 2, 5),
        (120, 10, 12),
        (255, 90, 15),
        (255, 255, 230),
    ),
    "terminal": (
        (0, 3, 0),
        (0, 22, 4),
        (0, 90, 20),
        (40, 220, 90),
        (220, 255, 220),
    ),
    "kalles-default": KALLES_DEFAULT_PALETTE_STOPS,
}
BUILTIN_INTERIOR_COLORS = {
    "midnight": (255, 255, 255),
    "ember-night": (255, 255, 255),
    "terminal": (255, 255, 255),
}
FORMULA_IDS = {
    "mandelbrot": 0,
    "julia": 1,
    "burning-ship": 2,
    "tricorn": 3,
}
FORMULA_DESCRIPTIONS = {
    "mandelbrot": "classic z²+c parameter plane; the e150+ optimized path",
    "julia": "fixed-c Julia set; use --julia-c to change the constant",
    "burning-ship": "absolute-value z²+c, with the characteristic ship symmetry",
    "tricorn": "conjugate z²+c (the Mandelbar)",
}
# Every formula has its own catalogue and default centre.  Julia presets also
# carry their fixed c value; the Mandelbrot catalogue is the production native
# e150+ path while alternate formulas use validated deep boundary targets.
FORMULA_DEFAULT_CENTERS = {
    "mandelbrot": (DEFAULT_X_CENTER, DEFAULT_Y_CENTER),
    "julia": (JULIA_POINTS[0].x, JULIA_POINTS[0].y),
    "burning-ship": (BURNING_SHIP_POINTS[0].x, BURNING_SHIP_POINTS[0].y),
    "tricorn": (TRICORN_POINTS[0].x, TRICORN_POINTS[0].y),
}
DEFAULT_JULIA_C = ("-0.8", "0.156")

# A single reference is normally ideal, but at deep zoom a frame-sized tile
# can contain a narrow boundary that is much farther from that reference than
# the rest of the image. The native renderer then spends most of its time in
# exact/replayed tails. Large deep tiles start with one bounded shared pass and
# split only connected regions whose pixels remain unresolved. This keeps MPFR
# setup and memory proportional to the genuinely difficult part of a tile.
# Radius-specific references begin at e20. Allow the glitch-driven repair path
# to handle those tiers immediately; keeping the old e40 gate let a strict
# e30-ish tile fail outright when its newly selected reference exposed a
# genuine perturbation glitch.
ATLAS_LOCAL_REFERENCE_MIN_LOG = 20.0
ATLAS_LOCAL_REFERENCE_MIN_DIMENSION = 96
ATLAS_LOCAL_REFERENCE_MAX_DIVISIONS = 32
ATLAS_LOCAL_REFERENCE_MIN_BUDGET_MS = 200
ATLAS_LOCAL_REFERENCE_MAX_BUDGET_MS = 4000
ATLAS_LOCAL_REFERENCE_MS_PER_PIXEL = 0.05
ATLAS_LOCAL_REFERENCE_FINAL_BUDGET_MS = 750
ATLAS_LOCAL_REFERENCE_TILE_BUDGET_MS = 30_000
# Arbitrary-precision direct recovery is intentionally a small escape hatch.
# A single deep pixel can take hundreds of milliseconds, so a large mask must
# continue through the native subdivision path rather than serialising a whole
# edge into Python big-number iterations.
ATLAS_LOCAL_REFERENCE_MAX_EXACT_PIXELS = 32
ATLAS_PROGRESS_INTERVAL_SECONDS = 20.0
ENCODER_BACKPRESSURE_NOTICE_SECONDS = 10.0
# A child that is smaller than a few display pixels is not carrying useful
# detail yet. Compositing it anyway makes a one-pixel KFP stencil flicker at
# every atlas boundary and reads as camera shake in the encoded video.
ATLAS_MIN_CHILD_PIXELS = 4
# On a reverse zoom, the active parent can be selected for one or two frames
# while its child still covers almost the whole viewport. Leaving the tiny
# uncovered strip to the parent is mathematically valid but visibly wrong for
# KFP's spatial shading: it becomes a rectangular border. Once the uncovered
# margin is below 2%, promote the child to the complete view instead.
ATLAS_NEAR_FULL_CHILD_FRACTION = 0.98
# Render each atlas tile a little wider than its nominal viewport. A camera
# can pull back between two adjacent levels; without this margin the child
# then covers only (for example) 88% of the frame and KFP's spatial stencil
# makes the low-resolution parent perimeter visible as a rectangular block.
# The same fixed overscan is used at every level, so adjacent tile scales and
# the exact crop geometry remain self-consistent.
ATLAS_TILE_OVERSCAN_FACTOR = 1.2
# Alternate formula perturbation uses a decimal reference once the viewport
# is narrower than a float64 centre can represent reliably. Keep this single
# threshold shared by the direct renderer and the video planner so an atlas
# cannot label a Python field as native or accidentally hand it a Mandelbrot
# BLA reference.
ALTERNATE_PERTURBATION_MIN_LOG = 7.0
MIN_LOG10_ZOOM = -300.0
MAX_LOG10_ZOOM = 9800.0
# A render can multiply the output dimensions by both --render-scale and a
# quality preset. Keep the public boundary finite so a typo cannot allocate a
# multi-gigabyte field before FFmpeg has even started.
MAX_RENDER_PIXELS = 100_000_000
MAX_AUDIO_FRAMES = 10_000_000
# Keep decoded audio bounded as well as video frames. A high requested sample
# rate can otherwise turn an otherwise valid multi-minute track into a
# multi-gigabyte float32 allocation before frame-count validation runs.
MAX_AUDIO_SAMPLES = 120_000_000
MAX_ITERATION_BUDGET = 10_000_000
MAX_KEYFRAME_LEVELS = 100_000
MAX_THREAD_COUNT = 4096
MAX_FPS = 1000
MAX_SAMPLE_RATE = 768_000
MAX_COORDINATE_TEXT_LENGTH = 50_000
MAX_NATIVE_PRECISION_BITS = 131_072
MAX_PALETTE_FILE_BYTES = 4 * 1024 * 1024
MAX_PALETTE_STOPS = 4096
MAX_CACHE_FILE_BYTES = 512 * 1024 * 1024
MAX_CACHE_ARCHIVE_MEMBERS = 16
MAX_CACHE_ARCHIVE_COMMENT_BYTES = 65_535
MAX_CACHE_HEADER_BYTES = 64 * 1024
MAX_CACHE_ARRAY_ELEMENTS = MAX_RENDER_PIXELS
DEMUCS_TIMEOUT_SECONDS = 30.0 * 60.0
MAX_DEMUCS_DISCOVERY_ENTRIES = 10_000
MAX_CACHE_SCAN_ENTRIES = 100_000
MAX_DIAGNOSTIC_LINE_CHARS = 16_384
FFMPEG_ENCODER_QUERY_TIMEOUT_SECONDS = 5.0
MAX_FFMPEG_ENCODER_QUERY_BYTES = 1 * 1024 * 1024
FFMPEG_FINALIZE_TIMEOUT_SECONDS = 30.0 * 60.0
FFMPEG_STDIN_STALL_TIMEOUT_SECONDS = 5.0 * 60.0
# A frame-count report is useful even when a single high-resolution frame or
# atlas tile takes longer than the old video-time reporting interval. The GTK
# launcher consumes these lines for its determinate progress bar.
RENDER_PROGRESS_INTERVAL_SECONDS = 5.0

# These are the original visualizer's liquid-gradient constants.  Keep the
# numerical field independent from them: the atlas stores raw iteration data,
# so the same expensive deep render can be recoloured without being rebuilt.
AURORA_BAND_THICKNESS = 0.15
# The original liquid engine peaked at 60 radians/second.  Keep its response
# shape, but halve the rate so the colour movement is calmer during a render.
AURORA_FLOW_SPEED = 30.0
# A completely silent tail must not turn the video into a frozen still.  This
# small carrier keeps the liquid field moving while the audio envelope still
# controls the energetic part of the flow.
AURORA_MIN_FLOW_FRACTION = 0.08
AURORA_COLOUR_SPLIT = 5.0
AURORA_GREEN_SPLIT = 0.4

# A deep-zoom centre is part of the numerical input, not merely a display
# preference.  Keep a guard band beyond the requested zoom so the supplied
# decimal does not become the dominant uncertainty in the viewport.  The
# bundled centre has 129 fractional decimal places: that is ample for the
# tested e100 path, but it is not a faithful e150/e4000 Kalles target.
CENTER_PRECISION_GUARD_DIGITS = 16

_native_library: Any = None
_native_checked = False
_native_notice_printed = False
_native_library_lock = threading.Lock()

NATIVE_STATS_FIELDS = (
    "pixels",
    "logical_iterations",
    "bla_blocks",
    "linear_blocks",
    "cubic_blocks",
    "exact_steps",
    "replay_steps",
    "bla_retries",
    "cycle_inside",
    "double_tail_pixels",
    "bla_disabled_pixels",
    "tail_steps",
    "max_tail_steps",
    "tail_rebases",
    "tail_rebase_fallbacks",
    "max_pixel_iterations",
)
NATIVE_EXTENDED_STATS_FIELDS = NATIVE_STATS_FIELDS + (
    "series_pixels",
    "series_jumps",
    "glitch_count",
    "unresolved_pixels",
    "deadline_aborts",
    "secondary_references",
    "render_ns",
) + tuple(f"bla_length_bin_{index}" for index in range(16))


class NativeRenderOptions(ctypes.Structure):
    """Versioned per-call controls for the native C ABI."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("strict", ctypes.c_int32),
        ("allow_recovery", ctypes.c_int32),
        ("time_budget_ms", ctypes.c_int32),
        ("disable_bla", ctypes.c_int32),
        ("disable_cycle", ctypes.c_int32),
        ("strict_cycle", ctypes.c_int32),
        ("series_min_terms", ctypes.c_int32),
        ("series_max_terms", ctypes.c_int32),
        ("max_bla_length", ctypes.c_int32),
        ("max_linear_bla_length", ctypes.c_int32),
        ("backend", ctypes.c_int32),
        ("reserved", ctypes.c_int32 * 3),
    ]

    def __init__(
        self,
        *,
        strict: bool = True,
        allow_recovery: bool = False,
        time_budget_ms: int = 0,
        disable_bla: bool = False,
        disable_cycle: bool = False,
        strict_cycle: bool = False,
        series_min_terms: int = 8,
        series_max_terms: int = 32,
        max_bla_length: int = 256,
        max_linear_bla_length: int = 4096,
        backend: int = 0,
    ) -> None:
        super().__init__()
        integer_values = {
            "time budget": time_budget_ms,
            "series minimum": series_min_terms,
            "series maximum": series_max_terms,
            "maximum BLA length": max_bla_length,
            "maximum linear BLA length": max_linear_bla_length,
            "backend": backend,
        }
        checked_values: dict[str, int] = {}
        for label, value in integer_values.items():
            try:
                checked_values[label] = int(operator.index(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} must be an integer") from error
        if checked_values["time budget"] < 0:
            raise ValueError("time budget cannot be negative")
        if not 1 <= checked_values["series minimum"] <= 32:
            raise ValueError("series minimum must be between 1 and 32")
        if not 1 <= checked_values["series maximum"] <= 32:
            raise ValueError("series maximum must be between 1 and 32")
        if checked_values["series minimum"] > checked_values["series maximum"]:
            raise ValueError("series minimum cannot exceed series maximum")
        if not 1 <= checked_values["maximum BLA length"] <= 4096:
            raise ValueError("maximum BLA length must be between 1 and 4096")
        if not 1 <= checked_values["maximum linear BLA length"] <= 4096:
            raise ValueError("maximum linear BLA length must be between 1 and 4096")
        if checked_values["backend"] not in {0, 1, 2}:
            raise ValueError("backend must be scalar (0), avx2 (1), or opencl (2)")
        if checked_values["time budget"] > 2_147_483_647:
            raise ValueError("time budget is too large for the native ABI")

        def binary_flag(value: Any, label: str) -> int:
            try:
                result = int(operator.index(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} must be 0 or 1") from error
            if result not in {0, 1}:
                raise ValueError(f"{label} must be 0 or 1")
            return result

        self.struct_size = ctypes.sizeof(self)
        self.version = 1
        self.strict = binary_flag(strict, "strict")
        self.allow_recovery = binary_flag(allow_recovery, "allow recovery")
        self.time_budget_ms = checked_values["time budget"]
        self.disable_bla = binary_flag(disable_bla, "disable BLA")
        self.disable_cycle = binary_flag(disable_cycle, "disable cycle detection")
        self.strict_cycle = binary_flag(strict_cycle, "strict cycle detection")
        self.series_min_terms = checked_values["series minimum"]
        self.series_max_terms = checked_values["series maximum"]
        self.max_bla_length = checked_values["maximum BLA length"]
        self.max_linear_bla_length = checked_values["maximum linear BLA length"]
        self.backend = checked_values["backend"]


NATIVE_KFP_MAX_MULTI_COLORS = 256


class NativeKfpOptions(ctypes.Structure):
    """ctypes mirror of the native Kalles Fraktaler transfer options."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("iter_div", ctypes.c_double),
        ("color_offset", ctypes.c_double),
        ("ratio", ctypes.c_double),
        ("color_method", ctypes.c_int32),
        ("smooth_method", ctypes.c_int32),
        ("smooth", ctypes.c_int32),
        ("flat", ctypes.c_int32),
        ("inverse_transition", ctypes.c_int32),
        ("phase_color_strength", ctypes.c_double),
        ("multi_color", ctypes.c_int32),
        ("blend_multi_color", ctypes.c_int32),
        ("multi_color_count", ctypes.c_uint32),
        ("multi_color_period", ctypes.c_double * NATIVE_KFP_MAX_MULTI_COLORS),
        ("multi_color_start", ctypes.c_int32 * NATIVE_KFP_MAX_MULTI_COLORS),
        ("multi_color_type", ctypes.c_int32 * NATIVE_KFP_MAX_MULTI_COLORS),
        ("power", ctypes.c_double),
        ("slopes", ctypes.c_int32),
        ("slope_power", ctypes.c_double),
        ("slope_ratio", ctypes.c_double),
        ("slope_angle", ctypes.c_double),
        ("differences", ctypes.c_int32),
        ("interior_color", ctypes.c_int32 * 3),
    ]

    @classmethod
    def from_profile(cls, profile: KfpPalette) -> "NativeKfpOptions":
        if len(profile.multi_colors) > NATIVE_KFP_MAX_MULTI_COLORS:
            raise ValueError("KFP profile contains too many multi-colour waves")
        result = cls()
        result.struct_size = ctypes.sizeof(cls)
        result.version = 1
        result.iter_div = profile.iter_div
        result.color_offset = profile.color_offset
        result.ratio = profile.ratio
        result.color_method = profile.color_method
        result.smooth_method = profile.smooth_method
        result.smooth = int(profile.smooth)
        result.flat = int(profile.flat)
        result.inverse_transition = int(profile.inverse_transition)
        result.phase_color_strength = profile.phase_color_strength
        result.multi_color = int(profile.multi_color)
        result.blend_multi_color = int(profile.blend_multi_color)
        result.multi_color_count = len(profile.multi_colors)
        for index, (period, start, colour_type) in enumerate(profile.multi_colors):
            result.multi_color_period[index] = period
            result.multi_color_start[index] = start
            result.multi_color_type[index] = colour_type
        result.power = profile.power
        result.slopes = int(profile.slopes)
        result.slope_power = profile.slope_power
        result.slope_ratio = profile.slope_ratio
        result.slope_angle = profile.slope_angle
        result.differences = profile.differences
        result.interior_color[:] = profile.interior_color
        return result


NATIVE_BACKEND_NAMES = {
    "scalar": 0,
    "avx2": 1,
    "opencl": 2,
}


def _native_backend_id(name: str, library: Any) -> int:
    """Resolve a user-facing backend name against the loaded ABI."""

    if name == "auto":
        # AVX2 is a low-overhead CPU choice for shallow/direct frames. OpenCL
        # is opt-in because a CPU OpenCL ICD can be slower than OpenMP and
        # deep frames intentionally stay on the validated CPU path.
        if library is not None and hasattr(library, "fractal_backend_capabilities"):
            return 1 if int(library.fractal_backend_capabilities()) & 2 else 0
        return 0
    try:
        backend = NATIVE_BACKEND_NAMES[name]
    except KeyError as error:
        raise ValueError(f"unknown native backend: {name}") from error
    if library is None or not hasattr(library, "fractal_backend_capabilities"):
        raise RuntimeError("native backend capabilities are unavailable")
    capabilities = int(library.fractal_backend_capabilities())
    required_bit = 1 << backend
    if not capabilities & required_bit:
        raise RuntimeError(
            f"native backend {name} is unavailable in this build/device "
            f"(capabilities bitmask {capabilities})"
        )
    return backend


def _field_renderer_cache_identity(renderer: str) -> str:
    """Return a content-based cache namespace for the active field renderer."""

    def file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()[:16]

    if renderer == "python":
        try:
            return "python-" + file_digest(Path(__file__))
        except OSError:
            return "python-unknown"

    configured = os.environ.get("MANDELBROT_LIBRARY")
    candidates = [
        Path(configured) if configured else Path(__file__).with_name("mandelbrot.so"),
        Path(__file__).with_name("libmandelbrot.so"),
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return "native-" + file_digest(candidate)
        except (OSError, ValueError):
            continue
    return "native-unavailable"


def _get_native_library() -> Any:
    """Load only the versioned renderer ABI, never an old incompatible .so."""

    global _native_checked, _native_library
    if _native_checked:
        return _native_library
    with _native_library_lock:
        # A render can be started from more than one Python worker. Do not let
        # one worker observe the in-progress probe as a permanent "no native
        # library" result.
        if _native_checked:
            return _native_library

        configured = os.environ.get("MANDELBROT_LIBRARY")
        default_library = Path(__file__).with_name("mandelbrot.so")
        try:
            configured_library = Path(configured) if configured else default_library
        except (TypeError, ValueError):
            configured_library = default_library
        candidates = [configured_library, Path(__file__).with_name("libmandelbrot.so")]
        for candidate in candidates:
            try:
                if not candidate.is_file():
                    continue
                library = ctypes.CDLL(str(candidate))
                if not hasattr(library, "fractal_abi_version"):
                    continue
                library.fractal_abi_version.restype = ctypes.c_int
                if library.fractal_abi_version() != 10:
                    continue
                if hasattr(library, "fractal_backend_capabilities"):
                    library.fractal_backend_capabilities.restype = ctypes.c_int
                library.fractal_last_error.restype = ctypes.c_char_p
                library.fractal_colourise.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_int,
                ]
                library.fractal_colourise.restype = ctypes.c_int
                if hasattr(library, "fractal_apply_aurora_accents"):
                    library.fractal_apply_aurora_accents.argtypes = [
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_double,
                        ctypes.c_int,
                    ]
                    library.fractal_apply_aurora_accents.restype = ctypes.c_int
                if hasattr(library, "fractal_colourise_kfp"):
                    library.fractal_colourise_kfp.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.POINTER(NativeKfpOptions),
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_colourise_kfp.restype = ctypes.c_int
                if hasattr(library, "fractal_crop_colourise_kfp"):
                    library.fractal_crop_colourise_kfp.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.POINTER(NativeKfpOptions),
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_crop_colourise_kfp.restype = ctypes.c_int
                if hasattr(library, "fractal_atlas_colourise_kfp"):
                    library.fractal_atlas_colourise_kfp.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.POINTER(NativeKfpOptions),
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_atlas_colourise_kfp.restype = ctypes.c_int
                if hasattr(library, "fractal_atlas_colourise_kfp_raw"):
                    library.fractal_atlas_colourise_kfp_raw.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.POINTER(NativeKfpOptions),
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_atlas_colourise_kfp_raw.restype = ctypes.c_int
                if hasattr(library, "fractal_crop_field"):
                    library.fractal_crop_field.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_int,
                    ]
                    library.fractal_crop_field.restype = ctypes.c_int
                library.fractal_crop_colourise.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_double,
                    ctypes.c_int,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_int,
                ]
                library.fractal_crop_colourise.restype = ctypes.c_int
                if hasattr(library, "fractal_crop_colourise_interior"):
                    library.fractal_crop_colourise_interior.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_crop_colourise_interior.restype = ctypes.c_int
                if hasattr(library, "fractal_crop_colourise_accents"):
                    library.fractal_crop_colourise_accents.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_crop_colourise_accents.restype = ctypes.c_int
                if hasattr(library, "fractal_atlas_colourise"):
                    library.fractal_atlas_colourise.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_int,
                    ]
                    library.fractal_atlas_colourise.restype = ctypes.c_int
                if hasattr(library, "fractal_atlas_colourise_interior"):
                    library.fractal_atlas_colourise_interior.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_atlas_colourise_interior.restype = ctypes.c_int
                if hasattr(library, "fractal_atlas_colourise_accents"):
                    library.fractal_atlas_colourise_accents.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_atlas_colourise_accents.restype = ctypes.c_int
                library.fractal_create_reference.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                library.fractal_create_reference.restype = ctypes.c_void_p
                if hasattr(library, "fractal_create_reference_reusable"):
                    library.fractal_create_reference_reusable.argtypes = [
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                    ]
                    library.fractal_create_reference_reusable.restype = ctypes.c_void_p
                if hasattr(library, "fractal_clone_reference"):
                    library.fractal_clone_reference.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_char_p,
                    ]
                    library.fractal_clone_reference.restype = ctypes.c_void_p
                library.fractal_destroy_reference.argtypes = [ctypes.c_void_p]
                library.fractal_destroy_reference.restype = None
                if hasattr(library, "fractal_get_reference_stats"):
                    library.fractal_get_reference_stats.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_uint64),
                        ctypes.c_int,
                    ]
                    library.fractal_get_reference_stats.restype = ctypes.c_int
                library.render_mandelbrot_reference.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                library.render_mandelbrot_reference.restype = ctypes.c_int
                if hasattr(library, "fractal_render_mandelbrot_reference_ex"):
                    library.fractal_render_mandelbrot_reference_ex.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_char_p,
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(NativeRenderOptions),
                    ]
                    library.fractal_render_mandelbrot_reference_ex.restype = ctypes.c_int
                if hasattr(library, "fractal_render_points"):
                    library.fractal_render_points.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_char_p,
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_int32),
                        ctypes.c_void_p,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(NativeRenderOptions),
                    ]
                    library.fractal_render_points.restype = ctypes.c_int
                library.render_mandelbrot.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                library.render_mandelbrot.restype = ctypes.c_int
                if hasattr(library, "render_mandelbrot_ex"):
                    library.render_mandelbrot_ex.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(NativeRenderOptions),
                    ]
                    library.render_mandelbrot_ex.restype = ctypes.c_int
                if hasattr(library, "render_fractal_ex"):
                    library.render_fractal_ex.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.POINTER(NativeRenderOptions),
                    ]
                    library.render_fractal_ex.restype = ctypes.c_int
                if hasattr(library, "fractal_set_stats_enabled"):
                    library.fractal_set_stats_enabled.argtypes = [ctypes.c_int]
                    library.fractal_set_stats_enabled.restype = None
                if hasattr(library, "fractal_get_last_stats"):
                    library.fractal_get_last_stats.argtypes = [
                        ctypes.POINTER(ctypes.c_uint64),
                        ctypes.c_int,
                    ]
                    library.fractal_get_last_stats.restype = ctypes.c_int
                if hasattr(library, "fractal_get_last_stats_ex"):
                    library.fractal_get_last_stats_ex.argtypes = [
                        ctypes.POINTER(ctypes.c_uint64),
                        ctypes.c_int,
                    ]
                    library.fractal_get_last_stats_ex.restype = ctypes.c_int
                _native_library = library
                _native_checked = True
                return library
            except (OSError, AttributeError, TypeError, ValueError):
                # Missing runtimes, malformed symbols, and incompatible shared
                # libraries should not prevent a later candidate or the Python
                # implementation from running.
                continue
        _native_checked = True
        return None


def _native_set_stats_enabled(library: Any, enabled: bool) -> bool:
    setter = getattr(library, "fractal_set_stats_enabled", None)
    if setter is None:
        return False
    setter(1 if enabled else 0)
    return True


def _native_get_stats(library: Any) -> Optional[dict[str, int]]:
    getter = getattr(library, "fractal_get_last_stats_ex", None)
    fields = NATIVE_EXTENDED_STATS_FIELDS
    if getter is None:
        getter = getattr(library, "fractal_get_last_stats", None)
        fields = NATIVE_STATS_FIELDS
    if getter is None:
        return None
    values = (ctypes.c_uint64 * len(fields))()
    count = int(getter(values, len(fields)))
    # Older libraries accept the legacy field count; current libraries expose
    # the extended histogram through the versioned entry point.
    if count != len(fields):
        if len(fields) == len(NATIVE_EXTENDED_STATS_FIELDS):
            return None
        if count != len(NATIVE_STATS_FIELDS):
            return None
    return {
        name: int(values[index])
        for index, name in enumerate(fields)
    }


def _crop_field_native(
    field: Any,
    output_width: int,
    output_height: int,
    zoom_factor: float,
    library: Any,
    threads: int,
) -> Any:
    """Reproject raw iteration data without converting it to RGB."""

    np = _require_numpy()
    cropper = getattr(library, "fractal_crop_field", None)
    if cropper is None:
        raise RuntimeError("native raw-field reprojection is unavailable")
    source = np.ascontiguousarray(field, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("raw-field reprojection requires a 2-D field")
    output = np.empty((output_height, output_width), dtype=np.float32)
    status = cropper(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(source.shape[1]),
        int(source.shape[0]),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(output_width),
        int(output_height),
        float(zoom_factor),
        int(threads),
    )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native raw-field reprojection error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def _scaled_log10_radius(log10_radius: float) -> tuple[float, int]:
    """Represent 10**log10_radius as a normalized mantissa and binary exponent."""

    if not math.isfinite(log10_radius):
        raise ValueError("exponential-map radius must be finite")
    log2_radius = float(log10_radius) * math.log2(10.0)
    exponent = math.floor(log2_radius)
    mantissa = math.pow(2.0, log2_radius - exponent)
    mantissa, shift = math.frexp(mantissa)
    return float(mantissa), int(exponent + shift)


def render_exponential_field(
    *,
    radial_samples: int,
    angular_samples: int,
    min_log10_radius: float,
    max_log10_radius: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    series_order: int = 3,
    series_block: int = 64,
    native_threads: int = 0,
    native_reference: Any = None,
    render_options: Optional[NativeRenderOptions] = None,
) -> Any:
    """Render a raw log-polar/exponential-map field.

    Rows are uniformly spaced in log10 radius and columns are uniformly
    spaced in angle.  The returned array is scalar iteration data, never
    RGB.  This is an opt-in building block for a GUI or a long zoom master;
    the normal command-line renderer continues to use the nested atlas until
    an exp-map has passed the same oracle suite at the requested sampling
    density.
    """

    np = _require_numpy()
    radial_samples = _index_value(radial_samples, "radial sample count")
    angular_samples = _index_value(angular_samples, "angular sample count")
    max_iter = _validate_iteration_count(max_iter)
    series_order = _index_value(series_order, "series order")
    series_block = _index_value(series_block, "series block")
    native_threads = _validate_thread_count(native_threads, "native thread count")
    if radial_samples <= 1 or angular_samples <= 3:
        raise ValueError("exponential-map sampling grid is too small")
    if radial_samples * angular_samples > MAX_RENDER_PIXELS:
        raise ValueError("exponential-map sampling grid is too large")
    if not 1 <= series_order <= 32 or not 2 <= series_block <= 4096:
        raise ValueError("series order/block are outside the supported range")
    min_log10_radius = _validate_log10_zoom(min_log10_radius, "minimum exp-map radius")
    max_log10_radius = _validate_log10_zoom(max_log10_radius, "maximum exp-map radius")
    if max_log10_radius <= min_log10_radius:
        raise ValueError("exponential-map radius range must be increasing")
    x_center = _validate_center_text(x_center, "real")
    y_center = _validate_center_text(y_center, "imaginary")
    library = _get_native_library()
    if library is None or not hasattr(library, "fractal_render_points"):
        raise RuntimeError("native point renderer is unavailable; run `make`")
    if native_reference is None:
        # The reference viewport radius is 2.8 / zoom.  Choose it from the
        # largest radius in the map so every point stays inside the BLA/series
        # validation domain.
        reference_log = math.log10(2.8) - float(max_log10_radius)
        _, native_reference = _create_native_reference(
            x_center,
            y_center,
            max_iter,
            reference_log,
            series_order,
            reference_log,
        )
        owns_reference = True
    else:
        owns_reference = False

    count = int(radial_samples) * int(angular_samples)
    real = np.empty(count, dtype=np.float64)
    imag = np.empty(count, dtype=np.float64)
    exponents = np.empty(count, dtype=np.int32)
    radius_step = (float(max_log10_radius) - float(min_log10_radius)) / radial_samples
    angle_step = 2.0 * math.pi / angular_samples
    for radial in range(radial_samples):
        log_radius = float(min_log10_radius) + (radial + 0.5) * radius_step
        mantissa, exponent = _scaled_log10_radius(log_radius)
        begin = radial * angular_samples
        end = begin + angular_samples
        angles = angle_step * (np.arange(angular_samples, dtype=np.float64) + 0.5)
        real[begin:end] = mantissa * np.cos(angles)
        imag[begin:end] = mantissa * np.sin(angles)
        exponents[begin:end] = exponent

    output = np.empty(count, dtype=np.float32)
    options = render_options or NativeRenderOptions()
    try:
        status = library.fractal_render_points(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            count,
            b"1e0",
            real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            native_reference,
            max_iter,
            native_threads,
            series_order,
            series_block,
            ctypes.byref(options),
        )
        if status != 0:
            message = library.fractal_last_error() or b"unknown native exponential-map renderer error"
            raise RuntimeError(message.decode("utf-8", errors="replace"))
    finally:
        if owns_reference:
            library.fractal_destroy_reference(native_reference)
    return output.reshape((radial_samples, angular_samples))


def reproject_exponential_field(
    field: Any,
    *,
    min_log10_radius: float,
    max_log10_radius: float,
    log10_zoom: float,
    output_width: int,
    output_height: int,
    max_iter: int,
) -> Any:
    """Reproject a raw exp-map into one centred rectangular view."""

    np = _require_numpy()
    source = np.asarray(field, dtype=np.float32)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] < 4:
        raise ValueError("invalid exponential-map field")
    if not np.isfinite(source).all():
        raise ValueError("exponential-map field contains non-finite samples")
    output_width, output_height = _validate_dimensions(output_width, output_height, "exp-map output")
    max_iter = _validate_iteration_count(max_iter)
    min_log10_radius = _validate_log10_zoom(min_log10_radius, "minimum exp-map radius")
    max_log10_radius = _validate_log10_zoom(max_log10_radius, "maximum exp-map radius")
    log10_zoom = _validate_log10_zoom(log10_zoom)
    if max_log10_radius <= min_log10_radius:
        raise ValueError("exponential-map radius range must be increasing")
    y_fraction = (
        (float(output_height - 1) / 2.0 - np.arange(output_height, dtype=np.float64))
        / float(output_height)
    )
    x_fraction = (
        (np.arange(output_width, dtype=np.float64) - float(output_width - 1) / 2.0)
        / float(output_width)
    )
    xx, yy = np.meshgrid(x_fraction, y_fraction)
    radius_fraction = np.hypot(xx, yy)
    valid = radius_fraction > 0.0
    log_radius = np.full(radius_fraction.shape, float(min_log10_radius), dtype=np.float64)
    log_radius[valid] = (
        math.log10(2.8)
        - float(log10_zoom)
        + np.log10(radius_fraction[valid])
    )
    radial_position = (
        (log_radius - float(min_log10_radius))
        / (float(max_log10_radius) - float(min_log10_radius))
        * source.shape[0]
        - 0.5
    )
    angular_position = (
        (np.arctan2(yy, xx) % (2.0 * math.pi))
        / (2.0 * math.pi)
        * source.shape[1]
        - 0.5
    ) % source.shape[1]
    r0 = np.clip(np.floor(radial_position).astype(np.int64), 0, source.shape[0] - 1)
    r1 = np.clip(r0 + 1, 0, source.shape[0] - 1)
    rw = np.clip(radial_position - np.floor(radial_position), 0.0, 1.0)
    a0 = np.floor(angular_position).astype(np.int64) % source.shape[1]
    a1 = (a0 + 1) % source.shape[1]
    aw = angular_position - np.floor(angular_position)
    top = source[r0, a0] * (1.0 - aw) + source[r0, a1] * aw
    bottom = source[r1, a0] * (1.0 - aw) + source[r1, a1] * aw
    result = (top * (1.0 - rw) + bottom * rw).astype(np.float32)
    result[~valid] = float(max_iter)
    outside = (log_radius < min_log10_radius) | (log_radius > max_log10_radius)
    result[outside & valid] = float(max_iter)
    return result


@dataclass
class AudioFeatures:
    """Frame-aligned controls used by the visual animation."""

    vocal: Any
    instrumental: Any
    phase: Any
    pitch: Any
    # The vocal/full-mix driver used by the original liquid gradient.  Keep it
    # separate from the attack/release-shaped control used for zoom decisions:
    # percentile clamping is useful for stable punches, but it can flatten a
    # perfectly usable quiet vocal passage into a grey three-channel palette.
    gradient: Any
    frame_count: int
    # Normalised onset strength at video-frame timestamps. Optional keeps the
    # dataclass compatible with small external callers that construct it.
    onset: Any = None


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("NumPy is required. Install it with: pip install numpy") from exc
    return np


def _require_librosa() -> Any:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "librosa is required for audio analysis. Install it with: pip install librosa"
        ) from exc
    return librosa


def _subprocess_group_options() -> dict[str, Any]:
    """Start optional tools in a private process group for reliable cleanup."""

    if os.name == "nt":
        return {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        }
    return {"start_new_session": True}


def _terminate_subprocess(process: Any, timeout: float = 2.0) -> None:
    """Stop a child and any descendants without allowing cleanup to hang."""

    try:
        if process.poll() is not None:
            return
    except (AttributeError, OSError):
        return
    if os.name == "nt":
        try:
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
        except (AttributeError, OSError, ValueError):
            try:
                process.terminate()
            except (AttributeError, OSError):
                return
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (AttributeError, OSError, ProcessLookupError):
            try:
                process.terminate()
            except (AttributeError, OSError):
                return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            try:
                process.kill()
            except (AttributeError, OSError):
                return
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (AttributeError, OSError, ProcessLookupError):
                try:
                    process.kill()
                except (AttributeError, OSError):
                    return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # The OS owns the final cleanup now. Do not make the caller wait
            # forever for a broken child that ignored both signals.
            return


def _read_bounded_line(stream: Any, limit: int) -> Optional[str]:
    """Read one subprocess line without buffering an unterminated flood."""

    chunk = stream.readline(limit + 1)
    if not chunk:
        return None
    is_bytes = isinstance(chunk, bytes)
    newline = b"\n" if is_bytes else "\n"
    truncated = len(chunk) > limit
    if truncated and not chunk.endswith(newline):
        # ``readline(size)`` stops at the size limit. Drain the remainder of
        # this logical line in equally bounded pieces so a child that never
        # emits a newline cannot grow the reader's temporary allocation.
        while True:
            remainder = stream.readline(limit + 1)
            if not remainder or remainder.endswith(newline):
                break
    if truncated:
        chunk = chunk[:limit]
        chunk += (
            b"... [line truncated]\n"
            if is_bytes
            else "… [line truncated]\n"
        )
    if is_bytes:
        return chunk.decode("utf-8", errors="replace")
    return str(chunk)


def _start_process_with_diagnostics(
    command: list[str],
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> tuple[Any, deque[str], threading.Thread]:
    """Start a child and continuously drain a bounded diagnostic buffer."""

    diagnostics: deque[str] = deque(maxlen=80)
    process = subprocess.Popen(
        command,
        stdin=stdin,
        stdout=stdout,
        stderr=subprocess.PIPE,
        **_subprocess_group_options(),
    )
    stream = process.stderr

    def drain_stderr() -> None:
        if stream is None:
            return
        try:
            while True:
                line = _read_bounded_line(stream, MAX_DIAGNOSTIC_LINE_CHARS)
                if line is None:
                    break
                diagnostics.append(line)
        except (OSError, ValueError):
            # Process teardown can close the descriptor while the reader is
            # between lines. The exit status remains the authoritative result.
            return
        finally:
            try:
                stream.close()
            except OSError:
                pass

    reader = threading.Thread(
        target=drain_stderr,
        name="fractal-ffmpeg-diagnostics",
        daemon=True,
    )
    try:
        reader.start()
    except BaseException:
        _terminate_subprocess(process)
        raise
    return process, diagnostics, reader


def _start_ffmpeg_process(
    command: list[str],
) -> tuple[Any, deque[str], threading.Thread]:
    """Start FFmpeg and continuously drain a bounded diagnostic buffer."""

    return _start_process_with_diagnostics(command, stdin=subprocess.PIPE)


def _diagnostic_snapshot(diagnostics: Optional[deque[str]]) -> tuple[str, ...]:
    """Take a race-tolerant snapshot of a background stderr ring buffer."""

    if diagnostics is None:
        return ()
    try:
        return tuple(diagnostics.copy())
    except (AttributeError, RuntimeError):
        # The normal object is a deque, whose copy operation is safe while the
        # reader thread appends.  A defensive fallback keeps error reporting
        # from masking the original process failure for test doubles or an
        # unusual deque implementation.
        return ()


def _ffmpeg_error_message(
    prefix: str,
    diagnostics: Optional[deque[str]] = None,
    return_code: Optional[int] = None,
    reader: Optional[threading.Thread] = None,
) -> str:
    """Format a compact FFmpeg failure without allowing stderr to grow output."""

    if reader is not None:
        reader.join(timeout=2.0)
    status = "" if return_code is None else f" (status {return_code})"
    details = "".join(_diagnostic_snapshot(diagnostics)).strip()
    if len(details) > 4000:
        details = details[-4000:]
        details = "[diagnostics truncated]\n" + details
    return f"{prefix}{status}" + (f":\n{details}" if details else "")


def _wait_for_ffmpeg(
    process: Any,
    diagnostics: Optional[deque[str]],
    reader: Optional[threading.Thread],
) -> int:
    """Wait for FFmpeg to finalize without allowing a broken child to hang."""

    try:
        return int(process.wait(timeout=FFMPEG_FINALIZE_TIMEOUT_SECONDS))
    except subprocess.TimeoutExpired as error:
        _terminate_subprocess(process)
        raise RuntimeError(
            _ffmpeg_error_message(
                f"ffmpeg exceeded the {FFMPEG_FINALIZE_TIMEOUT_SECONDS / 60.0:.0f}-minute "
                "finalization timeout",
                diagnostics,
                process.returncode,
                reader,
            )
        ) from error


class _FFmpegFrameWriter:
    """Write video frames through a bounded queue with stall detection."""

    def __init__(
        self,
        process: Any,
        diagnostics: Optional[deque[str]],
        reader: Optional[threading.Thread],
    ) -> None:
        if process.stdin is None:
            raise RuntimeError("FFmpeg did not provide a video input pipe")
        self.process = process
        self.diagnostics = diagnostics
        self.reader = reader
        self.stream = process.stdin
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=3)
        self.errors: list[BaseException] = []
        self.finished = threading.Event()
        self.progress = time.monotonic()
        self.last_backpressure_notice = 0.0
        self.thread = threading.Thread(
            target=self._worker,
            name="fractal-encoder-writer",
            daemon=True,
        )
        try:
            self.thread.start()
        except BaseException:
            _terminate_subprocess(process)
            raise

    def _drain_pending(self) -> None:
        # Items still queued have no worker left to acknowledge them after a
        # write failure. Mark them done so shutdown cannot deadlock in a join.
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return
            else:
                self.queue.task_done()

    def _worker(self) -> None:
        try:
            while True:
                frame = self.queue.get()
                try:
                    if frame is None:
                        self.finished.set()
                        return
                    self.stream.write(frame)
                    self.progress = time.monotonic()
                finally:
                    self.queue.task_done()
        except BaseException as error:  # surfaced by the producer thread
            self.errors.append(error)
            self._drain_pending()

    def _failure(self, prefix: str) -> RuntimeError:
        return RuntimeError(
            _ffmpeg_error_message(
                prefix,
                self.diagnostics,
                self.process.poll(),
                self.reader,
            )
        )

    def check_health(self) -> None:
        if self.errors:
            raise self._failure("FFmpeg writer failed") from self.errors[0]
        return_code = self.process.poll()
        if return_code is not None:
            raise self._failure("ffmpeg exited before the frame queue drained")
        if time.monotonic() - self.progress > FFMPEG_STDIN_STALL_TIMEOUT_SECONDS:
            _terminate_subprocess(self.process)
            raise self._failure("FFmpeg stopped draining video input for too long")

    def write(self, frame: Any) -> None:
        np = _require_numpy()
        contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
        while True:
            self.check_health()
            try:
                self.queue.put(memoryview(contiguous), timeout=0.5)
                return
            except queue.Full:
                now = time.monotonic()
                if now - self.last_backpressure_notice >= ENCODER_BACKPRESSURE_NOTICE_SECONDS:
                    print(
                        "  Encoder backpressure: native/compositor is "
                        "waiting for FFmpeg to drain the frame queue.",
                        flush=True,
                    )
                    self.last_backpressure_notice = now

    def _put_sentinel(self) -> None:
        while True:
            self.check_health()
            try:
                self.queue.put(None, timeout=0.5)
                return
            except queue.Full:
                continue

    def finish(self) -> int:
        self._put_sentinel()
        while not self.finished.wait(timeout=0.5):
            self.check_health()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError("FFmpeg writer thread did not shut down")
        if self.errors:
            raise self._failure("FFmpeg writer failed") from self.errors[0]
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        return _wait_for_ffmpeg(self.process, self.diagnostics, self.reader)

    def abort(self) -> None:
        """Best-effort queue shutdown for an outer render failure."""

        if not self.thread.is_alive():
            return
        try:
            self.queue.put(None, timeout=0.5)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)


def _normalise_path(path: Path) -> Path:
    """Return a stable absolute path without requiring the target to exist."""

    candidate = Path(path).expanduser()
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        # A broken symlink loop or an unusual network filesystem should not
        # make validation itself fail. ``absolute`` still gives us a useful
        # collision check for ordinary paths.
        return Path(os.path.abspath(str(candidate)))


def _absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without following its final symlink."""

    candidate = Path(path).expanduser()
    try:
        return Path(os.path.abspath(str(candidate)))
    except (OSError, RuntimeError, TypeError, ValueError):
        return candidate.absolute()


def _reject_final_symlink(path: Path, label: str) -> None:
    """Reject a final symlink before an atomic writer chooses its target."""

    candidate = Path(path).expanduser()
    try:
        if candidate.is_symlink():
            raise ValueError(f"{label} path must not be a symbolic link: {candidate}")
    except OSError as error:
        raise ValueError(f"could not inspect {label} path {candidate}: {error}") from error


def _paths_refer_to_same_target(first: Path, second: Path) -> bool:
    """Detect lexical, symlink, and hard-link aliases for two path targets."""

    first = _normalise_path(first)
    second = _normalise_path(second)
    if first == second:
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def _path_is_within_directory(path: Path, directory: Path) -> bool:
    """Return whether a path is inside a directory, including the directory."""

    try:
        _normalise_path(path).relative_to(_normalise_path(directory))
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _read_cache_bytes(stream: Any, size: int) -> Optional[bytes]:
    """Read a small cache header completely without trusting one short read."""

    if size < 0 or size > MAX_CACHE_HEADER_BYTES:
        return None
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        block = stream.read(remaining)
        if not block or len(block) > remaining:
            return None
        chunks.append(bytes(block))
        remaining -= len(block)
    return b"".join(chunks)


def _cache_npy_payload_is_safe(stream: Any, member_size: int) -> bool:
    """Validate a NumPy header before a claimed shape can trigger allocation."""

    # The header is parsed with literal_eval, never executed. The subsequent
    # dtype check deliberately accepts only plain numeric arrays: cache files
    # have no legitimate object, structured, or subarray payloads.
    magic = _read_cache_bytes(stream, 6)
    version = _read_cache_bytes(stream, 2)
    if magic != b"\x93NUMPY" or version is None or len(version) != 2:
        return False
    major, minor = version
    if minor != 0 or major not in {1, 2, 3}:
        return False
    if major == 1:
        raw_length = _read_cache_bytes(stream, 2)
        if raw_length is None or len(raw_length) != 2:
            return False
        header_length = struct.unpack("<H", raw_length)[0]
        preamble_size = 10
    else:
        raw_length = _read_cache_bytes(stream, 4)
        if raw_length is None or len(raw_length) != 4:
            return False
        header_length = struct.unpack("<I", raw_length)[0]
        preamble_size = 12
    if header_length > MAX_CACHE_HEADER_BYTES:
        return False
    header = _read_cache_bytes(stream, header_length)
    if header is None:
        return False
    try:
        metadata = ast.literal_eval(header.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return False
    if not isinstance(metadata, dict) or set(metadata) != {
        "descr", "fortran_order", "shape",
    }:
        return False
    if not isinstance(metadata["descr"], str) or not isinstance(
        metadata["fortran_order"], bool
    ):
        return False
    shape = metadata["shape"]
    if not isinstance(shape, tuple) or len(shape) > 2:
        return False
    element_count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            return False
        if dimension > MAX_CACHE_ARRAY_ELEMENTS:
            return False
        if dimension and element_count > MAX_CACHE_ARRAY_ELEMENTS // dimension:
            return False
        element_count *= dimension
    try:
        np = _require_numpy()
        dtype = np.dtype(metadata["descr"])
    except (ImportError, TypeError, ValueError):
        return False
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.kind not in {"f", "i", "u"}
        or dtype.itemsize <= 0
        or dtype.itemsize > 8
    ):
        return False
    payload_size = element_count * int(dtype.itemsize)
    data_offset = preamble_size + header_length
    return data_offset <= member_size and payload_size <= member_size - data_offset


def _open_cache_stream(path: Path) -> tuple[Any, int]:
    """Open a cache entry safely and return its stream plus stable file size."""

    candidate = Path(path)
    if candidate.is_symlink():
        raise OSError("cache entry is a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    # Linux/Unix can make the final-component check atomic with the open. The
    # explicit is_symlink check above remains the fallback on platforms that
    # do not expose O_NOFOLLOW.
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("cache entry is not a regular file")
        file_size = int(metadata.st_size)
        if file_size < 0 or file_size > MAX_CACHE_FILE_BYTES:
            raise ValueError("cache entry exceeds the size limit")
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return stream, file_size


def _cache_archive_directory_is_bounded(stream: Any, file_size: int) -> bool:
    """Reject an oversized ZIP central directory before ``ZipFile`` parses it."""

    # ``ZipFile`` builds one ZipInfo object per central-directory entry during
    # construction. Inspect the fixed-size end record first so a hostile NPZ
    # with millions of tiny members cannot force that allocation before our
    # member-count check runs. ZIP64 archives necessarily report 0xffff here;
    # under the 512 MiB cache-file limit such an entry count is not legitimate
    # for our 16-member cache format, so reject it conservatively.
    end_record_size = 22
    if file_size < end_record_size:
        return False
    tail_size = min(
        int(file_size),
        end_record_size + MAX_CACHE_ARCHIVE_COMMENT_BYTES,
    )
    stream.seek(file_size - tail_size)
    tail = stream.read(tail_size)
    if not isinstance(tail, bytes) or len(tail) != tail_size:
        return False
    signature = b"PK\x05\x06"
    for offset in range(len(tail) - end_record_size, -1, -1):
        if tail[offset : offset + 4] != signature:
            continue
        comment_length = struct.unpack_from("<H", tail, offset + 20)[0]
        if offset + end_record_size + comment_length != len(tail):
            continue
        member_count = struct.unpack_from("<H", tail, offset + 10)[0]
        return member_count <= MAX_CACHE_ARCHIVE_MEMBERS
    return False


def _cache_stream_is_safe(path: Path, stream: Any, file_size: int) -> bool:
    """Validate one already-open cache stream before deserialization."""

    suffix = path.suffix.casefold()
    if suffix == ".npy":
        return _cache_npy_payload_is_safe(stream, file_size)
    if suffix != ".npz":
        return False
    if not _cache_archive_directory_is_bounded(stream, file_size):
        return False
    with zipfile.ZipFile(stream) as archive:
        members = archive.infolist()
        if not members or len(members) > MAX_CACHE_ARCHIVE_MEMBERS:
            return False
        expanded_size = 0
        for member in members:
            if (
                member.is_dir()
                or not member.filename.casefold().endswith(".npy")
                or member.file_size < 0
                or member.file_size > MAX_CACHE_FILE_BYTES
            ):
                return False
            expanded_size += int(member.file_size)
            if expanded_size > MAX_CACHE_FILE_BYTES:
                return False
            with archive.open(member, "r") as member_stream:
                if not _cache_npy_payload_is_safe(member_stream, int(member.file_size)):
                    return False
        return True


def _cache_file_is_safe(path: Path) -> bool:
    """Reject oversized, deceptive, or zip-bomb-like cache entries."""

    try:
        stream, file_size = _open_cache_stream(path)
        try:
            return _cache_stream_is_safe(Path(path), stream, file_size)
        finally:
            stream.close()
    except (
        EOFError,
        KeyError,
        MemoryError,
        OSError,
        OverflowError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
    ):
        return False


@contextmanager
def _safe_cache_load(path: Path):
    """Validate and deserialize a cache entry through the same open handle."""

    stream, file_size = _open_cache_stream(path)
    cached = None
    try:
        try:
            safe = _cache_stream_is_safe(Path(path), stream, file_size)
        except (
            EOFError,
            KeyError,
            MemoryError,
            OSError,
            OverflowError,
            RuntimeError,
            ValueError,
            zipfile.BadZipFile,
        ):
            safe = False
        if not safe:
            yield None
            return
        stream.seek(0)
        np = _require_numpy()
        cached = np.load(stream, allow_pickle=False)
        yield cached
    finally:
        if cached is not None and hasattr(cached, "close"):
            try:
                cached.close()
            except (OSError, ValueError):
                pass
        stream.close()


def _temporary_sibling(path: Path, label: str) -> Path:
    """Return a unique same-directory temporary path for atomic output.

    The path is deliberately not created here. Callers can perform additional
    setup (native reference construction, cache creation, and encoder probing)
    after choosing it; avoiding a placeholder file means an exception during
    that setup cannot leave an orphaned output beside the user's target.
    """

    path = _absolute_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(label)
    ) or "temporary"
    return path.with_name(
        f".{path.name}.{safe_label}-{uuid.uuid4().hex}{path.suffix}"
    )


def _reserved_temporary_sibling(path: Path, label: str) -> Path:
    """Reserve a same-directory temporary file with exclusive creation."""

    for _ in range(3):
        candidate = _temporary_sibling(path, label)
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            continue
        try:
            os.close(descriptor)
        except OSError:
            try:
                candidate.unlink()
            except OSError:
                pass
            raise
        return candidate
    raise OSError("could not reserve a unique temporary output path")


def _validate_render_paths(
    audio_path: Path,
    output_path: Path,
    manifest_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
    """Validate all paths before analysis/encoding can mutate anything."""

    _reject_final_symlink(output_path, "output")
    if manifest_path is not None:
        _reject_final_symlink(manifest_path, "manifest")
    audio = _normalise_path(audio_path)
    output = _absolute_path(output_path)
    manifest = _absolute_path(manifest_path) if manifest_path is not None else None
    cache = _normalise_path(cache_dir) if cache_dir is not None else None

    if not audio.is_file():
        raise ValueError(f"audio file not found: {audio}")
    if output.exists() and output.is_dir():
        raise ValueError(f"output path is a directory: {output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise ValueError(f"output parent is not a directory: {output.parent}")
    if _paths_refer_to_same_target(audio, output):
        raise ValueError("output path must be different from the input audio file")

    if manifest is not None:
        if manifest.exists() and manifest.is_dir():
            raise ValueError(f"manifest path is a directory: {manifest}")
        if manifest.parent.exists() and not manifest.parent.is_dir():
            raise ValueError(f"manifest parent is not a directory: {manifest.parent}")
        if _paths_refer_to_same_target(manifest, audio):
            raise ValueError("manifest path must be different from the input audio file")
        if _paths_refer_to_same_target(manifest, output):
            raise ValueError("manifest path must be different from the output video")

    if cache is not None:
        if cache.exists() and not cache.is_dir():
            raise ValueError(f"cache path is not a directory: {cache}")
        if cache.parent.exists() and not cache.parent.is_dir():
            raise ValueError(f"cache parent is not a directory: {cache.parent}")
        for label, candidate in (
            ("input audio", audio),
            ("output", output),
            ("manifest", manifest),
        ):
            if candidate is not None and _path_is_within_directory(candidate, cache):
                raise ValueError(
                    f"cache directory must not contain the {label} path"
                )
        if _paths_refer_to_same_target(cache, output):
            raise ValueError("cache directory must be different from the output video")
        if manifest is not None and _paths_refer_to_same_target(cache, manifest):
            raise ValueError("cache directory must be different from the manifest file")

    return audio, output, manifest, cache


def _index_value(value: Any, label: str) -> int:
    try:
        return int(operator.index(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _validate_iteration_count(value: Any, label: str = "iteration count") -> int:
    result = _index_value(value, label)
    if result <= 0 or result > MAX_ITERATION_BUDGET:
        raise ValueError(
            f"{label} must be between 1 and {MAX_ITERATION_BUDGET:,}"
        )
    return result


def _validate_thread_count(value: Any, label: str = "thread count") -> int:
    result = _index_value(value, label)
    if result < 0 or result > MAX_THREAD_COUNT:
        raise ValueError(f"{label} must be between 0 and {MAX_THREAD_COUNT:,}")
    return result


def _validate_fps(value: Any, label: str = "fps") -> int:
    result = _index_value(value, label)
    if result <= 0 or result > MAX_FPS:
        raise ValueError(f"{label} must be between 1 and {MAX_FPS:,}")
    return result


def _validate_log10_zoom(value: Any, label: str = "zoom exponent") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result) or not MIN_LOG10_ZOOM <= result <= MAX_LOG10_ZOOM:
        raise ValueError(
            f"{label} must be between {MIN_LOG10_ZOOM:.0f} and "
            f"{MAX_LOG10_ZOOM:.0f}"
        )
    return result


def _validate_keyframe_factor(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("keyframe factor must be numeric") from error
    if not math.isfinite(result) or result <= 1.0:
        raise ValueError("keyframe factor must be finite and greater than 1")
    return result


def _validate_ffmpeg_token(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    token = value.strip()
    if (
        not token
        or token.startswith("-")
        or any(character.isspace() for character in token)
        or len(token) > 128
    ):
        raise ValueError(f"{label} must be one non-empty FFmpeg token")
    return token


def _validate_dimensions(width: Any, height: Any, label: str = "render") -> tuple[int, int]:
    width_value = _index_value(width, f"{label} width")
    height_value = _index_value(height, f"{label} height")
    if width_value <= 0 or height_value <= 0:
        raise ValueError(f"{label} width and height must be positive")
    if width_value * height_value > MAX_RENDER_PIXELS:
        raise ValueError(
            f"{label} dimensions contain too many pixels; maximum is {MAX_RENDER_PIXELS:,}"
        )
    return width_value, height_value


def _scaled_dimensions(
    width: Any,
    height: Any,
    scale: Any,
    label: str = "fractal source",
) -> tuple[int, int]:
    """Scale a frame size without allowing float/int overflow to escape."""

    width_value, height_value = _validate_dimensions(width, height, label)
    try:
        scale_value = float(scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} scale must be numeric") from error
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"{label} scale must be finite and positive")
    scaled_width = float(width_value) * scale_value
    scaled_height = float(height_value) * scale_value
    if not math.isfinite(scaled_width) or not math.isfinite(scaled_height):
        raise ValueError(f"{label} dimensions are too large")
    try:
        result_width = max(16, int(round(scaled_width)))
        result_height = max(16, int(round(scaled_height)))
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} dimensions are too large") from error
    return _validate_dimensions(result_width, result_height, label)


def _validate_numeric_series(
    values: Any,
    frame_count: int,
    name: str,
    *,
    bounded: bool = False,
) -> Any:
    np = _require_numpy()
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric samples") from error
    if array.ndim != 1 or array.size != frame_count:
        raise ValueError(f"{name} must contain exactly {frame_count} one-dimensional samples")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain real numeric samples")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric samples") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite samples")
    if bounded and (float(np.min(array)) < -1.0e-6 or float(np.max(array)) > 1.0 + 1.0e-6):
        raise ValueError(f"{name} samples must be between 0 and 1")
    return array.astype(np.float32, copy=False)


def _validate_audio_features(features: AudioFeatures) -> int:
    """Validate externally supplied controls before they reach the encoder."""

    try:
        frame_count = _index_value(features.frame_count, "audio frame count")
    except AttributeError as exc:
        raise ValueError("features must be an AudioFeatures-like object") from exc
    if frame_count <= 0 or frame_count > MAX_AUDIO_FRAMES:
        raise ValueError(
            f"audio frame count must be between 1 and {MAX_AUDIO_FRAMES:,}"
        )
    for name in ("vocal", "instrumental", "pitch", "gradient"):
        try:
            values = getattr(features, name)
        except AttributeError as exc:
            raise ValueError(f"features is missing the {name} control") from exc
        _validate_numeric_series(values, frame_count, name, bounded=True)
    try:
        phase = features.phase
    except AttributeError as exc:
        raise ValueError("features is missing the phase control") from exc
    _validate_numeric_series(phase, frame_count, "phase")
    onset = getattr(features, "onset", None)
    if onset is not None:
        _validate_numeric_series(onset, frame_count, "onset", bounded=True)
    return frame_count


def _normalise_audio_features(features: AudioFeatures) -> AudioFeatures:
    """Copy validated controls so rendering cannot mutate its caller's data."""

    np = _require_numpy()
    frame_count = _validate_audio_features(features)
    controls = {
        name: np.array(getattr(features, name), dtype=np.float32, copy=True)
        for name in ("vocal", "instrumental", "phase", "pitch", "gradient")
    }
    onset_value = getattr(features, "onset", None)
    onset = (
        None
        if onset_value is None
        else np.array(onset_value, dtype=np.float32, copy=True)
    )
    return AudioFeatures(
        controls["vocal"],
        controls["instrumental"],
        controls["phase"],
        controls["pitch"],
        controls["gradient"],
        frame_count,
        onset,
    )


def _validate_zoom_series(zooms: Any, frame_count: int) -> Any:
    np = _require_numpy()
    try:
        array = np.asarray(zooms)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("zoom path must contain real numeric samples") from error
    if array.ndim != 1 or array.size != frame_count or array.size == 0:
        raise ValueError(f"zoom path must contain exactly {frame_count} one-dimensional samples")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError("zoom path must contain real numeric samples")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("zoom path must contain real numeric samples") from error
    if not np.isfinite(array).all():
        raise ValueError("zoom path contains non-finite samples")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < MIN_LOG10_ZOOM or maximum > MAX_LOG10_ZOOM:
        raise ValueError(
            f"zoom path must stay between 10^{MIN_LOG10_ZOOM:.0f} and "
            f"10^{MAX_LOG10_ZOOM:.0f}"
        )
    return array


def _cached_audio_features(cached: Any) -> Optional[AudioFeatures]:
    """Read a cache entry only after validating every stored control array."""

    np = _require_numpy()
    frame_value = np.asarray(cached["frame_count"])
    if frame_value.size != 1:
        raise ValueError("cached audio frame count is not scalar")
    if not np.issubdtype(frame_value.dtype, np.integer):
        raise ValueError("cached audio frame count is not an integer")
    frame_count = int(frame_value.reshape(-1)[0])
    if frame_count <= 0 or frame_count > MAX_AUDIO_FRAMES:
        raise ValueError("cached audio frame count is out of range")
    vocal = _validate_numeric_series(cached["vocal"], frame_count, "cached vocal", bounded=True)
    instrumental = _validate_numeric_series(
        cached["instrumental"], frame_count, "cached instrumental", bounded=True
    )
    phase = _validate_numeric_series(cached["phase"], frame_count, "cached phase")
    pitch = _validate_numeric_series(cached["pitch"], frame_count, "cached pitch", bounded=True)
    gradient = _validate_numeric_series(
        cached["gradient"], frame_count, "cached gradient", bounded=True
    )
    onset = _validate_numeric_series(cached["onset"], frame_count, "cached onset", bounded=True)
    return AudioFeatures(vocal, instrumental, phase, pitch, gradient, frame_count, onset)


def _load_audio(path: Path, sample_rate: int) -> Any:
    np = _require_numpy()
    librosa = _require_librosa()
    sample_rate = _index_value(sample_rate, "sample rate")
    if sample_rate <= 0 or sample_rate > MAX_SAMPLE_RATE:
        raise ValueError(f"sample rate must be between 1 and {MAX_SAMPLE_RATE:,}")
    try:
        duration = float(librosa.get_duration(path=str(path)))
    except Exception:
        # Some codecs do not expose duration metadata. The decoded-size check
        # below remains authoritative for those files.
        duration = None
    if duration is not None:
        if not math.isfinite(duration) or duration < 0.0:
            raise RuntimeError(f"audio decoder returned an invalid duration: {path}")
        if duration * sample_rate > MAX_AUDIO_SAMPLES:
            raise RuntimeError(
                f"audio is too long at {sample_rate:,} Hz; decoded samples may not exceed "
                f"{MAX_AUDIO_SAMPLES:,}"
            )
    try:
        # Ask the decoder for one sentinel sample beyond the accepted budget.
        # This bounds backends that honour the duration argument while still
        # letting us reject a file whose true decoded length exceeds the limit
        # when metadata was unavailable or inaccurate.
        decode_duration = (MAX_AUDIO_SAMPLES + 1) / float(sample_rate)
        samples, _ = librosa.load(
            str(path),
            sr=sample_rate,
            mono=True,
            duration=decode_duration,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not decode audio file {path}: {exc}") from exc
    if samples.size == 0:
        raise RuntimeError(f"Audio file is empty: {path}")
    if samples.size > MAX_AUDIO_SAMPLES:
        raise RuntimeError(
            f"audio contains too many decoded samples; maximum is {MAX_AUDIO_SAMPLES:,}"
        )
    samples = np.asarray(samples, dtype=np.float32)
    if not np.isfinite(samples).all():
        # A damaged decoder window should not poison every downstream
        # percentile and phase calculation. Treat non-finite samples as
        # silence while retaining the rest of the track.
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(samples).all():
        raise RuntimeError(f"Audio decoder returned non-finite samples: {path}")
    return samples


def _demucs_is_available() -> bool:
    """Probe the optional separator without letting a broken install abort auto mode."""

    try:
        return importlib.util.find_spec("demucs") is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def _audio_cache_path(
    audio_path: Path,
    sample_rate: int,
    fps: int,
    separation: str,
    attack: float,
    release: float,
    cache_dir: Optional[Path],
) -> Optional[Path]:
    if cache_dir is None:
        return None
    digest = hashlib.sha256()
    digest.update(AUDIO_CACHE_SCHEMA.encode("ascii"))
    digest.update(str(sample_rate).encode("ascii"))
    digest.update(str(fps).encode("ascii"))
    digest.update(f"{float(attack):.9g},{float(release):.9g}".encode("ascii"))
    separation_signature = separation
    if separation == "auto":
        separation_signature += "-demucs" if _demucs_is_available() else "-fullmix"
    digest.update(separation_signature.encode("ascii"))
    with audio_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return cache_dir / f"audio-{digest.hexdigest()[:24]}.npz"


def _atomic_save_features(path: Path, features: AudioFeatures) -> None:
    np = _require_numpy()
    path = _absolute_path(path)
    _reject_final_symlink(path, "cache entry")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez(
                handle,
                vocal=np.asarray(features.vocal, dtype=np.float32),
                instrumental=np.asarray(features.instrumental, dtype=np.float32),
                phase=np.asarray(features.phase, dtype=np.float32),
                pitch=np.asarray(features.pitch, dtype=np.float32),
                gradient=np.asarray(features.gradient, dtype=np.float32),
                onset=np.asarray(
                    features.onset
                    if features.onset is not None
                    else np.zeros(features.frame_count, dtype=np.float32),
                    dtype=np.float32,
                ),
                frame_count=np.asarray(features.frame_count, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _find_demucs_stems(root: Path) -> Optional[tuple[Path, Path]]:
    """Return regular stem files that remain inside Demucs' private directory."""

    try:
        root = root.resolve(strict=True)
    except OSError:
        return None

    def safe_candidates(name: str) -> list[Path]:
        try:
            candidates = []
            for index, candidate in enumerate(root.rglob(name)):
                if index >= MAX_DEMUCS_DISCOVERY_ENTRIES:
                    return []
                candidates.append(candidate)
        except (OSError, RuntimeError):
            return []
        safe: list[Path] = []
        for candidate in candidates:
            try:
                # A separator is an optional third-party process. Do not let a
                # compromised/broken install make the parent decode an
                # arbitrary symlink target outside its private temp tree.
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            safe.append(resolved)
        return sorted(safe)

    vocals = safe_candidates("vocals.wav")
    no_vocals = safe_candidates("no_vocals.wav")
    if vocals and no_vocals:
        return vocals[0], no_vocals[0]
    # Some Demucs versions use ``instrumental.wav`` for the second stem.
    instruments = safe_candidates("instrumental.wav")
    if vocals and instruments:
        return vocals[0], instruments[0]
    return None


def _demucs_stems(audio_path: Path, output_dir: Path, mode: str) -> Optional[tuple[Path, Path]]:
    """Run Demucs if requested/available and return its two stem paths."""

    available = _demucs_is_available()
    if not available:
        if mode == "demucs":
            raise RuntimeError(
                "--separation demucs was requested, but Demucs is not installed. "
                "Install it with: pip install demucs"
            )
        print("Demucs is not installed; using full-song control.")
        return None

    # Demucs receives the original path rather than the bounded NumPy
    # samples. If a decoder cannot report duration, running it would let an
    # otherwise bounded render hand an unbounded source to a third-party
    # process. Auto mode can safely fall back; explicit mode gets an actionable
    # error instead.
    try:
        duration = float(_require_librosa().get_duration(path=str(audio_path)))
    except Exception as error:
        message = "could not determine audio duration safely for Demucs"
        if mode == "demucs":
            raise RuntimeError(message) from error
        print(f"{message}; using full-song control.")
        return None
    if not math.isfinite(duration) or duration < 0.0:
        message = "audio duration metadata is invalid for Demucs"
        if mode == "demucs":
            raise RuntimeError(message)
        print(f"{message}; using full-song control.")
        return None

    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems=vocals",
        "-n",
        "htdemucs",
        "--out",
        str(output_dir),
        str(audio_path),
    ]
    print("Separating vocals and instruments with Demucs...")
    process = None
    diagnostics: Optional[deque[str]] = None
    diagnostic_reader: Optional[threading.Thread] = None
    try:
        process, diagnostics, diagnostic_reader = _start_process_with_diagnostics(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=DEMUCS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            _terminate_subprocess(process)
            message = (
                f"Demucs exceeded the {DEMUCS_TIMEOUT_SECONDS / 60.0:.0f}-minute timeout"
            )
            if mode == "demucs":
                raise RuntimeError(message) from exc
            print(f"{message}; using full-song control.")
            return None
    except OSError as exc:
        message = f"could not start Demucs: {exc}"
        if mode == "demucs":
            raise RuntimeError(message) from exc
        print(f"Demucs {message}; using full-song control.")
        return None
    finally:
        if diagnostic_reader is not None:
            diagnostic_reader.join(timeout=2.0)
    if process is None:
        raise RuntimeError("Demucs process was not started")
    if process.returncode != 0:
        if mode == "demucs":
            details = "".join(_diagnostic_snapshot(diagnostics)).strip()
            if len(details) > 4000:
                details = "[diagnostics truncated]\n" + details[-4000:]
            raise RuntimeError(f"Demucs failed:\n{details}")
        print("Demucs failed; using full-song control.")
        details = "".join(_diagnostic_snapshot(diagnostics)).strip()
        if details:
            print(details)
        return None

    stems = _find_demucs_stems(output_dir)
    if stems is None:
        if mode == "demucs":
            raise RuntimeError("Demucs completed but did not produce vocals.wav and no_vocals.wav")
        print("Demucs produced no usable stems; using full-song control.")
    return stems


def _frame_rms(samples: Any, sample_rate: int, fps: int) -> Any:
    np = _require_numpy()
    librosa = _require_librosa()
    hop = max(1, round(sample_rate / fps))
    frame_length = max(1024, 2 * hop)
    values = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop,
        center=True,
    )[0]
    return np.asarray(values, dtype=np.float32)


def _frame_pitch(samples: Any, sample_rate: int, fps: int) -> Any:
    """Estimate a smooth 0..1 pitch control from a mono signal.

    YIN is deliberately used instead of a raw spectral centroid: the latter
    follows cymbals and broadband transients as if they were notes.  Invalid
    or unvoiced windows use the median detected pitch, producing the stable
    median-pitch gradient rather than sudden blue/yellow jumps.
    """

    np = _require_numpy()
    librosa = _require_librosa()
    hop = max(1, round(sample_rate / fps))
    frame_length = max(2048, 2 * hop)
    fmin = 55.0
    fmax = min(1600.0, sample_rate * 0.5 - 20.0)
    if fmax <= fmin:
        return np.full(1, 0.5, dtype=np.float32)
    try:
        frequencies = librosa.yin(
            samples,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop,
            center=True,
        )
    except Exception:
        # Pitch is an enhancement; a broken/old librosa pitch implementation
        # must not prevent the visualizer from rendering the song.
        return np.full(1, 0.5, dtype=np.float32)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    valid = np.isfinite(frequencies) & (frequencies >= fmin) & (frequencies <= fmax)
    if int(np.count_nonzero(valid)) < 3:
        return np.full(max(1, frequencies.size), 0.5, dtype=np.float32)
    log_frequencies = np.log2(np.clip(frequencies, fmin, fmax))
    median_pitch = float(np.median(log_frequencies[valid]))
    log_frequencies[~valid] = median_pitch
    normalized = (log_frequencies - math.log2(fmin)) / (math.log2(fmax) - math.log2(fmin))
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _frame_onset(samples: Any, sample_rate: int, fps: int) -> Any:
    """Return a normalised spectral-onset curve at the analysis hop rate."""

    np = _require_numpy()
    librosa = _require_librosa()
    hop = max(1, round(sample_rate / fps))
    try:
        strength = librosa.onset.onset_strength(
            y=samples,
            sr=sample_rate,
            hop_length=hop,
            center=True,
        )
    except Exception:
        # Onset sync is optional. A decoder or older librosa release should
        # not prevent the ordinary loudness-driven animation from rendering.
        return np.zeros(1, dtype=np.float32)
    strength = np.asarray(strength, dtype=np.float32)
    if strength.size == 0:
        return np.zeros(1, dtype=np.float32)
    return _normalise_minmax(strength)


def _relative_pitch(values: Any) -> Any:
    """Centre pitch around the song's robust average for colour control.

    The returned 0.0..1.0 value is deliberately centred at 0.5. A pitch equal
    to the song median leaves the legacy gradient unchanged; higher and lower
    notes rotate its two hues in opposite directions.
    Log-frequency pitch is already perceptual, so a robust deviation scale is
    more stable than treating Hz as a linear distance.
    """

    np = _require_numpy()
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, 0.5, dtype=np.float32)
    center = float(np.median(values[finite]))
    deviation = np.nan_to_num(values - center, nan=0.0, posinf=0.0, neginf=0.0)
    spread = float(np.percentile(np.abs(deviation[finite]), 80.0))
    spread = max(spread, 0.08)
    signed = np.clip(deviation / spread, -1.0, 1.0)
    return np.asarray(0.5 + 0.5 * signed, dtype=np.float32)


def _resample_features(values: Any, frame_count: int, fps: int, sample_rate: int) -> Any:
    np = _require_numpy()
    if frame_count <= 0:
        return np.zeros(1, dtype=np.float32)
    if values.size == 1:
        return np.full(frame_count, float(values[0]), dtype=np.float32)
    # librosa features are produced at the rounded integer hop used by
    # _frame_rms/_spectral_fallback.  Using sample_rate/fps here creates a
    # small but cumulative timing drift for frame rates that do not divide the
    # sample rate exactly (for example 59.94/60 fps variants).
    hop = max(1, round(sample_rate / fps))
    source_times = np.arange(values.size, dtype=np.float64) * hop / sample_rate
    target_times = np.arange(frame_count, dtype=np.float64) / fps
    return np.interp(
        target_times,
        source_times,
        values.astype(np.float64),
        left=float(values[0]),
        right=float(values[-1]),
    ).astype(np.float32)


def _normalise(values: Any) -> Any:
    np = _require_numpy()
    values = np.asarray(values, dtype=np.float32)
    if not np.any(np.isfinite(values)):
        return np.zeros_like(values)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(values, [5.0, 98.0])
    if high <= low + 1e-12:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _normalise_minmax(values: Any) -> Any:
    """Normalize a flow control with the original visualizer's full range."""

    np = _require_numpy()
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low + 1.0e-12:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _smooth(values: Any, radius: int) -> Any:
    np = _require_numpy()
    if radius <= 1 or values.size < 3:
        return values
    kernel = np.ones(radius, dtype=np.float32) / radius
    padded = np.pad(values, (radius // 2, radius - 1 - radius // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _envelope_follow(values: Any, fps: int, attack: float, release: float) -> Any:
    """Apply a causal attack/release follower to a normalized control signal."""

    np = _require_numpy()
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0
    )
    if values.size <= 1:
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    def coefficient(seconds: float) -> float:
        if seconds <= 0.0:
            return 0.0
        return math.exp(-1.0 / max(float(fps) * seconds, 1.0))

    attack_coefficient = coefficient(float(attack))
    release_coefficient = coefficient(float(release))
    output = np.empty_like(values, dtype=np.float32)
    previous = float(np.clip(values[0], 0.0, 1.0))
    output[0] = previous
    for index in range(1, values.size):
        target = float(np.clip(values[index], 0.0, 1.0))
        decay = attack_coefficient if target > previous else release_coefficient
        previous = target + decay * (previous - target)
        output[index] = previous
    return output


def _component_has_signal(component: Any, full_mix: Any) -> bool:
    """Return whether a stem/proxy has enough usable energy to drive alone."""

    np = _require_numpy()
    component = np.nan_to_num(np.asarray(component, dtype=np.float64))
    component_p95 = float(np.percentile(np.abs(component), 95.0))
    if component_p95 <= 1e-7:
        return False
    # A Demucs stem and an FFT-band proxy do not share an amplitude unit.  Do
    # not compare their raw percentiles; that made the result depend on the
    # FFT size rather than on whether the component actually had dynamics.
    component_p20 = float(np.percentile(np.abs(component), 20.0))
    return component_p95 - component_p20 > max(1e-7, component_p95 * 0.01)


def _controls_are_distinct(vocal: Any, instrumental: Any) -> bool:
    """Avoid pretending that two nearly identical controls are separated."""

    np = _require_numpy()
    vocal = _normalise(vocal).astype(np.float64)
    instrumental = _normalise(instrumental).astype(np.float64)
    if vocal.size < 2 or instrumental.size < 2:
        return False
    difference = float(np.mean(np.abs(vocal - instrumental)))
    return difference >= 0.05


def _spectral_fallback(samples: Any, sample_rate: int, fps: int) -> tuple[Any, Any]:
    """Estimate vocal/instrument energy without requiring separate files.

    This is intentionally a fallback, not a replacement for a neural source
    separator.  Speech and singing tend to concentrate in the middle band;
    bass, cymbals, and transients provide a useful instrumental control.
    """

    np = _require_numpy()
    librosa = _require_librosa()
    hop = max(1, round(sample_rate / fps))
    n_fft = 2048
    spectrum = np.abs(
        librosa.stft(samples, n_fft=n_fft, hop_length=hop, center=True)
    ).astype(np.float32)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    voice_band = (frequencies >= 250.0) & (frequencies <= 4200.0)
    instrument_band = ~((frequencies >= 450.0) & (frequencies <= 4200.0))

    vocal_energy = np.sqrt(np.mean(spectrum[voice_band] ** 2, axis=0) + 1e-12)
    instrumental_energy = np.sqrt(np.mean(spectrum[instrument_band] ** 2, axis=0) + 1e-12)
    # Keep analysis-rate values here.  The caller owns the one authoritative
    # conversion to exact video-frame timestamps; resampling twice subtly
    # smears transients and shifts the zoom punch.
    return vocal_energy, instrumental_energy


def analyse_audio(
    audio_path: Path,
    sample_rate: int,
    fps: int,
    separation: str,
    cache_dir: Optional[Path] = None,
    attack: float = 0.025,
    release: float = 0.12,
) -> AudioFeatures:
    """Load one song and produce frame-aligned vocal/instrument controls."""

    np = _require_numpy()
    audio_path = _normalise_path(audio_path)
    if not audio_path.is_file():
        raise ValueError(f"audio file not found: {audio_path}")
    sample_rate = _index_value(sample_rate, "sample rate")
    fps = _validate_fps(fps)
    if sample_rate <= 0 or sample_rate > MAX_SAMPLE_RATE:
        raise ValueError(f"sample rate must be between 1 and {MAX_SAMPLE_RATE:,}")
    if separation not in {"auto", "demucs", "spectral", "none"}:
        raise ValueError(f"unknown audio separation mode: {separation}")
    if not math.isfinite(float(attack)) or not math.isfinite(float(release)):
        raise ValueError("audio attack and release must be finite")
    if float(attack) < 0.0 or float(release) < 0.0:
        raise ValueError("audio attack and release cannot be negative")
    if cache_dir is not None:
        cache_dir = _normalise_path(cache_dir)
        if cache_dir.exists() and not cache_dir.is_dir():
            raise ValueError(f"cache path is not a directory: {cache_dir}")
    cache_path = _audio_cache_path(
        audio_path, sample_rate, fps, separation, attack, release, cache_dir
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _safe_cache_load(cache_path) as cached:
                if cached is None:
                    raise ValueError("unsafe cached audio controls")
                cached_features = _cached_audio_features(cached)
                if cached_features is not None:
                    print(f"Using cached audio controls {cache_path.name}.")
                    return cached_features
        except (
            EOFError,
            KeyError,
            MemoryError,
            OSError,
            TypeError,
            ValueError,
            OverflowError,
            zipfile.BadZipFile,
        ):
            pass

    source = _load_audio(audio_path, sample_rate)
    frame_count = max(1, math.ceil(len(source) * fps / sample_rate))
    if frame_count > MAX_AUDIO_FRAMES:
        raise ValueError(
            f"audio produces too many video frames; maximum is {MAX_AUDIO_FRAMES:,}"
        )
    full_mix_rms = _resample_features(_frame_rms(source, sample_rate, fps), frame_count, fps, sample_rate)
    onset = _resample_features(
        _frame_onset(source, sample_rate, fps), frame_count, fps, sample_rate
    )

    with tempfile.TemporaryDirectory(prefix="fractal-demucs-") as temp:
        stems = None
        pitch_source = source
        if separation in {"auto", "demucs"}:
            stems = _demucs_stems(audio_path, Path(temp), separation)

        if stems is not None:
            vocals = _load_audio(stems[0], sample_rate)
            instruments = _load_audio(stems[1], sample_rate)
            pitch_source = vocals
            vocal_rms = _frame_rms(vocals, sample_rate, fps)
            instrumental_rms = _frame_rms(instruments, sample_rate, fps)
            print("Using Demucs vocal and instrumental stems.")
        elif separation == "spectral":
            vocal_rms, instrumental_rms = _spectral_fallback(source, sample_rate, fps)
        else:
            vocal_rms = None
            instrumental_rms = None

    if stems is None and separation != "spectral":
        vocal_signal = full_mix_rms
        instrumental_signal = full_mix_rms
    else:
        vocal_signal = _resample_features(vocal_rms, frame_count, fps, sample_rate)
        instrumental_signal = _resample_features(instrumental_rms, frame_count, fps, sample_rate)
    # Preserve the first visualizer's liquid control independently from the
    # attack/release controls used for camera motion.  Its recipe was simply
    # min/max normalization followed by a short four-frame moving average.
    # The robust zoom normalizer intentionally clips outliers, which is good
    # for punch stability but made low-level full-mix/vocal passages collapse
    # to one phase split and appear grey.
    if stems is None and separation != "spectral":
        gradient = _smooth(_normalise_minmax(full_mix_rms), 4)
    else:
        gradient = _smooth(_normalise_minmax(vocal_signal), 4)
    full_mix = _envelope_follow(
        _smooth(_normalise(full_mix_rms), 5), fps, attack, release
    )
    pitch = _relative_pitch(_smooth(
        _resample_features(_frame_pitch(pitch_source, sample_rate, fps), frame_count, fps, sample_rate),
        5,
    ))
    pitch_radius = max(5, int(round(fps * 0.15)))
    if pitch_radius % 2 == 0:
        pitch_radius += 1
    pitch = _smooth(pitch, pitch_radius)

    vocal_detected = _component_has_signal(vocal_signal, full_mix_rms)
    instrumental_detected = _component_has_signal(instrumental_signal, full_mix_rms)
    separated_controls = (
        vocal_detected
        and instrumental_detected
        and _controls_are_distinct(vocal_signal, instrumental_signal)
    )
    if separated_controls:
        vocal = _envelope_follow(
            _smooth(_normalise(vocal_signal), 5), fps, attack, release
        )
        instrumental = _envelope_follow(
            _smooth(_normalise(instrumental_signal), 7), fps, attack, release
        )
        if stems is not None:
            print("Audio controls: Demucs vocals drive gradient; instruments drive zoom.")
        else:
            print("Audio controls: spectral vocal/instrument proxies drive gradient/zoom.")
    else:
        # This is the important fallback: no detected vocal/instrument split
        # means the song must still animate both dimensions.
        vocal = full_mix
        instrumental = full_mix
        print("No reliable vocal/instrument split; full-song energy drives gradient and zoom.")
    # Match the original visualizer's liquid engine.  ``vocal`` is already
    # attack/release smoothed, so the phase velocity changes continuously even
    # though the original peak rate is intentionally energetic.  When source
    # separation is unavailable ``vocal`` is the full-song envelope above, so
    # a normal one-file render still gets the same flowing gradient.
    flow_strength = np.clip(gradient.astype(np.float64), 0.0, 1.0) ** 2.5
    phase_rate = AURORA_FLOW_SPEED * (
        AURORA_MIN_FLOW_FRACTION
        + (1.0 - AURORA_MIN_FLOW_FRACTION) * flow_strength
    )
    phase = np.cumsum(phase_rate / max(float(fps), 1.0)).astype(np.float32)
    features = AudioFeatures(vocal, instrumental, phase, pitch, gradient, frame_count, onset)
    _validate_audio_features(features)
    if cache_path is not None:
        _atomic_save_features(cache_path, features)
    return features


def _fractional_decimal_places(value: str) -> int:
    """Return the decimal place of the least-significant supplied digit.

    Counting the digits after a literal ``.`` is wrong for scientific input:
    ``1.23e-4`` specifies a value to the 1e-6 place.  ``Decimal`` preserves
    that information without converting a deep coordinate through binary
    floating point.
    """

    try:
        decimal_value = Decimal(str(value).strip())
    except Exception:
        return 0
    if not decimal_value.is_finite():
        return 0
    exponent = decimal_value.as_tuple().exponent
    return max(0, -int(exponent))


def _coordinate_precision_digits(value: str) -> int:
    """Return usable fractional precision, preserving exact zero/integer axes.

    A literal ``0.0`` is not an under-specified deep coordinate: zero is
    exactly representable, and Kalles uses real-axis targets like this for
    several Julia presets.  Likewise, a coordinate written as an integer (or
    with a non-negative scientific exponent) has no omitted fractional part.
    Non-zero decimal fractions still report the least-significant supplied
    digit, because their omitted tail can change a deep zoom target.
    """

    try:
        text = str(value).strip()
        decimal_value = Decimal(text)
    except Exception:
        return 0
    if not decimal_value.is_finite():
        return 0
    if (
        decimal_value.is_zero()
        or decimal_value == decimal_value.to_integral_value()
        or decimal_value.as_tuple().exponent >= 0
    ):
        return MAX_COORDINATE_TEXT_LENGTH
    return _fractional_decimal_places(text)


def _center_precision_budget(
    x_center: str,
    y_center: str,
    log10_zoom: float,
) -> tuple[int, int]:
    """Return ``(available, required)`` decimal places for a deep centre.

    This is deliberately a diagnostic, not a claim that a short decimal is
    mathematically invalid.  A short decimal can be an intentional exact
    point; it simply cannot represent a separately discovered Kalles target
    whose omitted digits matter at the requested scale.
    """

    available = min(
        _coordinate_precision_digits(x_center),
        _coordinate_precision_digits(y_center),
    )
    required = max(0, int(math.ceil(max(0.0, float(log10_zoom))))) + CENTER_PRECISION_GUARD_DIGITS
    return available, required


def _center_precision_error(
    x_center: str,
    y_center: str,
    log10_zoom: float,
) -> Optional[str]:
    available, required = _center_precision_budget(x_center, y_center, log10_zoom)
    if available >= required:
        return None
    return (
        f"center coordinates provide only {available} fractional decimal places, "
        f"but max zoom 10^{float(log10_zoom):.3f} requires at least {required} "
        f"for a stable deep target (including a {CENTER_PRECISION_GUARD_DIGITS}-digit guard). "
        "This can produce a legitimate-looking black interior while following a "
        "different path from the intended Kalles target. Supply the full-precision "
        "--x-center and --y-center exported by Kalles, or lower --max-zoom. "
        "Use --allow-underspecified-center only for an explicitly exploratory render."
    )


def _deep_point_max_log10_zoom(point: DeepZoomPoint) -> float:
    """Return the depth supported by both the source and stored digits."""

    available = min(
        _coordinate_precision_digits(point.x),
        _coordinate_precision_digits(point.y),
    )
    return min(
        float(point.source_log10_zoom),
        float(max(0, available - CENTER_PRECISION_GUARD_DIGITS)),
    )


def _validate_center_text(value: str, label: str) -> str:
    text = str(value).strip()
    if not text or len(text) > MAX_COORDINATE_TEXT_LENGTH:
        raise ValueError(
            f"{label} coordinate must contain between 1 and "
            f"{MAX_COORDINATE_TEXT_LENGTH:,} characters"
        )
    try:
        parsed = Decimal(text)
    except Exception as error:
        raise ValueError(f"invalid {label} coordinate: {value}") from error
    if not parsed.is_finite():
        raise ValueError(f"{label} coordinate must be finite")
    return text


JULIA_TARGET_PRECISION_DIGITS = 384


def _julia_repelling_fixed_point(
    julia_constant: tuple[str, str],
) -> tuple[str, str]:
    """Return a high-precision fixed point on the selected Julia boundary.

    The fixed points solve ``z² - z + c = 0``.  Selecting the root with the
    larger modulus selects the repelling branch, which is on the Julia set;
    using a generated decimal target keeps a custom ``--julia-c`` usable at
    the same depth as the built-in presets.
    """

    if len(julia_constant) != 2:
        raise ValueError("Julia constant must contain real and imaginary coordinates")
    real_text = _validate_center_text(julia_constant[0], "Julia real")
    imag_text = _validate_center_text(julia_constant[1], "Julia imaginary")
    with localcontext() as context:
        context.prec = JULIA_TARGET_PRECISION_DIGITS
        c_real = Decimal(real_text)
        c_imag = Decimal(imag_text)
        discriminant_real = Decimal(1) - Decimal(4) * c_real
        discriminant_imag = -Decimal(4) * c_imag
        magnitude = (discriminant_real * discriminant_real
                     + discriminant_imag * discriminant_imag).sqrt()
        root_real = ((magnitude + discriminant_real) / Decimal(2)).sqrt()
        root_imag = ((magnitude - discriminant_real) / Decimal(2)).sqrt()
        if discriminant_imag < 0:
            root_imag = -root_imag
        root_a = ((Decimal(1) + root_real) / Decimal(2), root_imag / Decimal(2))
        root_b = ((Decimal(1) - root_real) / Decimal(2), -root_imag / Decimal(2))
        selected = max(
            (root_a, root_b),
            key=lambda root: root[0] * root[0] + root[1] * root[1],
        )
        real, imag = selected
        significant = JULIA_TARGET_PRECISION_DIGITS
        real_output = format(real, f".{significant}g")
        imag_output = "0.0" if imag == 0 else format(imag, f".{significant}g")
    return real_output, imag_output


def _finite_float_coordinate(value: str, label: str) -> float:
    """Convert a coordinate for a binary renderer without allowing infinity."""

    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} coordinate is outside the binary renderer range") from error
    if not math.isfinite(converted):
        raise ValueError(f"{label} coordinate is outside the binary renderer range")
    return converted


def _resolve_render_point(
    *,
    point_spec: Optional[str],
    random_point: bool,
    x_center: Optional[str],
    y_center: Optional[str],
    random_seed: Optional[int],
    max_log_zoom: float,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
) -> tuple[str, str, Optional[DeepZoomPoint]]:
    """Resolve a formula-specific preset, random point, pair, or x/y input."""

    formula = _formula_name(formula)
    catalogue = FORMULA_POINT_CATALOGUES[formula]
    points_by_slug = FORMULA_POINTS_BY_SLUG[formula]

    if random_point and point_spec is not None:
        raise ValueError("use either --random-point or --point, not both")
    if (random_point or point_spec is not None) and (
        x_center is not None or y_center is not None
    ):
        raise ValueError("--point/--random-point cannot be combined with --x-center/--y-center")
    if (x_center is None) != (y_center is None):
        raise ValueError("--x-center and --y-center must be supplied together")

    if point_spec is not None and not point_spec.strip():
        raise ValueError("--point cannot be empty")
    spec = "random" if random_point else (point_spec.strip() if point_spec is not None else None)
    if spec is None:
        if x_center is None:
            if formula == "julia":
                return (*_julia_repelling_fixed_point(julia_constant), None)
            default_x, default_y = FORMULA_DEFAULT_CENTERS[formula]
            return default_x, default_y, None
        return (
            _validate_center_text(x_center, "real"),
            _validate_center_text(y_center, "imaginary"),
            None,
        )

    if spec.casefold() == "random":
        if formula == "mandelbrot":
            candidates = [
                point
                for point in catalogue
                if _deep_point_max_log10_zoom(point) + 1.0e-9 >= float(max_log_zoom)
                and (
                    float(max_log_zoom) < 150.0
                    or point.screened_log10_zoom <= float(max_log_zoom) + 1.0e-9
                )
            ]
        else:
            # Never choose a shallow gallery framing point for a deep render.
            # Alternate formulas now expose only targets with an explicit
            # recommended depth, so random selection remains meaningful at
            # e150 instead of silently ending in an interior tile.
            candidates = [
                point
                for point in catalogue
                if _deep_point_max_log10_zoom(point) + 1.0e-9 >= float(max_log_zoom)
            ]
        if not candidates:
            deepest = max(
                (_deep_point_max_log10_zoom(point) for point in catalogue),
                default=0.0,
            )
            raise ValueError(
                f"no curated point safely supports 10^{float(max_log_zoom):.3f}; "
                f"the stored catalogue currently reaches about 10^{deepest:.0f}"
            )
        chooser = random.SystemRandom() if random_seed is None else random.Random(random_seed)
        selected = chooser.choice(candidates)
        if formula == "julia" and selected.julia_c is not None:
            target_constant = (
                selected.julia_c
                if julia_constant == DEFAULT_JULIA_C
                else julia_constant
            )
            return (*_julia_repelling_fixed_point(target_constant), selected)
        return selected.x, selected.y, selected

    preset = points_by_slug.get(spec.casefold())
    if preset is not None:
        supported = _deep_point_max_log10_zoom(preset)
        if supported + 1.0e-9 < float(max_log_zoom):
            raise ValueError(
                f"point '{preset.slug}' is stored safely through about 10^{supported:.0f}, "
                f"below requested 10^{float(max_log_zoom):.3f}"
            )
        if formula == "julia" and preset.julia_c is not None:
            target_constant = (
                preset.julia_c
                if julia_constant == DEFAULT_JULIA_C
                else julia_constant
            )
            return (*_julia_repelling_fixed_point(target_constant), preset)
        return preset.x, preset.y, preset

    if len(spec) <= 2 * MAX_COORDINATE_TEXT_LENGTH + 1:
        real_text, separator, imaginary_text = spec.partition(",")
    else:
        real_text, separator, imaginary_text = "", "", ""
    if (
        separator
        and "," not in imaginary_text
        and real_text.strip()
        and imaginary_text.strip()
    ):
        return (
            _validate_center_text(real_text, "real"),
            _validate_center_text(imaginary_text, "imaginary"),
            None,
        )
    raise ValueError(
        f"unknown {formula} point '{spec}'; use --list-points --formula {formula}, "
        "'random', or REAL,IMAG"
    )


def _print_deep_zoom_points(formula: str = "mandelbrot") -> None:
    formula = _formula_name(formula)
    points = FORMULA_POINT_CATALOGUES[formula]
    heading = "Curated deep-zoom points" if formula == "mandelbrot" else "Formula point presets"
    print(f"{heading} for {formula} ({len(points)}):")
    for point in points:
        supported = _deep_point_max_log10_zoom(point)
        derived = f", conjugate of {point.conjugate_of}" if point.conjugate_of else ""
        julia = f", c={','.join(point.julia_c)}" if point.julia_c else ""
        depth = f"stored-safe to ~1e{supported:.0f}"
        print(
            f"  {point.slug:<31} {depth:<28} {point.name}"
            f" [{point.source_name}{derived}{julia}]"
        )


def _decimal_precision(x_center: str, y_center: str, log10_zoom: float) -> int:
    precision = max(
        50,
        32 + int(math.ceil(max(0.0, log10_zoom))),
        _fractional_decimal_places(x_center),
        _fractional_decimal_places(y_center),
    )
    if precision > MAX_COORDINATE_TEXT_LENGTH:
        raise ValueError(
            f"coordinate precision exceeds the {MAX_COORDINATE_TEXT_LENGTH:,}-digit limit"
        )
    return precision


def _native_precision_bits(x_center: str, y_center: str, log10_zoom: float) -> int:
    """Return a precision accepted by the native MPFR ABI."""

    precision_bits = max(
        256,
        int(_decimal_precision(x_center, y_center, log10_zoom) * math.log2(10.0)) + 32,
    )
    if precision_bits > MAX_NATIVE_PRECISION_BITS:
        raise ValueError(
            f"native coordinate precision exceeds the {MAX_NATIVE_PRECISION_BITS:,}-bit limit"
        )
    return precision_bits


def _formula_name(value: str) -> str:
    """Validate and normalise a formula name used by the render pipeline."""

    name = str(value).strip().casefold()
    aliases = {
        "burningship": "burning-ship",
        "burning_ship": "burning-ship",
        "mandelbar": "tricorn",
    }
    name = aliases.get(name, name)
    if name not in FORMULA_CHOICES:
        raise ValueError(
            f"unknown formula '{value}'; choose one of: {', '.join(FORMULA_CHOICES)}"
        )
    return name


def _parse_coordinate_pair(value: str, label: str) -> tuple[str, str]:
    text = str(value).strip()
    if len(text) > 2 * MAX_COORDINATE_TEXT_LENGTH + 1:
        raise ValueError(
            f"{label} must contain no more than "
            f"{2 * MAX_COORDINATE_TEXT_LENGTH + 1:,} characters"
        )
    real_text, separator, imaginary_text = text.partition(",")
    if (
        not separator
        or "," in imaginary_text
        or not real_text.strip()
        or not imaginary_text.strip()
    ):
        raise ValueError(f"{label} must be REAL,IMAG, for example -0.8,0.156")
    return (
        _validate_center_text(real_text, f"{label} real"),
        _validate_center_text(imaginary_text, f"{label} imaginary"),
    )


def _formula_power(formula: str) -> int:
    _formula_name(formula)
    return 2


def _zoom_log(value: Any) -> float:
    """Return log10(value) without converting huge decimal zooms to float."""

    try:
        text = str(value).strip()
    except Exception as exc:
        raise ValueError("invalid zoom value") from exc
    if not text or len(text) > MAX_COORDINATE_TEXT_LENGTH:
        raise ValueError(
            f"zoom value must contain between 1 and {MAX_COORDINATE_TEXT_LENGTH:,} characters"
        )
    try:
        decimal_value = Decimal(text)
    except Exception as exc:
        raise ValueError("invalid zoom value") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise ValueError("zoom must be a finite positive decimal")
    exponent = decimal_value.adjusted()
    if exponent < math.floor(MIN_LOG10_ZOOM) - 1 or exponent > math.ceil(MAX_LOG10_ZOOM) + 1:
        raise ValueError(
            f"zoom exponent must be between {MIN_LOG10_ZOOM:.0f} and "
            f"{MAX_LOG10_ZOOM:.0f}"
        )
    mantissa = decimal_value.scaleb(-exponent)
    try:
        result = float(exponent) + math.log10(float(mantissa))
    except (OverflowError, ValueError) as exc:
        raise ValueError("invalid zoom value") from exc
    if result < MIN_LOG10_ZOOM or result > MAX_LOG10_ZOOM:
        raise ValueError(
            f"zoom exponent must be between {MIN_LOG10_ZOOM:.0f} and "
            f"{MAX_LOG10_ZOOM:.0f}"
        )
    return result


def _zoom_text(log10_zoom: float) -> bytes:
    """Encode 10**log10_zoom without overflowing Python's binary float."""

    log10_zoom = _validate_log10_zoom(log10_zoom)
    exponent = math.floor(log10_zoom)
    mantissa = 10.0 ** (log10_zoom - exponent)
    return f"{mantissa:.17g}e{exponent:+d}".encode("ascii")


def _zoom_label(log10_zoom: float) -> str:
    if log10_zoom > 300.0:
        return f"10^{log10_zoom:.3f}x"
    return f"{10.0 ** log10_zoom:.3e}x"


def _reference_orbit(
    x_center: str,
    y_center: str,
    max_iter: int,
    log10_zoom: float,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
    *,
    include_margins: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """Return a high-precision reference orbit for an alternate formula.

    ``real`` and ``imag`` are float projections used by the vectorised
    perturbation recurrence.  When requested, ``margins`` retains the
    high-precision value ``|z|² - 4`` for every reference sample.  Comparing
    the projected float orbit directly with ``4`` loses sub-ulp escape bands
    at deep zooms (notably the Tricorn and Burning Ship tips), so the margin
    lets the alternate renderer make the escape decision without rounding
    away the very quantity it is trying to detect.
    """

    np = _require_numpy()
    formula = _formula_name(formula)
    precision = _decimal_precision(x_center, y_center, log10_zoom)
    real = np.empty(max_iter + 1, dtype=np.float64)
    imag = np.empty(max_iter + 1, dtype=np.float64)
    margins = np.empty(max_iter + 1, dtype=np.float64) if include_margins else None

    def store_reference(index: int, zr: Any, zi: Any) -> None:
        real[index] = float(zr)
        imag[index] = float(zi)
        if margins is not None:
            try:
                margin = float(zr * zr + zi * zi - 4)
            except (OverflowError, ValueError):
                margin = math.inf
            margins[index] = margin

    def store_escaped_tail(index: int) -> None:
        real[index:] = math.inf
        imag[index:] = math.inf
        if margins is not None:
            margins[index:] = math.inf

    try:
        import mpmath as mp

        with mp.workdps(precision):
            cx = mp.mpf(x_center)
            cy = mp.mpf(y_center)
            julia_cx = mp.mpf(julia_constant[0])
            julia_cy = mp.mpf(julia_constant[1])
            if formula == "julia":
                zr, zi = cx, cy
                parameter_real, parameter_imag = julia_cx, julia_cy
            else:
                zr = mp.mpf("0")
                zi = mp.mpf("0")
                parameter_real, parameter_imag = cx, cy
            fixed_point = None
            if formula == "julia":
                # Julia presets are repelling fixed points.  Their decimal
                # exports intentionally carry far more digits than the
                # requested viewport, but even a 384-digit approximation
                # eventually leaves the fixed point when iterated for the
                # renderer's much larger iteration budget.  Recover the
                # algebraic root from c and hold this reference exactly at
                # the selected target; pixel perturbations still determine
                # which side of the Julia boundary each pixel occupies.
                discriminant = mp.sqrt(1 - 4 * mp.mpc(julia_cx, julia_cy))
                roots = (
                    (1 + discriminant) / 2,
                    (1 - discriminant) / 2,
                )
                candidate = mp.mpc(cx, cy)
                selected_root = min(roots, key=lambda root: abs(root - candidate))
                required_digits = max(
                    32,
                    int(math.ceil(max(0.0, log10_zoom)))
                    + CENTER_PRECISION_GUARD_DIGITS,
                )
                root_tolerance = mp.power(10, -required_digits)
                if (
                    abs(2 * selected_root) > 1
                    and abs(selected_root - candidate) <= root_tolerance
                ):
                    fixed_point = (mp.re(selected_root), mp.im(selected_root))
            float_limit = mp.mpf("1e300")
            if fixed_point is not None:
                fixed_real, fixed_imag = fixed_point
                real.fill(float(fixed_real))
                imag.fill(float(fixed_imag))
                if margins is not None:
                    margins.fill(float(fixed_real * fixed_real + fixed_imag * fixed_imag - 4))
            else:
                for index in range(max_iter + 1):
                    if abs(zr) > float_limit or abs(zi) > float_limit:
                        # The reference has escaped so far that its float64
                        # representation would overflow. Remaining pixels are
                        # handled as escaped by the perturbation loop.
                        store_escaped_tail(index)
                        break
                    store_reference(index, zr, zi)
                    if formula == "burning-ship":
                        next_real = abs(zr) * abs(zr) - abs(zi) * abs(zi) + parameter_real
                        next_imag = 2 * abs(zr) * abs(zi) + parameter_imag
                    elif formula == "tricorn":
                        next_real = zr * zr - zi * zi + parameter_real
                        next_imag = -2 * zr * zi + parameter_imag
                    else:
                        next_real = zr * zr - zi * zi + parameter_real
                        next_imag = 2 * zr * zi + parameter_imag
                    zr, zi = next_real, next_imag
    except ImportError:
        # Decimal is slower but keeps the centre precise without making mpmath
        # a hard dependency for ordinary-depth renders.
        with localcontext() as context:
            context.prec = precision
            cx = Decimal(x_center)
            cy = Decimal(y_center)
            julia_cx = Decimal(julia_constant[0])
            julia_cy = Decimal(julia_constant[1])
            if formula == "julia":
                zr, zi = cx, cy
                parameter_real, parameter_imag = julia_cx, julia_cy
            else:
                zr = Decimal(0)
                zi = Decimal(0)
                parameter_real, parameter_imag = cx, cy
            fixed_point = None
            if formula == "julia":
                discriminant_real = Decimal(1) - Decimal(4) * julia_cx
                discriminant_imag = -Decimal(4) * julia_cy
                magnitude = (
                    discriminant_real * discriminant_real
                    + discriminant_imag * discriminant_imag
                ).sqrt()
                root_real = ((magnitude + discriminant_real) / Decimal(2)).sqrt()
                root_imag = ((magnitude - discriminant_real) / Decimal(2)).sqrt()
                if discriminant_imag < 0:
                    root_imag = -root_imag
                roots = (
                    (
                        (Decimal(1) + root_real) / Decimal(2),
                        root_imag / Decimal(2),
                    ),
                    (
                        (Decimal(1) - root_real) / Decimal(2),
                        -root_imag / Decimal(2),
                    ),
                )
                selected_root = min(
                    roots,
                    key=lambda root: (root[0] - cx) ** 2 + (root[1] - cy) ** 2,
                )
                required_digits = max(
                    32,
                    int(math.ceil(max(0.0, log10_zoom)))
                    + CENTER_PRECISION_GUARD_DIGITS,
                )
                root_tolerance = Decimal(1).scaleb(-required_digits)
                distance_squared = (selected_root[0] - cx) ** 2 + (selected_root[1] - cy) ** 2
                multiplier_squared = 4 * (
                    selected_root[0] * selected_root[0]
                    + selected_root[1] * selected_root[1]
                )
                if multiplier_squared > 1 and distance_squared.sqrt() <= root_tolerance:
                    fixed_point = selected_root
            decimal_limit = Decimal("1e300")
            if fixed_point is not None:
                fixed_real, fixed_imag = fixed_point
                real.fill(float(fixed_real))
                imag.fill(float(fixed_imag))
                if margins is not None:
                    margins.fill(fixed_real * fixed_real + fixed_imag * fixed_imag - Decimal(4))
            else:
                for index in range(max_iter + 1):
                    if abs(zr) > decimal_limit or abs(zi) > decimal_limit:
                        store_escaped_tail(index)
                        break
                    store_reference(index, zr, zi)
                    if formula == "burning-ship":
                        next_real = abs(zr) * abs(zr) - abs(zi) * abs(zi) + parameter_real
                        next_imag = Decimal(2) * abs(zr) * abs(zi) + parameter_imag
                    elif formula == "tricorn":
                        next_real = zr * zr - zi * zi + parameter_real
                        next_imag = Decimal(-2) * zr * zi + parameter_imag
                    else:
                        next_real = zr * zr - zi * zi + parameter_real
                        next_imag = Decimal(2) * zr * zi + parameter_imag
                    zr, zi = next_real, next_imag

    if margins is None:
        return real, imag
    return real, imag, margins


def _stabilize_alternate_reference_cycle(
    reference_real: Any,
    reference_imag: Any,
    reference_margin: Any,
) -> Optional[int]:
    """Hold a detected bounded cycle instead of amplifying decimal roundoff.

    Alternate-formula catalogue targets are often repelling preperiodic
    points.  A finite decimal export is mathematically close to the target,
    but a repelling cycle amplifies its last few rounding digits until the
    centre falsely escapes after a few thousand iterations.  Three matching
    bounded cycles in the high-precision reference are enough evidence to
    repeat that cycle; neighbouring pixel perturbations still decide which
    side of the boundary they occupy.
    """

    np = _require_numpy()
    real = np.asarray(reference_real)
    imag = np.asarray(reference_imag)
    margins = np.asarray(reference_margin)
    if real.ndim != 1 or imag.shape != real.shape or margins.shape != real.shape:
        raise ValueError("alternate reference arrays must have matching vectors")
    finite = np.isfinite(real) & np.isfinite(imag)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size < 6:
        return None
    last_finite = int(finite_indices[-1])
    scan_limit = min(last_finite, 1024)
    tolerance = 1.0e-12
    max_period = min(64, (scan_limit + 1) // 4)
    for period in range(1, max_period + 1):
        # Compare three complete cycles, beginning at the earliest candidate
        # that leaves room for a preperiod and two independent repeats. This
        # avoids treating one coincident pair in a chaotic transient as a
        # proof that the reference is periodic.
        for end in range(4 * period - 1, scan_limit + 1):
            cycle_start = end - 3 * period + 1
            previous = slice(cycle_start, cycle_start + period)
            current = slice(cycle_start + period, cycle_start + 2 * period)
            following = slice(cycle_start + 2 * period, cycle_start + 3 * period)
            if not (
                finite[previous].all()
                and finite[current].all()
                and finite[following].all()
            ):
                continue
            cycle_real = real[previous]
            cycle_imag = imag[previous]
            current_real = real[current]
            current_imag = imag[current]
            following_real = real[following]
            following_imag = imag[following]
            radius_squared = cycle_real * cycle_real + cycle_imag * cycle_imag
            if not np.isfinite(radius_squared).all() or float(np.max(radius_squared)) >= 4.0:
                continue
            difference = np.hypot(
                current_real - cycle_real,
                current_imag - cycle_imag,
            )
            following_difference = np.hypot(
                following_real - current_real,
                following_imag - current_imag,
            )
            scale = np.maximum(
                1.0,
                np.maximum(
                    np.hypot(cycle_real, cycle_imag),
                    np.maximum(
                        np.hypot(current_real, current_imag),
                        np.hypot(following_real, following_imag),
                    ),
                ),
            )
            if not (
                np.all(difference <= tolerance * scale)
                and np.all(following_difference <= tolerance * scale)
            ):
                continue
            repeated_real = np.resize(cycle_real, real.size - cycle_start)
            repeated_imag = np.resize(cycle_imag, imag.size - cycle_start)
            repeated_margin = np.resize(margins[previous], margins.size - cycle_start)
            real[cycle_start:] = repeated_real
            imag[cycle_start:] = repeated_imag
            margins[cycle_start:] = repeated_margin
            return period
    return None


ALTERNATE_CYCLE_MAX_PERIOD = 4096
ALTERNATE_CYCLE_TOLERANCE = 1.0e-13
ALTERNATE_CYCLE_CONFIRMATIONS = 3


def _detect_float_cycle_period(values: Any) -> Optional[int]:
    """Find a short, numerically settled period in a bounded orbit."""

    np = _require_numpy()
    values = np.asarray(values, dtype=np.complex128).ravel()
    if values.size < 6 or not np.isfinite(values[-1]):
        return None
    maximum = min(ALTERNATE_CYCLE_MAX_PERIOD, (values.size - 1) // 3)
    for period in range(1, maximum + 1):
        window = min(values.size - period, max(3 * period, 256))
        if window < 3 * period:
            continue
        current = values[-window:]
        previous = values[-window - period:-period]
        difference = np.abs(current - previous)
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        if np.all(difference <= ALTERNATE_CYCLE_TOLERANCE * scale):
            return period
    return None


def _alternate_critical_orbit(
    formula: str,
    x_center: str,
    y_center: str,
    julia_constant: tuple[str, str],
) -> Any:
    """Return a small float critical orbit used for interior acceleration."""

    np = _require_numpy()
    if formula == "julia":
        parameter_real = _finite_float_coordinate(julia_constant[0], "Julia real")
        parameter_imag = _finite_float_coordinate(julia_constant[1], "Julia imaginary")
    else:
        parameter_real = _finite_float_coordinate(x_center, "real")
        parameter_imag = _finite_float_coordinate(y_center, "imaginary")
    zr = 0.0
    zi = 0.0
    values: list[complex] = []
    for _ in range(ALTERNATE_CYCLE_MAX_PERIOD * 3):
        zr, zi = _alternate_float_step(
            formula, zr, zi, parameter_real, parameter_imag
        )
        magnitude_squared = zr * zr + zi * zi
        if not math.isfinite(magnitude_squared) or magnitude_squared > 4.0:
            break
        values.append(complex(zr, zi))
    return np.asarray(values, dtype=np.complex128)


def _alternate_float_step(
    formula: str,
    real: float,
    imag: float,
    parameter_real: float,
    parameter_imag: float,
) -> tuple[float, float]:
    """Apply one alternate recurrence to a pair of ordinary floats."""

    if formula == "burning-ship":
        absolute_real = abs(real)
        absolute_imag = abs(imag)
        return (
            absolute_real * absolute_real
            - absolute_imag * absolute_imag
            + parameter_real,
            2.0 * absolute_real * absolute_imag + parameter_imag,
        )
    if formula == "tricorn":
        return (
            real * real - imag * imag + parameter_real,
            -2.0 * real * imag + parameter_imag,
        )
    return (
        real * real - imag * imag + parameter_real,
        2.0 * real * imag + parameter_imag,
    )


def _cycle_is_attracting(
    formula: str,
    values: Any,
    period: int,
    parameter_real: float,
    parameter_imag: float,
) -> bool:
    """Require a measured contraction before enabling cycle skipping."""

    if period <= 0 or values.size < period:
        return False
    base = complex(values[-period])
    perturbation = 1.0e-8 + 1.0e-8j
    perturbed = base + perturbation
    for _ in range(period):
        base_real, base_imag = _alternate_float_step(
            formula, base.real, base.imag, parameter_real, parameter_imag
        )
        perturbed_real, perturbed_imag = _alternate_float_step(
            formula,
            perturbed.real,
            perturbed.imag,
            parameter_real,
            parameter_imag,
        )
        base = complex(base_real, base_imag)
        perturbed = complex(perturbed_real, perturbed_imag)
    if not (math.isfinite(base.real) and math.isfinite(base.imag)):
        return False
    contraction = abs(perturbed - base) / abs(perturbation)
    return math.isfinite(contraction) and contraction < 0.95


def _alternate_cycle_period(
    formula: str,
    reference: Any,
    x_center: str,
    y_center: str,
    julia_constant: tuple[str, str],
) -> Optional[int]:
    """Choose a conservative period for skipping converged interior pixels."""

    if formula == "julia":
        # The Julia reference is deliberately stabilized at a repelling fixed
        # point, so its period is not the attracting cycle that bounds the
        # filled interior. Probe the critical orbit instead.
        values = _alternate_critical_orbit(
            formula, x_center, y_center, julia_constant
        )
        parameter_real = _finite_float_coordinate(julia_constant[0], "Julia real")
        parameter_imag = _finite_float_coordinate(julia_constant[1], "Julia imaginary")
    else:
        values = reference
        parameter_real = _finite_float_coordinate(x_center, "real")
        parameter_imag = _finite_float_coordinate(y_center, "imaginary")
    period = _detect_float_cycle_period(values)
    if period is None or not _cycle_is_attracting(
        formula, values, period, parameter_real, parameter_imag
    ):
        return None
    return period


def _view_offsets(width: int, height: int, log10_zoom: float) -> tuple[Any, Any]:
    np = _require_numpy()
    width, height = _validate_dimensions(width, height, "fractal")
    log10_zoom = _validate_log10_zoom(log10_zoom)
    if log10_zoom > 300.0:
        raise RuntimeError("Python rendering cannot represent zooms beyond 10^300; use the native renderer")
    zoom = 10.0 ** log10_zoom
    view_height = 2.8 / zoom
    view_width = view_height * width / height
    if not math.isfinite(view_height) or not math.isfinite(view_width):
        raise ValueError("zoom and frame dimensions produce an unrepresentable viewport")
    x = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * view_width / width
    y = ((height - 1) / 2.0 - np.arange(height, dtype=np.float64)) * view_height / height
    return np.meshgrid(x, y)


def _smooth_escape(iteration: int, magnitude_squared: Any, power: int = 2) -> Any:
    np = _require_numpy()
    # Orbit arithmetic can overflow after a pixel has escaped, especially in
    # the exploratory alternate-formula path. Keep the smooth colour finite
    # instead of turning an otherwise valid frame into -inf/NaN values.
    safe_squared = np.nan_to_num(
        np.asarray(magnitude_squared),
        nan=np.finfo(np.float64).max,
        posinf=np.finfo(np.float64).max,
        neginf=4.0000001,
    )
    magnitude = np.sqrt(np.maximum(safe_squared, 4.0000001))
    return iteration - np.log(np.log(magnitude)) / math.log(float(power))


def _render_direct(
    width: int,
    height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
) -> Any:
    """Vectorised float64 renderer for shallow Mandelbrot-family views."""

    np = _require_numpy()
    width, height = _validate_dimensions(width, height, "fractal")
    max_iter = _validate_iteration_count(max_iter)
    x_center = _validate_center_text(x_center, "real")
    y_center = _validate_center_text(y_center, "imaginary")
    if len(julia_constant) != 2:
        raise ValueError("Julia constant must contain real and imaginary coordinates")
    julia_constant = (
        _validate_center_text(julia_constant[0], "Julia real"),
        _validate_center_text(julia_constant[1], "Julia imaginary"),
    )
    formula = _formula_name(formula)
    x_offset, y_offset = _view_offsets(width, height, log10_zoom)
    # Float conversion is safe here because this path is used before a deep
    # zoom, where the pixel spacing is still much larger than float64 ulps.
    real = (_finite_float_coordinate(x_center, "real") + x_offset).ravel()
    imag = (_finite_float_coordinate(y_center, "imaginary") + y_offset).ravel()
    if formula == "julia":
        z_real = real.copy()
        z_imag = imag.copy()
        parameter_real = np.full_like(
            real, _finite_float_coordinate(julia_constant[0], "Julia real")
        )
        parameter_imag = np.full_like(
            imag, _finite_float_coordinate(julia_constant[1], "Julia imaginary")
        )
    else:
        z_real = np.zeros_like(real)
        z_imag = np.zeros_like(imag)
        parameter_real = real
        parameter_imag = imag
    escaped = np.zeros(real.shape, dtype=bool)
    smooth = np.full(real.shape, float(max_iter), dtype=np.float32)

    # These tests are valid only in the Mandelbrot parameter plane.
    if formula == "mandelbrot":
        q = (real - 0.25) ** 2 + imag**2
        cardioid = q * (q + real - 0.25) <= 0.25 * imag**2
        bulb = (real + 1.0) ** 2 + imag**2 <= 0.0625
        escaped[cardioid | bulb] = True

    for iteration in range(1, max_iter + 1):
        active_indices = np.flatnonzero(~escaped)
        if active_indices.size == 0:
            break
        active_real = z_real[active_indices]
        active_imag = z_imag[active_indices]
        active_parameter_real = parameter_real[active_indices]
        active_parameter_imag = parameter_imag[active_indices]
        if formula == "burning-ship":
            absolute_real = np.abs(active_real)
            absolute_imag = np.abs(active_imag)
            next_real = (
                absolute_real * absolute_real
                - absolute_imag * absolute_imag
                + active_parameter_real
            )
            next_imag = 2.0 * absolute_real * absolute_imag + active_parameter_imag
        elif formula == "tricorn":
            next_real = (
                active_real * active_real
                - active_imag * active_imag
                + active_parameter_real
            )
            next_imag = -2.0 * active_real * active_imag + active_parameter_imag
        else:
            next_real = (
                active_real * active_real
                - active_imag * active_imag
                + active_parameter_real
            )
            next_imag = 2.0 * active_real * active_imag + active_parameter_imag
        magnitude_squared = next_real * next_real + next_imag * next_imag
        newly_escaped = (magnitude_squared > 4.0) | ~np.isfinite(magnitude_squared)
        escaped_indices = active_indices[newly_escaped]
        smooth[escaped_indices] = _smooth_escape(
            iteration, magnitude_squared[newly_escaped], _formula_power(formula)
        )
        z_real[active_indices] = next_real
        z_imag[active_indices] = next_imag
        escaped[escaped_indices] = True

    return smooth.reshape((height, width))


def _render_perturbed(
    width: int,
    height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
) -> Any:
    """Render using a high-precision centre orbit and pixel perturbations."""

    np = _require_numpy()
    formula = _formula_name(formula)
    if formula != "mandelbrot":
        return _render_perturbed_alternate(
            width,
            height,
            log10_zoom,
            x_center,
            y_center,
            max_iter,
            formula,
            julia_constant,
        )
    x_offset, y_offset = _view_offsets(width, height, log10_zoom)
    reference_real, reference_imag = _reference_orbit(
        x_center, y_center, max_iter, log10_zoom
    )

    x_offset = x_offset.ravel()
    y_offset = y_offset.ravel()
    delta_real = np.zeros(width * height, dtype=np.float64)
    delta_imag = np.zeros(width * height, dtype=np.float64)
    escaped = np.zeros(width * height, dtype=bool)
    smooth = np.full(width * height, float(max_iter), dtype=np.float32)

    for iteration in range(max_iter):
        active_indices = np.flatnonzero(~escaped)
        if active_indices.size == 0:
            break
        ref_real = reference_real[iteration]
        ref_imag = reference_imag[iteration]
        if not np.isfinite(ref_real) or not np.isfinite(ref_imag):
            # A reference point outside the set eventually overflows even in
            # arbitrary precision. Do not turn that into NaNs in the vector
            # path; all still-active pixels have escaped by this point.
            smooth[active_indices] = iteration + 1
            escaped[active_indices] = True
            break
        active_delta_real = delta_real[active_indices]
        active_delta_imag = delta_imag[active_indices]
        next_delta_real = (
            2.0 * (ref_real * active_delta_real - ref_imag * active_delta_imag)
            + active_delta_real * active_delta_real
            - active_delta_imag * active_delta_imag
            + x_offset[active_indices]
        )
        next_delta_imag = (
            2.0 * (ref_real * active_delta_imag + ref_imag * active_delta_real)
            + 2.0 * active_delta_real * active_delta_imag
            + y_offset[active_indices]
        )
        next_real = reference_real[iteration + 1] + next_delta_real
        next_imag = reference_imag[iteration + 1] + next_delta_imag
        magnitude_squared = next_real * next_real + next_imag * next_imag
        newly_escaped = (magnitude_squared > 4.0) | ~np.isfinite(magnitude_squared)
        escaped_indices = active_indices[newly_escaped]
        smooth[escaped_indices] = _smooth_escape(
            iteration + 1, magnitude_squared[newly_escaped]
        )
        escaped[escaped_indices] = True

        # Only active indices are written back. Escaped pixels stay out of
        # every later vector operation, and their final smooth value is stored.
        delta_real[active_indices] = next_delta_real
        delta_imag[active_indices] = next_delta_imag

    return smooth.reshape((height, width))


def _render_perturbed_alternate(
    width: int,
    height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    formula: str,
    julia_constant: tuple[str, str],
) -> Any:
    """High-precision-reference perturbation fallback for alternate formulas.

    The validated native BLA hierarchy is specific to the Mandelbrot
    parameter plane. Alternate formulas still benefit from a high-precision
    reference orbit here, which makes Julia/Burning Ship/Tricorn views useful
    well past ordinary float-centre zooms without pretending they use the
    Mandelbrot BLA proof path.
    """

    np = _require_numpy()
    formula = _formula_name(formula)
    width, height = _validate_dimensions(width, height, "fractal")
    max_iter = _validate_iteration_count(max_iter)
    log10_zoom = _validate_log10_zoom(log10_zoom)
    x_center = _validate_center_text(x_center, "real")
    y_center = _validate_center_text(y_center, "imaginary")
    if len(julia_constant) != 2:
        raise ValueError("Julia constant must contain real and imaginary coordinates")
    julia_constant = (
        _validate_center_text(julia_constant[0], "Julia real"),
        _validate_center_text(julia_constant[1], "Julia imaginary"),
    )
    x_offset, y_offset = _view_offsets(width, height, log10_zoom)
    reference_real, reference_imag, reference_margin = _reference_orbit(
        x_center,
        y_center,
        max_iter,
        log10_zoom,
        formula,
        julia_constant,
        include_margins=True,
    )
    _stabilize_alternate_reference_cycle(
        reference_real,
        reference_imag,
        reference_margin,
    )
    offset = x_offset.ravel() + 1j * y_offset.ravel()
    reference = np.full(reference_real.shape, np.inf + 0j, dtype=np.complex128)
    finite_reference = np.isfinite(reference_real) & np.isfinite(reference_imag)
    reference[finite_reference] = (
        reference_real[finite_reference] + 1j * reference_imag[finite_reference]
    )
    # The alternate recurrence is evaluated around the selected centre.  In
    # the axis-crossing fallback below we must reconstruct the complete map
    # ``f(z) = |z|^2 + c`` rather than adding only the pixel's tiny offset.
    # Keeping this as the projected float centre also makes the recomputed
    # value use the same precision as ``reference`` and ``next_ref``.
    parameter = complex(
        _finite_float_coordinate(x_center, "real"),
        _finite_float_coordinate(y_center, "imaginary"),
    )
    cycle_period = _alternate_cycle_period(
        formula,
        reference,
        x_center,
        y_center,
        julia_constant,
    )
    cycle_start_iteration = max(
        256,
        int(math.ceil(max(0.0, log10_zoom) / math.log10(4.0))) + 32,
    )
    cycle_checkpoint = (
        np.empty(width * height, dtype=np.complex128)
        if cycle_period is not None
        else None
    )
    cycle_checkpoint_valid = (
        np.zeros(width * height, dtype=bool)
        if cycle_period is not None
        else None
    )
    cycle_hits = (
        np.zeros(width * height, dtype=np.uint8)
        if cycle_period is not None
        else None
    )
    delta = offset.copy() if formula == "julia" else np.zeros(width * height, dtype=np.complex128)
    escaped = np.zeros(width * height, dtype=bool)
    smooth = np.full(width * height, float(max_iter), dtype=np.float32)
    # A stabilized Julia reference is an analytic fixed point.  The exact
    # centre pixel is therefore known bounded; removing it up front also
    # prevents the tiny initial perturbations of neighbouring pixels from
    # looking like a false period before they have separated from the
    # repelling reference.
    reference_is_fixed = (
        formula == "julia"
        and np.isfinite(reference[0])
        and np.isfinite(reference[1])
        and reference[0] == reference[1]
    )
    if reference_is_fixed:
        escaped[offset == 0.0] = True
    power = _formula_power(formula)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for iteration in range(max_iter):
            active_indices = np.flatnonzero(~escaped)
            if active_indices.size == 0:
                break
            ref = reference[iteration]
            next_ref = reference[iteration + 1]
            if not np.isfinite(ref) or not np.isfinite(next_ref):
                smooth[active_indices] = iteration + 1
                escaped[active_indices] = True
                break
            active_delta = delta[active_indices]
            # ``ref`` is the float projection of a higher-precision reference
            # orbit.  The perturbation is defined relative to that orbit, so
            # its centre must remain exactly delta=0.  A residual calculated
            # from the rounded float reference is numerical noise (often
            # about 1e-16); adding it would swamp a 1e-150 pixel offset and
            # turn a valid deep view into a flat field.
            if formula == "julia":
                next_delta = (
                    2.0 * ref * active_delta
                    + active_delta * active_delta
                )
            elif formula == "tricorn":
                conjugate_delta = np.conjugate(active_delta)
                next_delta = (
                    2.0 * np.conjugate(ref) * conjugate_delta
                    + conjugate_delta * conjugate_delta
                    + offset[active_indices]
                )
            elif formula == "burning-ship":
                reference_abs = abs(float(ref.real)) + 1j * abs(float(ref.imag))
                sign_real = 1.0 if ref.real >= 0.0 else -1.0
                sign_imag = 1.0 if ref.imag >= 0.0 else -1.0
                signed_delta = (
                    sign_real * active_delta.real
                    + 1j * sign_imag * active_delta.imag
                )
                next_delta = (
                    2.0 * reference_abs * signed_delta
                    + signed_delta * signed_delta
                    + offset[active_indices]
                )
                # The absolute-value map is piecewise analytic. Recompute
                # pixels crossing an axis exactly for this step instead of
                # carrying a stale sign into subsequent perturbations.
                actual = ref + active_delta
                crossing = (
                    ((ref.real >= 0.0) & (actual.real < 0.0))
                    | ((ref.real < 0.0) & (actual.real >= 0.0))
                    | ((ref.imag >= 0.0) & (actual.imag < 0.0))
                    | ((ref.imag < 0.0) & (actual.imag >= 0.0))
                )
                if np.any(crossing):
                    absolute_actual = np.abs(actual.real[crossing]) + 1j * np.abs(actual.imag[crossing])
                    # Keep the tiny parameter offset out of the O(1) sum
                    # until after the projected centre has been subtracted;
                    # forming ``parameter + offset`` first rounds an e150
                    # pixel back to the centre and loses the crossing path.
                    next_delta[crossing] = (
                        absolute_actual * absolute_actual
                        + (parameter - next_ref)
                        + offset[active_indices][crossing]
                    )
            else:
                raise RuntimeError(f"unsupported alternate formula: {formula}")
            next_value = next_ref + next_delta
            # Do the escape test in terms of a high-precision reference
            # margin.  ``next_value.real**2 + next_value.imag**2`` rounds
            # values such as ``4 + 1e-300`` back to exactly 4.0, making the
            # e150+ boundary bands disappear.  Expanding around the exact
            # reference preserves that tiny margin while retaining a fully
            # vectorised float path for all pixels.
            reference_real_next = float(next_ref.real)
            reference_imag_next = float(next_ref.imag)
            delta_real_next = next_delta.real
            delta_imag_next = next_delta.imag
            margin = (
                reference_margin[iteration + 1]
                + 2.0 * (
                    reference_real_next * delta_real_next
                    + reference_imag_next * delta_imag_next
                )
                + delta_real_next * delta_real_next
                + delta_imag_next * delta_imag_next
            )
            newly_escaped = (margin > 0.0) | ~np.isfinite(margin)
            magnitude_squared = np.maximum(4.0000001, 4.0 + margin)
            escaped_indices = active_indices[newly_escaped]
            smooth[escaped_indices] = _smooth_escape(
                iteration + 1,
                magnitude_squared[newly_escaped],
                power,
            )
            escaped[escaped_indices] = True
            delta[active_indices] = next_delta
            if (
                cycle_period is not None
                and iteration + 1 >= cycle_start_iteration
                and (iteration + 1) % cycle_period == 0
            ):
                surviving = ~newly_escaped
                surviving_indices = active_indices[surviving]
                surviving_values = next_value[surviving]
                assert cycle_checkpoint is not None
                assert cycle_checkpoint_valid is not None
                assert cycle_hits is not None
                if surviving_indices.size:
                    uninitialized = ~cycle_checkpoint_valid[surviving_indices]
                    if np.any(uninitialized):
                        first_checkpoint = surviving_indices[uninitialized]
                        cycle_checkpoint[first_checkpoint] = surviving_values[uninitialized]
                        cycle_checkpoint_valid[first_checkpoint] = True
                        cycle_hits[first_checkpoint] = 0

                    checked_indices = surviving_indices[~uninitialized]
                    if checked_indices.size:
                        checked_values = surviving_values[~uninitialized]
                        previous_values = cycle_checkpoint[checked_indices]
                        difference = np.abs(checked_values - previous_values)
                        scale = np.maximum(
                            1.0,
                            np.maximum(
                                np.abs(checked_values), np.abs(previous_values)
                            ),
                        )
                        close = difference <= ALTERNATE_CYCLE_TOLERANCE * scale
                        if reference_is_fixed:
                            # A repelling fixed reference makes all pixels
                            # look period-matched while their perturbation is
                            # still microscopic. Only consider an attracting
                            # Julia cycle after the orbit has visibly left
                            # that reference neighbourhood.
                            close &= np.abs(checked_values - next_ref) > 1.0e-8
                        cycle_hits[checked_indices[~close]] = 0
                        close_indices = checked_indices[close]
                        cycle_hits[close_indices] = np.minimum(
                            255, cycle_hits[close_indices].astype(np.uint16) + 1
                        ).astype(np.uint8)
                        stable = close & (
                            cycle_hits[checked_indices] >= ALTERNATE_CYCLE_CONFIRMATIONS
                        )
                        stable_indices = checked_indices[stable]
                        if stable_indices.size:
                            smooth[stable_indices] = float(max_iter)
                            escaped[stable_indices] = True
                        update_mask = ~stable
                        if np.any(update_mask):
                            cycle_checkpoint[checked_indices[update_mask]] = checked_values[
                                update_mask
                            ]

    return smooth.reshape((height, width))


def _render_native(
    width: int,
    height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    native_threads: int,
    render_options: Optional[NativeRenderOptions] = None,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
) -> Any:
    np = _require_numpy()
    formula = _formula_name(formula)
    library = _get_native_library()
    if library is None:
        raise RuntimeError("native renderer is unavailable; run `make` inside `nix-shell`")

    output = np.empty((height, width), dtype=np.float32)
    zoom_text = _zoom_text(log10_zoom)
    precision_bits = _native_precision_bits(x_center, y_center, log10_zoom)
    use_perturbation = int(log10_zoom >= 12.0)
    options = render_options or NativeRenderOptions()
    formula_renderer = getattr(library, "render_fractal_ex", None)
    ex_renderer = getattr(library, "render_mandelbrot_ex", None)
    if formula_renderer is not None:
        status = formula_renderer(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            width,
            height,
            zoom_text,
            x_center.encode("ascii"),
            y_center.encode("ascii"),
            max_iter,
            precision_bits,
            int(log10_zoom >= 12.0),
            native_threads,
            FORMULA_IDS[formula],
            _finite_float_coordinate(julia_constant[0], "Julia real"),
            _finite_float_coordinate(julia_constant[1], "Julia imaginary"),
            ctypes.byref(options),
        )
    elif ex_renderer is not None and formula == "mandelbrot":
        status = ex_renderer(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            width,
            height,
            zoom_text,
            x_center.encode("ascii"),
            y_center.encode("ascii"),
            max_iter,
            precision_bits,
            use_perturbation,
            native_threads,
            ctypes.byref(options),
        )
    elif formula == "mandelbrot":
        status = library.render_mandelbrot(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            width,
            height,
            zoom_text,
            x_center.encode("ascii"),
            y_center.encode("ascii"),
            max_iter,
            precision_bits,
            use_perturbation,
            native_threads,
        )
    else:
        raise RuntimeError(
            "the native library does not expose render_fractal_ex; rebuild with make"
        )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native renderer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def _create_native_reference(
    x_center: str,
    y_center: str,
    max_iter: int,
    log10_zoom: float,
    series_order: int,
    bla_log10_zoom: Optional[float] = None,
    image_series_order: int = 32,
    reusable: bool = False,
) -> tuple[Any, Any]:
    """Prepare one reusable native reference orbit and BLA table.

    ``series_order`` is the degree of the per-pixel BLA map (1--3 in the
    public CLI).  The image-wide series is a separate polynomial and benefits
    from the full validated 8--32 term range, so references default to 32
    image-series terms without changing the BLA degree selected by callers.
    """

    x_center = _validate_center_text(x_center, "real")
    y_center = _validate_center_text(y_center, "imaginary")
    max_iter = _validate_iteration_count(max_iter)
    log10_zoom = _validate_log10_zoom(log10_zoom)
    if bla_log10_zoom is not None:
        bla_log10_zoom = _validate_log10_zoom(bla_log10_zoom, "BLA zoom exponent")
    series_order = _index_value(series_order, "series order")
    image_series_order = _index_value(image_series_order, "image series order")
    if not 1 <= series_order <= 32:
        raise ValueError("series order must be between 1 and 32")
    if not 8 <= image_series_order <= 32:
        raise ValueError("image series order must be between 8 and 32")
    library = _get_native_library()
    if library is None:
        raise RuntimeError("native renderer is unavailable; run `make` inside `nix-shell`")
    precision_bits = _native_precision_bits(x_center, y_center, log10_zoom)
    creator = (
        getattr(library, "fractal_create_reference_reusable", None)
        if reusable
        else None
    ) or library.fractal_create_reference
    handle = creator(
        x_center.encode("ascii"),
        y_center.encode("ascii"),
        _zoom_text(log10_zoom if bla_log10_zoom is None else bla_log10_zoom),
        max_iter,
        precision_bits,
        image_series_order,
    )
    if not handle:
        message = library.fractal_last_error() or b"unknown native reference error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return library, handle


def _clone_native_reference(
    library: Any,
    root_reference: Any,
    bla_log10_zoom: float,
) -> Any:
    """Build a radius-specific BLA tier while reusing one MPFR orbit."""

    clone = getattr(library, "fractal_clone_reference", None)
    if clone is None:
        raise RuntimeError("native reference tier cloning is unavailable")
    handle = clone(root_reference, _zoom_text(bla_log10_zoom))
    if not handle:
        message = library.fractal_last_error() or b"unknown native reference clone error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return handle


def _destroy_native_references(
    library: Any,
    references: list[tuple[float, Any]],
) -> None:
    """Release every native tier even when setup or rendering raises."""

    if library is None:
        references.clear()
        return
    for _, reference in references:
        try:
            library.fractal_destroy_reference(reference)
        except Exception:
            # Cleanup must not mask the original render/setup failure. The
            # native destroy entry point is itself idempotent for stale handles.
            pass
    references.clear()


def _native_reference_tier_logs(
    max_log_zoom: float,
    atlas_step: Optional[float] = None,
) -> list[float]:
    """Return BLA input-radius tiers needed by one render.

    A BLA hierarchy is not globally valid just because its reference orbit is
    long enough. Its input radius is part of the numerical contract. The old
    renderer built one hierarchy at e12 and reused it at e80/e100; that was
    correct only after expensive fallback work and could manufacture a black
    interior. Use decade-bucketed tiers instead: each tile selects the
    nearest prepared radius below it, keeping maps long without allocating a
    reference for every atlas level. When the atlas step is known, move each
    decade tier to the first atlas boundary at or below that decade. This
    matters because a tile is a complete factor-sized interval: a tier
    beginning at exactly 10^80 is too narrow for the tile whose lower edge is
    just below 10^80.

    ``atlas_step`` is optional to keep this helper useful to callers that do
    not have an atlas geometry. The render path supplies the actual
    log10(keyframe-factor) step.
    """

    maximum = float(max_log_zoom)
    if not math.isfinite(maximum) or maximum < 12.0:
        return []
    starts = [12.0]
    if maximum <= 32.0:
        return starts
    # The first reusable e12 table is intentionally conservative.  Leaving a
    # 28-decade gap before the next tier makes the e20--e40 atlas levels reuse
    # a shallow-radius BLA table and fall into a very expensive double-tail
    # path.  Start the regular deep ladder at e20; the extra two clones are
    # cheap compared with rendering even one 2K tile and remove that cliff.
    decade_starts = [
        float(value) for value in range(20, int(maximum) + 1, 10)
    ]
    if atlas_step is not None:
        step = float(atlas_step)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("atlas_step must be a finite positive value")
        # Atlas origins are integer multiples of the step, so a zero-origin
        # floor gives the same boundaries for every render path. Start three
        # complete atlas intervals before each decade boundary. A tier
        # placed at or immediately before a tile's lower edge has a
        # mathematically valid radius but no numerical margin at that edge;
        # the first tile using it can expose a narrow perturbation glitch. The
        # three-step lead leaves two whole tiles for the previous tier and keeps
        # the new radius comfortably wider than the complete viewport. A tier
        # exactly on a tile boundary can still make the BLA lookup land on a
        # floating-point radius edge at large source sizes, so the extra
        # interval is a deliberate numerical safety margin rather than wasted
        # setup. Never move a tier below e12: the
        # native deep path has no validated BLA contract for a broad shallow-
        # radius map, and an extreme keyframe factor can otherwise quantise
        # every decade boundary to e0.
        decade_starts = [
            max(12.0, math.floor(value / step) * step - 3.0 * step)
            for value in decade_starts
        ]
    starts.extend(decade_starts)
    starts = sorted(set(starts))
    # Give the final atlas level a radius-specific tier. It is cheap compared
    # with the movie and avoids making the last, deepest tile use a needlessly
    # conservative e.g. e80 table when the target is e100.
    if maximum > starts[-1] + 1.0e-9:
        starts.append(maximum)
    return sorted(set(starts))


def _select_native_reference(
    references: list[tuple[float, Any]],
    log_zoom: float,
    minimum_margin_log: float = 0.0,
) -> Any:
    """Select the deepest reference with a margin before a deep tile.

    The original e12 reference is also the compatibility entry point for a
    direct e12 render, so it remains eligible at its exact boundary. Deeper
    radius-specific tiers wait until the next atlas interval: using a newly
    built tier on the exact tile boundary can expose a one-pixel perturbation
    glitch even though the nominal BLA radius comparison passes.
    """

    margin = max(0.0, float(minimum_margin_log))
    selected = None
    for start_log, reference in references:
        if start_log <= 12.0 + 1.0e-9:
            if float(log_zoom) + 1.0e-9 < start_log:
                break
        elif margin > 0.0:
            if float(log_zoom) <= start_log + margin + 1.0e-9:
                break
        elif float(log_zoom) + 1.0e-9 < start_log:
            break
        selected = reference
    return selected


def _native_get_reference_stats(library: Any, handle: Any) -> Optional[dict[str, int]]:
    getter = getattr(library, "fractal_get_reference_stats", None)
    if getter is None:
        return None
    values = (ctypes.c_uint64 * 5)()
    if int(getter(handle, values, 5)) != 5:
        return None
    return {
        "reference_ns": int(values[0]),
        "series_ns": int(values[1]),
        "bla_ns": int(values[2]),
        "series_iteration": int(values[3]),
        "series_order": int(values[4]),
    }


def _render_native_reference(
    width: int,
    height: int,
    log10_zoom: float,
    max_iter: int,
    native_threads: int,
    native_reference: Any,
    series_order: int,
    series_block: int,
    render_options: Optional[NativeRenderOptions] = None,
) -> Any:
    np = _require_numpy()
    library = _get_native_library()
    if library is None:
        raise RuntimeError("native renderer is unavailable")

    output = np.empty((height, width), dtype=np.float32)
    zoom_text = _zoom_text(log10_zoom)
    options = render_options or NativeRenderOptions()
    ex_renderer = getattr(library, "fractal_render_mandelbrot_reference_ex", None)
    if ex_renderer is not None:
        status = ex_renderer(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            width,
            height,
            zoom_text,
            native_reference,
            max_iter,
            native_threads,
            series_order,
            series_block,
            ctypes.byref(options),
        )
    else:
        status = library.render_mandelbrot_reference(
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            width,
            height,
            zoom_text,
            native_reference,
            max_iter,
            native_threads,
            series_order,
            series_block,
        )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native renderer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def _decimal_to_scaled(value: Decimal) -> tuple[float, int]:
    """Convert a high-precision Decimal to the native mantissa/exponent form."""

    if not value.is_finite():
        raise ValueError("scaled Decimal value must be finite")
    if value.is_zero():
        return 0.0, 0
    sign = -1.0 if value.is_signed() else 1.0
    magnitude = abs(value)
    decimal_exponent = magnitude.adjusted()
    decimal_mantissa = float(magnitude.scaleb(-decimal_exponent))
    if not math.isfinite(decimal_mantissa) or decimal_mantissa <= 0.0:
        raise ValueError("scaled Decimal mantissa is not representable")
    log2_value = (
        math.log2(decimal_mantissa)
        + float(decimal_exponent) * math.log2(10.0)
    )
    binary_exponent = math.floor(log2_value)
    fractional = (
        float(decimal_exponent) * math.log2(10.0)
        + math.log2(decimal_mantissa)
        - binary_exponent
    )
    # ``fractional`` already contains log2(decimal_mantissa). Multiplying by
    # the decimal mantissa again inflated every non-power-of-two scaled
    # offset, eventually pushing otherwise valid local-reference cells beyond
    # their BLA radius at deep zooms.
    normalized = 2.0 ** fractional
    mantissa, shift = math.frexp(normalized)
    return sign * float(mantissa), int(binary_exponent + shift)


def _render_native_reference_points(
    *,
    render_width: int,
    render_height: int,
    log10_zoom: float,
    cell: tuple[int, int, int, int],
    probe: tuple[int, int],
    x_center: str,
    y_center: str,
    reference_x: str,
    reference_y: str,
    max_iter: int,
    native_threads: int,
    native_library: Any,
    native_reference: Any,
    series_order: int,
    series_block: int,
    render_options: NativeRenderOptions,
) -> Any:
    """Render exact global pixels around a probe-centred secondary reference.

    The point ABI accepts arbitrary scaled perturbations.  Using it here is
    both cheaper than rendering a padded local canvas and, importantly,
    avoids assuming that a probe-centred reference is aligned with the
    centre of an even-sized or non-square connected glitch box.
    """

    np = _require_numpy()
    if not hasattr(native_library, "fractal_render_points"):
        raise RuntimeError("native point renderer is unavailable")
    x0, x1, y0, y1 = cell
    probe_x, probe_y = probe
    if not (0 <= x0 < x1 <= render_width and 0 <= y0 < y1 <= render_height):
        raise ValueError("invalid point-render cell")
    if not (x0 <= probe_x < x1 and y0 <= probe_y < y1):
        raise ValueError("point-render probe is outside its cell")

    precision = max(
        96,
        _decimal_precision(x_center, y_center, log10_zoom) + 48,
        _decimal_precision(reference_x, reference_y, log10_zoom) + 48,
    )
    with localcontext() as context:
        context.prec = precision
        exponent = math.floor(log10_zoom)
        fractional_exponent = log10_zoom - exponent
        zoom_mantissa = Decimal(str(10.0 ** fractional_exponent))
        inverse_zoom = Decimal(1).scaleb(-exponent) / zoom_mantissa
        view_height = Decimal("2.8") * inverse_zoom
        view_width = view_height * Decimal(render_width) / Decimal(render_height)
        step_x = view_width / Decimal(render_width)
        step_y = view_height / Decimal(render_height)
        step_x_mantissa, step_x_exponent = _decimal_to_scaled(step_x)
        step_y_mantissa, step_y_exponent = _decimal_to_scaled(step_y)

    width = x1 - x0
    height = y1 - y0
    x_delta = np.arange(x0, x1, dtype=np.float64) - float(probe_x)
    y_delta = float(probe_y) - np.arange(y0, y1, dtype=np.float64)
    real_values = np.broadcast_to(
        step_x_mantissa * x_delta[None, :],
        (height, width),
    ).copy()
    imag_values = np.broadcast_to(
        step_y_mantissa * y_delta[:, None],
        (height, width),
    ).copy()
    real_mantissa, real_shift = np.frexp(real_values)
    imag_mantissa, imag_shift = np.frexp(imag_values)
    real_exponent = real_shift.astype(np.int32, copy=False) + int(step_x_exponent)
    imag_exponent = imag_shift.astype(np.int32, copy=False) + int(step_y_exponent)
    common_exponent = np.maximum(real_exponent, imag_exponent)
    real_mantissa = np.ldexp(
        real_mantissa,
        real_exponent.astype(np.int64) - common_exponent.astype(np.int64),
    )
    imag_mantissa = np.ldexp(
        imag_mantissa,
        imag_exponent.astype(np.int64) - common_exponent.astype(np.int64),
    )
    output = np.empty((height, width), dtype=np.float32)
    options = render_options
    status = native_library.fractal_render_points(
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        width * height,
        _zoom_text(log10_zoom),
        np.ascontiguousarray(real_mantissa, dtype=np.float64).ravel().ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
        np.ascontiguousarray(imag_mantissa, dtype=np.float64).ravel().ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
        np.ascontiguousarray(common_exponent, dtype=np.int32).ravel().ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)
        ),
        native_reference,
        max_iter,
        native_threads,
        series_order,
        series_block,
        ctypes.byref(options),
    )
    if status != 0:
        message = native_library.fractal_last_error() or b"unknown native point renderer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def render_fractal(
    width: int,
    height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    renderer: str = "auto",
    native_threads: int = 0,
    native_reference: Any = None,
    series_order: int = 3,
    series_block: int = 256,
    render_options: Optional[NativeRenderOptions] = None,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
) -> Any:
    """Render with the C-ABI backend, falling back to the Python backend."""

    global _native_notice_printed
    width, height = _validate_dimensions(width, height, "fractal")
    max_iter = _validate_iteration_count(max_iter)
    log10_zoom = _validate_log10_zoom(log10_zoom)
    x_center = _validate_center_text(x_center, "real")
    y_center = _validate_center_text(y_center, "imaginary")
    if len(julia_constant) != 2:
        raise ValueError("Julia constant must contain real and imaginary coordinates")
    julia_constant = (
        _validate_center_text(julia_constant[0], "Julia real"),
        _validate_center_text(julia_constant[1], "Julia imaginary"),
    )
    native_threads = _validate_thread_count(native_threads, "native thread count")
    series_order = _index_value(series_order, "series order")
    series_block = _index_value(series_block, "series block")
    if not 1 <= series_order <= 32:
        raise ValueError("series order must be between 1 and 32")
    if not 2 <= series_block <= 4096:
        raise ValueError("series block must be between 2 and 4096")
    formula = _formula_name(formula)
    if renderer not in {"auto", "native", "python"}:
        raise ValueError(f"unknown renderer: {renderer}")
    # The native direct ABI is intentionally a float64 preview path.  At
    # roughly e7 and deeper, an alternate-formula centre can be rounded by
    # more than a visible pixel before its orbit is iterated; the resulting
    # view is a valid image of a *different* coordinate and becomes especially
    # obvious as a blank Burning Ship/Julia zoom.  Native deep references are
    # Mandelbrot-only, so use the already vectorised high-precision reference
    # path for alternate formulas as soon as the pixel spacing needs it.  An
    # explicit native request remains an error rather than silently changing
    # backends.
    if (
        formula != "mandelbrot"
        and log10_zoom >= ALTERNATE_PERTURBATION_MIN_LOG
        and renderer == "native"
    ):
        raise RuntimeError(
            f"native alternate-formula rendering is only precise below 1e7; "
            f"use renderer=auto or python for {formula} at deeper zooms"
        )
    if formula != "mandelbrot" and log10_zoom >= ALTERNATE_PERTURBATION_MIN_LOG:
        return _render_perturbed(
            width,
            height,
            log10_zoom,
            x_center,
            y_center,
            max_iter,
            formula,
            julia_constant,
        )
    if renderer != "python":
        try:
            # Long-double direct iteration remains accurate while the pixel
            # spacing is comfortably above the decimal precision of the
            # selected centre.  Use scaled perturbation for genuinely deep
            # frames, where adding offsets directly would lose detail.
            if formula == "mandelbrot" and native_reference is not None and log10_zoom >= 12.0:
                return _render_native_reference(
                    width,
                    height,
                    log10_zoom,
                    max_iter,
                    native_threads,
                    native_reference,
                    series_order,
                    series_block,
                    render_options,
                )
            if formula != "mandelbrot" and log10_zoom >= 12.0:
                raise RuntimeError(
                    f"native e150 acceleration is currently Mandelbrot-only; "
                    f"{formula} uses the high-precision Python fallback"
                )
            if log10_zoom >= 12.0 and native_reference is None:
                raise RuntimeError("deep native rendering needs a prepared reference")
            return _render_native(
                width,
                height,
                log10_zoom,
                x_center,
                y_center,
                max_iter,
                native_threads,
                render_options,
                formula,
                julia_constant,
            )
        except RuntimeError as error:
            if renderer == "native" or (log10_zoom >= 12.0 and formula == "mandelbrot"):
                raise
            if not _native_notice_printed:
                print(f"Native renderer unavailable ({error}); using shallow Python fallback.")
                _native_notice_printed = True

    if log10_zoom > 300.0:
        raise RuntimeError("zoom beyond 10^300 requires the native MPFR renderer")

    # Once the pixel spacing is below about 1e-7, adding it to a normal
    # float64 centre starts to lose visible detail.  Perturbation keeps the
    # centre separate from the small per-pixel offsets.
    if log10_zoom < 7.0:
        return _render_direct(
            width, height, log10_zoom, x_center, y_center, max_iter,
            formula, julia_constant,
        )
    return _render_perturbed(
        width, height, log10_zoom, x_center, y_center, max_iter,
        formula, julia_constant,
    )


def max_iterations(
    log10_zoom: float,
    iteration_base: int,
    iterations_per_decade: int,
    iteration_cap: int,
) -> int:
    log10_zoom = _validate_log10_zoom(log10_zoom)
    iteration_base = _validate_iteration_count(iteration_base, "iteration base")
    iterations_per_decade = _index_value(iterations_per_decade, "iterations per decade")
    if iterations_per_decade > MAX_ITERATION_BUDGET:
        raise ValueError(
            f"iterations per decade must be at most {MAX_ITERATION_BUDGET:,}"
        )
    iteration_cap = _validate_iteration_count(iteration_cap, "iteration cap")
    if iterations_per_decade < 0:
        raise ValueError("iterations per decade cannot be negative")
    if iteration_cap < iteration_base:
        raise ValueError("iteration cap must be greater than or equal to iteration base")
    return min(
        iteration_cap,
        max(64, int(iteration_base + iterations_per_decade * max(0.0, log10_zoom))),
    )


def _zoom_plan(
    instrumental: Any,
    start_zoom: Any,
    max_zoom: Any,
    punch: float = 3.0,
    quiet_speed: float = -0.04,
    onset: Any = None,
    beat_strength: float = 0.0,
) -> Any:
    """Make loud instrumental moments punch the logarithmic zoom.

    The total travel is still exactly ``start_zoom`` to ``max_zoom`` at the
    final frame.  A small negative quiet-time velocity lets the camera ease
    back between beats; loud instrumental moments overcome that bias and
    advance the camera decisively. Pullbacks are bounded at the requested
    starting view, which keeps the absolute tile atlas reusable across songs
    while preserving backward motion after a punch.
    """

    np = _require_numpy()
    start_log = _zoom_log(start_zoom)
    max_log = _zoom_log(max_zoom)
    envelope = np.asarray(instrumental)
    if envelope.ndim != 1 or envelope.size == 0:
        raise ValueError("instrumental controls must be a non-empty one-dimensional series")
    if not np.issubdtype(envelope.dtype, np.number) or np.issubdtype(
        envelope.dtype, np.complexfloating
    ):
        raise ValueError("instrumental controls must be real numeric samples")
    envelope = np.asarray(envelope, dtype=np.float64)
    if not np.isfinite(envelope).all():
        raise ValueError("instrumental controls contain non-finite samples")
    try:
        punch_value = float(punch)
        quiet_speed_value = float(quiet_speed)
        beat_strength_value = float(beat_strength)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("zoom controls must be numeric") from error
    if not math.isfinite(punch_value) or punch_value < 0.0:
        raise ValueError("zoom punch must be finite and non-negative")
    if not math.isfinite(quiet_speed_value):
        raise ValueError("quiet zoom speed must be finite")
    if not math.isfinite(beat_strength_value) or beat_strength_value < 0.0:
        raise ValueError("beat strength must be finite and non-negative")
    if max_log <= start_log:
        return np.full(envelope.shape, start_log, dtype=np.float64)
    span = max_log - start_log
    envelope = np.clip(envelope, 0.0, 1.0)
    if envelope.size == 1:
        return np.asarray([max_log], dtype=np.float64)
    loudness = envelope ** 2.2
    # The default is intentionally slightly negative.  This is a velocity in
    # logarithmic zoom space, not a zoom position, so it makes quiet passages
    # pull back a little while strong beats still consume most of the travel.
    drive = quiet_speed_value + (1.0 + punch_value) * loudness
    if onset is not None and beat_strength_value != 0.0:
        try:
            beat_curve = np.asarray(onset, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("onset controls must be real numeric samples") from error
        if beat_curve.shape != envelope.shape:
            raise ValueError("onset and instrumental controls must have the same frame count")
        if not np.isfinite(beat_curve).all():
            raise ValueError("onset controls contain non-finite samples")
        beat_curve = np.clip(np.nan_to_num(beat_curve, nan=0.0), 0.0, 1.0)
        # A slightly sharper response keeps the transient itself visible
        # without making a sustained chorus drive the camera indefinitely.
        drive = drive + beat_strength_value * beat_curve ** 1.35
    if not np.isfinite(drive).all():
        raise ValueError("zoom controls produce a non-finite camera velocity")
    cumulative = np.concatenate(([0.0], np.cumsum(drive[:-1])))
    if not np.isfinite(cumulative).all():
        raise ValueError("zoom controls produce an unrepresentable camera path")
    # The last sample has no following frame over which to advance. Normalize
    # by the signed travelled intervals so the final video frame reaches
    # max_zoom even when the path briefly moves backwards.  A completely
    # silent track has no positive punches; make that degenerate case crawl
    # forward rather than reversing the entire movie.
    total = float(np.sum(drive[:-1])) if drive.size > 1 else 1.0
    if not math.isfinite(total):
        raise ValueError("zoom controls produce an unrepresentable camera path")
    if total <= 1.0e-6:
        drive = drive - float(np.min(drive)) + 1.0e-3
        cumulative = np.concatenate(([0.0], np.cumsum(drive[:-1])))
        total = max(float(np.sum(drive[:-1])), 1.0e-6) if drive.size > 1 else 1.0
        if not np.isfinite(cumulative).all() or not math.isfinite(total):
            raise ValueError("zoom controls produce an unrepresentable camera path")
    planned = start_log + span * cumulative / total
    if not np.isfinite(planned).all():
        raise ValueError("zoom controls produce an unrepresentable camera path")
    # Clipping creates a real still frame whenever a loud passage reaches the
    # ceiling before the final sample.  Fold the signed path at both bounds
    # instead, matching the old live renderer's direction reversal while
    # keeping every frame inside the requested zoom interval.  The final
    # sample has cumulative == total, so it still lands exactly on max_zoom.
    period = 2.0 * span
    folded = np.mod(planned - start_log, period)
    bounded = np.where(folded <= span, folded, period - folded)
    return start_log + bounded


def _crop_and_resize(
    field: Any,
    output_width: int,
    output_height: int,
    zoom_factor: float,
    resample: str = "lanczos",
) -> Any:
    np = _require_numpy()
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Pillow is required for high-resolution crop animation: pip install Pillow") from exc

    field = np.asarray(field)
    if field.ndim != 2 or field.shape[0] <= 0 or field.shape[1] <= 0:
        raise ValueError("crop source must be a non-empty two-dimensional field")
    if np.issubdtype(field.dtype, np.complexfloating):
        raise ValueError("crop source must contain real numeric samples")
    _validate_dimensions(field.shape[1], field.shape[0], "crop source")
    try:
        if not np.isfinite(field).all():
            raise ValueError("crop source contains non-finite samples")
    except TypeError as error:
        raise ValueError("crop source must contain numeric samples") from error
    field = np.asarray(field, dtype=np.float32)
    output_width, output_height = _validate_dimensions(output_width, output_height, "crop")
    if not math.isfinite(float(zoom_factor)) or float(zoom_factor) <= 0.0:
        raise ValueError("crop zoom factor must be a finite positive value")
    if resample not in {"lanczos", "bilinear"}:
        raise ValueError(f"unknown crop resample mode: {resample}")
    source_height, source_width = field.shape
    # Map output pixel centres continuously into the source image. Integer
    # crop widths/origins cause visible half-pixel jumps at small resolutions,
    # especially when the audio changes zoom velocity around a beat.
    zoom_factor = max(float(zoom_factor), 1.0)
    inverse_zoom = 1.0 / zoom_factor
    crop_width = source_width * inverse_zoom
    crop_height = source_height * inverse_zoom
    left = (source_width - crop_width) * 0.5
    top = (source_height - crop_height) * 0.5
    image = Image.fromarray(np.asarray(field, dtype=np.float32), mode="F")
    resampling = {
        # Bicubic is the highest-quality Pillow filter supported for a
        # floating-point crop box.
        "lanczos": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
    }[resample]
    # ``Image.transform(AFFINE)`` maps output coordinates directly to source
    # coordinates.  With a 125px source and 500px output it therefore samples
    # outside the source even at zoom 1, filling most of the frame with zero
    # iterations.  A floating-point crop box gives the intended centred resize
    # and avoids those visible brown/black borders and half-pixel jumps.
    resized = image.resize(
        (output_width, output_height),
        resample=resampling,
        box=(left, top, left + crop_width, top + crop_height),
    )
    return np.asarray(resized, dtype=np.float32)


def _crop_and_resize_preserving_interior(
    field: Any,
    output_width: int,
    output_height: int,
    zoom_factor: float,
    max_iter: int,
    resample: str = "lanczos",
) -> tuple[Any, Any]:
    """Resize a scalar field without interpolating its interior sentinel.

    Iteration fields use ``max_iter`` as a lossless interior marker.  Treating
    that marker as an ordinary number lets a bilinear/bicubic kernel blend it
    with an escaping neighbour; the resulting value looks like an escaped
    sample and can expose a sharp rectangular tile boundary.  Resample a
    separate coverage mask and keep a conservative half-covered sample as
    interior.  The scalar value remains interpolated only where all of the
    contributing coverage is exterior.
    """

    np = _require_numpy()
    field_array = np.asarray(field)
    if field_array.ndim != 2:
        raise ValueError("interior-aware crop source must be two-dimensional")
    max_iter = _validate_iteration_count(max_iter, "interior iteration cap")
    # The native renderer reserves the exact cap for bounded/interior pixels.
    # A smooth escaped pixel can legitimately land within half an iteration of
    # that cap; using ``cap - .5`` promoted those edge pixels to black
    # interiors and exposed rectangular atlas fills.
    source_inside = (
        np.isfinite(field_array)
        & (np.asarray(field_array, dtype=np.float64) >= float(max_iter))
    )
    if not np.any(source_inside):
        return (
            np.asarray(
                _crop_and_resize(
                    field_array,
                    output_width,
                    output_height,
                    zoom_factor,
                    resample,
                ),
                dtype=np.float32,
            ).copy(),
            np.zeros((output_height, output_width), dtype=bool),
        )

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - checked by _crop_and_resize
        raise RuntimeError("Pillow is required for high-resolution crop animation") from exc
    source_height, source_width = field_array.shape
    zoom_factor = max(float(zoom_factor), 1.0)
    inverse_zoom = 1.0 / zoom_factor
    crop_width = source_width * inverse_zoom
    crop_height = source_height * inverse_zoom
    left = (source_width - crop_width) * 0.5
    top = (source_height - crop_height) * 0.5
    # Resample exterior values and their coverage separately.  If the cap is
    # left in the scalar source, even a sub-half interior contribution can be
    # averaged into a high escaped count and reveal a rectangular tile edge.
    # Dividing by exterior coverage makes every non-interior result a true
    # average of exterior samples only, matching the native compositor.
    exterior = np.where(source_inside, 0.0, np.asarray(field_array, dtype=np.float32))
    exterior_coverage = np.asarray(~source_inside, dtype=np.float32)
    resized_exterior = np.asarray(
        _crop_and_resize(exterior, output_width, output_height, zoom_factor, resample),
        dtype=np.float64,
    )
    resized_coverage = np.asarray(
        _crop_and_resize(
            exterior_coverage,
            output_width,
            output_height,
            zoom_factor,
            resample,
        ),
        dtype=np.float64,
    )
    resized_coverage = np.clip(resized_coverage, 0.0, 1.0)
    interior_coverage = 1.0 - resized_coverage
    inside = interior_coverage >= 0.5
    valid_exterior = resized_coverage > 1.0e-6
    resized = np.zeros(resized_coverage.shape, dtype=np.float64)
    resized[valid_exterior] = (
        resized_exterior[valid_exterior] / resized_coverage[valid_exterior]
    )
    resized[inside] = float(max_iter)
    resized = np.asarray(np.nan_to_num(resized, nan=0.0, posinf=float(max_iter), neginf=0.0), dtype=np.float32)
    return resized, inside


def _zoom_chunks(zooms: Any, keyframe_factor: float) -> Any:
    """Partition an arbitrary zoom path into crop-safe keyframe ranges.

    A chunk is represented by its minimum zoom, because that is the widest
    source field needed to crop every frame in the chunk.  Tracking both
    extrema is important now that the audio response is allowed to pull back
    slightly between punches; the old monotonic-only partition could silently
    request a crop factor below one in that case.
    """

    keyframe_factor = _validate_keyframe_factor(keyframe_factor)
    limit = math.log10(max(1.05, keyframe_factor))
    start = 0
    total = len(zooms)
    chunk_count = 0
    while start < total:
        low = high = float(zooms[start])
        end = start + 1
        while end < total:
            candidate = float(zooms[end])
            candidate_low = min(low, candidate)
            candidate_high = max(high, candidate)
            if candidate_high - candidate_low >= limit:
                break
            low = candidate_low
            high = candidate_high
            end += 1
        chunk_count += 1
        if chunk_count > MAX_KEYFRAME_LEVELS:
            raise ValueError(
                f"zoom path would require more than {MAX_KEYFRAME_LEVELS:,} keyframes; "
                "increase keyframe factor or shorten the render"
            )
        yield start, end, low, high
        start = end


def _keyframe_count(zooms: Any, keyframe_factor: float) -> int:
    return sum(1 for _ in _zoom_chunks(zooms, keyframe_factor))


def _atlas_geometry(
    zooms: Any,
    keyframe_factor: float,
) -> tuple[float, float, int]:
    """Return the fixed logarithmic tile ladder for a camera path.

    Unlike the legacy chunker, the ladder is independent of audio timing. A
    tile at level ``n`` covers one factor-sized central region of its parent;
    the compositor uses the parent for the outer image and the child for the
    newly revealed centre. This keeps every tile at native sampling density
    while avoiding a factor-squared full-frame render at every level.
    """

    if len(zooms) == 0:
        raise ValueError("cannot build an atlas for an empty zoom path")
    keyframe_factor = _validate_keyframe_factor(keyframe_factor)
    step = math.log10(max(1.05, keyframe_factor))
    minimum = float(min(float(value) for value in zooms))
    maximum = float(max(float(value) for value in zooms))
    minimum = _validate_log10_zoom(minimum, "minimum zoom path value")
    maximum = _validate_log10_zoom(maximum, "maximum zoom path value")
    origin = math.floor(minimum / step) * step
    level_count = max(0, int(math.ceil((maximum - origin) / step - 1.0e-12)))
    if level_count > MAX_KEYFRAME_LEVELS:
        raise ValueError(
            f"zoom path would require more than {MAX_KEYFRAME_LEVELS:,} atlas levels; "
            "increase keyframe factor or shorten the render"
        )
    return origin, step, level_count


def _atlas_level_for_zoom(log_zoom: float, origin: float, step: float, level_count: int) -> int:
    level = int(math.floor((float(log_zoom) - origin) / step + 1.0e-9))
    return max(0, min(level, level_count))


def _trim_atlas_memory_cache(tile_cache: dict[int, Any], active_level: int) -> None:
    """Keep only the parent/current/child tiles needed around a level."""

    for old_level in tuple(tile_cache):
        if abs(old_level - active_level) > 1:
            del tile_cache[old_level]


def _atlas_tile_path(
    cache_dir: Optional[Path],
    cache_identity: str,
    render_width: int,
    render_height: int,
    level: int,
    log_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    series_order: int,
    series_block: int,
) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_key = hashlib.sha256(
        repr((
            ATLAS_CACHE_SCHEMA,
            cache_identity,
            render_width,
            render_height,
            level,
            log_zoom,
            x_center,
            y_center,
            max_iter,
            series_order,
            series_block,
        )).encode("utf-8")
    ).hexdigest()[:24]
    return cache_dir / f"atlas-tile-{cache_key}.npy"


def _atlas_local_reference_centres(
    render_width: int,
    render_height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    divisions: int = 2,
) -> list[tuple[int, int, int, int, str, str]]:
    """Return exact Decimal centres for a regular tile subdivision.

    The native viewport maps pixel centres as
    ``(pixel - (size - 1) / 2) / size``.  Computing the quadrant centre with
    integer numerators preserves that mapping when a parent tile is replaced
    by equal-ish views at their corresponding per-cell zoom. Binary floats
    cannot express the offsets once the zoom passes roughly 1e308, so only
    the small geometry fractions are floats; the scale and centre arithmetic
    stay in Decimal at a precision derived from the requested depth.
    """

    if divisions < 2:
        raise ValueError("local-reference subdivision needs at least 2 divisions")
    if render_width < 2 * divisions or render_height < 2 * divisions:
        raise ValueError("local-reference tile is too small for its subdivision")
    if not math.isfinite(log10_zoom):
        raise ValueError("tile zoom must be finite")

    precision = max(
        96,
        _decimal_precision(x_center, y_center, log10_zoom) + 48,
    )
    x_splits = [
        (index * render_width // divisions, (index + 1) * render_width // divisions)
        for index in range(divisions)
    ]
    y_splits = [
        (index * render_height // divisions, (index + 1) * render_height // divisions)
        for index in range(divisions)
    ]
    with localcontext() as context:
        context.prec = precision
        exponent = math.floor(log10_zoom)
        fractional_exponent = log10_zoom - exponent
        # 10**fractional_exponent is bounded in [1, 10), so this conversion
        # remains safe even when the integer exponent is several thousand.
        zoom_mantissa = Decimal(str(10.0 ** fractional_exponent))
        inverse_zoom = Decimal(1).scaleb(-exponent) / zoom_mantissa
        view_height = Decimal("2.8") * inverse_zoom
        view_width = view_height * Decimal(render_width) / Decimal(render_height)
        base_x = Decimal(x_center)
        base_y = Decimal(y_center)
        denominator_x = Decimal(2 * render_width)
        denominator_y = Decimal(2 * render_height)
        result: list[tuple[int, int, int, int, str, str]] = []
        for y0, y1 in y_splits:
            centre_y_numerator = y0 + y1 - 1
            y_fraction = (
                Decimal(render_height - 1 - centre_y_numerator)
                / denominator_y
            )
            local_y = base_y + view_height * y_fraction
            for x0, x1 in x_splits:
                centre_x_numerator = x0 + x1 - 1
                x_fraction = (
                    Decimal(centre_x_numerator - (render_width - 1))
                    / denominator_x
                )
                local_x = base_x + view_width * x_fraction
                result.append((x0, x1, y0, y1, str(local_x), str(local_y)))
        return result


def _atlas_local_reference_cell(
    render_width: int,
    render_height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    cell: tuple[int, int, int, int],
    probe: Optional[tuple[int, int]] = None,
) -> tuple[int, int, int, int, str, str, float]:
    """Compute one exact local view for an arbitrary glitch-cell rectangle."""

    x0, x1, y0, y1 = cell
    if not (0 <= x0 < x1 <= render_width and 0 <= y0 < y1 <= render_height):
        raise ValueError("invalid local-reference cell")
    precision = max(96, _decimal_precision(x_center, y_center, log10_zoom) + 48)
    with localcontext() as context:
        context.prec = precision
        exponent = math.floor(log10_zoom)
        fractional_exponent = log10_zoom - exponent
        zoom_mantissa = Decimal(str(10.0 ** fractional_exponent))
        inverse_zoom = Decimal(1).scaleb(-exponent) / zoom_mantissa
        view_height = Decimal("2.8") * inverse_zoom
        view_width = view_height * Decimal(render_width) / Decimal(render_height)
        base_x = Decimal(x_center)
        base_y = Decimal(y_center)
        if probe is None:
            x_numerator = Decimal(x0 + x1 - 1) - Decimal(render_width - 1)
            y_numerator = Decimal(render_height - 1 - (y0 + y1 - 1))
        else:
            probe_x, probe_y = probe
            if not (x0 <= probe_x < x1 and y0 <= probe_y < y1):
                raise ValueError("glitch probe is outside its reference cell")
            # Keep the local view scale based on the failed region, while
            # placing the high-precision reference on an actual failed pixel
            # rather than on the centre of a sparse bounding box.
            x_numerator = Decimal(2 * probe_x) - Decimal(render_width - 1)
            y_numerator = Decimal(render_height - 1) - Decimal(2 * probe_y)
        x_fraction = x_numerator / (Decimal(2) * Decimal(render_width))
        y_fraction = y_numerator / (Decimal(2) * Decimal(render_height))
        local_x = base_x + view_width * x_fraction
        local_y = base_y + view_height * y_fraction
    scale = max(
        render_width / float(x1 - x0),
        render_height / float(y1 - y0),
    )
    local_log_zoom = float(log10_zoom) + math.log10(scale)
    return x0, x1, y0, y1, str(local_x), str(local_y), local_log_zoom


def _unresolved_regions(mask: Any) -> list[tuple[int, int, int, int, int, int, int]]:
    """Return ``(box, count, probe)`` regions, largest unresolved first."""

    np = _require_numpy()
    unresolved = np.asarray(mask, dtype=bool)
    if unresolved.ndim != 2:
        raise ValueError("unresolved mask must be two-dimensional")
    height, width = unresolved.shape
    visited = np.zeros_like(unresolved, dtype=bool)
    regions: list[tuple[int, int, int, int, int, int, int]] = []
    for seed_y, seed_x in zip(*np.nonzero(unresolved)):
        if visited[seed_y, seed_x]:
            continue
        queue_cells = deque([(int(seed_y), int(seed_x))])
        visited[seed_y, seed_x] = True
        x0 = x1 = int(seed_x)
        y0 = y1 = int(seed_y)
        count = 0
        sum_x = 0
        sum_y = 0
        while queue_cells:
            y, x = queue_cells.pop()
            count += 1
            sum_x += x
            sum_y += y
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbour_y = y + dy
                    neighbour_x = x + dx
                    if (
                        0 <= neighbour_y < height
                        and 0 <= neighbour_x < width
                        and unresolved[neighbour_y, neighbour_x]
                        and not visited[neighbour_y, neighbour_x]
                    ):
                        visited[neighbour_y, neighbour_x] = True
                        queue_cells.append((neighbour_y, neighbour_x))
        # Include a one-pixel validation halo. It lets the local reference
        # prove the approximation at the boundary of the failed region too.
        probe_x = int(round(sum_x / max(count, 1)))
        probe_y = int(round(sum_y / max(count, 1)))
        regions.append((
            max(0, x0 - 1),
            min(width, x1 + 2),
            max(0, y0 - 1),
            min(height, y1 + 2),
            count,
            probe_x,
            probe_y,
        ))
    regions.sort(key=lambda region: region[4], reverse=True)
    return regions


def _spatial_recover_field(
    local_field: Any,
    fallback_field: Optional[Any] = None,
) -> tuple[Any, int]:
    """Fill deadline-marked pixels without creating a constant rectangle.

    ``fallback_field`` should be the same world-space region from a cheaper
    source, normally the central crop of the parent atlas tile. Any remaining
    holes are propagated from finite eight-neighbour samples. A completely
    unresolved field is rejected so callers can use their ordinary global
    reference fallback instead of silently producing a fake coloured block.
    """

    np = _require_numpy()
    result = np.asarray(local_field, dtype=np.float32).copy()
    if result.ndim != 2 or result.size == 0:
        raise ValueError("spatial recovery requires a non-empty 2-D field")
    missing_before = ~np.isfinite(result)
    if not missing_before.any():
        return result, 0

    if fallback_field is not None:
        fallback = np.asarray(fallback_field, dtype=np.float32)
        if fallback.shape != result.shape:
            raise ValueError("spatial recovery fallback shape does not match the field")
        fallback_finite = np.isfinite(fallback)
        use_fallback = missing_before & fallback_finite
        result[use_fallback] = fallback[use_fallback]

    shape = result.shape
    if not np.isfinite(result).all():
        # A damaged/partial parent should not reintroduce a flat rectangle.
        # Propagate nearby finite values inward with an eight-neighbour
        # average; this is only a last-resort edge repair for pixels
        # unavailable in both native passes.
        for _ in range(max(shape)):
            finite = np.isfinite(result)
            if finite.all():
                break
            padded = np.pad(
                result,
                1,
                mode="constant",
                constant_values=np.nan,
            )
            total = np.zeros(shape, dtype=np.float32)
            count = np.zeros(shape, dtype=np.float32)
            for dy in range(3):
                for dx in range(3):
                    if dx == 1 and dy == 1:
                        continue
                    neighbour = padded[dy:dy + shape[0], dx:dx + shape[1]]
                    valid = np.isfinite(neighbour)
                    total[valid] += neighbour[valid]
                    count[valid] += 1.0
            fillable = ~finite & (count > 0.0)
            if not fillable.any():
                break
            result[fillable] = total[fillable] / count[fillable]
    if not np.isfinite(result).all():
        raise RuntimeError("adaptive local reference has no spatially valid recovery")
    return result, int(missing_before.sum())


def _render_exact_mandelbrot_pixel(
    x_center: str,
    y_center: str,
    max_iter: int,
    log10_zoom: float,
) -> float:
    """Render one Mandelbrot pixel with arbitrary-precision arithmetic.

    This is a strict last-resort path for a very small unresolved mask after
    native perturbation/reference retries.  It evaluates the exact pixel
    coordinate directly, so it cannot inherit a bad BLA radius or a rounded
    double-tail state.  The normal image path never calls this function.
    """

    try:
        import mpmath as mp
    except ImportError as error:  # pragma: no cover - dependency is packaged
        raise RuntimeError(
            "mpmath is required for exact deep-pixel recovery"
        ) from error

    precision = max(
        64,
        _decimal_precision(x_center, y_center, log10_zoom) + 32,
    )
    with mp.workdps(precision):
        parameter_real = mp.mpf(x_center)
        parameter_imag = mp.mpf(y_center)
        value_real = mp.mpf("0")
        value_imag = mp.mpf("0")
        for iteration in range(1, int(max_iter) + 1):
            next_real = (
                value_real * value_real
                - value_imag * value_imag
                + parameter_real
            )
            next_imag = (
                2 * value_real * value_imag
                + parameter_imag
            )
            value_real = next_real
            value_imag = next_imag
            magnitude_squared = (
                value_real * value_real
                + value_imag * value_imag
            )
            if magnitude_squared > 4:
                magnitude = mp.sqrt(magnitude_squared)
                smooth = (
                    iteration
                    - mp.log(mp.log(magnitude)) / mp.log(2)
                )
                return float(smooth)
    return float(max_iter)


def _atlas_glitch_reference_field(
    *,
    render_width: int,
    render_height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    series_order: int,
    series_block: int,
    native_threads: int,
    native_library: Any,
    native_backend: int,
    native_reference: Any,
    fallback_field: Optional[Any],
    fallback_zoom_factor: float,
    fallback_max_iter: Optional[int],
    allow_recovery: bool,
    diagnostics: Optional[dict[str, int]] = None,
    native_reference_root: Any = None,
) -> Any:
    """Render a tile, refining only pixels that the shared reference cannot solve.

    The shared reference gets one whole-image pass.  A strict native timeout
    reports unresolved pixels as NaN; connected NaN regions are then used as
    the exact centres of secondary references.  This makes the expensive
    reference count proportional to real perturbation glitches rather than to
    an arbitrary 2x2/4x4 grid.  Quality modes never copy parent/interpolated
    values: a remaining unresolved region raises instead.
    """

    np = _require_numpy()
    field = np.full((render_height, render_width), np.nan, dtype=np.float32)
    fallback_array = None
    if fallback_field is not None:
        fallback_array = np.asarray(fallback_field, dtype=np.float32)
        if fallback_array.ndim != 2 or fallback_array.size == 0:
            raise ValueError("glitch recovery fallback must be a non-empty field")
    parent_fallback = None
    if fallback_array is not None:
        parent_fallback, _ = _crop_and_resize_preserving_interior(
            fallback_array,
            render_width,
            render_height,
            max(float(fallback_zoom_factor), 1.0),
            fallback_max_iter if fallback_max_iter is not None else max_iter,
            "bilinear",
        )
        parent_fallback = np.asarray(parent_fallback, dtype=np.float32)
        if fallback_max_iter is not None:
            parent_fallback = np.where(
                parent_fallback >= float(fallback_max_iter),
                float(max_iter),
                parent_fallback,
            ).astype(np.float32, copy=False)

    def budget_ms(
        cell_width: int,
        cell_height: int,
        final_retry: bool = False,
        whole_tile: bool = False,
    ) -> int:
        if final_retry:
            # Quality output must not convert a hard pixel into a parent
            # sample merely because a diagnostic deadline was too short. The
            # final selected glitch is a bounded exact perturbation job
            # (max_iter is finite), so let it finish. Draft keeps its short
            # deadline and may use labelled spatial recovery.
            return 0 if not allow_recovery else ATLAS_LOCAL_REFERENCE_FINAL_BUDGET_MS
        estimate = int(
            math.ceil(
                cell_width
                * cell_height
                * ATLAS_LOCAL_REFERENCE_MS_PER_PIXEL
            )
        )
        # Keep the initial pass short and bounded. Its unresolved mask is a
        # work queue, not a quality fallback; local references then spend
        # their budget only where the shared pass ran out of time.
        # The initial shared pass is deliberately a short bounded work-queue
        # pass; local references spend time only on its actual unresolved
        # regions. ``whole_tile`` remains an explicit argument to make that
        # policy visible at the call site and for future budget tiers.
        floor = ATLAS_LOCAL_REFERENCE_MIN_BUDGET_MS
        return max(
            floor,
            min(
                ATLAS_LOCAL_REFERENCE_TILE_BUDGET_MS,
                max(estimate, floor),
            ),
        )

    def render_with_reference(
        width: int,
        height: int,
        centre_x: str,
        centre_y: str,
        local_log_zoom: float,
        reference: Any,
        *,
        final_retry: bool = False,
    ) -> Any:
        options = NativeRenderOptions(
            # Keep native output strict even for draft mode. Draft recovery
            # belongs to this Python layer; native sentinel iteration values
            # would hide the actual glitch mask and prevent refinement.
            strict=True,
            allow_recovery=False,
            time_budget_ms=budget_ms(width, height, final_retry),
            disable_bla=final_retry,
            strict_cycle=True,
            backend=native_backend,
        )
        return np.asarray(
            render_fractal(
                width,
                height,
                local_log_zoom,
                centre_x,
                centre_y,
                max_iter,
                "native",
                native_threads,
                reference,
                series_order,
                series_block,
                options,
            ),
            dtype=np.float32,
        )

    def render_points_with_reference(
        cell: tuple[int, int, int, int],
        probe: tuple[int, int],
        local_x: str,
        local_y: str,
        local_log_zoom: float,
        reference: Any,
        *,
        final_retry: bool = False,
    ) -> Any:
        options = NativeRenderOptions(
            strict=True,
            allow_recovery=False,
            time_budget_ms=budget_ms(
                cell[1] - cell[0],
                cell[3] - cell[2],
                final_retry,
            ),
            disable_bla=final_retry,
            strict_cycle=True,
            backend=native_backend,
        )
        return np.asarray(
            _render_native_reference_points(
                render_width=render_width,
                render_height=render_height,
                log10_zoom=log10_zoom,
                cell=cell,
                probe=probe,
                x_center=x_center,
                y_center=y_center,
                reference_x=local_x,
                reference_y=local_y,
                max_iter=max_iter,
                native_threads=native_threads,
                native_library=native_library,
                native_reference=reference,
                series_order=series_order,
                series_block=series_block,
                render_options=options,
            ),
            dtype=np.float32,
        )

    def recover(cell: tuple[int, int, int, int], local_field: Any) -> None:
        if not allow_recovery:
            raise RuntimeError(
                "strict atlas render encountered unresolved pixels in a "
                "glitch-driven secondary reference"
            )
        x0, x1, y0, y1 = cell
        local = np.asarray(local_field, dtype=np.float32).copy()
        fallback = None if parent_fallback is None else parent_fallback[y0:y1, x0:x1]
        if fallback is None:
            existing = field[y0:y1, x0:x1]
            fallback = existing if np.isfinite(existing).any() else None
        local, _ = _spatial_recover_field(local, fallback)
        field[y0:y1, x0:x1] = local

    def exact_pixel_recovery(
        mask: Any,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> int:
        """Resolve a small global/local NaN mask without approximation."""

        unresolved_pixels = np.argwhere(np.asarray(mask, dtype=bool))
        if (
            unresolved_pixels.size == 0
            or allow_recovery
            or len(unresolved_pixels) > ATLAS_LOCAL_REFERENCE_MAX_EXACT_PIXELS
        ):
            return 0
        recovered = 0
        for local_y_index, local_x_index in unresolved_pixels:
            pixel_x = x_offset + int(local_x_index)
            pixel_y = y_offset + int(local_y_index)
            if not (
                0 <= pixel_x < render_width
                and 0 <= pixel_y < render_height
            ):
                raise ValueError("exact pixel recovery coordinate is outside the tile")
            pixel_cell = (pixel_x, pixel_x + 1, pixel_y, pixel_y + 1)
            pixel_geometry = _atlas_local_reference_cell(
                render_width,
                render_height,
                log10_zoom,
                x_center,
                y_center,
                pixel_cell,
                (pixel_x, pixel_y),
            )
            _, _, _, _, pixel_x_center, pixel_y_center, _ = pixel_geometry
            field[pixel_y, pixel_x] = _render_exact_mandelbrot_pixel(
                pixel_x_center,
                pixel_y_center,
                max_iter,
                log10_zoom,
            )
            recovered += 1
        if diagnostics is not None:
            diagnostics["exact_pixel_recoveries"] = (
                diagnostics.get("exact_pixel_recoveries", 0) + recovered
            )
        return recovered

    print(
        f"  Using glitch-driven shared reference for deep tile "
        f"({_zoom_label(log10_zoom)}).",
        flush=True,
    )
    shared_options = NativeRenderOptions(
        strict=True,
        allow_recovery=False,
        time_budget_ms=budget_ms(render_width, render_height, whole_tile=True),
        strict_cycle=True,
        backend=native_backend,
    )
    shared_field = np.asarray(
        render_fractal(
            render_width,
            render_height,
            log10_zoom,
            x_center,
            y_center,
            max_iter,
            "native",
            native_threads,
            native_reference,
            series_order,
            series_block,
            shared_options,
        ),
        dtype=np.float32,
    )
    finite = np.isfinite(shared_field)
    field[finite] = shared_field[finite]
    unresolved = ~finite
    if diagnostics is not None:
        diagnostics.update({
            "initial_unresolved_pixels": int(unresolved.sum()),
            "initial_regions": 0,
            "secondary_references": 0,
            "refined_regions": 0,
            "exact_pixel_recoveries": 0,
            "recovered_pixels": 0,
            "final_unresolved_pixels": 0,
        })
    if not unresolved.any():
        return field

    # A decade-spaced tier can be mathematically valid yet land on an
    # unfortunate BLA map boundary for this particular tile. Before building
    # a tree of small secondary references, retarget one clone of the reusable
    # root just inside the current viewport. This reuses the already-built
    # MPFR orbit, adds no approximation to the strict output, and is usually
    # enough to turn a large timeout mask into a handful of exact pixels.
    clone = getattr(native_library, "fractal_clone_reference", None)
    if (
        native_reference_root is not None
        and clone is not None
        and log10_zoom >= 12.0
    ):
        retarget_log = max(12.0, float(log10_zoom) - 0.1)
        retargeted_reference = None
        try:
            retargeted_reference = _clone_native_reference(
                native_library,
                native_reference_root,
                retarget_log,
            )
            retargeted = np.asarray(
                render_fractal(
                    render_width,
                    render_height,
                    log10_zoom,
                    x_center,
                    y_center,
                    max_iter,
                    "native",
                    native_threads,
                    retargeted_reference,
                    series_order,
                    series_block,
                    NativeRenderOptions(
                        strict=True,
                        allow_recovery=False,
                        time_budget_ms=0,
                        strict_cycle=True,
                        backend=native_backend,
                    ),
                ),
                dtype=np.float32,
            )
            retargeted_finite = np.isfinite(retargeted)
            repaired = unresolved & retargeted_finite
            if repaired.any():
                field[repaired] = retargeted[repaired]
                print(
                    f"  Retargeted BLA tier repaired {int(repaired.sum())} "
                    "deep pixels before local refinement.",
                    flush=True,
                )
            unresolved = ~np.isfinite(field)
            if not unresolved.any():
                return field
        except (RuntimeError, ValueError):
            # A retargeted tier is an optimization, not a correctness
            # dependency. Keep the existing connected-region repair path as
            # the fallback if clone support or the candidate radius is not
            # available for a particular build.
            pass
        finally:
            if retargeted_reference is not None:
                try:
                    native_library.fractal_destroy_reference(retargeted_reference)
                except Exception:
                    pass

    if not allow_recovery:
        exact_recovered = exact_pixel_recovery(unresolved)
        unresolved = ~np.isfinite(field)
        if exact_recovered:
            print(
                f"  Exact arbitrary-precision retry repaired {exact_recovered} "
                "deep pixels.",
                flush=True,
            )
        if not unresolved.any():
            return field

    regions = _unresolved_regions(unresolved)
    if diagnostics is not None:
        diagnostics["initial_regions"] = len(regions)
    pending: list[tuple[tuple[int, int, int, int], int, int, tuple[int, int]]] = [
        ((x0, x1, y0, y1), 0, count, (probe_x, probe_y))
        for x0, x1, y0, y1, count, probe_x, probe_y in regions
    ]
    max_depth = int(math.log2(ATLAS_LOCAL_REFERENCE_MAX_DIVISIONS))
    minimum_cell = 8
    secondary_references = 0
    refined_regions = 0
    recovered_pixels = 0
    tile_deadline = (
        time.monotonic() + ATLAS_LOCAL_REFERENCE_TILE_BUDGET_MS / 1000.0
        if allow_recovery else float("inf")
    )

    while pending:
        cell, depth, _, probe = pending.pop()
        x0, x1, y0, y1 = cell
        if x1 <= x0 or y1 <= y0:
            continue
        if time.monotonic() >= tile_deadline:
            # The deadline is enabled only in draft mode. Strict quality
            # renders continue until every selected glitch is resolved or a
            # native call reports an unresolved tail.
            recover(cell, np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32))
            recovered_pixels += (x1 - x0) * (y1 - y0)
            continue

        geometry = _atlas_local_reference_cell(
            render_width,
            render_height,
            log10_zoom,
            x_center,
            y_center,
            cell,
            probe,
        )
        _, _, _, _, local_x, local_y, local_log_zoom = geometry
        local_library, local_reference = _create_native_reference(
            local_x,
            local_y,
            max_iter,
            local_log_zoom,
            series_order,
            local_log_zoom,
        )
        secondary_references += 1
        if diagnostics is not None:
            diagnostics["secondary_references"] = secondary_references
        try:
            local_field = render_points_with_reference(
                cell,
                probe,
                local_x,
                local_y,
                local_log_zoom,
                local_reference,
            )
            local_finite = np.isfinite(local_field)
            local_view = field[y0:y1, x0:x1]
            local_view[local_finite] = local_field[local_finite]
            if local_finite.all():
                continue

            local_regions = _unresolved_regions(~local_finite)
            can_split = (
                depth < max_depth
                and max(x1 - x0, y1 - y0) > minimum_cell
            )
            if can_split:
                refined_regions += 1
                if diagnostics is not None:
                    diagnostics["refined_regions"] = refined_regions
                next_pending = []
                for lx0, lx1, ly0, ly1, count, probe_x, probe_y in local_regions:
                    global_region = (x0 + lx0, x0 + lx1, y0 + ly0, y0 + ly1)
                    global_probe = (x0 + probe_x, y0 + probe_y)
                    next_pending.append(
                        (global_region, depth + 1, count, global_probe)
                    )
                # `_unresolved_regions` is largest-first; the stack pops the
                # last item, so reverse it to process the largest glitch first
                # and keep the queue bounded by actual components.
                pending.extend(reversed(next_pending))
                continue

            # One exact scaled retry is the final quality-preserving recovery.
            # It still returns NaN on a deadline; strict modes then fail rather
            # than turning the unresolved tail into a flat rectangle.
            local_retry = render_points_with_reference(
                cell,
                probe,
                local_x,
                local_y,
                local_log_zoom,
                local_reference,
                final_retry=True,
            )
            retry_finite = np.isfinite(local_retry)
            local_view[retry_finite] = local_retry[retry_finite]
            if not retry_finite.all():
                unresolved_local = ~np.isfinite(local_view)
                exact_recovered = exact_pixel_recovery(
                    unresolved_local,
                    x_offset=x0,
                    y_offset=y0,
                )
                if exact_recovered:
                    print(
                        f"  Exact arbitrary-precision retry repaired {exact_recovered} "
                        "local deep pixels.",
                        flush=True,
                    )
                if not np.isfinite(local_view).all():
                    recover(cell, local_view)
        finally:
            local_library.fractal_destroy_reference(local_reference)

    if not np.isfinite(field).all():
        if allow_recovery and parent_fallback is not None:
            field, recovered = _spatial_recover_field(field, parent_fallback)
            recovered_pixels += recovered
        else:
            unresolved_count = int((~np.isfinite(field)).sum())
            raise RuntimeError(
                f"strict atlas render left {unresolved_count} unresolved pixels "
                "after glitch-driven references"
            )
    if diagnostics is not None:
        diagnostics["recovered_pixels"] = recovered_pixels
        diagnostics["final_unresolved_pixels"] = int((~np.isfinite(field)).sum())
    print(
        f"  Glitch-driven references: {secondary_references} secondary, "
        f"{len(regions)} initial regions, {refined_regions} refined regions"
        + (f", {recovered_pixels} draft-recovered pixels" if recovered_pixels else "")
        + ".",
        flush=True,
    )
    return field


def _atlas_local_reference_field(
    *,
    render_width: int,
    render_height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    series_order: int,
    series_block: int,
    renderer: str,
    native_threads: int,
    native_library: Any,
    native_backend: int = 0,
    native_reference: Any = None,
    native_reference_root: Any = None,
    fallback_field: Optional[Any] = None,
    fallback_zoom_factor: float = 2.0,
    fallback_max_iter: Optional[int] = None,
    allow_recovery: bool = False,
    diagnostics: Optional[dict[str, int]] = None,
) -> Optional[Any]:
    """Render a deep tile with an adaptive secondary-reference grid.

    With a shared reference, the first pass is a bounded native work queue and
    only connected unresolved regions receive probe-centred secondary
    references. Those regions are rendered through the point ABI, so the
    global pixel coordinates remain exact even for odd, rectangular boxes.
    The compatibility path without a shared reference retains the older local
    grid. Draft mode may recover a deadline tail from a parent crop; quality
    modes reject unresolved pixels instead of fabricating a rectangle.

    ``None`` means that the ordinary single-reference path should be used.
    """

    if renderer not in {"auto", "native"} or native_library is None:
        return None
    if log10_zoom < ATLAS_LOCAL_REFERENCE_MIN_LOG:
        return None
    if native_reference is not None:
        return _atlas_glitch_reference_field(
            render_width=render_width,
            render_height=render_height,
            log10_zoom=log10_zoom,
            x_center=x_center,
            y_center=y_center,
            max_iter=max_iter,
            series_order=series_order,
            series_block=series_block,
            native_threads=native_threads,
            native_library=native_library,
            native_backend=native_backend,
            native_reference=native_reference,
            native_reference_root=native_reference_root,
            fallback_field=fallback_field,
            fallback_zoom_factor=fallback_zoom_factor,
            fallback_max_iter=fallback_max_iter,
            allow_recovery=allow_recovery,
            diagnostics=diagnostics,
        )
    if (
        render_width < ATLAS_LOCAL_REFERENCE_MIN_DIMENSION
        or render_height < ATLAS_LOCAL_REFERENCE_MIN_DIMENSION
    ):
        return None

    np = _require_numpy()
    # At ultra-deep levels the feature scale can be much narrower than a
    # quadrant. Start one level finer there; at ordinary deep levels the
    # cheaper 2x2 probe still catches the usual boundary cliff.
    initial_divisions = 4 if log10_zoom >= 80.0 else 2
    if render_width < 2 * initial_divisions or render_height < 2 * initial_divisions:
        return None
    initial_cells = _atlas_local_reference_centres(
        render_width,
        render_height,
        log10_zoom,
        x_center,
        y_center,
        initial_divisions,
    )
    centres_by_divisions = {initial_divisions: initial_cells}
    field = np.full((render_height, render_width), np.nan, dtype=np.float32)
    fallback_array = None
    if fallback_field is not None:
        fallback_array = np.asarray(fallback_field, dtype=np.float32)
        if fallback_array.ndim != 2 or fallback_array.size == 0:
            raise ValueError("atlas recovery fallback must be a non-empty 2-D field")
    parent_fallback = None
    print(
        f"  Using adaptive local native references for deep tile "
        f"({_zoom_label(log10_zoom)}).",
        flush=True,
    )

    def budget_ms(cell_width: int, cell_height: int) -> int:
        estimate = int(
            math.ceil(
                cell_width
                * cell_height
                * ATLAS_LOCAL_REFERENCE_MS_PER_PIXEL
            )
        )
        return max(
            ATLAS_LOCAL_REFERENCE_MIN_BUDGET_MS,
            min(ATLAS_LOCAL_REFERENCE_MAX_BUDGET_MS, estimate),
        )

    def child_cells(cell: tuple[int, int, int, int, str, str], divisions: int):
        x0, x1, y0, y1, _, _ = cell
        next_divisions = divisions * 2
        all_children = centres_by_divisions.get(next_divisions)
        if all_children is None:
            all_children = _atlas_local_reference_centres(
                render_width,
                render_height,
                log10_zoom,
                x_center,
                y_center,
                next_divisions,
            )
            centres_by_divisions[next_divisions] = all_children
        return [
            candidate
            for candidate in all_children
            if candidate[0] >= x0
            and candidate[1] <= x1
            and candidate[2] >= y0
            and candidate[3] <= y1
        ]

    def render_cell(
        cell: tuple[int, int, int, int, str, str],
        divisions: int,
        final_retry: bool = False,
    ) -> Any:
        x0, x1, y0, y1, local_x, local_y = cell
        local_log_zoom = log10_zoom + math.log10(
            render_height / float(y1 - y0)
        )
        _, reference = _create_native_reference(
            local_x,
            local_y,
            max_iter,
            local_log_zoom,
            series_order,
            local_log_zoom,
        )
        try:
            render_options = NativeRenderOptions(
                strict=not allow_recovery,
                allow_recovery=allow_recovery,
                time_budget_ms=(
                    ATLAS_LOCAL_REFERENCE_FINAL_BUDGET_MS
                    if final_retry
                    else budget_ms(x1 - x0, y1 - y0)
                ),
                disable_bla=final_retry,
                strict_cycle=True,
                backend=native_backend,
            )
            return np.asarray(
                render_fractal(
                    x1 - x0,
                    y1 - y0,
                    local_log_zoom,
                    local_x,
                    local_y,
                    max_iter,
                    "native",
                    native_threads,
                    reference,
                    series_order,
                    series_block,
                    render_options,
                ),
                dtype=np.float32,
            )
        finally:
            native_library.fractal_destroy_reference(reference)

    pending = [(cell, initial_divisions) for cell in initial_cells]
    completed_cells = 0
    refined_cells = 0
    final_retries = 0
    recovered_cells = 0
    recovered_pixels = 0
    tile_deadline = (
        time.monotonic() + ATLAS_LOCAL_REFERENCE_TILE_BUDGET_MS / 1000.0
        if allow_recovery else float("inf")
    )

    def parent_fallback_view() -> Any:
        nonlocal parent_fallback
        if parent_fallback is not None:
            return parent_fallback
        if fallback_array is None:
            return None
        # A child atlas tile is one fixed zoom factor narrower than its
        # parent. Resample the parent's central crop once, lazily, only when
        # the deadline/recovery path is actually needed.
        parent_fallback, _ = _crop_and_resize_preserving_interior(
            fallback_array,
            render_width,
            render_height,
            max(float(fallback_zoom_factor), 1.0),
            fallback_max_iter if fallback_max_iter is not None else max_iter,
            "bilinear",
        )
        parent_fallback = np.asarray(parent_fallback, dtype=np.float32)
        if fallback_max_iter is not None:
            # Interior pixels in the parent are encoded at its iteration cap.
            # Translate that sentinel to the current tile's cap instead of
            # turning parent interiors into a false coloured escape band.
            parent_fallback = np.where(
                parent_fallback >= float(fallback_max_iter),
                float(max_iter),
                parent_fallback,
            ).astype(np.float32, copy=False)
        return parent_fallback

    def complete_with_recovery(
        cell: tuple[int, int, int, int, str, str],
        local_field: Optional[Any] = None,
    ) -> None:
        nonlocal recovered_cells, recovered_pixels
        if not allow_recovery:
            raise RuntimeError(
                "strict atlas render encountered unresolved pixels; "
                "increase native precision/iteration budget or use --quality draft"
            )
        x0, x1, y0, y1, _, _ = cell
        shape = (y1 - y0, x1 - x0)
        if local_field is None:
            local_field = np.full(shape, np.nan, dtype=np.float32)
        else:
            local_field = np.asarray(local_field, dtype=np.float32).copy()
        fallback = parent_fallback_view()
        fallback_values = None if fallback is None else fallback[y0:y1, x0:x1]
        local_field, recovered = _spatial_recover_field(local_field, fallback_values)
        if recovered:
            recovered_pixels += recovered
            recovered_cells += 1
        field[y0:y1, x0:x1] = local_field

    while pending:
            cell, divisions = pending.pop()
            x0, x1, y0, y1, _, _ = cell
            if time.monotonic() >= tile_deadline:
                complete_with_recovery(cell)
                for pending_cell, _ in pending:
                    complete_with_recovery(pending_cell)
                pending.clear()
                completed_cells += 1
                break
            local_field = render_cell(cell, divisions)
            if local_field.shape != (y1 - y0, x1 - x0):
                raise RuntimeError("native local reference returned an invalid tile shape")
            if np.isfinite(local_field).all():
                field[y0:y1, x0:x1] = local_field
                completed_cells += 1
                continue

            if divisions < ATLAS_LOCAL_REFERENCE_MAX_DIVISIONS:
                if time.monotonic() >= tile_deadline:
                    complete_with_recovery(cell, local_field)
                    completed_cells += 1
                    for pending_cell, _ in pending:
                        complete_with_recovery(pending_cell)
                    pending.clear()
                    break
                children = child_cells(cell, divisions)
                if len(children) != 4:
                    raise RuntimeError("local reference subdivision did not produce four children")
                pending.extend((child, divisions * 2) for child in children)
                refined_cells += 1
                if refined_cells <= 4 or refined_cells % 16 == 0:
                    print(
                        f"  Refining {x1 - x0}x{y1 - y0} deep cell "
                        f"at {divisions}x{divisions} grid ({len(pending)} pending).",
                        flush=True,
                    )
                continue

            # A cell this small no longer benefits from more references. Run
            # one bounded exact retry with BLA disabled. Truly pathological
            # pixels are recovered from the parent field instead of blocking
            # the atlas with a flat synthetic value.
            if time.monotonic() >= tile_deadline:
                complete_with_recovery(cell, local_field)
                completed_cells += 1
                for pending_cell, _ in pending:
                    complete_with_recovery(pending_cell)
                pending.clear()
                break
            final_retries += 1
            if final_retries <= 4 or final_retries % 16 == 0:
                print(
                    f"  Final exact retry for {x1 - x0}x{y1 - y0} deep cell.",
                    flush=True,
                )
            local_field = render_cell(cell, divisions, final_retry=True)
            complete_with_recovery(cell, local_field)
            completed_cells += 1
    if not np.isfinite(field).all():
        raise RuntimeError("adaptive local reference render left unresolved pixels")
    if refined_cells or recovered_cells:
        print(
            f"  Adaptive local references completed {completed_cells} cells "
            f"after refining {refined_cells} hard cells"
            + (
                f"; spatially recovered {recovered_pixels} unresolved pixels"
                if recovered_cells else ""
            )
            + ".",
            flush=True,
        )
    return field


def _atlas_tile_field(
    *,
    cache_dir: Optional[Path],
    cache_identity: str,
    render_width: int,
    render_height: int,
    level: int,
    log_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    series_order: int,
    series_block: int,
    renderer: str,
    native_reference: Any,
    native_threads: int,
    native_library: Any = None,
    native_backend: int = 0,
    fallback_field: Optional[Any] = None,
    fallback_zoom_factor: float = 2.0,
    fallback_max_iter: Optional[int] = None,
    allow_recovery: bool = False,
    durable_cache: bool = False,
    cache_evictor: Optional["_CacheEvictor"] = None,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
    native_reference_root: Any = None,
) -> Any:
    """Load or render one reusable tile in the fixed zoom atlas."""

    np = _require_numpy()
    cache_path = _atlas_tile_path(
        cache_dir,
        cache_identity,
        render_width,
        render_height,
        level,
        log_zoom,
        x_center,
        y_center,
        max_iter,
        series_order,
        series_block,
    )
    if cache_path is not None:
        try:
            with _safe_cache_load(cache_path) as cached:
                if cached is None:
                    raise ValueError("unsafe cached atlas tile")
                if _valid_field_array(cached, (render_height, render_width)):
                    if cache_evictor is not None:
                        cache_evictor.touch(cache_path)
                    print(f"Using cached atlas tile {level} ({cache_path.name}).", flush=True)
                    # Do not retain an mmap while the LRU evictor is allowed
                    # to remove neighbouring entries. A private copy keeps
                    # cache hits portable on Windows, where deleting an open
                    # mapping fails and can otherwise make pruning nondeterministic.
                    return np.array(cached, dtype=np.float32, copy=True)
        except (
            EOFError,
            KeyError,
            MemoryError,
            OSError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ):
            pass

    print(
        f"Rendering atlas tile {level} at zoom {_zoom_label(log_zoom)}, "
        f"iterations {max_iter}",
        flush=True,
    )
    tile_started = time.monotonic()
    watchdog_stop = threading.Event()

    def tile_watchdog() -> None:
        while not watchdog_stop.wait(ATLAS_PROGRESS_INTERVAL_SECONDS):
            elapsed = time.monotonic() - tile_started
            print(
                f"  Still rendering atlas tile {level} "
                f"({_zoom_label(log_zoom)}); {elapsed:.0f}s elapsed "
                "(native deep reference work is active).",
                flush=True,
            )

    watchdog = threading.Thread(
        target=tile_watchdog,
        name=f"atlas-tile-watchdog-{level}",
        daemon=True,
    )
    watchdog.start()
    try:
        field = None
        # The adaptive local-reference machinery is a Mandelbrot BLA path.
        # Calling it for an alternate formula silently rendered its cells with
        # the default Mandelbrot recurrence (and, in draft mode, could then
        # copy those wrong cells into a square fallback).  Alternate formulas
        # must go through their own direct/high-precision renderer instead.
        if native_library is not None and formula == "mandelbrot":
            try:
                field = _atlas_local_reference_field(
                    render_width=render_width,
                    render_height=render_height,
                    log10_zoom=log_zoom,
                    x_center=x_center,
                    y_center=y_center,
                    max_iter=max_iter,
                    series_order=series_order,
                    series_block=series_block,
                    renderer=renderer,
                    native_threads=native_threads,
                    native_library=native_library,
                    native_backend=native_backend,
                    native_reference=native_reference,
                    native_reference_root=native_reference_root,
                    fallback_field=fallback_field,
                    fallback_zoom_factor=fallback_zoom_factor,
                    fallback_max_iter=fallback_max_iter,
                    allow_recovery=allow_recovery,
                )
            except RuntimeError as error:
                if not allow_recovery:
                    raise
                # Secondary references are an optimization layer.  A transient
                # MPFR/native allocation failure must not make an otherwise
                # renderable tile fail; retry through the already prepared global
                # reference instead.
                print(
                    f"  Local references unavailable ({error}); "
                    "using the global reference.",
                    flush=True,
                )
        if field is None:
            field = render_fractal(
                render_width,
                render_height,
                log_zoom,
                x_center,
                y_center,
                max_iter,
                renderer,
                native_threads,
                native_reference,
                series_order,
                series_block,
                NativeRenderOptions(backend=native_backend),
                formula,
                julia_constant,
            )
            if (
                not _valid_field_array(field, (render_height, render_width))
                and native_library is not None
                and formula == "mandelbrot"
                and renderer in {"auto", "native"}
                and native_reference is not None
                and log_zoom >= 12.0
            ):
                # The ordinary e12--e20 path is intentionally a fast direct
                # reference render. A rare BLA/MPFR tail can still leave NaN
                # samples there, though; retry only that failed tile through
                # the strict glitch repair queue instead of rejecting the
                # complete export or fabricating a rectangular fill.
                print(
                    "  Native atlas tile was incomplete; retrying with "
                    "strict glitch repair.",
                    flush=True,
                )
                field = _atlas_glitch_reference_field(
                    render_width=render_width,
                    render_height=render_height,
                    log10_zoom=log_zoom,
                    x_center=x_center,
                    y_center=y_center,
                    max_iter=max_iter,
                    series_order=series_order,
                    series_block=series_block,
                    native_threads=native_threads,
                    native_library=native_library,
                    native_backend=native_backend,
                    native_reference=native_reference,
                    native_reference_root=native_reference_root,
                    fallback_field=fallback_field,
                    fallback_zoom_factor=fallback_zoom_factor,
                    fallback_max_iter=fallback_max_iter,
                    allow_recovery=allow_recovery,
                )
    finally:
        watchdog_stop.set()
        watchdog.join(timeout=1.0)
    print(
        f"  Completed atlas tile {level} in "
        f"{time.monotonic() - tile_started:.2f}s.",
        flush=True,
    )
    field = _validated_field(field, (render_height, render_width), "atlas tile")
    if cache_path is not None:
        _atomic_save_field(cache_path, field, durable=durable_cache)
        if cache_evictor is not None:
            cache_evictor.observe(cache_path)
        try:
            with _safe_cache_load(cache_path) as cached:
                if cached is None:
                    raise ValueError("unsafe cached atlas tile")
                if _valid_field_array(cached, (render_height, render_width)):
                    return np.array(cached, dtype=np.float32, copy=True)
        except (OSError, EOFError, ValueError, TypeError):
            pass
    return field


def _atlas_colourise_native(
    parent: Any,
    child: Any,
    output_width: int,
    output_height: int,
    parent_zoom: float,
    child_fraction: float,
    parent_iter: int,
    child_iter: Optional[int],
    phase: float,
    vocal: float,
    instrumental: float,
    native_threads: int,
    library: Any,
    pitch: float,
    interior_color: Optional[tuple[int, int, int]] = None,
    accents: Optional[tuple[tuple[int, int, int], ...]] = None,
) -> Any:
    """Fused parent/child atlas sampling and pitch-aware colourisation."""

    np = _require_numpy()
    parent = np.ascontiguousarray(parent, dtype=np.float32)
    parent_height, parent_width = parent.shape
    output = np.empty((output_height, output_width, 3), dtype=np.uint8)
    if child is None or child_iter is None:
        child_pointer = ctypes.POINTER(ctypes.c_float)()
        child_height = child_width = child_max_iter = 0
    else:
        child = np.ascontiguousarray(child, dtype=np.float32)
        child_height, child_width = child.shape
        child_max_iter = int(child_iter)
        child_pointer = child.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    palette_max_iter = max(int(parent_iter), int(child_iter or parent_iter))
    parent_pointer = parent.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    accent_function = getattr(library, "fractal_atlas_colourise_accents", None)
    if accents is not None and accent_function is not None:
        if len(accents) != 3 or any(len(colour) != 3 for colour in accents):
            raise ValueError("Aurora accents must contain three RGB colours")
        accent_values = (ctypes.c_uint8 * 9)(
            *(int(channel) for colour in accents for channel in colour)
        )
        selected_interior = interior_color or (0, 0, 0)
        if len(selected_interior) != 3 or not all(
            0 <= int(value) <= 255 for value in selected_interior
        ):
            raise ValueError("ordinary interior colour must contain three bytes")
        status = accent_function(
            parent_pointer,
            parent_width,
            parent_height,
            int(parent_iter),
            child_pointer,
            child_width,
            child_height,
            int(child_max_iter),
            output_pointer,
            output_width,
            output_height,
            float(parent_zoom),
            float(child_fraction),
            int(palette_max_iter),
            float(phase),
            float(vocal),
            float(instrumental),
            float(pitch),
            accent_values,
            *(int(value) for value in selected_interior),
            native_threads,
        )
        if status != 0:
            message = library.fractal_last_error() or b"unknown native accent atlas colourizer error"
            raise RuntimeError(message.decode("utf-8", errors="replace"))
        return output
    interior_function = getattr(library, "fractal_atlas_colourise_interior", None)
    base_pitch = 0.5 if accents is not None else pitch
    if interior_color is not None and interior_function is not None:
        if len(interior_color) != 3 or not all(0 <= int(value) <= 255 for value in interior_color):
            raise ValueError("ordinary interior colour must contain three bytes")
        status = interior_function(
            parent_pointer,
            parent_width,
            parent_height,
            int(parent_iter),
            child_pointer,
            child_width,
            child_height,
            child_max_iter,
            output_pointer,
            output_width,
            output_height,
            float(parent_zoom),
            float(child_fraction),
            palette_max_iter,
            float(phase),
            float(vocal),
            float(instrumental),
            float(base_pitch),
            *(int(value) for value in interior_color),
            native_threads,
        )
    else:
        status = library.fractal_atlas_colourise(
            parent_pointer,
            parent_width,
            parent_height,
            int(parent_iter),
            child_pointer,
            child_width,
            child_height,
            child_max_iter,
            output_pointer,
            output_width,
            output_height,
            float(parent_zoom),
            float(child_fraction),
            palette_max_iter,
            float(phase),
            float(vocal),
            float(instrumental),
            float(base_pitch),
            native_threads,
        )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native atlas colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    if accents is not None:
        interior_mask = None
        if interior_color is not None:
            interior_mask = np.all(
                output == np.asarray(interior_color, dtype=np.uint8), axis=-1
            )
        accented = _native_apply_aurora_accents(
            output,
            accents,
            pitch,
            library,
            native_threads,
        )
        if accented is None:
            return output
        if interior_mask is not None:
            accented[interior_mask] = np.asarray(interior_color, dtype=np.uint8)
        return accented
    if interior_color is not None and interior_function is None:
        # Compatibility with an older native .so. Reconstruct only the
        # boolean ownership mask in Python; the expensive crop and Aurora
        # colour pass above still stays native.
        _, parent_inside = _crop_and_resize_preserving_interior(
            parent,
            output_width,
            output_height,
            max(float(parent_zoom), 1.0),
            parent_iter,
            "bilinear",
        )
        inside = np.asarray(parent_inside, dtype=bool)
        if child is not None and child_iter is not None and child_fraction > 0.0:
            fraction = min(float(child_fraction), 1.0)
            child_view_width = (
                output_width if fraction >= 0.999999
                else max(1, int(round(output_width * fraction)))
            )
            child_view_height = (
                output_height if fraction >= 0.999999
                else max(1, int(round(output_height * fraction)))
            )
            _, child_inside_mask = _crop_and_resize_preserving_interior(
                child,
                child_view_width,
                child_view_height,
                1.0,
                child_iter,
                "bilinear",
            )
            if fraction >= 0.999999:
                inside = np.asarray(child_inside_mask, dtype=bool)
            else:
                left = (output_width - child_view_width) // 2
                top = (output_height - child_view_height) // 2
                inside = inside.copy()
                inside[top:top + child_view_height, left:left + child_view_width] = (
                    np.asarray(child_inside_mask, dtype=bool)
                )
        output[inside] = np.asarray(interior_color, dtype=np.uint8)
    return output


def _atlas_feather(width: int, height: int) -> int:
    """Return the shared scalar/RGB atlas seam width."""

    # KFP's finite-difference/slope pass is intentionally glossy and very
    # sensitive to a one-pixel change in source density. Give the nested tile
    # a real transition band; a 16px strip is effectively a hard rectangle at
    # 1080p. The cap keeps this from washing out small live/preview frames.
    return min(48, max(0, int(width) // 8), max(0, int(height) // 8))


def _atlas_colour_frame(
    parent: Any,
    child: Any,
    output_width: int,
    output_height: int,
    parent_zoom: float,
    child_fraction: float,
    parent_iter: int,
    child_iter: Optional[int],
    phase: float,
    vocal: float,
    instrumental: float,
    native_library: Any,
    native_threads: int,
    resample: str,
    palette_name: str,
    pitch: float,
    palette_file: Optional[Path] = None,
    child_zoom: Optional[float] = None,
) -> Any:
    """Compose a frame from a parent tile and its central child tile.

    ``child_fraction`` is the fraction of the output frame covered by the
    child. At the beginning of an atlas interval it is roughly 1/factor; at
    the end it reaches one. The child is therefore rendered only into its
    visible central rectangle instead of being expanded into a full temporary
    frame. Overscanned tiles can be slightly wider than the frame when the
    child takes over; ``child_zoom`` preserves that crop in the full-child
    path.
    """

    np = _require_numpy()
    if child_zoom is None:
        child_crop_zoom = 1.0
    else:
        try:
            child_crop_zoom = float(child_zoom)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("atlas child zoom must be numeric") from error
        if not math.isfinite(child_crop_zoom) or child_crop_zoom <= 0.0:
            raise ValueError("atlas child zoom must be finite and positive")
        child_crop_zoom = max(child_crop_zoom, 1.0)
    if (
        child is not None
        and child_iter is not None
        and float(child_fraction) > 0.0
        and min(int(output_width), int(output_height)) * float(child_fraction)
            < ATLAS_MIN_CHILD_PIXELS
    ):
        # Let the already valid parent approximation carry the first few
        # pixels of an interval. The child becomes useful once its KFP
        # neighbourhood has room to exist in the output image.
        child = None
        child_iter = None
        child_fraction = 0.0
    kfp_profile = _kfp_profile_for_selection(palette_name, palette_file)
    if (
        child is not None
        and child_iter is not None
        and float(child_fraction) >= ATLAS_NEAR_FULL_CHILD_FRACTION
        and float(child_fraction) < 0.999999
    ):
        # A reverse zoom can select the parent immediately after the child
        # reached the edge of its atlas interval. A narrow parent perimeter is
        # harmless for scalar Aurora colour, but KFP's neighbour stencil makes
        # the two independently coloured strips look like a hard rectangle.
        # The child is already a nearly complete viewport; promote it for this
        # short interval and avoid exposing the seam altogether.
        return _atlas_colour_frame(
            child,
            None,
            output_width,
            output_height,
            child_crop_zoom,
            0.0,
            int(child_iter),
            None,
            phase,
            vocal,
            instrumental,
            native_library,
            native_threads,
            resample,
            palette_name,
            pitch,
            palette_file,
        )
    if (
        kfp_profile is not None
        and child is not None
        and child_iter is not None
        and float(child_fraction) >= 0.999999
        and native_library is not None
        and hasattr(native_library, "fractal_crop_colourise_kfp")
        and resample == "bilinear"
    ):
        # At the end of an atlas interval the child is the complete viewport.
        # Going through the atlas seam code here would still feather it against
        # the old parent at the four output edges, causing a one-frame flash
        # when the next interval promotes that same child to its parent.
        return _crop_colourise_kfp_native(
            child,
            output_width,
            output_height,
            child_crop_zoom,
            int(child_iter),
            phase,
            vocal,
            instrumental,
            pitch,
            kfp_profile,
            native_library,
            native_threads,
        )
    if (
        child is not None
        and child_iter is not None
        and float(child_fraction) >= 0.999999
        and kfp_profile is None
        and native_library is not None
        and hasattr(native_library, "fractal_crop_colourise")
        and resample == "bilinear"
    ):
        # Overscan makes the full child wider than the nominal viewport. Use
        # the native crop compositor here instead of treating the whole stored
        # tile as the frame; otherwise the zoom would jump by the overscan
        # factor exactly at an atlas boundary.
        interior_color = _ordinary_interior_color(palette_name, palette_file)
        accents = (
            None
            if palette_name == "aurora" and palette_file is None
            else _aurora_accents_for_selection(palette_name, palette_file)
        )
        return _crop_and_colourise_native(
            child,
            output_width,
            output_height,
            child_crop_zoom,
            int(child_iter),
            phase,
            vocal,
            instrumental,
            native_threads,
            native_library,
            pitch,
            interior_color if interior_color != (0, 0, 0) else None,
            accents,
        )
    if (
        kfp_profile is not None
        and native_library is not None
        and hasattr(native_library, "fractal_atlas_colourise_kfp_raw")
        and resample == "bilinear"
    ):
        return _atlas_colourise_kfp_raw_native(
            parent,
            child,
            output_width,
            output_height,
            parent_zoom,
            child_fraction,
            parent_iter,
            child_iter,
            phase,
            vocal,
            instrumental,
            pitch,
            kfp_profile,
            native_library,
            native_threads,
        )
    if (
        native_library is not None
        and hasattr(native_library, "fractal_atlas_colourise")
        and resample == "bilinear"
    ):
        if palette_name == "aurora" and palette_file is None:
            return _atlas_colourise_native(
                parent,
                child,
                output_width,
                output_height,
                parent_zoom,
                child_fraction,
                parent_iter,
                child_iter,
                phase,
                vocal,
                instrumental,
                native_threads,
                native_library,
                pitch,
            )
        if _kfp_profile_for_selection(palette_name, palette_file) is None:
            interior_color = _ordinary_interior_color(palette_name, palette_file)
            return _atlas_colourise_native(
                parent,
                child,
                output_width,
                output_height,
                parent_zoom,
                child_fraction,
                parent_iter,
                child_iter,
                phase,
                vocal,
                instrumental,
                native_threads,
                native_library,
                pitch,
                interior_color if interior_color != (0, 0, 0) else None,
                _aurora_accents_for_selection(palette_name, palette_file),
            )
    parent_view, parent_inside_full = _crop_and_resize_preserving_interior(
        parent,
        output_width,
        output_height,
        max(float(parent_zoom), 1.0),
        parent_iter,
        resample,
    )
    parent_view = np.asarray(parent_view, dtype=np.float32)
    if child is None or child_iter is None or child_fraction <= 0.0:
        return _colourise_view(
            parent_view,
            parent_iter,
            phase,
            vocal,
            instrumental,
            native_library,
            native_threads,
            palette_name,
            pitch,
            palette_file,
        )
    child_fraction = min(float(child_fraction), 1.0)
    if child_fraction >= 0.999999:
        child_view, _ = _crop_and_resize_preserving_interior(
            child,
            output_width,
            output_height,
            child_crop_zoom,
            child_iter,
            resample,
        )
        child_view = np.asarray(child_view, dtype=np.float32)
        return _colourise_view(
            child_view,
            child_iter,
            phase,
            vocal,
            instrumental,
            native_library,
            native_threads,
            palette_name,
            pitch,
            palette_file,
        )

    child_width = max(1, int(round(output_width * child_fraction)))
    child_height = max(1, int(round(output_height * child_fraction)))
    child_view, child_inside = _crop_and_resize_preserving_interior(
        child,
        child_width,
        child_height,
        1.0,
        child_iter,
        resample,
    )
    child_view = np.asarray(child_view, dtype=np.float32)
    effective_iter = max(int(parent_iter), int(child_iter))
    # Parent and child tiles have different iteration caps. Once the child is
    # composited, colourise the complete frame against one cap; otherwise the
    # old parent sentinel becomes an escaped value outside the child rectangle
    # and draws a crisp square/black fill. Normalize every interior sample,
    # including parent pixels outside the visible child, before that pass.
    parent_inside_region = parent_inside_full
    parent_view[parent_inside_full] = float(effective_iter)
    child_view[child_inside] = float(effective_iter)
    left = (output_width - child_width) // 2
    top = (output_height - child_height) // 2
    right = left + child_width
    bottom = top + child_height
    if kfp_profile is not None:
        visible_feather = _atlas_feather(child_width, child_height)
        halo = min(
            2,
            left,
            output_width - left - child_width,
            top,
            output_height - top - child_height,
        )
        if halo > 0:
            # Kalles' difference/slope stencil needs neighbours outside the
            # moving child rectangle. Edge padding is the exact prepared-tile
            # equivalent of the native raw path's clamped bilinear halo.
            child_view = np.pad(
                child_view,
                ((halo, halo), (halo, halo)),
                mode="edge",
            )
            child_inside = np.pad(
                np.asarray(child_inside, dtype=bool),
                ((halo, halo), (halo, halo)),
                mode="edge",
            )
            child_width += 2 * halo
            child_height += 2 * halo
            left -= halo
            top -= halo
            right = left + child_width
            bottom = top + child_height
        feather = min(
            child_width,
            child_height,
            max(2, visible_feather + halo),
        )
        if (
            native_library is not None
            and hasattr(native_library, "fractal_atlas_colourise_kfp")
            and resample == "bilinear"
        ):
            return _atlas_colourise_kfp_native(
                parent_view,
                child_view,
                output_width,
                output_height,
                effective_iter,
                left,
                top,
                feather,
                phase,
                vocal,
                instrumental,
                pitch,
                kfp_profile,
                native_library,
                native_threads,
            )
        # KFP slope shading is derived from spatial gradients. Computing that
        # gradient after splicing a child into the parent manufactures a
        # rectangle at the tile boundary, even when both scalar fields are
        # mathematically aligned. Colourise each tile in its own coordinate
        # system, then feather RGB values at the seam instead.
        parent_rgb = _colourise_kfp(
            parent_view,
            effective_iter,
            phase,
            vocal,
            instrumental,
            pitch,
            kfp_profile,
            spatial_width=output_width,
        )
        child_rgb = _colourise_kfp(
            child_view,
            effective_iter,
            phase,
            vocal,
            instrumental,
            pitch,
            kfp_profile,
            spatial_width=output_width,
            dither_x=left,
            dither_y=top,
        )
        region_rgb = parent_rgb[top:bottom, left:right]
        parent_inside = parent_inside_region[top:bottom, left:right]
        if feather < 2:
            small_child = np.where(
                child_inside[..., None],
                child_rgb,
                np.where(parent_inside[..., None], child_rgb, region_rgb),
            )
            region_rgb[...] = small_child
            return parent_rgb
        yy, xx = np.ogrid[:child_height, :child_width]
        edge_distance = np.minimum(
            np.minimum(xx, yy),
            np.minimum(child_width - 1 - xx, child_height - 1 - yy),
        )
        alpha = np.clip(
            edge_distance.astype(np.float32) / float(feather), 0.0, 1.0
        )
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        blended_rgb = (
            region_rgb.astype(np.float32) * (1.0 - alpha[..., None])
            + child_rgb.astype(np.float32) * alpha[..., None]
        )
        # Even an interior sample participates in the feather band. Selecting
        # either classification unconditionally is what turns a valid black
        # interior into the four straight edges of a child tile. The deeper
        # child is still authoritative once alpha reaches one; the band itself
        # is intentionally a smooth RGB transition.
        region_rgb[...] = np.asarray(
            np.clip(np.rint(blended_rgb), 0.0, 255.0),
            dtype=np.uint8,
        )
        return parent_rgb
    feather = _atlas_feather(child_width, child_height)
    if feather < 2:
        region = parent_view[top:bottom, left:right]
        region[...] = child_view
        return _colourise_view(
            parent_view,
            effective_iter,
            phase,
            vocal,
            instrumental,
            native_library,
            native_threads,
            palette_name,
            pitch,
            palette_file,
        )

    yy, xx = np.ogrid[:child_height, :child_width]
    edge_distance = np.minimum(
        np.minimum(xx, yy),
        np.minimum(child_width - 1 - xx, child_height - 1 - yy),
    )
    alpha = np.clip(
        edge_distance.astype(np.float32) / float(feather), 0.0, 1.0
    )
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    # Colourize the two source densities independently. KFP needs this for its
    # spatial stencil, and doing the same in the portable ordinary path keeps
    # interior ownership consistent when native colourisation is unavailable.
    # Crucially, the feather band blends RGB even when one source is interior;
    # a hard scalar/classification choice would expose a rectangular tile.
    parent_rgb = _colourise_view(
        parent_view,
        effective_iter,
        phase,
        vocal,
        instrumental,
        native_library,
        native_threads,
        palette_name,
        pitch,
        palette_file,
    )
    child_rgb = _colourise_view(
        child_view,
        effective_iter,
        phase,
        vocal,
        instrumental,
        native_library,
        native_threads,
        palette_name,
        pitch,
        palette_file,
    )
    region_rgb = parent_rgb[top:bottom, left:right]
    blended = (
        region_rgb.astype(np.float32) * (1.0 - alpha[..., None])
        + child_rgb.astype(np.float32) * alpha[..., None]
    )
    region_rgb[...] = np.asarray(np.clip(np.rint(blended), 0.0, 255.0), dtype=np.uint8)
    return parent_rgb


def _valid_field_array(field: Any, shape: tuple[int, int]) -> bool:
    """Return whether a cached/rendered scalar field is safe to consume."""

    np = _require_numpy()
    try:
        array = np.asarray(field)
    except (TypeError, ValueError):
        return False
    if array.shape != shape or array.ndim != 2:
        return False
    if not np.issubdtype(array.dtype, np.floating):
        return False
    return bool(np.isfinite(array).all())


def _validated_field(field: Any, shape: tuple[int, int], label: str) -> Any:
    np = _require_numpy()
    raw = np.asarray(field)
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise RuntimeError(f"{label} must contain real numeric samples")
    converted = np.asarray(raw, dtype=np.float32)
    if not _valid_field_array(converted, shape):
        raise RuntimeError(f"{label} is not a finite {shape[1]}x{shape[0]} scalar field")
    return np.ascontiguousarray(converted, dtype=np.float32)


def _render_video_atlas(
    *,
    command: list[str],
    temporary_output: Path,
    output_path: Path,
    features: AudioFeatures,
    width: int,
    height: int,
    fps: int,
    keyframe_factor: float,
    x_center: str,
    y_center: str,
    zooms: Any,
    iteration_base: int,
    iterations_per_decade: int,
    iteration_cap: int,
    active_renderer: str,
    native_library: Any,
    native_references: list[tuple[float, Any]],
    native_threads: int,
    native_backend: int,
    series_order: int,
    series_block: int,
    allow_recovery: bool,
    render_width: int,
    render_height: int,
    resample: str,
    palette: str,
    cache_dir: Optional[Path],
    cache_limit_mb: float,
    durable_cache: bool,
    cache_identity: str,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
    palette_file: Optional[Path] = None,
    glow: float = 0.0,
    motion_blur: float = 0.0,
    hardware_encoder: bool = False,
) -> dict[str, Any]:
    """Render a song through a fixed nested tile atlas.

    The old renderer tied source fields to audio-dependent chunk boundaries.
    This path ties them only to absolute logarithmic zoom, so a tile is
    rendered once and can be reused by every camera path through the same
    centre. A bounded three-tile window is retained in memory so both forward
    and audio-driven reverse zooms remain cheap. Tiles include a small fixed
    overscan margin so reverse zooms never expose a low-resolution perimeter
    around a nearly complete child.
    """

    np = _require_numpy()
    # The atlas field already has the requested quality density. Colourising
    # at the final output size would repeat the crop and palette work for
    # undersampled custom renders. Keep the render surface at source
    # resolution and upscale once in FFmpeg when a custom scale requests it.
    frame_width = int(render_width)
    frame_height = int(render_height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("atlas frame dimensions must be positive")
    origin, step, level_count = _atlas_geometry(zooms, keyframe_factor)
    factor = 10.0 ** step
    tile_overscan_log = math.log10(ATLAS_TILE_OVERSCAN_FACTOR)
    maximum_log = float(max(float(value) for value in zooms))
    total_frames = features.frame_count
    print(
        f"Atlas source: {render_width}x{render_height}; levels: {level_count + 1}; "
        f"log step: {step:.6f}; origin: {_zoom_label(origin)}",
        flush=True,
    )

    tile_cache: dict[int, Any] = {}
    tile_iterations: dict[int, int] = {}
    cache_evictor = _CacheEvictor(cache_dir, cache_limit_mb)
    tile_seconds = 0.0
    tile_futures: dict[int, Future[Any]] = {}
    future_started: dict[int, float] = {}
    native_reference_root = None
    if (
        native_library is not None
        and len(native_references) > 1
        and hasattr(native_library, "fractal_clone_reference")
    ):
        # The first entry is created with the reusable builder orbit whenever
        # clone tiers are available. Keep the root alive for rare per-tile
        # retargets; the normal decade tiers continue to serve the hot path.
        native_reference_root = native_references[0][1]
    # Deep native tiles already saturate a CPU-only export. Running a
    # speculative tile beside x264 (and beside the next foreground tile when
    # the queue catches up) creates a second OpenMP team and can nearly double
    # wall time on small machines. A real hardware encoder leaves those CPU
    # cores available, so permit one bounded look-ahead worker there and cap
    # its native team below the foreground team.
    deep_native_export = native_library is not None and maximum_log >= 12.0
    deep_hardware_prefetch = deep_native_export and hardware_encoder
    prefetch_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="fractal-field")
        if cache_limit_mb <= 0.0 and (not deep_native_export or deep_hardware_prefetch)
        else None
    )
    prefetch_native_threads = (
        max(1, min(native_threads, max(1, (os.cpu_count() or 2) // 2)))
        if deep_hardware_prefetch
        else native_threads
    )

    def tile_display_log(level: int) -> float:
        """Nominal camera boundary represented by an atlas level."""

        return origin + step * float(level)

    def tile_log(level: int) -> float:
        """Effective field zoom, widened by the fixed atlas overscan."""

        return tile_display_log(level) - tile_overscan_log

    def tile_iter(level: int) -> int:
        cached = tile_iterations.get(level)
        if cached is not None:
            return cached
        # Iteration count follows the nominal tile interval, not the slightly
        # wider stored field. This remains conservative for every pixel in
        # the central viewport while avoiding a depth change in the public
        # zoom/quality contract.
        current_log = tile_display_log(level)
        end_log = min(maximum_log, current_log + step)
        value = max_iterations(
            end_log,
            iteration_base,
            iterations_per_decade,
            iteration_cap,
        )
        tile_iterations[level] = value
        return value

    def render_tile(
        level: int,
        parent_field: Any = None,
        parent_max_iter: Optional[int] = None,
        tile_native_threads: Optional[int] = None,
    ) -> Any:
        if level < 0 or level > level_count:
            return None
        if parent_field is None:
            parent_field = tile_cache.get(level - 1)
            parent_max_iter = (
                tile_iterations.get(level - 1)
                if parent_field is not None
                else None
            )
        field = _atlas_tile_field(
            cache_dir=cache_dir,
            cache_identity=cache_identity,
            render_width=render_width,
            render_height=render_height,
            level=level,
            log_zoom=tile_log(level),
            x_center=x_center,
            y_center=y_center,
            max_iter=tile_iter(level),
            series_order=series_order,
            series_block=series_block,
            renderer=active_renderer,
            # The tier ladder is already placed several complete atlas
            # intervals before each decade boundary. Waiting for one more
            # interval here needlessly sends the first tile after a tier
            # boundary through a shallow BLA table and can turn a sub-second
            # tile into a long exact-tail replay. Keep the optional margin in
            # the helper for callers that need it, but the production atlas
            # uses the validated tier as soon as its radius starts.
            native_reference=_select_native_reference(
                native_references,
                tile_log(level),
            ),
            native_threads=(
                native_threads
                if tile_native_threads is None
                else tile_native_threads
            ),
            native_library=native_library,
            native_backend=native_backend,
            native_reference_root=native_reference_root,
            fallback_field=parent_field,
            fallback_zoom_factor=factor,
            fallback_max_iter=parent_max_iter,
            allow_recovery=allow_recovery,
            durable_cache=durable_cache,
            cache_evictor=cache_evictor,
            formula=formula,
            julia_constant=julia_constant,
        )
        return field

    def get_tile(level: int) -> Any:
        nonlocal tile_seconds
        if level < 0 or level > level_count:
            return None
        if level in tile_cache:
            return tile_cache[level]
        future = tile_futures.pop(level, None)
        if future is not None:
            started_waiting = time.monotonic()
            while True:
                try:
                    field = future.result(timeout=ATLAS_PROGRESS_INTERVAL_SECONDS)
                    break
                except FutureTimeout:
                    waited = time.monotonic() - started_waiting
                    print(
                        f"  Waiting for prefetched atlas tile {level}; "
                        f"{waited:.0f}s elapsed.",
                        flush=True,
                    )
            started = future_started.pop(level, time.perf_counter())
            tile_seconds += time.perf_counter() - started
        else:
            started = time.perf_counter()
            field = render_tile(level)
            tile_seconds += time.perf_counter() - started
        tile_cache[level] = field
        protected_paths: set[Path] = set()
        if active_level is not None and cache_dir is not None:
            for protected_level in range(active_level - 1, active_level + 2):
                if 0 <= protected_level <= level_count:
                    protected_path = _atlas_tile_path(
                        cache_dir,
                        cache_identity,
                        render_width,
                        render_height,
                        protected_level,
                        tile_log(protected_level),
                        x_center,
                        y_center,
                        tile_iter(protected_level),
                        series_order,
                        series_block,
                    )
                    if protected_path is not None:
                        protected_paths.add(protected_path)
        _prune_cache(cache_dir, cache_limit_mb, cache_evictor, protected_paths)
        return tile_cache[level]

    def prefetch_tile(level: int) -> None:
        if prefetch_executor is None or level < 0 or level > level_count:
            return
        if level in tile_cache or level in tile_futures:
            return
        parent_field = tile_cache.get(level - 1)
        if parent_field is None:
            return
        future_started[level] = time.perf_counter()
        tile_futures[level] = prefetch_executor.submit(
            render_tile,
            level,
            parent_field,
            tile_iterations.get(level - 1),
            prefetch_native_threads,
        )

    process = None
    ffmpeg_diagnostics: Optional[deque[str]] = None
    ffmpeg_reader: Optional[threading.Thread] = None
    frame_writer: Optional[_FFmpegFrameWriter] = None
    frame_seconds = 0.0
    encoder_seconds = 0.0
    previous_rgb = None
    render_started = time.perf_counter()
    last_progress_report = render_started
    active_level = None
    try:
        process, ffmpeg_diagnostics, ffmpeg_reader = _start_ffmpeg_process(command)
        frame_writer = _FFmpegFrameWriter(
            process,
            ffmpeg_diagnostics,
            ffmpeg_reader,
        )

        def enqueue_frame(frame: Any) -> None:
            assert frame_writer is not None
            frame_writer.write(frame)

        for frame_index in range(total_frames):
            frame_started = time.perf_counter()
            frame_log_zoom = float(zooms[frame_index])
            level = _atlas_level_for_zoom(frame_log_zoom, origin, step, level_count)
            if level != active_level:
                active_level = level
                print(
                    f"Entering atlas level {level}/{level_count} at "
                    f"frame {frame_index}/{total_frames}.",
                    flush=True,
                )
                # The child is needed for the central high-resolution region
                # throughout the parent's interval, so prepare it before the
                # first frame that can display it.
                get_tile(level)
                if level < level_count:
                    get_tile(level + 1)
                prefetch_tile(level + 2)
                _trim_atlas_memory_cache(tile_cache, level)

            parent_log = tile_log(level)
            parent = get_tile(level)
            child = get_tile(level + 1) if level < level_count else None
            parent_zoom = max(1.0, 10.0 ** (frame_log_zoom - parent_log))
            child_fraction = min(1.0, parent_zoom / factor) if child is not None else 0.0
            parent_max_iter = tile_iter(level)
            child_max_iter = tile_iter(level + 1) if child is not None else None
            # With fixed tile overscan, a full child can still be wider than
            # the nominal camera viewport. Preserve that crop when it takes
            # over the frame instead of showing the whole stored tile.
            child_zoom = max(1.0, parent_zoom / factor) if child is not None else 1.0
            phase = float(features.phase[frame_index])
            gradient = float(features.gradient[frame_index])
            instrumental = float(features.instrumental[frame_index])
            pitch = float(features.pitch[frame_index])
            rgb = _atlas_colour_frame(
                parent,
                child,
                frame_width,
                frame_height,
                parent_zoom,
                child_fraction,
                parent_max_iter,
                child_max_iter,
                phase,
                gradient,
                instrumental,
                native_library,
                native_threads,
                resample,
                palette,
                pitch,
                palette_file,
                child_zoom,
            )
            rgb = _apply_frame_effects(rgb, glow, motion_blur, previous_rgb)
            previous_rgb = rgb
            enqueue_frame(rgb)
            frame_seconds += time.perf_counter() - frame_started
            progress_now = time.perf_counter()
            if (
                frame_index == 0
                or frame_index + 1 == total_frames
                or progress_now - last_progress_report
                    >= RENDER_PROGRESS_INTERVAL_SECONDS
            ):
                print(
                    f"  frame {frame_index + 1}/{total_frames} "
                    f"({100.0 * (frame_index + 1) / total_frames:5.1f}%) "
                    f"· atlas level {level}/{level_count}",
                    flush=True,
                )
                last_progress_report = progress_now

        encoder_started = time.perf_counter()
        assert frame_writer is not None
        return_code = frame_writer.finish()
        encoder_seconds += time.perf_counter() - encoder_started
        if return_code != 0:
            raise RuntimeError(_ffmpeg_error_message(
                "ffmpeg exited with an error",
                ffmpeg_diagnostics,
                return_code,
                ffmpeg_reader,
            ))
        os.replace(temporary_output, output_path)
        elapsed = time.perf_counter() - render_started
        print(
            f"Atlas timing: tiles {tile_seconds:.2f}s, frame/reproject/queue "
            f"{frame_seconds:.2f}s, encoder drain {encoder_seconds:.2f}s, "
            f"total {elapsed:.2f}s.",
            flush=True,
        )
        return {
            "keyframe_seconds": float(tile_seconds),
            "frame_seconds": float(frame_seconds),
            "encoder_seconds": float(encoder_seconds),
            "total_seconds": float(elapsed),
        }
    except BrokenPipeError as exc:
        if process is not None:
            _terminate_subprocess(process)
        raise RuntimeError(
            _ffmpeg_error_message(
                "ffmpeg stopped while receiving video frames",
                ffmpeg_diagnostics,
                reader=ffmpeg_reader,
            )
        ) from exc
    except BaseException:
        if process is not None:
            _terminate_subprocess(process)
        raise
    finally:
        if frame_writer is not None:
            frame_writer.abort()
        if prefetch_executor is not None:
            # Native tile workers borrow the prepared reference handles. They
            # cannot be safely abandoned on an error: the outer cleanup may
            # destroy those handles while a speculative render is still in
            # MPFR/BLA code. Always join the one bounded worker before the
            # references are released; the native time budgets keep this
            # wait finite even for a pathological tile.
            prefetch_executor.shutdown(
                wait=True,
                cancel_futures=True,
            )
        if ffmpeg_reader is not None:
            ffmpeg_reader.join(timeout=2.0)
        if temporary_output.exists():
            try:
                temporary_output.unlink()
            except OSError:
                pass


def _ffmpeg_encoder_names() -> set[str]:
    """Return encoders advertised by the local FFmpeg, if queryable."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return set()
    known_encoders = {
        "h264_nvenc",
        "h264_qsv",
        "h264_vaapi",
        "h264_videotoolbox",
        "libx264",
    }
    process = None
    reader: Optional[threading.Thread] = None
    captured = bytearray()
    try:
        process = subprocess.Popen(
            [ffmpeg, "-nostdin", "-hide_banner", "-encoders"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            **_subprocess_group_options(),
        )
        stream = process.stdout

        def drain_stdout() -> None:
            if stream is None:
                return
            try:
                while True:
                    block = stream.read(64 * 1024)
                    if not block:
                        break
                    remaining = MAX_FFMPEG_ENCODER_QUERY_BYTES - len(captured)
                    if remaining > 0:
                        captured.extend(block[:remaining])
            except (OSError, ValueError):
                return
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        reader = threading.Thread(
            target=drain_stdout,
            name="fractal-ffmpeg-encoder-query",
            daemon=True,
        )
        try:
            reader.start()
        except BaseException:
            _terminate_subprocess(process)
            return set()
        try:
            process.wait(timeout=FFMPEG_ENCODER_QUERY_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _terminate_subprocess(process)
            return set()
    except (OSError, subprocess.SubprocessError):
        return set()
    finally:
        if reader is not None:
            reader.join(timeout=2.0)
        if process is not None and process.poll() is None:
            _terminate_subprocess(process)
    if process is None or process.returncode != 0:
        return set()

    names: set[str] = set()
    for line in bytes(captured).decode("utf-8", errors="replace").splitlines():
        parts = line.split()
        if (
            len(parts) >= 2
            and parts[0]
            and not parts[0].startswith("#")
            and parts[1] in known_encoders
        ):
            names.add(parts[1])
    return names


@lru_cache(maxsize=4)
def _vaapi_encoder_usable(
    device: str = "/dev/dri/renderD128",
    lossless: bool = False,
    width: int = 128,
    height: int = 128,
    fps: int = 30,
) -> bool:
    """Probe the complete upload/encode path, not just FFmpeg's name list."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not Path(device).exists():
        return False
    try:
        width, height = _validate_dimensions(width, height, "VAAPI probe")
        fps = _validate_fps(fps)
    except ValueError:
        return False
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-vaapi_device",
                device,
                "-f",
                "lavfi",
                "-i",
                f"color=black:s={width}x{height}:r={fps}:d=0.05,format=rgb24",
                "-r",
                str(fps),
                "-vf",
                "format=nv12,hwupload",
                "-c:v",
                "h264_vaapi",
                "-qp",
                "0" if lossless else "24",
                "-f",
                "null",
                "-",
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            **_subprocess_group_options(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@lru_cache(maxsize=16)
def _hardware_encoder_usable(
    encoder: str,
    lossless: bool = False,
    near_lossless: bool = False,
    width: int = 256,
    height: int = 256,
    fps: int = 30,
    preserve_chroma: Optional[bool] = None,
    *,
    video_preset: str = "ultrafast",
) -> bool:
    """Probe the encoder at the dimensions and quality the render will use."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        width, height = _validate_dimensions(width, height, "hardware encoder probe")
        fps = _validate_fps(fps)
        video_preset = _validate_ffmpeg_token(video_preset, "video preset")
        if video_preset not in VIDEO_PRESET_CHOICES:
            return False
    except ValueError:
        return False
    if preserve_chroma is None:
        # Preserve the old private-helper behaviour for callers that do not
        # know the selected palette yet. The render path passes an explicit
        # value so smooth near-lossless palettes can stay on yuv420p.
        preserve_chroma = lossless or near_lossless
    # 4:2:0 cannot represent odd frame dimensions. The render path selects
    # 4:4:4 for odd RGB output on the codecs that support it; make the NVENC
    # probe follow that same choice so auto-selection cannot pass one format
    # and fail when the first real frame arrives.
    if encoder == "h264_nvenc" and (width % 2 or height % 2):
        preserve_chroma = True
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        # Use the real input dimensions. A tiny probe can pass even when the
        # selected hardware rejects an 8K frame, causing the first real frame
        # to fail after expensive atlas preparation has already started.
        f"color=black:s={width}x{height}:r={fps}:d=0.05,format=rgb24",
        "-r",
        str(fps),
        "-frames:v",
        "1",
    ]
    try:
        _, rate_control, _ = _video_encoder_settings(
            encoder,
            video_preset,
            0 if lossless else (NEAR_LOSSLESS_CRF if near_lossless else 24),
            lossless=lossless,
        )
    except ValueError:
        return False
    output_pixel_format = (
        "yuv444p"
        if preserve_chroma and encoder == "h264_nvenc"
        else "yuv420p"
    )
    command.extend([
        "-c:v",
        encoder,
        *rate_control,
        "-pix_fmt",
        output_pixel_format,
        "-f",
        "null",
        "-",
    ])
    try:
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            **_subprocess_group_options(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@lru_cache(maxsize=16)
def _software_encoder_usable(
    encoder: str = "libx264",
    lossless: bool = False,
    near_lossless: bool = False,
    width: int = 256,
    height: int = 256,
    fps: int = 30,
    preserve_chroma: bool = False,
    *,
    video_preset: str = "ultrafast",
) -> bool:
    """Probe the software fallback at the same RGB frame shape as a render."""

    if encoder != "libx264":
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        width, height = _validate_dimensions(width, height, "software encoder probe")
        fps = _validate_fps(fps)
        video_preset = _validate_ffmpeg_token(video_preset, "video preset")
        if video_preset not in VIDEO_PRESET_CHOICES:
            return False
    except ValueError:
        return False
    # Keep this in lockstep with _video_pixel_format for software output. A
    # yuv420p probe would incorrectly bless odd dimensions that x264 rejects.
    output_pixel_format = (
        "yuv444p"
        if preserve_chroma or width % 2 or height % 2
        else "yuv420p"
    )
    try:
        preset, rate_control, _ = _video_encoder_settings(
            encoder,
            video_preset,
            0 if lossless else (NEAR_LOSSLESS_CRF if near_lossless else 24),
            lossless=lossless,
        )
    except ValueError:
        return False
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=black:s={width}x{height}:r={fps}:d=0.05,format=rgb24",
        "-r",
        str(fps),
        "-frames:v",
        "1",
        "-c:v",
        encoder,
        "-preset",
        preset,
        *rate_control,
        "-pix_fmt",
        output_pixel_format,
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            **_subprocess_group_options(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _video_encoder_settings(
    encoder: str,
    video_preset: str,
    crf: int,
    *,
    lossless: bool = False,
) -> tuple[str, list[str], bool]:
    """Return ``(preset, rate-control, needs-vaapi-upload)`` for one codec."""

    if lossless:
        crf = 0
    if encoder == "h264_nvenc":
        nvenc_presets = {
            "ultrafast": "p1",
            "superfast": "p2",
            "veryfast": "p3",
            "faster": "p4",
            "fast": "p5",
            "medium": "p6",
            "slow": "p7",
        }
        if lossless:
            # NVENC's lossless tune is materially different from CQ 0: CQ 0
            # still uses the VBR quantiser and can soften high-frequency KFP
            # edges.  constqp/qp 0 plus 4:4:4 input preserves the RGB frame
            # after the encoder's colour conversion.
            return nvenc_presets.get(video_preset, "p3"), [
                "-tune", "lossless", "-rc", "constqp", "-qp", "0",
            ], False
        return nvenc_presets.get(video_preset, "p3"), [
            "-cq", str(crf), "-rc", "vbr",
        ], False
    if encoder == "h264_vaapi":
        return "", ["-qp", str(crf)], True
    if encoder == "h264_qsv":
        # QSV accepts system-memory input and performs its own upload.  Using
        # global_quality keeps the probe and the real rawvideo command on the
        # same path; a bare hwupload filter would require a device context
        # whose name differs across Linux driver stacks.
        return "", ["-global_quality", str(crf)], False
    if encoder == "h264_videotoolbox":
        # VideoToolbox uses a quality scale rather than CRF.  Map the CLI's
        # 0..51 quality range onto its documented 1..63 range.
        quality = 1 if lossless else max(1, min(63, int(round(float(crf) * 63.0 / 51.0))))
        return "", ["-q:v", str(quality)], False
    return video_preset, ["-crf", str(crf)], False


def _select_video_encoder(
    requested: str,
    video_preset: str,
    crf: int,
    lossless: bool = False,
    *,
    probe_width: int = 256,
    probe_height: int = 256,
    probe_fps: int = 30,
    preserve_chroma: bool = False,
) -> tuple[str, str, list[str]]:
    """Resolve ``auto`` without making a render depend on unavailable hardware."""

    requested = _validate_ffmpeg_token(requested, "video codec")
    video_preset = _validate_ffmpeg_token(video_preset, "video preset")
    if video_preset not in VIDEO_PRESET_CHOICES:
        raise ValueError(
            f"video preset must be one of: {', '.join(VIDEO_PRESET_CHOICES)}"
        )
    crf = _index_value(crf, "crf")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    hardware = {"h264_nvenc", "h264_qsv", "h264_vaapi", "h264_videotoolbox"}
    lossless = bool(lossless)
    if lossless:
        crf = 0
    preserve_chroma = bool(preserve_chroma or lossless)
    near_lossless = not lossless and crf <= NEAR_LOSSLESS_CRF
    if requested in hardware:
        usable = (
            _vaapi_encoder_usable(
                lossless=lossless,
                width=probe_width,
                height=probe_height,
                fps=probe_fps,
            )
            if requested == "h264_vaapi"
            else _hardware_encoder_usable(
                requested,
                lossless,
                near_lossless,
                probe_width,
                probe_height,
                probe_fps,
                preserve_chroma,
                video_preset=video_preset,
            )
        )
        if not usable:
            raise RuntimeError(
                f"{requested} was requested, but its complete FFmpeg encode path "
                "failed the hardware probe"
            )
        preset, rate_control, _ = _video_encoder_settings(
            requested, video_preset, crf, lossless=lossless
        )
        return requested, preset, rate_control
    if requested == "libx264":
        if not _software_encoder_usable(
            requested,
            lossless,
            near_lossless,
            probe_width,
            probe_height,
            probe_fps,
            preserve_chroma,
            video_preset=video_preset,
        ):
            raise RuntimeError(
                "libx264 was requested, but its complete FFmpeg encode path "
                "failed the software probe"
            )
        preset, rate_control, _ = _video_encoder_settings(
            requested, video_preset, crf, lossless=lossless
        )
        return requested, preset, rate_control
    if requested != "auto":
        preset, rate_control, _ = _video_encoder_settings(
            requested, video_preset, crf, lossless=lossless
        )
        return requested, preset, rate_control
    available = _ffmpeg_encoder_names()
    candidates: list[str] = []
    if shutil.which("nvidia-smi") is not None:
        candidates.append("h264_nvenc")
    if Path("/dev/dri/renderD128").exists():
        candidates.extend(["h264_qsv", "h264_vaapi"])
    if sys.platform == "darwin":
        candidates.append("h264_videotoolbox")
    for encoder in candidates:
        if encoder not in available:
            continue
        if encoder == "h264_nvenc":
            if not _hardware_encoder_usable(
                encoder,
                lossless,
                near_lossless,
                probe_width,
                probe_height,
                probe_fps,
                preserve_chroma,
                video_preset=video_preset,
            ):
                continue
            preset, rate_control, _ = _video_encoder_settings(
                encoder, video_preset, crf, lossless=lossless
            )
            return encoder, preset, rate_control
        if encoder == "h264_vaapi":
            if not _vaapi_encoder_usable(
                lossless=lossless,
                width=probe_width,
                height=probe_height,
                fps=probe_fps,
            ):
                continue
            preset, rate_control, _ = _video_encoder_settings(
                encoder, video_preset, crf, lossless=lossless
            )
            return encoder, preset, rate_control
        if encoder in {"h264_qsv", "h264_videotoolbox"}:
            if not _hardware_encoder_usable(
                encoder,
                lossless,
                near_lossless,
                probe_width,
                probe_height,
                probe_fps,
                preserve_chroma,
                video_preset=video_preset,
            ):
                continue
        preset, rate_control, _ = _video_encoder_settings(
            encoder, video_preset, crf, lossless=lossless
        )
        return encoder, preset, rate_control
    if available and "libx264" not in available:
        raise RuntimeError(
            "no usable hardware encoder was found and this FFmpeg build does not "
            "provide libx264"
        )
    if not _software_encoder_usable(
        "libx264",
        lossless,
        near_lossless,
        probe_width,
        probe_height,
        probe_fps,
        preserve_chroma,
        video_preset=video_preset,
    ):
        raise RuntimeError(
            "no usable hardware encoder was found and the libx264 software "
            "fallback failed its complete FFmpeg probe"
        )
    return "libx264", video_preset, ["-crf", str(crf)]


def _video_pixel_format(
    codec: str,
    palette: str,
    palette_file: Optional[Path] = None,
    lossless: bool = False,
    crf: int = 18,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    """Choose a pixel format that keeps high-frequency KFP colour detail.

    The normal H.264 default is 4:2:0, which is a good compatibility choice
    for Aurora-like gradients but smears KFP's one-pixel cyan/green edge
    detail into visibly blocky chroma squares. KFP profiles and exact
    lossless output retain that detail with 4:4:4; ordinary CRF 10 output
    stays on 4:2:0 for faster, smaller files. Other hardware H.264 paths keep
    4:2:0 because their accepted input formats are device-dependent and
    VAAPI is explicitly NV12.
    """

    kfp_profile = _kfp_profile_for_selection(palette, palette_file)
    odd_dimensions = (
        width is not None
        and height is not None
        and (int(width) % 2 or int(height) % 2)
    )
    preserve_chroma = lossless or kfp_profile is not None
    if odd_dimensions and codec in {"libx264", "h264_nvenc"}:
        # yuv420p rejects odd dimensions. These two encoders accept 4:4:4,
        # which preserves the requested output size without padding it.
        preserve_chroma = True
    if codec == "libx264" and preserve_chroma:
        return "yuv444p"
    if codec == "h264_nvenc" and preserve_chroma:
        return "yuv444p"
    return "yuv420p"


@lru_cache(maxsize=8)
def _palette_basis(max_iter: int) -> tuple[Any, Any, float]:
    """Return the static cosine basis for one smooth-iteration palette."""

    np = _require_numpy()
    palette_size = min(65536, max(4096, int(max_iter) * 4))
    palette_field = np.linspace(0.0, float(max_iter), palette_size, dtype=np.float32)
    angle = np.asarray(AURORA_BAND_THICKNESS * palette_field, dtype=np.float32)
    return np.cos(angle), np.sin(angle), (palette_size - 1) / max(float(max_iter), 1.0)


def _pitch_hue_angle(pitch: float) -> float:
    """Return the hue rotation applied to the legacy blue/yellow palette."""

    np = _require_numpy()
    signed = float(np.clip(2.0 * (float(pitch) - 0.5), -1.0, 1.0))
    # Median pitch leaves the old palette unchanged. Extremes rotate it by
    # roughly 58 degrees, preserving the two distinct gradient anchors.
    return signed * 0.16 * (2.0 * math.pi)


def _rotate_hue_rgb(palette: Any, pitch: float) -> Any:
    """Rotate a palette's chroma in YIQ while preserving its luminance."""

    np = _require_numpy()
    angle = _pitch_hue_angle(pitch)
    if abs(angle) <= 1.0e-15:
        return palette
    cosine = math.cos(angle)
    sine = math.sin(angle)
    red = palette[:, 0]
    green = palette[:, 1]
    blue = palette[:, 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    in_phase = 0.596 * red - 0.275 * green - 0.321 * blue
    quadrature = 0.212 * red - 0.523 * green + 0.311 * blue
    rotated_i = in_phase * cosine - quadrature * sine
    rotated_q = in_phase * sine + quadrature * cosine
    return np.stack((
        luminance + 0.956 * rotated_i + 0.621 * rotated_q,
        luminance - 0.272 * rotated_i - 0.647 * rotated_q,
        luminance - 1.106 * rotated_i + 1.703 * rotated_q,
    ), axis=1)


def _colourise(
    field: Any,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    pitch: float = 0.5,
) -> Any:
    np = _require_numpy()
    field = np.asarray(field, dtype=np.float32)
    inside = ~np.isfinite(field) | (field >= max_iter)
    # This is the original three-wave liquid gradient.  The field is a smooth
    # iteration count, so each channel is exactly
    #   0.5 - 0.5*cos(band_thickness*field - phase_channel).
    # Pitch rotates both anchors together; it does not replace the flowing
    # field gradient.
    cosine_basis, sine_basis, scale = _palette_basis(int(max_iter))
    split = AURORA_COLOUR_SPLIT * float(np.clip(vocal, 0.0, 1.0)) ** 2.0
    red_wave = cosine_basis * math.cos(phase) + sine_basis * math.sin(phase)
    green_wave = cosine_basis * math.cos(phase + split * AURORA_GREEN_SPLIT) \
        + sine_basis * math.sin(phase + split * AURORA_GREEN_SPLIT)
    blue_wave = cosine_basis * math.cos(phase + split) \
        + sine_basis * math.sin(phase + split)
    palette = np.empty((cosine_basis.size, 3), dtype=np.float32)
    palette[:, 0] = np.clip(
        (0.5 - 0.5 * red_wave) * 140.0, 0.0, 255.0
    )
    palette[:, 1] = np.clip(
        (0.5 - 0.5 * green_wave) * 140.0, 0.0, 255.0
    )
    palette[:, 2] = np.clip(
        (0.5 - 0.5 * blue_wave) * 140.0, 0.0, 255.0
    )
    palette = np.clip(_rotate_hue_rgb(palette, pitch), 0.0, 255.0).astype(np.uint8)
    # Do the arithmetic in float64 and sanitize before converting to an
    # integer index. A finite float32 field can still overflow during the
    # multiplication, and NumPy's float-to-int overflow result is platform
    # dependent (and can become a negative palette index).
    scaled = np.asarray(field, dtype=np.float64) * float(scale)
    scaled = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=float(palette.shape[0] - 1),
        neginf=0.0,
    )
    palette_indices = np.clip(
        scaled, 0.0, float(palette.shape[0] - 1)
    ).astype(np.intp)
    rgb = palette[palette_indices]
    rgb[inside] = 0
    return rgb


@lru_cache(maxsize=16)
def _custom_palette(name: str, size: int = 4096) -> Any:
    """Build a compact RGB palette once; frame controls only index it."""

    np = _require_numpy()
    if name not in BUILTIN_PALETTE_STOPS:
        raise ValueError(f"unknown palette: {name}")
    stops = BUILTIN_PALETTE_STOPS[name]
    anchors = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
    positions = np.linspace(0.0, 1.0, size, dtype=np.float32)
    output = np.empty((size, 3), dtype=np.float32)
    for channel in range(3):
        output[:, channel] = np.interp(
            positions,
            anchors,
            [stop[channel] for stop in stops],
        )
    return np.asarray(np.rint(output), dtype=np.uint8)


def _parse_palette_colour(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3 and all(character in "0123456789abcdefABCDEF" for character in text):
        text = "".join(character * 2 for character in text)
    if len(text) == 6 and all(character in "0123456789abcdefABCDEF" for character in text):
        return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
    components = text.replace(",", " ").split(None, 3)
    if len(components) != 3:
        raise ValueError(f"invalid palette colour '{value}'")
    channels = tuple(int(component) for component in components)
    if not all(0 <= channel <= 255 for channel in channels):
        raise ValueError(f"palette RGB values must be between 0 and 255: '{value}'")
    return channels  # type: ignore[return-value]


def _parse_kfp_integer_values(value: str, path: Path) -> list[int]:
    """Parse one comma-separated KFP ``Colors`` fragment."""

    value = value.split("#", 1)[0]
    tokens = value.replace(",", " ").split()
    try:
        values = [int(token, 10) for token in tokens]
    except ValueError as error:
        raise ValueError(f"invalid KFP Colors field: {path}") from error
    if len(values) > MAX_PALETTE_STOPS * 3:
        raise ValueError(
            f"palette file exceeds the {MAX_PALETTE_STOPS:,}-stop limit: {path}"
        )
    return values


def _parse_kfp_fields(palette_text: str) -> dict[str, str]:
    """Parse KFP ``key: value`` rows and wrapped continuation values."""

    fields: dict[str, str] = {}
    current_key: Optional[str] = None
    for raw_line in palette_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if separator:
            current_key = key.strip().casefold()
            fields[current_key] = value.strip()
        elif current_key is not None:
            previous = fields.get(current_key, "")
            fields[current_key] = f"{previous} {line}".strip()
    return fields


def _parse_kfp_stops(palette_text: str, path: Path) -> list[tuple[int, int, int]]:
    """Read Kalles Fraktaler's text ``Colors: r,g,b,...`` field."""

    fields = _parse_kfp_fields(palette_text)
    if "colors" not in fields:
        raise ValueError(f"KFP palette is missing a Colors field: {path}")
    values = _parse_kfp_integer_values(fields["colors"], path)
    # Some third-party exporters prefix the RGB stream with the number of
    # stops. Kalles' own files do not, but accepting this form is inexpensive.
    if (
        len(values) >= 7
        and 2 <= values[0] <= MAX_PALETTE_STOPS
        and len(values) == 1 + values[0] * 3
    ):
        values = values[1:]
    if len(values) < 6 or len(values) % 3:
        raise ValueError(
            f"KFP Colors field needs at least two complete RGB stops: {path}"
        )
    stops = []
    for index in range(0, len(values), 3):
        channels = tuple(values[index:index + 3])
        if not all(0 <= channel <= 255 for channel in channels):
            raise ValueError(f"KFP RGB values must be between 0 and 255: {path}")
        stops.append(channels)  # type: ignore[arg-type]
    return stops


def _parse_kfp_float(
    fields: dict[str, str],
    key: str,
    default: float,
    path: Path,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    value = fields.get(key.casefold())
    if value is None or not value.strip():
        return float(default)
    token = value.replace(",", " ").split()[0]
    try:
        result = float(token)
    except ValueError as error:
        raise ValueError(f"invalid KFP {key} field: {path}") from error
    if not math.isfinite(result):
        raise ValueError(f"KFP {key} must be finite: {path}")
    if minimum is not None and result < minimum:
        raise ValueError(f"KFP {key} is below its supported range: {path}")
    if maximum is not None and result > maximum:
        raise ValueError(f"KFP {key} is above its supported range: {path}")
    return result


def _parse_kfp_int(
    fields: dict[str, str],
    key: str,
    default: int,
    path: Path,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    value = fields.get(key.casefold())
    if value is None or not value.strip():
        return int(default)
    token = value.replace(",", " ").split()[0]
    try:
        result = int(token, 10)
    except ValueError as error:
        raise ValueError(f"invalid KFP {key} field: {path}") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"KFP {key} is below its supported range: {path}")
    if maximum is not None and result > maximum:
        raise ValueError(f"KFP {key} is above its supported range: {path}")
    return result


def _parse_kfp_bool(
    fields: dict[str, str],
    key: str,
    default: bool,
    path: Path,
) -> bool:
    value = fields.get(key.casefold())
    if value is None or not value.strip():
        return bool(default)
    token = value.strip().casefold()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid KFP {key} field: {path}")


def _parse_kfp_multi_colors(
    value: str,
    path: Path,
) -> tuple[tuple[int, int, int], ...]:
    """Parse Kalles' ``period,start,type`` multi-colour triples."""

    if not value.strip():
        return ()
    tokens = value.replace(",", " ").replace(";", " ").split()
    if len(tokens) % 3:
        raise ValueError(f"KFP MultiColors needs period,start,type triples: {path}")
    if len(tokens) > 3 * 256:
        raise ValueError(f"KFP MultiColors field is too large: {path}")
    triples: list[tuple[int, int, int]] = []
    for index in range(0, len(tokens), 3):
        try:
            period = float(tokens[index])
            start = float(tokens[index + 1])
            colour_type = int(tokens[index + 2], 10)
        except ValueError as error:
            raise ValueError(f"invalid KFP MultiColors field: {path}") from error
        if (
            not math.isfinite(period)
            or not math.isfinite(start)
            or period == 0.0
            or abs(period) > 1.0e12
            or abs(start) > 1.0e12
            or colour_type not in {0, 1, 2}
        ):
            raise ValueError(f"invalid KFP MultiColors value: {path}")
        if not period.is_integer() or not start.is_integer():
            raise ValueError(f"KFP MultiColors values must be integers: {path}")
        triples.append((int(period), int(start), colour_type))
    return tuple(triples)


def _parse_kfp_profile(palette_text: str, path: Path) -> KfpPalette:
    """Read a KFP gradient and the transfer settings that give it its look."""

    fields = _parse_kfp_fields(palette_text)
    stops = tuple(_parse_kfp_stops(palette_text, path))
    interior_text = fields.get("interiorcolor", "0,0,0").rstrip(" ,;")
    try:
        interior_color = _parse_palette_colour(interior_text)
    except ValueError as error:
        raise ValueError(f"invalid KFP InteriorColor field: {path}") from error
    return KfpPalette(
        stops=stops,
        iter_div=_parse_kfp_float(
            fields, "IterDiv", 1.0, path, minimum=1.0e-12, maximum=1.0e12
        ),
        color_offset=_parse_kfp_float(
            fields, "ColorOffset", 0.0, path, minimum=-1.0e12, maximum=1.0e12
        ),
        ratio=_parse_kfp_float(
            fields, "Ratio", 360.0, path, minimum=0.0, maximum=1.0e6
        ),
        color_method=_parse_kfp_int(
            fields, "ColorMethod", 0, path, minimum=0, maximum=11
        ),
        smooth_method=_parse_kfp_int(
            fields, "SmoothMethod", 0, path, minimum=0, maximum=2
        ),
        smooth=_parse_kfp_bool(fields, "Smooth", True, path),
        flat=_parse_kfp_bool(fields, "Flat", False, path),
        inverse_transition=_parse_kfp_bool(
            fields, "InverseTransition", False, path
        ),
        phase_color_strength=_parse_kfp_float(
            fields, "ColorPhaseStrength", 0.0, path, minimum=0.0, maximum=100.0
        ),
        multi_color=_parse_kfp_bool(fields, "MultiColor", False, path),
        blend_multi_color=_parse_kfp_bool(fields, "BlendMC", False, path),
        multi_colors=_parse_kfp_multi_colors(fields.get("multicolors", ""), path),
        power=_parse_kfp_float(fields, "Power", 2.0, path, minimum=1.0e-6, maximum=64.0),
        slopes=_parse_kfp_bool(fields, "Slopes", False, path),
        slope_power=_parse_kfp_float(
            fields, "SlopePower", 50.0, path, minimum=0.0, maximum=1.0e4
        ),
        slope_ratio=_parse_kfp_float(
            fields, "SlopeRatio", 20.0, path, minimum=0.0, maximum=1.0e4
        ),
        slope_angle=_parse_kfp_float(
            fields, "SlopeAngle", 45.0, path, minimum=-360.0, maximum=360.0
        ),
        differences=_parse_kfp_int(
            fields, "Differences", 3, path, minimum=0, maximum=7
        ),
        interior_color=interior_color,
    )


def _read_palette_text(path: Path) -> str:
    """Read a bounded UTF-8 palette file once its path has been validated."""

    try:
        file_size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"cannot read palette file: {path}") from error
    if file_size > MAX_PALETTE_FILE_BYTES:
        raise ValueError(
            f"palette file exceeds the {MAX_PALETTE_FILE_BYTES:,}-byte limit: {path}"
        )
    try:
        with path.open("rb") as handle:
            raw_text = handle.read(MAX_PALETTE_FILE_BYTES + 1)
    except OSError as error:
        raise ValueError(f"cannot read palette file: {path}") from error
    if len(raw_text) > MAX_PALETTE_FILE_BYTES:
        raise ValueError(
            f"palette file exceeds the {MAX_PALETTE_FILE_BYTES:,}-byte limit: {path}"
        )
    try:
        return raw_text.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"palette file is not valid UTF-8: {path}") from error


@lru_cache(maxsize=16)
def _kfp_file_profile(path_text: str, mtime_ns: int) -> KfpPalette:
    path = Path(path_text)
    return _parse_kfp_profile(_read_palette_text(path), path)


def _interpolate_palette_stops(
    stops: Any, size: int, np: Any
) -> Any:
    anchors = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
    positions = np.linspace(0.0, 1.0, size, dtype=np.float32)
    output = np.empty((size, 3), dtype=np.float32)
    for channel in range(3):
        output[:, channel] = np.interp(
            positions,
            anchors,
            [stop[channel] for stop in stops],
        )
    return np.asarray(np.rint(output), dtype=np.uint8)


@lru_cache(maxsize=16)
def _kfp_palette_lut(profile: KfpPalette, size: int = 1024) -> Any:
    """Expand KFP key colours with Kalles' cyclic interpolation."""

    np = _require_numpy()
    size = max(2, int(size))
    # Kalles builds m_cPos with double-precision temporaries and truncates
    # each channel only after the cyclic sine interpolation.  Keeping the
    # lookup construction in float64 avoids a one-bin drift at deep zooms,
    # especially for palettes with short, high-contrast key sequences.
    stops = np.asarray(profile.stops, dtype=np.float64)
    if stops.shape[0] < 2:
        raise ValueError("KFP palette needs at least two colour stops")
    # Kalles treats the key colours as a cycle.  The final table entry blends
    # the last key back into the first; a non-cyclic linspace leaves a visible
    # discontinuity whenever ColorOffset crosses 1024.
    position = (
        np.arange(size, dtype=np.float64) * float(stops.shape[0]) / float(size)
    )
    left = np.floor(position).astype(np.intp)
    right = (left + 1) % stops.shape[0]
    fraction = position - left.astype(np.float64)
    # Keep Kalles' source expression (rather than only its equivalent cosine
    # identity) so values that land exactly on a byte boundary follow the same
    # libm rounding path before the final truncating conversion.
    fraction = np.sin((fraction - 0.5) * np.pi) / 2.0 + 0.5
    lut = stops[left] * (1.0 - fraction[:, None]) + stops[right] * fraction[:, None]
    # Kalles stores m_cPos as unsigned bytes with a C-style conversion, so
    # the fractional channel is truncated rather than rounded. This matters
    # at the many half-way values in a short cyclic KFP gradient.
    return np.asarray(np.clip(lut, 0.0, 255.0), dtype=np.uint8)


@lru_cache(maxsize=16)
def _palette_file_palette(path_text: str, mtime_ns: int, size: int = 4096) -> Any:
    """Load ``#rrggbb`` or ``r g b`` stops and interpolate them once."""

    np = _require_numpy()
    path = Path(path_text)
    palette_text = _read_palette_text(path)
    if path.suffix.casefold() == ".kfp":
        stops = _parse_kfp_stops(palette_text, path)
        return _interpolate_palette_stops(stops, size, np)

    stops = []
    for line in palette_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") and len(line) not in {4, 7}:
            continue
        if not line.startswith("#"):
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        stops.append(_parse_palette_colour(line))
        if len(stops) > MAX_PALETTE_STOPS:
            raise ValueError(
                f"palette file exceeds the {MAX_PALETTE_STOPS:,}-stop limit: {path}"
            )
    if len(stops) < 2:
        raise ValueError(f"palette file needs at least two colour stops: {path}")
    return _interpolate_palette_stops(stops, size, np)


def _palette_from_file(path: Path, size: int = 4096) -> Any:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _palette_file_palette(str(path), int(stat.st_mtime_ns), int(size))


def _kfp_profile_for_selection(
    palette_name: str,
    palette_file: Optional[Path],
) -> Optional[KfpPalette]:
    """Return the full KFP profile, leaving ordinary files on the Aurora path."""

    if palette_file is not None:
        path = Path(palette_file).expanduser().resolve()
        if path.suffix.casefold() != ".kfp":
            return None
        stat = path.stat()
        return _kfp_file_profile(str(path), int(stat.st_mtime_ns))
    if palette_name == "kalles-default":
        return KALLES_DEFAULT_KFP
    return None


@lru_cache(maxsize=16)
def _palette_file_aurora_accents(path_text: str, mtime_ns: int) -> tuple[tuple[int, int, int], ...]:
    """Convert a normal RGB-stop file into three Aurora wave accents."""

    palette = _palette_file_palette(path_text, mtime_ns, 64)
    indices = (8, 32, 56)
    return tuple(
        tuple(int(value) for value in palette[index])
        for index in indices
    )


def _aurora_accents_for_selection(
    palette_name: str,
    palette_file: Optional[Path],
) -> tuple[tuple[int, int, int], ...]:
    if palette_file is not None:
        path = Path(palette_file).expanduser().resolve()
        stat = path.stat()
        return _palette_file_aurora_accents(str(path), int(stat.st_mtime_ns))
    try:
        return BUILTIN_AURORA_ACCENTS[palette_name]
    except KeyError as error:
        raise ValueError(f"unknown ordinary palette: {palette_name}") from error


def _ordinary_interior_color(
    palette_name: str,
    palette_file: Optional[Path] = None,
) -> tuple[int, int, int]:
    """Return the set colour for an ordinary (non-KFP) palette."""

    if palette_file is not None:
        return (0, 0, 0)
    return BUILTIN_INTERIOR_COLORS.get(palette_name, (0, 0, 0))


def _apply_interior_color(
    rgb: Any,
    field: Any,
    max_iter: int,
    colour: tuple[int, int, int],
) -> Any:
    """Paint interior samples without allocating a second RGB frame."""

    np = _require_numpy()
    output = np.asarray(rgb, dtype=np.uint8)
    scalar = np.asarray(field, dtype=np.float32)
    if output.ndim != 3 or output.shape[-1] != 3 or output.shape[:2] != scalar.shape:
        raise ValueError("interior colour input shapes do not match")
    inside = ~np.isfinite(scalar) | (scalar >= float(max_iter))
    output[inside] = np.asarray(colour, dtype=np.uint8)
    return output


def _apply_aurora_accents(
    base_rgb: Any,
    accents: tuple[tuple[int, int, int], ...],
    pitch: float = 0.5,
) -> Any:
    """Recolour Aurora's three waves without throwing away their phase detail."""

    np = _require_numpy()
    base = np.asarray(base_rgb, dtype=np.float32)
    if base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError("Aurora accent input must have shape (height, width, 3)")
    accent_array = np.asarray(accents, dtype=np.float32)
    if accent_array.shape != (3, 3):
        raise ValueError("Aurora accents must contain three RGB colours")
    # Native Aurora emits one 0..140 wave per channel.  The weighted sum keeps
    # all three waves visible; dividing by 1.8 leaves headroom so a broad band
    # does not collapse into a clipped rectangle of one solid colour.
    weights = np.clip(base / 140.0, 0.0, 1.0)
    rgb = np.einsum("...c,cd->...d", weights, accent_array) / 1.8
    if abs(float(pitch) - 0.5) > 1.0e-12:
        shape = rgb.shape
        rgb = _rotate_hue_rgb(rgb.reshape(-1, 3), pitch).reshape(shape)
    return np.asarray(np.clip(np.rint(rgb), 0.0, 255.0), dtype=np.uint8)


def _native_apply_aurora_accents(
    rgb: Any,
    accents: tuple[tuple[int, int, int], ...],
    pitch: float,
    native_library: Any,
    native_threads: int,
) -> Optional[Any]:
    """Run the ordinary-palette accent pass in native code when available."""

    function = getattr(native_library, "fractal_apply_aurora_accents", None)
    if function is None:
        return None
    np = _require_numpy()
    output = np.ascontiguousarray(rgb, dtype=np.uint8)
    if output.ndim != 3 or output.shape[-1] != 3:
        raise ValueError("Aurora accent output must have shape (height, width, 3)")
    if len(accents) != 3 or any(len(colour) != 3 for colour in accents):
        raise ValueError("Aurora accents must contain three RGB colours")
    accent_values = (ctypes.c_uint8 * 9)(
        *(int(channel) for colour in accents for channel in colour)
    )
    status = function(
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(output.shape[1]),
        int(output.shape[0]),
        accent_values,
        float(pitch),
        int(native_threads),
    )
    if status != 0:
        message = native_library.fractal_last_error() or b"unknown native Aurora accent error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def _colourise_aurora_accents(
    field: Any,
    max_iter: int,
    phase: float,
    vocal: float,
    accents: tuple[tuple[int, int, int], ...],
    pitch: float = 0.5,
    interior_color: tuple[int, int, int] = (0, 0, 0),
) -> Any:
    """Python fallback for ordinary palettes using Aurora's raw phase waves."""

    np = _require_numpy()
    field = np.asarray(field, dtype=np.float32)
    inside = ~np.isfinite(field) | (field >= float(max_iter))
    safe_field = np.nan_to_num(
        np.asarray(field, dtype=np.float64),
        nan=0.0,
        posinf=float(max_iter),
        neginf=0.0,
    )
    safe_field = np.clip(safe_field, 0.0, float(max_iter))
    angle = AURORA_BAND_THICKNESS * safe_field
    split = AURORA_COLOUR_SPLIT * float(np.clip(vocal, 0.0, 1.0)) ** 2.0
    waves = np.stack((
        0.5 - 0.5 * np.cos(angle - float(phase)),
        0.5 - 0.5 * np.cos(angle - (float(phase) + split * AURORA_GREEN_SPLIT)),
        0.5 - 0.5 * np.cos(angle - (float(phase) + split)),
    ), axis=-1).astype(np.float32)
    rgb = _apply_aurora_accents(waves * 140.0, accents, pitch)
    rgb[inside] = np.asarray(interior_color, dtype=np.uint8)
    return rgb


def _kfp_slope_gradient(field: Any, np: Any) -> tuple[Any, Any]:
    """Return Kalles' one-sided gradient used by the slope pass."""

    height, width = field.shape
    dx = np.zeros_like(field, dtype=np.float64)
    dy = np.zeros_like(field, dtype=np.float64)
    if width > 1:
        # The normal Kalles renderer starts with its CPU colour path. Its
        # finite-difference slope prefers the previous neighbour and only
        # falls forward at the left edge; matching that convention keeps the
        # default .kfp relief oriented like Kalles' own non-OpenGL output.
        dx[:, 0] = field[:, 0] - field[:, 1]
        dx[:, 1:] = field[:, :-1] - field[:, 1:]
    if height > 1:
        dy[0, :] = field[0, :] - field[1, :]
        dy[1:, :] = field[:-1, :] - field[1:, :]
    return dx, dy


def _kfp_difference_magnitude(field: Any, differences: int, np: Any) -> Any:
    """Apply Kalles' selectable 3x3 difference operators.

    A scalar iteration image cannot provide Kalles' analytic distance
    derivative, but using the selected finite-difference stencil preserves
    Kalles' local scale.  Kalles reflects missing samples at image boundaries
    before differencing; edge padding looks harmless but changes the slope and
    distance transfer exactly where a tile seam is most visible.
    """

    values = np.asarray(field, dtype=np.float64)

    height, width = values.shape
    centre = values

    # Cache the one-dimensional index/mask work for all eight shifts. A
    # single shared halo cannot represent Kalles' exact corner rule: for
    # example, the missing top-left neighbour uses the opposite diagonal,
    # while the missing top neighbour uses the opposite vertical sample.
    # Precomputing these vectors retains the exact rule and avoids rebuilding
    # them inside every shift.
    x = np.arange(width, dtype=np.intp)
    y = np.arange(height, dtype=np.intp)
    x_data: dict[int, tuple[Any, Any]] = {}
    y_data: dict[int, tuple[Any, Any]] = {}
    for offset in (-1, 0, 1):
        source_x = x + offset
        source_y = y + offset
        valid_x = (source_x >= 0) & (source_x < width)
        valid_y = (source_y >= 0) & (source_y < height)
        x_data[offset] = (np.clip(source_x, 0, width - 1), valid_x)
        y_data[offset] = (np.clip(source_y, 0, height - 1), valid_y)

    def reflected_shift(offset_x: int, offset_y: int) -> Any:
        source_x, valid_x = x_data[offset_x]
        source_y, valid_y = y_data[offset_y]
        result = values[np.ix_(source_y, source_x)]
        valid = valid_y[:, None] & valid_x[None, :]
        if np.all(valid):
            return result
        opposite_x = x - offset_x
        opposite_y = y - offset_y
        opposite_valid_x = (opposite_x >= 0) & (opposite_x < width)
        opposite_valid_y = (opposite_y >= 0) & (opposite_y < height)
        opposite = values[
            np.ix_(
                np.clip(opposite_y, 0, height - 1),
                np.clip(opposite_x, 0, width - 1),
            )
        ]
        opposite_valid = opposite_valid_y[:, None] & opposite_valid_x[None, :]
        return np.where(
            valid,
            result,
            np.where(opposite_valid, 2.0 * centre - opposite, centre),
        )

    left = reflected_shift(-1, 0)
    right = reflected_shift(1, 0)
    up = reflected_shift(0, -1)
    down = reflected_shift(0, 1)
    top_left = reflected_shift(-1, -1)
    top_right = reflected_shift(1, -1)
    bottom_left = reflected_shift(-1, 1)
    bottom_right = reflected_shift(1, 1)
    diagonal_distance_squared = 2.0
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    if differences == 0:  # Traditional, as in CFraktalSFT::SetColor.
        return (
            np.abs(left - centre) * math.sqrt(2.0)
            + np.abs(up - centre) * math.sqrt(2.0)
            # Kalles multiplies every delta by sqrt(2) before dividing by
            # its geometric neighbour distance. Diagonal neighbours are
            # already sqrt(2) pixels away, so their net coefficient is one.
            + np.abs(top_left - centre)
            + np.abs(bottom_left - centre)
        )
    if differences == 1:  # Forward 3x3, eight radial differences.
        squared = (
            (left - centre) ** 2
            + (right - centre) ** 2
            + (up - centre) ** 2
            + (down - centre) ** 2
            + ((top_left - centre) ** 2 + (bottom_right - centre) ** 2)
                * inv_sqrt_two ** 2
            + ((bottom_left - centre) ** 2 + (top_right - centre) ** 2)
                * inv_sqrt_two ** 2
        )
        return np.sqrt(np.maximum(0.0, squared * 0.25)) * 2.8284271247461903
    if differences == 2:  # Central 3x3, four diameter differences.
        squared = (
            (right - left) ** 2 / 4.0
            + (down - up) ** 2 / 4.0
            + (bottom_right - top_left) ** 2 / 8.0
            + (top_right - bottom_left) ** 2 / 8.0
        )
        return np.sqrt(np.maximum(0.0, squared * 0.5)) * 2.8284271247461903
    if differences == 3:  # Diagonal 2x2 / Roberts Cross.
        squared = (
            (top_left - centre) ** 2 / diagonal_distance_squared
            + (left - up) ** 2 / diagonal_distance_squared
        )
        return np.sqrt(np.maximum(0.0, squared)) * 2.8284271247461903
    if differences == 4:  # Least-squares 2x2.
        # The 2x2 stencil is centred at (+/-0.5,+/-0.5), so its two
        # least-squares slopes reduce to these two four-sample contrasts.
        dx = ((up - top_left) + (centre - left)) * 0.5
        dy = ((left - top_left) + (centre - up)) * 0.5
        return np.hypot(dx, dy) * 2.8284271247461903
    if differences == 5:  # Least-squares 3x3.
        dx = (right + top_right + bottom_right - left - top_left - bottom_left) / 6.0
        dy = (down + bottom_left + bottom_right - up - top_left - top_right) / 6.0
        return np.hypot(dx, dy) * 2.8284271247461903
    if differences == 6:  # Laplacian 3x3.
        laplacian = (
            top_left + 4.0 * up + top_right
            + 4.0 * left - 20.0 * centre + 4.0 * right
            + bottom_left + 4.0 * down + bottom_right
        )
        return np.sqrt(np.abs(laplacian / 6.0 * 1.4426950408889634)) * 2.8284271247461903
    # Analytic (7) is not available from a scalar field. Central differences
    # are the least surprising fallback for imported profiles using it.
    return np.sqrt(
        np.maximum(
            0.0,
            (right - left) ** 2 / 4.0 + (down - up) ** 2 / 4.0,
        )
    ) * 2.8284271247461903


def _hsv_to_rgb(hue: Any, saturation: Any, value: Any, np: Any) -> Any:
    """Vectorised HSV conversion matching Kalles' 0..1 colour coordinates."""

    hue = np.mod(hue, 1.0)
    saturation = np.clip(saturation, 0.0, 1.0)
    value = np.clip(value, 0.0, 1.0)
    scaled_hue = hue * 6.0
    sector = np.floor(scaled_hue).astype(np.intp)
    fraction = scaled_hue - np.floor(scaled_hue)
    fraction = np.where((sector & 1) == 0, 1.0 - fraction, fraction)
    minimum = value * (1.0 - saturation)
    transition = value * (1.0 - saturation * fraction)
    choices = (
        np.stack((minimum, transition, value), axis=-1),
        np.stack((minimum, value, transition), axis=-1),
        np.stack((transition, value, minimum), axis=-1),
        np.stack((value, transition, minimum), axis=-1),
        np.stack((value, minimum, transition), axis=-1),
        np.stack((transition, minimum, value), axis=-1),
    )
    output = np.empty(hue.shape + (3,), dtype=np.float64)
    for index, choice in enumerate(choices):
        output[sector == index] = choice[sector == index]
    return output


def _kfp_dither_rgb(
    rgb: Any,
    np: Any,
    x_offset: int = 0,
    y_offset: int = 0,
) -> Any:
    """Apply Kalles' deterministic ordered dither before converting to RGB8."""

    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("KFP dither input must have shape (height, width, 3)")
    height, width, _ = values.shape
    x = (np.arange(width, dtype=np.uint64) + np.uint64(max(0, int(x_offset))))[None, :]
    y = (np.arange(height, dtype=np.uint64) + np.uint64(max(0, int(y_offset))))[:, None]
    output = np.empty(values.shape, dtype=np.uint8)
    for channel in range(3):
        mixed = (
            x
            + np.uint64(channel * 67)
            + y * np.uint64(236)
        ) * np.uint64(119)
        mask = (mixed & np.uint64(255)).astype(np.float64) / 256.0
        channel_values = np.nan_to_num(
            values[..., channel], nan=0.0, posinf=255.0, neginf=0.0
        )
        output[..., channel] = np.asarray(
            np.clip(np.floor(np.clip(channel_values, 0.0, 255.0) + mask), 0.0, 255.0),
            dtype=np.uint8,
        )
    return output


def _colourise_kfp(
    field: Any,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    pitch: float,
    profile: KfpPalette,
    spatial_width: Optional[int] = None,
    dither_x: int = 0,
    dither_y: int = 0,
) -> Any:
    """Apply the portable Kalles transfer, multi-colour, and slope stages."""

    # A KFP is a complete colour recipe.  Kalles applies the imported colours
    # and slope pass directly; the visualizer's audio/pitch colour transforms
    # belong to the ordinary Aurora palettes and would otherwise make the
    # bundled Kalles defaults drift into a different palette while rendering.
    del phase, vocal, instrumental, pitch
    np = _require_numpy()
    field = np.asarray(field, dtype=np.float32)
    inside = np.isfinite(field) & (field >= float(max_iter))
    safe = np.nan_to_num(
        np.asarray(field, dtype=np.float64),
        # Non-finite samples are numerical faults, not bounded-set markers.
        # Keep them on the escaped fallback path so they cannot turn an
        # entire atlas tile into the interior colour.
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe = np.clip(safe, 0.0, float(max_iter))
    smooth_iter = np.maximum(0.0, safe)
    colour_iter = np.floor(smooth_iter) if profile.flat else smooth_iter
    method = int(profile.color_method)
    needs_difference = method in {5, 6, 7, 8}
    needs_slopes = bool(
        profile.slopes
        and profile.slope_power > 0.0
        and profile.slope_ratio > 0.0
    )
    # Kalles represents an interior sample as (max_iter, transition=0), which
    # reconstructs to max_iter + 1 when that sample is used by a neighbour
    # stencil. Keep the centre colour marker at max_iter, but retain the
    # reconstructed value for the distance and slope passes so boundaries do
    # not turn into a flat rectangular fill.
    stencil_iter = np.where(inside, float(max_iter) + 1.0, smooth_iter)
    if needs_difference:
        gradient = _kfp_difference_magnitude(stencil_iter, profile.differences, np)
    else:
        gradient = np.zeros_like(smooth_iter, dtype=np.float64)
    if needs_slopes:
        slope_dx, slope_dy = _kfp_slope_gradient(stencil_iter, np)
    else:
        slope_dx = slope_dy = None
    # Atlas children are rendered into a smaller rectangle, but their pixels
    # represent the same final-screen density as the parent crop. Use the
    # output width for Kalles' resolution-dependent distance/slope scale so a
    # child does not suddenly change palette contrast at its rectangular edge.
    width = max(1, int(spatial_width or field.shape[1]))
    # A scalar iteration field does not carry Kalles' analytic DE derivatives.
    # Its selected finite-difference magnitude is the closest portable local
    # distance proxy and, unlike normalising by max_iter, retains detail at any
    # absolute zoom depth.
    # The individual Kalles difference operators return their own native
    # scale (some include the historical 2*sqrt(2) factor); only the image
    # width normalization belongs here.
    distance = gradient * width / 640.0
    distance = np.nan_to_num(distance, nan=0.0, posinf=1.0e12, neginf=0.0)
    if method == 1:
        transfer = np.sqrt(colour_iter)
    elif method == 2:
        transfer = np.cbrt(colour_iter)
    elif method == 3:
        transfer = np.log(np.maximum(1.0, colour_iter))
    elif method == 4:
        # Kalles' GetIterations() range is based on integer nIter0 values,
        # not the fractional transition used for the final pixel colour.
        escaped = np.floor(smooth_iter[~inside])
        minimum = float(np.min(escaped)) if escaped.size else 0.0
        maximum = float(np.max(escaped)) if escaped.size else minimum + 1.0
        transfer = 1024.0 * (colour_iter - minimum) / max(maximum - minimum, 1.0e-12)
    elif method == 5:
        transfer = np.minimum(distance, 1024.0)
    elif method == 6:
        # Kalles uses the square-root distance transfer for DEPlusStandard
        # before deciding whether to fall back to the ordinary iteration.
        distance_transfer = np.minimum(
            np.sqrt(np.maximum(0.0, distance)),
            1024.0,
        )
        transfer = np.where(
            distance_transfer > profile.iter_div,
            # CFraktalSFT restores nIter + 1 - offs here.  That is the
            # original smooth scalar even when Flat only affected the first
            # colour-method transform.
            smooth_iter,
            distance_transfer,
        )
    elif method == 7:
        transfer = np.log(np.maximum(1.0, distance + 1.0))
    elif method == 8:
        transfer = np.sqrt(np.maximum(0.0, distance))
    elif method == 9:
        transfer = np.log1p(np.log1p(colour_iter))
    elif method == 10:
        transfer = np.arctan(colour_iter)
    elif method == 11:
        transfer = np.power(colour_iter, 0.25)
    else:
        transfer = colour_iter
    transfer = np.nan_to_num(
        transfer,
        nan=0.0,
        posinf=float(max(max_iter, 1024)),
        neginf=0.0,
    )
    if method in {5, 7, 8}:
        transfer = np.clip(transfer, 0.0, 1024.0)

    lut = _kfp_palette_lut(profile)
    position = np.mod(transfer / profile.iter_div + profile.color_offset, 1024.0)
    lower = np.floor(position).astype(np.intp) % 1024
    if profile.smooth:
        fraction = position - np.floor(position)
        if profile.inverse_transition:
            fraction = 1.0 - fraction
        upper = (lower + 1) % 1024
        rgb = lut[lower].astype(np.float64) * (1.0 - fraction[..., None])
        rgb += lut[upper].astype(np.float64) * fraction[..., None]
    else:
        rgb = lut[lower].astype(np.float64)

    if profile.multi_color and profile.multi_colors:
        hues: list[Any] = []
        saturations: list[Any] = []
        values: list[Any] = []
        wave_input = transfer / profile.iter_div + profile.color_offset
        if not profile.smooth:
            wave_input = np.floor(wave_input)
        for period, _start, colour_type in profile.multi_colors:
            if period < 0:
                wave = np.full_like(transfer, -float(period) / 100.0)
            else:
                wave = 0.5 + 0.5 * np.sin(
                    np.pi * wave_input / float(period)
                )
            if colour_type == 0:
                hues.append(wave)
            elif colour_type == 1:
                saturations.append(wave)
            else:
                values.append(wave)
        hue = np.mean(hues, axis=0) if hues else np.zeros_like(transfer)
        saturation = np.mean(saturations, axis=0) if saturations else np.ones_like(transfer)
        value = np.mean(values, axis=0) if values else np.ones_like(transfer)
        multi_rgb = _hsv_to_rgb(hue, saturation, value, np) * 255.0
        rgb = (rgb + multi_rgb) * 0.5 if profile.blend_multi_color else multi_rgb

    if needs_slopes:
        assert slope_dx is not None and slope_dy is not None
        angle = math.radians(profile.slope_angle)
        projected = slope_dx * math.cos(angle) + slope_dy * math.sin(angle)
        projected *= profile.slope_power * width / 640.0
        strength = np.clip(
            np.arctan(np.abs(projected)) / (math.pi / 2.0)
            * profile.slope_ratio / 100.0,
            0.0,
            1.0,
        )[..., None]
        dark = projected >= 0.0
        rgb = np.where(
            dark[..., None],
            rgb * (1.0 - strength),
            rgb * (1.0 - strength) + 255.0 * strength,
        )

    rgb = np.clip(rgb, 0.0, 255.0)
    rgb = _kfp_dither_rgb(rgb, np, dither_x, dither_y)
    rgb[inside] = np.asarray(profile.interior_color, dtype=np.uint8)
    return rgb


@lru_cache(maxsize=32)
def _native_kfp_options(profile: KfpPalette) -> NativeKfpOptions:
    """Build one immutable ctypes transfer block per imported KFP profile."""

    return NativeKfpOptions.from_profile(profile)


def _colourise_kfp_native(
    field: Any,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    pitch: float,
    profile: KfpPalette,
    native_library: Any,
    native_threads: int,
) -> Any:
    """Apply KFP's transfer in one OpenMP-native pass over the scalar field."""

    function = getattr(native_library, "fractal_colourise_kfp", None)
    if function is None:
        return _colourise_kfp(
            field, max_iter, phase, vocal, instrumental, pitch, profile
        )
    np = _require_numpy()
    scalar = np.ascontiguousarray(field, dtype=np.float32)
    if scalar.ndim != 2:
        raise ValueError("KFP colour input must be two-dimensional")
    output = np.empty(scalar.shape + (3,), dtype=np.uint8)
    lut = np.ascontiguousarray(_kfp_palette_lut(profile), dtype=np.uint8)
    options = _native_kfp_options(profile)
    status = function(
        scalar.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(scalar.shape[1]),
        int(scalar.shape[0]),
        int(max_iter),
        float(phase),
        float(vocal),
        float(instrumental),
        float(pitch),
        ctypes.byref(options),
        lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(lut.shape[0]),
        int(native_threads),
    )
    if status != 0:
        message = native_library.fractal_last_error() or b"unknown native KFP colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def _crop_colourise_kfp_native(
    field: Any,
    output_width: int,
    output_height: int,
    zoom_factor: float,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    pitch: float,
    profile: KfpPalette,
    native_library: Any,
    native_threads: int,
) -> Any:
    """Crop and colourise a KFP frame without a Python/Pillow round trip."""

    function = getattr(native_library, "fractal_crop_colourise_kfp", None)
    if function is None:
        raise RuntimeError("native KFP crop colourizer is unavailable")
    np = _require_numpy()
    scalar = np.ascontiguousarray(field, dtype=np.float32)
    if scalar.ndim != 2:
        raise ValueError("KFP crop input must be two-dimensional")
    output = np.empty((int(output_height), int(output_width), 3), dtype=np.uint8)
    lut = np.ascontiguousarray(_kfp_palette_lut(profile), dtype=np.uint8)
    options = _native_kfp_options(profile)
    status = function(
        scalar.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(scalar.shape[1]),
        int(scalar.shape[0]),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(output_width),
        int(output_height),
        float(zoom_factor),
        int(max_iter),
        float(phase),
        float(vocal),
        float(instrumental),
        float(pitch),
        ctypes.byref(options),
        lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(lut.shape[0]),
        int(native_threads),
    )
    if status != 0:
        message = native_library.fractal_last_error() or b"unknown native KFP crop colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def _atlas_colourise_kfp_native(
    parent: Any,
    child: Any,
    output_width: int,
    output_height: int,
    max_iter: int,
    child_left: int,
    child_top: int,
    feather: int,
    phase: float,
    vocal: float,
    instrumental: float,
    pitch: float,
    profile: KfpPalette,
    native_library: Any,
    native_threads: int,
) -> Any:
    """Colour a nested KFP frame without materialising two RGB atlases."""

    function = getattr(native_library, "fractal_atlas_colourise_kfp", None)
    if function is None:
        raise RuntimeError("native KFP atlas colourizer is unavailable")
    np = _require_numpy()
    parent_array = np.ascontiguousarray(parent, dtype=np.float32)
    child_array = np.ascontiguousarray(child, dtype=np.float32)
    if parent_array.shape != (int(output_height), int(output_width)):
        raise ValueError("native KFP atlas parent does not match output dimensions")
    if child_array.ndim != 2:
        raise ValueError("native KFP atlas child must be two-dimensional")
    child_height, child_width = child_array.shape
    halo = min(
        2,
        int(child_left),
        int(output_width) - int(child_left) - child_width,
        int(child_top),
        int(output_height) - int(child_top) - child_height,
    )
    if halo > 0:
        child_array = np.pad(
            child_array,
            ((halo, halo), (halo, halo)),
            mode="edge",
        )
        child_height, child_width = child_array.shape
        child_left -= halo
        child_top -= halo
        feather = min(
            child_width,
            child_height,
            max(2, int(feather) + halo),
        )
    output = np.empty((int(output_height), int(output_width), 3), dtype=np.uint8)
    lut = np.ascontiguousarray(_kfp_palette_lut(profile), dtype=np.uint8)
    options = _native_kfp_options(profile)
    status = function(
        parent_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(output_width),
        int(output_height),
        child_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(child_width),
        int(child_height),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(output_width),
        int(output_height),
        int(max_iter),
        int(child_left),
        int(child_top),
        int(feather),
        float(phase),
        float(vocal),
        float(instrumental),
        float(pitch),
        ctypes.byref(options),
        lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(lut.shape[0]),
        int(native_threads),
    )
    if status != 0:
        message = native_library.fractal_last_error() or b"unknown native KFP atlas colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def _atlas_colourise_kfp_raw_native(
    parent: Any,
    child: Any,
    output_width: int,
    output_height: int,
    parent_zoom: float,
    child_fraction: float,
    parent_iter: int,
    child_iter: Optional[int],
    phase: float,
    vocal: float,
    instrumental: float,
    pitch: float,
    profile: KfpPalette,
    native_library: Any,
    native_threads: int,
) -> Any:
    """Reproject raw atlas tiles and colourise them without Python image work."""

    function = getattr(native_library, "fractal_atlas_colourise_kfp_raw", None)
    if function is None:
        raise RuntimeError("native raw KFP atlas colourizer is unavailable")
    np = _require_numpy()
    parent_array = np.ascontiguousarray(parent, dtype=np.float32)
    if parent_array.ndim != 2:
        raise ValueError("native raw KFP atlas parent must be two-dimensional")
    try:
        requested_parent_zoom = float(parent_zoom)
        requested_child_fraction = float(child_fraction)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("native raw KFP atlas zoom controls must be numeric") from error
    if not math.isfinite(requested_parent_zoom) or requested_parent_zoom <= 0.0:
        raise ValueError("native raw KFP atlas parent zoom must be finite and positive")
    if not math.isfinite(requested_child_fraction):
        raise ValueError("native raw KFP atlas child fraction must be finite")
    effective_parent_zoom = max(requested_parent_zoom, 1.0)
    effective_child_fraction = min(max(requested_child_fraction, 0.0), 1.0)
    use_child = child is not None and child_iter is not None and effective_child_fraction > 0.0
    if not use_child:
        effective_child_fraction = 0.0
    if not use_child:
        child_array = np.empty((0, 0), dtype=np.float32)
        child_pointer = ctypes.POINTER(ctypes.c_float)()
        child_width = child_height = child_max_iter = 0
    else:
        child_array = np.ascontiguousarray(child, dtype=np.float32)
        if child_array.ndim != 2:
            raise ValueError("native raw KFP atlas child must be two-dimensional")
        child_height, child_width = child_array.shape
        child_max_iter = int(child_iter)
        child_pointer = child_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output = np.empty((int(output_height), int(output_width), 3), dtype=np.uint8)
    lut = np.ascontiguousarray(_kfp_palette_lut(profile), dtype=np.uint8)
    options = _native_kfp_options(profile)
    effective_iter = max(int(parent_iter), int(child_iter or parent_iter))
    status = function(
        parent_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(parent_array.shape[1]),
        int(parent_array.shape[0]),
        int(parent_iter),
        child_pointer,
        int(child_width),
        int(child_height),
        int(child_max_iter),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(output_width),
        int(output_height),
        effective_parent_zoom,
        effective_child_fraction,
        int(effective_iter),
        float(phase),
        float(vocal),
        float(instrumental),
        float(pitch),
        ctypes.byref(options),
        lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(lut.shape[0]),
        int(native_threads),
    )
    if status != 0:
        message = native_library.fractal_last_error() or b"unknown native raw KFP atlas error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


def _colourise_custom(
    field: Any,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    palette_name: str,
    pitch: float = 0.5,
    palette_file: Optional[Path] = None,
) -> Any:
    profile = _kfp_profile_for_selection(palette_name, palette_file)
    if profile is not None:
        return _colourise_kfp(
            field, max_iter, phase, vocal, instrumental, pitch, profile
        )
    return _colourise_aurora_accents(
        field,
        max_iter,
        phase,
        vocal,
        _aurora_accents_for_selection(palette_name, palette_file),
        pitch,
        _ordinary_interior_color(palette_name, palette_file),
    )


def _colourise_native(
    field: Any,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    native_threads: int,
    library: Any,
    pitch: float = 0.5,
) -> Any:
    """Colour one smooth-iteration frame through the OpenMP C ABI."""

    np = _require_numpy()
    field = np.ascontiguousarray(field, dtype=np.float32)
    height, width = field.shape
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    status = library.fractal_colourise(
        field.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        width,
        height,
        max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        native_threads,
    )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return rgb


def _crop_and_colourise_native(
    field: Any,
    output_width: int,
    output_height: int,
    zoom_factor: float,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    native_threads: int,
    library: Any,
    pitch: float = 0.5,
    interior_color: Optional[tuple[int, int, int]] = None,
    accents: Optional[tuple[tuple[int, int, int], ...]] = None,
) -> Any:
    """Fused centred bilinear crop and colour through the native C ABI."""

    np = _require_numpy()
    field = np.ascontiguousarray(field, dtype=np.float32)
    source_height, source_width = field.shape
    rgb = np.empty((output_height, output_width, 3), dtype=np.uint8)
    source_pointer = field.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_pointer = rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    accent_function = getattr(library, "fractal_crop_colourise_accents", None)
    if accents is not None and accent_function is not None:
        if len(accents) != 3 or any(len(colour) != 3 for colour in accents):
            raise ValueError("Aurora accents must contain three RGB colours")
        accent_values = (ctypes.c_uint8 * 9)(
            *(int(channel) for colour in accents for channel in colour)
        )
        selected_interior = interior_color or (0, 0, 0)
        if len(selected_interior) != 3 or not all(
            0 <= int(value) <= 255 for value in selected_interior
        ):
            raise ValueError("ordinary interior colour must contain three bytes")
        status = accent_function(
            source_pointer,
            source_width,
            source_height,
            output_pointer,
            output_width,
            output_height,
            float(zoom_factor),
            int(max_iter),
            float(phase),
            float(vocal),
            float(instrumental),
            float(pitch),
            accent_values,
            *(int(value) for value in selected_interior),
            native_threads,
        )
        if status != 0:
            message = library.fractal_last_error() or b"unknown native accent crop colourizer error"
            raise RuntimeError(message.decode("utf-8", errors="replace"))
        return rgb
    interior_function = getattr(library, "fractal_crop_colourise_interior", None)
    base_pitch = 0.5 if accents is not None else pitch
    if interior_color is not None and interior_function is not None:
        if len(interior_color) != 3 or not all(0 <= int(value) <= 255 for value in interior_color):
            raise ValueError("ordinary interior colour must contain three bytes")
        status = interior_function(
            source_pointer,
            source_width,
            source_height,
            output_pointer,
            output_width,
            output_height,
            zoom_factor,
            max_iter,
            phase,
            vocal,
            instrumental,
            base_pitch,
            *(int(value) for value in interior_color),
            native_threads,
        )
    else:
        status = library.fractal_crop_colourise(
            source_pointer,
            source_width,
            source_height,
            output_pointer,
            output_width,
            output_height,
            zoom_factor,
            max_iter,
            phase,
            vocal,
            instrumental,
            base_pitch,
            native_threads,
        )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native crop/colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    if accents is not None:
        interior_mask = None
        if interior_color is not None:
            interior_mask = np.all(
                rgb == np.asarray(interior_color, dtype=np.uint8), axis=-1
            )
        accented = _native_apply_aurora_accents(
            rgb,
            accents,
            pitch,
            library,
            native_threads,
        )
        if accented is None:
            return rgb
        if interior_mask is not None:
            accented[interior_mask] = np.asarray(interior_color, dtype=np.uint8)
        return accented
    if interior_color is not None and interior_function is None:
        # Compatibility with an older native .so: retain its fast crop/color
        # pass, then apply the exact interior mask through the portable mapper.
        _, inside = _crop_and_resize_preserving_interior(
            field, output_width, output_height, zoom_factor, max_iter, "bilinear"
        )
        rgb[inside] = np.asarray(interior_color, dtype=np.uint8)
    return rgb


def _atomic_save_field(path: Path, field: Any, *, durable: bool = False) -> None:
    """Write a keyframe atomically, optionally forcing it to stable storage.

    Atomic replacement is sufficient for normal resume behaviour and avoids a
    blocking fsync for every multi-megabyte atlas tile. ``durable`` is an
    explicit opt-in for users who need cache entries to survive a sudden power
    loss rather than merely an interrupted process.
    """

    np = _require_numpy()
    path = _absolute_path(path)
    _reject_final_symlink(path, "cache entry")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, np.asarray(field, dtype=np.float32), allow_pickle=False)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        if durable and hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


class _CacheEvictor:
    """Maintain a cheap incremental LRU index for generated cache files.

    The old implementation rescanned and sorted the entire cache directory
    after every tile.  That is harmless for a handful of preview files but
    becomes quadratic for a deep atlas.  This index scans existing files once,
    then updates a heap as files are loaded or written.
    """

    _PATTERNS = ("keyframe-*.npy", "atlas-tile-*.npy", "audio-*.npz")

    def __init__(self, cache_dir: Optional[Path], limit_mb: float) -> None:
        self.cache_dir = cache_dir
        self.limit_bytes = (
            int(limit_mb * 1024 * 1024)
            if cache_dir is not None and limit_mb > 0.0
            else 0
        )
        self._entries: dict[Path, tuple[int, int]] = {}
        self._heap: list[tuple[int, Path]] = []
        self._total_bytes = 0
        self._sequence = 0
        self._scanned = False
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None and self.limit_bytes > 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _scan_once(self) -> None:
        if self._scanned or not self.enabled:
            return
        self._scanned = True
        assert self.cache_dir is not None
        scanned = 0
        for pattern in self._PATTERNS:
            try:
                for path in self.cache_dir.glob(pattern):
                    if scanned >= MAX_CACHE_SCAN_ENTRIES:
                        return
                    scanned += 1
                    try:
                        if path.is_symlink() or not path.is_file():
                            continue
                        size = path.stat().st_size
                    except OSError:
                        continue
                    sequence = self._next_sequence()
                    self._entries[path] = (size, sequence)
                    self._total_bytes += size
                    heapq.heappush(self._heap, (sequence, path))
            except (OSError, RuntimeError):
                # Cache entries are advisory. A directory can be modified or
                # become inaccessible while it is being indexed; the render
                # should continue with the entries it could inspect.
                continue

    def observe(self, path: Path) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._scan_once()
            try:
                if path.is_symlink() or not path.is_file():
                    return
                size = path.stat().st_size
            except OSError:
                return
            previous = self._entries.get(path)
            if previous is not None:
                self._total_bytes -= previous[0]
            sequence = self._next_sequence()
            self._entries[path] = (size, sequence)
            self._total_bytes += size
            heapq.heappush(self._heap, (sequence, path))

    def touch(self, path: Path) -> None:
        """Mark a valid cache hit as recently used without rewriting it."""

        self.observe(path)

    def prune(self, protected: Optional[set[Path]] = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._scan_once()
            if self._total_bytes <= self.limit_bytes:
                return
            protected = protected or set()

            deferred: list[tuple[int, Path]] = []
            while self._total_bytes > self.limit_bytes and self._heap:
                sequence, path = heapq.heappop(self._heap)
                current = self._entries.get(path)
                if current is None or current[1] != sequence:
                    continue
                if path in protected:
                    deferred.append((sequence, path))
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    # Do not spin on an undeletable entry during this render.
                    deferred.append((sequence, path))
                    break
                self._total_bytes -= current[0]
                self._entries.pop(path, None)
            for item in deferred:
                heapq.heappush(self._heap, item)


def _keyframe_field(
    *,
    cache_dir: Optional[Path],
    cache_identity: str,
    renderer: str,
    render_width: int,
    render_height: int,
    log_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    series_order: int,
    series_block: int,
    native_reference: Any,
    native_threads: int,
    render_options: Optional[NativeRenderOptions] = None,
    durable_cache: bool = False,
    cache_evictor: Optional[_CacheEvictor] = None,
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
) -> Any:
    """Load or render one absolute-zoom field, with an atomic cache."""

    np = _require_numpy()
    cache_path = None
    if cache_dir is not None:
        cache_key = hashlib.sha256(
            repr((
                KEYFRAME_CACHE_SCHEMA,
                cache_identity,
                renderer,
                render_width,
                render_height,
                log_zoom,
                x_center,
                y_center,
                max_iter,
                series_order,
                series_block,
            )).encode("utf-8")
        ).hexdigest()[:20]
        cache_path = cache_dir / f"keyframe-{cache_key}.npy"
        try:
            with _safe_cache_load(cache_path) as cached:
                if cached is None:
                    raise ValueError("unsafe cached keyframe")
                if _valid_field_array(cached, (render_height, render_width)):
                    if cache_evictor is not None:
                        cache_evictor.touch(cache_path)
                    print(f"Using cached keyframe {cache_path.name}.", flush=True)
                    # Keep the returned field independent of a memory map that
                    # cache eviction may unlink while the render is running.
                    return np.array(cached, dtype=np.float32, copy=True)
        except (OSError, EOFError, ValueError, TypeError):
            pass

    field = render_fractal(
        render_width,
        render_height,
        log_zoom,
        x_center,
        y_center,
        max_iter,
        renderer,
        native_threads,
        native_reference,
        series_order,
        series_block,
        render_options,
        formula,
        julia_constant,
    )
    field = _validated_field(field, (render_height, render_width), "keyframe")
    if cache_path is not None:
        _atomic_save_field(cache_path, field, durable=durable_cache)
        if cache_evictor is not None:
            cache_evictor.observe(cache_path)
    return field


def _prune_cache(
    cache_dir: Optional[Path],
    limit_mb: float,
    evictor: Optional[_CacheEvictor] = None,
    protected: Optional[set[Path]] = None,
) -> None:
    """Optionally bound cache growth through an incremental eviction index."""

    active = evictor or _CacheEvictor(cache_dir, limit_mb)
    active.prune(protected or set())


def _colourise_view(
    view: Any,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    native_library: Any,
    native_threads: int,
    palette_name: str,
    pitch: float,
    palette_file: Optional[Path] = None,
) -> Any:
    """Colour an already-resampled scalar iteration field."""

    kfp_profile = _kfp_profile_for_selection(palette_name, palette_file)
    if kfp_profile is not None:
        if native_library is not None and hasattr(native_library, "fractal_colourise_kfp"):
            return _colourise_kfp_native(
                view,
                max_iter,
                phase,
                vocal,
                instrumental,
                pitch,
                kfp_profile,
                native_library,
                native_threads,
            )
        return _colourise_custom(
            view,
            max_iter,
            phase,
            vocal,
            instrumental,
            palette_name,
            pitch,
            palette_file,
        )
    if native_library is not None and palette_name == "aurora" and palette_file is None:
        return _colourise_native(
            view,
            max_iter,
            phase,
            vocal,
            instrumental,
            native_threads,
            native_library,
            pitch,
        )
    if native_library is not None:
        base = _colourise_native(
            view,
            max_iter,
            phase,
            vocal,
            instrumental,
            native_threads,
            native_library,
            0.5,
        )
        accents = _aurora_accents_for_selection(palette_name, palette_file)
        accented = _native_apply_aurora_accents(
            base, accents, pitch, native_library, native_threads
        )
        if accented is None:
            accented = _colourise_aurora_accents(
                view,
                max_iter,
                phase,
                vocal,
                accents,
                pitch,
                _ordinary_interior_color(palette_name, palette_file),
            )
        return _apply_interior_color(
            accented,
            view,
            max_iter,
            _ordinary_interior_color(palette_name, palette_file),
        )
    if palette_name != "aurora" or palette_file is not None:
        return _colourise_custom(
            view,
            max_iter,
            phase,
            vocal,
            instrumental,
            palette_name,
            pitch,
            palette_file,
        )
    return _colourise(view, max_iter, phase, vocal, instrumental, pitch)


def _colour_frame(
    field: Any,
    output_width: int,
    output_height: int,
    zoom_factor: float,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    native_library: Any,
    native_threads: int,
    resample: str,
    palette_name: str,
    pitch: float = 0.5,
    palette_file: Optional[Path] = None,
) -> Any:
    if (
        palette_file is None
        and native_library is not None
        and resample == "bilinear"
        and palette_name == "aurora"
    ):
        return _crop_and_colourise_native(
            field,
            output_width,
            output_height,
            zoom_factor,
            max_iter,
            phase,
            vocal,
            instrumental,
            native_threads,
            native_library,
            pitch,
        )
    kfp_profile = _kfp_profile_for_selection(palette_name, palette_file)
    if (
        kfp_profile is not None
        and native_library is not None
        and hasattr(native_library, "fractal_crop_colourise_kfp")
        and resample == "bilinear"
    ):
        return _crop_colourise_kfp_native(
            field,
            output_width,
            output_height,
            zoom_factor,
            max_iter,
            phase,
            vocal,
            instrumental,
            pitch,
            kfp_profile,
            native_library,
            native_threads,
        )
    if (
        native_library is not None
        and resample == "bilinear"
        and _kfp_profile_for_selection(palette_name, palette_file) is None
    ):
        interior_color = _ordinary_interior_color(palette_name, palette_file)
        return _crop_and_colourise_native(
            field,
            output_width,
            output_height,
            zoom_factor,
            max_iter,
            phase,
            vocal,
            instrumental,
            native_threads,
            native_library,
            pitch,
            interior_color if interior_color != (0, 0, 0) else None,
            _aurora_accents_for_selection(palette_name, palette_file),
        )
    view, _ = _crop_and_resize_preserving_interior(
        field,
        output_width,
        output_height,
        zoom_factor,
        max_iter,
        resample,
    )
    return _colourise_view(
        view,
        max_iter,
        phase,
        vocal,
        instrumental,
        native_library,
        native_threads,
        palette_name,
        pitch,
        palette_file,
    )


def _apply_frame_effects(
    rgb: Any,
    glow: float = 0.0,
    motion_blur: float = 0.0,
    previous: Any = None,
) -> Any:
    """Apply optional compositor effects after the scalar field is coloured."""

    np = _require_numpy()
    output = np.asarray(rgb, dtype=np.uint8)
    if glow > 0.0:
        try:
            from PIL import Image, ImageFilter

            height, width = output.shape[:2]
            small_size = (max(1, width // 8), max(1, height // 8))
            bloom = Image.fromarray(output, mode="RGB").resize(
                small_size, Image.Resampling.BILINEAR
            ).filter(ImageFilter.GaussianBlur(radius=2.0))
            bloom = bloom.resize((width, height), Image.Resampling.BILINEAR)
            bloom_array = np.asarray(bloom, dtype=np.float32)
            output = np.clip(
                output.astype(np.float32) + bloom_array * (0.65 * float(glow)),
                0.0,
                255.0,
            ).astype(np.uint8)
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError("Pillow is required for --glow") from exc
    if previous is not None and motion_blur > 0.0:
        output = np.asarray(
            np.rint(
                output.astype(np.float32) * (1.0 - float(motion_blur))
                + np.asarray(previous, dtype=np.float32) * float(motion_blur)
            ),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(output, dtype=np.uint8)


def _quality_source_settings(
    fractal_scale: float,
    quality: str,
    keyframe_mode: str,
    keyframe_factor: float,
    source_mode: str = "native",
) -> tuple[float, float]:
    """Return the scalar-field scale and transition for an export.

    ``source_mode`` is deliberately separate from the quality preset.  The
    normal/native path keeps at least one scalar-field sample per output
    pixel, while the explicit upscaled path uses the old quarter-resolution
    source that made 4K quick renders practical.  The final FFmpeg filter is
    still responsible for enlarging the latter to the requested video size.
    """

    if source_mode not in SOURCE_MODE_CHOICES:
        raise ValueError(f"unknown source mode: {source_mode}")
    if keyframe_mode == "atlas":
        quality_settings = {
            "draft": (max(fractal_scale, UPSCALED_SOURCE_SCALE), 0.0),
            "balanced": (max(fractal_scale, UPSCALED_SOURCE_SCALE), 0.0),
            "quality": (max(fractal_scale, 1.0), 0.0),
            "extreme": (max(fractal_scale, 1.25), 0.0),
        }
    elif keyframe_mode == "legacy":
        quality_settings = {
            "draft": (max(fractal_scale, UPSCALED_SOURCE_SCALE), 0.0),
            "balanced": (max(fractal_scale, 1.0), 0.12),
            "quality": (max(fractal_scale, keyframe_factor), 0.20),
            "extreme": (max(fractal_scale, keyframe_factor * 2.0), 0.28),
        }
    else:
        raise ValueError(f"unknown keyframe mode: {keyframe_mode}")
    if quality not in quality_settings:
        raise ValueError(f"unknown quality preset: {quality}")
    source_scale, transition_fraction = quality_settings[quality]
    if source_mode == "native":
        source_scale = max(source_scale, 1.0)
    else:
        source_scale = min(source_scale, UPSCALED_SOURCE_SCALE)
    return source_scale, transition_fraction


def render_video(
    audio_path: Path,
    output_path: Path,
    features: AudioFeatures,
    width: int,
    height: int,
    fps: int,
    render_scale: float,
    fractal_scale: float,
    keyframe_factor: float,
    x_center: str,
    y_center: str,
    zooms: Any,
    iteration_base: int,
    iterations_per_decade: int,
    iteration_cap: int,
    renderer: str,
    native_threads: int,
    series_order: int,
    series_block: int,
    video_preset: str,
    resample: str,
    encoder_threads: int,
    cache_dir: Optional[Path],
    quality: str = "balanced",
    palette: str = "aurora",
    cache_limit_mb: float = 0.0,
    video_codec: str = "auto",
    crf: int = 18,
    keyframe_mode: str = "atlas",
    durable_cache: bool = False,
    native_backend: str = "auto",
    formula: str = "mandelbrot",
    julia_constant: tuple[str, str] = DEFAULT_JULIA_C,
    palette_file: Optional[Path] = None,
    glow: float = 0.0,
    motion_blur: float = 0.0,
    lossless: bool = False,
    source_mode: str = "native",
) -> dict[str, Any]:
    np = _require_numpy()
    audio_path, output_path, _, cache_dir = _validate_render_paths(
        audio_path,
        output_path,
        cache_dir=cache_dir,
    )
    width, height = _validate_dimensions(width, height, "video")
    fps = _validate_fps(fps)
    features = _normalise_audio_features(features)
    frame_count = features.frame_count
    zooms = _validate_zoom_series(zooms, frame_count)
    render_scale = float(render_scale)
    fractal_scale = float(fractal_scale)
    keyframe_factor = _validate_keyframe_factor(keyframe_factor)
    if not math.isfinite(render_scale) or render_scale < 1.0:
        raise ValueError("render scale must be finite and at least 1")
    if not math.isfinite(fractal_scale) or fractal_scale <= 0.0:
        raise ValueError("fractal scale must be finite and positive")
    iteration_base = _validate_iteration_count(iteration_base, "iteration base")
    iterations_per_decade = _index_value(iterations_per_decade, "iterations per decade")
    if iterations_per_decade > MAX_ITERATION_BUDGET:
        raise ValueError(
            f"iterations per decade must be at most {MAX_ITERATION_BUDGET:,}"
        )
    iteration_cap = _validate_iteration_count(iteration_cap, "iteration cap")
    if iterations_per_decade < 0:
        raise ValueError("iterations per decade cannot be negative")
    if iteration_cap < iteration_base:
        raise ValueError("iteration cap must be greater than or equal to iteration base")
    native_threads = _validate_thread_count(native_threads, "native thread count")
    encoder_threads = _validate_thread_count(encoder_threads, "encoder thread count")
    series_order = _index_value(series_order, "series order")
    series_block = _index_value(series_block, "series block")
    if not 1 <= series_order <= 32 or not 2 <= series_block <= 4096:
        raise ValueError("series order/block are outside the supported range")
    if renderer not in {"auto", "native", "python"}:
        raise ValueError(f"unknown renderer: {renderer}")
    if native_backend not in NATIVE_BACKEND_NAMES and native_backend != "auto":
        raise ValueError(f"unknown native backend: {native_backend}")
    x_center = _validate_center_text(x_center, "real")
    y_center = _validate_center_text(y_center, "imaginary")
    if len(julia_constant) != 2:
        raise ValueError("Julia constant must contain real and imaginary coordinates")
    julia_constant = (
        _validate_center_text(julia_constant[0], "Julia real"),
        _validate_center_text(julia_constant[1], "Julia imaginary"),
    )
    cache_limit_mb = float(cache_limit_mb)
    if not math.isfinite(cache_limit_mb) or cache_limit_mb < 0.0:
        raise ValueError("cache limit must be finite and non-negative")
    if cache_limit_mb > 1_000_000.0:
        raise ValueError("cache limit is too large")
    glow = float(glow)
    motion_blur = float(motion_blur)
    if not math.isfinite(glow) or not 0.0 <= glow <= 1.0:
        raise ValueError("glow must be between 0 and 1")
    if not math.isfinite(motion_blur) or not 0.0 <= motion_blur < 1.0:
        raise ValueError("motion blur must be between 0 (off) and 1")
    video_codec = _validate_ffmpeg_token(video_codec, "video codec")
    video_preset = _validate_ffmpeg_token(video_preset, "video preset")
    crf = _index_value(crf, "crf")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    lossless = bool(lossless)
    if lossless:
        crf = 0
    source_scale, transition_fraction = _quality_source_settings(
        fractal_scale,
        quality,
        keyframe_mode,
        keyframe_factor,
        source_mode,
    )
    render_width, render_height = _scaled_dimensions(
        width,
        height,
        render_scale * float(source_scale),
        "fractal source",
    )
    formula = _formula_name(formula)
    if palette_file is not None:
        palette_file = _normalise_path(palette_file)
        if not palette_file.is_file():
            raise RuntimeError(f"palette file not found: {palette_file}")
        if _paths_refer_to_same_target(palette_file, audio_path):
            raise ValueError("palette file must be different from the input audio file")
        if _paths_refer_to_same_target(palette_file, output_path):
            raise ValueError("palette file must be different from the output video")
    preserve_chroma = (
        lossless or _kfp_profile_for_selection(palette, palette_file) is not None
    )
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to encode the video, but it was not found on PATH")
    selected_codec, selected_preset, rate_control = _select_video_encoder(
        video_codec,
        video_preset,
        crf,
        lossless,
        # The optional scale filter runs before the encoder, so probe the
        # dimensions that the codec actually receives, not the scalar-field
        # dimensions used to calculate the fractal.
        probe_width=width,
        probe_height=height,
        probe_fps=fps,
        preserve_chroma=preserve_chroma,
    )
    if selected_codec == "h264_vaapi" and (
        width < 128
        or height < 128
        or width > 4096
        or height > 4096
        or width % 2
        or height % 2
    ):
        if video_codec != "auto":
            raise RuntimeError(
                "h264_vaapi requires even dimensions between 128 and 4096 pixels"
            )
        selected_codec, selected_preset, rate_control = _select_video_encoder(
            "libx264",
            video_preset,
            crf,
            lossless,
            probe_width=width,
            probe_height=height,
            probe_fps=fps,
            preserve_chroma=preserve_chroma,
        )

    if formula != "mandelbrot" and native_backend == "auto":
        native_backend = "scalar"
    if formula != "mandelbrot" and native_backend in {"avx2", "opencl"}:
        raise RuntimeError(
            f"--native-backend {native_backend} is only available for Mandelbrot; "
            "use --native-backend scalar for alternate formulas"
        )
    if not math.isfinite(float(glow)) or not 0.0 <= float(glow) <= 1.0:
        raise ValueError("glow must be between 0 and 1")
    if not math.isfinite(float(motion_blur)) or not 0.0 <= float(motion_blur) < 1.0:
        raise ValueError("motion-blur must be between 0 (off) and 1")

    # Parent crops and neighbour interpolation are useful for a fast preview
    # but cannot be part of a quality master: they are exactly the source of
    # the rectangular/deep-black artefacts seen in earlier renders.
    allow_recovery = quality == "draft"
    cpu_count = max(2, os.cpu_count() or 2)
    if native_threads == 0:
        native_threads = min(MAX_THREAD_COUNT, max(1, (cpu_count * 2) // 3))
    frame_width = int(render_width)
    frame_height = int(render_height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("fractal frame dimensions must be positive")
    # The encoder destination is reserved only after validation and native
    # setup below have succeeded. This avoids orphaning a placeholder when a
    # hardware probe or deep-reference build fails before encoding starts.
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    total_frames = features.frame_count
    chunks = list(_zoom_chunks(zooms, keyframe_factor)) if keyframe_mode == "legacy" else []

    native_library = None
    native_references: list[tuple[float, Any]] = []
    active_renderer = renderer
    max_log_zoom = float(np.max(zooms))
    native_backend_id = 0
    native_render_options = NativeRenderOptions()
    if renderer != "python":
        native_library = _get_native_library()
        if native_library is None:
            if native_backend != "auto":
                raise RuntimeError(
                    f"native backend {native_backend} was requested, but the native library is unavailable"
                )
            if renderer == "native":
                raise RuntimeError("native renderer is unavailable; run `make` inside `nix-shell`")
            active_renderer = "python"
        else:
            native_backend_id = _native_backend_id(native_backend, native_library)
            native_render_options = NativeRenderOptions(backend=native_backend_id)
            if native_backend_id == 2 and max_log_zoom >= 6.0:
                raise RuntimeError(
                    "--native-backend opencl currently supports only direct zooms below 1e6"
                )
        if formula != "mandelbrot" and max_log_zoom >= ALTERNATE_PERTURBATION_MIN_LOG:
            # The native reference/BLA context is mathematically specific to
            # z²+c in the Mandelbrot parameter plane. Alternate formulas use
            # the high-precision Python direct renderer instead. Keep the
            # native library alive for palette colourisation: this avoids
            # sending every KFP pixel through Python just because the scalar
            # field itself cannot use Mandelbrot's reference orbit.
            if renderer == "native":
                raise RuntimeError(
                    f"--renderer native does not support deep {formula} yet; "
                    "use --renderer auto or python"
                )
            active_renderer = "python"
            native_references.clear()
            native_backend_id = 0
        if formula == "mandelbrot" and native_library is not None and float(np.max(zooms)) >= 12.0:
            reference_iter = max(
                max_iterations(
                    float(np.max(zooms)),
                    iteration_base,
                    iterations_per_decade,
                    iteration_cap,
                ),
                iteration_base,
            )
            try:
                reference_logs = _native_reference_tier_logs(
                    max_log_zoom,
                    math.log10(max(1.05, float(keyframe_factor))),
                )
                clone_tiers = (
                    len(reference_logs) > 1
                    and hasattr(native_library, "fractal_create_reference_reusable")
                    and hasattr(native_library, "fractal_clone_reference")
                )
                reference_setup_started = time.perf_counter()

                def record_reference(
                    tier_index: int,
                    bla_start_log: float,
                    reference: Any,
                    reused_orbit: bool,
                ) -> None:
                    native_references.append((bla_start_log, reference))
                    print(
                        f"Prepared native reference tier {tier_index}/{len(reference_logs)} "
                        f"(BLA starts at {_zoom_label(bla_start_log)}, "
                        f"{reference_iter} iterations"
                        + (", shared MPFR orbit" if reused_orbit else "")
                        + ").",
                        flush=True,
                    )
                    reference_stats = _native_get_reference_stats(native_library, reference)
                    if reference_stats is not None:
                        print(
                            "Native setup timing: reference "
                            f"{reference_stats['reference_ns'] / 1.0e9:.3f}s, series "
                            f"{reference_stats['series_ns'] / 1.0e9:.3f}s, BLA "
                            f"{reference_stats['bla_ns'] / 1.0e9:.3f}s; "
                            f"series jump {reference_stats['series_iteration']} "
                            f"({reference_stats['series_order']} terms).",
                            flush=True,
                        )

                _, root_reference = _create_native_reference(
                    x_center,
                    y_center,
                    reference_iter,
                    max_log_zoom,
                    series_order,
                    reference_logs[0],
                    reusable=clone_tiers,
                )
                record_reference(1, reference_logs[0], root_reference, False)

                if clone_tiers:
                    setup_workers = max(
                        1,
                        min(len(reference_logs) - 1, native_threads, cpu_count),
                    )
                    print(
                        f"Preparing {len(reference_logs) - 1} radius-specific BLA tiers "
                        f"on {setup_workers} setup workers.",
                        flush=True,
                    )

                    def clone_tier(bla_start_log: float) -> tuple[Any, Optional[RuntimeError]]:
                        try:
                            return (
                                _clone_native_reference(
                                    native_library,
                                    root_reference,
                                    bla_start_log,
                                ),
                                None,
                            )
                        except RuntimeError as error:
                            return None, error

                    with ThreadPoolExecutor(
                        max_workers=setup_workers,
                        thread_name_prefix="fractal-reference",
                    ) as reference_executor:
                        clone_results = list(
                            reference_executor.map(clone_tier, reference_logs[1:])
                        )
                    for tier_index, (bla_start_log, clone_result) in enumerate(
                        zip(reference_logs[1:], clone_results),
                        start=2,
                    ):
                        reference, clone_error = clone_result
                        reused_orbit = clone_error is None
                        if clone_error is not None:
                            print(
                                f"Native tier clone unavailable ({clone_error}); "
                                "rebuilding this tier.",
                                flush=True,
                            )
                            _, reference = _create_native_reference(
                                x_center,
                                y_center,
                                reference_iter,
                                max_log_zoom,
                                series_order,
                                bla_start_log,
                            )
                        record_reference(
                            tier_index,
                            bla_start_log,
                            reference,
                            reused_orbit,
                        )
                else:
                    for tier_index, bla_start_log in enumerate(reference_logs[1:], start=2):
                        _, reference = _create_native_reference(
                            x_center,
                            y_center,
                            reference_iter,
                            max_log_zoom,
                            series_order,
                            bla_start_log,
                        )
                        record_reference(tier_index, bla_start_log, reference, False)
                print(
                    f"Prepared {len(native_references)} depth-safe native reference "
                    f"tier(s) in {time.perf_counter() - reference_setup_started:.2f}s, "
                    f"max zoom {_zoom_label(max_log_zoom)}.",
                    flush=True,
                )
            except RuntimeError as error:
                _destroy_native_references(native_library, native_references)
                if renderer == "native" or float(np.max(zooms)) > 300.0:
                    raise
                print(f"Native deep reference unavailable ({error}); using Python fallback.")
                active_renderer = "python"
                native_library = None
            except BaseException:
                _destroy_native_references(native_library, native_references)
                raise

    cache_identity = _field_renderer_cache_identity(active_renderer)
    cache_identity = (
        f"{cache_identity}-formula-{formula}"
        + (f"-julia-{julia_constant[0]}-{julia_constant[1]}" if formula == "julia" else "")
        + f"-backend-{native_backend_id}"
    )
    colourizer_backend = (
        f"native-{('scalar', 'avx2', 'opencl')[native_backend_id]}"
        if native_library is not None and 0 <= native_backend_id <= 2
        else "python"
    )
    print(
        f"Keyframe source: {render_width}x{render_height} ({quality}, {source_mode}); "
        f"planned keyframes: "
        f"{_keyframe_count(zooms, keyframe_factor) if keyframe_mode == 'legacy' else _atlas_geometry(zooms, keyframe_factor)[2] + 1}; "
        f"mode: {keyframe_mode}; "
        f"field renderer: {active_renderer}; "
        f"colourizer: {colourizer_backend}; "
        f"native threads: {native_threads}; encoder: {selected_codec}; "
        f"encoder threads: {encoder_threads if encoder_threads > 0 else 'auto'}",
        flush=True,
    )

    temporary_output: Optional[Path] = None
    try:
        temporary_output = _reserved_temporary_sibling(output_path, "rendering")
        using_vaapi = selected_codec == "h264_vaapi"
        using_hardware_encoder = selected_codec in {
            "h264_nvenc",
            "h264_qsv",
            "h264_vaapi",
            "h264_videotoolbox",
        }
        video_pixel_format = _video_pixel_format(
            selected_codec,
            palette,
            palette_file,
            lossless,
            crf,
            width=width,
            height=height,
        )
        preset_arguments = ["-preset", selected_preset] if selected_preset else []
        video_filters: list[str] = []
        if (frame_width, frame_height) != (int(width), int(height)):
            video_filters.append(f"scale={int(width)}:{int(height)}:flags={resample}")
        if using_vaapi:
            video_filters.extend(["format=nv12", "hwupload"])
        filter_arguments = ["-vf", ",".join(video_filters)] if video_filters else []
        command = [
            ffmpeg_path,
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            *(["-vaapi_device", "/dev/dri/renderD128"] if using_vaapi else []),
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{frame_width}x{frame_height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            *filter_arguments,
            "-c:v",
            selected_codec,
            *preset_arguments,
            *(["-threads", str(encoder_threads)] if not using_vaapi and encoder_threads > 0 else []),
            *rate_control,
            *([] if using_vaapi else ["-pix_fmt", video_pixel_format]),
            "-c:a",
            "aac",
            # The decoded sample count is rounded up to a video frame.  Padding
            # the audio covers that sub-frame amount, while the explicit output
            # duration below prevents AAC padding from creating a frozen tail.
            "-af",
            "apad",
            "-t",
            f"{features.frame_count / max(float(fps), 1.0):.9f}",
            "-movflags",
            "+faststart",
            str(temporary_output),
        ]
    except BaseException:
        _destroy_native_references(native_library, native_references)
        if temporary_output is not None and temporary_output.exists():
            try:
                temporary_output.unlink()
            except OSError:
                pass
        raise
    assert temporary_output is not None

    if keyframe_mode == "atlas":
        atlas_result = None
        try:
            atlas_result = _render_video_atlas(
                command=command,
                temporary_output=temporary_output,
                output_path=output_path,
                features=features,
                width=width,
                height=height,
                fps=fps,
                keyframe_factor=keyframe_factor,
                x_center=x_center,
                y_center=y_center,
                zooms=zooms,
                iteration_base=iteration_base,
                iterations_per_decade=iterations_per_decade,
                iteration_cap=iteration_cap,
                active_renderer=active_renderer,
                native_library=native_library,
                native_references=native_references,
                native_threads=native_threads,
                native_backend=native_backend_id,
                series_order=series_order,
                series_block=series_block,
                allow_recovery=allow_recovery,
                render_width=render_width,
                render_height=render_height,
                resample=resample,
                palette=palette,
                cache_dir=cache_dir,
                cache_limit_mb=cache_limit_mb,
                durable_cache=durable_cache,
                cache_identity=cache_identity,
                formula=formula,
                julia_constant=julia_constant,
                palette_file=palette_file,
                glow=glow,
                motion_blur=motion_blur,
                hardware_encoder=using_hardware_encoder,
            )
        finally:
            _destroy_native_references(native_library, native_references)
            if temporary_output.exists():
                try:
                    temporary_output.unlink()
                except OSError:
                    pass
        return {
            **(atlas_result or {}),
            "codec": selected_codec,
            "preset": selected_preset,
            "renderer": active_renderer,
            "formula": formula,
            "source_width": render_width,
            "source_height": render_height,
            "frames": total_frames,
        }

    process = None
    ffmpeg_diagnostics: Optional[deque[str]] = None
    ffmpeg_reader: Optional[threading.Thread] = None
    frame_writer: Optional[_FFmpegFrameWriter] = None
    render_started = time.perf_counter()
    keyframe_seconds = 0.0
    frame_seconds = 0.0
    last_progress_report = render_started
    cache_evictor = _CacheEvictor(cache_dir, cache_limit_mb)
    previous_rgb = None

    try:
        process, ffmpeg_diagnostics, ffmpeg_reader = _start_ffmpeg_process(command)
        frame_writer = _FFmpegFrameWriter(
            process,
            ffmpeg_diagnostics,
            ffmpeg_reader,
        )
        chunk_index = 0
        field = None
        field_source_zoom = None
        field_max_zoom = None
        field_iter = None
        while chunk_index < len(chunks):
            chunk_start, chunk_end, chunk_log_zoom, chunk_max_zoom = chunks[chunk_index]

            # Budget for the deepest crop represented by this field, rather
            # than the first frame of the chunk.  This removes the iteration
            # pop that previously occurred exactly when a keyframe changed.
            current_iter = max_iterations(
                chunk_max_zoom,
                iteration_base,
                iterations_per_decade,
                iteration_cap,
            )
            print(
                f"Rendering keyframe at frame {chunk_start}/{total_frames}, "
                f"zoom {_zoom_label(chunk_log_zoom)}, iterations {current_iter}",
                flush=True,
            )
            keyframe_started = time.perf_counter()
            if field is None:
                field = _keyframe_field(
                    cache_dir=cache_dir,
                    cache_identity=cache_identity,
                    renderer=active_renderer,
                    render_width=render_width,
                    render_height=render_height,
                    log_zoom=chunk_log_zoom,
                    x_center=x_center,
                    y_center=y_center,
                    max_iter=current_iter,
                    series_order=series_order,
                    series_block=series_block,
                    native_reference=_select_native_reference(
                        native_references,
                        chunk_log_zoom,
                    ),
                    native_threads=native_threads,
                    render_options=native_render_options,
                    durable_cache=durable_cache,
                    cache_evictor=cache_evictor,
                    formula=formula,
                    julia_constant=julia_constant,
                )
                field_source_zoom = chunk_log_zoom
                field_max_zoom = chunk_max_zoom
                field_iter = current_iter
                _prune_cache(cache_dir, cache_limit_mb, cache_evictor)
            else:
                # The preceding chunk rendered this field ahead of time for
                # a transition. Promote it in memory instead of rendering or
                # loading the same absolute zoom again.
                if (
                    field_source_zoom != chunk_log_zoom
                    or field_max_zoom != chunk_max_zoom
                    or field_iter != current_iter
                ):
                    raise RuntimeError("prefetched keyframe metadata mismatch")
                print("Using prefetched next keyframe.", flush=True)

            next_field = None
            next_log_zoom = None
            next_max_zoom = None
            next_iter = None
            if transition_fraction > 0.0 and chunk_index + 1 < len(chunks):
                _, _, next_log_zoom, next_max_zoom = chunks[chunk_index + 1]
                # Pre-render only when the two source views overlap in world
                # space. In the ordinary forward zoom case the next source is
                # narrower than every current frame, so a crossfade cannot be
                # geometrically aligned and prefetching only increases peak
                # memory and startup latency.
                if next_log_zoom <= chunk_max_zoom:
                    next_iter = max_iterations(
                        next_max_zoom,
                        iteration_base,
                        iterations_per_decade,
                        iteration_cap,
                    )
                    next_field = _keyframe_field(
                        cache_dir=cache_dir,
                        cache_identity=cache_identity,
                        renderer=active_renderer,
                        render_width=render_width,
                        render_height=render_height,
                        log_zoom=next_log_zoom,
                        x_center=x_center,
                        y_center=y_center,
                        max_iter=next_iter,
                        series_order=series_order,
                        series_block=series_block,
                        native_reference=_select_native_reference(
                            native_references,
                            next_log_zoom,
                        ),
                        native_threads=native_threads,
                        render_options=native_render_options,
                        durable_cache=durable_cache,
                        cache_evictor=cache_evictor,
                        formula=formula,
                        julia_constant=julia_constant,
                    )
                    _prune_cache(cache_dir, cache_limit_mb, cache_evictor)
                else:
                    next_log_zoom = None
                    next_max_zoom = None
            keyframe_seconds += time.perf_counter() - keyframe_started

            for frame_index in range(chunk_start, chunk_end):
                frame_started = time.perf_counter()
                frame_log_zoom = float(zooms[frame_index])
                relative_zoom = 10.0 ** (frame_log_zoom - chunk_log_zoom)
                phase = float(features.phase[frame_index])
                gradient = float(features.gradient[frame_index])
                instrumental = float(features.instrumental[frame_index])
                pitch = float(features.pitch[frame_index])
                rgb = _colour_frame(
                    field,
                    frame_width,
                    frame_height,
                    relative_zoom,
                    current_iter,
                    phase,
                    gradient,
                    instrumental,
                    native_library,
                    native_threads,
                    resample,
                    palette,
                    pitch,
                    palette_file,
                )
                if next_field is not None and next_log_zoom is not None and next_iter is not None:
                    progress = (frame_index - chunk_start) / max(
                        chunk_end - chunk_start - 1,
                        1,
                    )
                    blend_start = 1.0 - transition_fraction
                    alpha = np.clip(
                        (progress - blend_start) / transition_fraction,
                        0.0,
                        1.0,
                    )
                    next_relative_zoom = 10.0 ** (frame_log_zoom - next_log_zoom)
                    # A source field can only crop inward. If the next field
                    # is narrower than this frame, wait for overlap instead
                    # of zooming it out and creating a misregistered flash.
                    if alpha > 0.0 and next_relative_zoom >= 1.0:
                        next_rgb = _colour_frame(
                            next_field,
                            frame_width,
                            frame_height,
                            next_relative_zoom,
                            next_iter,
                            phase,
                            gradient,
                            instrumental,
                            native_library,
                            native_threads,
                            resample,
                            palette,
                            pitch,
                            palette_file,
                        )
                        rgb = np.asarray(
                            np.rint(
                                rgb.astype(np.float32) * (1.0 - alpha)
                                + next_rgb.astype(np.float32) * alpha
                            ),
                            dtype=np.uint8,
                        )
                rgb = _apply_frame_effects(rgb, glow, motion_blur, previous_rgb)
                previous_rgb = rgb
                assert frame_writer is not None
                # A bounded writer keeps the compatibility path responsive if
                # FFmpeg stops consuming input while its process remains alive.
                frame_writer.write(rgb)
                frame_seconds += time.perf_counter() - frame_started
                progress_now = time.perf_counter()
                if (
                    frame_index == 0
                    or frame_index + 1 == total_frames
                    or progress_now - last_progress_report
                        >= RENDER_PROGRESS_INTERVAL_SECONDS
                ):
                    print(
                        f"  frame {frame_index + 1}/{total_frames} "
                        f"({100.0 * (frame_index + 1) / total_frames:5.1f}%)",
                        flush=True,
                    )
                    last_progress_report = progress_now

            if next_field is None:
                del field
                field = None
                field_source_zoom = None
                field_max_zoom = None
                field_iter = None
            else:
                # Critical rolling-keyframe reuse: this field becomes the
                # current field for the next chunk without another render.
                field = next_field
                field_source_zoom = next_log_zoom
                field_max_zoom = next_max_zoom
                field_iter = next_iter
            chunk_index += 1

        assert frame_writer is not None
        return_code = frame_writer.finish()
        if return_code != 0:
            raise RuntimeError(_ffmpeg_error_message(
                "ffmpeg exited with an error",
                ffmpeg_diagnostics,
                return_code,
                ffmpeg_reader,
            ))
        os.replace(temporary_output, output_path)
        elapsed = time.perf_counter() - render_started
        print(
            f"Timing: keyframes {keyframe_seconds:.2f}s, frame/crop/pipe "
            f"{frame_seconds:.2f}s, total {elapsed:.2f}s.",
            flush=True,
        )
        result = {
            "keyframe_seconds": float(keyframe_seconds),
            "frame_seconds": float(frame_seconds),
            "encoder_seconds": 0.0,
            "total_seconds": float(elapsed),
        }
    except BrokenPipeError as exc:
        assert process is not None
        _terminate_subprocess(process)
        raise RuntimeError(
            _ffmpeg_error_message(
                "ffmpeg stopped while receiving video frames",
                ffmpeg_diagnostics,
                reader=ffmpeg_reader,
            )
        ) from exc
    except BaseException:
        if process is not None:
            _terminate_subprocess(process)
        raise
    finally:
        if frame_writer is not None:
            frame_writer.abort()
        _destroy_native_references(native_library, native_references)
        if temporary_output.exists():
            try:
                temporary_output.unlink()
            except OSError:
                pass
        if ffmpeg_reader is not None:
            ffmpeg_reader.join(timeout=2.0)
    return {
        **result,
        "codec": selected_codec,
        "preset": selected_preset,
        "renderer": active_renderer,
        "formula": formula,
        "source_width": render_width,
        "source_height": render_height,
        "frames": total_frames,
    }


def _print_profiles() -> None:
    print("Built-in profiles:")
    for name in CANONICAL_PROFILE_CHOICES:
        print(f"  {name:<12} {PROFILE_DESCRIPTIONS[name]}")
    print("Fast compatibility profiles:")
    for name in FAST_PROFILE_CHOICES:
        print(f"  {name:<12} {PROFILE_DESCRIPTIONS[name]}")
    print("Compatibility aliases:")
    for alias, canonical in PROFILE_ALIASES.items():
        print(f"  {alias:<20} {canonical}")


def _print_formulas() -> None:
    print("Supported formulas:")
    for name in FORMULA_CHOICES:
        print(f"  {name:<14} {FORMULA_DESCRIPTIONS[name]}")


def _manifest_path(
    output_path: Path,
    requested: Optional[Path],
    disabled: bool,
) -> Optional[Path]:
    if disabled:
        return None
    if requested is not None:
        _reject_final_symlink(requested, "manifest")
        return _absolute_path(requested)
    _reject_final_symlink(output_path, "output")
    return _absolute_path(output_path).with_suffix(".json")


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _json_safe_arguments(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    """Atomically write a human-readable sidecar without partial JSON."""

    path = _absolute_path(Path(path))
    _reject_final_symlink(path, "manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _build_manifest(
    args: argparse.Namespace,
    x_center: str,
    y_center: str,
    selected_point: Optional[DeepZoomPoint],
    max_log_zoom: float,
    features: AudioFeatures,
    zooms: Any,
) -> dict[str, Any]:
    np = _require_numpy()
    point: dict[str, Any] = {
        "formula": args.formula,
        "x": x_center,
        "y": y_center,
        "catalogue_slug": selected_point.slug if selected_point else None,
        "catalogue_name": selected_point.name if selected_point else None,
        "catalogue_source": selected_point.source_name if selected_point else None,
        "preset_julia_c": list(selected_point.julia_c) if selected_point and selected_point.julia_c else None,
    }
    try:
        audio_size = args.audio.stat().st_size
    except OSError:
        audio_size = None
    return {
        "schema": "fractal-audio-viz.render-manifest.v1",
        "status": "running",
        "started_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "command": list(sys.argv),
        "git_revision": _git_revision(),
        "profile": args.profile,
        "formula": args.formula,
        "julia_c": args.julia_c,
        "audio": {"path": str(args.audio), "bytes": audio_size},
        "output": str(args.output),
        "point": point,
        "zoom": {
            "base": str(args.base_zoom),
            "max": str(args.max_zoom),
            "max_log10": float(max_log_zoom),
            "planned_min_log10": float(np.min(zooms)),
            "planned_max_log10": float(np.max(zooms)),
        },
        "frames": {
            "count": int(features.frame_count),
            "fps": int(args.fps),
            "duration_seconds": float(features.frame_count / max(args.fps, 1)),
        },
        "audio_controls": {
            "separation": args.separation,
            "beat_strength": float(args.beat_strength),
            "onset_sync": bool(args.beat_strength != 0.0),
        },
        "settings": _json_safe_arguments(args),
    }


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", default=DEFAULT_AUDIO, type=Path)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=DEFAULT_PROFILE,
        help=(
            "start from a named preset; defaults to the native-density "
            f"{DEFAULT_PROFILE} quality profile"
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="list built-in profiles and exit",
    )
    parser.add_argument("--width", type=int, default=1920, help="output width in pixels")
    parser.add_argument("--height", type=int, default=1080, help="output height in pixels")
    parser.add_argument("--fps", type=int, default=30, help="output frames per second")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument(
        "--separation",
        choices=("auto", "demucs", "spectral", "none"),
        default="auto",
        help=(
            "auto uses Demucs when available and otherwise full-mix control; "
            "spectral explicitly enables frequency-band proxies"
        ),
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=1.0,
        help="keyframe resolution multiplier; 1.0 is native output resolution",
    )
    parser.add_argument(
        "--fractal-scale",
        type=float,
        default=1.0,
        help="minimum fractal source resolution multiplier; draft mode permits undersampling",
    )
    parser.add_argument(
        "--quality",
        choices=("draft", "balanced", "quality", "extreme"),
        default="balanced",
        help=(
            "draft is fastest; balanced uses output-resolution nested tiles; "
            "quality preserves source scale; extreme supersamples modestly"
        ),
    )
    parser.add_argument(
        "--source-mode",
        choices=SOURCE_MODE_CHOICES,
        default="native",
        help=(
            "fractal source density: native keeps at least output resolution; "
            "upscaled renders a quarter-size source and enlarges it for a faster export"
        ),
    )
    parser.add_argument(
        "--upscaling",
        dest="source_mode",
        action="store_const",
        const="upscaled",
        default=argparse.SUPPRESS,
        help="shorthand for --source-mode upscaled",
    )
    parser.add_argument(
        "--keyframe-factor",
        type=float,
        default=2.0,
        help="maximum zoom change between high-resolution keyframes",
    )
    parser.add_argument(
        "--keyframe-mode",
        choices=("atlas", "legacy"),
        default="atlas",
        help="use reusable nested tiles (atlas) or the previous full-field chunk renderer",
    )
    parser.add_argument(
        "--point",
        default=None,
        help=(
            "curated point slug, 'random', or an exact REAL,IMAG pair "
            "(for a negative pair use --point=-0.7,0.1)"
        ),
    )
    parser.add_argument(
        "--random-point",
        action="store_true",
        help="choose a curated point that safely supports --max-zoom",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="make random point selection reproducible",
    )
    parser.add_argument(
        "--list-points",
        action="store_true",
        help="list the curated deep-zoom catalogue and exit",
    )
    parser.add_argument(
        "--list-formulas",
        action="store_true",
        help="list Mandelbrot-family formulas and exit",
    )
    parser.add_argument(
        "--formula",
        choices=FORMULA_CHOICES,
        default="mandelbrot",
        help="fractal family; alternate formulas use the direct/high-precision Python path",
    )
    parser.add_argument(
        "--julia-c",
        default=f"{DEFAULT_JULIA_C[0]},{DEFAULT_JULIA_C[1]}",
        help="Julia constant as REAL,IMAG when --formula julia is selected",
    )
    parser.add_argument(
        "--x-center",
        default=None,
        help="exact custom real coordinate; requires --y-center",
    )
    parser.add_argument(
        "--y-center",
        default=None,
        help="exact custom imaginary coordinate; requires --x-center",
    )
    parser.add_argument("--base-zoom", default="1.0", help="starting zoom, e.g. 1e0")
    parser.add_argument(
        "--max-zoom",
        default="1e32",
        help="final zoom; native scaled arithmetic supports decimal exponents up to about 9800",
    )
    parser.add_argument(
        "--allow-underspecified-center",
        action="store_true",
        help=(
            "allow a deep render when the supplied centre has too few decimal places; "
            "for exploratory output only, because the path may differ from the intended target"
        ),
    )
    parser.add_argument(
        "--iteration-base",
        type=int,
        default=384,
        help="minimum iterations for shallow frames",
    )
    parser.add_argument(
        "--iterations-per-decade",
        type=int,
        default=500,
        help="additional iterations per decimal zoom; 500 suits the bundled deep centre",
    )
    parser.add_argument(
        "--iteration-cap",
        type=int,
        default=100000,
        help="maximum iteration budget; increase for unusually stubborn deep interiors",
    )
    parser.add_argument(
        "--zoom-punch",
        type=float,
        default=3.0,
        help="loudness contrast in zoom speed; larger values make loud beats punch harder",
    )
    parser.add_argument(
        "--zoom-speed",
        type=float,
        default=-0.04,
        help=(
            "quiet-time logarithmic zoom velocity; the slightly negative default "
            "lets the camera pull back between loud punches"
        ),
    )
    parser.add_argument(
        "--beat-strength",
        type=float,
        default=0.0,
        help="additional onset/beat contribution to zoom speed; 0 keeps loudness-only motion",
    )
    parser.add_argument(
        "--attack",
        type=float,
        default=0.025,
        help="audio-control attack time in seconds; lower is more percussive",
    )
    parser.add_argument(
        "--release",
        type=float,
        default=0.12,
        help="audio-control release time in seconds; higher gives calmer motion",
    )
    parser.add_argument(
        "--series-order",
        type=int,
        default=3,
        help="native BLA polynomial degree (1-3; values 4-32 remain ABI-compatible)",
    )
    parser.add_argument(
        "--series-block",
        type=int,
        default=256,
        help=(
            "requested BLA block length; 256 is the safe default for deep video tiles; "
            "larger values can be faster at benign locations but may amplify glitches"
        ),
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "native", "python"),
        default="auto",
        help="use the MPFR/OpenMP C-ABI renderer when available",
    )
    parser.add_argument(
        "--native-threads",
        type=int,
        default=0,
        help="OpenMP threads for the native renderer; 0 uses the runtime default",
    )
    parser.add_argument(
        "--native-backend",
        choices=("auto", "scalar", "avx2", "opencl"),
        default="auto",
        help=(
            "native field backend: auto selects AVX2 when available; opencl is "
            "an opt-in direct/shallow preview backend"
        ),
    )
    parser.add_argument(
        "--video-preset",
        choices=VIDEO_PRESET_CHOICES,
        default="ultrafast",
        help="x264 speed/size tradeoff; ultrafast is the low-power default",
    )
    parser.add_argument(
        "--codec",
        default="auto",
        help=(
            "FFmpeg video encoder name, or auto to detect a usable NVENC/QSV/"
            "VAAPI/VideoToolbox encoder and otherwise use libx264"
        ),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="constant-rate-factor passed to the selected video encoder (0-51)",
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help=(
            "use lossless H.264 rate control where the selected encoder supports it; "
            "also preserves 4:4:4 chroma"
        ),
    )
    parser.add_argument(
        "--resample",
        choices=("lanczos", "bilinear"),
        default="bilinear",
        help="crop resize filter; bilinear is fastest, lanczos is sharper",
    )
    parser.add_argument(
        "--palette",
        choices=PALETTE_CHOICES,
        default="aurora",
        help="colour palette; kalles-default matches Kalles Fraktaler's default stops",
    )
    parser.add_argument(
        "--palette-file",
        type=Path,
        default=None,
        help="optional .txt or KFP colour-stop file, interpolated per frame",
    )
    parser.add_argument(
        "--glow",
        type=float,
        default=0.0,
        help="subtle low-resolution bloom amount after colourisation (0-1)",
    )
    parser.add_argument(
        "--motion-blur",
        type=float,
        default=0.0,
        help="blend each frame with the previous frame (0-<1; adds compositor work)",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=0,
        help="FFmpeg encoder threads; 0 lets FFmpeg choose, lower values reduce CPU contention",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="optional directory for reusable .npy keyframes",
    )
    parser.add_argument(
        "--cache-limit-mb",
        type=float,
        default=0.0,
        help="optional maximum cache size in MB; 0 keeps all cache entries",
    )
    parser.add_argument(
        "--durable-cache",
        action="store_true",
        help="fsync each cache tile before replacement; slower but safer after power loss",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON render manifest path; defaults to OUTPUT with a .json suffix",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="disable the automatic JSON sidecar manifest",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="analyse audio and print keyframe workload without rendering",
    )
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--profile", choices=PROFILE_CHOICES, default=None)
    probe_argv = sys.argv[1:] if argv is None else argv
    selected = probe.parse_known_args(probe_argv)[0].profile or DEFAULT_PROFILE
    parser.set_defaults(profile=selected, **PROFILE_DEFAULTS[selected])
    return parser


def _main_impl() -> None:
    args = build_parser(sys.argv[1:]).parse_args()
    if args.list_profiles:
        _print_profiles()
        return
    if args.list_formulas:
        _print_formulas()
        return
    try:
        args.formula = _formula_name(args.formula)
        julia_constant = _parse_coordinate_pair(args.julia_c, "--julia-c")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        args.width, args.height = _validate_dimensions(args.width, args.height, "video")
        args.fps = _validate_fps(args.fps)
        args.sample_rate = _index_value(args.sample_rate, "sample rate")
        if args.sample_rate <= 0 or args.sample_rate > MAX_SAMPLE_RATE:
            raise ValueError(f"sample rate must be between 1 and {MAX_SAMPLE_RATE:,}")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        base_log = _zoom_log(args.base_zoom)
        max_log = _zoom_log(args.max_zoom)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if max_log < base_log:
        raise SystemExit("max-zoom must be greater than or equal to base-zoom")
    if max_log > MAX_LOG10_ZOOM:
        raise SystemExit("max-zoom is beyond the native scaled-exponent range; use a zoom below 1e9800")
    if args.list_points:
        _print_deep_zoom_points(args.formula)
        return
    if args.formula != "mandelbrot":
        if max_log > 300.0:
            raise SystemExit(
                "alternate formulas currently support Python high-precision views up to 1e300; "
                "the native e150+BLA path is reserved for Mandelbrot"
            )
    try:
        args.x_center, args.y_center, selected_point = _resolve_render_point(
            point_spec=args.point,
            random_point=args.random_point,
            x_center=args.x_center,
            y_center=args.y_center,
            random_seed=args.random_seed,
            max_log_zoom=max_log,
            formula=args.formula,
            julia_constant=julia_constant,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if (
        args.formula == "julia"
        and selected_point is not None
        and selected_point.julia_c is not None
        and args.julia_c == f"{DEFAULT_JULIA_C[0]},{DEFAULT_JULIA_C[1]}"
    ):
        # A Julia preset owns both its viewport point and its fixed c.  An
        # explicitly different --julia-c remains authoritative.
        args.julia_c = ",".join(selected_point.julia_c)
        julia_constant = selected_point.julia_c
    if selected_point is not None:
        selection_kind = (
            "Random point"
            if args.random_point
            or (args.point is not None and args.point.casefold() == "random")
            else "Point"
        )
        depth = (
            f"stored-safe depth ~1e{_deep_point_max_log10_zoom(selected_point):.0f}."
        )
        print(
            f"{selection_kind}: {selected_point.name} ({selected_point.slug}); {depth}",
            flush=True,
        )
    center_error = _center_precision_error(args.x_center, args.y_center, max_log)
    if center_error is not None:
        if not args.allow_underspecified_center:
            raise SystemExit(center_error)
        print(f"WARNING: {center_error}", file=sys.stderr, flush=True)
    finite_options = (
        ("render-scale", args.render_scale),
        ("fractal-scale", args.fractal_scale),
        ("keyframe-factor", args.keyframe_factor),
        ("zoom-punch", args.zoom_punch),
        ("zoom-speed", args.zoom_speed),
        ("beat-strength", args.beat_strength),
        ("attack", args.attack),
        ("release", args.release),
        ("cache-limit-mb", args.cache_limit_mb),
        ("glow", args.glow),
        ("motion-blur", args.motion_blur),
    )
    for option_name, option_value in finite_options:
        if not math.isfinite(option_value):
            raise SystemExit(f"{option_name} must be finite")
    if args.render_scale < 1.0:
        raise SystemExit("render-scale must be at least 1")
    if args.fractal_scale <= 0:
        raise SystemExit("fractal-scale must be positive")
    if args.keyframe_factor <= 1.0:
        raise SystemExit("keyframe-factor must be greater than 1")
    try:
        args.iteration_base = _validate_iteration_count(args.iteration_base, "iteration-base")
        args.iteration_cap = _validate_iteration_count(args.iteration_cap, "iteration-cap")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not 0 <= args.iterations_per_decade <= MAX_ITERATION_BUDGET:
        raise SystemExit(
            f"iterations-per-decade must be between 0 and {MAX_ITERATION_BUDGET:,}"
        )
    if args.iteration_cap < args.iteration_base:
        raise SystemExit("iteration-cap must be greater than iteration-base")
    if args.zoom_punch < 0:
        raise SystemExit("zoom-punch cannot be negative")
    if args.beat_strength < 0:
        raise SystemExit("beat-strength cannot be negative")
    if args.attack < 0 or args.release < 0:
        raise SystemExit("attack and release cannot be negative")
    if not 0.0 <= args.glow <= 1.0:
        raise SystemExit("glow must be between 0 and 1")
    if not 0.0 <= args.motion_blur < 1.0:
        raise SystemExit("motion-blur must be between 0 (off) and 1")
    if not 1 <= args.series_order <= 32:
        raise SystemExit("series-order must be between 1 and 32")
    if not 2 <= args.series_block <= 4096:
        raise SystemExit("series-block must be between 2 and 4096")
    try:
        args.native_threads = _validate_thread_count(args.native_threads, "native-threads")
        args.encoder_threads = _validate_thread_count(args.encoder_threads, "encoder-threads")
        _validate_ffmpeg_token(args.codec, "codec")
        _validate_ffmpeg_token(args.video_preset, "video-preset")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not 0 <= args.crf <= 51:
        raise SystemExit("crf must be between 0 and 51")
    if args.cache_limit_mb < 0:
        raise SystemExit("cache-limit-mb cannot be negative")
    if args.sample_rate <= 0 or args.sample_rate > MAX_SAMPLE_RATE:
        raise SystemExit(f"sample-rate must be between 1 and {MAX_SAMPLE_RATE:,}")
    try:
        # Keep the final destination lexical. Resolving an output symlink here
        # would make os.replace overwrite the symlink's target instead of
        # replacing the link itself, which is an unsafe and surprising write.
        _reject_final_symlink(args.output, "output")
        if args.manifest is not None:
            _reject_final_symlink(args.manifest, "manifest")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.audio = _normalise_path(args.audio)
    args.output = _absolute_path(args.output)
    if args.cache_dir is not None:
        args.cache_dir = _normalise_path(args.cache_dir)
    if args.palette_file is not None:
        args.palette_file = _normalise_path(args.palette_file)
        if not args.palette_file.is_file():
            raise SystemExit(f"palette file not found: {args.palette_file}")
    manifest_path = _manifest_path(args.output, args.manifest, args.no_manifest)
    try:
        args.audio, args.output, manifest_path, args.cache_dir = _validate_render_paths(
            args.audio,
            args.output,
            manifest_path,
            args.cache_dir,
        )
        if args.palette_file is not None and _paths_refer_to_same_target(
            args.palette_file, args.output
        ):
            raise ValueError("palette file must be different from the output video")
        if args.palette_file is not None and _paths_refer_to_same_target(
            args.palette_file, args.audio
        ):
            raise ValueError("palette file must be different from the input audio file")
        if args.palette_file is not None and manifest_path is not None and _paths_refer_to_same_target(
            args.palette_file, manifest_path
        ):
            raise ValueError("palette file must be different from the manifest file")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    audio_started = time.perf_counter()
    features = analyse_audio(
        args.audio,
        args.sample_rate,
        args.fps,
        args.separation,
        args.cache_dir,
        args.attack,
        args.release,
    )
    print(f"Audio analysis: {time.perf_counter() - audio_started:.2f}s", flush=True)
    zooms = _zoom_plan(
        features.instrumental,
        args.base_zoom,
        args.max_zoom,
        args.zoom_punch,
        args.zoom_speed,
        features.onset,
        args.beat_strength,
    )
    print(
        f"{features.frame_count} frames ({features.frame_count / args.fps:.1f}s) -> "
        f"{args.output}"
    )
    if args.estimate:
        estimate_scale, _ = _quality_source_settings(
            args.fractal_scale,
            args.quality,
            args.keyframe_mode,
            args.keyframe_factor,
            args.source_mode,
        )
        try:
            render_width, render_height = _scaled_dimensions(
                args.width,
                args.height,
                args.render_scale * estimate_scale,
                "fractal source",
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        planned_units = (
            _atlas_geometry(zooms, args.keyframe_factor)[2] + 1
            if args.keyframe_mode == "atlas"
            else _keyframe_count(zooms, args.keyframe_factor)
        )
        print(
            f"Estimate: {render_width}x{render_height} fractal source, "
            f"{planned_units} {args.keyframe_mode} levels, "
            f"{args.video_preset} encoder, {args.quality} quality."
        )
        return
    manifest = _build_manifest(
        args,
        args.x_center,
        args.y_center,
        selected_point,
        max_log,
        features,
        zooms,
    )
    if manifest_path is not None:
        _write_manifest(manifest_path, manifest)
    try:
        render_result = render_video(
            args.audio,
            args.output,
            features,
            args.width,
            args.height,
            args.fps,
            args.render_scale,
            args.fractal_scale,
            args.keyframe_factor,
            args.x_center,
            args.y_center,
            zooms,
            args.iteration_base,
            args.iterations_per_decade,
            args.iteration_cap,
            args.renderer,
            args.native_threads,
            args.series_order,
            args.series_block,
            args.video_preset,
            args.resample,
            args.encoder_threads,
            args.cache_dir,
            args.quality,
            args.palette,
            args.cache_limit_mb,
            args.codec,
            args.crf,
            args.keyframe_mode,
            args.durable_cache,
            args.native_backend,
            args.formula,
            julia_constant,
            args.palette_file,
            args.glow,
            args.motion_blur,
            args.lossless,
            args.source_mode,
        )
    except Exception as error:
        if manifest_path is not None:
            manifest.update({
                "status": "failed",
                "finished_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
                "error": str(error),
            })
            _write_manifest(manifest_path, manifest)
        raise
    if manifest_path is not None:
        manifest.update({
            "status": "complete",
            "finished_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "timing": render_result or {},
        })
        _write_manifest(manifest_path, manifest)
    print(f"Done -> {args.output}")


def main() -> None:
    """Run the CLI with concise errors instead of implementation tracebacks."""

    try:
        _main_impl()
    except SystemExit:
        raise
    except (OSError, RuntimeError, ValueError, OverflowError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
