#!/usr/bin/env python3
"""Low-latency, audio-reactive GTK live view.

The export renderer is intentionally built for quality and reproducibility.
That makes it the wrong thing to run once per screen-refresh in a
screensaver-like preview.  This module renders a small ladder of native
scalar fields, colourises a continuous parent/child atlas between them, and
keeps the audio decoder, renderer, and GTK main loop separate.  The result
starts quickly, remains responsive, and is smoothly upscaled by the window
when it is fullscreen.  The ladder follows the selected base/max zoom range
and the song resets to the base view when it reaches the selected maximum.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import visualizer


LIVE_AUDIO_SAMPLE_RATE = 8_000
LIVE_DEFAULT_WIDTH = 640
LIVE_DEFAULT_HEIGHT = 360
LIVE_DEFAULT_FPS = 30
LIVE_MAX_FPS = 60
# Live view deliberately uses the old fast-render density, reduced one more
# step for a screensaver. GTK/Cairo performs the only upscale to the actual
# window or monitor; no 4K surface is allocated for the live path.
LIVE_NATIVE_MAX_WIDTH = 480
LIVE_NATIVE_MAX_HEIGHT = 270
LIVE_PYTHON_MAX_WIDTH = 240
LIVE_PYTHON_MAX_HEIGHT = 135
LIVE_DEFAULT_MAX_ZOOM = "1e4"
# The live view is a preview, but it should still follow normal deep-zoom
# selections (including the GUI's usual e150 range).  Beyond e300 the Python
# alternate-formula path is deliberately not made a blocking screensaver.
LIVE_MAX_PREVIEW_LOG_ZOOM = 300.0
# Keep live view's atlas at about 0.75 decades per replacement: that gives it
# enough intermediate coverage to hide tile changes without paying for export
# resolution. Its fields stay at the smaller screensaver source size above.
# The first fields are still shown immediately and later fields are built in
# the background.
LIVE_MAX_SOURCE_KEYFRAMES = 168
LIVE_SOURCE_LOG_STEP = 0.75
# Render only the first few sources before opening the window. The remaining
# ladder is filled by a worker while audio and display playback are already
# running; a screensaver should never wait for the deepest source set.
LIVE_INITIAL_SOURCE_COUNT = 3
# Keep the old name as the minimum for callers/tests that used the original
# fixed-budget preview.  Deep alternate formulas need a larger budget: a
# fixed 192-iteration cap classifies an e150 boundary tile as entirely
# interior, which is indistinguishable from a black rendering failure.
LIVE_ITERATIONS = 192
LIVE_MIN_ITERATIONS = LIVE_ITERATIONS
LIVE_MAX_ITERATIONS = 4096
LIVE_ITERATION_QUANTUM = 32
LIVE_FRAME_READ_BYTES = 256 * 1024
LIVE_FALLBACK_DURATION = 300.0
LIVE_DEFAULT_NATIVE_THREADS = 4
# Source fields are prepared in the background. When playback catches the
# builder, let the latest field carry the camera only to the next requested
# source boundary, and approach that boundary at a bounded rate. Without
# this, the camera freezes at the last completed field and then jumps when the
# worker appends the next one, which looks like an atlas-tile slideshow.
LIVE_MAX_ZOOM_RATE = 2.5


class _LiveCancelled(Exception):
    """Internal signal used when a live window is closed during decoding."""


@dataclass(frozen=True)
class LiveViewConfig:
    """Validated settings for one live view window."""

    audio_path: Path
    formula: str
    x_center: str
    y_center: str
    julia_constant: tuple[str, str] = visualizer.DEFAULT_JULIA_C
    palette: str = "aurora"
    palette_file: Optional[Path] = None
    width: int = LIVE_DEFAULT_WIDTH
    height: int = LIVE_DEFAULT_HEIGHT
    fps: int = LIVE_DEFAULT_FPS
    native_threads: int = 0
    loop: bool = True
    fullscreen: bool = True
    base_zoom: str = "1.0"
    max_zoom: str = LIVE_DEFAULT_MAX_ZOOM

    def __post_init__(self) -> None:
        audio_path = Path(self.audio_path).expanduser()
        if not audio_path.is_file():
            raise ValueError(f"audio file not found: {audio_path}")
        object.__setattr__(self, "audio_path", audio_path.resolve())
        formula = visualizer._formula_name(self.formula)
        object.__setattr__(self, "formula", formula)
        object.__setattr__(self, "x_center", visualizer._validate_center_text(self.x_center, "real"))
        object.__setattr__(self, "y_center", visualizer._validate_center_text(self.y_center, "imaginary"))
        base_zoom = str(self.base_zoom).strip()
        max_zoom = str(self.max_zoom).strip()
        base_log_zoom = visualizer._zoom_log(base_zoom)
        max_log_zoom = visualizer._zoom_log(max_zoom)
        if max_log_zoom < base_log_zoom:
            raise ValueError("live max zoom must be greater than or equal to live base zoom")
        if base_log_zoom > LIVE_MAX_PREVIEW_LOG_ZOOM:
            raise ValueError(
                f"live base zoom cannot exceed 1e{LIVE_MAX_PREVIEW_LOG_ZOOM:.0f}"
            )
        precision_error = visualizer._center_precision_error(
            self.x_center,
            self.y_center,
            min(max_log_zoom, LIVE_MAX_PREVIEW_LOG_ZOOM),
        )
        if precision_error is not None:
            raise ValueError(precision_error)
        object.__setattr__(self, "base_zoom", base_zoom)
        object.__setattr__(self, "max_zoom", max_zoom)
        if len(self.julia_constant) != 2:
            raise ValueError("Julia constant must contain real and imaginary coordinates")
        julia_constant = (
            visualizer._validate_center_text(self.julia_constant[0], "Julia real"),
            visualizer._validate_center_text(self.julia_constant[1], "Julia imaginary"),
        )
        object.__setattr__(self, "julia_constant", julia_constant)
        if self.palette not in visualizer.PALETTE_CHOICES:
            raise ValueError(f"unknown palette: {self.palette}")
        if self.palette_file is not None:
            palette_file = Path(self.palette_file).expanduser()
            if not palette_file.is_file():
                raise ValueError(f"palette file not found: {palette_file}")
            object.__setattr__(
                self,
                "palette_file",
                palette_file.resolve(),
            )
        width, height = live_dimensions(self.width, self.height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        try:
            fps = int(self.fps)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("live fps must be an integer") from error
        if not 1 <= fps <= LIVE_MAX_FPS:
            raise ValueError(f"live fps must be between 1 and {LIVE_MAX_FPS}")
        object.__setattr__(self, "fps", fps)
        try:
            native_threads = int(self.native_threads)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("live native threads must be an integer") from error
        if native_threads < 0:
            raise ValueError("live native threads cannot be negative")
        if native_threads > visualizer.MAX_THREAD_COUNT:
            raise ValueError(
                f"live native threads must be at most {visualizer.MAX_THREAD_COUNT}"
            )
        object.__setattr__(self, "native_threads", native_threads)

    @property
    def base_log_zoom(self) -> float:
        return visualizer._zoom_log(self.base_zoom)

    @property
    def max_log_zoom(self) -> float:
        return visualizer._zoom_log(self.max_zoom)

    @property
    def preview_max_log_zoom(self) -> float:
        return max(self.base_log_zoom, min(self.max_log_zoom, LIVE_MAX_PREVIEW_LOG_ZOOM))

    @property
    def preview_zoom_is_capped(self) -> bool:
        return self.preview_max_log_zoom < self.max_log_zoom - 1.0e-9


@dataclass(frozen=True)
class LiveAudioTrack:
    """Small frame-aligned control arrays used by the live compositor.

    ``zoom`` stores base-10 logarithms, rather than an enormous binary zoom
    factor.  This keeps e150 selections finite and lets the compositor choose
    the correct absolute-zoom source tile without losing precision.
    """

    energy: Any
    onset: Any
    phase: Any
    zoom: Any
    duration: float
    fps: int


@dataclass(frozen=True)
class LiveZoomSources:
    """Absolute-zoom scalar fields used by the live parent/child compositor."""

    log_zooms: Any
    fields: tuple[Any, ...]
    iteration_caps: tuple[int, ...] = ()
    capped: bool = False


@dataclass
class LiveZoomSourceStore:
    """Thread-safe, incrementally populated source ladder for live playback."""

    requested_logs: Any
    _log_zooms: list[float] = field(default_factory=list, init=False, repr=False)
    _fields: list[Any] = field(default_factory=list, init=False, repr=False)
    _iteration_caps: list[int] = field(default_factory=list, init=False, repr=False)
    _capped: bool = field(default=False, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)
    _error: Optional[BaseException] = field(default=None, init=False, repr=False)
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
        repr=False,
    )

    @property
    def count(self) -> int:
        with self._condition:
            return len(self._fields)

    @property
    def finished(self) -> bool:
        with self._condition:
            return self._finished

    @property
    def error(self) -> Optional[BaseException]:
        with self._condition:
            return self._error

    def append(self, log_zoom: float, field_value: Any, iteration_cap: int) -> None:
        with self._condition:
            self._log_zooms.append(float(log_zoom))
            self._fields.append(field_value)
            self._iteration_caps.append(int(iteration_cap))
            self._condition.notify_all()

    def finish(self, *, capped: bool = False) -> None:
        with self._condition:
            self._capped = self._capped or bool(capped)
            self._finished = True
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            self._error = error
            self._finished = True
            self._condition.notify_all()

    def display_zoom_limit(self) -> Optional[float]:
        """Return the furthest zoom the current source set can display.

        While the ladder is still being built, the next requested source is a
        safe look-ahead boundary: the current last field can be continuously
        cropped up to that point. Once building finishes (including a
        deliberate deep-zoom cap), never extrapolate beyond the last valid
        field.
        """

        with self._condition:
            if not self._fields:
                return None
            last_log = float(self._log_zooms[-1])
            if not self._finished and len(self._fields) < len(self.requested_logs):
                try:
                    next_log = float(self.requested_logs[len(self._fields)])
                except (IndexError, TypeError, ValueError, OverflowError):
                    next_log = last_log
                if math.isfinite(next_log) and next_log > last_log:
                    return next_log
            return last_log

    def snapshot(self) -> LiveZoomSources:
        np = visualizer._require_numpy()
        with self._condition:
            if not self._fields:
                if self._error is not None:
                    raise self._error
                raise RuntimeError("live zoom source ladder is empty")
            return LiveZoomSources(
                np.asarray(tuple(self._log_zooms), dtype=np.float64),
                tuple(self._fields),
                tuple(self._iteration_caps),
                self._capped,
            )


def live_dimensions(
    width: int,
    height: int,
    *,
    native_available: bool = True,
) -> tuple[int, int]:
    """Return a bounded 16:9-ish source size that the live view can upscale.

    The requested dimensions describe the window/aspect ratio, not a promise
    to calculate a full 4K field every 1/30 second. Native colourisation uses
    a 480x270 ceiling on the supported machines; the Python fallback is capped
    at 240x135 so a missing native library cannot make the GUI appear hung.
    """

    try:
        requested_width = int(width)
        requested_height = int(height)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("live dimensions must be integers") from error
    if requested_width <= 0 or requested_height <= 0:
        raise ValueError("live dimensions must be positive")
    max_width = LIVE_NATIVE_MAX_WIDTH if native_available else LIVE_PYTHON_MAX_WIDTH
    max_height = LIVE_NATIVE_MAX_HEIGHT if native_available else LIVE_PYTHON_MAX_HEIGHT
    scale = min(1.0, max_width / requested_width, max_height / requested_height)
    result_width = max(1, int(round(requested_width * scale)))
    result_height = max(1, int(round(requested_height * scale)))
    return result_width, result_height


def live_zoom_ladder(base_zoom: Any, max_zoom: Any) -> Any:
    """Return a bounded absolute-zoom ladder for a live preview.

    The export atlas can contain thousands of levels. A live window needs a
    small fixed setup cost, so it samples the selected logarithmic range at
    no more than ``LIVE_MAX_SOURCE_KEYFRAMES`` levels and composes the
    in-between frames continuously.  The final source is capped at e300,
    where the native Mandelbrot path and the bounded Python fallback remain
    practical for an interactive preview.
    """

    np = visualizer._require_numpy()
    base_log_zoom = visualizer._zoom_log(base_zoom)
    max_log_zoom = visualizer._zoom_log(max_zoom)
    if max_log_zoom < base_log_zoom:
        raise ValueError("live max zoom must be greater than or equal to live base zoom")
    if base_log_zoom > LIVE_MAX_PREVIEW_LOG_ZOOM:
        raise ValueError(
            f"live base zoom cannot exceed 1e{LIVE_MAX_PREVIEW_LOG_ZOOM:.0f}"
        )
    preview_max_log_zoom = max(
        base_log_zoom,
        min(max_log_zoom, LIVE_MAX_PREVIEW_LOG_ZOOM),
    )
    if preview_max_log_zoom <= base_log_zoom:
        return np.asarray([base_log_zoom], dtype=np.float64)
    interval_count = max(
        1,
        int(math.ceil((preview_max_log_zoom - base_log_zoom) / LIVE_SOURCE_LOG_STEP)),
    )
    point_count = min(
        LIVE_MAX_SOURCE_KEYFRAMES,
        max(2, interval_count + 1),
    )
    return np.linspace(
        base_log_zoom,
        preview_max_log_zoom,
        point_count,
        dtype=np.float64,
    )


def _live_zoom_text(log_zoom: float) -> str:
    """Format one live ladder logarithm without overflowing a float zoom."""

    return visualizer._zoom_text(float(log_zoom)).decode("ascii")


def live_iteration_cap(formula: str, log_zoom: float) -> int:
    """Choose a bounded live iteration budget for one absolute-zoom tile.

    Iteration depth is independent of coordinate precision, but a boundary
    point can require substantially more iterations as the viewport narrows.
    Kalles uses the exact iteration cap as the interior marker; leaving every
    live tile at 192 therefore turns valid deep alternate-formula views into
    a solid black rectangle.  The schedule is deliberately much smaller than
    export defaults and is quantised so neighbouring tiles have stable cache
    and colourisation behaviour.
    """

    formula = visualizer._formula_name(formula)
    try:
        log_zoom = float(log_zoom)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("live source zoom must be numeric") from error
    if not math.isfinite(log_zoom):
        raise ValueError("live source zoom must be finite")
    log_zoom = max(0.0, log_zoom)
    if formula == "mandelbrot":
        base, per_decade = 192.0, 2.5
    elif formula == "burning-ship":
        base, per_decade = 256.0, 3.5
    else:
        # Julia and Tricorn boundary orbits are more likely to have a long
        # preperiod than the Mandelbrot reference path.
        base, per_decade = 256.0, 5.0
    requested = max(LIVE_MIN_ITERATIONS, int(math.ceil(base + per_decade * log_zoom)))
    quantised = int(
        math.ceil(requested / LIVE_ITERATION_QUANTUM) * LIVE_ITERATION_QUANTUM
    )
    return min(LIVE_MAX_ITERATIONS, max(LIVE_MIN_ITERATIONS, quantised))


def _normalise_live_energy(values: Any) -> Any:
    """Map raw frame RMS values to a stable, beat-visible 0..1 envelope."""

    np = visualizer._require_numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(1, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if float(np.max(values)) <= 1.0e-12:
        return np.zeros(values.shape, dtype=np.float32)
    low = float(np.percentile(values, 8.0))
    high = float(np.percentile(values, 98.0))
    if high - low <= 1.0e-12:
        high = float(np.max(values))
        low = float(np.min(values))
    if high - low <= 1.0e-12:
        return np.zeros(values.shape, dtype=np.float32)
    normalised = np.clip((values - low) / (high - low), 0.0, 1.0)
    # A gentle lift keeps quieter musical passages visible without turning
    # the live view into a flat white flash on loud transients.
    return np.asarray(np.power(normalised, 0.72), dtype=np.float32)


def build_live_track(
    raw_energy: Any,
    fps: int,
    base_zoom: Any = "1.0",
    max_zoom: Any = LIVE_DEFAULT_MAX_ZOOM,
) -> LiveAudioTrack:
    """Build cheap audio controls without invoking the full export analysis.

    The zoom control is an absolute log10 position.  Keeping it in log space
    avoids the overflow and quantisation that made deep live previews drift
    into a different crop.
    """

    np = visualizer._require_numpy()
    fps = int(fps)
    if fps <= 0:
        raise ValueError("live fps must be positive")
    energy = _normalise_live_energy(raw_energy)
    if energy.size == 1:
        onset = np.zeros(1, dtype=np.float32)
    else:
        onset = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
        onset_scale = float(np.percentile(onset, 99.0))
        if onset_scale <= 1.0e-9:
            onset = np.zeros(onset.shape, dtype=np.float32)
        else:
            onset = np.asarray(np.clip(onset / onset_scale, 0.0, 1.0), dtype=np.float32)

    # Keep the established Aurora phase behaviour, but calculate it from the
    # small live envelope instead of the export-quality librosa controls.
    flow_strength = np.clip(energy.astype(np.float64), 0.0, 1.0) ** 1.8
    phase_rate = visualizer.AURORA_FLOW_SPEED * (
        visualizer.AURORA_MIN_FLOW_FRACTION
        + (1.0 - visualizer.AURORA_MIN_FLOW_FRACTION) * flow_strength
    )
    phase = np.cumsum(phase_rate / float(fps)).astype(np.float32)
    zoom = visualizer._zoom_plan(
        energy,
        base_zoom,
        max_zoom,
        punch=2.5,
        quiet_speed=-0.02,
        onset=onset,
        beat_strength=0.8,
    )
    zoom = np.asarray(zoom, dtype=np.float64)
    duration = max(1.0 / float(fps), float(energy.size) / float(fps))
    return LiveAudioTrack(energy, onset, phase, zoom, duration, fps)


def _probe_audio_duration(audio_path: Path) -> Optional[float]:
    """Ask ffprobe for a duration when PCM decoding is unavailable."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
            check=False,
        )
        duration = float(result.stdout.strip())
    except (OSError, ValueError, OverflowError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


def _decode_audio_energy(
    audio_path: Path,
    fps: int,
    stop_event: Optional[threading.Event] = None,
    register_process: Optional[Callable[[Any], None]] = None,
) -> tuple[Any, float]:
    """Decode mono 8 kHz PCM and reduce it to one RMS value per live frame.

    The stream is reduced while it is read; a long song therefore never turns
    into a multi-hundred-megabyte NumPy allocation and skips the full
    pitch/onset/stem analysis used by final exports.
    """

    np = visualizer._require_numpy()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is unavailable for live audio analysis")
    frame_samples = max(1, round(LIVE_AUDIO_SAMPLE_RATE / int(fps)))
    process = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(audio_path),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(LIVE_AUDIO_SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        **visualizer._subprocess_group_options(),
    )
    if register_process is not None:
        register_process(process)
    raw_remainder = b""
    sample_remainder = np.empty(0, dtype=np.float32)
    energies: list[float] = []
    try:
        assert process.stdout is not None
        while True:
            if stop_event is not None and stop_event.is_set():
                raise _LiveCancelled
            chunk = process.stdout.read(LIVE_FRAME_READ_BYTES)
            if not chunk:
                break
            raw = raw_remainder + chunk
            usable = len(raw) - (len(raw) % 4)
            raw_remainder = raw[usable:]
            if usable == 0:
                continue
            samples = np.frombuffer(raw[:usable], dtype="<f4")
            if sample_remainder.size:
                samples = np.concatenate((sample_remainder, samples))
            complete = (samples.size // frame_samples) * frame_samples
            if complete:
                blocks = samples[:complete].reshape(-1, frame_samples)
                with np.errstate(over="ignore", invalid="ignore"):
                    block_energy = np.sqrt(np.mean(blocks * blocks, axis=1))
                energies.extend(
                    float(value) if math.isfinite(float(value)) else 0.0
                    for value in block_energy
                )
            sample_remainder = np.asarray(samples[complete:], dtype=np.float32).copy()
        if sample_remainder.size:
            with np.errstate(over="ignore", invalid="ignore"):
                final_energy = float(np.sqrt(np.mean(sample_remainder * sample_remainder)))
            energies.append(final_energy if math.isfinite(final_energy) else 0.0)
        return_code = process.wait(timeout=2.0)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg could not decode {audio_path}")
        return np.asarray(energies, dtype=np.float32), max(
            1.0 / float(fps), float(len(energies)) / float(fps)
        )
    finally:
        if register_process is not None:
            register_process(None)
        if process.poll() is None and stop_event is not None and stop_event.is_set():
            visualizer._terminate_subprocess(process)
        else:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                visualizer._terminate_subprocess(process)


def _fallback_live_track(
    audio_path: Path,
    fps: int,
    base_zoom: Any = "1.0",
    max_zoom: Any = LIVE_DEFAULT_MAX_ZOOM,
) -> LiveAudioTrack:
    """Keep visuals usable when a host lacks an FFmpeg decoder."""

    np = visualizer._require_numpy()
    duration = _probe_audio_duration(audio_path) or LIVE_FALLBACK_DURATION
    count = max(1, int(math.ceil(duration * fps)))
    timestamps = np.arange(count, dtype=np.float64) / float(fps)
    pulse = 0.42 + 0.24 * (0.5 + 0.5 * np.sin(timestamps * 2.1))
    track = build_live_track(pulse, fps, base_zoom, max_zoom)
    # Keep the visual clock aligned with the player even when the fallback
    # duration is not an exact multiple of the frame period.
    return LiveAudioTrack(
        track.energy,
        track.onset,
        track.phase,
        track.zoom,
        duration,
        track.fps,
    )


def _render_live_source(
    config: LiveViewConfig,
    source_width: int,
    source_height: int,
    log_zoom: float,
    native_library: Any,
    max_iter: int = LIVE_ITERATIONS,
    native_references: Optional[list[tuple[float, Any]]] = None,
    native_threads_override: Optional[int] = None,
) -> Any:
    """Render one absolute-zoom source with the cheapest valid backend."""

    max_iter = visualizer._validate_iteration_count(max_iter, "live iteration cap")
    if native_threads_override is None:
        render_threads = (
            config.native_threads
            if config.native_threads > 0
            else LIVE_DEFAULT_NATIVE_THREADS
        )
    else:
        render_threads = visualizer._validate_thread_count(
            native_threads_override,
            "live source native thread count",
        )
    render_options = None
    if native_library is not None:
        try:
            render_options = visualizer.NativeRenderOptions(
                strict=False,
                allow_recovery=True,
                backend=visualizer._native_backend_id("auto", native_library),
            )
        except (OSError, RuntimeError, ValueError):
            render_options = visualizer.NativeRenderOptions(
                strict=False,
                allow_recovery=True,
            )

    np = visualizer._require_numpy()

    def validated_native_result(
        result: Any,
        reference: Any,
        reference_root: Any = None,
    ) -> Any:
        """Repair a rare strict-BLA NaN mask before it reaches the GUI."""

        array = np.asarray(result, dtype=np.float32)
        if np.isfinite(array).all():
            return array
        if native_library is not None and reference is not None:
            backend = int(getattr(render_options, "backend", 0))
            try:
                repaired = visualizer._atlas_glitch_reference_field(
                    render_width=source_width,
                    render_height=source_height,
                    log10_zoom=log_zoom,
                    x_center=config.x_center,
                    y_center=config.y_center,
                    max_iter=max_iter,
                    series_order=3,
                    series_block=128,
                    native_threads=render_threads,
                    native_library=native_library,
                    native_backend=backend,
                    native_reference=reference,
                    native_reference_root=reference_root,
                    fallback_field=None,
                    fallback_zoom_factor=1.0,
                    fallback_max_iter=None,
                    allow_recovery=False,
                )
                repaired = np.asarray(repaired, dtype=np.float32)
                if np.isfinite(repaired).all():
                    return repaired
            except (RuntimeError, ValueError):
                pass
        # This path is exceptional and only runs when the native strict pass
        # could not cover every pixel. It is still preferable to handing GTK a
        # partially initialized buffer, which presents as black noise or a
        # rectangular fill artifact.
        if log_zoom <= 300.0:
            return np.asarray(
                visualizer.render_fractal(
                    source_width,
                    source_height,
                    log_zoom,
                    config.x_center,
                    config.y_center,
                    max_iter,
                    renderer="python",
                    native_threads=render_threads,
                    formula=config.formula,
                    julia_constant=config.julia_constant,
                ),
                dtype=np.float32,
            )
        return array

    # Reuse the same depth-safe references for every live source. Rebuilding a
    # reference orbit once per ladder entry was the main reason a deep live
    # preview could take longer to prepare than a cached export. The optional
    # empty-list path retains the standalone helper's safe compatibility
    # behaviour when no shared reference could be prepared.
    if config.formula == "mandelbrot" and log_zoom >= 12.0 and native_library is not None:
        shared_reference = None
        if native_references:
            shared_reference = visualizer._select_native_reference(
                native_references,
                log_zoom,
            )
        if shared_reference is not None:
            try:
                return validated_native_result(
                    visualizer.render_fractal(
                        source_width,
                        source_height,
                        log_zoom,
                        config.x_center,
                        config.y_center,
                        max_iter,
                        renderer="native",
                        native_threads=render_threads,
                        native_reference=shared_reference,
                        series_order=3,
                        series_block=128,
                        render_options=render_options,
                        formula=config.formula,
                        julia_constant=config.julia_constant,
                    ),
                    shared_reference,
                    native_references[0][1] if native_references else None,
                )
            except RuntimeError:
                # Keep the live preview useful if one shared tier is rejected
                # by a native build with a stricter radius contract. The
                # bounded Python fallback preserves the exact decimal centre.
                if log_zoom > 300.0:
                    raise
                return visualizer.render_fractal(
                    source_width,
                    source_height,
                    log_zoom,
                    config.x_center,
                    config.y_center,
                    max_iter,
                    renderer="python",
                    native_threads=render_threads,
                    formula=config.formula,
                    julia_constant=config.julia_constant,
                )
        reference_library: Any = None
        reference: Any = None
        try:
            reference_library, reference = visualizer._create_native_reference(
                config.x_center,
                config.y_center,
                max_iter,
                log_zoom,
                3,
                log_zoom,
                image_series_order=16,
            )
            return validated_native_result(
                visualizer.render_fractal(
                    source_width,
                    source_height,
                    log_zoom,
                    config.x_center,
                    config.y_center,
                    max_iter,
                    renderer="native",
                    native_threads=render_threads,
                    native_reference=reference,
                    series_order=3,
                    series_block=128,
                    render_options=render_options,
                    formula=config.formula,
                    julia_constant=config.julia_constant,
                ),
                reference,
            )
        except RuntimeError:
            # A live window must remain usable when a native build rejects a
            # coordinate spelling (for example a locale-specific value copied
            # from a GUI field). The validated Python perturbation path still
            # preserves the decimal centre through e300; final exports keep
            # their stricter native error instead of hiding it.
            if log_zoom > 300.0:
                raise
            return visualizer.render_fractal(
                source_width,
                source_height,
                log_zoom,
                config.x_center,
                config.y_center,
                max_iter,
                renderer="python",
                native_threads=render_threads,
                formula=config.formula,
                julia_constant=config.julia_constant,
            )
        finally:
            if reference_library is not None and reference is not None:
                reference_library.fractal_destroy_reference(reference)

    renderer = "auto"
    if config.formula == "mandelbrot" and log_zoom >= 12.0:
        # This is only reached when the native library is missing.  Explicitly
        # select the bounded high-precision fallback so the error from the
        # native reference API does not turn the live window into a blank one.
        renderer = "python"
    try:
        return visualizer.render_fractal(
            source_width,
            source_height,
            log_zoom,
            config.x_center,
            config.y_center,
            max_iter,
            renderer=renderer,
            native_threads=render_threads,
            render_options=render_options,
            formula=config.formula,
            julia_constant=config.julia_constant,
        )
    except RuntimeError:
        # A reference can be rejected when a user supplied centre has fewer
        # digits than the selected deep target. The live preview should still
        # be useful; its explicit Python fallback is preferable to a stale or
        # synthetic black field. Final export validation remains strict.
        if renderer != "python" and log_zoom <= 300.0:
            return visualizer.render_fractal(
                source_width,
                source_height,
                log_zoom,
                config.x_center,
                config.y_center,
                max_iter,
                renderer="python",
                native_threads=render_threads,
                formula=config.formula,
                julia_constant=config.julia_constant,
            )
        raise


def _prepare_live_native_references(
    config: LiveViewConfig,
    native_library: Any,
) -> list[tuple[float, Any]]:
    """Prepare one reusable reference ladder for all deep live sources."""

    if (
        native_library is None
        or config.formula != "mandelbrot"
        or config.preview_max_log_zoom < 12.0
    ):
        return []
    np = visualizer._require_numpy()
    requested_logs = live_zoom_ladder(
        config.base_zoom,
        _live_zoom_text(config.preview_max_log_zoom),
    )
    if requested_logs.size < 2:
        atlas_step = 0.75
    else:
        atlas_step = float(np.min(np.diff(requested_logs)))
    reference_logs = visualizer._native_reference_tier_logs(
        config.preview_max_log_zoom,
        atlas_step,
    )
    if not reference_logs:
        return []
    references: list[tuple[float, Any]] = []
    reference_iter = max(
        LIVE_MAX_ITERATIONS,
        live_iteration_cap(config.formula, config.preview_max_log_zoom),
    )
    clone_tiers = (
        len(reference_logs) > 1
        and hasattr(native_library, "fractal_create_reference_reusable")
        and hasattr(native_library, "fractal_clone_reference")
    )
    try:
        _, root_reference = visualizer._create_native_reference(
            config.x_center,
            config.y_center,
            reference_iter,
            config.preview_max_log_zoom,
            3,
            reference_logs[0],
            image_series_order=16,
            reusable=clone_tiers,
        )
        references.append((reference_logs[0], root_reference))
        for start_log in reference_logs[1:]:
            reference = None
            if clone_tiers:
                try:
                    reference = visualizer._clone_native_reference(
                        native_library,
                        root_reference,
                        start_log,
                    )
                except RuntimeError:
                    reference = None
            if reference is None:
                _, reference = visualizer._create_native_reference(
                    config.x_center,
                    config.y_center,
                    reference_iter,
                    config.preview_max_log_zoom,
                    3,
                    start_log,
                    image_series_order=16,
                )
            references.append((start_log, reference))
    except (OSError, RuntimeError, ValueError, OverflowError):
        visualizer._destroy_native_references(native_library, references)
        return []
    return references


def build_live_zoom_sources(
    config: LiveViewConfig,
    source_width: int,
    source_height: int,
    native_library: Any,
    stop_event: Optional[threading.Event] = None,
    status: Optional[Callable[[str], None]] = None,
    native_references: Optional[list[tuple[float, Any]]] = None,
    store: Optional[LiveZoomSourceStore] = None,
    max_sources: Optional[int] = None,
    source_native_threads: Optional[int] = None,
) -> LiveZoomSources:
    """Prepare the small absolute-zoom atlas used by the live compositor."""

    np = visualizer._require_numpy()
    requested_logs = live_zoom_ladder(config.base_zoom, config.max_zoom)
    if store is None:
        store = LiveZoomSourceStore(requested_logs)
    elif len(store.requested_logs) != len(requested_logs) or not np.array_equal(
        np.asarray(store.requested_logs, dtype=np.float64),
        np.asarray(requested_logs, dtype=np.float64),
    ):
        raise ValueError("live source store does not match the requested zoom ladder")
    if max_sources is not None:
        try:
            max_sources = int(max_sources)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("live source limit must be an integer") from error
        if max_sources <= 0:
            raise ValueError("live source limit must be positive")
    truncated_for_budget = False
    start_index = store.count
    try:
        for index in range(start_index, len(requested_logs)):
            if max_sources is not None and store.count >= max_sources:
                break
            if stop_event is not None and stop_event.is_set():
                raise _LiveCancelled
            source_log_zoom = float(requested_logs[index])
            source_iter = live_iteration_cap(config.formula, source_log_zoom)
            if status is not None:
                status(
                    f"rendering live zoom source {index + 1}/{len(requested_logs)} "
                    f"({_live_zoom_text(source_log_zoom)}, {source_iter} iterations)…"
                )
            # A shallow cap can make a valid deep target look entirely bounded.
            # Retry only that pathological case, and only as far as the bounded
            # live budget allows; ordinary sources still pay for one render.
            field = None
            final_iter = source_iter
            while True:
                if native_references is None and source_native_threads is None:
                    field = _render_live_source(
                        config,
                        source_width,
                        source_height,
                        source_log_zoom,
                        native_library,
                        final_iter,
                    )
                elif source_native_threads is None:
                    field = _render_live_source(
                        config,
                        source_width,
                        source_height,
                        source_log_zoom,
                        native_library,
                        final_iter,
                        native_references,
                    )
                else:
                    field = _render_live_source(
                        config,
                        source_width,
                        source_height,
                        source_log_zoom,
                        native_library,
                        final_iter,
                        native_references,
                        source_native_threads,
                    )
                field = visualizer._validated_field(
                    field,
                    (source_height, source_width),
                    "live zoom source",
                )
                finite_exterior = np.isfinite(field) & (field < float(final_iter))
                if np.any(finite_exterior) or final_iter >= LIVE_MAX_ITERATIONS:
                    break
                if final_iter <= LIVE_MIN_ITERATIONS:
                    break
                next_iter = min(LIVE_MAX_ITERATIONS, final_iter * 2)
                if next_iter == final_iter:
                    break
                final_iter = next_iter
                if status is not None:
                    status(
                        f"deep live source {_live_zoom_text(source_log_zoom)} reached "
                        f"its cap; retrying with {final_iter} iterations…"
                    )

            assert field is not None
            finite_exterior = np.isfinite(field) & (field < float(final_iter))
            previous_log = (
                float(requested_logs[index - 1])
                if index > 0
                else None
            )
            if not np.any(finite_exterior) and store.count and (
                previous_log is None or source_log_zoom > previous_log + 1.0e-9
            ):
                # Do not expose an all-interior source as a black rectangular
                # viewport. Hold the last source and clamp the live camera there;
                # this is especially important for connected Julia targets whose
                # boundary does not escape within a practical screensaver budget.
                truncated_for_budget = True
                if status is not None:
                    status(
                        f"live preview holding at "
                        f"{_live_zoom_text(float(requested_logs[index - 1]))}; "
                        f"{_live_zoom_text(source_log_zoom)} needs more than "
                        f"{final_iter} iterations"
                    )
                break
            store.append(source_log_zoom, field, final_iter)
        is_complete = store.count >= len(requested_logs) or truncated_for_budget
        if max_sources is None or is_complete:
            store.finish(
                capped=config.preview_zoom_is_capped or truncated_for_budget,
            )
        return store.snapshot()
    except _LiveCancelled:
        raise
    except BaseException as error:
        store.fail(error)
        raise


def _live_zoom_factor(log_zoom: float, source_log_zoom: float) -> float:
    """Convert a local live interval position to a finite crop factor."""

    delta = max(0.0, float(log_zoom) - float(source_log_zoom))
    return max(1.0, math.pow(10.0, delta))


def _live_source_zoom_limit(
    sources: LiveZoomSources | LiveZoomSourceStore,
) -> Optional[float]:
    """Return the current live camera ceiling for a source ladder."""

    if isinstance(sources, LiveZoomSourceStore):
        return sources.display_zoom_limit()
    try:
        logs = sources.log_zooms
    except (AttributeError, TypeError):
        return None
    if len(logs) == 0:
        return None
    try:
        value = float(logs[-1])
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _live_colour_frame(
    sources: LiveZoomSources | LiveZoomSourceStore,
    log_zoom: float,
    output_width: int,
    output_height: int,
    phase: float,
    energy: float,
    native_library: Any,
    config: LiveViewConfig,
) -> Any:
    """Compose one continuous frame from the absolute live source ladder."""

    np = visualizer._require_numpy()
    source_store = sources if isinstance(sources, LiveZoomSourceStore) else None
    if isinstance(sources, LiveZoomSourceStore):
        sources = sources.snapshot()
    logs = np.asarray(sources.log_zooms, dtype=np.float64)
    if logs.size == 0 or not sources.fields:
        raise RuntimeError("live zoom source ladder is empty")
    source_limit = _live_source_zoom_limit(source_store or sources)
    if source_limit is None:
        source_limit = float(logs[-1])
    # An incomplete source store exposes one safe interval beyond its last
    # completed field. This preserves continuous crop motion while the next
    # field is rendering; the playback loop rate-limits the approach so an
    # append can never turn into a one-frame jump.
    bounded_log_zoom = min(
        max(float(log_zoom), float(logs[0])),
        max(float(logs[-1]), source_limit),
    )
    index = int(np.searchsorted(logs, bounded_log_zoom, side="right") - 1)
    index = max(0, min(index, len(sources.fields) - 1))
    parent = sources.fields[index]
    parent_log_zoom = float(logs[index])
    caps = sources.iteration_caps
    if len(caps) != len(sources.fields):
        # Keep hand-constructed LiveZoomSources values compatible with the
        # original fixed-budget helper API.
        caps = (LIVE_ITERATIONS,) * len(sources.fields)
    parent_iter = int(caps[index])
    parent_zoom = _live_zoom_factor(bounded_log_zoom, parent_log_zoom)
    child = None
    child_iter: Optional[int] = None
    child_fraction = 0.0
    if index + 1 < len(sources.fields):
        child = sources.fields[index + 1]
        child_log_zoom = float(logs[index + 1])
        child_interval_factor = _live_zoom_factor(child_log_zoom, parent_log_zoom)
        child_fraction = min(1.0, parent_zoom / child_interval_factor)
        child_iter = int(caps[index + 1])
    colour_threads = (
        config.native_threads
        if config.native_threads > 0
        else LIVE_DEFAULT_NATIVE_THREADS
    )
    return visualizer._atlas_colour_frame(
        parent,
        child,
        output_width,
        output_height,
        parent_zoom,
        child_fraction,
        parent_iter,
        child_iter,
        phase,
        energy,
        energy,
        native_library,
        colour_threads,
        "bilinear",
        config.palette,
        0.5,
        config.palette_file,
    )


def _audio_player_command(audio_path: Path) -> Optional[list[str]]:
    """Return a quiet ffplay command, or None when ffplay is not installed."""

    ffplay = shutil.which("ffplay")
    if ffplay is None:
        return None
    return [
        ffplay,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-nodisp",
        "-autoexit",
        str(audio_path),
    ]


def _start_audio_player(audio_path: Path) -> Optional[subprocess.Popen[Any]]:
    command = _audio_player_command(audio_path)
    if command is None:
        return None
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **visualizer._subprocess_group_options(),
        )
    except OSError:
        return None


try:
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    gi.require_version("GdkPixbuf", "2.0")
    from gi.repository import Gdk, GdkPixbuf, GLib, Gtk
except (ImportError, ValueError) as error:  # pragma: no cover - environment dependent
    Gdk = GdkPixbuf = GLib = Gtk = None  # type: ignore[assignment]
    GTK_IMPORT_ERROR: Exception | None = error
else:
    GTK_IMPORT_ERROR = None


if Gtk is not None:

    class LiveViewWindow(Gtk.Window):
        """Fullscreen-capable live compositor with a one-frame back buffer."""

        def __init__(
            self,
            config: LiveViewConfig,
            *,
            transient_for: Optional[Gtk.Window] = None,
            on_closed: Optional[Callable[["LiveViewWindow"], None]] = None,
        ) -> None:
            super().__init__(title="Fractal Audio Viz — live view")
            self.config = config
            self._on_closed_callback = on_closed
            self._stop_event = threading.Event()
            self._child_lock = threading.Lock()
            self._audio_process: Any = None
            self._decoder_process: Any = None
            self._frame_lock = threading.Lock()
            self._pending_frame: Any = None
            self._frame_idle_scheduled = False
            self._closed = False
            self._fullscreen = False
            self._pixbuf: Any = None
            self._status_visible = True
            self._dismiss_status_on_frame = True

            if transient_for is not None:
                self.set_transient_for(transient_for)
            self.set_default_size(min(1280, config.width), min(720, config.height))
            self.set_size_request(320, 180)
            # Fullscreen is intentionally chrome-free; a windowed launch
            # should retain normal GTK decorations so it can be moved and
            # closed like the rest of the desktop.
            self.set_decorated(not config.fullscreen)
            self.connect("delete-event", self._on_delete_event)
            self.connect("destroy", self._on_destroy)
            self.connect("key-press-event", self._on_key_press)

            self.drawing_area = Gtk.DrawingArea()
            self.drawing_area.set_hexpand(True)
            self.drawing_area.set_vexpand(True)
            self.drawing_area.set_double_buffered(True)
            self.drawing_area.connect("draw", self._draw)

            self.status = Gtk.Label(label="preparing live view…")
            self.status.set_halign(Gtk.Align.START)
            self.status.set_valign(Gtk.Align.END)
            self.status.set_margin_start(18)
            self.status.set_margin_end(18)
            self.status.set_margin_bottom(14)
            self.status.get_style_context().add_class("dim-label")

            overlay = Gtk.Overlay()
            overlay.add(self.drawing_area)
            overlay.add_overlay(self.status)
            self.add(overlay)

            self._worker = threading.Thread(
                target=self._worker_main,
                name="fractal-live-view",
                daemon=True,
            )
            self._worker.start()
            if config.fullscreen:
                GLib.idle_add(self._apply_initial_fullscreen)

        def _apply_initial_fullscreen(self) -> bool:
            if not self._closed and self.config.fullscreen:
                self.fullscreen()
                self._fullscreen = True
            return False

        def _set_child(self, name: str, process: Any) -> None:
            with self._child_lock:
                setattr(self, name, process)

        def _stop_child(self, name: str) -> None:
            with self._child_lock:
                process = getattr(self, name)
                setattr(self, name, None)
            if process is None:
                return
            try:
                if process.poll() is None:
                    process.terminate()
            except (AttributeError, OSError):
                pass

        def stop(self) -> None:
            if self._stop_event.is_set():
                return
            self._stop_event.set()
            self._stop_child("_audio_process")
            self._stop_child("_decoder_process")

        def _post_status(
            self,
            text: str,
            *,
            visible: bool = True,
            dismiss_on_frame: bool = True,
        ) -> None:
            if GLib is None:
                return

            def apply() -> bool:
                if self._closed:
                    return False
                self.status.set_text(text)
                self._dismiss_status_on_frame = dismiss_on_frame
                if visible:
                    self.status.show()
                    self._status_visible = True
                else:
                    self.status.hide()
                    self._status_visible = False
                return False

            GLib.idle_add(apply)

        def _publish_frame(self, frame: Any) -> None:
            if GLib is None:
                return
            with self._frame_lock:
                if self._closed:
                    return
                self._pending_frame = frame
                if self._frame_idle_scheduled:
                    return
                self._frame_idle_scheduled = True
            GLib.idle_add(self._drain_frame)

        def _drain_frame(self) -> bool:
            with self._frame_lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is not None and not self._closed:
                np = visualizer._require_numpy()
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    self._post_status("live renderer returned an invalid frame", visible=True)
                else:
                    height, width, _channels = frame.shape
                    data = GLib.Bytes(frame.tobytes())
                    self._pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                        data,
                        GdkPixbuf.Colorspace.RGB,
                        False,
                        8,
                        int(width),
                        int(height),
                        int(width * 3),
                    )
                    self.drawing_area.queue_draw()
                    if self._status_visible and self._dismiss_status_on_frame:
                        self.status.hide()
                        self._status_visible = False
            with self._frame_lock:
                if self._pending_frame is not None and not self._closed:
                    return True
                self._frame_idle_scheduled = False
            return False

        def _draw(self, _widget: Any, context: Any) -> bool:
            allocation = self.drawing_area.get_allocation()
            width = max(1, int(allocation.width))
            height = max(1, int(allocation.height))
            context.set_source_rgb(0.0, 0.0, 0.0)
            context.paint()
            if self._pixbuf is None:
                return False
            source_width = self._pixbuf.get_width()
            source_height = self._pixbuf.get_height()
            scale = min(width / source_width, height / source_height)
            draw_width = source_width * scale
            draw_height = source_height * scale
            left = (width - draw_width) * 0.5
            top = (height - draw_height) * 0.5
            context.save()
            context.translate(left, top)
            context.scale(scale, scale)
            Gdk.cairo_set_source_pixbuf(context, self._pixbuf, 0.0, 0.0)
            context.get_source().set_filter(4)  # Cairo FILTER_BILINEAR.
            context.paint()
            context.restore()
            return False

        def _on_key_press(self, _widget: Any, event: Any) -> bool:
            if event.keyval in (Gdk.KEY_Escape, Gdk.KEY_q):
                self.close()
                return True
            if event.keyval == Gdk.KEY_F11:
                if self._fullscreen:
                    self.unfullscreen()
                    self._fullscreen = False
                else:
                    self.fullscreen()
                    self._fullscreen = True
                return True
            return False

        def _on_delete_event(self, _window: Any, _event: Any) -> bool:
            self.stop()
            return False

        def _on_destroy(self, _window: Any) -> None:
            if self._closed:
                return
            self._closed = True
            self.stop()
            with self._frame_lock:
                self._pending_frame = None
            if self._on_closed_callback is not None:
                self._on_closed_callback(self)

        def _worker_main(self) -> None:
            player: Any = None
            source_builder: Optional[threading.Thread] = None
            native_library: Any = None
            native_references: list[tuple[float, Any]] = []
            try:
                self._post_status("analysing audio…")
                try:
                    raw_energy, duration = _decode_audio_energy(
                        self.config.audio_path,
                        self.config.fps,
                        self._stop_event,
                        lambda process: self._set_child("_decoder_process", process),
                    )
                    track = build_live_track(
                        raw_energy,
                        self.config.fps,
                        self.config.base_zoom,
                        _live_zoom_text(self.config.preview_max_log_zoom),
                    )
                    if raw_energy.size == 0:
                        raise RuntimeError("ffmpeg returned no audio samples")
                    # A truncated/odd decoder stream should not make playback
                    # wrap at a different time from the visuals.
                    if duration > 0.0 and abs(duration - track.duration) > 1.0:
                        track = LiveAudioTrack(
                            track.energy,
                            track.onset,
                            track.phase,
                            track.zoom,
                            duration,
                            track.fps,
                        )
                except _LiveCancelled:
                    return
                except (OSError, RuntimeError, ValueError, OverflowError):
                    track = _fallback_live_track(
                        self.config.audio_path,
                        self.config.fps,
                        self.config.base_zoom,
                        _live_zoom_text(self.config.preview_max_log_zoom),
                    )
                    self._post_status("audio analysis unavailable; using a live clock", visible=True)
                if self._stop_event.is_set():
                    return

                self._post_status("preparing live zoom ladder…")
                try:
                    native_library = visualizer._get_native_library()
                except (OSError, RuntimeError):
                    native_library = None
                native_preview = native_library is not None and (
                    self.config.formula == "mandelbrot"
                    or self.config.max_log_zoom < visualizer.ALTERNATE_PERTURBATION_MIN_LOG
                )
                source_width, source_height = live_dimensions(
                    self.config.width,
                    self.config.height,
                    native_available=native_preview,
                )
                # Keep the reference handles in one stable list so the
                # background source worker can populate it before it reaches
                # the first deep source. For the normal shallow-base case,
                # defer the expensive MPFR/BLA setup until the first three
                # displayable fields are already available.
                native_references = []
                requested_logs = live_zoom_ladder(
                    self.config.base_zoom,
                    self.config.max_zoom,
                )
                source_store = LiveZoomSourceStore(requested_logs)
                initial_count = min(
                    len(requested_logs),
                    LIVE_INITIAL_SOURCE_COUNT,
                )
                initial_last_log = float(requested_logs[initial_count - 1])
                if initial_last_log >= 12.0:
                    native_references.extend(
                        _prepare_live_native_references(
                            self.config,
                            native_library,
                        )
                    )
                sources = build_live_zoom_sources(
                    self.config,
                    source_width,
                    source_height,
                    native_library,
                    self._stop_event,
                    lambda message: self._post_status(message, dismiss_on_frame=False),
                    native_references,
                    source_store,
                    initial_count,
                )
                if self._stop_event.is_set():
                    return

                if source_store.count < len(requested_logs) and not source_store.finished:
                    def finish_source_ladder() -> None:
                        try:
                            if (
                                not native_references
                                and native_library is not None
                                and self.config.formula == "mandelbrot"
                                and self.config.preview_max_log_zoom >= 12.0
                            ):
                                native_references.extend(
                                    _prepare_live_native_references(
                                        self.config,
                                        native_library,
                                    )
                                )
                            build_live_zoom_sources(
                                self.config,
                                source_width,
                                source_height,
                                native_library,
                                self._stop_event,
                                native_references=native_references,
                                store=source_store,
                                source_native_threads=(
                                    max(1, self.config.native_threads // 2)
                                    if self.config.native_threads > 0
                                    else 1
                                ),
                            )
                        except _LiveCancelled:
                            return
                        except BaseException as error:
                            source_store.fail(error)
                            self._post_status(
                                f"live zoom ladder stopped: {error}",
                                visible=True,
                                dismiss_on_frame=False,
                            )

                    source_builder = threading.Thread(
                        target=finish_source_ladder,
                        name="fractal-live-sources",
                        daemon=True,
                    )
                    source_builder.start()

                if sources.capped:
                    actual_log_zoom = float(sources.log_zooms[-1])
                    self._post_status(
                        "live preview capped at "
                        f"{_live_zoom_text(actual_log_zoom)}; Esc to exit · "
                        "F11 toggles fullscreen",
                        visible=True,
                        dismiss_on_frame=False,
                    )
                elif source_builder is not None:
                    self._post_status(
                        "live view ready · warming deep zoom sources · "
                        "Esc to exit · F11 toggles fullscreen",
                        visible=True,
                        dismiss_on_frame=False,
                    )

                player = _start_audio_player(self.config.audio_path)
                self._set_child("_audio_process", player)
                if player is None:
                    self._post_status(
                        "ffplay unavailable; showing visual preview only",
                        visible=True,
                        dismiss_on_frame=False,
                    )
                else:
                    self._post_status("Esc to exit · F11 toggles fullscreen", visible=True)
                self._run_frames(source_store, track, native_library)
            except _LiveCancelled:
                pass
            except Exception as error:  # keep a broken preview from killing GTK
                self._post_status(f"live view error: {error}", visible=True)
            finally:
                self._stop_event.set()
                if source_builder is not None:
                    source_builder.join(timeout=5.0)
                if native_references and (
                    source_builder is None or not source_builder.is_alive()
                ):
                    visualizer._destroy_native_references(
                        native_library,
                        native_references,
                    )
                self._stop_child("_audio_process")
                if player is not None:
                    try:
                        if player.poll() is None:
                            visualizer._terminate_subprocess(player)
                    except (AttributeError, OSError):
                        pass

        def _run_frames(
            self,
            sources: LiveZoomSources | LiveZoomSourceStore,
            track: LiveAudioTrack,
            native_library: Any,
        ) -> None:
            np = visualizer._require_numpy()
            initial_sources = (
                sources.snapshot()
                if isinstance(sources, LiveZoomSourceStore)
                else sources
            )
            if not initial_sources.fields:
                raise RuntimeError("live zoom source ladder is empty")
            source_shape = np.asarray(initial_sources.fields[0]).shape
            if len(source_shape) != 2:
                raise RuntimeError("live zoom source has invalid dimensions")
            # Keep the expensive KFP/Aurora colour pass at the bounded source
            # density. The DrawingArea applies one bilinear upscale to the
            # window; rendering a 1920x1080 RGB frame here would do the same
            # resize twice and make the GUI the dominant live-view bottleneck.
            source_height, source_width = int(source_shape[0]), int(source_shape[1])
            frame_interval = 1.0 / float(track.fps)
            started = time.monotonic()
            next_deadline = started
            previous_cycle = -1
            phase_span = float(track.phase[-1]) if track.phase.size else 0.0
            display_log_zoom: Optional[float] = None
            last_zoom_update = started
            while not self._stop_event.is_set():
                elapsed = time.monotonic() - started
                if track.duration <= 0.0:
                    local_time = 0.0
                    cycle = 0
                else:
                    cycle = int(elapsed / track.duration)
                    if not self.config.loop and cycle > 0:
                        self._post_status(
                            "song finished · Esc to close",
                            visible=True,
                            dismiss_on_frame=False,
                        )
                        return
                    local_time = elapsed - cycle * track.duration
                cycle_changed = cycle != previous_cycle
                if cycle_changed:
                    if cycle and self.config.loop:
                        self._stop_child("_audio_process")
                        player = _start_audio_player(self.config.audio_path)
                        self._set_child("_audio_process", player)
                    previous_cycle = cycle
                index = min(
                    track.energy.size - 1,
                    max(0, int(local_time * float(track.fps))),
                )
                energy = float(track.energy[index])
                phase = float(track.phase[index]) + cycle * phase_span
                requested_log_zoom = float(track.zoom[index])
                source_limit = _live_source_zoom_limit(sources)
                if source_limit is not None:
                    requested_log_zoom = min(requested_log_zoom, source_limit)
                now = time.monotonic()
                if display_log_zoom is None or cycle_changed:
                    display_log_zoom = requested_log_zoom
                else:
                    elapsed_zoom = max(0.0, now - last_zoom_update)
                    max_step = LIVE_MAX_ZOOM_RATE * max(elapsed_zoom, frame_interval)
                    delta = requested_log_zoom - display_log_zoom
                    display_log_zoom += max(-max_step, min(max_step, delta))
                last_zoom_update = now
                rgb = _live_colour_frame(
                    sources,
                    display_log_zoom,
                    source_width,
                    source_height,
                    phase,
                    energy,
                    native_library,
                    self.config,
                )
                self._publish_frame(np.asarray(rgb, dtype=np.uint8))
                next_deadline += frame_interval
                now = time.monotonic()
                if next_deadline < now:
                    next_deadline = now
                self._stop_event.wait(max(0.0, next_deadline - now))


else:

    class LiveViewWindow:  # pragma: no cover - exercised only without GTK
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            detail = str(GTK_IMPORT_ERROR) if GTK_IMPORT_ERROR is not None else "GTK3 is unavailable"
            raise RuntimeError(f"GTK3/PyGObject is required for live view: {detail}")


def _resolve_cli_config(args: argparse.Namespace) -> LiveViewConfig:
    julia_constant = visualizer._parse_coordinate_pair(args.julia_c, "--julia-c")
    max_log_zoom = visualizer._zoom_log(args.max_zoom)
    x_center, y_center, _selected = visualizer._resolve_render_point(
        point_spec=args.point,
        random_point=args.random_point,
        x_center=args.x_center,
        y_center=args.y_center,
        random_seed=args.random_seed,
        max_log_zoom=max_log_zoom,
        formula=args.formula,
        julia_constant=julia_constant,
    )
    palette_file = Path(args.palette_file).expanduser() if args.palette_file else None
    return LiveViewConfig(
        audio_path=Path(args.audio),
        formula=args.formula,
        x_center=x_center,
        y_center=y_center,
        julia_constant=julia_constant,
        palette=args.palette,
        palette_file=palette_file,
        width=args.width,
        height=args.height,
        fps=args.fps,
        native_threads=args.native_threads,
        loop=args.loop,
        fullscreen=args.fullscreen,
        base_zoom=args.base_zoom,
        max_zoom=args.max_zoom,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path, default=Path(visualizer.DEFAULT_AUDIO))
    parser.add_argument("--formula", choices=visualizer.FORMULA_CHOICES, default="mandelbrot")
    parser.add_argument("--point", default=None, help="catalogue slug, random, or REAL,IMAG")
    parser.add_argument("--random-point", action="store_true")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--x-center", default=None)
    parser.add_argument("--y-center", default=None)
    parser.add_argument(
        "--julia-c",
        default=f"{visualizer.DEFAULT_JULIA_C[0]},{visualizer.DEFAULT_JULIA_C[1]}",
    )
    parser.add_argument("--palette", choices=visualizer.PALETTE_CHOICES, default="aurora")
    parser.add_argument("--palette-file", type=Path, default=None)
    parser.add_argument("--base-zoom", default="1.0")
    parser.add_argument("--max-zoom", default=LIVE_DEFAULT_MAX_ZOOM)
    parser.add_argument("--width", type=int, default=LIVE_DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=LIVE_DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=LIVE_DEFAULT_FPS)
    parser.add_argument("--native-threads", type=int, default=0)
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.set_defaults(loop=True)
    parser.add_argument("--windowed", dest="fullscreen", action="store_false")
    parser.set_defaults(fullscreen=True)
    return parser


def main() -> None:
    if Gtk is None:
        detail = str(GTK_IMPORT_ERROR) if GTK_IMPORT_ERROR is not None else "GTK3 is unavailable"
        raise SystemExit(f"GTK3/PyGObject is required for live view: {detail}")
    args = build_parser().parse_args()
    try:
        config = _resolve_cli_config(args)
    except (OSError, RuntimeError, ValueError, OverflowError) as error:
        raise SystemExit(str(error)) from error

    window_holder: list[LiveViewWindow] = []

    def closed(window: LiveViewWindow) -> None:
        if window in window_holder:
            window_holder.remove(window)
        Gtk.main_quit()

    window = LiveViewWindow(config, on_closed=closed)
    window_holder.append(window)
    window.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
