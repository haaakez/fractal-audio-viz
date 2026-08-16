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
import ctypes
import hashlib
import heapq
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


# Do this before importing numerical libraries.  It helps BLAS-backed NumPy
# without forcing an unnecessarily large number of worker threads.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 3))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")


DEFAULT_AUDIO = "song.mp3"
DEFAULT_OUTPUT = "fractal_viz.mp4"
DEFAULT_X_CENTER = (
    "-1.711030826576984823314722728180246694222252112777834549259732560022287905717123892927883662257081287304281205446785464750361251745"
)
DEFAULT_Y_CENTER = (
    "0.000001509818957972609043170877177447547323633361751210706181530872644435995661269979265353802853564243259051551728584671844401805"
)

# Fields are expensive, so they can be cached; their cache identity must
# change when native numerical behaviour changes.  This prevents a new
# renderer from silently reusing an old, under-resolved .npy keyframe.
KEYFRAME_CACHE_SCHEMA = "bla-keyframe-blend-colour-v6"
AUDIO_CACHE_SCHEMA = "audio-controls-v5-dynamics-pitch"
ATLAS_CACHE_SCHEMA = "nested-atlas-v5-spatial-recovery"

# A single reference is normally ideal, but at deep zoom a frame-sized tile
# can contain a narrow boundary that is much farther from that reference than
# the rest of the image.  The native renderer then spends most of its time in
# exact/replayed tails. Large deep tiles start with four local views and split
# only the cells whose render exceeds a short native deadline. This keeps MPFR
# setup and memory proportional to the genuinely difficult part of a tile.
ATLAS_LOCAL_REFERENCE_MIN_LOG = 40.0
ATLAS_LOCAL_REFERENCE_MIN_DIMENSION = 96
ATLAS_LOCAL_REFERENCE_MAX_DIVISIONS = 32
ATLAS_LOCAL_REFERENCE_MIN_BUDGET_MS = 200
ATLAS_LOCAL_REFERENCE_MAX_BUDGET_MS = 4000
ATLAS_LOCAL_REFERENCE_MS_PER_PIXEL = 0.05
ATLAS_LOCAL_REFERENCE_FINAL_BUDGET_MS = 750
ATLAS_LOCAL_REFERENCE_TILE_BUDGET_MS = 30_000

_native_library: Any = None
_native_checked = False
_native_notice_printed = False

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


def _field_renderer_cache_identity(renderer: str) -> str:
    """Return a content-based cache namespace for the active field renderer."""

    if renderer == "python":
        try:
            return "python-" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
        except OSError:
            return "python-unknown"

    configured = os.environ.get("MANDELBROT_LIBRARY")
    candidates = [
        Path(configured) if configured else Path(__file__).with_name("mandelbrot.so"),
        Path(__file__).with_name("libmandelbrot.so"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return "native-" + hashlib.sha256(candidate.read_bytes()).hexdigest()[:16]
        except OSError:
            continue
    return "native-unavailable"


def _get_native_library() -> Any:
    """Load only the versioned renderer ABI, never an old incompatible .so."""

    global _native_checked, _native_library
    if _native_checked:
        return _native_library
    _native_checked = True

    configured = os.environ.get("MANDELBROT_LIBRARY")
    candidates = [
        Path(configured) if configured else Path(__file__).with_name("mandelbrot.so"),
        Path(__file__).with_name("libmandelbrot.so"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            library = ctypes.CDLL(str(candidate))
            if not hasattr(library, "fractal_abi_version"):
                continue
            library.fractal_abi_version.restype = ctypes.c_int
            if library.fractal_abi_version() != 8:
                continue
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
            library.fractal_create_reference.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            library.fractal_create_reference.restype = ctypes.c_void_p
            library.fractal_destroy_reference.argtypes = [ctypes.c_void_p]
            library.fractal_destroy_reference.restype = None
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
            if hasattr(library, "fractal_set_stats_enabled"):
                library.fractal_set_stats_enabled.argtypes = [ctypes.c_int]
                library.fractal_set_stats_enabled.restype = None
            if hasattr(library, "fractal_get_last_stats"):
                library.fractal_get_last_stats.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_int,
                ]
                library.fractal_get_last_stats.restype = ctypes.c_int
            _native_library = library
            return library
        except OSError:
            # Missing MPFR/OpenMP runtime libraries should not prevent the
            # Python implementation from running.
            continue
    return None


def _native_set_stats_enabled(library: Any, enabled: bool) -> bool:
    setter = getattr(library, "fractal_set_stats_enabled", None)
    if setter is None:
        return False
    setter(1 if enabled else 0)
    return True


def _native_get_stats(library: Any) -> Optional[dict[str, int]]:
    getter = getattr(library, "fractal_get_last_stats", None)
    if getter is None:
        return None
    values = (ctypes.c_uint64 * len(NATIVE_STATS_FIELDS))()
    count = int(getter(values, len(NATIVE_STATS_FIELDS)))
    if count != len(NATIVE_STATS_FIELDS):
        return None
    return {
        name: int(values[index])
        for index, name in enumerate(NATIVE_STATS_FIELDS)
    }


@dataclass
class AudioFeatures:
    """Frame-aligned controls used by the visual animation."""

    vocal: Any
    instrumental: Any
    phase: Any
    pitch: Any
    frame_count: int


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


def _load_audio(path: Path, sample_rate: int) -> Any:
    np = _require_numpy()
    librosa = _require_librosa()
    try:
        samples, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Could not decode audio file {path}: {exc}") from exc
    if samples.size == 0:
        raise RuntimeError(f"Audio file is empty: {path}")
    return np.asarray(samples, dtype=np.float32)


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
        separation_signature += "-demucs" if importlib.util.find_spec("demucs") else "-fullmix"
    digest.update(separation_signature.encode("ascii"))
    with audio_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return cache_dir / f"audio-{digest.hexdigest()[:24]}.npz"


def _atomic_save_features(path: Path, features: AudioFeatures) -> None:
    np = _require_numpy()
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
    vocals = sorted(root.rglob("vocals.wav"))
    no_vocals = sorted(root.rglob("no_vocals.wav"))
    if vocals and no_vocals:
        return vocals[0], no_vocals[0]
    # Some Demucs versions use ``instrumental.wav`` for the second stem.
    instruments = sorted(root.rglob("instrumental.wav"))
    if vocals and instruments:
        return vocals[0], instruments[0]
    return None


def _demucs_stems(audio_path: Path, output_dir: Path, mode: str) -> Optional[tuple[Path, Path]]:
    """Run Demucs if requested/available and return its two stem paths."""

    available = importlib.util.find_spec("demucs") is not None
    if not available:
        if mode == "demucs":
            raise RuntimeError(
                "--separation demucs was requested, but Demucs is not installed. "
                "Install it with: pip install demucs"
            )
        print("Demucs is not installed; using full-song control.")
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
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        if mode == "demucs":
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Demucs failed:\n{details}")
        print("Demucs failed; using full-song control.")
        if result.stderr:
            print(result.stderr.strip())
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
    cache_path = _audio_cache_path(
        audio_path, sample_rate, fps, separation, attack, release, cache_dir
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                vocal = np.asarray(cached["vocal"], dtype=np.float32)
                instrumental = np.asarray(cached["instrumental"], dtype=np.float32)
                phase = np.asarray(cached["phase"], dtype=np.float32)
                pitch = np.asarray(cached["pitch"], dtype=np.float32)
                frame_count = int(cached["frame_count"])
            if (
                frame_count > 0
                and vocal.shape == (frame_count,)
                and instrumental.shape == (frame_count,)
                and phase.shape == (frame_count,)
                and pitch.shape == (frame_count,)
            ):
                print(f"Using cached audio controls {cache_path.name}.")
                return AudioFeatures(vocal, instrumental, phase, pitch, frame_count)
        except (OSError, KeyError, ValueError):
            pass

    source = _load_audio(audio_path, sample_rate)
    frame_count = max(1, math.ceil(len(source) * fps / sample_rate))
    full_mix_rms = _resample_features(_frame_rms(source, sample_rate, fps), frame_count, fps, sample_rate)

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
    # Phase is a gentle spatial drift, not a strobe.  The old coefficient
    # advanced the pattern by one radian per frame at peak vocals, which is
    # nearly ten full visual cycles per second at 60 FPS.
    phase_rate = 0.02 + 0.58 * (vocal.astype(np.float64) ** 2.2)
    phase = np.cumsum(phase_rate / max(float(fps), 1.0)).astype(np.float32)
    features = AudioFeatures(vocal, instrumental, phase, pitch, frame_count)
    if cache_path is not None:
        _atomic_save_features(cache_path, features)
    return features


def _decimal_precision(x_center: str, y_center: str, log10_zoom: float) -> int:
    def fractional_digits(value: str) -> int:
        value = value.lower().split("e", 1)[0]
        return len(value.split(".", 1)[1]) if "." in value else 0

    return max(
        50,
        32 + int(math.ceil(max(0.0, log10_zoom))),
        fractional_digits(x_center),
        fractional_digits(y_center),
    )


def _zoom_log(value: Any) -> float:
    """Return log10(value) without converting huge decimal zooms to float."""

    try:
        decimal_value = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"invalid zoom value: {value}") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise ValueError(f"zoom must be a finite positive decimal: {value}")
    exponent = decimal_value.adjusted()
    mantissa = decimal_value.scaleb(-exponent)
    return float(exponent) + math.log10(float(mantissa))


def _zoom_text(log10_zoom: float) -> bytes:
    """Encode 10**log10_zoom without overflowing Python's binary float."""

    if not math.isfinite(log10_zoom):
        raise ValueError("zoom exponent must be finite")
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
) -> tuple[Any, Any]:
    """Return a high-precision reference orbit as two NumPy arrays."""

    np = _require_numpy()
    precision = _decimal_precision(x_center, y_center, log10_zoom)
    real = np.empty(max_iter + 1, dtype=np.float64)
    imag = np.empty(max_iter + 1, dtype=np.float64)

    try:
        import mpmath as mp

        with mp.workdps(precision):
            cx = mp.mpf(x_center)
            cy = mp.mpf(y_center)
            zr = mp.mpf("0")
            zi = mp.mpf("0")
            float_limit = mp.mpf("1e300")
            for index in range(max_iter + 1):
                if abs(zr) > float_limit or abs(zi) > float_limit:
                    # The reference has escaped so far that its float64
                    # representation would overflow. Remaining pixels are
                    # handled as escaped by the perturbation loop.
                    real[index:] = math.inf
                    imag[index:] = math.inf
                    break
                real[index] = float(zr)
                imag[index] = float(zi)
                zr, zi = zr * zr - zi * zi + cx, 2 * zr * zi + cy
    except ImportError:
        # Decimal is slower but keeps the centre precise without making mpmath
        # a hard dependency for ordinary-depth renders.
        with localcontext() as context:
            context.prec = precision
            cx = Decimal(x_center)
            cy = Decimal(y_center)
            zr = Decimal(0)
            zi = Decimal(0)
            decimal_limit = Decimal("1e300")
            for index in range(max_iter + 1):
                if abs(zr) > decimal_limit or abs(zi) > decimal_limit:
                    real[index:] = math.inf
                    imag[index:] = math.inf
                    break
                real[index] = float(zr)
                imag[index] = float(zi)
                zr, zi = zr * zr - zi * zi + cx, Decimal(2) * zr * zi + cy

    return real, imag


def _view_offsets(width: int, height: int, log10_zoom: float) -> tuple[Any, Any]:
    np = _require_numpy()
    if log10_zoom > 300.0:
        raise RuntimeError("Python rendering cannot represent zooms beyond 10^300; use the native renderer")
    zoom = 10.0 ** log10_zoom
    view_height = 2.8 / zoom
    view_width = view_height * width / height
    x = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * view_width / width
    y = ((height - 1) / 2.0 - np.arange(height, dtype=np.float64)) * view_height / height
    return np.meshgrid(x, y)


def _smooth_escape(iteration: int, magnitude_squared: Any) -> Any:
    np = _require_numpy()
    magnitude = np.sqrt(np.maximum(magnitude_squared, 4.0000001))
    return iteration - np.log(np.log(magnitude)) / math.log(2.0)


def _render_direct(
    width: int,
    height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
) -> Any:
    """Vectorised float64 renderer for the shallow part of the journey."""

    np = _require_numpy()
    x_offset, y_offset = _view_offsets(width, height, log10_zoom)
    # Float conversion is safe here because this path is used before a deep
    # zoom, where the pixel spacing is still much larger than float64 ulps.
    real = (float(x_center) + x_offset).ravel()
    imag = (float(y_center) + y_offset).ravel()
    z_real = np.zeros_like(real)
    z_imag = np.zeros_like(imag)
    escaped = np.zeros(real.shape, dtype=bool)
    smooth = np.full(real.shape, float(max_iter), dtype=np.float32)

    # Analytic tests remove a large amount of work in the main cardioid and
    # period-2 bulb, which is especially helpful for high-resolution masters.
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
        next_real = active_real * active_real - active_imag * active_imag + real[active_indices]
        next_imag = 2.0 * active_real * active_imag + imag[active_indices]
        magnitude_squared = next_real * next_real + next_imag * next_imag
        newly_escaped = magnitude_squared > 4.0
        escaped_indices = active_indices[newly_escaped]
        smooth[escaped_indices] = _smooth_escape(
            iteration, magnitude_squared[newly_escaped]
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
) -> Any:
    """Render using a high-precision centre orbit and pixel perturbations."""

    np = _require_numpy()
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
        newly_escaped = magnitude_squared > 4.0
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


def _render_native(
    width: int,
    height: int,
    log10_zoom: float,
    x_center: str,
    y_center: str,
    max_iter: int,
    native_threads: int,
) -> Any:
    np = _require_numpy()
    library = _get_native_library()
    if library is None:
        raise RuntimeError("native renderer is unavailable; run `make` inside `nix-shell`")

    output = np.empty((height, width), dtype=np.float32)
    zoom_text = _zoom_text(log10_zoom)
    precision_bits = max(
        256,
        int(_decimal_precision(x_center, y_center, log10_zoom) * math.log2(10.0)) + 32,
    )
    use_perturbation = int(log10_zoom >= 12.0)
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
) -> tuple[Any, Any]:
    """Prepare one reusable native reference orbit and BLA table."""

    library = _get_native_library()
    if library is None:
        raise RuntimeError("native renderer is unavailable; run `make` inside `nix-shell`")
    precision_bits = max(
        256,
        int(_decimal_precision(x_center, y_center, log10_zoom) * math.log2(10.0)) + 32,
    )
    handle = library.fractal_create_reference(
        x_center.encode("ascii"),
        y_center.encode("ascii"),
        _zoom_text(log10_zoom if bla_log10_zoom is None else bla_log10_zoom),
        max_iter,
        precision_bits,
        series_order,
    )
    if not handle:
        message = library.fractal_last_error() or b"unknown native reference error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return library, handle


def _render_native_reference(
    width: int,
    height: int,
    log10_zoom: float,
    max_iter: int,
    native_threads: int,
    native_reference: Any,
    series_order: int,
    series_block: int,
) -> Any:
    np = _require_numpy()
    library = _get_native_library()
    if library is None:
        raise RuntimeError("native renderer is unavailable")

    output = np.empty((height, width), dtype=np.float32)
    zoom_text = _zoom_text(log10_zoom)
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
) -> Any:
    """Render with the C-ABI backend, falling back to the Python backend."""

    global _native_notice_printed
    if renderer not in {"auto", "native", "python"}:
        raise ValueError(f"unknown renderer: {renderer}")
    if renderer != "python":
        try:
            # Long-double direct iteration remains accurate while the pixel
            # spacing is comfortably above the decimal precision of the
            # selected centre.  Use scaled perturbation for genuinely deep
            # frames, where adding offsets directly would lose detail.
            if native_reference is not None and log10_zoom >= 12.0:
                return _render_native_reference(
                    width,
                    height,
                    log10_zoom,
                    max_iter,
                    native_threads,
                    native_reference,
                    series_order,
                    series_block,
                )
            if log10_zoom >= 12.0 and native_reference is None:
                raise RuntimeError("deep native rendering needs a prepared reference")
            return _render_native(width, height, log10_zoom, x_center, y_center, max_iter, native_threads)
        except RuntimeError as error:
            if renderer == "native" or log10_zoom >= 12.0:
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
        return _render_direct(width, height, log10_zoom, x_center, y_center, max_iter)
    return _render_perturbed(width, height, log10_zoom, x_center, y_center, max_iter)


def max_iterations(
    log10_zoom: float,
    iteration_base: int,
    iterations_per_decade: int,
    iteration_cap: int,
) -> int:
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
    if max_log <= start_log:
        return np.full(instrumental.shape, start_log, dtype=np.float64)
    span = max_log - start_log
    envelope = np.clip(np.asarray(instrumental, dtype=np.float64), 0.0, 1.0)
    if envelope.size == 1:
        return np.asarray([max_log], dtype=np.float64)
    loudness = envelope ** 2.2
    # The default is intentionally slightly negative.  This is a velocity in
    # logarithmic zoom space, not a zoom position, so it makes quiet passages
    # pull back a little while strong beats still consume most of the travel.
    drive = float(quiet_speed) + (1.0 + punch) * loudness
    cumulative = np.concatenate(([0.0], np.cumsum(drive[:-1])))
    # The last sample has no following frame over which to advance. Normalize
    # by the signed travelled intervals so the final video frame reaches
    # max_zoom even when the path briefly moves backwards.  A completely
    # silent track has no positive punches; make that degenerate case crawl
    # forward rather than reversing the entire movie.
    total = float(np.sum(drive[:-1])) if drive.size > 1 else 1.0
    if total <= 1.0e-6:
        drive = drive - float(np.min(drive)) + 1.0e-3
        cumulative = np.concatenate(([0.0], np.cumsum(drive[:-1])))
        total = max(float(np.sum(drive[:-1])), 1.0e-6) if drive.size > 1 else 1.0
    planned = start_log + span * cumulative / total
    return np.clip(planned, start_log, max_log)


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


def _zoom_chunks(zooms: Any, keyframe_factor: float) -> Any:
    """Partition an arbitrary zoom path into crop-safe keyframe ranges.

    A chunk is represented by its minimum zoom, because that is the widest
    source field needed to crop every frame in the chunk.  Tracking both
    extrema is important now that the audio response is allowed to pull back
    slightly between punches; the old monotonic-only partition could silently
    request a crop factor below one in that case.
    """

    limit = math.log10(max(1.05, keyframe_factor))
    start = 0
    total = len(zooms)
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
    step = math.log10(max(1.05, float(keyframe_factor)))
    minimum = float(min(float(value) for value in zooms))
    maximum = float(max(float(value) for value in zooms))
    origin = math.floor(minimum / step) * step
    level_count = max(0, int(math.ceil((maximum - origin) / step - 1.0e-12)))
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
    fallback_field: Optional[Any] = None,
    fallback_zoom_factor: float = 2.0,
    fallback_max_iter: Optional[int] = None,
) -> Optional[Any]:
    """Render a deep tile with an adaptive secondary-reference grid.

    The first pass uses four nearby references. A narrow parabolic feature
    can still make one of those cells pathological, so the native renderer
    accepts a short, opt-in deadline and marks unfinished pixels with NaN.
    Only those cells are recursively split into four smaller reference views;
    easy cells are never rendered again. The final subdivision gets a bounded
    exact perturbation retry with BLA disabled. A tile-level deadline prevents
    a pathological cluster from consuming the whole render. If it fires,
    unresolved pixels are recovered from the already-rendered parent tile at
    the matching world-space crop, keeping the result spatially continuous
    instead of painting a whole cell with one median iteration value.

    ``None`` means that the ordinary single-reference path should be used.
    """

    if renderer not in {"auto", "native"} or native_library is None:
        return None
    if log10_zoom < ATLAS_LOCAL_REFERENCE_MIN_LOG:
        return None
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
    previous_cycle_setting = os.environ.get("FRACTAL_STRICT_CYCLE")
    previous_budget_setting = os.environ.get("FRACTAL_TIME_BUDGET_MS")
    previous_bla_setting = os.environ.get("FRACTAL_DISABLE_BLA")
    # A local reference can sit in an attracting basin while a nearby pixel
    # is still a very late escape.  Use the native strict Brent mode here: it
    # waits longer and requires a much tighter recurrence than the throughput
    # mode, while keeping BLA and normal escape checks enabled.
    os.environ["FRACTAL_STRICT_CYCLE"] = "1"
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
        cell_previous_budget = os.environ.get("FRACTAL_TIME_BUDGET_MS")
        cell_previous_bla = os.environ.get("FRACTAL_DISABLE_BLA")
        try:
            if final_retry:
                os.environ["FRACTAL_TIME_BUDGET_MS"] = str(
                    ATLAS_LOCAL_REFERENCE_FINAL_BUDGET_MS
                )
                os.environ["FRACTAL_DISABLE_BLA"] = "1"
            else:
                os.environ["FRACTAL_TIME_BUDGET_MS"] = str(
                    budget_ms(x1 - x0, y1 - y0)
                )
                if cell_previous_bla is None:
                    os.environ.pop("FRACTAL_DISABLE_BLA", None)
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
                ),
                dtype=np.float32,
            )
        finally:
            native_library.fractal_destroy_reference(reference)
            if cell_previous_budget is None:
                os.environ.pop("FRACTAL_TIME_BUDGET_MS", None)
            else:
                os.environ["FRACTAL_TIME_BUDGET_MS"] = cell_previous_budget
            if cell_previous_bla is None:
                os.environ.pop("FRACTAL_DISABLE_BLA", None)
            else:
                os.environ["FRACTAL_DISABLE_BLA"] = cell_previous_bla

    pending = [(cell, initial_divisions) for cell in initial_cells]
    completed_cells = 0
    refined_cells = 0
    final_retries = 0
    recovered_cells = 0
    recovered_pixels = 0
    tile_deadline = time.monotonic() + ATLAS_LOCAL_REFERENCE_TILE_BUDGET_MS / 1000.0

    def parent_fallback_view() -> Any:
        nonlocal parent_fallback
        if parent_fallback is not None:
            return parent_fallback
        if fallback_array is None:
            return None
        # A child atlas tile is one fixed zoom factor narrower than its
        # parent. Resample the parent's central crop once, lazily, only when
        # the deadline/recovery path is actually needed.
        parent_fallback = np.asarray(
            _crop_and_resize(
                fallback_array,
                render_width,
                render_height,
                max(float(fallback_zoom_factor), 1.0),
                "bilinear",
            ),
            dtype=np.float32,
        )
        if fallback_max_iter is not None:
            # Interior pixels in the parent are encoded at its iteration cap.
            # Translate that sentinel to the current tile's cap instead of
            # turning parent interiors into a false coloured escape band.
            parent_fallback = np.where(
                parent_fallback >= float(fallback_max_iter) - 0.5,
                float(max_iter),
                parent_fallback,
            ).astype(np.float32, copy=False)
        return parent_fallback

    def complete_with_recovery(
        cell: tuple[int, int, int, int, str, str],
        local_field: Optional[Any] = None,
    ) -> None:
        nonlocal recovered_cells, recovered_pixels
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

    try:
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
    finally:
        if previous_cycle_setting is None:
            os.environ.pop("FRACTAL_STRICT_CYCLE", None)
        else:
            os.environ["FRACTAL_STRICT_CYCLE"] = previous_cycle_setting
        if previous_budget_setting is None:
            os.environ.pop("FRACTAL_TIME_BUDGET_MS", None)
        else:
            os.environ["FRACTAL_TIME_BUDGET_MS"] = previous_budget_setting
        if previous_bla_setting is None:
            os.environ.pop("FRACTAL_DISABLE_BLA", None)
        else:
            os.environ["FRACTAL_DISABLE_BLA"] = previous_bla_setting


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
    fallback_field: Optional[Any] = None,
    fallback_zoom_factor: float = 2.0,
    fallback_max_iter: Optional[int] = None,
    durable_cache: bool = False,
    cache_evictor: Optional["_CacheEvictor"] = None,
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
            cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if cached.shape == (render_height, render_width):
                if cache_evictor is not None:
                    cache_evictor.touch(cache_path)
                print(f"Using cached atlas tile {level} ({cache_path.name}).", flush=True)
                return cached
        except (OSError, ValueError):
            pass

    print(
        f"Rendering atlas tile {level} at zoom {_zoom_label(log_zoom)}, "
        f"iterations {max_iter}",
        flush=True,
    )
    field = None
    if native_library is not None:
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
                fallback_field=fallback_field,
                fallback_zoom_factor=fallback_zoom_factor,
                fallback_max_iter=fallback_max_iter,
            )
        except RuntimeError as error:
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
        )
    if cache_path is not None:
        _atomic_save_field(cache_path, field, durable=durable_cache)
        if cache_evictor is not None:
            cache_evictor.observe(cache_path)
        try:
            return np.load(cache_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            pass
    return np.asarray(field, dtype=np.float32)


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
    status = library.fractal_atlas_colourise(
        parent.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        parent_width,
        parent_height,
        int(parent_iter),
        child_pointer,
        child_width,
        child_height,
        child_max_iter,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output_width,
        output_height,
        float(parent_zoom),
        float(child_fraction),
        palette_max_iter,
        float(phase),
        float(vocal),
        float(instrumental),
        float(pitch),
        native_threads,
    )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native atlas colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return output


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
) -> Any:
    """Compose a frame from a parent tile and its central child tile.

    ``child_fraction`` is the fraction of the output frame covered by the
    child. At the beginning of an atlas interval it is 1/factor; at the end
    it reaches one. The child is therefore rendered only into its visible
    central rectangle instead of being expanded into a full temporary frame.
    """

    np = _require_numpy()
    if (
        native_library is not None
        and hasattr(native_library, "fractal_atlas_colourise")
        and resample == "bilinear"
        and palette_name == "aurora"
    ):
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
    parent_view = np.array(
        _crop_and_resize(
            parent,
            output_width,
            output_height,
            max(float(parent_zoom), 1.0),
            resample,
        ),
        dtype=np.float32,
        copy=True,
    )
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
        )
    child_fraction = min(float(child_fraction), 1.0)
    if child_fraction >= 0.999999:
        child_view = _crop_and_resize(
            child,
            output_width,
            output_height,
            1.0,
            resample,
        )
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
        )

    child_width = max(1, int(round(output_width * child_fraction)))
    child_height = max(1, int(round(output_height * child_fraction)))
    child_view = _crop_and_resize(
        child,
        child_width,
        child_height,
        1.0,
        resample,
    )
    left = (output_width - child_width) // 2
    top = (output_height - child_height) // 2
    right = left + child_width
    bottom = top + child_height
    region = parent_view[top:bottom, left:right]
    feather = min(16, child_width // 8, child_height // 8)
    if feather < 2:
        region[...] = child_view
        return _colourise_view(
            parent_view,
            max(int(parent_iter), int(child_iter)),
            phase,
            vocal,
            instrumental,
            native_library,
            native_threads,
            palette_name,
            pitch,
        )

    yy, xx = np.ogrid[:child_height, :child_width]
    edge_distance = np.minimum(
        np.minimum(xx, yy),
        np.minimum(child_width - 1 - xx, child_height - 1 - yy),
    )
    alpha = np.clip(edge_distance.astype(np.float32) / float(feather), 0.0, 1.0)
    parent_inside = ~np.isfinite(region) | (region >= float(parent_iter) - 0.5)
    child_inside = ~np.isfinite(child_view) | (child_view >= float(child_iter) - 0.5)
    blended = (
        region.astype(np.float32) * (1.0 - alpha)
        + child_view.astype(np.float32) * alpha
    )
    blended = np.where(parent_inside & ~child_inside, child_view, blended)
    blended = np.where(child_inside & ~parent_inside, region, blended)
    effective_iter = max(int(parent_iter), int(child_iter))
    blended = np.where(
        parent_inside & child_inside,
        float(effective_iter),
        blended,
    )
    region[...] = np.asarray(blended, dtype=np.float32)
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
    )


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
    native_reference: Any,
    native_threads: int,
    series_order: int,
    series_block: int,
    render_width: int,
    render_height: int,
    resample: str,
    palette: str,
    cache_dir: Optional[Path],
    cache_limit_mb: float,
    durable_cache: bool,
    cache_identity: str,
) -> None:
    """Render a song through a fixed nested tile atlas.

    The old renderer tied source fields to audio-dependent chunk boundaries.
    This path ties them only to absolute logarithmic zoom, so a tile is
    rendered once and can be reused by every camera path through the same
    centre. A bounded three-tile window is retained in memory so both forward
    and audio-driven reverse zooms remain cheap.
    """

    np = _require_numpy()
    origin, step, level_count = _atlas_geometry(zooms, keyframe_factor)
    factor = 10.0 ** step
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

    def tile_log(level: int) -> float:
        return origin + step * float(level)

    def tile_iter(level: int) -> int:
        cached = tile_iterations.get(level)
        if cached is not None:
            return cached
        current_log = tile_log(level)
        end_log = min(maximum_log, current_log + step)
        value = max_iterations(
            end_log,
            iteration_base,
            iterations_per_decade,
            iteration_cap,
        )
        tile_iterations[level] = value
        return value

    def get_tile(level: int) -> Any:
        nonlocal tile_seconds
        if level < 0 or level > level_count:
            return None
        if level in tile_cache:
            return tile_cache[level]
        started = time.perf_counter()
        parent_field = tile_cache.get(level - 1)
        parent_max_iter = (
            tile_iterations.get(level - 1)
            if parent_field is not None
            else None
        )
        tile_cache[level] = _atlas_tile_field(
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
            native_reference=native_reference,
            native_threads=native_threads,
            native_library=native_library,
            fallback_field=parent_field,
            fallback_zoom_factor=factor,
            fallback_max_iter=parent_max_iter,
            durable_cache=durable_cache,
            cache_evictor=cache_evictor,
        )
        tile_seconds += time.perf_counter() - started
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

    process = None
    frame_seconds = 0.0
    render_started = time.perf_counter()
    active_level = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
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
                _trim_atlas_memory_cache(tile_cache, level)

            parent_log = tile_log(level)
            parent = get_tile(level)
            child = get_tile(level + 1) if level < level_count else None
            parent_zoom = max(1.0, 10.0 ** (frame_log_zoom - parent_log))
            child_fraction = min(1.0, parent_zoom / factor) if child is not None else 0.0
            parent_max_iter = tile_iter(level)
            child_max_iter = tile_iter(level + 1) if child is not None else None
            phase = float(features.phase[frame_index])
            vocal = float(features.vocal[frame_index])
            instrumental = float(features.instrumental[frame_index])
            pitch = float(features.pitch[frame_index])
            rgb = _atlas_colour_frame(
                parent,
                child,
                width,
                height,
                parent_zoom,
                child_fraction,
                parent_max_iter,
                child_max_iter,
                phase,
                vocal,
                instrumental,
                native_library,
                native_threads,
                resample,
                palette,
                pitch,
            )
            assert process.stdin is not None
            process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8))
            frame_seconds += time.perf_counter() - frame_started
            if frame_index % max(1, fps * 5) == 0:
                print(f"  encoded {100.0 * frame_index / total_frames:5.1f}%")

        assert process.stdin is not None
        process.stdin.close()
        process.stdin = None
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        os.replace(temporary_output, output_path)
        elapsed = time.perf_counter() - render_started
        print(
            f"Atlas timing: tiles {tile_seconds:.2f}s, frame/crop/pipe "
            f"{frame_seconds:.2f}s, total {elapsed:.2f}s.",
            flush=True,
        )
    except BrokenPipeError as exc:
        if process is not None:
            process.kill()
            process.wait()
        raise RuntimeError("ffmpeg stopped while receiving video frames") from exc
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            process.wait()
        raise
    finally:
        if temporary_output.exists():
            try:
                temporary_output.unlink()
            except OSError:
                pass


@lru_cache(maxsize=8)
def _palette_basis(max_iter: int) -> tuple[Any, Any, float]:
    """Return the static cosine basis for one smooth-iteration palette."""

    np = _require_numpy()
    palette_size = min(65536, max(4096, int(max_iter) * 4))
    palette_field = np.linspace(0.0, float(max_iter), palette_size, dtype=np.float32)
    angle = np.asarray(0.19 * palette_field, dtype=np.float32)
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
    inside = field >= max_iter - 0.5
    # Restore the original three phase-offset channel waves. They produce the
    # characteristic blue/yellow gradient; pitch rotates both hues together.
    cosine_basis, sine_basis, scale = _palette_basis(int(max_iter))
    split = 0.7 + 3.0 * float(vocal) * float(vocal)
    brightness = 0.65 + 0.35 * float(instrumental)
    red_wave = cosine_basis * math.cos(phase) + sine_basis * math.sin(phase)
    green_wave = cosine_basis * math.cos(phase + split * 0.35) \
        + sine_basis * math.sin(phase + split * 0.35)
    blue_wave = cosine_basis * math.cos(phase + split) \
        + sine_basis * math.sin(phase + split)
    palette = np.empty((cosine_basis.size, 3), dtype=np.float32)
    palette[:, 0] = np.clip((0.5 - 0.5 * red_wave)
                            * (150.0 + 80.0 * float(vocal)) * brightness, 0.0, 255.0)
    palette[:, 1] = np.clip((0.5 - 0.5 * green_wave) * 180.0 * brightness, 0.0, 255.0)
    palette[:, 2] = np.clip((0.5 - 0.5 * blue_wave) * 210.0 * brightness, 0.0, 255.0)
    palette = np.clip(_rotate_hue_rgb(palette, pitch), 0.0, 255.0).astype(np.uint8)
    palette_indices = np.clip(field * scale, 0, palette.shape[0] - 1).astype(np.intp)
    rgb = palette[palette_indices]
    rgb[inside] = 0
    return rgb


@lru_cache(maxsize=16)
def _custom_palette(name: str, size: int = 4096) -> Any:
    """Build a compact RGB palette once; frame controls only index it."""

    np = _require_numpy()
    stops = {
        "fire": ((4, 0, 12), (90, 5, 8), (220, 45, 5), (255, 220, 80), (255, 255, 255)),
        "ocean": ((2, 5, 24), (5, 55, 125), (0, 180, 210), (140, 255, 235), (255, 255, 255)),
        "neon": ((10, 0, 35), (170, 0, 180), (0, 220, 255), (220, 255, 40), (255, 255, 255)),
        "sunset": ((18, 2, 30), (100, 8, 70), (235, 45, 55), (255, 150, 40), (255, 245, 180)),
        "mono": ((0, 0, 0), (80, 80, 80), (180, 180, 180), (255, 255, 255)),
    }
    if name not in stops:
        raise ValueError(f"unknown palette: {name}")
    anchors = np.linspace(0.0, 1.0, len(stops[name]), dtype=np.float32)
    positions = np.linspace(0.0, 1.0, size, dtype=np.float32)
    output = np.empty((size, 3), dtype=np.float32)
    for channel in range(3):
        output[:, channel] = np.interp(
            positions,
            anchors,
            [stop[channel] for stop in stops[name]],
        )
    return np.asarray(np.rint(output), dtype=np.uint8)


def _colourise_custom(
    field: Any,
    max_iter: int,
    phase: float,
    vocal: float,
    instrumental: float,
    palette_name: str,
    pitch: float = 0.5,
) -> Any:
    np = _require_numpy()
    palette = _custom_palette(palette_name)
    inside = field >= max_iter - 0.5
    position = np.asarray(field, dtype=np.float32) / max(float(max_iter), 1.0)
    position = np.mod(position + phase * 0.006 + vocal * 0.08 + float(pitch) * 0.12, 1.0)
    indices = np.clip(
        position * (palette.shape[0] - 1), 0, palette.shape[0] - 1
    ).astype(np.intp)
    rgb = palette[indices].astype(np.float32)
    saturation = 0.75 + 0.45 * float(vocal)
    brightness = 0.75 + 0.35 * float(instrumental)
    luminance = np.sum(
        rgb * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    rgb = luminance + (rgb - luminance) * saturation
    rgb = np.clip(rgb * brightness, 0.0, 255.0).astype(np.uint8)
    rgb[inside] = 0
    return rgb


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
) -> Any:
    """Fused centred bilinear crop and colour through the native C ABI."""

    np = _require_numpy()
    field = np.ascontiguousarray(field, dtype=np.float32)
    source_height, source_width = field.shape
    rgb = np.empty((output_height, output_width, 3), dtype=np.uint8)
    status = library.fractal_crop_colourise(
        field.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        source_width,
        source_height,
        rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output_width,
        output_height,
        zoom_factor,
        max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        native_threads,
    )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native crop/colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return rgb


def _atomic_save_field(path: Path, field: Any, *, durable: bool = False) -> None:
    """Write a keyframe atomically, optionally forcing it to stable storage.

    Atomic replacement is sufficient for normal resume behaviour and avoids a
    blocking fsync for every multi-megabyte atlas tile. ``durable`` is an
    explicit opt-in for users who need cache entries to survive a sudden power
    loss rather than merely an interrupted process.
    """

    np = _require_numpy()
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
        for pattern in self._PATTERNS:
            for path in self.cache_dir.glob(pattern):
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                sequence = self._next_sequence()
                self._entries[path] = (size, sequence)
                self._total_bytes += size
                heapq.heappush(self._heap, (sequence, path))

    def observe(self, path: Path) -> None:
        if not self.enabled:
            return
        self._scan_once()
        try:
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
    durable_cache: bool = False,
    cache_evictor: Optional[_CacheEvictor] = None,
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
            cached = np.load(cache_path, allow_pickle=False)
            if cached.shape == (render_height, render_width):
                if cache_evictor is not None:
                    cache_evictor.touch(cache_path)
                print(f"Using cached keyframe {cache_path.name}.", flush=True)
                return np.asarray(cached, dtype=np.float32)
        except (OSError, ValueError):
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
    )
    if cache_path is not None:
        _atomic_save_field(cache_path, field, durable=durable_cache)
        if cache_evictor is not None:
            cache_evictor.observe(cache_path)
    return np.asarray(field, dtype=np.float32)


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
) -> Any:
    """Colour an already-resampled scalar iteration field."""

    if native_library is not None and palette_name == "aurora":
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
    if palette_name != "aurora":
        return _colourise_custom(view, max_iter, phase, vocal, instrumental, palette_name, pitch)
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
) -> Any:
    if native_library is not None and resample == "bilinear" and palette_name == "aurora":
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
    view = _crop_and_resize(field, output_width, output_height, zoom_factor, resample)
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
    )


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
    video_codec: str = "libx264",
    crf: int = 18,
    keyframe_mode: str = "atlas",
    durable_cache: bool = False,
) -> None:
    np = _require_numpy()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the video, but it was not found on PATH")

    if keyframe_mode not in {"atlas", "legacy"}:
        raise ValueError(f"unknown keyframe mode: {keyframe_mode}")
    if keyframe_mode == "atlas":
        # Nested tiles already provide the missing factor-two density at the
        # centre of every interval. Keep quality presets as modest optional
        # supersampling instead of rendering every tile at factor squared.
        quality_settings = {
            "draft": (max(fractal_scale, 0.5), 0.0),
            "balanced": (max(fractal_scale, 1.0), 0.0),
            "quality": (max(fractal_scale, 1.0), 0.0),
            "extreme": (max(fractal_scale, 1.25), 0.0),
        }
    else:
        # Legacy full-field mode is retained for visual regression and for
        # comparing the new atlas against the previous renderer.
        quality_settings = {
            "draft": (max(fractal_scale, 0.25), 0.0),
            "balanced": (max(fractal_scale, 1.0), 0.12),
            "quality": (max(fractal_scale, keyframe_factor), 0.20),
            "extreme": (max(fractal_scale, keyframe_factor * 2.0), 0.28),
        }
    if quality not in quality_settings:
        raise ValueError(f"unknown quality preset: {quality}")
    source_scale, transition_fraction = quality_settings[quality]
    cpu_count = max(2, os.cpu_count() or 2)
    if native_threads == 0:
        native_threads = max(1, (cpu_count * 2) // 3)
    if encoder_threads == 0:
        encoder_threads = max(1, cpu_count // 3)
    render_width = max(16, int(round(width * render_scale * source_scale)))
    render_height = max(16, int(round(height * render_scale * source_scale)))
    temporary_output = output_path.with_name(
        f".{output_path.stem}.rendering-{os.getpid()}{output_path.suffix}"
    )
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
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
        "-c:v",
        video_codec,
        "-preset",
        video_preset,
        "-threads",
        str(encoder_threads),
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(temporary_output),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    total_frames = features.frame_count
    chunks = list(_zoom_chunks(zooms, keyframe_factor)) if keyframe_mode == "legacy" else []

    native_library = None
    native_reference = None
    active_renderer = renderer
    max_log_zoom = float(np.max(zooms))
    if renderer != "python":
        native_library = _get_native_library()
        if native_library is None:
            if renderer == "native":
                raise RuntimeError("native renderer is unavailable; run `make` inside `nix-shell`")
            active_renderer = "python"
        elif float(np.max(zooms)) >= 12.0:
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
                # Python routes frames below 12 decades through the direct
                # native path. Size the reusable BLA bound from the first
                # frame that actually uses this reference, not an unnecessarily
                # shallow e6 viewport that would make composition too
                # conservative.
                bla_start_log = max(12.0, float(np.min(zooms)))
                _, native_reference = _create_native_reference(
                    x_center,
                    y_center,
                    reference_iter,
                    float(np.max(zooms)),
                    series_order,
                    bla_start_log,
                )
                print(
                    f"Prepared native reference orbit ({reference_iter} iterations, "
                    f"scaled perturbation + BLA, max zoom {_zoom_label(float(np.max(zooms)))}).",
                    flush=True,
                )
            except RuntimeError as error:
                if renderer == "native" or float(np.max(zooms)) > 300.0:
                    raise
                print(f"Native deep reference unavailable ({error}); using Python fallback.")
                active_renderer = "python"
                native_library = None

    cache_identity = _field_renderer_cache_identity(active_renderer)
    print(
        f"Keyframe source: {render_width}x{render_height} ({quality}); "
        f"planned keyframes: "
        f"{_keyframe_count(zooms, keyframe_factor) if keyframe_mode == 'legacy' else _atlas_geometry(zooms, keyframe_factor)[2] + 1}; "
        f"mode: {keyframe_mode}; "
        f"field renderer: {active_renderer}; "
        f"native threads: {native_threads}; encoder threads: {encoder_threads}",
        flush=True,
    )

    if keyframe_mode == "atlas":
        try:
            _render_video_atlas(
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
                native_reference=native_reference,
                native_threads=native_threads,
                series_order=series_order,
                series_block=series_block,
                render_width=render_width,
                render_height=render_height,
                resample=resample,
                palette=palette,
                cache_dir=cache_dir,
                cache_limit_mb=cache_limit_mb,
                durable_cache=durable_cache,
                cache_identity=_field_renderer_cache_identity(active_renderer),
            )
        finally:
            if native_reference is not None and native_library is not None:
                native_library.fractal_destroy_reference(native_reference)
            if temporary_output.exists():
                try:
                    temporary_output.unlink()
                except OSError:
                    pass
        return

    process = None
    render_started = time.perf_counter()
    keyframe_seconds = 0.0
    frame_seconds = 0.0
    cache_evictor = _CacheEvictor(cache_dir, cache_limit_mb)

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
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
                    native_reference=native_reference,
                    native_threads=native_threads,
                    durable_cache=durable_cache,
                    cache_evictor=cache_evictor,
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
                        native_reference=native_reference,
                        native_threads=native_threads,
                        durable_cache=durable_cache,
                        cache_evictor=cache_evictor,
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
                vocal = float(features.vocal[frame_index])
                instrumental = float(features.instrumental[frame_index])
                pitch = float(features.pitch[frame_index])
                rgb = _colour_frame(
                    field,
                    width,
                    height,
                    relative_zoom,
                    current_iter,
                    phase,
                    vocal,
                    instrumental,
                    native_library,
                    native_threads,
                    resample,
                    palette,
                    pitch,
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
                            width,
                            height,
                            next_relative_zoom,
                            next_iter,
                            phase,
                            vocal,
                            instrumental,
                            native_library,
                            native_threads,
                            resample,
                            palette,
                            pitch,
                        )
                        rgb = np.asarray(
                            np.rint(
                                rgb.astype(np.float32) * (1.0 - alpha)
                                + next_rgb.astype(np.float32) * alpha
                            ),
                            dtype=np.uint8,
                        )
                assert process.stdin is not None
                # NumPy exposes a contiguous buffer here; writing it directly
                # avoids allocating one 0.75 MB bytes object per 500x500 frame.
                process.stdin.write(rgb)
                frame_seconds += time.perf_counter() - frame_started
                if frame_index % max(1, fps * 5) == 0:
                    print(f"  encoded {100.0 * frame_index / total_frames:5.1f}%")

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

        assert process.stdin is not None
        process.stdin.close()
        process.stdin = None
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        os.replace(temporary_output, output_path)
        elapsed = time.perf_counter() - render_started
        print(
            f"Timing: keyframes {keyframe_seconds:.2f}s, frame/crop/pipe "
            f"{frame_seconds:.2f}s, total {elapsed:.2f}s.",
            flush=True,
        )
    except BrokenPipeError as exc:
        assert process is not None
        process.kill()
        process.wait()
        raise RuntimeError("ffmpeg stopped while receiving video frames") from exc
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            process.wait()
        raise
    finally:
        if native_reference is not None and native_library is not None:
            native_library.fractal_destroy_reference(native_reference)
        if temporary_output.exists():
            try:
                temporary_output.unlink()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", default=DEFAULT_AUDIO, type=Path)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
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
    parser.add_argument("--x-center", default=DEFAULT_X_CENTER)
    parser.add_argument("--y-center", default=DEFAULT_Y_CENTER)
    parser.add_argument("--base-zoom", default="1.0", help="starting zoom, e.g. 1e0")
    parser.add_argument(
        "--max-zoom",
        default="1e32",
        help="final zoom; native scaled arithmetic supports decimal exponents up to about 9800",
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
        "--video-preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"),
        default="ultrafast",
        help="x264 speed/size tradeoff; ultrafast is the low-power default",
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        help="FFmpeg video encoder name, for example libx264, libx265, or a hardware encoder",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="constant-rate-factor passed to the selected video encoder (0-51)",
    )
    parser.add_argument(
        "--resample",
        choices=("lanczos", "bilinear"),
        default="bilinear",
        help="crop resize filter; bilinear is fastest, lanczos selects bicubic quality filtering",
    )
    parser.add_argument(
        "--palette",
        choices=("aurora", "fire", "ocean", "neon", "sunset", "mono"),
        default="aurora",
        help="colour palette; aurora uses the fused native colour path",
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
        "--estimate",
        action="store_true",
        help="analyse audio and print keyframe workload without rendering",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height, and fps must be positive")
    try:
        base_log = _zoom_log(args.base_zoom)
        max_log = _zoom_log(args.max_zoom)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if max_log < base_log:
        raise SystemExit("max-zoom must be greater than or equal to base-zoom")
    if max_log > 9800.0:
        raise SystemExit("max-zoom is beyond the native scaled-exponent range; use a zoom below 1e9800")
    finite_options = (
        ("render-scale", args.render_scale),
        ("fractal-scale", args.fractal_scale),
        ("keyframe-factor", args.keyframe_factor),
        ("zoom-punch", args.zoom_punch),
        ("zoom-speed", args.zoom_speed),
        ("attack", args.attack),
        ("release", args.release),
        ("cache-limit-mb", args.cache_limit_mb),
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
    if args.iteration_cap < args.iteration_base:
        raise SystemExit("iteration-cap must be greater than iteration-base")
    if args.zoom_punch < 0:
        raise SystemExit("zoom-punch cannot be negative")
    if args.attack < 0 or args.release < 0:
        raise SystemExit("attack and release cannot be negative")
    if not 1 <= args.series_order <= 32:
        raise SystemExit("series-order must be between 1 and 32")
    if not 2 <= args.series_block <= 4096:
        raise SystemExit("series-block must be between 2 and 4096")
    if args.native_threads < 0:
        raise SystemExit("native-threads cannot be negative")
    if args.encoder_threads < 0:
        raise SystemExit("encoder-threads cannot be negative")
    if not args.codec or args.codec.startswith("-"):
        raise SystemExit("codec must be a non-empty FFmpeg encoder name")
    if not 0 <= args.crf <= 51:
        raise SystemExit("crf must be between 0 and 51")
    if args.cache_limit_mb < 0:
        raise SystemExit("cache-limit-mb cannot be negative")
    if args.sample_rate <= 0:
        raise SystemExit("sample-rate must be positive")
    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")

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
    )
    print(
        f"{features.frame_count} frames ({features.frame_count / args.fps:.1f}s) -> "
        f"{args.output}"
    )
    if args.estimate:
        if args.keyframe_mode == "atlas":
            estimate_scale = {
                "draft": max(args.fractal_scale, 0.5),
                "balanced": max(args.fractal_scale, 1.0),
                "quality": max(args.fractal_scale, 1.0),
                "extreme": max(args.fractal_scale, 1.25),
            }[args.quality]
        else:
            estimate_scale = {
                "draft": max(args.fractal_scale, 0.25),
                "balanced": max(args.fractal_scale, 1.0),
                "quality": max(args.fractal_scale, args.keyframe_factor),
                "extreme": max(args.fractal_scale, args.keyframe_factor * 2.0),
            }[args.quality]
        render_width = max(16, int(round(args.width * args.render_scale * estimate_scale)))
        render_height = max(16, int(round(args.height * args.render_scale * estimate_scale)))
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
    render_video(
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
    )
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
