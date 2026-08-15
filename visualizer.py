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
KEYFRAME_CACHE_SCHEMA = "bla-keyframe-blend-colour-v5"
AUDIO_CACHE_SCHEMA = "audio-controls-v2"

_native_library: Any = None
_native_checked = False
_native_notice_printed = False


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
            if library.fractal_abi_version() != 6:
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
                ctypes.c_int,
            ]
            library.fractal_crop_colourise.restype = ctypes.c_int
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
            _native_library = library
            return library
        except OSError:
            # Missing MPFR/OpenMP runtime libraries should not prevent the
            # Python implementation from running.
            continue
    return None


@dataclass
class AudioFeatures:
    """Frame-aligned controls used by the visual animation."""

    vocal: Any
    instrumental: Any
    phase: Any
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
    cache_dir: Optional[Path],
) -> Optional[Path]:
    if cache_dir is None:
        return None
    digest = hashlib.sha256()
    digest.update(AUDIO_CACHE_SCHEMA.encode("ascii"))
    digest.update(str(sample_rate).encode("ascii"))
    digest.update(str(fps).encode("ascii"))
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
) -> AudioFeatures:
    """Load one song and produce frame-aligned vocal/instrument controls."""

    np = _require_numpy()
    cache_path = _audio_cache_path(audio_path, sample_rate, fps, separation, cache_dir)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                vocal = np.asarray(cached["vocal"], dtype=np.float32)
                instrumental = np.asarray(cached["instrumental"], dtype=np.float32)
                phase = np.asarray(cached["phase"], dtype=np.float32)
                frame_count = int(cached["frame_count"])
            if (
                frame_count > 0
                and vocal.shape == (frame_count,)
                and instrumental.shape == (frame_count,)
                and phase.shape == (frame_count,)
            ):
                print(f"Using cached audio controls {cache_path.name}.")
                return AudioFeatures(vocal, instrumental, phase, frame_count)
        except (OSError, KeyError, ValueError):
            pass

    source = _load_audio(audio_path, sample_rate)
    frame_count = max(1, math.ceil(len(source) * fps / sample_rate))
    full_mix_rms = _resample_features(_frame_rms(source, sample_rate, fps), frame_count, fps, sample_rate)

    with tempfile.TemporaryDirectory(prefix="fractal-demucs-") as temp:
        stems = None
        if separation in {"auto", "demucs"}:
            stems = _demucs_stems(audio_path, Path(temp), separation)

        if stems is not None:
            vocals = _load_audio(stems[0], sample_rate)
            instruments = _load_audio(stems[1], sample_rate)
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
    full_mix = _smooth(_normalise(full_mix_rms), 5)

    vocal_detected = _component_has_signal(vocal_signal, full_mix_rms)
    instrumental_detected = _component_has_signal(instrumental_signal, full_mix_rms)
    separated_controls = (
        vocal_detected
        and instrumental_detected
        and _controls_are_distinct(vocal_signal, instrumental_signal)
    )
    if separated_controls:
        vocal = _smooth(_normalise(vocal_signal), 5)
        instrumental = _smooth(_normalise(instrumental_signal), 7)
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
    phase = np.cumsum((vocal.astype(np.float64) ** 2.5) * 60.0 / fps).astype(np.float32)
    features = AudioFeatures(vocal, instrumental, phase, frame_count)
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
    series_order: int = 8,
    series_block: int = 32,
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
) -> Any:
    """Make loud instrumental moments advance the logarithmic zoom faster.

    The total travel is still exactly ``start_zoom`` to ``max_zoom``.  The
    loudness changes the local velocity, so a loud beat consumes more of that
    zoom distance and produces a visibly larger punch than a quiet passage.
    The envelope is already smoothed during analysis; the power curve makes
    the response musical instead of letting low-level noise move the camera.
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
    # A quiet signal still moves smoothly, while a loud signal gets a larger
    # share of the logarithmic zoom distance.  Normalising by the sum below
    # keeps the final zoom bounded regardless of the audio or punch setting.
    drive = 0.15 + (1.0 + punch) * loudness
    cumulative = np.concatenate(([0.0], np.cumsum(drive[:-1])))
    # The last sample has no following frame over which to advance. Normalize
    # by the travelled intervals so the final video frame reaches max_zoom.
    total = max(float(np.sum(drive[:-1])), 1e-12) if drive.size > 1 else 1.0
    return start_log + span * cumulative / total


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


def _keyframe_count(zooms: Any, keyframe_factor: float) -> int:
    limit = math.log10(max(1.05, keyframe_factor))
    count = 0
    index = 0
    total = len(zooms)
    while index < total:
        start = float(zooms[index])
        index += 1
        while index < total and float(zooms[index]) - start < limit:
            index += 1
        count += 1
    return count


@lru_cache(maxsize=8)
def _palette_basis(max_iter: int) -> tuple[Any, Any, float]:
    """Return the static cosine basis for one smooth-iteration palette."""

    np = _require_numpy()
    palette_size = min(65536, max(4096, int(max_iter) * 4))
    palette_field = np.linspace(0.0, float(max_iter), palette_size, dtype=np.float32)
    angle = np.asarray(0.19 * palette_field, dtype=np.float32)
    return np.cos(angle), np.sin(angle), (palette_size - 1) / max(float(max_iter), 1.0)


def _colourise(field: Any, max_iter: int, phase: float, vocal: float, instrumental: float) -> Any:
    np = _require_numpy()
    inside = field >= max_iter - 0.5
    # Colour changes every video frame, but it only depends on the smooth
    # iteration value.  Precomputing sin/cos(angle) once per keyframe lets a
    # frame rotate the palette with just six scalar trig calls, rather than
    # evaluating three 65k-element cosine arrays every time.
    cosine_basis, sine_basis, scale = _palette_basis(int(max_iter))
    split = 0.7 + 3.0 * vocal * vocal
    brightness = 0.65 + 0.35 * instrumental
    palette = np.empty((cosine_basis.size, 3), dtype=np.uint8)
    wave = np.empty_like(cosine_basis)
    for channel, offset, gain in (
        (0, phase, 150.0 + 80.0 * vocal),
        (1, phase + split * 0.35, 180.0),
        (2, phase + split, 210.0),
    ):
        np.multiply(cosine_basis, math.cos(offset), out=wave)
        np.add(wave, sine_basis * math.sin(offset), out=wave)
        np.multiply(wave, -0.5, out=wave)
        np.add(wave, 0.5, out=wave)
        np.multiply(wave, gain * brightness, out=wave)
        np.clip(wave, 0.0, 255.0, out=wave)
        palette[:, channel] = wave
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
) -> Any:
    np = _require_numpy()
    palette = _custom_palette(palette_name)
    inside = field >= max_iter - 0.5
    position = np.asarray(field, dtype=np.float32) / max(float(max_iter), 1.0)
    position = np.mod(position + phase * 0.006 + vocal * 0.08, 1.0)
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
        native_threads,
    )
    if status != 0:
        message = library.fractal_last_error() or b"unknown native crop/colourizer error"
        raise RuntimeError(message.decode("utf-8", errors="replace"))
    return rgb


def _atomic_save_field(path: Path, field: Any) -> None:
    """Write a keyframe without leaving a half-written cache entry."""

    np = _require_numpy()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, np.asarray(field, dtype=np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


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
        _atomic_save_field(cache_path, field)
    return np.asarray(field, dtype=np.float32)


def _prune_cache(cache_dir: Optional[Path], limit_mb: float) -> None:
    """Optionally bound cache growth, only within the explicitly chosen dir."""

    if cache_dir is None or limit_mb <= 0.0:
        return
    limit_bytes = int(limit_mb * 1024 * 1024)
    files = [
        path
        for pattern in ("keyframe-*.npy", "audio-*.npz")
        for path in cache_dir.glob(pattern)
        if path.is_file()
    ]
    total = sum(path.stat().st_size for path in files)
    if total <= limit_bytes:
        return
    for path in sorted(files, key=lambda item: item.stat().st_mtime):
        if total <= limit_bytes:
            break
        try:
            size = path.stat().st_size
            path.unlink()
            total -= size
        except OSError:
            continue


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
        )
    view = _crop_and_resize(field, output_width, output_height, zoom_factor, resample)
    if native_library is not None and palette_name == "aurora":
        return _colourise_native(
            view,
            max_iter,
            phase,
            vocal,
            instrumental,
            native_threads,
            native_library,
        )
    if palette_name != "aurora":
        return _colourise_custom(view, max_iter, phase, vocal, instrumental, palette_name)
    return _colourise(view, max_iter, phase, vocal, instrumental)


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
) -> None:
    np = _require_numpy()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the video, but it was not found on PATH")

    quality_settings = {
        # Draft intentionally permits undersampling for quick musical timing
        # previews.  All other modes retain at least one output sample per
        # output pixel at the keyframe boundary.
        "draft": (max(fractal_scale, 0.25), 0.0),
        "balanced": (max(fractal_scale, 1.0), 0.12),
        # A factor-two keyframe needs two samples per output dimension if the
        # crop itself is to remain native-resolution at the end of its span.
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
        "libx264",
        "-preset",
        video_preset,
        "-threads",
        str(encoder_threads),
        "-crf",
        "18",
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
    max_zoom_factor = max(1.05, keyframe_factor)
    max_log_factor = math.log10(max_zoom_factor)

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
                # The BLA input bound must cover the shallowest deep frame,
                # not the final microscopic frame.  Using max(zooms) would
                # disable every BLA lookup until the very end of the movie.
                bla_start_log = max(6.0, float(np.min(zooms)))
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
        f"planned keyframes: {_keyframe_count(zooms, keyframe_factor)}; "
        f"field renderer: {active_renderer}; "
        f"native threads: {native_threads}; encoder threads: {encoder_threads}",
        flush=True,
    )

    def chunk_end_for(start: int) -> int:
        end = start + 1
        start_zoom = float(zooms[start])
        while end < total_frames and float(zooms[end]) - start_zoom < max_log_factor:
            end += 1
        return end

    process = None
    render_started = time.perf_counter()
    keyframe_seconds = 0.0
    frame_seconds = 0.0

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        frame_index = 0
        while frame_index < total_frames:
            chunk_start = frame_index
            chunk_log_zoom = float(zooms[chunk_start])
            chunk_end = chunk_end_for(chunk_start)
            chunk_last_log_zoom = float(zooms[chunk_end - 1])

            # Budget for the deepest crop represented by this field, rather
            # than the first frame of the chunk.  This removes the iteration
            # pop that previously occurred exactly when a keyframe changed.
            current_iter = max_iterations(
                chunk_last_log_zoom,
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
            )
            _prune_cache(cache_dir, cache_limit_mb)

            next_field = None
            next_log_zoom = None
            next_iter = None
            if transition_fraction > 0.0 and chunk_end < total_frames:
                next_log_zoom = float(zooms[chunk_end])
                next_end = chunk_end_for(chunk_end)
                next_iter = max_iterations(
                    float(zooms[next_end - 1]),
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
                )
                _prune_cache(cache_dir, cache_limit_mb)
            keyframe_seconds += time.perf_counter() - keyframe_started

            for frame_index in range(chunk_start, chunk_end):
                frame_started = time.perf_counter()
                relative_zoom = 10.0 ** (float(zooms[frame_index]) - chunk_log_zoom)
                phase = float(features.phase[frame_index])
                vocal = float(features.vocal[frame_index])
                instrumental = float(features.instrumental[frame_index])
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
                )
                if next_field is not None and next_log_zoom is not None and next_iter is not None:
                    boundary_span = max(next_log_zoom - chunk_log_zoom, 1e-12)
                    progress = (float(zooms[frame_index]) - chunk_log_zoom) / boundary_span
                    blend_start = 1.0 - transition_fraction
                    alpha = np.clip(
                        (progress - blend_start) / transition_fraction,
                        0.0,
                        1.0,
                    )
                    if alpha > 0.0:
                        next_rgb = _colour_frame(
                            next_field,
                            width,
                            height,
                            1.0,
                            next_iter,
                            phase,
                            vocal,
                            instrumental,
                            native_library,
                            native_threads,
                            resample,
                            palette,
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

            del field
            if next_field is not None:
                del next_field
            # The for-loop leaves frame_index at chunk_end - 1. Advance the
            # outer render loop or the final chunk is rendered forever.
            frame_index = chunk_end

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
            "draft is fastest; balanced avoids undersampling; quality renders "
            "factor-sized keyframes; extreme supersamples further"
        ),
    )
    parser.add_argument(
        "--keyframe-factor",
        type=float,
        default=2.0,
        help="maximum zoom change between high-resolution keyframes",
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
        "--series-order",
        type=int,
        default=3,
        help="native BLA polynomial degree (1-3; values 4-32 remain ABI-compatible)",
    )
    parser.add_argument(
        "--series-block",
        type=int,
        default=256,
        help="maximum BLA block length (2-4096)",
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
    if not 1 <= args.series_order <= 32:
        raise SystemExit("series-order must be between 1 and 32")
    if args.series_block < 2:
        raise SystemExit("series-block must be at least 2")
    if args.native_threads < 0:
        raise SystemExit("native-threads cannot be negative")
    if args.encoder_threads < 0:
        raise SystemExit("encoder-threads cannot be negative")
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
    )
    print(f"Audio analysis: {time.perf_counter() - audio_started:.2f}s", flush=True)
    zooms = _zoom_plan(
        features.instrumental,
        args.base_zoom,
        args.max_zoom,
        args.zoom_punch,
    )
    print(
        f"{features.frame_count} frames ({features.frame_count / args.fps:.1f}s) -> "
        f"{args.output}"
    )
    if args.estimate:
        estimate_scale = {
            "draft": max(args.fractal_scale, 0.25),
            "balanced": max(args.fractal_scale, 1.0),
            "quality": max(args.fractal_scale, args.keyframe_factor),
            "extreme": max(args.fractal_scale, args.keyframe_factor * 2.0),
        }[args.quality]
        render_width = max(16, int(round(args.width * args.render_scale * estimate_scale)))
        render_height = max(16, int(round(args.height * args.render_scale * estimate_scale)))
        print(
            f"Estimate: {render_width}x{render_height} fractal source, "
            f"{_keyframe_count(zooms, args.keyframe_factor)} keyframes, "
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
    )
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
