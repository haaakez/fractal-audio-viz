#!/usr/bin/env python3
"""Create a tiny dependency-free WAV for smoke-testing the renderer."""

from __future__ import annotations

import argparse
import math
import os
import struct
import tempfile
import wave
from pathlib import Path


MAX_SECONDS = 60.0 * 60.0
MAX_SAMPLE_RATE = 768_000
MAX_FRAMES = 120_000_000


def _absolute_path(path: Path) -> Path:
    """Make a destination absolute without resolving its final symlink."""

    return Path(os.path.abspath(str(Path(path).expanduser())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=Path("test-tone.wav"))
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0.0
        or args.seconds > MAX_SECONDS
        or args.sample_rate <= 0
        or args.sample_rate > MAX_SAMPLE_RATE
    ):
        raise SystemExit(
            "seconds must be finite and between 0 and "
            f"{MAX_SECONDS / 60.0:.0f} minutes; sample-rate must be between 1 and "
            f"{MAX_SAMPLE_RATE:,}"
        )
    frame_estimate = args.seconds * args.sample_rate
    if not math.isfinite(frame_estimate) or frame_estimate > MAX_FRAMES:
        raise SystemExit(f"the tone may contain at most {MAX_FRAMES:,} samples")
    frames = max(1, int(round(frame_estimate)))
    output_path = args.output.expanduser()
    try:
        if output_path.is_symlink():
            raise SystemExit(f"output path must not be a symbolic link: {output_path}")
    except OSError as error:
        raise SystemExit(f"could not inspect output path {output_path}: {error}") from error
    output_path = _absolute_path(output_path)
    if output_path.exists() and output_path.is_dir():
        raise SystemExit(f"output path is a directory: {output_path}")
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise SystemExit(f"output parent is not a directory: {output_path.parent}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"could not create output directory: {error}") from error
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        with wave.open(str(temporary_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(args.sample_rate)
            for index in range(frames):
                time_value = index / args.sample_rate
                pulse = 0.35 + 0.65 * (
                    0.5 + 0.5 * math.sin(2.0 * math.pi * 2.0 * time_value)
                )
                sample = pulse * (
                    0.45 * math.sin(2.0 * math.pi * 220.0 * time_value)
                    + 0.25 * math.sin(2.0 * math.pi * 330.0 * time_value)
                )
                output.writeframesraw(
                    struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767))
                )
        os.replace(temporary_path, output_path)
    except (OSError, wave.Error) as error:
        raise SystemExit(f"could not write tone {output_path}: {error}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
