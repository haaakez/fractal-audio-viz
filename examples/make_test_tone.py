#!/usr/bin/env python3
"""Create a tiny dependency-free WAV for smoke-testing the renderer."""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=Path("test-tone.wav"))
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()
    if args.seconds <= 0 or args.sample_rate <= 0:
        raise SystemExit("seconds and sample-rate must be positive")
    frames = int(round(args.seconds * args.sample_rate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(args.sample_rate)
        for index in range(frames):
            time_value = index / args.sample_rate
            pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 2.0 * time_value))
            sample = pulse * (
                0.45 * math.sin(2.0 * math.pi * 220.0 * time_value)
                + 0.25 * math.sin(2.0 * math.pi * 330.0 * time_value)
            )
            output.writeframesraw(
                struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767))
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
