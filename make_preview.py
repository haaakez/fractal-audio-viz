#!/usr/bin/env python3
"""Turn a render into a small GIF or silent social-media preview.

With no input, the newest filename containing ``4k`` and ``e150`` is chosen.
If no such filename exists, the newest video in the selected directory is
used and the choice is reported.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}


def _latest_video(directory: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"no video files found in {directory}")
    preferred = [
        path for path in candidates
        if "4k" in path.stem.casefold()
        and any(marker in path.stem.casefold() for marker in ("e150", "e-150", "150"))
    ]
    if preferred:
        return preferred[0]
    print("No filename matching 4k/e150 was found; using the newest video.")
    return candidates[0]


def _output_path(input_path: Path, requested: Path | None, format_name: str) -> Path:
    if requested is not None:
        return requested
    suffix = ".gif" if format_name == "gif" else ".mp4"
    return input_path.with_name(f"{input_path.stem}-preview{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make a short GIF or MP4 preview from a rendered video."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="source video; omit it to choose the newest 4K/e150 render",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="directory searched when input is omitted (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help="preview format (default: gif)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="clip duration in seconds")
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="maximum preview width; aspect ratio is preserved",
    )
    parser.add_argument("--fps", type=int, default=12, help="GIF/preview frame rate")
    parser.add_argument(
        "--with-audio",
        action="store_true",
        help="keep audio for MP4 output; GIF output is always silent",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.start < 0.0 or args.duration <= 0.0:
        raise SystemExit("start must be non-negative and duration must be positive")
    if args.width < 2 or args.fps < 1:
        raise SystemExit("width must be at least 2 and fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")

    input_path = args.input or _latest_video(args.directory)
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input video not found: {input_path}")
    output_path = _output_path(input_path, args.output, args.format).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale='min(iw,{args.width})':-2:flags=lanczos"

    if args.format == "gif":
        filter_graph = (
            f"fps={args.fps},{scale},split[s0][s1];"
            "[s0]palettegen=stats_mode=diff:max_colors=256[p];"
            "[s1][p]paletteuse=dither=sierra2_4a"
        )
        command = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(args.start), "-t", str(args.duration), "-i", str(input_path),
            "-vf", filter_graph, "-loop", "0", str(output_path),
        ]
    else:
        command = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(args.start), "-t", str(args.duration), "-i", str(input_path),
            "-vf", f"fps={args.fps},{scale}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
        ]
        if args.with_audio:
            command.extend(["-map", "0:v:0", "-map", "0:a:0?", "-c:a", "aac", "-shortest"])
        else:
            command.append("-an")
        command.append(str(output_path))

    print(f"Creating {output_path.name} from {input_path.name}...")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"ffmpeg failed with status {error.returncode}") from error
    print(f"Preview ready: {output_path}")


if __name__ == "__main__":
    main()
