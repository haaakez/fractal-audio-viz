#!/usr/bin/env python3
"""Turn a render into a small GIF or silent social-media preview.

With no input, the newest filename containing ``4k`` and ``e150`` is chosen.
If no such filename exists, the newest video in the selected directory is
used and the choice is reported.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
import signal
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}
PREVIEW_TIMEOUT_SECONDS = 30.0 * 60.0
MAX_PREVIEW_DURATION_SECONDS = 60.0 * 60.0
MAX_PREVIEW_FRAMES = 2_000_000
MAX_PREVIEW_DIMENSION = 16_384
MAX_PREVIEW_PIXELS = 100_000_000
MAX_VIDEO_SCAN_ENTRIES = 100_000
MAX_DIAGNOSTIC_LINE_CHARS = 16_384


def _absolute_path(path: Path) -> Path:
    """Make a path absolute without resolving its final symlink."""

    return Path(os.path.abspath(str(Path(path).expanduser())))


def _reject_final_symlink(path: Path, label: str) -> None:
    try:
        if Path(path).expanduser().is_symlink():
            raise SystemExit(f"{label} path must not be a symbolic link: {path}")
    except OSError as error:
        raise SystemExit(f"could not inspect {label} path {path}: {error}") from error


def _process_group_options() -> dict[str, object]:
    if os.name == "nt":
        return {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        }
    return {"start_new_session": True}


def _terminate_process(process: subprocess.Popen[str], timeout: float = 2.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
        except (OSError, ValueError):
            try:
                process.terminate()
            except OSError:
                return
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                process.terminate()
            except OSError:
                return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            try:
                process.kill()
            except OSError:
                return
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    process.kill()
                except OSError:
                    return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return


def _read_bounded_line(stream: object, limit: int) -> str | None:
    """Read one diagnostic line without buffering an unterminated flood."""

    readline = getattr(stream, "readline")
    chunk = readline(limit + 1)
    if not chunk:
        return None
    truncated = len(chunk) > limit
    if truncated and not chunk.endswith("\n"):
        while True:
            remainder = readline(limit + 1)
            if not remainder or remainder.endswith("\n"):
                break
    if truncated:
        chunk = chunk[:limit] + "… [line truncated]\n"
    return str(chunk)


def _diagnostic_snapshot(diagnostics: deque[str] | None) -> tuple[str, ...]:
    """Take a stable snapshot of the background stderr ring buffer."""

    if diagnostics is None:
        return ()
    try:
        return tuple(diagnostics.copy())
    except (AttributeError, RuntimeError):
        return ()


def _start_process_with_diagnostics(
    command: list[str],
) -> tuple[subprocess.Popen[str], deque[str], threading.Thread]:
    """Start FFmpeg without allowing an unbounded stderr pipe to deadlock."""

    diagnostics: deque[str] = deque(maxlen=80)
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        **_process_group_options(),
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
            return
        finally:
            try:
                stream.close()
            except OSError:
                pass

    reader = threading.Thread(
        target=drain_stderr,
        name="fractal-preview-diagnostics",
        daemon=True,
    )
    try:
        reader.start()
    except BaseException:
        _terminate_process(process)
        raise
    return process, diagnostics, reader


def _latest_video(directory: Path) -> Path:
    if not directory.is_dir():
        raise SystemExit(f"video search directory not found: {directory}")
    newest: tuple[Path, int] | None = None
    preferred: tuple[Path, int] | None = None
    scanned = 0
    truncated = False
    try:
        entries = directory.iterdir()
        for path in entries:
            if scanned >= MAX_VIDEO_SCAN_ENTRIES:
                truncated = True
                break
            scanned += 1
            try:
                if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES:
                    timestamp = path.stat().st_mtime_ns
                    candidate = (path, timestamp)
                    if newest is None or timestamp > newest[1]:
                        newest = candidate
                    stem = path.stem.casefold()
                    if "4k" in stem and any(
                        marker in stem for marker in ("e150", "e-150", "150")
                    ) and (preferred is None or timestamp > preferred[1]):
                        preferred = candidate
            except OSError:
                # Files can disappear or become inaccessible while a search
                # is in progress. Ignore that entry and keep the launcher
                # useful for the remaining videos.
                continue
    except OSError as error:
        raise SystemExit(f"could not search video directory {directory}: {error}") from error
    if preferred is not None:
        return preferred[0]
    if newest is None:
        suffix = f" (first {MAX_VIDEO_SCAN_ENTRIES:,} entries)" if truncated else ""
        raise SystemExit(f"no video files found in {directory}{suffix}")
    if truncated:
        print(
            f"Video search reached the {MAX_VIDEO_SCAN_ENTRIES:,}-entry limit; "
            "using the newest matching file seen."
        )
    print("No filename matching 4k/e150 was found; using the newest video.")
    return newest[0]


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
        help="maximum preview width (2..16384); aspect ratio is preserved",
    )
    parser.add_argument("--fps", type=int, default=12, help="GIF/preview frame rate (1..1000)")
    parser.add_argument(
        "--with-audio",
        action="store_true",
        help="keep audio for MP4 output; GIF output is always silent",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        not math.isfinite(args.start)
        or not math.isfinite(args.duration)
        or args.start < 0.0
        or args.duration <= 0.0
        or args.duration > MAX_PREVIEW_DURATION_SECONDS
    ):
        raise SystemExit(
            "start must be non-negative and duration must be positive and no more than "
            f"{MAX_PREVIEW_DURATION_SECONDS / 60.0:.0f} minutes"
        )
    if args.width < 2 or args.width > MAX_PREVIEW_DIMENSION or args.fps < 1 or args.fps > 1000:
        raise SystemExit(
            f"width must be between 2 and {MAX_PREVIEW_DIMENSION} and fps between 1 and 1000"
        )
    if args.duration * args.fps > MAX_PREVIEW_FRAMES:
        raise SystemExit(
            f"preview duration and fps may produce at most {MAX_PREVIEW_FRAMES:,} frames"
        )
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")

    search_directory = args.directory.expanduser().resolve()
    input_path = args.input or _latest_video(search_directory)
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input video not found: {input_path}")
    output_path = _output_path(input_path, args.output, args.format).expanduser()
    _reject_final_symlink(output_path, "output")
    output_path = _absolute_path(output_path)
    if output_path == input_path:
        raise SystemExit("output preview must be different from the input video")
    try:
        if output_path.exists() and output_path.is_file() and output_path.samefile(input_path):
            raise SystemExit("output preview must be different from the input video")
    except OSError:
        pass
    if output_path.exists() and output_path.is_dir():
        raise SystemExit(f"output path is a directory: {output_path}")
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise SystemExit(f"output parent is not a directory: {output_path.parent}")
    expected_suffix = ".gif" if args.format == "gif" else ".mp4"
    if output_path.suffix.casefold() != expected_suffix:
        raise SystemExit(
            f"--output must use the {expected_suffix} extension for --format {args.format}"
        )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"could not create output directory {output_path.parent}: {error}") from error
    try:
        temporary_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.previewing-",
            suffix=output_path.suffix,
            delete=False,
        )
    except OSError as error:
        raise SystemExit(f"could not reserve temporary preview: {error}") from error
    temporary_output = Path(temporary_handle.name)
    temporary_handle.close()
    max_height = min(MAX_PREVIEW_DIMENSION, MAX_PREVIEW_PIXELS // args.width)
    scale = (
        f"scale=w='min(iw,{args.width})':h='min(ih,{max_height})':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos"
    )

    if args.format == "gif":
        filter_graph = (
            f"fps={args.fps},{scale},split[s0][s1];"
            "[s0]palettegen=stats_mode=diff:max_colors=256[p];"
            "[s1][p]paletteuse=dither=sierra2_4a"
        )
        command = [
            ffmpeg_path, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(args.start), "-t", str(args.duration), "-i", str(input_path),
            "-vf", filter_graph, "-loop", "0", str(temporary_output),
        ]
    else:
        command = [
            ffmpeg_path, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(args.start), "-t", str(args.duration), "-i", str(input_path),
            "-vf", f"fps={args.fps},{scale}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
        ]
        if args.with_audio:
            command.extend(["-map", "0:v:0", "-map", "0:a:0?", "-c:a", "aac", "-shortest"])
        else:
            command.append("-an")
        command.append(str(temporary_output))

    print(f"Creating {output_path.name} from {input_path.name}...")
    process = None
    diagnostics: deque[str] | None = None
    diagnostic_reader: threading.Thread | None = None
    try:
        try:
            process, diagnostics, diagnostic_reader = _start_process_with_diagnostics(command)
        except OSError as error:
            raise SystemExit(f"could not start ffmpeg: {error}") from error
        try:
            process.wait(timeout=PREVIEW_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            _terminate_process(process)
            raise SystemExit(
                f"ffmpeg exceeded the {PREVIEW_TIMEOUT_SECONDS / 60.0:.0f}-minute timeout"
            ) from error
        finally:
            if diagnostic_reader is not None:
                diagnostic_reader.join(timeout=2.0)
        if process.returncode != 0:
            details = "".join(_diagnostic_snapshot(diagnostics)).strip()
            if len(details) > 4000:
                details = "[diagnostics truncated]\n" + details[-4000:]
            message = f"ffmpeg failed with status {process.returncode}"
            if details:
                message += f":\n{details}"
            raise SystemExit(message)
        try:
            os.replace(temporary_output, output_path)
        except OSError as error:
            raise SystemExit(f"could not finalize preview: {error}") from error
    finally:
        if process is not None and process.poll() is None:
            _terminate_process(process)
        if diagnostic_reader is not None:
            diagnostic_reader.join(timeout=2.0)
        if temporary_output.exists():
            try:
                temporary_output.unlink()
            except OSError:
                pass
    print(f"Preview ready: {output_path}")


if __name__ == "__main__":
    main()
