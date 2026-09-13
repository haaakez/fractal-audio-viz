#!/usr/bin/env python3
"""Build a small, deterministic runtime archive for the visualizer.

The application is intentionally distributed as Python source plus one native
renderer library.  Python, FFmpeg, GTK, and the numerical libraries remain
normal platform dependencies; bundling those would make a Linux tarball
machine-specific and would hide the useful error messages from the GUI.

Linux output is a ``.tar.gz``.  Windows output is a ``.zip`` and can include
the MinGW runtime DLLs copied by the Windows CI job with ``--runtime-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import gzip
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]

# Keep the bundle focused on files needed to run or rebuild the application.
# In particular, do not walk the repository: local music, videos, caches,
# interrupted renders, and ignored native products must never enter a release.
RUNTIME_FILES = (
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "IMPROVEMENT_PLAN.md",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "shell.nix",
    "visualizer.py",
    "live_view.py",
    "gui.py",
    "profiles.py",
    "deep_zoom_points.py",
    "make_preview.py",
    "point_sheet.py",
    "benchmark.py",
    "renderer.cpp",
    "renderer.h",
    "packaging/release.py",
    "packaging/build_windows_exe.py",
    "opencl/mandelbrot.cl",
    "palettes/kalles-default.kfp",
    "examples/make_test_tone.py",
    "examples/palette-neon.txt",
    "images/gui.png",
    "images/preview.GIF",
)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _project_version() -> str:
    match = re.search(
        r"^version\s*=\s*[\"']([^\"']+)[\"']\s*$",
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else "dev"


def _safe_component(value: str) -> str:
    value = value.strip()
    if not value:
        return "dev"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
        raise SystemExit(f"invalid release version: {value!r}")
    return value


def _git_revision() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = False
    return f"{revision}{'-dirty' if dirty else ''}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"required release file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_launchers(root: Path, platform: str) -> None:
    if platform == "linux":
        launchers = {
            "run_gui.sh": (
                "#!/bin/sh\n"
                "set -eu\n"
                "BASE_DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
                "exec python3 \"$BASE_DIR/gui.py\" \"$@\"\n"
            ),
            "render.sh": (
                "#!/bin/sh\n"
                "set -eu\n"
                "BASE_DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
                "exec python3 \"$BASE_DIR/visualizer.py\" \"$@\"\n"
            ),
            "live-view.sh": (
                "#!/bin/sh\n"
                "set -eu\n"
                "BASE_DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
                "exec python3 \"$BASE_DIR/live_view.py\" \"$@\"\n"
            ),
        }
        for name, content in launchers.items():
            path = root / name
            path.write_text(content, encoding="utf-8")
            path.chmod(0o755)
        return

    launchers = {
        "run_gui.bat": (
            "@echo off\r\n"
            "setlocal\r\n"
            "where py >nul 2>nul\r\n"
            "if not errorlevel 1 (py -3 \"%~dp0gui.py\" %*) else (python \"%~dp0gui.py\" %*)\r\n"
        ),
        "render.bat": (
            "@echo off\r\n"
            "setlocal\r\n"
            "where py >nul 2>nul\r\n"
            "if not errorlevel 1 (py -3 \"%~dp0visualizer.py\" %*) else (python \"%~dp0visualizer.py\" %*)\r\n"
        ),
        "live-view.bat": (
            "@echo off\r\n"
            "setlocal\r\n"
            "where py >nul 2>nul\r\n"
            "if not errorlevel 1 (py -3 \"%~dp0live_view.py\" %*) else (python \"%~dp0live_view.py\" %*)\r\n"
        ),
    }
    for name, content in launchers.items():
        (root / name).write_text(content, encoding="utf-8", newline="")


def _write_bundle_readme(root: Path, platform: str, version: str, native_name: str) -> None:
    if platform == "linux":
        commands = """\
1. install python 3.10+, ffmpeg, and the packages in requirements.txt.
2. for the gui/live view, install gtk 3 and pygobject as system packages.
3. run ./run_gui.sh, ./render.sh, or ./live-view.sh.
"""
    else:
        commands = """\
1. install python 3.10+, ffmpeg, and the packages in requirements.txt.
2. for the gui/live view, install gtk 3 and pygobject for windows.
3. run run_gui.bat, render.bat, or live-view.bat.
"""
    text = f"""fractal audio viz {version} ({platform})

this archive contains the python application and the compiled native renderer
({native_name}). it intentionally does not contain your music, rendered video,
cache, or a platform-wide python/ffmpeg/gtk installation.

setup:
{commands}

the native loader looks for {native_name} beside visualizer.py automatically.
set MANDELBROT_LIBRARY if you want to use a different build.

the full project documentation is in README.md.
"""
    (root / "BUNDLE-README.txt").write_text(text, encoding="utf-8")


def _write_manifest(root: Path, platform: str, version: str, native_name: str) -> None:
    files = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "RELEASE-MANIFEST.json":
            continue
        files.append({"path": relative, "sha256": _sha256(path)})
    manifest = {
        "name": "fractal-audio-viz",
        "version": version,
        "platform": platform,
        "architecture": "x86_64",
        "native_library": native_name,
        "git_revision": _git_revision(),
        "files": files,
    }
    (root / "RELEASE-MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _iter_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _write_tar(source_root: Path, archive: Path, archive_stem: str) -> None:
    with archive.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            compresslevel=9,
            mtime=0,
        ) as compressed_output:
            with tarfile.open(fileobj=compressed_output, mode="w") as output:
                for path in sorted(source_root.rglob("*")):
                    relative = path.relative_to(source_root)
                    archive_name = Path(archive_stem) / relative
                    info = output.gettarinfo(str(path), arcname=archive_name.as_posix())
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            output.addfile(info, handle)
                    else:
                        output.addfile(info)


def _write_zip(source_root: Path, archive: Path, archive_stem: str) -> None:
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as output:
        for path in _iter_files(source_root):
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(f"{archive_stem}/{relative}")
            # ZIP cannot represent the Unix epoch. A fixed DOS timestamp keeps
            # repeated builds byte-stable without preserving local mtimes.
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            output.writestr(info, path.read_bytes())


def build_archive(
    platform: str,
    native: Path,
    output_dir: Path,
    version: str,
    runtime_dir: Path | None,
) -> Path:
    if platform not in {"linux", "windows"}:
        raise SystemExit(f"unsupported release platform: {platform}")
    if not native.is_file():
        raise SystemExit(f"native renderer does not exist: {native}")

    native_name = "mandelbrot.so" if platform == "linux" else "mandelbrot.dll"
    archive_stem = f"fractal-audio-viz-{version}-{platform}-x86_64"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / (
        f"{archive_stem}.tar.gz" if platform == "linux" else f"{archive_stem}.zip"
    )

    with tempfile.TemporaryDirectory(prefix="fractal-audio-viz-") as temporary:
        stage = Path(temporary) / archive_stem
        for relative in RUNTIME_FILES:
            _copy_file(ROOT / relative, stage / relative)

        if runtime_dir is not None:
            if not runtime_dir.is_dir():
                raise SystemExit(f"runtime DLL directory does not exist: {runtime_dir}")
            for dependency in sorted(runtime_dir.iterdir()):
                if dependency.is_file() and dependency.name != native_name:
                    _copy_file(dependency, stage / dependency.name)
        _copy_file(native, stage / native_name)
        _write_launchers(stage, platform)
        _write_bundle_readme(stage, platform, version, native_name)
        _write_manifest(stage, platform, version, native_name)

        if archive.exists():
            archive.unlink()
        if platform == "linux":
            _write_tar(stage, archive, archive_stem)
        else:
            _write_zip(stage, archive, archive_stem)

    print(f"created {archive} ({archive.stat().st_size / (1024 * 1024):.1f} MiB)")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=("linux", "windows"))
    parser.add_argument("--native", required=True, help="compiled .so or .dll")
    parser.add_argument("--output", default="dist", help="archive output directory")
    parser.add_argument("--version", default=None, help="archive version (default: pyproject.toml)")
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help="optional directory of runtime DLLs to place beside the Windows renderer",
    )
    args = parser.parse_args()
    version = _safe_component(args.version or _project_version())
    native = _resolve_path(args.native)
    output_dir = _resolve_path(args.output)
    runtime_dir = _resolve_path(args.runtime_dir) if args.runtime_dir else None
    build_archive(args.platform, native, output_dir, version, runtime_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
