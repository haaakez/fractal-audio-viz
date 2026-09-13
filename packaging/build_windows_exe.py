#!/usr/bin/env python3
"""Build the standalone Windows executable release.

This script is intentionally run on Windows. PyInstaller embeds the Python
runtime and application modules into one executable; the native renderer,
MinGW/GMP/MPFR DLL dependencies, and FFmpeg tools are added as embedded
binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAMES = (
    "audioread",
    "librosa",
    "llvmlite",
    "mpmath",
    "numba",
    "numpy",
    "PIL",
    "scipy",
    "soundfile",
)
METADATA_NAMES = (
    "librosa",
    "llvmlite",
    "numba",
    "numpy",
    "scipy",
    "soundfile",
)
HIDDEN_IMPORTS = (
    "audioread.ffdec",
    "audioread.ffmpeg",
    "deep_zoom_points",
    "profiles",
)
FFMPEG_TOOLS = ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe")


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    marker = 'version = "'
    start = text.index(marker) + len(marker)
    return text[start : text.index('"', start)]


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"required release file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_zip(source_root: Path, archive: Path, archive_stem: str) -> None:
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as output:
        for path in sorted(path for path in source_root.rglob("*") if path.is_file()):
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(f"{archive_stem}/{relative}")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            output.writestr(info, path.read_bytes())


def _write_manifest(stage: Path, version: str) -> None:
    files = []
    for path in sorted(path for path in stage.rglob("*") if path.is_file()):
        relative = path.relative_to(stage).as_posix()
        if relative == "RELEASE-MANIFEST.json":
            continue
        files.append({"path": relative, "sha256": _sha256(path)})
    manifest = {
        "name": "fractal-audio-viz",
        "version": version,
        "platform": "windows",
        "architecture": "x86_64",
        "executable": "fractal-viz.exe",
        "git_revision": _git_revision(),
        "files": files,
    }
    (stage / "RELEASE-MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_readme(stage: Path, version: str, bundled_ffmpeg: list[str]) -> None:
    tools = ", ".join(bundled_ffmpeg) if bundled_ffmpeg else "none"
    text = f"""fractal audio viz {version} — standalone windows renderer

this is the easy, self-contained version. it already includes python, the
native fractal renderer, its numerical/runtime dlls, and the media tools. you
do not need to install python or any pip packages.

render a file from powershell or command prompt:

  .\\fractal-viz.exe "C:\\Music\\song.mp3" --output fractal_viz.mp4 --profile fhd60

the bundled media tools are: {tools}.
they are unpacked into the executable's private runtime directory when needed.

the command-line renderer supports the normal formulas, palettes, deep zoom
profiles, and kalles .kfp palettes. the executable creates the output video
next to the path you give it.
"""
    (stage / "README.txt").write_text(text, encoding="utf-8")
    (stage / "render.bat").write_text(
        '@echo off\r\n"%~dp0fractal-viz.exe" %*\r\n',
        encoding="utf-8",
        newline="",
    )


def _pyinstaller_command(
    *,
    native: Path,
    runtime_dlls: list[Path],
    media_binaries: list[Path],
    build_root: Path,
) -> list[str]:
    separator = ";"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name",
        "fractal-viz",
        "--distpath",
        str(build_root / "dist"),
        "--workpath",
        str(build_root / "work"),
        "--specpath",
        str(build_root),
        "--paths",
        str(ROOT),
        "--add-binary",
        f"{native}{separator}.",
        "--add-data",
        f"{ROOT / 'palettes'}{separator}palettes",
        str(ROOT / "visualizer.py"),
    ]
    for runtime_dll in runtime_dlls:
        command.extend(["--add-binary", f"{runtime_dll}{separator}."])
    for media_binary in media_binaries:
        command.extend(["--add-binary", f"{media_binary}{separator}."])
    for package in PACKAGE_NAMES:
        command.extend(["--collect-all", package])
    for package in METADATA_NAMES:
        command.extend(["--copy-metadata", package])
    for module in HIDDEN_IMPORTS:
        command.extend(["--hidden-import", module])
    return command


def build(args: argparse.Namespace) -> Path:
    if sys.platform != "win32":
        raise SystemExit("packaging/build_windows_exe.py must run on Windows")

    native = Path(args.native).expanduser().resolve()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    ffmpeg_dir = Path(args.ffmpeg_dir).expanduser().resolve()
    if native.suffix.casefold() != ".dll":
        raise SystemExit("--native must point to a Windows .dll")
    if not native.is_file():
        raise SystemExit(f"native DLL does not exist: {native}")
    if not runtime_dir.is_dir():
        raise SystemExit(f"runtime DLL directory does not exist: {runtime_dir}")
    if not ffmpeg_dir.is_dir():
        raise SystemExit(f"FFmpeg directory does not exist: {ffmpeg_dir}")
    ffmpeg = ffmpeg_dir / "ffmpeg.exe"
    if not ffmpeg.is_file():
        raise SystemExit(f"bundled FFmpeg is missing: {ffmpeg}")

    runtime_dlls = sorted(path for path in runtime_dir.glob("*.dll") if path.is_file())
    if not runtime_dlls:
        raise SystemExit(f"no runtime DLLs found in {runtime_dir}")

    # FFmpeg is part of the standalone bundle too. PyInstaller places these
    # binaries in its private one-file extraction directory, where the app's
    # bundled-tool lookup can find them without requiring PATH changes.
    media_binaries: list[Path] = []
    bundled_ffmpeg: list[str] = []
    for tool in FFMPEG_TOOLS:
        source = ffmpeg_dir / tool
        if source.is_file():
            media_binaries.append(source)
            bundled_ffmpeg.append(tool)
    if "ffmpeg.exe" not in bundled_ffmpeg:
        raise SystemExit(f"bundled FFmpeg is missing: {ffmpeg_dir / 'ffmpeg.exe'}")

    # Chocolatey's FFmpeg builds usually keep their dependent DLLs beside the
    # executables. Include them, while avoiding duplicate basenames already
    # supplied by the native renderer runtime.
    occupied_names = {
        native.name.casefold(),
        *(path.name.casefold() for path in runtime_dlls),
        *(path.name.casefold() for path in media_binaries),
    }
    for dependency in sorted(ffmpeg_dir.glob("*.dll")):
        if dependency.is_file() and dependency.name.casefold() not in occupied_names:
            media_binaries.append(dependency)
            occupied_names.add(dependency.name.casefold())

    version = _project_version()
    build_root = ROOT / ".release-build" / "windows-exe"
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)
    command = _pyinstaller_command(
        native=native,
        runtime_dlls=runtime_dlls,
        media_binaries=media_binaries,
        build_root=build_root,
    )
    subprocess.run(command, cwd=ROOT, check=True)

    built_executable = build_root / "dist" / "fractal-viz.exe"
    if not built_executable.is_file():
        raise SystemExit(f"PyInstaller did not create {built_executable}")

    stage = build_root / "stage"
    stage.mkdir()
    _copy(built_executable, stage / "fractal-viz.exe")
    _write_readme(stage, version, bundled_ffmpeg)
    _write_manifest(stage, version)

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_stem = f"fractal-audio-viz-{version}-windows-x86_64-exe"
    archive = output / f"{archive_stem}.zip"
    _write_zip(stage, archive, archive_stem)
    print(f"created {archive} ({archive.stat().st_size / 1024 / 1024:.1f} MiB)")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, help="portable mandelbrot.dll")
    parser.add_argument("--runtime-dir", required=True, help="MinGW/GMP/MPFR DLL directory")
    parser.add_argument("--ffmpeg-dir", required=True, help="directory containing ffmpeg.exe")
    parser.add_argument("--output", default="dist")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
