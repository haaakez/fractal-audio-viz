import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gui


class GuiHelperTests(unittest.TestCase):
    def test_renderer_command_uses_source_cli(self):
        self.assertEqual(
            gui._renderer_command(["song.mp3"]),
            [sys.executable, "-u", str(gui.VISUALIZER), "song.mp3"],
        )

    def test_renderer_command_reuses_frozen_gui_for_cli_child(self):
        with mock.patch.object(gui.sys, "frozen", True, create=True):
            self.assertEqual(
                gui._renderer_command(["song.mp3"]),
                [sys.executable, gui.FROZEN_CLI_FLAG, "song.mp3"],
            )

    def test_source_native_build_detects_stale_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("renderer.cpp", "renderer.h"):
                (root / name).write_text("source", encoding="utf-8")
            (root / "opencl").mkdir()
            (root / "opencl" / "mandelbrot.cl").write_text("kernel", encoding="utf-8")
            target = root / "mandelbrot.so"
            target.write_bytes(b"old")
            old_time = target.stat().st_mtime_ns - 10_000_000
            os.utime(target, ns=(old_time, old_time))
            with mock.patch.object(gui, "RESOURCE_ROOT", root):
                self.assertTrue(gui._source_native_build_required())

    def test_source_native_build_runs_make_for_stale_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("renderer.cpp", "renderer.h"):
                (root / name).write_text("source", encoding="utf-8")
            (root / "opencl").mkdir()
            (root / "opencl" / "mandelbrot.cl").write_text("kernel", encoding="utf-8")
            (root / "mandelbrot.so").write_bytes(b"old")
            result = mock.Mock(returncode=0, stdout="build ok\n")
            with (
                mock.patch.object(gui, "RESOURCE_ROOT", root),
                mock.patch.object(gui, "_source_native_build_required", return_value=True),
                mock.patch.object(gui.subprocess, "run", return_value=result) as run,
            ):
                self.assertEqual(gui._ensure_source_native_build(), "build ok\n")
            run.assert_called_once_with(
                ["make", "-B", "mandelbrot.so"],
                cwd=root,
                stdout=gui.subprocess.PIPE,
                stderr=gui.subprocess.STDOUT,
                text=True,
                check=False,
            )

    def test_progress_helper_accepts_percent_lines(self):
        self.assertEqual(gui._progress_from_output_line("  encoded  25.0%"), (0.25, "25.0%"))
        self.assertEqual(gui._progress_from_output_line("encoded 100%"), (1.0, "100.0%"))

    def test_progress_helper_accepts_frame_counts(self):
        fraction, label = gui._progress_from_output_line("frame 12/100")
        self.assertAlmostEqual(fraction, 0.12)
        self.assertEqual(label, "Frame 12 / 100 (12.0%)")

    def test_progress_helper_ignores_unrelated_diagnostics(self):
        self.assertEqual(gui._progress_from_output_line("Entering atlas level 3/10"), (None, None))
        self.assertEqual(gui._progress_from_output_line("encoder selected: libx264"), (None, None))


if __name__ == "__main__":
    unittest.main()
