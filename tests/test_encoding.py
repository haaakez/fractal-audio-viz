"""Small end-to-end checks for the near-lossless media pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

try:
    import visualizer
except (ImportError, RuntimeError) as error:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"visualizer dependencies are unavailable: {error}") from error


class EncodingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = shutil.which("ffmpeg")
        cls.ffprobe = shutil.which("ffprobe")
        if cls.ffmpeg is None or cls.ffprobe is None:
            raise unittest.SkipTest("ffmpeg and ffprobe are required")
        encoders = subprocess.run(
            [cls.ffmpeg, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if encoders.returncode != 0 or "libx264" not in encoders.stdout:
            raise unittest.SkipTest("the local FFmpeg has no libx264 encoder")

    @staticmethod
    def _write_fixture_audio(path: Path) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8_000)
            handle.writeframes(b"\0\0" * 2_000)

    def test_cli_near_lossless_output_has_matching_media_and_manifest(self):
        with tempfile.TemporaryDirectory(prefix="fractal-encoding-test-") as root_text:
            root = Path(root_text)
            audio = root / "fixture.wav"
            output = root / "fixture.mp4"
            manifest_path = output.with_suffix(".json")
            self._write_fixture_audio(audio)
            argv = [
                "visualizer.py",
                str(audio),
                "--output",
                str(output),
                "--profile",
                "sd60",
                "--width",
                "64",
                "--height",
                "48",
                "--fps",
                "10",
                "--sample-rate",
                "8000",
                "--separation",
                "none",
                "--max-zoom",
                "1.0",
                "--base-zoom",
                "1.0",
                "--iteration-base",
                "32",
                "--iterations-per-decade",
                "0",
                "--iteration-cap",
                "64",
                "--series-block",
                "32",
                "--native-threads",
                "1",
                "--renderer",
                "python",
                "--codec",
                "libx264",
                "--video-preset",
                "ultrafast",
                "--crf",
                "10",
                "--resample",
                "bilinear",
            ]
            with mock.patch.object(sys, "argv", argv):
                visualizer._main_impl()

            self.assertTrue(output.is_file())
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            expected_frames = int(manifest["frames"]["count"])
            expected_duration = float(manifest["frames"]["duration_seconds"])
            self.assertGreater(expected_frames, 0)

            probe = subprocess.run(
                [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_entries",
                    "stream=codec_type,width,height,pix_fmt,nb_read_frames,sample_rate",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            metadata = json.loads(probe.stdout)
            video = next(
                stream for stream in metadata["streams"]
                if stream["codec_type"] == "video"
            )
            audio_stream = next(
                stream for stream in metadata["streams"]
                if stream["codec_type"] == "audio"
            )
            self.assertEqual((video["width"], video["height"]), (64, 48))
            self.assertEqual(video["pix_fmt"], "yuv420p")
            self.assertEqual(int(video["nb_read_frames"]), expected_frames)
            self.assertEqual(int(audio_stream["sample_rate"]), 8_000)
            self.assertAlmostEqual(
                float(metadata["format"]["duration"]),
                expected_duration,
                delta=0.15,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
