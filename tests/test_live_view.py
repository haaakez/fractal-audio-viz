import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

try:
    import numpy  # noqa: F401
except ImportError as error:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"NumPy is unavailable: {error}") from error

import live_view


class LiveViewHelperTests(unittest.TestCase):
    def test_live_dimensions_cap_the_source_but_keep_aspect(self):
        self.assertEqual(live_view.live_dimensions(3840, 2160), (640, 360))
        self.assertEqual(
            live_view.live_dimensions(3840, 2160, native_available=False),
            (320, 180),
        )
        self.assertEqual(live_view.live_dimensions(400, 200), (400, 200))

    def test_live_dimensions_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            live_view.live_dimensions(0, 200)
        with self.assertRaises(ValueError):
            live_view.live_dimensions(200, -1)

    def test_live_track_is_finite_and_bounded(self):
        track = live_view.build_live_track([0.0, 0.2, 1.0, 0.4, 0.8], 30)
        for values in (track.energy, track.onset, track.phase, track.zoom):
            self.assertTrue(all(float(value) == float(value) for value in values))
        self.assertGreaterEqual(float(track.energy.min()), 0.0)
        self.assertLessEqual(float(track.energy.max()), 1.0)
        self.assertGreaterEqual(float(track.zoom.min()), 1.0)
        self.assertLessEqual(
            float(track.zoom.max()),
            live_view.LIVE_MAX_ZOOM_FACTOR + 1.0e-5,
        )
        self.assertAlmostEqual(track.duration, 5.0 / 30.0)

    def test_silence_does_not_create_nan_controls(self):
        track = live_view.build_live_track([0.0, 0.0, 0.0], 24)
        self.assertTrue((track.energy == 0.0).all())
        self.assertTrue((track.onset == 0.0).all())
        self.assertTrue((track.zoom >= 1.0).all())

    def test_audio_player_command_is_quiet_and_uses_a_list(self):
        with mock.patch("live_view.shutil.which", return_value="/usr/bin/ffplay"):
            command = live_view._audio_player_command(Path("song.mp3"))
        self.assertIsNotNone(command)
        assert command is not None
        self.assertIn("-nodisp", command)
        self.assertEqual(command[-1], "song.mp3")

    def test_config_rejects_missing_audio(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                live_view.LiveViewConfig(
                    audio_path=Path(directory) / "missing.mp3",
                    formula="mandelbrot",
                    x_center="0.0",
                    y_center="0.0",
                )


if __name__ == "__main__":
    unittest.main()
