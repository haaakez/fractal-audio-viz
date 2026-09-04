import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"NumPy is unavailable: {error}") from error

import live_view
import visualizer


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
        track = live_view.build_live_track(
            [0.0, 0.2, 1.0, 0.4, 0.8],
            30,
            "1e2",
            "1e24",
        )
        for values in (track.energy, track.onset, track.phase, track.zoom):
            self.assertTrue(all(float(value) == float(value) for value in values))
        self.assertGreaterEqual(float(track.energy.min()), 0.0)
        self.assertLessEqual(float(track.energy.max()), 1.0)
        self.assertGreaterEqual(float(track.zoom.min()), 2.0)
        self.assertAlmostEqual(float(track.zoom[-1]), 24.0)
        self.assertAlmostEqual(track.duration, 5.0 / 30.0)

    def test_silence_does_not_create_nan_controls(self):
        track = live_view.build_live_track([0.0, 0.0, 0.0], 24, "1.0", "1e4")
        self.assertTrue((track.energy == 0.0).all())
        self.assertTrue((track.onset == 0.0).all())
        self.assertTrue((track.zoom >= 0.0).all())
        self.assertAlmostEqual(float(track.zoom[-1]), 4.0)

    def test_live_zoom_ladder_reaches_selected_endpoint_without_overflow(self):
        ladder = live_view.live_zoom_ladder("1e0", "1e150")
        self.assertLessEqual(len(ladder), live_view.LIVE_MAX_SOURCE_KEYFRAMES)
        self.assertAlmostEqual(float(ladder[0]), 0.0)
        self.assertAlmostEqual(float(ladder[-1]), 150.0)
        self.assertTrue(all(float(right) > float(left) for left, right in zip(ladder, ladder[1:])))

        capped = live_view.live_zoom_ladder("1e0", "1e1000")
        self.assertAlmostEqual(float(capped[-1]), live_view.LIVE_MAX_PREVIEW_LOG_ZOOM)
        self.assertLessEqual(float(np.max(np.diff(ladder))), 150.0 / 95.0 + 1.0e-9)

        default_range = live_view.live_zoom_ladder("1e0", "1e24")
        self.assertEqual(len(default_range), 33)
        self.assertLessEqual(
            float(np.max(np.diff(default_range))),
            live_view.LIVE_SOURCE_LOG_STEP + 1.0e-9,
        )

    def test_live_zoom_ladder_rejects_reverse_range(self):
        with self.assertRaises(ValueError):
            live_view.live_zoom_ladder("1e4", "1e3")
        with self.assertRaises(ValueError):
            live_view.live_zoom_ladder("1e301", "1e302")

    def test_deep_formula_live_budgets_are_not_fixed_at_192(self):
        self.assertEqual(live_view.live_iteration_cap("mandelbrot", 0.0), 192)
        self.assertGreaterEqual(live_view.live_iteration_cap("burning-ship", 150.0), 700)
        self.assertGreaterEqual(live_view.live_iteration_cap("tricorn", 150.0), 900)
        self.assertGreaterEqual(live_view.live_iteration_cap("julia", 300.0), 1500)
        self.assertLessEqual(
            live_view.live_iteration_cap("julia", 300.0),
            live_view.LIVE_MAX_ITERATIONS,
        )

    def test_live_sources_preserve_the_iteration_cap_for_each_tile(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="tricorn",
                x_center="-1.00000000000000000000",
                y_center="0.10000000000000000000",
                max_zoom="1e4",
            )
            calls = []

            def fake_render(_config, width, height, _log_zoom, _library, max_iter):
                calls.append(max_iter)
                return np.full((height, width), max_iter - 1.0, dtype=np.float32)

            with mock.patch("live_view._render_live_source", side_effect=fake_render):
                sources = live_view.build_live_zoom_sources(config, 8, 4, None)
            self.assertEqual(tuple(calls), sources.iteration_caps)
            self.assertEqual(len(sources.fields), len(sources.iteration_caps))
            self.assertTrue(all(cap >= live_view.LIVE_MIN_ITERATIONS for cap in calls))

    def test_live_sources_can_be_populated_incrementally(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="tricorn",
                x_center="-1.00000000000000000000",
                y_center="0.10000000000000000000",
                max_zoom="1e4",
            )

            def fake_render(_config, width, height, _log_zoom, _library, max_iter):
                return np.full((height, width), max_iter - 1.0, dtype=np.float32)

            ladder = live_view.live_zoom_ladder(config.base_zoom, config.max_zoom)
            store = live_view.LiveZoomSourceStore(ladder)
            with mock.patch("live_view._render_live_source", side_effect=fake_render):
                first = live_view.build_live_zoom_sources(
                    config,
                    8,
                    4,
                    None,
                    store=store,
                    max_sources=2,
                )
                self.assertEqual(len(first.fields), 2)
                self.assertFalse(store.finished)
                complete = live_view.build_live_zoom_sources(
                    config,
                    8,
                    4,
                    None,
                    store=store,
                )
            self.assertEqual(len(complete.fields), len(ladder))
            self.assertTrue(store.finished)

    def test_shared_live_reference_is_used_without_rebuilding_each_source(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="-0.743643887037151000000000000000000000000000",
                y_center="0.131825904205330000000000000000000000000000",
                max_zoom="1e12",
            )
            reference = object()
            with mock.patch.object(
                live_view.visualizer,
                "_select_native_reference",
                return_value=reference,
            ), mock.patch.object(
                live_view.visualizer,
                "render_fractal",
                return_value=np.ones((4, 8), dtype=np.float32),
            ) as render:
                result = live_view._render_live_source(
                    config,
                    8,
                    4,
                    12.0,
                    object(),
                    192,
                    [(12.0, reference)],
                )
            self.assertEqual(result.shape, (4, 8))
            self.assertIs(render.call_args.kwargs["native_reference"], reference)

    def test_nonfinite_shared_live_field_is_repaired_before_display(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="-0.743643887037151000000000000000000000000000",
                y_center="0.131825904205330000000000000000000000000000",
                max_zoom="1e12",
            )
            reference = object()
            repaired = np.ones((4, 8), dtype=np.float32)
            with mock.patch.object(
                live_view.visualizer,
                "_select_native_reference",
                return_value=reference,
            ), mock.patch.object(
                live_view.visualizer,
                "render_fractal",
                return_value=np.full((4, 8), np.nan, dtype=np.float32),
            ), mock.patch.object(
                live_view.visualizer,
                "_atlas_glitch_reference_field",
                return_value=repaired,
            ):
                result = live_view._render_live_source(
                    config,
                    8,
                    4,
                    12.0,
                    object(),
                    192,
                    [(12.0, reference)],
                )
            np.testing.assert_array_equal(result, repaired)

    def test_live_sources_drop_an_unresolved_deep_tile(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="julia",
                x_center="1.0",
                y_center="0.0",
                max_zoom="1e4",
            )
            calls = []

            def fake_render(_config, width, height, log_zoom, _library, max_iter):
                calls.append((log_zoom, max_iter))
                if log_zoom > 0.0:
                    return np.full((height, width), max_iter, dtype=np.float32)
                return np.zeros((height, width), dtype=np.float32)

            with mock.patch("live_view._render_live_source", side_effect=fake_render):
                sources = live_view.build_live_zoom_sources(config, 8, 4, None)
            self.assertEqual(len(sources.fields), 1)
            self.assertTrue(sources.capped)
            self.assertEqual(float(sources.log_zooms[-1]), 0.0)
            self.assertGreater(len(calls), 1)  # the deep tile was retried

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

    def test_config_rejects_reverse_zoom_range(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            with self.assertRaises(ValueError):
                live_view.LiveViewConfig(
                    audio_path=audio,
                    formula="mandelbrot",
                    x_center="0.0",
                    y_center="0.0",
                    base_zoom="1e4",
                    max_zoom="1e3",
                )

    def test_default_cli_config_resolves_a_native_safe_centre(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            args = live_view.build_parser().parse_args([str(audio)])
            config = live_view._resolve_cli_config(args)
            self.assertTrue(config.x_center)
            self.assertTrue(config.y_center)
            self.assertIsNone(visualizer._center_precision_error(
                config.x_center,
                config.y_center,
                config.max_log_zoom,
            ))

    def test_deep_native_coordinate_error_has_a_live_python_fallback(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="0.0",
                y_center="0.0",
                max_zoom="1e24",
            )
            fallback = np.zeros((9, 16), dtype=np.float32)
            with mock.patch(
                "live_view.visualizer._create_native_reference",
                side_effect=RuntimeError("invalid real coordinate"),
            ), mock.patch(
                "live_view.visualizer.render_fractal",
                return_value=fallback,
            ) as render:
                result = live_view._render_live_source(config, 16, 9, 12.0, object())
            np.testing.assert_array_equal(result, fallback)
            self.assertEqual(render.call_args.kwargs["renderer"], "python")

    def test_live_config_rejects_a_base_beyond_preview_precision(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            with self.assertRaisesRegex(ValueError, "base zoom"):
                live_view.LiveViewConfig(
                    audio_path=audio,
                    formula="mandelbrot",
                    x_center="0.0",
                    y_center="0.0",
                    base_zoom="1e301",
                    max_zoom="1e302",
                )


if __name__ == "__main__":
    unittest.main()
