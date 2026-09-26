import unittest
from dataclasses import replace
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
    def test_live_cli_preserves_kfp_glitch_override(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            args = live_view.build_parser().parse_args(
                [str(audio), "--no-kfp-glitches"]
            )
            config = live_view._resolve_cli_config(args)
            self.assertFalse(config.kfp_glitches)

    def test_live_source_snapshots_are_cached_and_refresh_on_updates(self):
        store = live_view.LiveZoomSourceStore(np.asarray([0.0, 1.0]))
        store.append(0.0, np.zeros((4, 8), dtype=np.float32), 192)
        first = store.snapshot()
        self.assertIs(first, store.snapshot())

        store.finish(capped=True)
        finished = store.snapshot()
        self.assertIsNot(first, finished)
        self.assertTrue(finished.capped)

        store.append(1.0, np.ones((4, 8), dtype=np.float32), 224)
        updated = store.snapshot()
        self.assertIsNot(finished, updated)
        self.assertEqual(len(updated.fields), 2)

    def test_native_live_frames_render_for_every_formula_and_palette(self):
        """Exercise the complete bounded live frame sequence, not one sample."""

        native_library = visualizer._get_native_library()
        if native_library is None:
            self.skipTest("native renderer is unavailable")

        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            track = live_view.build_live_track(
                np.linspace(0.0, 1.0, 90, dtype=np.float64),
                30,
                "1.0",
                "1e12",
            )

            for formula in visualizer.FORMULA_CHOICES:
                x_center, y_center, _point = visualizer._resolve_render_point(
                    point_spec=None,
                    random_point=False,
                    x_center=None,
                    y_center=None,
                    random_seed=None,
                    max_log_zoom=12.0,
                    formula=formula,
                    julia_constant=visualizer.DEFAULT_JULIA_C,
                )
                config = live_view.LiveViewConfig(
                    audio_path=audio,
                    formula=formula,
                    x_center=x_center,
                    y_center=y_center,
                    julia_constant=visualizer.DEFAULT_JULIA_C,
                    max_zoom="1e12",
                    width=64,
                    height=36,
                    native_threads=2,
                )
                references = live_view._prepare_live_native_references(
                    config,
                    native_library,
                )
                try:
                    sources = live_view.build_live_zoom_sources(
                        config,
                        64,
                        36,
                        native_library,
                        native_references=references,
                        source_native_threads=2,
                    )
                    self.assertEqual(
                        len(sources.fields),
                        len(live_view.live_zoom_ladder("1.0", "1e12")),
                        formula,
                    )
                    frame_zooms = np.asarray(track.zoom, dtype=np.float64).copy()
                    # Make the final frame an exact source endpoint; the
                    # normalised audio envelope can otherwise end at the
                    # next representable float just below 12.0.
                    frame_zooms[-1] = float(sources.log_zooms[-1])

                    for palette in ("aurora", "kalles-default"):
                        palette_config = replace(config, palette=palette)
                        kfp_cache = {} if palette == "kalles-default" else None
                        first = None
                        last = None
                        for index in range(track.energy.size):
                            frame = live_view._live_colour_frame(
                                sources,
                                float(frame_zooms[index]),
                                64,
                                36,
                                float(track.phase[index]),
                                float(track.energy[index]),
                                native_library,
                                palette_config,
                                kfp_cache,
                            )
                            frame = np.asarray(frame)
                            self.assertEqual(
                                frame.shape,
                                (36, 64, 3),
                                f"{formula}/{palette} frame {index}",
                            )
                            self.assertEqual(frame.dtype, np.uint8)
                            self.assertTrue(
                                np.isfinite(frame).all(),
                                f"{formula}/{palette} frame {index} is not finite",
                            )
                            if first is None:
                                first = frame.copy()
                            last = frame
                        assert first is not None and last is not None
                        self.assertGreater(
                            int(np.count_nonzero(first != last)),
                            0,
                            f"{formula}/{palette} did not produce a changing sequence",
                        )
                        if kfp_cache is not None:
                            self.assertEqual(len(kfp_cache), len(sources.fields))
                finally:
                    visualizer._destroy_native_references(
                        native_library,
                        references,
                    )

    def test_live_dimensions_cap_the_source_but_keep_aspect(self):
        self.assertEqual(
            (live_view.LIVE_DEFAULT_WIDTH, live_view.LIVE_DEFAULT_HEIGHT),
            (960, 540),
        )
        self.assertEqual(live_view.live_dimensions(3840, 2160), (960, 540))
        self.assertEqual(
            live_view.live_dimensions(3840, 2160, formula="burning-ship"),
            (960, 540),
        )
        self.assertEqual(
            live_view.live_dimensions(3840, 2160, native_available=False),
            (960, 540),
        )
        self.assertEqual(live_view.live_dimensions(400, 200), (400, 200))

    def test_kfp_live_display_uses_a_smooth_filter(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "song.mp3"
            audio.write_bytes(b"audio")
            kfp = root / "palette.kfp"
            kfp.write_text("Colors: 0,0,0,255,255,255\n", encoding="utf-8")

            ordinary = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="0.0",
                y_center="0.0",
            )
            imported = replace(ordinary, palette_file=kfp)
            builtin = replace(ordinary, palette="kalles-default")

            self.assertEqual(
                live_view._live_display_filter(ordinary),
                live_view.LIVE_CAIRO_FILTER_NEAREST,
            )
            self.assertEqual(
                live_view._live_display_filter(imported),
                live_view.LIVE_CAIRO_FILTER_BEST,
            )
            self.assertEqual(
                live_view._live_display_filter(builtin),
                live_view.LIVE_CAIRO_FILTER_BEST,
            )

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

    def test_prerender_target_covers_the_highest_zoom_in_the_prefix(self):
        track = live_view.LiveAudioTrack(
            np.zeros(10, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
            np.asarray(
                [0.0, 0.8, 2.2, 1.4, 3.7, 2.9, 5.0, 4.0, 6.0, 7.0],
                dtype=np.float64,
            ),
            1.0,
            10,
        )
        count, target = live_view.live_prerender_target(
            track,
            np.arange(8.0, dtype=np.float64),
            0.60,
        )
        # Explicitly exercise a 60% prefix; the 3.7 excursion requires the
        # source at 4.0, even though the prefix ends by pulling back to 2.9.
        self.assertEqual(count, 5)
        self.assertAlmostEqual(target, 3.7)

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
        self.assertEqual(
            len(default_range),
            int(np.ceil(24.0 / live_view.LIVE_SOURCE_LOG_STEP)) + 1,
        )
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
        self.assertGreaterEqual(live_view.live_iteration_cap("mandelbrot", 96.0), 8192)
        self.assertEqual(live_view.live_iteration_cap("mandelbrot", 150.0), 16384)
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
                max_zoom="1e1",
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

    def test_incomplete_live_sources_allow_continuous_lookahead(self):
        """The live camera must not freeze, then jump, as sources arrive."""

        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="-0.50000000000000000000",
                y_center="0.00000000000000000000",
                max_zoom="1e2",
            )
            requested = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
            store = live_view.LiveZoomSourceStore(requested)
            store.append(0.0, np.zeros((4, 8), dtype=np.float32), 192)
            store.append(1.0, np.ones((4, 8), dtype=np.float32), 192)
            observed = []

            def fake_colour(*args):
                observed.append(args)
                return np.zeros((4, 8, 3), dtype=np.uint8)

            with mock.patch.object(
                live_view.visualizer,
                "_atlas_colour_frame",
                side_effect=fake_colour,
            ):
                live_view._live_colour_frame(
                    store,
                    1.5,
                    8,
                    4,
                    0.0,
                    0.0,
                    None,
                    config,
                )

            self.assertAlmostEqual(
                live_view._live_source_zoom_limit(store),
                2.0,
            )
            # The old clamp used the last completed source (1.0), producing a
            # parent zoom of exactly 1.0. The look-ahead must keep the crop
            # moving through the not-yet-rendered interval.
            self.assertEqual(len(observed), 1)
            self.assertAlmostEqual(observed[0][4], 10.0 ** 0.5, places=6)

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

    def test_kfp_live_source_uses_the_kalles_bailout_mode(self):
        """Live KFP fields must use the same high bailout as export fields."""

        palette_file = Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp"
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="-0.743643887037151000000000000000000000000000",
                y_center="0.131825904205330000000000000000000000000000",
                palette_file=palette_file,
                max_zoom="1e12",
            )
            reference = object()
            native_library = type(
                "CapabilityLibrary",
                (),
                {"fractal_backend_capabilities": lambda self: 2 | 4 | 8},
            )()
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
                    native_library,
                    192,
                    [(12.0, reference)],
                )

            self.assertEqual(result.shape, (4, 8))
            self.assertEqual(live_view._live_escape_radius_mode(config), 1)
            self.assertEqual(
                render.call_args.kwargs["render_options"].escape_radius_mode,
                1,
            )
            self.assertEqual(
                render.call_args.kwargs["render_options"].coordinate_mode,
                1,
            )
            # This compact default recipe has no metadata/3D plane needs;
            # its high-bailout 540p field is faster on AVX2 than OpenCL.
            self.assertEqual(render.call_args.kwargs["render_options"].backend, 1)
            self.assertTrue(render.call_args.kwargs["render_options"].strict)
            self.assertFalse(render.call_args.kwargs["render_options"].allow_recovery)

    def test_kfp_live_coordinate_mode_is_independent_of_bailout_mode(self):
        """Classic-radius KFP files still use Kalles' viewport geometry."""

        palette_file = Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp"
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="0.0",
                y_center="0.0",
                palette_file=palette_file,
                max_zoom="1e12",
            )
            with mock.patch.object(
                live_view.visualizer,
                "_kfp_escape_radius_mode",
                return_value=0,
            ):
                self.assertEqual(live_view._live_escape_radius_mode(config), 0)
            self.assertEqual(live_view._live_coordinate_mode(config), 1)

    def test_kfp_deep_alternate_formula_live_source_keeps_orbit_planes(self):
        """Deep alternate-formula tiles must use the native KFP plane path."""

        palette_file = Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp"
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="tricorn",
                x_center="-1.000000000000000000000000000000000000000000",
                y_center="0.100000000000000000000000000000000000000000",
                palette_file=palette_file,
                base_zoom="1e12",
                max_zoom="1e12",
            )
            native_library = type(
                "NativeStub",
                (),
                {"render_fractal_ex_planes": object()},
            )()
            calls = []

            def fake_render(_config, width, height, _log_zoom, _library, max_iter, **kwargs):
                calls.append(kwargs.get("return_planes", False))
                field = np.zeros((height, width), dtype=np.float32)
                return (field, object()) if kwargs.get("return_planes") else field

            with mock.patch("live_view._render_live_source", side_effect=fake_render):
                sources = live_view.build_live_zoom_sources(
                    config,
                    8,
                    4,
                    native_library,
                    max_sources=1,
                )

            self.assertEqual(calls, [True])
            self.assertEqual(len(sources.planes), 1)
            self.assertIsNotNone(sources.planes[0])

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

    def test_deep_non_kfp_live_source_uses_opencl_backend(self):
        """The GUI preview should use the accelerated deep scalar path."""

        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="-0.743643887037151000000000000000000000000000",
                y_center="0.131825904205330000000000000000000000000000",
                # This fixture intentionally starts at the normal deep
                # threshold; it does not need an artificial e80 probe.
                max_zoom="1e12",
            )
            reference = object()
            field = np.zeros((4, 8), dtype=np.float32)
            with mock.patch.object(
                live_view.visualizer,
                "_native_backend_id",
                side_effect=lambda name, _library: 2 if name == "opencl" else 1,
            ), mock.patch.object(
                live_view.visualizer,
                "_select_native_reference",
                return_value=reference,
            ), mock.patch.object(
                live_view.visualizer,
                "render_fractal",
                return_value=field,
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

            np.testing.assert_array_equal(result, field)
            self.assertEqual(render.call_args.kwargs["render_options"].backend, 2)
            self.assertFalse(render.call_args.kwargs["return_planes"])

    def test_shallow_non_kfp_live_source_uses_opencl_backend_when_supported(self):
        """The 540p live source should use the fast direct GPU field."""

        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="tricorn",
                x_center="-0.10000000000000000000",
                y_center="0.70000000000000000000",
                max_zoom="1e1",
            )
            field = np.zeros((4, 8), dtype=np.float32)
            with mock.patch.object(
                live_view.visualizer,
                "_native_backend_id",
                side_effect=lambda name, _library: 2 if name == "opencl" else 1,
            ), mock.patch.object(
                live_view.visualizer,
                "render_fractal",
                return_value=field,
            ) as render:
                result = live_view._render_live_source(
                    config,
                    8,
                    4,
                    1.0,
                    object(),
                    192,
                )

            np.testing.assert_array_equal(result, field)
            self.assertEqual(render.call_args.kwargs["render_options"].backend, 2)

    def test_explicit_scalar_live_backend_is_not_overridden(self):
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="tricorn",
                x_center="-0.10000000000000000000",
                y_center="0.70000000000000000000",
                max_zoom="1e1",
                native_backend="scalar",
            )
            field = np.zeros((4, 8), dtype=np.float32)
            with mock.patch.object(
                live_view.visualizer,
                "_native_backend_id",
                side_effect=lambda name, _library: 2 if name == "opencl" else 1,
            ), mock.patch.object(
                live_view.visualizer,
                "render_fractal",
                return_value=field,
            ) as render:
                result = live_view._render_live_source(
                    config,
                    8,
                    4,
                    1.0,
                    object(),
                    192,
                )

            np.testing.assert_array_equal(result, field)
            self.assertEqual(render.call_args.kwargs["render_options"].backend, 1)

    def test_ultra_deep_mandelbrot_kfp_live_source_uses_gpu_plane_backend(self):
        """Deep Mandelbrot KFP metadata uses the validated GPU plane path."""

        palette_file = Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp"
        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="-0.743643887037151000000000000000000000000000",
                y_center="0.131825904205330000000000000000000000000000",
                palette_file=palette_file,
                max_zoom="1e12",
            )
            reference = object()
            native_library = type(
                "CapabilityLibrary",
                (),
                {"fractal_backend_capabilities": lambda self: 2 | 4 | 8},
            )()
            field = np.zeros((4, 8), dtype=np.float32)
            planes = visualizer.KfpFramePlanes(
                orbit_iteration=np.zeros((4, 8), dtype=np.int64),
                phase=np.zeros((4, 8), dtype=np.float64),
                de_x=np.ones((4, 8), dtype=np.float64),
                de_y=np.ones((4, 8), dtype=np.float64),
                test1=np.ones((4, 8), dtype=np.float64),
                test2=np.ones((4, 8), dtype=np.float64),
            )
            with mock.patch.object(
                live_view.visualizer,
                "_native_backend_id",
                side_effect=lambda name, _library: 2 if name == "opencl" else 1,
            ), mock.patch.object(
                live_view.visualizer,
                "_select_native_reference",
                return_value=reference,
            ), mock.patch.object(
                live_view.visualizer,
                "render_fractal",
                return_value=(field, planes),
            ) as render:
                result, result_planes = live_view._render_live_source(
                    config,
                    8,
                    4,
                    80.0,
                    native_library,
                    192,
                    [(80.0, reference)],
                    return_planes=True,
                )

            np.testing.assert_array_equal(result, field)
            self.assertIsNotNone(result_planes)
            self.assertEqual(render.call_args.kwargs["render_options"].backend, 2)
            self.assertTrue(render.call_args.kwargs["return_planes"])

    def test_incomplete_kfp_planes_are_repaired_before_live_colour(self):
        """Finite iteration fields cannot bypass Kalles plane validation."""

        def make_planes(shape, phase):
            return visualizer.KfpFramePlanes(
                orbit_iteration=np.full(shape, 8, dtype=np.int64),
                phase=np.full(shape, phase, dtype=np.float64),
                de_x=np.full(shape, 1.0, dtype=np.float64),
                de_y=np.full(shape, -1.0, dtype=np.float64),
                test1=np.full(shape, 4.0, dtype=np.float64),
                test2=np.full(shape, 2.0, dtype=np.float64),
            )

        with TemporaryDirectory() as directory:
            audio = Path(directory) / "song.mp3"
            audio.write_bytes(b"audio")
            palette_file = Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp"
            config = live_view.LiveViewConfig(
                audio_path=audio,
                formula="mandelbrot",
                x_center="-0.743643887037151000000000000000000000000000",
                y_center="0.131825904205330000000000000000000000000000",
                palette_file=palette_file,
                max_zoom="1e12",
            )
            reference = object()
            field = np.ones((4, 8), dtype=np.float32)
            broken_planes = make_planes(field.shape, np.nan)
            repaired_planes = make_planes(field.shape, 0.5)
            with mock.patch.object(
                live_view.visualizer,
                "_select_native_reference",
                return_value=reference,
            ), mock.patch.object(
                live_view.visualizer,
                "render_fractal",
                return_value=(field, broken_planes),
            ), mock.patch.object(
                live_view.visualizer,
                "_atlas_glitch_reference_field",
                return_value=(field, repaired_planes),
            ) as repair:
                result, planes = live_view._render_live_source(
                    config,
                    8,
                    4,
                    12.0,
                    object(),
                    192,
                    [(12.0, reference)],
                    return_planes=True,
                )
            repair.assert_called_once()
            np.testing.assert_array_equal(result, field)
            self.assertIsNot(planes, repaired_planes)
            self.assertIs(planes.orbit_iteration, repaired_planes.orbit_iteration)
            self.assertIs(planes.test1, repaired_planes.test1)
            self.assertIsNone(planes.test2)
            self.assertIsNone(planes.phase)
            self.assertIsNone(planes.de_x)
            self.assertIsNone(planes.de_y)

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
