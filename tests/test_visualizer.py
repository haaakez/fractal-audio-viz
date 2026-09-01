import unittest
import tempfile
import math
from decimal import Decimal
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"NumPy is unavailable: {error}") from error

import visualizer
from profiles import PROFILE_DEFAULTS


class AnimationTests(unittest.TestCase):
    def test_fractional_decimal_places_respect_scientific_notation(self):
        self.assertEqual(visualizer._fractional_decimal_places("1.23e-4"), 6)
        self.assertEqual(visualizer._fractional_decimal_places("-1.2300"), 4)
        self.assertEqual(visualizer._fractional_decimal_places("12e3"), 0)

    def test_deep_center_precision_guard_catches_e150_bundled_center(self):
        available, required = visualizer._center_precision_budget(
            visualizer.DEFAULT_X_CENTER,
            visualizer.DEFAULT_Y_CENTER,
            100.0,
        )
        self.assertEqual(available, 129)
        self.assertEqual(required, 116)
        self.assertIsNone(
            visualizer._center_precision_error(
                visualizer.DEFAULT_X_CENTER,
                visualizer.DEFAULT_Y_CENTER,
                100.0,
            )
        )
        available, required = visualizer._center_precision_budget(
            visualizer.DEFAULT_X_CENTER,
            visualizer.DEFAULT_Y_CENTER,
            150.0,
        )
        self.assertEqual(available, 129)
        self.assertEqual(required, 166)
        self.assertIn(
            "full-precision --x-center and --y-center",
            visualizer._center_precision_error(
                visualizer.DEFAULT_X_CENTER,
                visualizer.DEFAULT_Y_CENTER,
                150.0,
            ),
        )

    def test_curated_point_catalogue_has_at_least_twenty_e150_locations(self):
        points = visualizer.DEEP_ZOOM_POINTS
        self.assertGreaterEqual(len(points), 20)
        self.assertEqual(len({point.slug for point in points}), len(points))
        self.assertGreaterEqual(
            sum(point.conjugate_of is None for point in points),
            14,
        )
        self.assertTrue(
            all(visualizer._deep_point_max_log10_zoom(point) >= 150.0 for point in points)
        )

    def test_random_point_is_reproducible_and_filtered_for_e150(self):
        first = visualizer._resolve_render_point(
            point_spec="random",
            random_point=False,
            x_center=None,
            y_center=None,
            random_seed=417,
            max_log_zoom=150.0,
        )
        second = visualizer._resolve_render_point(
            point_spec=None,
            random_point=True,
            x_center=None,
            y_center=None,
            random_seed=417,
            max_log_zoom=150.0,
        )
        self.assertEqual(first, second)
        self.assertIsNotNone(first[2])
        self.assertLessEqual(first[2].screened_log10_zoom, 150.0)
        self.assertIsNone(visualizer._center_precision_error(first[0], first[1], 150.0))

    def test_point_accepts_exact_custom_pair_and_rejects_conflicts(self):
        x, y, preset = visualizer._resolve_render_point(
            point_spec="-0.743643887037151, 0.131825904205330",
            random_point=False,
            x_center=None,
            y_center=None,
            random_seed=None,
            max_log_zoom=12.0,
        )
        self.assertEqual(x, "-0.743643887037151")
        self.assertEqual(y, "0.131825904205330")
        self.assertIsNone(preset)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            visualizer._resolve_render_point(
                point_spec="oldwooddish",
                random_point=False,
                x_center="-0.7",
                y_center="0.1",
                random_seed=None,
                max_log_zoom=12.0,
            )

    def test_preset_rejects_depth_beyond_stored_coordinate(self):
        with self.assertRaisesRegex(ValueError, "below requested"):
            visualizer._resolve_render_point(
                point_spec="oldwooddish",
                random_point=False,
                x_center=None,
                y_center=None,
                random_seed=None,
                max_log_zoom=160.0,
            )

    def test_native_reference_tiers_switch_before_deep_bla_can_fail(self):
        self.assertEqual(visualizer._native_reference_tier_logs(20.0), [12.0])
        self.assertEqual(
            visualizer._native_reference_tier_logs(83.0),
            [12.0, 40.0, 50.0, 60.0, 70.0, 80.0, 83.0],
        )
        references = [
            (12.0, "shallow"),
            (40.0, "e40"),
            (50.0, "e50"),
            (60.0, "e60"),
        ]
        self.assertIsNone(visualizer._select_native_reference(references, 11.9))
        self.assertEqual(visualizer._select_native_reference(references, 20.0), "shallow")
        self.assertEqual(visualizer._select_native_reference(references, 49.9), "e40")
        self.assertEqual(visualizer._select_native_reference(references, 59.9), "e50")
        self.assertEqual(visualizer._select_native_reference(references, 83.0), "e60")

    def test_native_reference_tiers_align_to_atlas_boundaries(self):
        step = math.log10(2.0)
        tiers = visualizer._native_reference_tier_logs(83.0, step)
        expected_first_deep = math.floor(40.0 / step) * step
        self.assertAlmostEqual(tiers[1], expected_first_deep, places=12)
        self.assertLessEqual(tiers[1], 40.0)
        self.assertEqual(tiers[-1], 83.0)

    def test_audio_without_stems_uses_full_mix_for_flow_and_zoom(self):
        source = np.ones(1000, dtype=np.float32)
        rms = np.linspace(0.05, 1.0, 100, dtype=np.float32)
        pitch = np.linspace(0.2, 0.8, 100, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_path = Path(audio_file.name)
            with mock.patch.object(visualizer, "_load_audio", return_value=source), \
                mock.patch.object(visualizer, "_frame_rms", return_value=rms), \
                mock.patch.object(visualizer, "_frame_pitch", return_value=pitch):
                features = visualizer.analyse_audio(
                    audio_path,
                    sample_rate=100,
                    fps=10,
                    separation="none",
                )
        np.testing.assert_allclose(features.vocal, features.instrumental)
        expected_gradient = visualizer._smooth(
            visualizer._normalise_minmax(rms), 4
        )
        np.testing.assert_allclose(features.gradient, expected_gradient)
        expected_phase = np.cumsum(
            visualizer.AURORA_FLOW_SPEED
            * (
                visualizer.AURORA_MIN_FLOW_FRACTION
                + (1.0 - visualizer.AURORA_MIN_FLOW_FRACTION)
                * np.clip(expected_gradient.astype(np.float64), 0.0, 1.0) ** 2.5
            )
            / 10.0
        ).astype(np.float32)
        np.testing.assert_allclose(features.phase, expected_phase)
        self.assertGreater(float(np.ptp(features.phase)), 5.0)
        self.assertGreaterEqual(float(np.min(np.diff(features.phase))), 0.0)

    def test_audio_flow_keeps_moving_through_a_silent_tail(self):
        source = np.ones(1000, dtype=np.float32)
        rms = np.concatenate((
            np.ones(50, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
        ))
        pitch = np.full(100, 0.5, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_path = Path(audio_file.name)
            with mock.patch.object(visualizer, "_load_audio", return_value=source), \
                mock.patch.object(visualizer, "_frame_rms", return_value=rms), \
                mock.patch.object(visualizer, "_frame_pitch", return_value=pitch):
                features = visualizer.analyse_audio(
                    audio_path,
                    sample_rate=100,
                    fps=10,
                    separation="none",
                )
        self.assertGreater(float(features.phase[-1] - features.phase[-2]), 0.0)
        np.testing.assert_allclose(
            features.phase[-1] - features.phase[-2],
            visualizer.AURORA_FLOW_SPEED
            * visualizer.AURORA_MIN_FLOW_FRACTION
            / 10.0,
            rtol=0.0,
            atol=1.0e-5,
        )

    def test_zoom_plan_reflects_at_ceiling_instead_of_freezing(self):
        instrumental = np.concatenate((
            np.ones(100, dtype=np.float32),
            np.zeros(300, dtype=np.float32),
        ))
        zooms = visualizer._zoom_plan(instrumental, "1", "1e12", punch=3.0)
        self.assertAlmostEqual(float(zooms[-1]), 12.0, places=12)
        self.assertGreater(float(np.max(zooms)), 11.0)
        self.assertLess(float(np.min(zooms[95:105])), 12.0)
        self.assertLess(float(np.min(np.diff(zooms))), 0.0)

    def test_onset_strength_can_add_transient_zoom_energy(self):
        instrumental = np.zeros(8, dtype=np.float32)
        onset = np.zeros(8, dtype=np.float32)
        onset[3] = 1.0
        loudness_only = visualizer._zoom_plan(
            instrumental, "1", "1e8", punch=0.0, quiet_speed=0.0
        )
        onset_sync = visualizer._zoom_plan(
            instrumental, "1", "1e8", punch=0.0, quiet_speed=0.0,
            onset=onset, beat_strength=2.0,
        )
        self.assertAlmostEqual(float(loudness_only[-1]), 8.0, places=12)
        self.assertAlmostEqual(float(onset_sync[-1]), 8.0, places=12)
        self.assertFalse(np.array_equal(loudness_only, onset_sync))

    def test_formula_defaults_and_profile_defaults_are_available(self):
        self.assertEqual(visualizer._formula_name("burningship"), "burning-ship")
        self.assertEqual(visualizer._parse_coordinate_pair("-0.8,0.156", "julia"), ("-0.8", "0.156"))
        self.assertEqual(PROFILE_DEFAULTS["fullhd"], PROFILE_DEFAULTS["1080p"])
        self.assertEqual(PROFILE_DEFAULTS["fullhd"]["width"], 1920)
        self.assertEqual(PROFILE_DEFAULTS["fullhd"]["height"], 1080)
        self.assertEqual(PROFILE_DEFAULTS["4k-e150"]["width"], 3840)
        self.assertGreater(PROFILE_DEFAULTS["4k-e150"]["max_zoom"].count("e"), 0)

    def test_palette_file_and_frame_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            palette_path = Path(directory) / "palette.txt"
            palette_path.write_text("#000000\n#ffffff\n", encoding="utf-8")
            palette = visualizer._palette_from_file(palette_path, 8)
            self.assertEqual(tuple(palette[0]), (0, 0, 0))
            self.assertEqual(tuple(palette[-1]), (255, 255, 255))
            field = np.linspace(0.0, 10.0, 64).reshape(8, 8).astype(np.float32)
            rgb = visualizer._colourise_custom(
                field, 10, 0.0, 0.5, 0.5, "aurora", palette_file=palette_path
            )
            self.assertEqual(rgb.shape, (8, 8, 3))
            effected = visualizer._apply_frame_effects(
                rgb, glow=0.2, motion_blur=0.5, previous=np.zeros_like(rgb)
            )
            self.assertEqual(effected.dtype, np.uint8)
            self.assertTrue(np.isfinite(effected).all())

    def test_native_formula_modes_produce_fields(self):
        if visualizer._get_native_library() is None:
            raise unittest.SkipTest("native renderer is unavailable")
        for formula in ("julia", "burning-ship", "tricorn", "multibrot3"):
            field = visualizer.render_fractal(
                40,
                30,
                0.0,
                *visualizer.FORMULA_DEFAULT_CENTERS[formula],
                128,
                "native",
                2,
                formula=formula,
            )
            self.assertEqual(field.shape, (30, 40))
            self.assertTrue(np.isfinite(field).all())
            expected = visualizer._render_direct(
                40,
                30,
                0.0,
                *visualizer.FORMULA_DEFAULT_CENTERS[formula],
                128,
                formula,
            )
            np.testing.assert_allclose(field, expected, rtol=0.0, atol=1.0e-4)

    def test_deep_local_reference_geometry_supports_odd_tiles(self):
        centres = visualizer._atlas_local_reference_centres(
            125,
            125,
            4000.25,
            visualizer.DEFAULT_X_CENTER,
            visualizer.DEFAULT_Y_CENTER,
        )
        self.assertEqual(len(centres), 4)
        self.assertEqual(
            [(x0, x1, y0, y1) for x0, x1, y0, y1, _, _ in centres],
            [(0, 62, 0, 62), (62, 125, 0, 62),
             (0, 62, 62, 125), (62, 125, 62, 125)],
        )
        self.assertTrue(all(
            Decimal(x).is_finite() and Decimal(y).is_finite()
            for _, _, _, _, x, y in centres
        ))

    def test_deep_local_references_match_single_reference_field(self):
        library = visualizer._get_native_library()
        if library is None:
            raise unittest.SkipTest("native renderer is unavailable")
        width = height = 96
        max_iter = 1800
        library, reference = visualizer._create_native_reference(
            visualizer.DEFAULT_X_CENTER,
            visualizer.DEFAULT_Y_CENTER,
            max_iter,
            40.0,
            3,
            40.0,
        )
        try:
            single = visualizer.render_fractal(
                width,
                height,
                40.0,
                visualizer.DEFAULT_X_CENTER,
                visualizer.DEFAULT_Y_CENTER,
                max_iter,
                "native",
                2,
                reference,
                3,
                256,
            )
        finally:
            library.fractal_destroy_reference(reference)
        local = visualizer._atlas_local_reference_field(
            render_width=width,
            render_height=height,
            log10_zoom=40.0,
            x_center=visualizer.DEFAULT_X_CENTER,
            y_center=visualizer.DEFAULT_Y_CENTER,
            max_iter=max_iter,
            series_order=3,
            series_block=256,
            renderer="native",
            native_threads=2,
            native_library=library,
        )
        np.testing.assert_allclose(local, single, rtol=0.0, atol=1.0e-3)

    def test_probe_centred_point_reference_preserves_global_pixel_alignment(self):
        library = visualizer._get_native_library()
        if library is None or not hasattr(library, "fractal_render_points"):
            raise unittest.SkipTest("native point renderer is unavailable")
        width = height = 64
        max_iter = 1800
        _, global_reference = visualizer._create_native_reference(
            visualizer.DEFAULT_X_CENTER,
            visualizer.DEFAULT_Y_CENTER,
            max_iter,
            40.0,
            3,
            40.0,
        )
        cell = (5, 14, 7, 17)
        probe = (12, 9)
        geometry = visualizer._atlas_local_reference_cell(
            width,
            height,
            40.0,
            visualizer.DEFAULT_X_CENTER,
            visualizer.DEFAULT_Y_CENTER,
            cell,
            probe,
        )
        _, _, _, _, local_x, local_y, local_log_zoom = geometry
        _, local_reference = visualizer._create_native_reference(
            local_x,
            local_y,
            max_iter,
            local_log_zoom,
            3,
            local_log_zoom,
        )
        try:
            expected = visualizer.render_fractal(
                width,
                height,
                40.0,
                visualizer.DEFAULT_X_CENTER,
                visualizer.DEFAULT_Y_CENTER,
                max_iter,
                "native",
                2,
                global_reference,
                3,
                256,
                visualizer.NativeRenderOptions(strict=True, strict_cycle=True),
            )
            actual = visualizer._render_native_reference_points(
                render_width=width,
                render_height=height,
                log10_zoom=40.0,
                cell=cell,
                probe=probe,
                x_center=visualizer.DEFAULT_X_CENTER,
                y_center=visualizer.DEFAULT_Y_CENTER,
                reference_x=local_x,
                reference_y=local_y,
                max_iter=max_iter,
                native_threads=2,
                native_library=library,
                native_reference=local_reference,
                series_order=3,
                series_block=256,
                render_options=visualizer.NativeRenderOptions(
                    strict=True,
                    strict_cycle=True,
                ),
            )
        finally:
            library.fractal_destroy_reference(local_reference)
            library.fractal_destroy_reference(global_reference)
        np.testing.assert_allclose(
            actual,
            expected[cell[2]:cell[3], cell[0]:cell[1]],
            rtol=0.0,
            atol=2.0e-3,
        )

    def test_atlas_memory_cache_is_bounded_in_both_directions(self):
        tiles = {level: object() for level in range(7)}
        visualizer._trim_atlas_memory_cache(tiles, 2)
        self.assertEqual(set(tiles), {1, 2, 3})
        tiles.update({0: object(), 4: object(), 5: object()})
        visualizer._trim_atlas_memory_cache(tiles, 4)
        self.assertEqual(set(tiles), {3, 4, 5})

    def test_cache_evictor_is_incremental_and_protects_active_tiles(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            paths = [cache_dir / f"atlas-tile-{index}.npy" for index in range(4)]
            for path in paths:
                path.write_bytes(b"01234567")
            limit_mb = 16.0 / (1024.0 * 1024.0)
            evictor = visualizer._CacheEvictor(cache_dir, limit_mb)
            evictor.prune({paths[-1]})
            remaining = [path for path in paths if path.exists()]
            self.assertIn(paths[-1], remaining)
            self.assertLessEqual(sum(path.stat().st_size for path in remaining), 16)
            self.assertTrue(evictor._scanned)

    def test_zoom_plan_has_default_pullback_and_reaches_end(self):
        instrumental = np.asarray([0.0, 0.1, 1.0, 0.0, 0.8, 0.0], dtype=np.float32)
        zooms = visualizer._zoom_plan(instrumental, "1", "1e12", punch=3.0)
        self.assertEqual(zooms.shape, instrumental.shape)
        self.assertAlmostEqual(float(zooms[0]), 0.0, places=12)
        self.assertAlmostEqual(float(zooms[-1]), 12.0, places=9)
        self.assertGreaterEqual(float(np.min(zooms)), 0.0)
        self.assertLess(float(np.min(np.diff(zooms))), 0.0)
        self.assertGreater(float(np.max(np.diff(zooms))), 0.0)
        self.assertGreater(float(zooms[3] - zooms[2]), 0.0)

    def test_zoom_chunks_cover_signed_path_without_zooming_out(self):
        zooms = np.asarray([0.0, -0.1, 0.2, 0.45, 0.1, 0.8], dtype=np.float64)
        chunks = list(visualizer._zoom_chunks(zooms, 2.0))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], zooms.size)
        for _, _, low, high in chunks:
            self.assertLess(high - low, np.log10(2.0))

    def test_atlas_has_fixed_levels_independent_of_frame_boundaries(self):
        zooms = np.linspace(0.0, 100.0, 1001, dtype=np.float64)
        origin, step, level_count = visualizer._atlas_geometry(zooms, 2.0)
        self.assertAlmostEqual(origin, 0.0, places=12)
        self.assertAlmostEqual(step, np.log10(2.0), places=12)
        self.assertEqual(level_count, 333)
        self.assertEqual(visualizer._atlas_level_for_zoom(0.0, origin, step, level_count), 0)
        self.assertEqual(
            visualizer._atlas_level_for_zoom(100.0, origin, step, level_count),
            332,
        )
        self.assertEqual(
            visualizer._atlas_level_for_zoom(origin + step * level_count, origin, step, level_count),
            level_count,
        )

    def test_atlas_compositor_uses_child_only_in_central_region(self):
        parent = np.zeros((8, 8), dtype=np.float32)
        child = np.ones((8, 8), dtype=np.float32)
        original = visualizer._colourise_view

        def fake_colour(view, *args):
            output_height, output_width = view.shape
            value = 20 if float(view[view.shape[0] // 2, view.shape[1] // 2]) > 0.5 else 10
            result = np.full((output_height, output_width, 3), value, dtype=np.uint8)
            result[view < 0.5] = 10
            result[view > 0.5] = 20
            return result

        visualizer._colourise_view = fake_colour
        try:
            result = visualizer._atlas_colour_frame(
                parent,
                child,
                32,
                32,
                1.0,
                0.5,
                100,
                100,
                0.0,
                0.0,
                0.0,
                None,
                1,
                "bilinear",
                "aurora",
                0.5,
            )
        finally:
            visualizer._colourise_view = original
        self.assertEqual(result.shape, (32, 32, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result[0, 0, 0]), 10)
        self.assertEqual(int(result[16, 16, 0]), 20)

    def test_custom_palettes_are_cached_and_finite(self):
        field = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        for name in ("fire", "ocean", "neon", "sunset", "mono"):
            rgb = visualizer._colourise_custom(field, 100, 1.5, 0.8, 0.6, name)
            self.assertEqual(rgb.shape, (64, 64, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertTrue(np.isfinite(rgb).all())
        self.assertIs(visualizer._custom_palette("fire"), visualizer._custom_palette("fire"))

    def test_pitch_rotates_legacy_two_hue_gradient(self):
        field = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        baseline = visualizer._colourise(field, 100, 0.0, 0.5, 0.8, 0.5)
        low = visualizer._colourise(field, 100, 0.0, 0.5, 0.8, 0.0)
        high = visualizer._colourise(field, 100, 0.0, 0.5, 0.8, 1.0)
        channel_spread = np.max(
            baseline.astype(np.int16).max(axis=-1)
            - baseline.astype(np.int16).min(axis=-1)
        )
        self.assertGreater(int(channel_spread), 20)
        self.assertGreater(int(np.unique(baseline.reshape(-1, 3), axis=0).shape[0]), 32)
        self.assertEqual(visualizer._pitch_hue_angle(0.5), 0.0)
        self.assertLess(visualizer._pitch_hue_angle(0.0), 0.0)
        self.assertGreater(visualizer._pitch_hue_angle(1.0), 0.0)
        self.assertFalse(np.array_equal(low, high))
        self.assertFalse(np.array_equal(baseline, low))
        self.assertFalse(np.array_equal(baseline, high))

    def test_spatial_recovery_preserves_fallback_structure(self):
        local = np.full((6, 6), np.nan, dtype=np.float32)
        fallback = np.arange(36, dtype=np.float32).reshape(6, 6)
        recovered, count = visualizer._spatial_recover_field(local, fallback)
        self.assertEqual(count, 36)
        self.assertTrue(np.isfinite(recovered).all())
        np.testing.assert_array_equal(recovered, fallback)

    def test_spatial_recovery_does_not_flat_fill_partial_holes(self):
        local = np.arange(49, dtype=np.float32).reshape(7, 7)
        local[2:5, 2:5] = np.nan
        recovered, count = visualizer._spatial_recover_field(local)
        self.assertEqual(count, 9)
        self.assertTrue(np.isfinite(recovered).all())
        self.assertGreater(int(np.unique(recovered[2:5, 2:5]).size), 1)

    def test_unresolved_regions_keep_a_real_probe_pixel(self):
        mask = np.zeros((10, 12), dtype=bool)
        mask[2:5, 7:10] = True
        regions = visualizer._unresolved_regions(mask)
        self.assertEqual(len(regions), 1)
        x0, x1, y0, y1, count, probe_x, probe_y = regions[0]
        self.assertEqual(count, 9)
        self.assertEqual((x0, x1, y0, y1), (6, 11, 1, 6))
        self.assertTrue(mask[probe_y, probe_x])

    def test_explicit_video_codec_keeps_x264_rate_control(self):
        codec, preset, rate_control = visualizer._select_video_encoder(
            "libx264", "ultrafast", 18
        )
        self.assertEqual(codec, "libx264")
        self.assertEqual(preset, "ultrafast")
        self.assertEqual(rate_control, ["-crf", "18"])

    def test_auto_codec_uses_probed_vaapi_when_available(self):
        with mock.patch.object(
            visualizer, "_ffmpeg_encoder_names", return_value={"h264_vaapi"}
        ), mock.patch.object(
            visualizer, "_vaapi_encoder_usable", return_value=True
        ), mock.patch.object(visualizer.Path, "exists", return_value=True):
            codec, preset, rate_control = visualizer._select_video_encoder(
                "auto", "ultrafast", 18
            )
        self.assertEqual(codec, "h264_vaapi")
        self.assertEqual(preset, "")
        self.assertEqual(rate_control, ["-qp", "18"])

    def test_crop_factor_is_clamped_to_native_source(self):
        field = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        view = visualizer._crop_and_resize(field, 32, 32, 2.0, "bilinear")
        self.assertEqual(view.shape, (32, 32))
        self.assertTrue(np.isfinite(view).all())

    def test_scaled_exponential_radius_preserves_e4000(self):
        mantissa, exponent = visualizer._scaled_log10_radius(-4000.0)
        self.assertTrue(np.isfinite(mantissa))
        self.assertGreaterEqual(mantissa, 0.5)
        self.assertLess(mantissa, 1.0)
        self.assertLess(exponent, -12000)

    def test_exponential_field_smoke_is_raw_and_reprojectable(self):
        library = visualizer._get_native_library()
        if library is None or not hasattr(library, "fractal_render_points"):
            raise unittest.SkipTest("native point renderer is unavailable")
        field = visualizer.render_exponential_field(
            radial_samples=8,
            angular_samples=16,
            min_log10_radius=-8.0,
            max_log10_radius=-6.0,
            x_center=visualizer.DEFAULT_X_CENTER,
            y_center=visualizer.DEFAULT_Y_CENTER,
            max_iter=400,
            native_threads=2,
        )
        self.assertEqual(field.shape, (8, 16))
        self.assertEqual(field.dtype, np.float32)
        self.assertTrue(np.isfinite(field).all())
        view = visualizer.reproject_exponential_field(
            field,
            min_log10_radius=-8.0,
            max_log10_radius=-6.0,
            log10_zoom=6.5,
            output_width=12,
            output_height=10,
            max_iter=400,
        )
        self.assertEqual(view.shape, (10, 12))
        self.assertTrue(np.isfinite(view).all())

    def test_envelope_follower_is_bounded_and_release_is_smooth(self):
        values = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        followed = visualizer._envelope_follow(values, 60, 0.02, 0.2)
        self.assertTrue(np.all((followed >= 0.0) & (followed <= 1.0)))
        self.assertGreater(float(followed[1]), 0.0)
        self.assertLess(float(followed[2]), float(followed[1]))
        self.assertLess(float(followed[2]), 1.0)


class AudioTimingTests(unittest.TestCase):
    def test_feature_resampling_uses_rounded_analysis_hop(self):
        values = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        result = visualizer._resample_features(values, 4, 59, 44100)
        hop = round(44100 / 59)
        source_times = np.arange(values.size) * hop / 44100.0
        target_times = np.arange(4) / 59.0
        expected = np.interp(target_times, source_times, values)
        np.testing.assert_allclose(result, expected.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
