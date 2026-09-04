import unittest
import tempfile
import math
import os
import struct
from decimal import Decimal
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"NumPy is unavailable: {error}") from error

import visualizer
from deep_zoom_points import FORMULA_POINT_CATALOGUES
from profiles import DEFAULT_PROFILE, PROFILE_DEFAULTS


class AnimationTests(unittest.TestCase):
    def test_fractional_decimal_places_respect_scientific_notation(self):
        self.assertEqual(visualizer._fractional_decimal_places("1.23e-4"), 6)
        self.assertEqual(visualizer._fractional_decimal_places("-1.2300"), 4)
        self.assertEqual(visualizer._fractional_decimal_places("12e3"), 0)

    def test_decimal_to_scaled_preserves_non_power_of_two_magnitude(self):
        value = Decimal("2.4604481033564476e-68")
        mantissa, exponent = visualizer._decimal_to_scaled(value)
        reconstructed = math.ldexp(mantissa, exponent)
        self.assertTrue(math.isfinite(reconstructed))
        self.assertAlmostEqual(reconstructed, float(value), places=80)

        negative_mantissa, negative_exponent = visualizer._decimal_to_scaled(-value)
        self.assertAlmostEqual(
            math.ldexp(negative_mantissa, negative_exponent),
            -float(value),
            places=80,
        )

    def test_default_center_has_precision_for_the_default_e150_profile(self):
        available, required = visualizer._center_precision_budget(
            visualizer.DEFAULT_X_CENTER,
            visualizer.DEFAULT_Y_CENTER,
            100.0,
        )
        self.assertGreaterEqual(available, required)
        self.assertIsNone(
            visualizer._center_precision_error(
                visualizer.DEFAULT_X_CENTER,
                visualizer.DEFAULT_Y_CENTER,
                100.0,
            )
        )
        available, required = visualizer._center_precision_budget(
            "-1.234567890123456789",
            "0.123456789012345678",
            150.0,
        )
        self.assertLess(available, required)
        self.assertIn(
            "full-precision --x-center and --y-center",
            visualizer._center_precision_error(
                "-1.234567890123456789",
                "0.123456789012345678",
                150.0,
            ),
        )

    def test_curated_point_catalogue_has_screened_e150_locations(self):
        points = visualizer.DEEP_ZOOM_POINTS
        self.assertGreaterEqual(len(points), 18)
        self.assertEqual(len({point.slug for point in points}), len(points))
        self.assertGreaterEqual(
            sum(point.conjugate_of is None for point in points),
            9,
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

    def test_formula_catalogues_are_individual(self):
        all_slugs = []
        for formula, points in FORMULA_POINT_CATALOGUES.items():
            self.assertGreater(len(points), 0)
            self.assertTrue(all(point.formula == formula for point in points))
            all_slugs.extend(point.slug for point in points)
        self.assertEqual(len(all_slugs), len(set(all_slugs)))
        self.assertNotIn(
            "oldwooddish",
            {point.slug for point in FORMULA_POINT_CATALOGUES["burning-ship"]},
        )

    def test_formula_point_resolution_uses_the_selected_catalogue(self):
        burning_ship = visualizer._resolve_render_point(
            point_spec="burning-ship-mini-ship",
            random_point=False,
            x_center=None,
            y_center=None,
            random_seed=None,
            max_log_zoom=50.0,
            formula="burning-ship",
        )
        self.assertEqual(burning_ship[2].formula, "burning-ship")
        self.assertNotEqual((burning_ship[0], burning_ship[1]), ("-2.0", "0.0"))
        self.assertNotIn(
            "burning-ship-left-tip",
            {point.slug for point in FORMULA_POINT_CATALOGUES["burning-ship"]},
        )

        tricorn = visualizer._resolve_render_point(
            point_spec="tricorn-branching-junction",
            random_point=False,
            x_center=None,
            y_center=None,
            random_seed=None,
            max_log_zoom=150.0,
            formula="tricorn",
        )
        self.assertEqual(tricorn[2].formula, "tricorn")
        self.assertNotEqual((tricorn[0], tricorn[1]), ("-2.0", "0.0"))
        self.assertNotIn(
            "tricorn-left-tip",
            {point.slug for point in FORMULA_POINT_CATALOGUES["tricorn"]},
        )

        julia = visualizer._resolve_render_point(
            point_spec="julia-douady-rabbit",
            random_point=False,
            x_center=None,
            y_center=None,
            random_seed=None,
            max_log_zoom=1.0,
            formula="julia",
        )
        self.assertEqual(julia[2].julia_c, ("-0.123", "0.745"))

        self.assertEqual(
            visualizer.FORMULA_CHOICES,
            ("mandelbrot", "julia", "burning-ship", "tricorn"),
        )

        with self.assertRaisesRegex(ValueError, "unknown burning-ship point"):
            visualizer._resolve_render_point(
                point_spec="oldwooddish",
                random_point=False,
                x_center=None,
                y_center=None,
                random_seed=None,
                max_log_zoom=1.0,
                formula="burning-ship",
            )

    def test_alternate_e150_catalogue_targets_are_finite_and_structured(self):
        for formula, points in visualizer.FORMULA_POINT_CATALOGUES.items():
            if formula == "mandelbrot":
                continue
            for point in points:
                x_center, y_center, selected = visualizer._resolve_render_point(
                    point_spec=point.slug,
                    random_point=False,
                    x_center=None,
                    y_center=None,
                    random_seed=None,
                    max_log_zoom=150.0,
                    formula=formula,
                )
                constant = (
                    selected.julia_c
                    if formula == "julia" and selected is not None
                    else visualizer.DEFAULT_JULIA_C
                )
                field = visualizer.render_fractal(
                    12,
                    8,
                    150.0,
                    x_center,
                    y_center,
                    4000,
                    "python",
                    formula=formula,
                    julia_constant=constant,
                )
                # A valid deep target can have a filament thinner than this
                # deliberately tiny smoke-test frame. Recheck those sparse
                # targets at the source-preview geometry before rejecting the
                # preset as a flat/blank view.
                if float(np.ptp(field)) == 0.0:
                    field = visualizer.render_fractal(
                        256,
                        192,
                        150.0,
                        x_center,
                        y_center,
                        4000,
                        "python",
                        formula=formula,
                        julia_constant=constant,
                    )
                self.assertTrue(np.isfinite(field).all(), point.slug)
                self.assertGreater(float(np.ptp(field)), 0.0, point.slug)

    def test_burning_ship_e150_axis_crossing_does_not_false_escape(self):
        point = visualizer.FORMULA_POINT_CATALOGUES["burning-ship"][0]
        field = visualizer.render_fractal(
            5,
            5,
            150.0,
            point.x,
            point.y,
            1000,
            "python",
            formula="burning-ship",
        )
        # The exported target is a repelling boundary cycle.  Stabilizing the
        # reference keeps the exact centre bounded while nearby pixels still
        # escape; carrying an incorrect absolute-value sign into the
        # perturbation path used to turn this whole sample into one flat cap.
        self.assertEqual(float(field[2, 2]), 1000.0)
        self.assertLess(float(np.min(field)), 400.0)
        self.assertGreater(float(np.ptp(field)), 500.0)

    def test_exact_minus_two_boundary_margin_is_preserved(self):
        for formula in ("burning-ship", "tricorn"):
            field = visualizer.render_fractal(
                3,
                3,
                150.0,
                "-2.0",
                "0.0",
                1000,
                "python",
                formula=formula,
            )
            # The upper-right pixel is just outside the exact -2 tip and
            # escapes around iteration 251. It must not be marked bounded by
            # the period-one interior shortcut before that happens.
            self.assertAlmostEqual(float(field[0, 2]), 251.0, delta=2.0)

    def test_tricorn_e150_boundary_margin_is_preserved(self):
        point = visualizer.FORMULA_POINTS_BY_SLUG["tricorn"][
            "tricorn-branching-junction"
        ]
        field = visualizer.render_fractal(
            5,
            5,
            150.0,
            point.x,
            point.y,
            1000,
            "python",
            formula="tricorn",
        )
        self.assertTrue(np.isfinite(field).all())
        # The exact odd-sized centre is the stabilized boundary target. Its
        # neighbours must still escape, proving that the cycle repair did not
        # flatten the whole Tricorn viewport into one interior block.
        self.assertEqual(float(field[2, 2]), 1000.0)
        self.assertGreater(float(np.ptp(field)), 10.0)
        self.assertLess(float(np.min(field)), 900.0)

    def test_julia_e150_fixed_point_reference_is_stabilized(self):
        for point in visualizer.JULIA_POINTS:
            field = visualizer.render_fractal(
                3,
                3,
                150.0,
                point.x,
                point.y,
                1000,
                "python",
                formula="julia",
                julia_constant=point.julia_c,
            )
            # The centre is the analytically recovered repelling fixed point,
            # not a finite-decimal orbit that eventually drifts away after
            # its last supplied digit.
            self.assertEqual(float(field[1, 1]), 1000.0, point.slug)

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
            [12.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 83.0],
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
        expected_first_deep = max(
            12.0,
            math.floor(20.0 / step) * step - 3.0 * step,
        )
        self.assertAlmostEqual(tiers[1], expected_first_deep, places=12)
        self.assertLessEqual(tiers[1], 20.0)
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
        self.assertEqual(PROFILE_DEFAULTS["4k-e150"]["fps"], 60)
        self.assertGreater(PROFILE_DEFAULTS["4k-e150"]["max_zoom"].count("e"), 0)
        self.assertEqual(PROFILE_DEFAULTS["4k-e150"]["fractal_scale"], 0.25)
        self.assertEqual(PROFILE_DEFAULTS["4k-e150"]["keyframe_factor"], 8.0)
        self.assertEqual(PROFILE_DEFAULTS["4k-e150-lossless"]["fps"], 60)
        self.assertEqual(PROFILE_DEFAULTS["4k-e150-lossless"]["fractal_scale"], 1.0)
        self.assertTrue(PROFILE_DEFAULTS["4k-e150-lossless"]["lossless"])
        self.assertEqual(PROFILE_DEFAULTS["4k-e150-lossless"]["crf"], 0)
        self.assertEqual(PROFILE_DEFAULTS["4k-e150-lossless"]["resample"], "lanczos")
        args = visualizer.build_parser(["--profile", "4k-e150-lossless"]).parse_args([])
        self.assertTrue(args.lossless)
        self.assertEqual(args.crf, 0)
        self.assertEqual(args.fractal_scale, 1.0)
        defaults = visualizer.build_parser().parse_args([])
        self.assertEqual(defaults.profile, DEFAULT_PROFILE)
        self.assertTrue(defaults.lossless)
        self.assertEqual(defaults.width, 3840)
        self.assertEqual(defaults.height, 2160)
        self.assertEqual(PROFILE_DEFAULTS["preview"]["fractal_scale"], 1.0)

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

    def test_kfp_palette_import_and_bundled_default(self):
        bundled = Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp"
        palette = visualizer._palette_from_file(bundled, 17)
        self.assertEqual(tuple(palette[0]), (255, 255, 255))
        self.assertEqual(tuple(palette[-1]), (0, 0, 255))
        profile = visualizer._kfp_profile_for_selection("kalles-default", None)
        self.assertIsNotNone(profile)
        assert profile is not None
        file_profile = visualizer._kfp_profile_for_selection("aurora", bundled)
        self.assertEqual(file_profile, profile)
        self.assertEqual(profile.stops, (
            (255, 255, 255),
            (128, 0, 64),
            (160, 0, 0),
            (192, 128, 0),
            (64, 128, 0),
            (0, 255, 255),
            (64, 128, 255),
            (0, 0, 255),
        ))
        self.assertEqual(
            (profile.iter_div, profile.color_method, profile.differences),
            (0.01, 7, 3),
        )
        self.assertTrue(profile.slopes)
        self.assertEqual(profile.interior_color, (0, 0, 0))
        self.assertEqual(
            tuple(visualizer._custom_palette("kalles-default")[0]),
            (255, 255, 255),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continued.KFP"
            path.write_text(
                "Colors: 0,0,0,\n 255, 255, 255\nSmooth: 1\n",
                encoding="utf-8",
            )
            imported = visualizer._palette_from_file(path, 3)
            np.testing.assert_array_equal(
                imported,
                np.asarray([[0, 0, 0], [128, 128, 128], [255, 255, 255]], dtype=np.uint8),
            )
            profile = visualizer._kfp_profile_for_selection("aurora", path)
            self.assertIsNotNone(profile)
            assert profile is not None
            self.assertEqual(profile.interior_color, (0, 0, 0))

            extended = Path(directory) / "extended.kfp"
            extended.write_text(
                "Colors: 0,0,0,255,255,255,\n"
                "InteriorColor: 1,2,3,\n"
                "ColorMethod: 11\nFlat: 1\nInverseTransition: 1\n",
                encoding="utf-8",
            )
            extended_profile = visualizer._kfp_profile_for_selection(
                "aurora", extended
            )
            self.assertIsNotNone(extended_profile)
            assert extended_profile is not None
            self.assertEqual(extended_profile.color_method, 11)
            self.assertTrue(extended_profile.flat)
            self.assertTrue(extended_profile.inverse_transition)
            self.assertEqual(extended_profile.interior_color, (1, 2, 3))

    def test_kfp_palette_lut_wraps_from_last_stop_to_first(self):
        profile = visualizer.KfpPalette(
            stops=((0, 0, 0), (255, 255, 255)),
        )
        lut = visualizer._kfp_palette_lut(profile, 16)
        self.assertEqual(tuple(lut[0]), (0, 0, 0))
        self.assertLess(int(lut[-1, 0]), 255)

    def test_kfp_palette_lut_matches_kalles_sine_interpolation(self):
        profile = visualizer.KfpPalette(
            stops=((7, 19, 241), (240, 31, 11), (63, 222, 95)),
        )
        size = 37
        lut = visualizer._kfp_palette_lut(profile, size)
        expected = []
        for index in range(size):
            position = index * len(profile.stops) / size
            left = int(math.floor(position))
            right = (left + 1) % len(profile.stops)
            fraction = position - left
            fraction = math.sin((fraction - 0.5) * math.pi) / 2.0 + 0.5
            expected.append(tuple(
                int(
                    fraction * profile.stops[right][channel]
                    + (1.0 - fraction) * profile.stops[left][channel]
                )
                for channel in range(3)
            ))
        np.testing.assert_array_equal(lut, np.asarray(expected, dtype=np.uint8))

    def test_kfp_zero_distance_uses_the_imported_first_colour(self):
        field = np.zeros((5, 7), dtype=np.float32)
        rgb = visualizer._colourise_kfp(
            field, 200, 0.0, 0.0, 0.0, 0.5, visualizer.KALLES_DEFAULT_KFP
        )
        np.testing.assert_array_equal(rgb, np.full_like(rgb, (255, 255, 255)))

    def test_dark_builtin_themes_have_white_interiors(self):
        field = np.asarray([[20.0, 100.0]], dtype=np.float32)
        for name in ("midnight", "ember-night", "terminal"):
            rgb = visualizer._colourise_custom(field, 100, 0.0, 0.5, 0.5, name)
            self.assertEqual(tuple(rgb[0, 1]), (255, 255, 255), name)
            self.assertLess(int(np.mean(rgb[0, 0])), 220, name)

    def test_kfp_difference_matches_kalles_traditional_stencil(self):
        field = np.asarray(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [20.0, 7.0, 8.0]],
            dtype=np.float32,
        )
        gradient = visualizer._kfp_difference_magnitude(field, 0, np)
        # Kalles multiplies each delta by sqrt(2), then divides diagonal
        # neighbours by their geometric distance sqrt(2). Axis terms retain
        # sqrt(2), while diagonal terms retain unit weight.
        expected = 20.0 + 4.0 * math.sqrt(2.0)
        self.assertAlmostEqual(float(gradient[1, 1]), expected, places=12)

    def test_kfp_de_plus_standard_uses_kalles_sqrt_distance(self):
        profile = visualizer.KfpPalette(
            stops=((0, 0, 0), (255, 255, 255)),
            iter_div=11.0,
            color_offset=256.0,
            color_method=6,
            smooth=False,
            slopes=False,
            differences=0,
        )
        field = np.asarray([[0.0, 5000.0, 20000.0]], dtype=np.float32)
        sqrt_rgb = visualizer._colourise_kfp(
            field, 100000, 0.0, 0.0, 0.0, 0.5, profile
        )
        linear_rgb = visualizer._colourise_kfp(
            field,
            100000,
            0.0,
            0.0,
            0.0,
            0.5,
            visualizer.KfpPalette(
                stops=profile.stops,
                iter_div=profile.iter_div,
                color_offset=profile.color_offset,
                color_method=5,
                smooth=False,
                slopes=False,
                differences=profile.differences,
            ),
        )
        # The left sample sees a non-zero traditional distance.  Kalles'
        # square root keeps DEPlusStandard below IterDiv here, while
        # DistanceLinear retains the unrooted distance, so the two colours
        # must differ.
        self.assertGreater(int(linear_rgb[0, 0, 0]), int(sqrt_rgb[0, 0, 0]))

    def test_kfp_de_plus_standard_flat_keeps_smooth_fallback(self):
        field = np.asarray([[0.25, 100.75, 0.25]], dtype=np.float32)
        common = dict(
            stops=((0, 0, 0), (255, 255, 255)),
            iter_div=0.1,
            color_method=6,
            smooth=True,
            slopes=False,
            differences=0,
        )
        flat = visualizer._colourise_kfp(
            field, 200, 0.0, 0.0, 0.0, 0.5,
            visualizer.KfpPalette(flat=True, **common),
        )
        fractional = visualizer._colourise_kfp(
            field, 200, 0.0, 0.0, 0.0, 0.5,
            visualizer.KfpPalette(flat=False, **common),
        )
        # The distance branch falls back to nIter + 1 - offs in Kalles,
        # independently of Flat. The high-gradient centre therefore keeps
        # its fractional transfer in both profiles.
        self.assertEqual(tuple(flat[0, 1]), tuple(fractional[0, 1]))

    def test_native_kfp_colouriser_matches_portable_boundary_stencils(self):
        library = visualizer._get_native_library()
        if library is None or not hasattr(library, "fractal_colourise_kfp"):
            raise unittest.SkipTest("native KFP colouriser is unavailable")
        rng = np.random.default_rng(714)
        field = rng.uniform(0.0, 180.0, size=(17, 23)).astype(np.float32)
        field[0, 0] = 200.0
        field[8, 11] = 200.0
        field[-1, -1] = 200.0
        field[1, 2] = np.nan
        field[9, 13] = np.inf
        field[15, 20] = -np.inf
        portable = visualizer._colourise_kfp(
            field, 200, 0.0, 0.0, 0.0, 0.5, visualizer.KALLES_DEFAULT_KFP
        )
        native = visualizer._colourise_kfp_native(
            field,
            200,
            0.0,
            0.0,
            0.0,
            0.5,
            visualizer.KALLES_DEFAULT_KFP,
            library,
            2,
        )
        difference = np.abs(native.astype(np.int16) - portable.astype(np.int16))
        self.assertLessEqual(int(np.max(difference)), 1)

        # Exercise a non-default imported transfer as well; this protects the
        # native DEPlusStandard square-root path from drifting away from the
        # portable Kalles reference.
        de_profile = visualizer.KfpPalette(
            stops=((0, 0, 0), (255, 255, 255)),
            iter_div=1000.0,
            color_offset=128.0,
            color_method=6,
            smooth=False,
            slopes=True,
            slope_power=30.0,
            slope_ratio=20.0,
            differences=0,
        )
        portable = visualizer._colourise_kfp(
            field, 200, 0.0, 0.0, 0.0, 0.5, de_profile
        )
        native = visualizer._colourise_kfp_native(
            field,
            200,
            0.0,
            0.0,
            0.0,
            0.5,
            de_profile,
            library,
            2,
        )
        difference = np.abs(native.astype(np.int16) - portable.astype(np.int16))
        self.assertLessEqual(int(np.max(difference)), 1)

    def test_native_kfp_crop_matches_direct_at_unit_zoom(self):
        library = visualizer._get_native_library()
        if library is None or not hasattr(library, "fractal_crop_colourise_kfp"):
            raise unittest.SkipTest("native KFP crop colourizer is unavailable")
        rng = np.random.default_rng(1204)
        field = rng.uniform(0.0, 180.0, size=(19, 27)).astype(np.float32)
        field[2:5, 8:12] = 200.0
        cropped = visualizer._crop_colourise_kfp_native(
            field,
            field.shape[1],
            field.shape[0],
            1.0,
            200,
            0.0,
            0.0,
            0.0,
            0.5,
            visualizer.KALLES_DEFAULT_KFP,
            library,
            2,
        )
        direct = visualizer._colourise_kfp_native(
            field,
            200,
            0.0,
            0.0,
            0.0,
            0.5,
            visualizer.KALLES_DEFAULT_KFP,
            library,
            2,
        )
        difference = np.abs(cropped.astype(np.int16) - direct.astype(np.int16))
        self.assertLessEqual(int(np.max(difference)), 1)
        np.testing.assert_array_equal(cropped[3, 9], (0, 0, 0))

    def test_native_kfp_atlas_colouriser_matches_portable_compositor(self):
        library = visualizer._get_native_library()
        if library is None or not hasattr(library, "fractal_atlas_colourise_kfp"):
            raise unittest.SkipTest("native KFP atlas colouriser is unavailable")
        rng = np.random.default_rng(901)
        parent = rng.uniform(0.0, 180.0, size=(24, 32)).astype(np.float32)
        child = rng.uniform(0.0, 180.0, size=(12, 16)).astype(np.float32)
        parent[0, 0] = 200.0
        child[3, 4] = 200.0
        portable = visualizer._atlas_colour_frame(
            parent,
            child,
            32,
            24,
            1.0,
            0.5,
            200,
            200,
            0.0,
            0.0,
            0.0,
            None,
            2,
            "bilinear",
            "aurora",
            0.5,
            Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp",
        )
        native = visualizer._atlas_colour_frame(
            parent,
            child,
            32,
            24,
            1.0,
            0.5,
            200,
            200,
            0.0,
            0.0,
            0.0,
            library,
            2,
            "bilinear",
            "aurora",
            0.5,
            Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp",
        )
        difference = np.abs(native.astype(np.int16) - portable.astype(np.int16))
        self.assertLessEqual(int(np.max(difference)), 1)

    def test_native_raw_kfp_atlas_matches_prepared_tiles(self):
        library = visualizer._get_native_library()
        if library is None or not hasattr(library, "fractal_atlas_colourise_kfp_raw"):
            raise unittest.SkipTest("native raw KFP atlas colourizer is unavailable")
        rng = np.random.default_rng(1907)
        parent = rng.uniform(0.0, 180.0, size=(24, 32)).astype(np.float32)
        child = rng.uniform(0.0, 180.0, size=(12, 16)).astype(np.float32)
        parent[0, 0] = 200.0
        child[3, 4] = 200.0
        profile = visualizer.KALLES_DEFAULT_KFP
        prepared = visualizer._atlas_colourise_kfp_native(
            parent,
            child,
            32,
            24,
            200,
            8,
            6,
            visualizer._atlas_feather(16, 12),
            0.0,
            0.0,
            0.0,
            0.5,
            profile,
            library,
            2,
        )
        raw = visualizer._atlas_colourise_kfp_raw_native(
            parent,
            child,
            32,
            24,
            1.0,
            0.5,
            200,
            200,
            0.0,
            0.0,
            0.0,
            0.5,
            profile,
            library,
            2,
        )
        difference = np.abs(raw.astype(np.int16) - prepared.astype(np.int16))
        self.assertLessEqual(int(np.max(difference)), 1)

    def test_public_limits_reject_aliases_and_unbounded_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "song.wav"
            audio.write_bytes(b"audio")
            output = root / "render.mp4"
            manifest = root / "render.json"
            valid = visualizer._validate_render_paths(audio, output, manifest)
            self.assertEqual(valid[0], audio.resolve())
            with self.assertRaisesRegex(ValueError, "input audio"):
                visualizer._validate_render_paths(audio, audio)
            with self.assertRaisesRegex(ValueError, "input audio"):
                visualizer._validate_render_paths(audio, output, audio)

            hardlink = root / "audio-alias.bin"
            os.link(audio, hardlink)
            with self.assertRaisesRegex(ValueError, "input audio"):
                visualizer._validate_render_paths(audio, hardlink)

            protected = root / "protected.bin"
            protected.write_bytes(b"must survive")
            output_link = root / "render-link.mp4"
            try:
                output_link.symlink_to(protected)
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaisesRegex(ValueError, "symbolic link"):
                    visualizer._validate_render_paths(audio, output_link)
                self.assertEqual(protected.read_bytes(), b"must survive")

            with self.assertRaises(ValueError):
                visualizer._validate_dimensions(10_000, 10_001, "test")
            with self.assertRaises(ValueError):
                visualizer._validate_log10_zoom(float("inf"))
            with self.assertRaises(ValueError):
                visualizer.NativeRenderOptions(series_min_terms=0)
            with self.assertRaises(ValueError):
                visualizer.NativeRenderOptions(strict=2)

            sibling = visualizer._temporary_sibling(output, "rendering")
            self.assertEqual(sibling.parent, output.parent)
            self.assertFalse(sibling.exists())

    def test_colourisers_sanitize_extreme_and_nonfinite_fields(self):
        field = np.asarray(
            [[0.0, np.finfo(np.float32).max, np.inf, -np.inf, np.nan]],
            dtype=np.float32,
        )
        aurora = visualizer._colourise(field, 100, 0.0, 0.5, 0.5)
        custom = visualizer._colourise_custom(
            field, 100, 0.0, 0.5, 0.5, "fire"
        )
        self.assertEqual(aurora.shape, (1, 5, 3))
        self.assertEqual(custom.shape, (1, 5, 3))
        self.assertEqual(aurora.dtype, np.uint8)
        self.assertEqual(custom.dtype, np.uint8)
        np.testing.assert_array_equal(aurora[0, 2:], 0)
        np.testing.assert_array_equal(custom[0, 2:], 0)

    def test_cache_safety_rejects_malformed_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "broken.npz"
            malformed.write_bytes(b"not a zip archive")
            self.assertFalse(visualizer._cache_file_is_safe(malformed))
            unknown = root / "unknown.cache"
            unknown.write_bytes(b"not a NumPy cache")
            self.assertFalse(visualizer._cache_file_is_safe(unknown))
            valid = root / "valid.npz"
            np.savez(valid, field=np.zeros((2, 2), dtype=np.float32))
            self.assertTrue(visualizer._cache_file_is_safe(valid))
            empty = root / "empty.npz"
            np.savez(empty)
            self.assertFalse(visualizer._cache_file_is_safe(empty))

            deceptive = root / "deceptive.npy"
            header = (
                "{'descr': '<f4', 'fortran_order': False, "
                "'shape': (100000000000,), }"
            ).encode("ascii")
            header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
            deceptive.write_bytes(
                b"\x93NUMPY\x01\x00"
                + struct.pack("<H", len(header))
                + header
            )
            self.assertFalse(visualizer._cache_file_is_safe(deceptive))

            symlink = root / "cache-link.npy"
            try:
                symlink.symlink_to(valid)
            except (OSError, NotImplementedError):
                pass
            else:
                self.assertFalse(visualizer._cache_file_is_safe(symlink))

    def test_native_formula_modes_produce_fields(self):
        if visualizer._get_native_library() is None:
            raise unittest.SkipTest("native renderer is unavailable")
        for formula in ("julia", "burning-ship", "tricorn"):
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

    def test_glitch_atlas_repairs_deep_tile_without_black_fill(self):
        """A compact reference glitch is refined instead of becoming a tile."""

        library = visualizer._get_native_library()
        if library is None or not hasattr(library, "fractal_render_points"):
            raise unittest.SkipTest("native deep reference renderer is unavailable")
        point = next(
            candidate for candidate in visualizer.DEEP_ZOOM_POINTS
            if candidate.slug == "oldwooddish"
        )
        width = height = 17
        max_iter = 4000
        library, reference = visualizer._create_native_reference(
            point.x,
            point.y,
            max_iter,
            60.0,
            3,
            60.0,
        )
        diagnostics: dict[str, int] = {}
        try:
            field = visualizer._atlas_local_reference_field(
                render_width=width,
                render_height=height,
                log10_zoom=60.0,
                x_center=point.x,
                y_center=point.y,
                max_iter=max_iter,
                series_order=3,
                series_block=256,
                renderer="native",
                native_threads=2,
                native_library=library,
                native_backend=0,
                native_reference=reference,
                allow_recovery=False,
                diagnostics=diagnostics,
            )
        finally:
            library.fractal_destroy_reference(reference)
        self.assertTrue(np.isfinite(field).all())
        self.assertGreater(diagnostics["initial_unresolved_pixels"], 0)
        self.assertGreater(
            diagnostics["secondary_references"]
            + diagnostics.get("exact_pixel_recoveries", 0),
            0,
        )
        self.assertGreater(float(np.ptp(field)), 10.0)
        self.assertTrue(np.any(field < max_iter - 0.5))

    def test_invalid_fast_deep_tile_retries_strict_glitch_repair(self):
        repaired = np.arange(16, dtype=np.float32).reshape(4, 4)
        with mock.patch.object(
            visualizer,
            "_atlas_local_reference_field",
            return_value=None,
        ), mock.patch.object(
            visualizer,
            "render_fractal",
            return_value=np.full((4, 4), np.nan, dtype=np.float32),
        ), mock.patch.object(
            visualizer,
            "_atlas_glitch_reference_field",
            return_value=repaired,
        ) as repair:
            field = visualizer._atlas_tile_field(
                cache_dir=None,
                cache_identity="test",
                render_width=4,
                render_height=4,
                level=0,
                log_zoom=15.0,
                x_center="-0.7",
                y_center="0.1",
                max_iter=128,
                series_order=3,
                series_block=256,
                renderer="auto",
                native_reference=object(),
                native_threads=1,
                native_library=object(),
            )
        np.testing.assert_array_equal(field, repaired)
        repair.assert_called_once()

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

    def test_atlas_compositor_promotes_a_nearly_full_child(self):
        """A reverse-boundary child must not leave a visible parent frame."""

        parent = np.zeros((8, 8), dtype=np.float32)
        child = np.ones((8, 8), dtype=np.float32)
        original = visualizer._colourise_view

        def fake_colour(view, *args):
            value = 20 if float(np.mean(view)) > 0.5 else 10
            return np.full((*view.shape, 3), value, dtype=np.uint8)

        visualizer._colourise_view = fake_colour
        try:
            result = visualizer._atlas_colour_frame(
                parent,
                child,
                64,
                64,
                7.9,
                0.9875,
                100,
                200,
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
        np.testing.assert_array_equal(
            result,
            np.full((64, 64, 3), 20, dtype=np.uint8),
        )

    def test_atlas_compositor_crops_an_overscanned_full_child(self):
        """A wider stored child keeps its true scale at takeover."""

        parent = np.zeros((8, 8), dtype=np.float32)
        child = np.ones((8, 8), dtype=np.float32)
        original_crop = visualizer._crop_and_resize_preserving_interior
        original_colour = visualizer._colourise_view
        observed_zooms = []

        def fake_crop(field, output_width, output_height, zoom, max_iter, resample):
            observed_zooms.append(float(zoom))
            return (
                np.full((output_height, output_width), float(np.mean(field)), dtype=np.float32),
                np.zeros((output_height, output_width), dtype=bool),
            )

        def fake_colour(view, *args):
            value = 20 if float(np.mean(view)) > 0.5 else 10
            return np.full((*view.shape, 3), value, dtype=np.uint8)

        visualizer._crop_and_resize_preserving_interior = fake_crop
        visualizer._colourise_view = fake_colour
        try:
            result = visualizer._atlas_colour_frame(
                parent,
                child,
                64,
                64,
                9.6,
                1.0,
                100,
                200,
                0.0,
                0.0,
                0.0,
                None,
                1,
                "bilinear",
                "aurora",
                0.5,
                child_zoom=1.2,
            )
        finally:
            visualizer._crop_and_resize_preserving_interior = original_crop
            visualizer._colourise_view = original_colour
        self.assertEqual(observed_zooms, [9.6, 1.2])
        np.testing.assert_array_equal(
            result,
            np.full((64, 64, 3), 20, dtype=np.uint8),
        )

    def test_atlas_compositor_normalizes_parent_and_child_interior_caps(self):
        """A deeper child cap must not turn the parent interior into a rectangle."""

        parent = np.full((8, 8), 100.0, dtype=np.float32)
        child = np.full((8, 8), 200.0, dtype=np.float32)
        original = visualizer._colourise_view

        def fake_colour(view, max_iter, *args):
            result = np.full((*view.shape, 3), 31, dtype=np.uint8)
            result[view >= float(max_iter) - 0.5] = 0
            return result

        visualizer._colourise_view = fake_colour
        try:
            result = visualizer._atlas_colour_frame(
                parent,
                child,
                64,
                64,
                1.0,
                0.5,
                100,
                200,
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
        np.testing.assert_array_equal(result, np.zeros((64, 64, 3), dtype=np.uint8))

    def test_atlas_compositor_prefers_deeper_child_interior(self):
        """A child cap must win when the coarse parent escaped the same pixel."""

        parent = np.full((8, 8), 10.0, dtype=np.float32)
        child = np.full((8, 8), 200.0, dtype=np.float32)
        original = visualizer._colourise_view

        def fake_colour(view, max_iter, *args):
            result = np.full((*view.shape, 3), 31, dtype=np.uint8)
            result[view >= float(max_iter) - 0.5] = 0
            return result

        visualizer._colourise_view = fake_colour
        try:
            result = visualizer._atlas_colour_frame(
                parent,
                child,
                64,
                64,
                1.0,
                0.5,
                100,
                200,
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
        self.assertEqual(tuple(result[32, 32]), (0, 0, 0))
        self.assertEqual(tuple(result[0, 0]), (31, 31, 31))

    def test_atlas_seam_feathers_before_an_interior_child(self):
        """A capped child must not expose a hard rectangular tile boundary."""

        parent = np.full((64, 64), 10.0, dtype=np.float32)
        child = np.full((64, 64), 200.0, dtype=np.float32)
        palette_file = (
            Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp"
        )
        library = visualizer._get_native_library()
        cases = [(None, None), (palette_file, None)]
        if library is not None:
            cases.extend(((None, library), (palette_file, library)))
        for selected_file, native_library in cases:
            result = visualizer._atlas_colour_frame(
                parent,
                child,
                64,
                64,
                1.0,
                0.5,
                100,
                200,
                0.0,
                0.5,
                0.5,
                native_library,
                1,
                "bilinear",
                "aurora",
                0.5,
                selected_file,
            )
            # The child starts at y=16 and the shared four-pixel feather ends
            # at y=20. The first two rows must retain parent colour; only the
            # fully-owned child interior is allowed to be the interior colour.
            self.assertNotEqual(tuple(result[16, 32]), (0, 0, 0))
            self.assertNotEqual(tuple(result[17, 32]), (0, 0, 0))
            self.assertEqual(tuple(result[20, 32]), (0, 0, 0))

    def test_custom_palettes_are_cached_and_finite(self):
        field = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        for name in ("fire", "ocean", "neon", "sunset", "mono", "kalles-default"):
            rgb = visualizer._colourise_custom(field, 100, 1.5, 0.8, 0.6, name)
            self.assertEqual(rgb.shape, (64, 64, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertTrue(np.isfinite(rgb).all())
        self.assertIs(visualizer._custom_palette("fire"), visualizer._custom_palette("fire"))

    def test_kfp_profile_preserves_transfer_and_slope_settings(self):
        bundled = Path(__file__).resolve().parents[1] / "palettes" / "kalles-default.kfp"
        profile = visualizer._kfp_profile_for_selection("aurora", bundled)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertAlmostEqual(profile.iter_div, 0.01)
        self.assertEqual(profile.color_method, 7)
        self.assertTrue(profile.slopes)
        self.assertEqual(profile.differences, 3)

        field = visualizer.render_fractal(
            24,
            18,
            0.0,
            "-1.0",
            "0.1",
            200,
            "python",
            formula="tricorn",
        )
        rgb = visualizer._colourise_custom(
            field, 200, 0.0, 0.5, 0.5, "aurora", palette_file=bundled
        )
        self.assertGreater(np.unique(rgb.reshape(-1, 3), axis=0).shape[0], 4)
        self.assertTrue(np.isfinite(rgb).all())

        # KFP owns its colours. Audio pitch/loudness controls must not turn
        # Kalles' imported default gradient into a different palette.
        baseline = visualizer._colourise_kfp(
            field, 200, 0.0, 0.0, 0.0, 0.5, profile
        )
        varied = visualizer._colourise_kfp(
            field, 200, 1.7, 1.0, 0.9, 0.0, profile
        )
        np.testing.assert_array_equal(varied, baseline)

    def test_ordinary_palettes_keep_aurora_detail_beyond_iteration_cap(self):
        field = np.asarray([[10.0, 11.0, 1000.0, 1001.0]], dtype=np.float32)
        rgb = visualizer._colourise_custom(field, 2000, 0.0, 0.8, 0.5, "fire")
        self.assertFalse(np.array_equal(rgb[0, 0], rgb[0, 2]))
        self.assertGreater(np.unique(rgb.reshape(-1, 3), axis=0).shape[0], 2)

    def test_non_aurora_palettes_reach_high_contrast_anchors(self):
        luminance_weights = np.asarray([0.2126, 0.7152, 0.0722])
        for name in visualizer.PALETTE_CHOICES:
            if name == "aurora":
                continue
            stops = np.asarray(visualizer.BUILTIN_PALETTE_STOPS[name], dtype=np.float32)
            luminance = stops @ luminance_weights
            self.assertLessEqual(float(np.min(luminance)), 35.0, name)
            self.assertGreaterEqual(float(np.max(luminance)), 245.0, name)
            palette = visualizer._custom_palette(name, len(stops))
            np.testing.assert_array_equal(palette[0], stops[0], err_msg=name)
            np.testing.assert_array_equal(palette[-1], stops[-1], err_msg=name)

    def test_kfp_software_video_keeps_chroma_detail(self):
        self.assertEqual(
            visualizer._video_pixel_format("libx264", "kalles-default"),
            "yuv444p",
        )
        self.assertEqual(
            visualizer._video_pixel_format("libx264", "aurora"),
            "yuv420p",
        )
        # Hardware H.264 inputs remain on their device-compatible format.
        self.assertEqual(
            visualizer._video_pixel_format("h264_nvenc", "kalles-default"),
            "yuv420p",
        )
        self.assertEqual(
            visualizer._video_pixel_format("h264_nvenc", "aurora", lossless=True),
            "yuv444p",
        )

    def test_lossless_encoder_settings_use_real_lossless_controls(self):
        self.assertEqual(
            visualizer._video_encoder_settings(
                "h264_nvenc",
                "ultrafast",
                18,
                lossless=True,
            )[1],
            ["-tune", "lossless", "-rc", "constqp", "-qp", "0"],
        )
        codec, preset, rate_control = visualizer._select_video_encoder(
            "libx264",
            "slow",
            18,
            lossless=True,
        )
        self.assertEqual((codec, preset, rate_control), ("libx264", "slow", ["-crf", "0"]))

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

    def test_hardware_probe_uses_nvenc_safe_dimensions(self):
        probe_result = mock.Mock(returncode=0)
        visualizer._hardware_encoder_usable.cache_clear()
        try:
            with mock.patch.object(
                visualizer.shutil, "which", return_value="/usr/bin/ffmpeg"
            ), mock.patch.object(
                visualizer.subprocess, "run", return_value=probe_result
            ) as run:
                self.assertTrue(visualizer._hardware_encoder_usable("h264_nvenc"))
        finally:
            visualizer._hardware_encoder_usable.cache_clear()
        command = run.call_args.args[0]
        self.assertIn("color=black:s=256x256:d=0.05", command)

    def test_lossless_hardware_probe_matches_nvenc_output_format(self):
        probe_result = mock.Mock(returncode=0)
        visualizer._hardware_encoder_usable.cache_clear()
        try:
            with mock.patch.object(
                visualizer.shutil, "which", return_value="/usr/bin/ffmpeg"
            ), mock.patch.object(
                visualizer.subprocess, "run", return_value=probe_result
            ) as run:
                self.assertTrue(visualizer._hardware_encoder_usable("h264_nvenc", True))
        finally:
            visualizer._hardware_encoder_usable.cache_clear()
        command = run.call_args.args[0]
        self.assertIn("-tune", command)
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv444p")
        self.assertGreater(command.index("-pix_fmt"), command.index("-i"))
        output_format_index = len(command) - 1 - command[::-1].index("-f")
        self.assertLess(command.index("-pix_fmt"), output_format_index)

    def test_lossless_vaapi_probe_uses_qp_zero(self):
        probe_result = mock.Mock(returncode=0)
        visualizer._vaapi_encoder_usable.cache_clear()
        try:
            with mock.patch.object(
                visualizer.shutil, "which", return_value="/usr/bin/ffmpeg"
            ), mock.patch.object(
                visualizer.Path, "exists", return_value=True
            ), mock.patch.object(
                visualizer.subprocess, "run", return_value=probe_result
            ) as run:
                self.assertTrue(
                    visualizer._vaapi_encoder_usable(lossless=True)
                )
        finally:
            visualizer._vaapi_encoder_usable.cache_clear()
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-qp") + 1], "0")

    def test_crop_factor_is_clamped_to_native_source(self):
        field = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        view = visualizer._crop_and_resize(field, 32, 32, 2.0, "bilinear")
        self.assertEqual(view.shape, (32, 32))
        self.assertTrue(np.isfinite(view).all())

    def test_crop_resampling_preserves_partial_interior_coverage(self):
        field = np.asarray(
            [[100.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        view, inside = visualizer._crop_and_resize_preserving_interior(
            field,
            2,
            2,
            1.0,
            100,
            "bilinear",
        )
        self.assertTrue(inside[0, 0])
        self.assertEqual(float(view[0, 0]), 100.0)
        self.assertFalse(inside[1, 1])

    def test_crop_resampling_does_not_leak_subhalf_interior_into_exterior(self):
        field = np.asarray(
            [[100.0, 10.0], [10.0, 10.0]],
            dtype=np.float32,
        )
        view, inside = visualizer._crop_and_resize_preserving_interior(
            field,
            8,
            8,
            1.0,
            100,
            "lanczos",
        )
        exterior = view[~inside]
        self.assertTrue(exterior.size > 0)
        self.assertLessEqual(float(np.max(exterior)), 20.0)

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
