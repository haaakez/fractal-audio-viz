import unittest

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"NumPy is unavailable: {error}") from error

import visualizer


class AnimationTests(unittest.TestCase):
    def test_zoom_plan_has_default_pullback_and_reaches_end(self):
        instrumental = np.asarray([0.0, 0.1, 1.0, 0.0, 0.8, 0.0], dtype=np.float32)
        zooms = visualizer._zoom_plan(instrumental, "1", "1e12", punch=3.0)
        self.assertEqual(zooms.shape, instrumental.shape)
        self.assertAlmostEqual(float(zooms[0]), 0.0, places=12)
        self.assertAlmostEqual(float(zooms[-1]), 12.0, places=9)
        self.assertLess(float(np.min(zooms)), 0.0)
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

    def test_custom_palettes_are_cached_and_finite(self):
        field = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        for name in ("fire", "ocean", "neon", "sunset", "mono"):
            rgb = visualizer._colourise_custom(field, 100, 1.5, 0.8, 0.6, name)
            self.assertEqual(rgb.shape, (64, 64, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertTrue(np.isfinite(rgb).all())
        self.assertIs(visualizer._custom_palette("fire"), visualizer._custom_palette("fire"))

    def test_pitch_is_grey_at_average_and_moves_around_the_hue_wheel(self):
        field = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        neutral = visualizer._colourise(field, 100, 0.0, 0.5, 0.8, 0.5)
        low = visualizer._colourise(field, 100, 0.0, 0.5, 0.8, 0.0)
        high = visualizer._colourise(field, 100, 0.0, 0.5, 0.8, 1.0)
        self.assertLessEqual(
            int(np.max(np.abs(neutral[..., 0].astype(np.int16) - neutral[..., 1].astype(np.int16)))),
            1,
        )
        self.assertLessEqual(
            int(np.max(np.abs(neutral[..., 1].astype(np.int16) - neutral[..., 2].astype(np.int16)))),
            1,
        )
        self.assertFalse(np.array_equal(low, high))
        low_hue, low_saturation = visualizer._pitch_colour_controls(0.0)
        high_hue, high_saturation = visualizer._pitch_colour_controls(1.0)
        self.assertGreater(low_saturation, 0.7)
        self.assertGreater(high_saturation, 0.7)
        self.assertNotAlmostEqual(low_hue, high_hue, places=3)

    def test_crop_factor_is_clamped_to_native_source(self):
        field = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        view = visualizer._crop_and_resize(field, 32, 32, 2.0, "bilinear")
        self.assertEqual(view.shape, (32, 32))
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
