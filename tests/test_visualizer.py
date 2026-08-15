import unittest

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"NumPy is unavailable: {error}") from error

import visualizer


class AnimationTests(unittest.TestCase):
    def test_zoom_plan_is_monotonic_and_reaches_end(self):
        instrumental = np.asarray([0.0, 0.1, 1.0, 0.0, 0.8, 0.0], dtype=np.float32)
        zooms = visualizer._zoom_plan(instrumental, "1", "1e12", punch=3.0)
        self.assertEqual(zooms.shape, instrumental.shape)
        self.assertTrue(np.all(np.diff(zooms) >= 0.0))
        self.assertAlmostEqual(float(zooms[0]), 0.0, places=12)
        self.assertAlmostEqual(float(zooms[-1]), 12.0, places=9)
        self.assertGreater(float(zooms[3] - zooms[2]), 0.0)

    def test_custom_palettes_are_cached_and_finite(self):
        field = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        for name in ("fire", "ocean", "neon", "sunset", "mono"):
            rgb = visualizer._colourise_custom(field, 100, 1.5, 0.8, 0.6, name)
            self.assertEqual(rgb.shape, (64, 64, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertTrue(np.isfinite(rgb).all())
        self.assertIs(visualizer._custom_palette("fire"), visualizer._custom_palette("fire"))

    def test_crop_factor_is_clamped_to_native_source(self):
        field = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        view = visualizer._crop_and_resize(field, 32, 32, 2.0, "bilinear")
        self.assertEqual(view.shape, (32, 32))
        self.assertTrue(np.isfinite(view).all())


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
