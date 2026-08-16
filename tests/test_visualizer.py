import unittest
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"NumPy is unavailable: {error}") from error

import visualizer


class AnimationTests(unittest.TestCase):
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
        self.assertEqual(visualizer._pitch_hue_angle(0.5), 0.0)
        self.assertLess(visualizer._pitch_hue_angle(0.0), 0.0)
        self.assertGreater(visualizer._pitch_hue_angle(1.0), 0.0)
        self.assertFalse(np.array_equal(low, high))
        self.assertFalse(np.array_equal(baseline, low))
        self.assertFalse(np.array_equal(baseline, high))

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
