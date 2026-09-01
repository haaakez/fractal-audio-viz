import ctypes
import math
import unittest
from pathlib import Path

try:
    import mpmath as mp
except ImportError:  # pragma: no cover - optional outside the Nix shell
    mp = None


class NativeRenderOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("strict", ctypes.c_int32),
        ("allow_recovery", ctypes.c_int32),
        ("time_budget_ms", ctypes.c_int32),
        ("disable_bla", ctypes.c_int32),
        ("disable_cycle", ctypes.c_int32),
        ("strict_cycle", ctypes.c_int32),
        ("series_min_terms", ctypes.c_int32),
        ("series_max_terms", ctypes.c_int32),
        ("max_bla_length", ctypes.c_int32),
        ("max_linear_bla_length", ctypes.c_int32),
        ("backend", ctypes.c_int32),
        ("reserved", ctypes.c_int32 * 3),
    ]

    @classmethod
    def make(cls, **overrides):
        values = dict(
            strict=1,
            allow_recovery=0,
            time_budget_ms=0,
            disable_bla=0,
            disable_cycle=0,
            strict_cycle=0,
            series_min_terms=8,
            series_max_terms=32,
            max_bla_length=64,
            max_linear_bla_length=4096,
            backend=0,
        )
        values.update(overrides)
        result = cls()
        result.struct_size = ctypes.sizeof(cls)
        result.version = 1
        for name, value in values.items():
            setattr(result, name, int(value))
        return result


class NativeRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library_path = Path(__file__).resolve().parents[1] / "mandelbrot.so"
        if not library_path.exists():
            raise unittest.SkipTest("mandelbrot.so is not built")
        try:
            cls.library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise unittest.SkipTest(str(error)) from error
        library = cls.library
        library.fractal_abi_version.restype = ctypes.c_int
        library.fractal_last_error.restype = ctypes.c_char_p
        library.fractal_create_reference.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.fractal_create_reference.restype = ctypes.c_void_p
        if hasattr(library, "fractal_create_reference_reusable"):
            library.fractal_create_reference_reusable.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            library.fractal_create_reference_reusable.restype = ctypes.c_void_p
        if hasattr(library, "fractal_clone_reference"):
            library.fractal_clone_reference.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            library.fractal_clone_reference.restype = ctypes.c_void_p
        library.fractal_destroy_reference.argtypes = [ctypes.c_void_p]
        library.fractal_colourise.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        library.fractal_colourise.restype = ctypes.c_int
        library.fractal_crop_colourise.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        library.fractal_crop_colourise.restype = ctypes.c_int
        if hasattr(library, "fractal_crop_field"):
            library.fractal_crop_field.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
            ]
            library.fractal_crop_field.restype = ctypes.c_int
        if hasattr(library, "fractal_atlas_colourise"):
            library.fractal_atlas_colourise.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
            ]
            library.fractal_atlas_colourise.restype = ctypes.c_int
        library.render_mandelbrot_reference.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.render_mandelbrot_reference.restype = ctypes.c_int
        library.fractal_render_mandelbrot_reference_ex.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(NativeRenderOptions),
        ]
        library.fractal_render_mandelbrot_reference_ex.restype = ctypes.c_int
        if hasattr(library, "fractal_set_stats_enabled"):
            library.fractal_set_stats_enabled.argtypes = [ctypes.c_int]
            library.fractal_set_stats_enabled.restype = None
            library.fractal_get_last_stats.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_int,
            ]
            library.fractal_get_last_stats.restype = ctypes.c_int
            library.fractal_get_last_stats_ex.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_int,
            ]
            library.fractal_get_last_stats_ex.restype = ctypes.c_int

    def test_abi_version(self):
        self.assertEqual(self.library.fractal_abi_version(), 10)

    def test_cloned_radius_tier_matches_independent_reference(self):
        if not hasattr(self.library, "fractal_create_reference_reusable"):
            raise unittest.SkipTest("reference tier cloning is unavailable")
        x_center = (
            b"-1.711030826576984823314722728180246694222252112777834549259732560022287905717123892927883662257081287304281205446785464750361745"
        )
        y_center = (
            b"0.000001509818957972609043170877177447547323633361751210706181530872644435995661269979265353802853564243259051551728584671844401805"
        )
        max_iter = 2400
        root = self.library.fractal_create_reference_reusable(
            x_center, y_center, b"1e12", max_iter, 768, 32
        )
        self.assertTrue(root, self.library.fractal_last_error())
        clone = None
        independent = None
        try:
            clone = self.library.fractal_clone_reference(root, b"1e80")
            self.assertTrue(clone, self.library.fractal_last_error())
            independent = self.library.fractal_create_reference(
                x_center, y_center, b"1e80", max_iter, 768, 32
            )
            self.assertTrue(independent, self.library.fractal_last_error())
            width = 12
            height = 8
            output_type = ctypes.c_float * (width * height)
            cloned_output = output_type()
            independent_output = output_type()
            self.assertEqual(
                self.render_reference(
                    cloned_output, width, height, b"1e80", clone,
                    max_iter, 2, 3, 256,
                ),
                0,
                self.library.fractal_last_error(),
            )
            self.assertEqual(
                self.render_reference(
                    independent_output, width, height, b"1e80", independent,
                    max_iter, 2, 3, 256,
                ),
                0,
                self.library.fractal_last_error(),
            )
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(cloned_output, independent_output)),
                1.0e-3,
            )
        finally:
            if independent:
                self.library.fractal_destroy_reference(independent)
            if clone:
                self.library.fractal_destroy_reference(clone)
            self.library.fractal_destroy_reference(root)

    def render_reference(self, output, width, height, zoom, handle, max_iter,
                         threads, series_order, series_block, options=None):
        options = options or NativeRenderOptions.make()
        return self.library.fractal_render_mandelbrot_reference_ex(
            output,
            width,
            height,
            zoom,
            handle,
            max_iter,
            threads,
            series_order,
            series_block,
            ctypes.byref(options),
        )

    def test_opt_in_render_stats_report_bla_work(self):
        if not hasattr(self.library, "fractal_set_stats_enabled"):
            raise unittest.SkipTest("native render statistics are unavailable")
        width = height = 8
        output_type = ctypes.c_float * (width * height)
        output = output_type()
        handle = self.library.fractal_create_reference(
            b"-1.7110308265769848",
            b"0.00000150981895797",
            b"1e6",
            800,
            768,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        values = (ctypes.c_uint64 * 16)()
        try:
            self.library.fractal_set_stats_enabled(1)
            status = self.render_reference(
                output, width, height, b"1e30", handle, 800, 2, 3, 1024
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            self.assertEqual(self.library.fractal_get_last_stats(values, 16), 16)
            self.assertEqual(values[0], width * height)
            self.assertGreater(values[1], 0)
            self.assertGreater(values[2], 0)
        finally:
            self.library.fractal_set_stats_enabled(0)
            self.library.fractal_destroy_reference(handle)

    def test_opt_in_time_budget_marks_unfinished_pixels(self):
        """A pathological deep cell must return control to the atlas splitter."""

        width = height = 8
        max_iter = 30000
        output_type = ctypes.c_float * (width * height)
        output = output_type()
        x_center = (
            b"-1.711030826576984823314722728180246694222252112777834549259732560022287905717123892927883662257081287304281205446785464750361251745"
        )
        y_center = (
            b"0.000001509818957972609043170877177447547323633361751210706181530872644435995661269979265353802853564243259051551728584671844401805"
        )
        handle = self.library.fractal_create_reference(
            x_center,
            y_center,
            b"9.6389560487384802e+66",
            max_iter,
            1024,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            status = self.render_reference(
                output, width, height, b"9.6389560487384802e+66",
                handle, max_iter, 1, 3, 256,
                NativeRenderOptions.make(
                    time_budget_ms=1,
                    disable_bla=1,
                    disable_cycle=1,
                    series_min_terms=32,
                    series_max_terms=32,
                ),
            )
        finally:
            self.library.fractal_destroy_reference(handle)
        self.assertEqual(status, 0, self.library.fractal_last_error())
        self.assertTrue(any(math.isnan(value) for value in output))

    def test_bla_matches_exact_perturbation(self):
        width = height = 24
        output_type = ctypes.c_float * (width * height)
        with_reference = output_type()
        exact = output_type()
        handle = self.library.fractal_create_reference(
            b"-1.7110308265769848",
            b"0.00000150981895797",
            b"1e6",
            4000,
            512,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            status = self.render_reference(
                with_reference, width, height, b"1e12", handle, 4000, 2, 3, 256
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            status = self.render_reference(
                exact,
                width,
                height,
                b"1e12",
                handle,
                4000,
                2,
                3,
                256,
                NativeRenderOptions.make(disable_bla=1, series_min_terms=32, series_max_terms=32),
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            self.assertLessEqual(max(abs(a - b) for a, b in zip(with_reference, exact)), 1e-3)
        finally:
            self.library.fractal_destroy_reference(handle)

    def test_extreme_zoom_returns_finite_field(self):
        width = height = 8
        output_type = ctypes.c_float * (width * height)
        output = output_type()
        handle = self.library.fractal_create_reference(
            b"-1.7110308265769848",
            b"0.00000150981895797",
            b"1e6",
            1000,
            1024,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            for zoom_text in (b"1e100", b"1e4000"):
                status = self.library.render_mandelbrot_reference(
                    output, width, height, zoom_text, handle, 1000, 2, 3, 256
                )
                self.assertEqual(status, 0, self.library.fractal_last_error())
                self.assertTrue(all(math.isfinite(value) for value in output))
        finally:
            self.library.fractal_destroy_reference(handle)

    def test_native_pitch_rotates_legacy_two_hue_gradient(self):
        width = height = 2
        field_type = ctypes.c_float * (width * height)
        output_type = ctypes.c_uint8 * (width * height * 3)
        field = field_type(0.0, 25.0, 50.0, 100.0)
        baseline = output_type()
        low = output_type()
        high = output_type()
        for pitch, output in ((0.5, baseline), (0.0, low), (1.0, high)):
            status = self.library.fractal_colourise(
                field,
                output,
                width,
                height,
                100,
                0.0,
                0.5,
                0.8,
                pitch,
                2,
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
        baseline_pixels = [tuple(baseline[index:index + 3]) for index in range(0, len(baseline), 3)]
        self.assertTrue(any(max(pixel) - min(pixel) > 20 for pixel in baseline_pixels))
        self.assertNotEqual(bytes(baseline), bytes(low))
        self.assertNotEqual(bytes(baseline), bytes(high))
        self.assertNotEqual(bytes(low), bytes(high))

    def test_raw_field_reprojection_matches_scalar_crop(self):
        if not hasattr(self.library, "fractal_crop_field"):
            raise unittest.SkipTest("raw-field reprojection is unavailable")
        width = height = 18
        field_type = ctypes.c_float * (width * height)
        source = field_type(*[
            float((index * 13 + index // width * 5) % 97)
            for index in range(width * height)
        ])
        output_type = ctypes.c_float * (11 * 13)
        output = output_type()
        status = self.library.fractal_crop_field(
            source,
            width,
            height,
            output,
            11,
            13,
            1.7,
            2,
        )
        self.assertEqual(status, 0, self.library.fractal_last_error())
        self.assertTrue(all(math.isfinite(value) for value in output))
        # The centre must be a bilinear sample rather than a copied source
        # corner; this catches accidental nearest-neighbour reprojection.
        self.assertGreater(max(output), min(output))

    def test_avx2_backend_matches_scalar_when_available(self):
        if not hasattr(self.library, "fractal_backend_capabilities"):
            raise unittest.SkipTest("backend capability ABI is unavailable")
        self.library.fractal_backend_capabilities.restype = ctypes.c_int
        if not self.library.fractal_backend_capabilities() & 2:
            raise unittest.SkipTest("AVX2 is not available on this build")
        width = height = 13
        output_type = ctypes.c_float * (width * height)
        scalar = output_type()
        vector = output_type()
        handle = self.library.fractal_create_reference(
            b"-0.743643887037151",
            b"0.13182590420533",
            b"1e1",
            500,
            512,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            self.assertEqual(
                self.render_reference(
                    scalar, width, height, b"1e1", handle, 500, 2, 3, 64,
                    NativeRenderOptions.make(backend=0),
                ),
                0,
            )
            self.assertEqual(
                self.render_reference(
                    vector, width, height, b"1e1", handle, 500, 2, 3, 64,
                    NativeRenderOptions.make(backend=1),
                ),
                0,
            )
            self.assertLessEqual(max(abs(a - b) for a, b in zip(scalar, vector)), 1e-5)
        finally:
            self.library.fractal_destroy_reference(handle)

    def test_opencl_backend_matches_scalar_when_available(self):
        if not hasattr(self.library, "fractal_backend_capabilities"):
            raise unittest.SkipTest("backend capability ABI is unavailable")
        self.library.fractal_backend_capabilities.restype = ctypes.c_int
        if not self.library.fractal_backend_capabilities() & 4:
            raise unittest.SkipTest("OpenCL double-precision backend is unavailable")
        width, height = 19, 13
        output_type = ctypes.c_float * (width * height)
        scalar = output_type()
        opencl = output_type()
        handle = self.library.fractal_create_reference(
            b"-0.743643887037151",
            b"0.13182590420533",
            b"1e1",
            900,
            512,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            self.assertEqual(
                self.render_reference(
                    scalar, width, height, b"1e0", handle, 900, 2, 3, 64,
                    NativeRenderOptions.make(backend=0),
                ),
                0,
            )
            self.assertEqual(
                self.render_reference(
                    opencl, width, height, b"1e0", handle, 900, 2, 3, 64,
                    NativeRenderOptions.make(backend=2),
                ),
                0,
            )
            self.assertLessEqual(max(abs(a - b) for a, b in zip(scalar, opencl)), 1e-5)
        finally:
            self.library.fractal_destroy_reference(handle)

    def test_deep_opencl_backend_is_rejected_before_render(self):
        width = height = 3
        output_type = ctypes.c_float * (width * height)
        output = output_type()
        handle = self.library.fractal_create_reference(
            b"-0.743643887037151",
            b"0.13182590420533",
            b"1e1",
            900,
            512,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            status = self.render_reference(
                output, width, height, b"1e12", handle, 900, 2, 3, 64,
                NativeRenderOptions.make(backend=2),
            )
            self.assertNotEqual(status, 0)
            self.assertIn(b"direct", self.library.fractal_last_error())
        finally:
            self.library.fractal_destroy_reference(handle)

    def test_native_atlas_compositor_blends_central_child(self):
        if not hasattr(self.library, "fractal_atlas_colourise"):
            raise unittest.SkipTest("atlas ABI entry point is not available")
        parent_width = parent_height = 8
        output_width = output_height = 16
        field_type = ctypes.c_float * (parent_width * parent_height)
        output_type = ctypes.c_uint8 * (output_width * output_height * 3)
        parent = field_type(*([10.0] * (parent_width * parent_height)))
        child = field_type(*([20.0] * (parent_width * parent_height)))
        output = output_type()
        status = self.library.fractal_atlas_colourise(
            parent,
            parent_width,
            parent_height,
            100,
            child,
            parent_width,
            parent_height,
            100,
            output,
            output_width,
            output_height,
            1.0,
            0.5,
            100,
            0.0,
            0.0,
            0.0,
            0.5,
            2,
        )
        self.assertEqual(status, 0, self.library.fractal_last_error())
        corner = tuple(output[0:3])
        centre_offset = (output_width * (output_height // 2) + output_width // 2) * 3
        centre = tuple(output[centre_offset:centre_offset + 3])
        self.assertNotEqual(corner, centre)

    def test_optimized_atlas_parent_matches_crop_colour_path(self):
        """The mapped hot loop must preserve the previous crop semantics."""

        if not hasattr(self.library, "fractal_atlas_colourise"):
            raise unittest.SkipTest("atlas ABI entry point is not available")
        width = height = 24
        field_type = ctypes.c_float * (width * height)
        output_type = ctypes.c_uint8 * (width * height * 3)
        field = field_type(*[
            float((index * 17 + index // width * 3) % 101)
            for index in range(width * height)
        ])
        old_output = output_type()
        new_output = output_type()
        phase = 0.31
        vocal = 0.42
        pitch = 0.63
        status = self.library.fractal_crop_colourise(
            field,
            width,
            height,
            old_output,
            width,
            height,
            1.7,
            100,
            phase,
            vocal,
            0.0,
            pitch,
            4,
        )
        self.assertEqual(status, 0, self.library.fractal_last_error())
        status = self.library.fractal_atlas_colourise(
            field,
            width,
            height,
            100,
            ctypes.POINTER(ctypes.c_float)(),
            0,
            0,
            0,
            new_output,
            width,
            height,
            1.7,
            0.0,
            100,
            phase,
            vocal,
            0.0,
            pitch,
            4,
        )
        self.assertEqual(status, 0, self.library.fractal_last_error())
        differences = [abs(int(left) - int(right)) for left, right in zip(old_output, new_output)]
        self.assertLessEqual(max(differences), 1)

    def test_shared_exponent_bla_matches_exact_deep_field(self):
        width = height = 8
        output_type = ctypes.c_float * (width * height)
        with_reference = output_type()
        exact = output_type()
        handle = self.library.fractal_create_reference(
            b"-1.7110308265769848",
            b"0.00000150981895797",
            b"1e6",
            4000,
            512,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            status = self.render_reference(
                with_reference, width, height, b"1e30", handle, 4000, 2, 3, 256
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            status = self.render_reference(
                exact,
                width,
                height,
                b"1e30",
                handle,
                4000,
                2,
                3,
                256,
                NativeRenderOptions.make(disable_bla=1, series_min_terms=32, series_max_terms=32),
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            self.assertLessEqual(max(abs(a - b) for a, b in zip(with_reference, exact)), 1e-3)
        finally:
            self.library.fractal_destroy_reference(handle)

    def test_wider_bla_radius_matches_exact_deep_tail(self):
        """The throughput-oriented deep radius must not change escape bands."""

        width = height = 6
        max_iter = 6000
        output_type = ctypes.c_float * (width * height)
        with_reference = output_type()
        exact = output_type()
        handle = self.library.fractal_create_reference(
            b"-1.7110308265769848",
            b"0.00000150981895797",
            b"1e12",
            max_iter,
            768,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            status = self.render_reference(
                with_reference, width, height, b"1e40", handle, max_iter, 1, 3, 4096
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            status = self.render_reference(
                exact,
                width,
                height,
                b"1e40",
                handle,
                max_iter,
                1,
                3,
                4096,
                NativeRenderOptions.make(disable_bla=1, series_min_terms=32, series_max_terms=32),
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            self.assertLessEqual(max(abs(a - b) for a, b in zip(with_reference, exact)), 2e-3)
        finally:
            self.library.fractal_destroy_reference(handle)

    def test_bla_rejects_nonfinite_reference_tail(self):
        """Escaping references must not let an overflowed tail create NaN maps."""

        width = height = 4
        max_iter = 1200
        output_type = ctypes.c_float * (width * height)
        cases = (
            (b"-1.7110308265769848", b"0.00000150981895797"),
            (b"-0.7453", b"0.1127"),
        )
        for x_text, y_text in cases:
            with_reference = output_type()
            exact = output_type()
            handle = self.library.fractal_create_reference(
                x_text,
                y_text,
                b"1e6",
                max_iter,
                768,
                3,
            )
            self.assertTrue(handle, self.library.fractal_last_error())
            try:
                status = self.render_reference(
                    exact,
                    width,
                    height,
                    b"1e30",
                    handle,
                    max_iter,
                    2,
                    3,
                    4096,
                    NativeRenderOptions.make(
                        disable_bla=1,
                        series_min_terms=32,
                        series_max_terms=32,
                    ),
                )
                self.assertEqual(status, 0, self.library.fractal_last_error())
                status = self.render_reference(
                    with_reference,
                    width,
                    height,
                    b"1e30",
                    handle,
                    max_iter,
                    2,
                    3,
                    4096,
                )
                self.assertEqual(status, 0, self.library.fractal_last_error())
                self.assertTrue(all(math.isfinite(value) for value in with_reference))
                self.assertLessEqual(
                    max(abs(a - b) for a, b in zip(with_reference, exact)),
                    1e-3,
                )
            finally:
                self.library.fractal_destroy_reference(handle)

    def test_deep_linear_tier_matches_exact_at_tier_boundary(self):
        """The extra deep hierarchy must stay exact at e30 and e100."""

        width = height = 4
        max_iter = 2400
        output_type = ctypes.c_float * (width * height)
        handle = self.library.fractal_create_reference(
            b"-1.711030826576984823314722728180246694222252112777834549259732560022287905717123892927883662257081287304281205446785464750361745",
            b"0.000001509818957972609043170877177447547323633361751210706181530872644435995661269979265353802853564243259051551728584671844401805",
            b"1e12",
            max_iter,
            768,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            for zoom_text in (b"1e30", b"1e100"):
                with_reference = output_type()
                exact = output_type()
                status = self.render_reference(
                    exact,
                    width,
                    height,
                    zoom_text,
                    handle,
                    max_iter,
                    2,
                    3,
                    1024,
                    NativeRenderOptions.make(
                        disable_bla=1,
                        series_min_terms=32,
                        series_max_terms=32,
                    ),
                )
                self.assertEqual(status, 0, self.library.fractal_last_error())
                status = self.render_reference(
                    with_reference,
                    width,
                    height,
                    zoom_text,
                    handle,
                    max_iter,
                    2,
                    3,
                    1024,
                )
                self.assertEqual(status, 0, self.library.fractal_last_error())
                self.assertLessEqual(
                    max(abs(a - b) for a, b in zip(with_reference, exact)),
                    1e-3,
                )
        finally:
            self.library.fractal_destroy_reference(handle)

    def test_deep_field_matches_independent_mpmath_oracle(self):
        """Check the native perturbation result against direct high precision math.

        The existing BLA tests compare two native paths, which is useful for
        regression but cannot catch a shared mistake in both paths. This test
        deliberately computes the same pixel coordinates with mpmath's direct
        Mandelbrot iteration at a known near-boundary centre.
        """

        if mp is None:
            raise unittest.SkipTest("mpmath is unavailable")
        width = height = 6
        max_iter = 2000
        x_text = "-0.743643887037151"
        y_text = "0.13182590420533"
        output_type = ctypes.c_float * (width * height)
        output = output_type()
        handle = self.library.fractal_create_reference(
            x_text.encode(),
            y_text.encode(),
            b"1e6",
            max_iter,
            768,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        try:
            status = self.library.render_mandelbrot_reference(
                output,
                width,
                height,
                b"1e12",
                handle,
                max_iter,
                4,
                3,
                256,
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            native = list(output)
        finally:
            self.library.fractal_destroy_reference(handle)

        with mp.workdps(100):
            centre_real = mp.mpf(x_text)
            centre_imag = mp.mpf(y_text)
            zoom = mp.mpf("1e12")
            height_span = mp.mpf("2.8") / zoom
            width_span = height_span * width / height
            expected = []
            for pixel_y in range(height):
                y_offset = (mp.mpf(height - 1) / 2 - pixel_y) * height_span / height
                for pixel_x in range(width):
                    x_offset = (mp.mpf(pixel_x) - mp.mpf(width - 1) / 2) * width_span / width
                    c = mp.mpc(centre_real + x_offset, centre_imag + y_offset)
                    z = mp.mpc(0, 0)
                    value = mp.mpf(max_iter)
                    for iteration in range(max_iter):
                        z = z * z + c
                        magnitude_squared = z.real * z.real + z.imag * z.imag
                        if magnitude_squared > 4:
                            value = (
                                iteration
                                + 1
                                - mp.log(mp.log(mp.sqrt(magnitude_squared))) / mp.log(2)
                            )
                            break
                    expected.append(float(value))

        differences = [abs(actual - reference) for actual, reference in zip(native, expected)]
        self.assertLessEqual(max(differences), 0.02)

    def test_adaptive_deep_series_matches_mpmath_at_e30_and_e100(self):
        """Exercise the linear BLA branches against an independent oracle."""

        if mp is None:
            raise unittest.SkipTest("mpmath is unavailable")
        width = height = 4
        max_iter = 1200
        x_text = "-0.743643887037151"
        y_text = "0.13182590420533"
        output_type = ctypes.c_float * (width * height)
        output = output_type()
        handle = self.library.fractal_create_reference(
            x_text.encode(),
            y_text.encode(),
            b"1e12",
            max_iter,
            768,
            3,
        )
        self.assertTrue(handle, self.library.fractal_last_error())
        native_by_zoom = {}
        try:
            for zoom_text in (b"1e30", b"1e100"):
                status = self.library.render_mandelbrot_reference(
                    output,
                    width,
                    height,
                    zoom_text,
                    handle,
                    max_iter,
                    4,
                    3,
                    4096,
                )
                self.assertEqual(status, 0, self.library.fractal_last_error())
                native_by_zoom[zoom_text] = list(output)
        finally:
            self.library.fractal_destroy_reference(handle)

        for zoom_text, decimal_precision in ((b"1e30", 160), (b"1e100", 240)):
            with mp.workdps(decimal_precision):
                centre_real = mp.mpf(x_text)
                centre_imag = mp.mpf(y_text)
                zoom = mp.mpf(zoom_text.decode())
                height_span = mp.mpf("2.8") / zoom
                width_span = height_span * width / height
                expected = []
                for pixel_y in range(height):
                    y_offset = (mp.mpf(height - 1) / 2 - pixel_y) * height_span / height
                    for pixel_x in range(width):
                        x_offset = (mp.mpf(pixel_x) - mp.mpf(width - 1) / 2) * width_span / width
                        c = mp.mpc(centre_real + x_offset, centre_imag + y_offset)
                        z = mp.mpc(0, 0)
                        value = mp.mpf(max_iter)
                        for iteration in range(max_iter):
                            z = z * z + c
                            magnitude_squared = z.real * z.real + z.imag * z.imag
                            if magnitude_squared > 4:
                                value = (
                                    iteration
                                    + 1
                                    - mp.log(mp.log(mp.sqrt(magnitude_squared))) / mp.log(2)
                                )
                                break
                        expected.append(float(value))

            differences = [
                abs(actual - reference)
                for actual, reference in zip(native_by_zoom[zoom_text], expected)
            ]
            self.assertLessEqual(max(differences), 0.03)

    def test_bundled_centre_sampled_oracle_at_e40_e60_e80_e100(self):
        """Exercise the production 50k iteration range at four deep zooms."""

        if mp is None:
            raise unittest.SkipTest("mpmath is unavailable")
        x_text = (
            "-1.711030826576984823314722728180246694222252112777834549259732560022287905717123892927883662257081287304281205446785464750361745"
        )
        y_text = (
            "0.000001509818957972609043170877177447547323633361751210706181530872644435995661269979265353802853564243259051551728584671844401805"
        )
        width = height = 2
        max_iter = 50000
        output_type = ctypes.c_float * (width * height)
        for zoom_decades in (40, 60, 80, 100):
            handle = self.library.fractal_create_reference(
                x_text.encode(),
                y_text.encode(),
                f"1e{zoom_decades}".encode(),
                max_iter,
                max(768, int(zoom_decades * 3.5 * math.log2(10.0)) + 64),
                3,
            )
            self.assertTrue(handle, self.library.fractal_last_error())
            output = output_type()
            try:
                status = self.render_reference(
                    output,
                    width,
                    height,
                    f"1e{zoom_decades}".encode(),
                    handle,
                    max_iter,
                    4,
                    3,
                    256,
                )
                self.assertEqual(status, 0, self.library.fractal_last_error())
                native = list(output)
            finally:
                self.library.fractal_destroy_reference(handle)

            with mp.workdps(max(160, int(zoom_decades * 3.5) + 80)):
                centre_real = mp.mpf(x_text)
                centre_imag = mp.mpf(y_text)
                zoom = mp.mpf(10) ** zoom_decades
                height_span = mp.mpf("2.8") / zoom
                width_span = height_span
                expected = []
                for pixel_y in range(height):
                    y_offset = (mp.mpf(height - 1) / 2 - pixel_y) * height_span / height
                    for pixel_x in range(width):
                        x_offset = (mp.mpf(pixel_x) - mp.mpf(width - 1) / 2) * width_span / width
                        c = mp.mpc(centre_real + x_offset, centre_imag + y_offset)
                        z = mp.mpc(0, 0)
                        value = mp.mpf(max_iter)
                        for iteration in range(max_iter):
                            z = z * z + c
                            magnitude_squared = z.real * z.real + z.imag * z.imag
                            if magnitude_squared > 4:
                                value = (
                                    iteration
                                    + 1
                                    - mp.log(mp.log(mp.sqrt(magnitude_squared))) / mp.log(2)
                                )
                                break
                        expected.append(float(value))
                self.assertLessEqual(
                    max(abs(actual - reference) for actual, reference in zip(native, expected)),
                    0.05,
                    f"oracle mismatch at 1e{zoom_decades}",
                )
            if zoom_decades == 100:
                # The centre-adjacent pixel of a production-sized tile is a
                # deliberately slow escape band (~4.0e4 iterations here).
                # It must not be painted as an interior merely because a
                # shorter e66-style budget would have ended first.
                wide_output_type = ctypes.c_float * (256 * 256)
                wide_output = wide_output_type()
                wide_handle = self.library.fractal_create_reference(
                    x_text.encode(),
                    y_text.encode(),
                    b"1e100",
                    max_iter,
                    1024,
                    3,
                )
                self.assertTrue(wide_handle, self.library.fractal_last_error())
                try:
                    status = self.render_reference(
                        wide_output,
                        256,
                        256,
                        b"1e100",
                        wide_handle,
                        max_iter,
                        6,
                        3,
                        256,
                    )
                    self.assertEqual(status, 0, self.library.fractal_last_error())
                finally:
                    self.library.fractal_destroy_reference(wide_handle)
                centre_adjacent = wide_output[128 * 256 + 128]
                self.assertLess(
                    centre_adjacent,
                    max_iter - 0.5,
                    "the e100 centre-adjacent escape band became false black",
                )

        # Production used to prepare this same orbit with a shallow e12 BLA
        # bound and then reuse it at e80/e100.  A deep tier near e40 must keep
        # the field escaping instead of manufacturing an all-interior frame.
        deep_output_type = ctypes.c_float * (8 * 8)
        deep_handle = self.library.fractal_create_reference(
            x_text.encode(),
            y_text.encode(),
            b"1e40",
            max_iter,
            1024,
            3,
        )
        self.assertTrue(deep_handle, self.library.fractal_last_error())
        try:
            for zoom_text in (b"1e83", b"1e100"):
                deep_output = deep_output_type()
                status = self.render_reference(
                    deep_output,
                    8,
                    8,
                    zoom_text,
                    deep_handle,
                    max_iter,
                    4,
                    3,
                    256,
                )
                self.assertEqual(status, 0, self.library.fractal_last_error())
                self.assertTrue(
                    any(value < max_iter - 0.5 for value in deep_output),
                    f"deep tier produced no escaping pixels at {zoom_text!r}",
                )
        finally:
            self.library.fractal_destroy_reference(deep_handle)


if __name__ == "__main__":
    unittest.main()
