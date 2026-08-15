import ctypes
import math
import os
import unittest
from pathlib import Path

try:
    import mpmath as mp
except ImportError:  # pragma: no cover - optional outside the Nix shell
    mp = None


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

    def test_abi_version(self):
        self.assertEqual(self.library.fractal_abi_version(), 8)

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
            status = self.library.render_mandelbrot_reference(
                with_reference, width, height, b"1e12", handle, 4000, 2, 3, 256
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            old_value = os.environ.get("FRACTAL_DISABLE_BLA")
            os.environ["FRACTAL_DISABLE_BLA"] = "1"
            try:
                status = self.library.render_mandelbrot_reference(
                    exact, width, height, b"1e12", handle, 4000, 2, 3, 256
                )
            finally:
                if old_value is None:
                    os.environ.pop("FRACTAL_DISABLE_BLA", None)
                else:
                    os.environ["FRACTAL_DISABLE_BLA"] = old_value
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

    def test_native_pitch_colour_is_neutral_at_average(self):
        width = height = 2
        field_type = ctypes.c_float * (width * height)
        output_type = ctypes.c_uint8 * (width * height * 3)
        field = field_type(0.0, 25.0, 50.0, 100.0)
        neutral = output_type()
        low = output_type()
        high = output_type()
        for pitch, output in ((0.5, neutral), (0.0, low), (1.0, high)):
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
        self.assertLessEqual(abs(int(neutral[0]) - int(neutral[1])), 1)
        self.assertLessEqual(abs(int(neutral[1]) - int(neutral[2])), 1)
        self.assertNotEqual(bytes(low), bytes(high))

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
            status = self.library.render_mandelbrot_reference(
                with_reference, width, height, b"1e30", handle, 4000, 2, 3, 256
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            old_value = os.environ.get("FRACTAL_DISABLE_BLA")
            os.environ["FRACTAL_DISABLE_BLA"] = "1"
            try:
                status = self.library.render_mandelbrot_reference(
                    exact, width, height, b"1e30", handle, 4000, 2, 3, 256
                )
            finally:
                if old_value is None:
                    os.environ.pop("FRACTAL_DISABLE_BLA", None)
                else:
                    os.environ["FRACTAL_DISABLE_BLA"] = old_value
            self.assertEqual(status, 0, self.library.fractal_last_error())
            self.assertLessEqual(max(abs(a - b) for a, b in zip(with_reference, exact)), 1e-3)
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

    def test_adaptive_deep_series_matches_mpmath_at_e30(self):
        """Exercise the cheap linear BLA branch at a genuinely deep view."""

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
        try:
            status = self.library.render_mandelbrot_reference(
                output,
                width,
                height,
                b"1e30",
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

        with mp.workdps(160):
            centre_real = mp.mpf(x_text)
            centre_imag = mp.mpf(y_text)
            zoom = mp.mpf("1e30")
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
        self.assertLessEqual(max(differences), 0.03)


if __name__ == "__main__":
    unittest.main()
