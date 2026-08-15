import ctypes
import math
import os
import unittest
from pathlib import Path


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
        self.assertEqual(self.library.fractal_abi_version(), 6)

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
            status = self.library.render_mandelbrot_reference(
                output, width, height, b"1e100", handle, 1000, 2, 3, 256
            )
            self.assertEqual(status, 0, self.library.fractal_last_error())
            self.assertTrue(all(math.isfinite(value) for value in output))
        finally:
            self.library.fractal_destroy_reference(handle)


if __name__ == "__main__":
    unittest.main()
