import unittest

import gui


class GuiHelperTests(unittest.TestCase):
    def test_progress_helper_accepts_percent_lines(self):
        self.assertEqual(gui._progress_from_output_line("  encoded  25.0%"), (0.25, "25.0%"))
        self.assertEqual(gui._progress_from_output_line("encoded 100%"), (1.0, "100.0%"))

    def test_progress_helper_accepts_frame_counts(self):
        fraction, label = gui._progress_from_output_line("frame 12/100")
        self.assertAlmostEqual(fraction, 0.12)
        self.assertEqual(label, "Frame 12 / 100 (12.0%)")

    def test_progress_helper_ignores_unrelated_diagnostics(self):
        self.assertEqual(gui._progress_from_output_line("Entering atlas level 3/10"), (None, None))
        self.assertEqual(gui._progress_from_output_line("encoder selected: libx264"), (None, None))


if __name__ == "__main__":
    unittest.main()
