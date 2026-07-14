import unittest

import numpy as np

from heartbeat_preprocessor.core import normalize_for_wav, peak_from_dbfs


class OutputLevelTest(unittest.TestCase):
    def test_target_peak_controls_export_level(self) -> None:
        source = np.asarray([-0.2, 0.5, -0.8], dtype=np.float32)
        target_peak = peak_from_dbfs(-6.0)
        exported = normalize_for_wav(source, target_peak=target_peak)

        self.assertAlmostEqual(float(np.max(np.abs(exported))), target_peak, places=6)
        self.assertAlmostEqual(target_peak, 10.0 ** (-6.0 / 20.0), places=6)

    def test_peak_target_keeps_headroom_when_requested_level_is_too_high(self) -> None:
        self.assertAlmostEqual(peak_from_dbfs(6.0), peak_from_dbfs(-0.1), places=6)
        self.assertLess(peak_from_dbfs(-0.1), 1.0)


if __name__ == "__main__":
    unittest.main()
