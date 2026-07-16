import io
import unittest

import numpy as np
from scipy.io import wavfile

from heartbeat_preprocessor.core import (
    ProcessingParams,
    confirm_beats_with_template,
    estimate_bpm_with_consensus,
    process_audio_bytes,
)


def pulse_envelope(sample_rate: int, seconds: float, beat_times: np.ndarray) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    envelope = np.full_like(time, 0.03)
    for beat in beat_times:
        envelope += np.exp(-0.5 * ((time - beat) / 0.045) ** 2).astype(np.float32)
    return envelope


class RobustAnalysisTest(unittest.TestCase):
    def test_multi_window_consensus_recovers_periodic_bpm(self) -> None:
        sample_rate = 100
        beat_times = np.arange(0.8, 23.5, 0.8, dtype=np.float32)
        envelope = pulse_envelope(sample_rate, 24.0, beat_times)
        params = ProcessingParams(
            analysis_window_seconds=6.0,
            analysis_window_hop_seconds=3.0,
            peak_prominence=0.1,
        )
        estimate, windows = estimate_bpm_with_consensus(envelope, sample_rate, params)
        self.assertEqual(estimate["method"], "multi_window_median_consensus")
        self.assertAlmostEqual(estimate["estimated_bpm"], 75.0, delta=2.0)
        self.assertGreaterEqual(estimate["consensus_window_count"], 2)
        self.assertGreaterEqual(len(windows), 2)

    def test_template_confirmation_preserves_plausible_beat_count(self) -> None:
        sample_rate = 100
        beats = np.asarray([0.8, 1.8, 2.8, 3.8, 4.8], dtype=np.float32)
        signal_data = pulse_envelope(sample_rate, 6.0, beats)
        confirmed, info, analysis, template = confirm_beats_with_template(
            signal_data,
            beats,
            sample_rate,
            expected_period_seconds=1.0,
            params=ProcessingParams(template_correlation_threshold=0.8),
        )
        self.assertEqual(len(confirmed), len(beats))
        self.assertEqual(info["method"], "kept_timing_complete")
        self.assertEqual(len(analysis), len(beats))
        self.assertGreater(len(template), 0)

    def test_manual_beats_regenerate_denoising_outputs(self) -> None:
        sample_rate = 8000
        signal_data = np.zeros(sample_rate * 5, dtype=np.float32)
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, (signal_data * 32767).astype(np.int16))
        manual_beats = [0.5, 1.3, 2.1, 2.9, 3.7]
        result = process_audio_bytes(
            "manual_test.wav",
            buffer.getvalue(),
            manual_beat_times=manual_beats,
        )
        self.assertEqual(result["summary"]["tempo"]["detected_beats"], len(manual_beats))
        self.assertEqual(result["summary"]["template_confirmation"]["method"], "manual_beat_times")
        self.assertIn("cleaned.wav", result["artifacts"])
        self.assertNotIn("best_loop.wav", result["artifacts"])


if __name__ == "__main__":
    unittest.main()
