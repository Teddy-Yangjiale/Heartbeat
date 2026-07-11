import unittest

import numpy as np

from heartbeat_preprocessor.core import (
    ProcessingParams,
    confirm_beats_with_template,
    estimate_bpm_with_consensus,
    rank_loop_candidates,
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

    def test_loop_ranking_prefers_regular_candidate(self) -> None:
        sample_rate = 100
        beats = np.asarray([0.0, 1.0, 2.25, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        envelope = pulse_envelope(sample_rate, 8.0, beats)
        selected, candidates = rank_loop_candidates(
            beats,
            envelope,
            sample_rate,
            8.0,
            target_loop_beats=3,
            fallback_period=0.8,
            candidate_limit=5,
        )
        self.assertGreaterEqual(len(candidates), 2)
        self.assertGreaterEqual(selected["quality_score"], candidates[-1]["quality_score"])
        self.assertLess(selected["regularity_score"], 0.05)

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


if __name__ == "__main__":
    unittest.main()
