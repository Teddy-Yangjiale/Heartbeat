import io
import unittest

import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile

from heartbeat_preprocessor.core import (
    ProcessingParams,
    assess_recording_quality,
    cycle_consistency_denoise,
    measure_focal_cycle_contamination,
    measure_rhythm_preservation,
    phase_aware_noise_reduction,
    process_audio_bytes,
    suppress_detected_hum,
)


def heartbeat_signal(sr: int, seconds: float, beats: np.ndarray, seed: int = 7) -> np.ndarray:
    time = np.arange(int(sr * seconds), dtype=np.float32) / sr
    heart = np.zeros_like(time)
    rng = np.random.default_rng(seed)
    for beat in beats:
        for offset, scale in ((0.0, 1.0), (0.22, 0.62)):
            start = int((beat + offset) * sr)
            length = int(0.055 * sr)
            if start < 0 or start + length > len(heart):
                continue
            burst = rng.standard_normal(length).astype(np.float32) * np.hanning(length).astype(np.float32)
            heart[start : start + length] += scale * burst
    return heart


def wav_data(sr: int, signal_data: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, sr, (np.clip(signal_data, -1.0, 1.0) * 32767).astype(np.int16))
    return buffer.getvalue()


class HeartbeatDenoisingTest(unittest.TestCase):
    def test_detected_mains_hum_is_notched_without_erasing_heart_events(self) -> None:
        sr = 8000
        seconds = 12
        time = np.arange(sr * seconds, dtype=np.float32) / sr
        beats = np.arange(0.8, seconds - 0.5, 0.8)
        heart = 0.55 * heartbeat_signal(sr, seconds, beats)
        contaminated = heart + 0.20 * np.sin(2.0 * np.pi * 50.0 * time)
        output, report = suppress_detected_hum(contaminated, sr, ProcessingParams())

        frequencies, before = sp_signal.welch(contaminated, fs=sr, nperseg=8192)
        _, after = sp_signal.welch(output, fs=sr, nperseg=8192)
        hum_bin = int(np.argmin(np.abs(frequencies - 50.0)))
        reduction_db = 10.0 * np.log10((after[hum_bin] + 1e-18) / (before[hum_bin] + 1e-18))
        heart_mask = np.abs(heart) > 0.03
        correlation = float(np.corrcoef(heart[heart_mask], output[heart_mask])[0, 1])

        self.assertTrue(report["applied"])
        self.assertIn(50.0, report["detected_frequencies_hz"])
        self.assertLess(reduction_db, -12.0)
        self.assertGreater(correlation, 0.70)

    def test_phase_noise_model_reduces_diastolic_floor_and_builds_cycle_pool(self) -> None:
        sr = 8000
        seconds = 15
        time = np.arange(sr * seconds, dtype=np.float32) / sr
        beats = np.arange(0.8, seconds - 0.5, 0.8)
        rng = np.random.default_rng(31)
        heart = 0.62 * heartbeat_signal(sr, seconds, beats, seed=32)
        contaminated = heart + rng.normal(0.0, 0.035, len(time)).astype(np.float32)
        reduced, report = phase_aware_noise_reduction(
            contaminated,
            sr,
            beats,
            ProcessingParams(),
        )
        phase = ((time - beats[0]) % 0.8) / 0.8
        quiet = (time >= beats[0]) & (time <= beats[-1]) & (phase >= 0.55) & (phase <= 0.85)
        heart_mask = np.abs(heart) > 0.03

        def rms(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(values[mask]))))

        quiet_change_db = 20.0 * np.log10((rms(reduced, quiet) + 1e-12) / (rms(contaminated, quiet) + 1e-12))
        heart_change_db = 20.0 * np.log10((rms(reduced, heart_mask) + 1e-12) / (rms(contaminated, heart_mask) + 1e-12))
        result = process_audio_bytes(
            "pool.wav",
            wav_data(sr, contaminated),
            manual_beat_times=beats,
            artifact_profile="web",
            create_zip=False,
        )

        self.assertTrue(report["applied"])
        self.assertLess(quiet_change_db, -2.0)
        self.assertGreater(heart_change_db, quiet_change_db + 1.0)
        self.assertGreater(len(result["cycle_pool"]), 4)
        self.assertIn("heartbeat_cycle_pool_preview.wav", result["artifacts"])

    def test_rejects_audio_over_configured_duration_limit(self) -> None:
        sr = 8000
        pcm = np.zeros(sr * 2, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "maximum allowed duration is 1 seconds"):
            process_audio_bytes(
                "too_long.wav",
                wav_data(sr, pcm),
                max_duration_seconds=1.0,
            )

    def test_reduces_sustained_voice_between_beats(self) -> None:
        sr = 8000
        seconds = 12
        time = np.arange(sr * seconds, dtype=np.float32) / sr
        beats = np.arange(0.8, seconds - 0.5, 0.8)
        voice = 0.22 * (np.sin(2 * np.pi * 110 * time) + 0.45 * np.sin(2 * np.pi * 220 * time))
        pcm = voice + 0.55 * heartbeat_signal(sr, seconds, beats)

        without_denoising = process_audio_bytes(
            "synthetic_voice_contaminated.wav",
            wav_data(sr, pcm),
            ProcessingParams(enable_denoising=False),
        )
        with_denoising = process_audio_bytes("synthetic_voice_contaminated.wav", wav_data(sr, pcm))

        between_beats = np.zeros_like(time, dtype=bool)
        for beat in beats:
            between_beats |= (time >= beat + 0.38) & (time <= beat + 0.68)

        def rms(signal_data: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(signal_data[between_beats]))))

        reduction_db = 20.0 * np.log10(
            (rms(with_denoising["cleaned"]) + 1e-12)
            / (rms(without_denoising["cleaned"]) + 1e-12)
        )
        self.assertLess(reduction_db, -20.0)
        self.assertGreaterEqual(len(with_denoising["beat_times"]), 8)

    def test_cycle_consistency_reduces_non_repeating_friction_without_thinning_heartbeats(self) -> None:
        sr = 8000
        seconds = 15
        time = np.arange(sr * seconds, dtype=np.float32) / sr
        beats = np.arange(0.8, seconds - 0.5, 0.8)
        heart = heartbeat_signal(sr, seconds, beats, seed=21)
        friction = np.zeros_like(time)
        rng = np.random.default_rng(22)
        contaminated_cycles = ((2, 0.38), (5, 0.51), (9, 0.34), (13, 0.57))
        for cycle_index, offset in contaminated_cycles:
            start = int((beats[cycle_index] + offset) * sr)
            length = int(0.07 * sr)
            friction[start : start + length] += (
                0.75
                * rng.standard_normal(length).astype(np.float32)
                * np.hanning(length).astype(np.float32)
            )

        contaminated = 0.58 * heart + friction
        denoised, info = cycle_consistency_denoise(contaminated, sr, beats, ProcessingParams())
        friction_mask = np.abs(friction) > 1e-5
        heartbeat_mask = np.abs(heart) > 0.02

        def rms(signal_data: np.ndarray, mask: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(signal_data[mask]))))

        friction_reduction_db = 20.0 * np.log10(
            (rms(denoised, friction_mask) + 1e-12) / (rms(contaminated, friction_mask) + 1e-12)
        )
        heartbeat_change_db = 20.0 * np.log10(
            (rms(denoised, heartbeat_mask) + 1e-12) / (rms(contaminated, heartbeat_mask) + 1e-12)
        )
        heartbeat_correlation = float(np.corrcoef(contaminated[heartbeat_mask], denoised[heartbeat_mask])[0, 1])

        self.assertTrue(info["applied"])
        self.assertEqual(info["method"], "attenuation_only_cycle_envelope")
        self.assertLess(friction_reduction_db, -10.0)
        self.assertGreater(heartbeat_change_db, -1.0)
        self.assertGreater(heartbeat_correlation, 0.99)

    def test_exports_before_after_and_quality_contract(self) -> None:
        sr = 8000
        seconds = 15
        beats = np.arange(0.8, seconds - 0.5, 0.8)
        result = process_audio_bytes(
            "contract.wav",
            wav_data(sr, 0.55 * heartbeat_signal(sr, seconds, beats)),
            manual_beat_times=beats,
        )

        self.assertTrue(result["cycle_consistency"]["applied"])
        for artifact in (
            "input_reference.wav",
            "spectral_filtered.wav",
            "filtered_detection.wav",
            "cleaned.wav",
            "cleanest_heartbeat_loop.wav",
            "cleanest_heartbeat_loop_loud.wav",
            "cleanest_segment.json",
            "cleanest_segment_candidates.csv",
            "cycle_consistency.json",
            "rhythm_preservation.json",
            "focal_cycle_contamination.json",
            "postprocess_beat_times.csv",
            "recording_quality.json",
        ):
            self.assertIn(artifact, result["artifacts"])
        self.assertNotIn("best_loop.wav", result["artifacts"])
        self.assertEqual(result["cleanest_segment"]["cycle_count"], 4)
        self.assertFalse(result["cleanest_segment"]["is_fallback"])
        self.assertTrue(
            result["cleanest_segment"]["playback_loudness"]["is_playback_optimized"]
        )
        self.assertLessEqual(
            result["cleanest_segment"]["playback_loudness"]["achieved_peak_dbfs"],
            -0.99,
        )
        self.assertEqual(
            result["summary"]["quality"]["reconstruction_policy"],
            "attenuation_only_no_template_replacement",
        )
        rhythm = result["rhythm_preservation"]
        self.assertTrue(rhythm["applied"])
        self.assertTrue(rhythm["is_preserved"])
        self.assertGreaterEqual(rhythm["matched_fraction"], 0.95)
        self.assertEqual(rhythm["count_delta"], 0)
        self.assertEqual(result["focal_cycle_contamination"]["severe_cycle_count"], 0)

    def test_isolated_high_energy_non_template_cycle_is_flagged(self) -> None:
        sr = 8000
        seconds = 12
        beats = np.arange(0.8, seconds - 0.5, 0.8)
        clean = np.zeros(sr * seconds, dtype=np.float32)
        pulse_length = int(0.07 * sr)
        pulse_time = np.arange(pulse_length, dtype=np.float32) / sr
        pulse = (
            np.sin(2 * np.pi * 62.0 * pulse_time)
            * np.hanning(pulse_length).astype(np.float32)
        )
        for beat in beats:
            for offset, scale in ((0.0, 0.45), (0.22, 0.28)):
                start = int((beat + offset) * sr)
                clean[start : start + pulse_length] += scale * pulse
        contaminated = clean.copy()
        contaminated_index = 5
        start = int((beats[contaminated_index] - 0.09) * sr)
        end = int((beats[contaminated_index] + 0.30) * sr)
        contaminated[start:end] *= -3.0

        focal = measure_focal_cycle_contamination(
            contaminated,
            sr,
            beats,
            ProcessingParams(),
        )

        self.assertTrue(focal["applied"])
        self.assertGreaterEqual(focal["severe_cycle_count"], 1)
        self.assertIn(
            contaminated_index,
            [item["beat_index"] for item in focal["severe_cycles"]],
        )

    def test_rhythm_check_rejects_a_destroyed_postprocess_signal(self) -> None:
        sr = 8000
        seconds = 12
        beats = np.arange(0.8, seconds - 0.5, 0.8)
        rhythm = measure_rhythm_preservation(
            np.zeros(sr * seconds, dtype=np.float32),
            sr,
            beats,
            0.8,
            ProcessingParams(),
        )

        self.assertTrue(rhythm["applied"])
        self.assertFalse(rhythm["is_preserved"])
        self.assertEqual(rhythm["processed_beat_count"], 0)
        self.assertEqual(rhythm["matched_beat_count"], 0)
        self.assertEqual(rhythm["count_delta"], -len(beats))

    def test_failed_rhythm_verification_forces_rerecording(self) -> None:
        sr = 8000
        seconds = 12
        beats = np.arange(0.8, seconds - 0.5, 0.8)
        raw = heartbeat_signal(sr, seconds, beats)
        envelope = np.abs(raw)
        quality = assess_recording_quality(
            raw,
            raw,
            envelope,
            beats,
            sr,
            {
                "estimated_bpm": 75.0,
                "consensus_window_count": 3,
                "window_count": 3,
            },
            [],
            {
                "is_clipping_suspected": False,
                "interbeat_noise_reduction_db": -10.0,
                "heartbeat_preservation_correlation": 1.0,
                "rhythm_preservation": {
                    "applied": True,
                    "is_preserved": False,
                    "matched_fraction": 0.5,
                    "count_delta": -7,
                    "median_timing_error_ms": 20.0,
                    "median_ibi_error_fraction": 0.01,
                },
            },
            {
                "enabled": False,
                "candidate_count": len(beats),
                "confirmation_fraction": 1.0,
            },
            {
                "enabled": False,
                "applied": False,
                "outlier_fraction": 0.0,
                "cycles_used": 0,
            },
            ProcessingParams(),
        )

        self.assertTrue(quality["needs_rerecording"])
        self.assertEqual(quality["denoising_status"], "rerecord")
        self.assertTrue(
            any("count or timing" in reason.lower() for reason in quality["rerecord_reasons"])
        )

    def test_low_cycle_confidence_is_limited_without_inventing_a_rerecord_failure(self) -> None:
        sr = 8000
        seconds = 12
        beats = np.asarray(
            [0.7, 1.2, 2.2, 2.7, 3.7, 4.2, 5.2, 5.7, 6.7, 7.2, 8.2, 8.7, 9.7, 10.2, 11.2],
            dtype=np.float64,
        )
        raw = heartbeat_signal(sr, seconds, beats)
        quality = assess_recording_quality(
            raw,
            raw,
            np.abs(raw),
            beats,
            sr,
            {
                "estimated_bpm": 75.0,
                "consensus_window_count": 3,
                "window_count": 3,
            },
            [],
            {
                "is_clipping_suspected": False,
                "interbeat_noise_reduction_db": -10.0,
                "heartbeat_preservation_correlation": 1.0,
                "rhythm_preservation": {
                    "applied": True,
                    "is_preserved": True,
                    "matched_fraction": 1.0,
                    "count_delta": 0,
                    "median_timing_error_ms": 0.0,
                    "median_ibi_error_fraction": 0.0,
                },
                "focal_cycle_contamination": {
                    "applied": True,
                    "severe_cycle_count": 0,
                    "severe_cycle_fraction": 0.0,
                },
            },
            {
                "enabled": True,
                "candidate_count": len(beats),
                "confirmation_fraction": 1.0,
                "median_correlation": 0.2,
            },
            {
                "enabled": True,
                "applied": True,
                "median_cycle_correlation": 0.9,
                "outlier_fraction": 0.0,
                "cycles_used": len(beats),
            },
            ProcessingParams(),
        )

        self.assertFalse(quality["needs_rerecording"])
        self.assertEqual(quality["denoising_status"], "limited")
        self.assertLess(quality["score"], 75.0)
        self.assertGreater(
            quality["metrics"]["ibi_coefficient_of_variation"],
            ProcessingParams().quality_high_ibi_cv,
        )

    def test_irrecoverable_clipping_requests_rerecording(self) -> None:
        sr = 8000
        clipped = np.ones(sr * 10, dtype=np.float32)
        result = process_audio_bytes("clipped.wav", wav_data(sr, clipped))
        quality = result["recording_quality"]
        self.assertTrue(quality["needs_rerecording"])
        self.assertEqual(quality["denoising_status"], "rerecord")
        self.assertTrue(any("clipping" in reason.lower() for reason in quality["rerecord_reasons"]))


if __name__ == "__main__":
    unittest.main()
