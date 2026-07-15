import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

from legasynth.mixing import mix_heartbeat_with_song
from legasynth.remix_pipeline import process_song_remix
from legasynth.song_analysis import analyze_song_audio, save_song_analysis
from legasynth.stems import extract_vocal_melody
from heartbeat_preprocessor.core import ProcessingParams


class SongRemixTest(unittest.TestCase):
    def test_song_override_builds_phase_anchored_grid(self) -> None:
        sr = 8000
        duration = 4.0
        y = np.zeros(int(sr * duration), dtype=np.float32)
        for time_seconds in np.arange(0.25, duration, 0.5):
            start = int(time_seconds * sr)
            y[start : start + 80] += np.hanning(80).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            song_path = Path(tmp) / "song.wav"
            sf.write(song_path, y, sr)
            analysis = analyze_song_audio(
                song_path,
                bpm_override=120.0,
                first_beat_override=0.25,
                beats_per_bar=3,
            )
            paths = save_song_analysis(analysis, Path(tmp) / "analysis")

            self.assertEqual(analysis["estimated_bpm"], 120.0)
            self.assertEqual(analysis["first_beat_seconds"], 0.25)
            self.assertEqual(analysis["beats_per_bar"], 3)
            self.assertEqual(analysis["beat_grid_times_seconds"][:3], [0.25, 0.75, 1.25])
            grid = pd.read_csv(paths["song_beat_grid_csv"])
            self.assertEqual(grid["beat_in_bar"].head(4).tolist(), [1, 2, 3, 1])

    def test_song_model_disagreement_requires_review(self) -> None:
        sr = 8000
        duration = 6.0
        y = np.zeros(int(sr * duration), dtype=np.float32)
        for time_seconds in np.arange(0.25, duration, 0.5):
            start = int(time_seconds * sr)
            y[start : start + 100] += np.hanning(100).astype(np.float32)
        model_beats = np.arange(0.4, duration, 2.0 / 3.0)
        model = {
            "backend": "beatnet_crnn_dbn",
            "beat_times_seconds": model_beats.tolist(),
            "beat_numbers": [(index % 3) + 1 for index in range(len(model_beats))],
            "downbeat_times_seconds": model_beats[::3].tolist(),
            "estimated_bpm": 90.0,
            "estimated_meter": 3,
        }

        with tempfile.TemporaryDirectory() as tmp:
            song_path = Path(tmp) / "song.wav"
            sf.write(song_path, y, sr)
            with patch("legasynth.song_analysis.beatnet_available", return_value=True), patch(
                "legasynth.song_analysis.run_beatnet", return_value=model
            ):
                analysis = analyze_song_audio(song_path)

        self.assertTrue(analysis["requires_manual_review"])
        self.assertGreater(analysis["tempo_ensemble_relative_error"], 0.12)
        self.assertTrue(any("disagree" in warning for warning in analysis["warnings"]))

    def test_mix_starts_heartbeat_at_first_beat_and_fits_exact_loop_duration(self) -> None:
        sr = 8000
        song = np.zeros((sr * 5, 2), dtype=np.float32)
        loop_time = np.arange(sr, dtype=np.float32) / sr
        loop = (0.5 * np.sin(2 * np.pi * 60 * loop_time))[:, None]
        summary = {"best_loop": {"num_beats": 4, "local_bpm": 240.0}, "tempo": {"estimated_bpm": 240.0}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song_path = root / "song.wav"
            loop_path = root / "loop.wav"
            sf.write(song_path, song, sr)
            sf.write(loop_path, loop, sr)
            report = mix_heartbeat_with_song(
                song_path,
                loop_path,
                summary,
                song_bpm=120.0,
                out_dir=root / "mix",
                heartbeat_gain_db=-6.0,
                first_beat_seconds=0.5,
            )

            layer, layer_sr = sf.read(report["heartbeat_layer_wav"], always_2d=True)
            self.assertEqual(layer_sr, sr)
            self.assertTrue(np.allclose(layer[: int(0.5 * sr)], 0.0, atol=1e-5))
            self.assertGreater(float(np.max(np.abs(layer[int(0.5 * sr) :]))), 0.05)
            self.assertAlmostEqual(report["target_loop_duration_seconds"], 2.0, places=6)
            beats = pd.read_csv(report["aligned_heartbeat_beats_csv"])
            self.assertAlmostEqual(float(beats.iloc[0]["time_seconds"]), 0.5, places=6)
            persisted = json.loads((root / "mix" / "mix_report.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["first_beat_seconds"], 0.5)

    def test_dynamic_tempo_map_prevents_accumulated_loop_drift(self) -> None:
        sr = 8000
        song = np.zeros((sr * 7, 2), dtype=np.float32)
        loop_time = np.arange(sr * 2, dtype=np.float32) / sr
        loop = (0.25 * np.sin(2 * np.pi * 55 * loop_time))[:, None]
        summary = {"best_loop": {"num_beats": 4, "local_bpm": 120.0}}
        beat_grid = [0.25, 0.75, 1.26, 1.78, 2.31, 2.85, 3.40, 3.96, 4.53, 5.11, 5.70, 6.30]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song_path = root / "song.wav"
            loop_path = root / "loop.wav"
            sf.write(song_path, song, sr)
            sf.write(loop_path, loop, sr)
            report = mix_heartbeat_with_song(
                song_path,
                loop_path,
                summary,
                song_bpm=112.0,
                out_dir=root / "dynamic",
                song_beat_times=beat_grid,
                first_beat_seconds=0.25,
            )

            self.assertEqual(report["grid_mode"], "dynamic_tempo_map")
            self.assertEqual(report["tempo_mapped_segment_count"], 3)
            exported = pd.read_csv(report["aligned_heartbeat_beats_csv"])
            self.assertTrue(np.allclose(exported["time_seconds"], beat_grid, atol=1e-6))
            self.assertAlmostEqual(report["tempo_mapped_segment_duration_min_seconds"], 2.06, places=6)

    def test_end_to_end_audio_pipeline_exports_stage2_contract(self) -> None:
        sr = 8000
        heartbeat = np.zeros(sr * 7, dtype=np.float32)
        pulse = np.hanning(160).astype(np.float32)
        for second in range(7):
            heartbeat[second * sr : second * sr + len(pulse)] += pulse
        song = np.zeros((sr * 4, 2), dtype=np.float32)
        for time_seconds in np.arange(0.25, 4.0, 0.5):
            start = int(time_seconds * sr)
            song[start : start + len(pulse), :] += pulse[:, None] * 0.2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heartbeat_path = root / "heartbeat.wav"
            song_path = root / "song.wav"
            sf.write(heartbeat_path, heartbeat, sr)
            sf.write(song_path, song, sr)
            report = process_song_remix(
                heartbeat_path,
                song_path,
                root / "output",
                params=ProcessingParams(
                    min_bpm=50.0,
                    max_bpm=90.0,
                    target_loop_beats=4,
                    enable_speech_suppression=False,
                    enable_template_confirmation=False,
                ),
                song_bpm_override=120.0,
                first_beat_override=0.25,
                beats_per_bar=4,
            )

            self.assertEqual(report["stage"], "stage2_song_aligned_heartbeat_remix")
            self.assertTrue(Path(report["outputs"]["heartbeat_best_loop_wav"]).is_file())
            self.assertTrue(Path(report["outputs"]["heartbeat_layer_wav"]).is_file())
            self.assertTrue(Path(report["outputs"]["final_mix_wav"]).is_file())
            self.assertTrue(Path(report["outputs"]["all_outputs_zip"]).is_file())

    def test_vocal_melody_extraction_tracks_monophonic_pitch(self) -> None:
        sr = 16000
        time = np.arange(sr * 2, dtype=np.float32) / sr
        vocals = (0.2 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocals_path = root / "vocals.wav"
            sf.write(vocals_path, vocals, sr)
            summary = extract_vocal_melody(vocals_path, root / "melody", prefer_basic_pitch=False)

            self.assertGreater(summary["voiced_fraction"], 0.5)
            self.assertAlmostEqual(summary["median_midi_note"], 69.0, delta=0.25)
            self.assertTrue(Path(summary["melody_csv"]).is_file())


if __name__ == "__main__":
    unittest.main()
