from __future__ import annotations

import io
import json
import unittest
import zipfile

import numpy as np
import soundfile as sf
from scipy.io import wavfile

from heartbeat_preprocessor.core import process_audio_bytes
from music_processor.core import (
    MixParams,
    RegionEdit,
    _repair_missing_beats,
    analyze_song_bytes,
    build_region_schedule,
    process_music_bytes,
)


def wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(
        buffer,
        sample_rate,
        (np.asarray(audio).clip(-1.0, 1.0) * 32767.0).astype("<i2"),
    )
    return buffer.getvalue()


def synthetic_heartbeat(sample_rate: int = 8000, duration: float = 12.0) -> bytes:
    rng = np.random.default_rng(4)
    audio = rng.normal(0.0, 0.006, int(sample_rate * duration))
    for beat in np.arange(0.5, duration - 0.5, 0.8):
        for offset, amplitude, frequency, length in (
            (0.0, 0.82, 70.0, 0.11),
            (0.23, 0.50, 95.0, 0.08),
        ):
            begin = int(round((beat + offset) * sample_rate))
            count = min(int(round(length * sample_rate)), len(audio) - begin)
            time = np.arange(count) / sample_rate
            audio[begin : begin + count] += (
                amplitude
                * np.sin(2.0 * np.pi * frequency * time)
                * np.exp(-time * 25.0)
            )
    return wav_bytes(sample_rate, audio)


def synthetic_song(sample_rate: int = 22050, duration: float = 12.0) -> bytes:
    time = np.arange(int(sample_rate * duration)) / sample_rate
    mono = 0.10 * np.sin(2.0 * np.pi * 220.0 * time)
    for index, beat in enumerate(np.arange(0.3, duration, 0.5)):
        begin = int(round(beat * sample_rate))
        count = min(int(round(0.035 * sample_rate)), len(mono) - begin)
        local = np.arange(count) / sample_rate
        frequency = 1300.0 if index % 4 == 0 else 900.0
        mono[begin : begin + count] += (
            0.55 * np.sin(2.0 * np.pi * frequency * local) * np.exp(-local * 75.0)
        )
    return wav_bytes(sample_rate, np.column_stack([mono, mono]))


def mp3_bytes(wav_data: bytes) -> bytes:
    audio, sample_rate = sf.read(io.BytesIO(wav_data), dtype="float32", always_2d=True)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="MP3", subtype="MPEG_LAYER_III")
    return buffer.getvalue()


class MusicProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.heartbeat = process_audio_bytes(
            "heartbeat.wav",
            synthetic_heartbeat(),
            max_duration_seconds=30.0,
        )
        cls.song_bytes = synthetic_song()

    def test_manual_song_grid_uses_requested_bpm_and_anchor(self) -> None:
        analysis = analyze_song_bytes(
            "song.wav",
            self.song_bytes,
            manual_bpm=120.0,
            manual_first_beat=0.3,
        )
        grid = np.asarray(analysis["beat_grid_times_seconds"])
        self.assertEqual(analysis["grid_mode"], "constant_manual")
        self.assertAlmostEqual(grid[0], 0.3, places=5)
        np.testing.assert_allclose(np.diff(grid[:8]), 0.5, atol=1e-6)

    def test_mp3_song_analysis_and_render(self) -> None:
        encoded_song = mp3_bytes(self.song_bytes)
        analysis = analyze_song_bytes(
            "song.MP3",
            encoded_song,
            manual_bpm=120.0,
            manual_first_beat=0.3,
        )
        self.assertEqual(analysis["source"]["format"], "mp3")
        self.assertEqual(analysis["sample_rate"], 22050)
        self.assertEqual(analysis["channels"], 2)

        result = process_music_bytes(
            "song.MP3",
            encoded_song,
            self.heartbeat,
            analysis,
            render_duration_seconds=2.0,
        )
        rendered_audio, rendered_rate = sf.read(
            io.BytesIO(result["artifacts"]["final_mix.wav"]),
            always_2d=True,
        )
        self.assertEqual(rendered_rate, 22050)
        self.assertEqual(rendered_audio.shape[1], 2)
        self.assertAlmostEqual(len(rendered_audio) / rendered_rate, 2.0, places=3)

    def test_region_schedule_changes_density_and_can_mute(self) -> None:
        beats = np.arange(0.0, 8.0, 0.5)
        edits = [
            RegionEdit(2.0, 4.0, "Fast", pulse_mode="double"),
            RegionEdit(5.0, 6.0, "Break", pulse_mode="mute"),
        ]
        schedule = build_region_schedule(beats, 8.0, "normal", "gap", 4, 0.0, None, edits)
        times = np.asarray([item["time_seconds"] for item in schedule])
        self.assertGreater(np.sum((times >= 2.0) & (times < 4.0)), 4)
        self.assertFalse(np.any((times >= 5.0) & (times < 6.0)))

    def test_missing_beat_repair_requires_local_onset_support(self) -> None:
        beats = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0])
        repaired, inserted = _repair_missing_beats(beats, np.asarray([3.01]))
        self.assertEqual(inserted, 1)
        self.assertTrue(np.any(np.isclose(repaired, 3.0, atol=0.02)))
        unchanged, inserted_without_support = _repair_missing_beats(beats, np.asarray([]))
        self.assertEqual(inserted_without_support, 0)
        np.testing.assert_array_equal(unchanged, beats)

    def test_end_to_end_exports_final_mix_and_edit_contract(self) -> None:
        analysis = analyze_song_bytes(
            "song.wav",
            self.song_bytes,
            manual_bpm=120.0,
            manual_first_beat=0.3,
        )
        result = process_music_bytes(
            "song.wav",
            self.song_bytes,
            self.heartbeat,
            analysis,
            MixParams(master_target_lufs=-17.0),
            [
                RegionEdit(
                    3.0,
                    6.0,
                    "Verse",
                    song_gain_db=-3.0,
                    heartbeat_gain_db=4.0,
                    pulse_mode="double",
                    fit_mode="stretch",
                )
            ],
            render_duration_seconds=8.0,
        )
        self.assertAlmostEqual(result["duration_seconds"], 8.0, places=3)
        self.assertIn("final_mix.wav", result["artifacts"])
        self.assertIn("heartbeat_aligned.wav", result["artifacts"])
        report = json.loads(result["artifacts"]["mix_report.json"])
        self.assertEqual(report["render"]["regions"][0]["label"], "Verse")
        self.assertLessEqual(report["master"]["output_peak_dbfs"], -0.95)
        rendered_audio, rendered_rate = sf.read(
            io.BytesIO(result["artifacts"]["final_mix.wav"]),
            always_2d=True,
        )
        self.assertEqual(rendered_rate, 22050)
        self.assertEqual(rendered_audio.shape[1], 2)
        self.assertAlmostEqual(len(rendered_audio) / rendered_rate, 8.0, places=3)
        with zipfile.ZipFile(io.BytesIO(result["zip_bytes"])) as archive:
            self.assertIn("final_mix.wav", archive.namelist())
            self.assertIn("heartbeat_timeline.csv", archive.namelist())


if __name__ == "__main__":
    unittest.main()
