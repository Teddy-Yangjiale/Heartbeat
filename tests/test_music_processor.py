from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile

from heartbeat_preprocessor.core import process_audio_bytes
from music_processor.core import (
    HeartbeatCycle,
    MixParams,
    RegionEdit,
    _detect_s1_anchor,
    _repair_missing_beats,
    analyze_song_bytes,
    build_adaptive_pulse_grid,
    build_region_schedule,
    fit_cycle,
    get_style_preset,
    process_music_bytes,
    render_heartbeat_layer,
    trim_song_content,
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
        self.assertNotIn("audio", analysis)
        self.assertLessEqual(len(analysis["waveform_overview_values"]), 10000)
        self.assertGreater(len(analysis["downbeat_times_seconds"]), 1)

    def test_song_file_like_input_is_reused_without_losing_position(self) -> None:
        source = io.BytesIO(self.song_bytes)
        source.seek(17)
        analysis = analyze_song_bytes(
            "song.wav",
            source,
            manual_bpm=120.0,
            manual_first_beat=0.3,
        )
        self.assertEqual(source.tell(), 17)
        self.assertEqual(analysis["channels"], 2)

    def test_adaptive_grid_uses_real_beats_and_marks_only_gap_guides(self) -> None:
        beats = np.concatenate([np.arange(0.0, 4.0, 0.5), np.arange(4.0, 8.0, 0.75)])
        pulses, model_backed, relaxations = build_adaptive_pulse_grid(
            beats,
            75.0,
            8.0,
            downbeats=np.asarray([0.0, 2.0, 4.0, 7.0]),
            active_duration_seconds=0.35,
        )
        self.assertGreater(len(pulses), 4)
        self.assertTrue(np.all(model_backed))
        self.assertEqual(relaxations, 0)
        for pulse in pulses:
            self.assertTrue(np.any(np.isclose(beats, pulse, atol=1e-9)))

        gap_beats = np.asarray([0.0, 0.5, 1.0, 1.5, 5.0, 5.5, 6.0])
        _, gap_model_backed, _ = build_adaptive_pulse_grid(
            gap_beats,
            75.0,
            6.0,
            active_duration_seconds=0.35,
        )
        self.assertTrue(np.any(~gap_model_backed))

    def test_s1_anchor_is_rendered_at_the_requested_pulse(self) -> None:
        sample_rate = 8000
        audio = np.zeros((int(0.7 * sample_rate), 1), dtype=np.float32)
        onset = int(0.08 * sample_rate)
        local = np.arange(int(0.12 * sample_rate)) / sample_rate
        audio[onset : onset + len(local), 0] = np.sin(2 * np.pi * 70 * local) * np.exp(-25 * local)
        anchor = _detect_s1_anchor(audio, sample_rate)
        self.assertGreater(anchor, int(0.03 * sample_rate))
        cycle = HeartbeatCycle(audio, anchor, int(0.3 * sample_rate), 0)
        schedule = [
            {
                "time_seconds": 1.0,
                "fit_mode": "gap",
                "velocity": 1.0,
            }
        ]
        rendered, report = render_heartbeat_layer(
            [cycle], schedule, sample_rate, 1, int(2.0 * sample_rate)
        )
        self.assertGreater(float(np.max(np.abs(rendered))), 0.1)
        self.assertEqual(report["skipped_count"], 0)
        self.assertLessEqual(report["maximum_error_ms"], 1e-6)

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
        self.assertAlmostEqual(result["song_duration_seconds"], 2.0, places=3)
        self.assertGreater(result["duration_seconds"], result["song_duration_seconds"])
        self.assertAlmostEqual(
            len(rendered_audio) / rendered_rate,
            result["duration_seconds"],
            places=3,
        )

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

    def test_preserve_mode_cuts_noisy_tail_without_stretching_s1_s2(self) -> None:
        sample_rate = 8000
        audio = np.zeros((int(0.8 * sample_rate), 1), dtype=np.float32)
        audio[int(0.08 * sample_rate) : int(0.30 * sample_rate)] = 0.5
        audio[int(0.45 * sample_rate) :] = 0.12
        cycle = HeartbeatCycle(
            audio=audio,
            anchor_offset_samples=int(0.08 * sample_rate),
            active_samples=int(0.34 * sample_rate),
            source_cycle_index=0,
        )
        fitted, anchor = fit_cycle(cycle, int(1.1 * sample_rate), sample_rate, "preserve")
        self.assertEqual(anchor, cycle.anchor_offset_samples)
        np.testing.assert_allclose(
            fitted[: int(0.30 * sample_rate)],
            audio[: int(0.30 * sample_rate)],
            atol=1e-7,
        )
        self.assertLess(float(np.max(np.abs(fitted[int(0.38 * sample_rate) :]))), 1e-7)

    def test_groove_controls_and_detected_kick_role_are_applied(self) -> None:
        beats = np.arange(0.0, 6.0, 0.5)
        kicks = np.asarray([0.04, 1.02, 2.06, 3.01, 4.05, 5.02])
        schedule = build_region_schedule(
            beats,
            6.0,
            "kick",
            "preserve",
            4,
            0.0,
            None,
            [],
            kick_times=kicks,
            humanize_ms=12.0,
            swing=0.10,
        )
        targets = np.asarray([item["target_time_seconds"] for item in schedule])
        offsets = np.asarray([item["groove_offset_ms"] for item in schedule])
        np.testing.assert_allclose(targets, kicks, atol=1e-9)
        self.assertTrue(np.any(np.abs(offsets) > 1.0))
        self.assertLess(float(np.max(np.abs(offsets))), 50.0)

    def test_style_presets_and_compact_exports_change_the_audio_contract(self) -> None:
        cinematic = get_style_preset("cinematic")
        lofi = get_style_preset("lofi")
        self.assertNotEqual(cinematic["quantize_strength"], lofi["quantize_strength"])
        self.assertNotEqual(cinematic["saturation"], lofi["saturation"])
        analysis = analyze_song_bytes(
            "song.wav",
            self.song_bytes,
            manual_bpm=120.0,
            manual_first_beat=0.3,
        )
        wav_result = process_music_bytes(
            "song.wav",
            self.song_bytes,
            self.heartbeat,
            analysis,
            MixParams(output_format="wav24"),
            render_duration_seconds=4.0,
            export_stems=False,
            export_debug=False,
            create_zip=False,
        )
        flac_result = process_music_bytes(
            "song.wav",
            self.song_bytes,
            self.heartbeat,
            analysis,
            MixParams(output_format="flac16"),
            render_duration_seconds=4.0,
            export_stems=False,
            export_debug=False,
            create_zip=False,
        )
        mp3_result = process_music_bytes(
            "song.wav",
            self.song_bytes,
            self.heartbeat,
            analysis,
            MixParams(output_format="mp3"),
            render_duration_seconds=4.0,
            export_stems=False,
            export_debug=False,
            create_zip=False,
        )
        self.assertIn("final_mix.flac", flac_result["artifacts"])
        self.assertLess(
            len(flac_result["artifacts"]["final_mix.flac"]),
            len(wav_result["artifacts"]["final_mix.wav"]),
        )
        decoded, rate = sf.read(io.BytesIO(flac_result["artifacts"]["final_mix.flac"]), always_2d=True)
        self.assertEqual(rate, 22050)
        self.assertEqual(decoded.shape[1], 2)
        self.assertIn("final_mix.mp3", mp3_result["artifacts"])
        self.assertLess(
            len(mp3_result["artifacts"]["final_mix.mp3"]),
            len(wav_result["artifacts"]["final_mix.wav"]),
        )
        mp3_decoded, mp3_rate = sf.read(
            io.BytesIO(mp3_result["artifacts"]["final_mix.mp3"]),
            always_2d=True,
        )
        self.assertEqual(mp3_rate, 22050)
        self.assertEqual(mp3_decoded.shape[1], 2)

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
        self.assertAlmostEqual(result["song_duration_seconds"], 8.0, places=3)
        self.assertGreater(result["duration_seconds"], result["song_duration_seconds"])
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
        self.assertAlmostEqual(
            len(rendered_audio) / rendered_rate,
            result["duration_seconds"],
            places=3,
        )
        with zipfile.ZipFile(io.BytesIO(result["zip_bytes"])) as archive:
            self.assertIn("final_mix.wav", archive.namelist())
            self.assertIn("heartbeat_timeline.csv", archive.namelist())

    def test_disk_output_keeps_large_wav_bytes_out_of_result(self) -> None:
        analysis = analyze_song_bytes(
            "song.wav",
            self.song_bytes,
            manual_bpm=120.0,
            manual_first_beat=0.3,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = process_music_bytes(
                "song.wav",
                self.song_bytes,
                self.heartbeat,
                analysis,
                render_duration_seconds=2.0,
                output_dir=directory,
                export_stems=False,
                export_debug=False,
                create_zip=False,
            )
            self.assertNotIn("artifacts", result)
            self.assertNotIn("zip_bytes", result)
            self.assertEqual(
                set(result["artifact_paths"]),
                {
                    "final_mix.wav",
                    "mix_report.json",
                    "heartbeat_timeline.csv",
                    "region_edits.json",
                },
            )
            self.assertTrue(Path(result["artifact_paths"]["final_mix.wav"]).is_file())

    def test_hybrid_content_trim_and_manual_overrides(self) -> None:
        sample_rate = 8000
        audio = np.zeros((5 * sample_rate, 1), dtype=np.float32)
        time = np.arange(int(2.5 * sample_rate), dtype=np.float32) / sample_rate
        audio[int(1.5 * sample_rate) : int(4.0 * sample_rate), 0] = (
            0.2 * np.sin(2.0 * np.pi * 30.0 * time)
        )
        trimmed, report = trim_song_content(
            audio,
            sample_rate,
            enabled=True,
            top_db=30.0,
            manual_start_seconds=None,
            manual_end_seconds=None,
        )
        self.assertGreater(report["used_start_seconds"], 1.2)
        self.assertLess(report["used_start_seconds"], 1.7)
        self.assertGreater(len(trimmed), 2 * sample_rate)
        manual, manual_report = trim_song_content(
            audio,
            sample_rate,
            enabled=True,
            top_db=30.0,
            manual_start_seconds=1.0,
            manual_end_seconds=4.5,
        )
        self.assertEqual(len(manual), int(3.5 * sample_rate))
        self.assertAlmostEqual(manual_report["used_start_seconds"], 1.0)
        self.assertAlmostEqual(manual_report["used_end_seconds"], 4.5)

    def test_sync_contract_exports_five_pcm24_files_and_report_sections(self) -> None:
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
            MixParams(
                intro_pulses=2,
                outro_pulses=2,
                sync_contract_exports=True,
            ),
            render_duration_seconds=4.0,
            export_stems=False,
            export_debug=False,
            create_zip=False,
        )
        artifacts = result["artifacts"]
        expected = {
            "preview_mix.wav",
            "heartbeat_aligned.wav",
            "debug_click_mix.wav",
            "heartbeat_detection_mix.wav",
            "analysis_report.json",
        }
        self.assertTrue(expected.issubset(artifacts))
        for filename in expected - {"analysis_report.json"}:
            self.assertEqual(sf.info(io.BytesIO(artifacts[filename])).subtype, "PCM_24")
        report = json.loads(artifacts["analysis_report.json"])
        self.assertEqual(
            set(report),
            {
                "run",
                "inputs",
                "audio",
                "content_trim",
                "song_analysis",
                "heartbeat_analysis",
                "arrangement",
                "loudness_analysis",
                "alignment",
            },
        )
        self.assertEqual(report["arrangement"]["effective_intro_pulses"], 2)
        self.assertEqual(report["arrangement"]["outro_pulses"], 2)
        self.assertTrue(report["run"]["exports_enabled"])


if __name__ == "__main__":
    unittest.main()
