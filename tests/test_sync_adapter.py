from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from music_processor.sync_adapter import (
    SyncServiceConfig,
    adapt_sync_result,
    build_sync_command,
    run_sync_cli,
)


class SyncAdapterTests(unittest.TestCase):
    def test_build_command_is_argument_array_and_preserves_unicode_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SyncServiceConfig(root, root / ".venv/Scripts/python.exe")
            heartbeat = root / "输入 空格/心跳.wav"
            song = root / "输入 空格/歌曲.mp3"
            output = root / "输出 文件"
            command = build_sync_command(
                config,
                heartbeat_path=heartbeat,
                song_path=song,
                output_root=output,
                manual_song_bpm=92.5,
                fit_mode="preserve",
            )
            self.assertIsInstance(command, list)
            self.assertIn(str(heartbeat.resolve()), command)
            self.assertIn(str(song.resolve()), command)
            self.assertIn(str(output.resolve()), command)
            self.assertEqual(command[command.index("--fit-mode") + 1], "gap")

    def test_run_validates_and_adapts_five_file_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "输出 root"
            run_dir = output_root / "run_001"
            run_dir.mkdir(parents=True)
            files = {
                "mix": "preview_mix.wav",
                "heartbeat_stem": "heartbeat_aligned.wav",
                "beat_check": "debug_click_mix.wav",
                "heartbeat_check": "heartbeat_detection_mix.wav",
                "report": "analysis_report.json",
            }
            audio = np.zeros((8000, 1), dtype=np.float32)
            for filename in list(files.values())[:-1]:
                sf.write(run_dir / filename, audio, 8000, subtype="PCM_24")
            report = {
                "audio": {"song_processed_duration_seconds": 1.0},
                "arrangement": {
                    "pulse_times_seconds": [0.2, 0.8],
                    "pulse_mode_requested": "auto",
                    "song_offset_seconds": 0.2,
                    "effective_intro_pulses": 1,
                    "outro_pulses": 1,
                },
                "loudness_analysis": {
                    "output_mix_integrated_lufs": -16.0,
                    "output_peak_dbfs": -1.0,
                },
            }
            (run_dir / files["report"]).write_text(
                json.dumps(report), encoding="utf-8"
            )
            summary = {
                "output_dir": str(run_dir),
                "files": files,
                "warnings": [],
            }
            completed = subprocess.CompletedProcess(
                ["python"], 0, stdout=json.dumps(summary), stderr=""
            )
            config = SyncServiceConfig(root, root / ".venv/Scripts/python.exe")
            with patch("music_processor.sync_adapter.subprocess.run", return_value=completed) as mocked:
                result = run_sync_cli(
                    config,
                    ["python", "-m", "heartbeat_sync"],
                    output_root=output_root,
                )
            self.assertFalse(mocked.call_args.kwargs.get("shell", False))
            adapted = adapt_sync_result(result)
            self.assertEqual(adapted["report"]["render"]["engine"], "heartbeat_sync_cli")
            self.assertEqual(adapted["sample_rate"], 8000)
            self.assertEqual(len(adapted["artifact_paths"]), 5)

    def test_run_rejects_output_escape_and_safe_error_is_concise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            completed = subprocess.CompletedProcess(
                ["python"],
                0,
                stdout=json.dumps({"output_dir": str(outside), "files": {}}),
                stderr="",
            )
            config = SyncServiceConfig(root, root / ".venv/Scripts/python.exe")
            with patch("music_processor.sync_adapter.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(RuntimeError, "escaped"):
                    run_sync_cli(config, ["python"], output_root=root / "safe")

            failed = subprocess.CompletedProcess(
                ["python"], 2, stdout="", stderr="traceback\nuser-facing error"
            )
            with patch("music_processor.sync_adapter.subprocess.run", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "user-facing error"):
                    run_sync_cli(config, ["python"], output_root=root / "safe")


if __name__ == "__main__":
    unittest.main()
