from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf


@dataclass(frozen=True)
class SyncServiceConfig:
    repo: Path
    python: Path


def discover_sync_service(repo: str | Path | None = None) -> SyncServiceConfig | None:
    """Find the separately installed heartbeat_sync Windows CLI service."""
    configured = Path(repo) if repo is not None else None
    if configured is None:
        value = os.environ.get("HEARTBEAT_SYNC_REPO", "").strip()
        configured = Path(value) if value else None
    if configured is None:
        return None
    root = configured.expanduser().resolve()
    python = root / ".venv" / "Scripts" / "python.exe"
    if not root.is_dir() or not python.is_file():
        return None
    return SyncServiceConfig(repo=root, python=python)


def build_sync_command(
    config: SyncServiceConfig,
    *,
    heartbeat_path: str | Path,
    song_path: str | Path,
    output_root: str | Path,
    song_analysis_backend: str = "auto",
    beat_device: str = "auto",
    manual_song_bpm: float | None = None,
    manual_first_beat: float | None = None,
    trim_silence: bool = True,
    trim_top_db: float = 30.0,
    song_start_seconds: float | None = None,
    song_end_seconds: float | None = None,
    max_song_seconds: float | None = None,
    beats_per_loop: int = 4,
    intro_pulses: int = 4,
    outro_pulses: int = 4,
    pulse_mode: str = "auto",
    pulse_min: float = 55.0,
    pulse_max: float = 110.0,
    song_gain_db: float = 0.0,
    heartbeat_gain_db: float = 0.0,
    auto_loudness: bool = True,
    song_target_lufs: float = -18.0,
    heartbeat_relative_lu: float = 3.0,
    intro_outro_boost_db: float = 4.0,
    fit_mode: str = "gap",
) -> list[str]:
    """Build an argument array; never build a shell command string."""
    command = [
        str(config.python),
        "-m",
        "heartbeat_sync",
        "--heartbeat",
        str(Path(heartbeat_path).resolve()),
        "--song",
        str(Path(song_path).resolve()),
        "--output-dir",
        str(Path(output_root).resolve()),
        "--song-analysis-backend",
        song_analysis_backend,
        "--beat-device",
        beat_device,
        "--trim-top-db",
        str(float(trim_top_db)),
        "--beats-per-loop",
        str(max(1, int(beats_per_loop))),
        "--intro-pulses",
        str(max(0, int(intro_pulses))),
        "--outro-pulses",
        str(max(0, int(outro_pulses))),
        "--pulse-mode",
        pulse_mode,
        "--pulse-min",
        str(float(pulse_min)),
        "--pulse-max",
        str(float(pulse_max)),
        "--song-gain-db",
        str(float(song_gain_db)),
        "--heartbeat-gain-db",
        str(float(heartbeat_gain_db)),
        "--song-target-lufs",
        str(float(song_target_lufs)),
        "--heartbeat-relative-lu",
        str(float(heartbeat_relative_lu)),
        "--intro-outro-boost-db",
        str(float(intro_outro_boost_db)),
        "--fit-mode",
        "gap" if fit_mode == "preserve" else fit_mode,
    ]
    optional = (
        ("--manual-song-bpm", manual_song_bpm),
        ("--manual-first-beat", manual_first_beat),
        ("--song-start-seconds", song_start_seconds),
        ("--song-end-seconds", song_end_seconds),
        ("--max-song-seconds", max_song_seconds),
    )
    for flag, value in optional:
        if value is not None:
            command.extend([flag, str(float(value))])
    if not trim_silence:
        command.append("--no-trim-silence")
    if not auto_loudness:
        command.append("--no-auto-loudness")
    return command


def _safe_error(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout or "heartbeat_sync failed").strip()
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    return lines[-1] if lines else "heartbeat_sync failed"


def run_sync_cli(
    config: SyncServiceConfig,
    command: list[str],
    *,
    output_root: str | Path,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Run heartbeat_sync and validate its stdout and file-path contract."""
    completed = subprocess.run(
        command,
        cwd=config.repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=float(timeout_seconds),
    )
    if completed.returncode != 0:
        raise RuntimeError(_safe_error(completed))
    try:
        summary = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("heartbeat_sync returned invalid JSON output") from exc
    run_dir = Path(summary["output_dir"]).resolve()
    root = Path(output_root).resolve()
    if run_dir != root and root not in run_dir.parents:
        raise RuntimeError("heartbeat_sync output directory escaped the job root")
    files = summary.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("heartbeat_sync summary did not contain a files mapping")
    resolved: dict[str, str] = {}
    for key, filename in files.items():
        path = (run_dir / str(filename)).resolve()
        if run_dir not in path.parents or not path.is_file():
            raise RuntimeError(f"heartbeat_sync output is missing or unsafe: {key}")
        resolved[str(filename)] = str(path)
    report_path = run_dir / str(files["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "analysis_report": report,
        "artifact_paths": resolved,
        "run_dir": str(run_dir),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def adapt_sync_result(result: dict[str, Any]) -> dict[str, Any]:
    """Expose external CLI output through the website's result renderer."""
    summary = result["summary"]
    analysis = result["analysis_report"]
    paths = result["artifact_paths"]
    mix_name = str(summary["files"]["mix"])
    info = sf.info(paths[mix_name])
    arrangement = analysis.get("arrangement", {})
    loudness = analysis.get("loudness_analysis", {})
    pulse_times = arrangement.get("pulse_times_seconds", [])
    website_report = {
        "render": {
            "duration_seconds": float(info.duration),
            "song_duration_seconds": float(
                analysis.get("audio", {}).get("song_processed_duration_seconds", 0.0)
            ),
            "pulse_count": len(pulse_times),
            "pulse_mode_requested": arrangement.get("pulse_mode_requested", "auto"),
            "pulse_mode_resolved": arrangement.get("pulse_mode_requested", "auto"),
            "maximum_anchor_alignment_error_ms": 0.0,
            "model_backed_pulse_count": len(pulse_times),
            "guide_pulse_count": 0,
            "song_offset_seconds": float(arrangement.get("song_offset_seconds", 0.0)),
            "effective_intro_pulses": int(arrangement.get("effective_intro_pulses", 0)),
            "outro_pulses": int(arrangement.get("outro_pulses", 0)),
            "engine": "heartbeat_sync_cli",
        },
        "master": {
            "output_lufs": float(loudness.get("output_mix_integrated_lufs", -120.0)),
            "output_peak_dbfs": float(loudness.get("output_peak_dbfs", -120.0)),
        },
        "memory": {"peak_observed_mb": 0.0, "rss_mb_by_stage": {}},
        "output": {
            "filename": mix_name,
            "format": "wav24",
            "mime": "audio/wav",
            "bit_depth": 24,
        },
        "content_trim": analysis.get("content_trim", {}),
        "sync_contract": analysis,
        "warnings": summary.get("warnings", []),
    }
    return {
        "sample_rate": int(info.samplerate),
        "duration_seconds": float(info.duration),
        "song_duration_seconds": website_report["render"]["song_duration_seconds"],
        "report": website_report,
        "schedule": [
            {"pulse_index": index, "time_seconds": float(value)}
            for index, value in enumerate(pulse_times)
        ],
        "artifact_paths": paths,
        "sync_summary": summary,
        "sync_run_dir": result["run_dir"],
    }
