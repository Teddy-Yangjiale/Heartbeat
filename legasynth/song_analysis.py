from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from legasynth.model_backends import beatnet_available, run_beatnet


def analyze_song_audio(
    audio_path: str | os.PathLike[str],
    bpm_override: float | None = None,
    first_beat_override: float | None = None,
    beats_per_bar: int | None = None,
) -> dict[str, Any]:
    """Analyze a WAV/MP3 song and expose the beat grid used by the remixer.

    ``first_beat_seconds`` is the alignment anchor. BeatNet uses its first
    downbeat; the librosa fallback can only supply a first detected beat. The
    caller can override either BPM or phase after listening.
    """
    path = Path(audio_path)
    y, sr = librosa.load(str(path), sr=None, mono=True)
    duration = float(len(y) / sr) if sr else 0.0
    if not len(y) or sr <= 0:
        raise ValueError(f"Song audio is empty or unreadable: {path}")

    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    librosa_tempo, librosa_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sr,
        units="frames",
    )
    librosa_bpm = float(np.atleast_1d(librosa_tempo)[0])
    librosa_times = librosa.frames_to_time(librosa_frames, sr=sr)
    backend = "librosa_dynamic_beat"
    backend_warnings: list[str] = []
    beat_numbers: list[int] = []
    downbeat_times: list[float] = []
    estimated_meter: int | None = None
    model_result: dict[str, Any] | None = None
    # A full model pass is unnecessary when the user has explicitly fixed both
    # tempo and phase; the override grid is authoritative in that case.
    if beatnet_available() and not (bpm_override is not None and first_beat_override is not None):
        try:
            model_result = run_beatnet(path)
        except Exception as exc:
            backend_warnings.append(f"BeatNet failed; used librosa fallback: {str(exc)[-400:]}")
    else:
        backend_warnings.append("BeatNet is unavailable; used librosa dynamic beat tracking.")

    if model_result:
        backend = str(model_result.get("backend") or "beatnet_crnn_dbn")
        detected_times = _clean_beat_times(model_result.get("beat_times_seconds", []), duration)
        estimated_bpm = float(model_result.get("estimated_bpm") or _bpm_from_beats(detected_times))
        beat_numbers = [int(value) for value in model_result.get("beat_numbers", [])][: len(detected_times)]
        downbeat_times = [float(value) for value in model_result.get("downbeat_times_seconds", [])]
        estimated_meter = int(model_result["estimated_meter"]) if model_result.get("estimated_meter") else None
        beat_frames = librosa.time_to_frames(detected_times, sr=sr)
    else:
        beat_frames = librosa_frames
        estimated_bpm = librosa_bpm
        detected_times = librosa_times
    if bpm_override is not None:
        if bpm_override <= 0:
            raise ValueError("bpm_override must be positive")
        estimated_bpm = float(bpm_override)

    if first_beat_override is not None:
        first_beat = max(0.0, float(first_beat_override))
    elif downbeat_times:
        first_beat = float(downbeat_times[0])
    elif len(detected_times):
        first_beat = float(detected_times[0])
    else:
        first_beat = 0.0

    if estimated_bpm <= 0:
        raise ValueError("Could not estimate song BPM; provide --bpm explicitly")
    beat_period = 60.0 / estimated_bpm
    use_dynamic_grid = (
        bpm_override is None
        and first_beat_override is None
        and len(detected_times) >= 3
        and backend.startswith("beatnet")
    )
    if use_dynamic_grid:
        beat_grid = detected_times[detected_times >= first_beat - 1e-6].astype(np.float64)
        grid_mode = "model_tempo_map"
    else:
        beat_grid = np.arange(first_beat, duration + 1e-9, beat_period, dtype=np.float64)
        grid_mode = "constant_override" if bpm_override is not None or first_beat_override is not None else "constant_fallback"
    beat_strengths = onset_envelope[np.clip(beat_frames, 0, max(0, len(onset_envelope) - 1))]
    confidence = _beat_confidence(beat_strengths, onset_envelope, detected_times, duration, beat_period)
    stability = _tempo_stability(detected_times, estimated_bpm, first_beat)
    meter = max(1, int(beats_per_bar or estimated_meter or 4))
    review_warnings: list[str] = []
    tempo_ensemble_error = (
        _metrical_tempo_error(estimated_bpm, librosa_bpm)
        if model_result and bpm_override is None and librosa_bpm > 0
        else None
    )
    if tempo_ensemble_error is not None and tempo_ensemble_error > 0.12:
        review_warnings.append(
            "BeatNet and librosa tempo estimates disagree; verify the musical BPM or provide an override."
        )
    if confidence < 0.45:
        review_warnings.append("Beat tracking confidence is low; verify BPM and first beat manually.")
    if stability["interval_cv"] is not None and stability["interval_cv"] > 0.08:
        backend_warnings.append("Detected beat intervals vary; exported a local tempo map for alignment.")
    if (
        grid_mode == "constant_fallback"
        and stability["grid_error_p95_seconds"] is not None
        and stability["grid_error_p95_seconds"] > beat_period * 0.15
    ):
        review_warnings.append("Fallback beats do not fit a constant-tempo grid; provide BPM/first-beat overrides or enable BeatNet.")
    if len(detected_times) < max(2, int(duration / max(beat_period, 1e-6) * 0.35)):
        review_warnings.append("Too few song beats were detected for reliable full-track alignment.")

    tempo_map = _build_tempo_map(beat_grid)
    warnings = backend_warnings + review_warnings

    return {
        "filename": path.name,
        "path": str(path),
        "sample_rate": int(sr),
        "duration_seconds": duration,
        "backend": backend,
        "grid_mode": grid_mode,
        "estimated_bpm": estimated_bpm,
        "tempo_estimates_bpm": {
            "selected": estimated_bpm,
            "beatnet": float(model_result["estimated_bpm"]) if model_result else None,
            "librosa": librosa_bpm,
        },
        "tempo_ensemble_relative_error": tempo_ensemble_error,
        "bpm_overridden": bpm_override is not None,
        "beat_period_seconds": beat_period,
        "first_beat_seconds": first_beat,
        "first_beat_overridden": first_beat_override is not None,
        "phase_offset_seconds": float(first_beat % beat_period),
        "beats_per_bar": meter,
        "beats_per_bar_overridden": beats_per_bar is not None,
        "estimated_meter": estimated_meter,
        "bar_duration_seconds": meter * beat_period,
        "beat_tracking_confidence": confidence,
        "tempo_stability": stability,
        "requires_manual_review": bool(review_warnings),
        "warnings": warnings,
        "detected_beat_times_seconds": [round(float(v), 6) for v in detected_times],
        "detected_beat_numbers": beat_numbers,
        "downbeat_times_seconds": [round(float(v), 6) for v in downbeat_times],
        "beat_grid_times_seconds": [round(float(v), 6) for v in beat_grid],
        "tempo_map": tempo_map,
        "alignment_note": (
            "first_beat_seconds is the loop alignment anchor. Automatic analysis does not "
            "guarantee that it is the musical downbeat; use an override after listening."
        ),
    }


def save_song_analysis(analysis: dict[str, Any], out_dir: str | os.PathLike[str]) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "song_analysis.json"
    beat_grid_path = out / "song_beat_grid.csv"
    detected_path = out / "detected_song_beats.csv"
    tempo_map_path = out / "song_tempo_map.csv"
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        {
            "beat_index": np.arange(len(analysis["beat_grid_times_seconds"])),
            "time_seconds": analysis["beat_grid_times_seconds"],
            "beat_in_bar": [
                int(i % analysis["beats_per_bar"]) + 1
                for i in range(len(analysis["beat_grid_times_seconds"]))
            ],
            "bar_index": [
                int(i // analysis["beats_per_bar"])
                for i in range(len(analysis["beat_grid_times_seconds"]))
            ],
        }
    ).to_csv(beat_grid_path, index=False)
    pd.DataFrame(
        {
            "beat_index": np.arange(len(analysis["detected_beat_times_seconds"])),
            "time_seconds": analysis["detected_beat_times_seconds"],
        }
    ).to_csv(detected_path, index=False)
    pd.DataFrame(analysis.get("tempo_map", [])).to_csv(tempo_map_path, index=False)
    return {
        "song_analysis_json": str(json_path),
        "song_beat_grid_csv": str(beat_grid_path),
        "detected_song_beats_csv": str(detected_path),
        "song_tempo_map_csv": str(tempo_map_path),
    }


def _clean_beat_times(values: Any, duration: float) -> np.ndarray:
    times = np.asarray(values, dtype=np.float64)
    times = times[np.isfinite(times)]
    times = times[(times >= 0.0) & (times <= duration + 1e-6)]
    return np.unique(times)


def _bpm_from_beats(beat_times: np.ndarray) -> float:
    intervals = np.diff(beat_times)
    intervals = intervals[intervals > 1e-4]
    return float(60.0 / np.median(intervals)) if len(intervals) else 0.0


def _metrical_tempo_error(primary_bpm: float, alternative_bpm: float) -> float:
    if primary_bpm <= 0 or alternative_bpm <= 0:
        return float("inf")
    return float(
        min(abs(primary_bpm - alternative_bpm * factor) / primary_bpm for factor in (0.5, 1.0, 2.0))
    )


def _build_tempo_map(beat_times: np.ndarray) -> list[dict[str, float | int]]:
    if len(beat_times) < 2:
        return []
    intervals = np.diff(beat_times.astype(np.float64))
    local_bpm = 60.0 / np.maximum(intervals, 1e-6)
    smoothed = np.asarray(
        [np.median(local_bpm[max(0, i - 2) : min(len(local_bpm), i + 3)]) for i in range(len(local_bpm))],
        dtype=np.float64,
    )
    return [
        {
            "beat_index": int(index),
            "start_seconds": round(float(beat_times[index]), 6),
            "end_seconds": round(float(beat_times[index + 1]), 6),
            "interval_seconds": round(float(intervals[index]), 6),
            "instantaneous_bpm": round(float(local_bpm[index]), 4),
            "smoothed_bpm": round(float(smoothed[index]), 4),
        }
        for index in range(len(intervals))
    ]


def _beat_confidence(
    beat_strengths: np.ndarray,
    onset_envelope: np.ndarray,
    detected_times: np.ndarray,
    duration: float,
    beat_period: float,
) -> float:
    if not len(beat_strengths) or not len(onset_envelope) or duration <= 0:
        return 0.0
    expected_count = max(1.0, duration / beat_period)
    coverage = min(1.0, len(detected_times) / expected_count)
    reference = float(np.percentile(onset_envelope, 90)) + 1e-9
    strength = float(np.clip(np.median(beat_strengths) / reference, 0.0, 1.0))
    return float(np.clip(0.55 * coverage + 0.45 * strength, 0.0, 1.0))


def _tempo_stability(detected_times: np.ndarray, bpm: float, first_beat: float) -> dict[str, float | None]:
    if len(detected_times) < 3 or bpm <= 0:
        return {
            "detected_interval_median_seconds": None,
            "interval_cv": None,
            "grid_error_median_seconds": None,
            "grid_error_p95_seconds": None,
        }
    intervals = np.diff(detected_times.astype(np.float64))
    median_interval = float(np.median(intervals))
    interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
    period = 60.0 / bpm
    grid_indices = np.round((detected_times - first_beat) / period)
    nearest_grid = first_beat + grid_indices * period
    errors = np.abs(detected_times - nearest_grid)
    return {
        "detected_interval_median_seconds": median_interval,
        "interval_cv": interval_cv,
        "grid_error_median_seconds": float(np.median(errors)),
        "grid_error_p95_seconds": float(np.percentile(errors, 95)),
    }
