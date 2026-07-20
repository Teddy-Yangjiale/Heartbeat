from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import librosa
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class RegionEdit:
    start_seconds: float
    end_seconds: float
    label: str = "Region"
    song_gain_db: float = 0.0
    heartbeat_gain_db: float = 0.0
    pulse_mode: str = "inherit"
    fit_mode: str = "inherit"
    fade_ms: float = 80.0


@dataclass(frozen=True)
class MixParams:
    pulse_mode: str = "auto"
    fit_mode: str = "gap"
    beats_per_bar: int = 4
    heartbeat_start_seconds: float = 0.0
    heartbeat_end_seconds: float | None = None
    song_gain_db: float = 0.0
    heartbeat_gain_db: float = 0.0
    auto_balance: bool = True
    song_target_lufs: float = -18.0
    heartbeat_relative_lu: float = 1.0
    ducking_db: float = 2.5
    ducking_cutoff_hz: float = 280.0
    master_target_lufs: float = -16.0
    output_ceiling_dbfs: float = -1.0


PULSE_MODES = {"auto", "bar", "half", "normal", "double", "mute"}
FIT_MODES = {"gap", "stretch"}


def analyze_song_bytes(
    filename: str,
    data: bytes,
    *,
    manual_bpm: float | None = None,
    manual_first_beat: float | None = None,
    force_constant_grid: bool = False,
) -> dict[str, Any]:
    audio, sample_rate, source = read_song_bytes(filename, data)
    duration = len(audio) / sample_rate
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    analysis_rate = min(sample_rate, 22050)
    analysis_audio = resample_audio(mono[:, None], sample_rate, analysis_rate)[:, 0]
    hop_length = 512
    onset = librosa.onset.onset_strength(
        y=analysis_audio,
        sr=analysis_rate,
        hop_length=hop_length,
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=analysis_rate,
        hop_length=hop_length,
        units="frames",
    )
    detected = librosa.frames_to_time(
        beat_frames,
        sr=analysis_rate,
        hop_length=hop_length,
    ).astype(np.float64)
    detected = _clean_beat_grid(detected, duration)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset,
        sr=analysis_rate,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
    )
    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=analysis_rate,
        hop_length=hop_length,
    ).astype(np.float64)
    detected, inserted_beats = _repair_missing_beats(detected, onset_times)
    detected_bpm = _bpm_from_beats(detected)
    librosa_bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if detected_bpm <= 0:
        detected_bpm = librosa_bpm
    selected_bpm = float(manual_bpm) if manual_bpm is not None else detected_bpm
    if not np.isfinite(selected_bpm) or selected_bpm <= 0:
        raise ValueError("Could not estimate a valid song BPM. Enter a manual BPM.")

    first_detected = float(detected[0]) if len(detected) else 0.0
    first_beat = (
        max(0.0, float(manual_first_beat))
        if manual_first_beat is not None
        else first_detected
    )
    use_constant = bool(force_constant_grid or manual_bpm is not None)
    warnings: list[str] = []
    if inserted_beats:
        warnings.append(
            f"Inserted {inserted_beats} locally supported missing beat(s) into the dynamic grid."
        )
    if use_constant or len(detected) < 3:
        beat_grid = _constant_grid(duration, selected_bpm, first_beat)
        grid_mode = "constant_manual" if manual_bpm is not None else "constant_fallback"
        if len(detected) < 3:
            warnings.append("Too few song beats were detected; a constant grid was used.")
    else:
        shift = first_beat - first_detected
        beat_grid = detected + shift
        beat_grid = beat_grid[(beat_grid >= 0.0) & (beat_grid <= duration)]
        grid_mode = "dynamic_librosa"

    beat_strengths = onset[
        np.clip(np.asarray(beat_frames, dtype=int), 0, max(0, len(onset) - 1))
    ] if len(onset) else np.asarray([], dtype=np.float32)
    expected = max(1.0, duration * selected_bpm / 60.0)
    coverage = min(1.0, len(detected) / expected)
    reference = float(np.percentile(onset, 90)) + 1e-9 if len(onset) else 1.0
    strength = (
        float(np.clip(np.median(beat_strengths) / reference, 0.0, 1.0))
        if len(beat_strengths)
        else 0.0
    )
    confidence = float(np.clip(0.55 * coverage + 0.45 * strength, 0.0, 1.0))
    intervals = np.diff(detected)
    interval_cv = (
        float(np.std(intervals) / np.mean(intervals))
        if len(intervals) >= 2 and float(np.mean(intervals)) > 1e-6
        else None
    )
    if confidence < 0.45:
        warnings.append("Beat confidence is low; verify BPM and the first beat with the click track.")
    if interval_cv is not None and interval_cv > 0.12:
        warnings.append("The song has variable or uncertain beat intervals; use the dynamic grid or edit the anchor.")

    return {
        "filename": filename,
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "duration_seconds": float(duration),
        "source": source,
        "backend": "librosa",
        "grid_mode": grid_mode,
        "estimated_bpm": float(selected_bpm),
        "detected_bpm": float(detected_bpm),
        "first_beat_seconds": float(first_beat),
        "manual_bpm": manual_bpm,
        "manual_first_beat": manual_first_beat,
        "beat_tracking_confidence": confidence,
        "detected_interval_cv": interval_cv,
        "detected_beat_times_seconds": detected.tolist(),
        "beat_grid_times_seconds": beat_grid.tolist(),
        "warnings": warnings,
        "audio": audio,
    }


def process_music_bytes(
    song_filename: str,
    song_data: bytes,
    heartbeat_result: dict[str, Any],
    song_analysis: dict[str, Any],
    params: MixParams | None = None,
    region_edits: list[RegionEdit] | None = None,
    *,
    render_duration_seconds: float | None = None,
) -> dict[str, Any]:
    params = params or MixParams()
    edits = validate_region_edits(region_edits or [], song_analysis["duration_seconds"])
    song, sample_rate, _ = read_song_bytes(song_filename, song_data)
    full_duration = len(song) / sample_rate
    render_duration = (
        min(full_duration, max(1.0, float(render_duration_seconds)))
        if render_duration_seconds is not None
        else full_duration
    )
    target_samples = min(len(song), int(round(render_duration * sample_rate)))
    song = song[:target_samples].astype(np.float32, copy=True)
    duration = len(song) / sample_rate
    beat_grid = np.asarray(song_analysis["beat_grid_times_seconds"], dtype=np.float64)
    beat_grid = beat_grid[(beat_grid >= 0.0) & (beat_grid < duration + 1e-6)]
    if len(beat_grid) < 2:
        beat_grid = _constant_grid(
            duration,
            float(song_analysis["estimated_bpm"]),
            float(song_analysis["first_beat_seconds"]),
        )
    if len(beat_grid) < 2:
        raise ValueError("The song needs at least two usable beat positions.")

    cycles, heartbeat_source_rate = extract_heartbeat_cycles(heartbeat_result)
    cycles = [resample_audio(cycle, heartbeat_source_rate, sample_rate) for cycle in cycles]
    song_bpm = float(song_analysis["estimated_bpm"])
    heartbeat_bpm = float(
        heartbeat_result["cleanest_segment"].get("local_bpm")
        or heartbeat_result["summary"]["tempo"]["estimated_bpm"]
    )
    default_mode = resolve_auto_mode(
        params.pulse_mode,
        song_bpm,
        heartbeat_bpm,
        params.beats_per_bar,
    )
    schedule = build_region_schedule(
        beat_grid,
        duration,
        default_mode,
        params.fit_mode,
        params.beats_per_bar,
        params.heartbeat_start_seconds,
        params.heartbeat_end_seconds,
        edits,
    )
    heartbeat_raw = render_heartbeat_layer(
        cycles,
        schedule,
        sample_rate,
        song.shape[1],
        len(song),
    )

    song_automation = region_gain_envelope(len(song), sample_rate, edits, "song_gain_db")
    heartbeat_automation = region_gain_envelope(
        len(song), sample_rate, edits, "heartbeat_gain_db"
    )
    song_track = song * song_automation[:, None]
    heartbeat_track = heartbeat_raw * heartbeat_automation[:, None]

    song_stats_before = analyze_loudness(song_track, sample_rate)
    heartbeat_stats_before = analyze_loudness(heartbeat_track, sample_rate)
    song_gain_db = float(params.song_gain_db)
    heartbeat_gain_db = float(params.heartbeat_gain_db)
    if params.auto_balance:
        if np.isfinite(song_stats_before["integrated_lufs"]):
            song_gain_db += float(params.song_target_lufs) - song_stats_before["integrated_lufs"]
        if np.isfinite(heartbeat_stats_before["active_lufs"]):
            song_reference = song_stats_before["active_lufs"] + song_gain_db
            heartbeat_gain_db += (
                song_reference
                + float(params.heartbeat_relative_lu)
                - heartbeat_stats_before["active_lufs"]
            )
    song_gain_db = float(np.clip(song_gain_db, -24.0, 18.0))
    heartbeat_gain_db = float(np.clip(heartbeat_gain_db, -24.0, 30.0))
    song_track *= db_to_gain(song_gain_db)
    heartbeat_track *= db_to_gain(heartbeat_gain_db)

    ducked_song, duck_report = frequency_selective_duck(
        song_track,
        heartbeat_track,
        sample_rate,
        depth_db=params.ducking_db,
        cutoff_hz=params.ducking_cutoff_hz,
    )
    raw_mix = ducked_song + heartbeat_track
    mastered, master_report = master_mix(
        raw_mix,
        sample_rate,
        target_lufs=params.master_target_lufs,
        ceiling_dbfs=params.output_ceiling_dbfs,
    )
    applied_master_gain = db_to_gain(master_report["applied_gain_db"])
    exported_song = ducked_song * applied_master_gain
    exported_heartbeat = heartbeat_track * applied_master_gain

    click_mix = make_click_mix(
        song_track,
        beat_grid,
        sample_rate,
        params.beats_per_bar,
    )
    timeline = pd.DataFrame(schedule)
    report = {
        "song": {
            key: value
            for key, value in song_analysis.items()
            if key not in {"audio", "detected_beat_times_seconds", "beat_grid_times_seconds"}
        },
        "heartbeat": {
            "quality": heartbeat_result["recording_quality"],
            "estimated_bpm": heartbeat_bpm,
            "selected_cycle_count": len(cycles),
            "source_artifact": "cleanest_heartbeat_loop.wav",
        },
        "render": {
            "duration_seconds": duration,
            "is_preview": render_duration_seconds is not None and duration < full_duration - 1e-6,
            "pulse_count": len(schedule),
            "pulse_mode_requested": params.pulse_mode,
            "pulse_mode_resolved": default_mode,
            "fit_mode": params.fit_mode,
            "beats_per_bar": params.beats_per_bar,
            "song_gain_db": song_gain_db,
            "heartbeat_gain_db": heartbeat_gain_db,
            "regions": [asdict(edit) for edit in edits],
        },
        "ducking": duck_report,
        "master": master_report,
        "parameters": asdict(params),
    }
    artifacts = {
        "final_mix.wav": wav_bytes(sample_rate, mastered),
        "heartbeat_aligned.wav": wav_bytes(sample_rate, exported_heartbeat),
        "song_processed.wav": wav_bytes(sample_rate, exported_song),
        "debug_click_mix.wav": wav_bytes(sample_rate, click_mix),
        "mix_report.json": json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
        "heartbeat_timeline.csv": timeline.to_csv(index=False).encode("utf-8"),
        "region_edits.json": json.dumps(
            [asdict(edit) for edit in edits], indent=2, ensure_ascii=False
        ).encode("utf-8"),
    }
    return {
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "report": report,
        "schedule": schedule,
        "artifacts": artifacts,
        "zip_bytes": make_zip(artifacts),
    }


def read_song_bytes(filename: str, data: bytes) -> tuple[np.ndarray, int, dict[str, Any]]:
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in {"wav", "mp3"}:
        raise ValueError("Song input must be a WAV or MP3 file.")
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not decode {extension.upper()} song audio: {exc}") from exc
    if sample_rate <= 0 or not len(audio):
        raise ValueError("The song audio is empty or has an invalid sample rate.")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    return np.asarray(audio, dtype=np.float32), int(sample_rate), {
        "filename": filename,
        "format": extension,
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
    }


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if source_rate == target_rate:
        return data.copy()
    ratio = Fraction(int(target_rate), int(source_rate)).limit_denominator(1000)
    return signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def _clean_beat_grid(values: np.ndarray, duration: float) -> np.ndarray:
    beats = np.unique(np.asarray(values, dtype=np.float64))
    beats = beats[np.isfinite(beats) & (beats >= 0.0) & (beats <= duration)]
    if len(beats) < 3:
        return beats
    intervals = np.diff(beats)
    typical = float(np.median(intervals[intervals > 1e-4]))
    output = [float(beats[0])]
    for value in beats[1:]:
        if value - output[-1] >= typical * 0.42:
            output.append(float(value))
    return np.asarray(output, dtype=np.float64)


def _bpm_from_beats(beats: np.ndarray) -> float:
    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.15) & (intervals < 3.0)]
    return float(60.0 / np.median(intervals)) if len(intervals) else 0.0


def _repair_missing_beats(
    beats: np.ndarray,
    onset_times: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Repair only omissions supported by stable local tempo and nearby onsets."""
    values = np.asarray(beats, dtype=np.float64)
    supports = np.asarray(onset_times, dtype=np.float64)
    if len(values) < 6:
        return values, 0
    inserted: list[float] = []
    for index, (left_beat, right_beat) in enumerate(zip(values[:-1], values[1:])):
        before = np.diff(values[max(0, index - 4) : index + 1])
        after = np.diff(values[index + 1 : min(len(values), index + 6)])
        before = before[before > 1e-4]
        after = after[after > 1e-4]
        if not len(before) or not len(after):
            continue
        left_period = float(np.median(before))
        right_period = float(np.median(after))
        local_period = float(np.median([left_period, right_period]))
        if abs(left_period - right_period) / max(local_period, 1e-9) > 0.15:
            continue
        gap = float(right_beat - left_beat)
        subdivisions = int(round(gap / max(local_period, 1e-9)))
        if subdivisions < 2 or subdivisions > 4:
            continue
        if abs(gap / subdivisions - local_period) / max(local_period, 1e-9) > 0.10:
            continue
        predicted = [left_beat + gap * part / subdivisions for part in range(1, subdivisions)]
        tolerance = max(0.08, 0.20 * local_period)
        supported = sum(
            bool(np.any(np.abs(supports - prediction) <= tolerance))
            for prediction in predicted
        )
        if supported / max(1, len(predicted)) >= 0.60:
            inserted.extend(predicted)
    if not inserted:
        return values, 0
    repaired = np.unique(np.concatenate([values, np.asarray(inserted, dtype=np.float64)]))
    return repaired, len(repaired) - len(values)


def _constant_grid(duration: float, bpm: float, first_beat: float) -> np.ndarray:
    if duration <= 0 or bpm <= 0:
        return np.asarray([], dtype=np.float64)
    return np.arange(max(0.0, first_beat), duration + 1e-9, 60.0 / bpm, dtype=np.float64)


def validate_region_edits(edits: list[RegionEdit], duration: float) -> list[RegionEdit]:
    normalized: list[RegionEdit] = []
    for edit in edits:
        start = max(0.0, float(edit.start_seconds))
        end = min(float(duration), float(edit.end_seconds))
        if end <= start:
            raise ValueError(f"Region '{edit.label}' must end after it starts.")
        if edit.pulse_mode not in PULSE_MODES | {"inherit"}:
            raise ValueError(f"Unknown pulse mode in region '{edit.label}': {edit.pulse_mode}")
        if edit.fit_mode not in FIT_MODES | {"inherit"}:
            raise ValueError(f"Unknown fit mode in region '{edit.label}': {edit.fit_mode}")
        normalized.append(
            RegionEdit(
                start_seconds=start,
                end_seconds=end,
                label=edit.label or "Region",
                song_gain_db=float(np.clip(edit.song_gain_db, -60.0, 18.0)),
                heartbeat_gain_db=float(np.clip(edit.heartbeat_gain_db, -60.0, 30.0)),
                pulse_mode=edit.pulse_mode,
                fit_mode=edit.fit_mode,
                fade_ms=float(np.clip(edit.fade_ms, 0.0, 5000.0)),
            )
        )
    normalized.sort(key=lambda item: (item.start_seconds, item.end_seconds))
    for left, right in zip(normalized, normalized[1:]):
        if right.start_seconds < left.end_seconds - 1e-6:
            raise ValueError(f"Regions '{left.label}' and '{right.label}' overlap.")
    return normalized


def resolve_auto_mode(
    requested: str,
    song_bpm: float,
    heartbeat_bpm: float,
    beats_per_bar: int,
) -> str:
    if requested != "auto":
        return requested
    candidates = {
        "bar": 1.0 / max(1, beats_per_bar),
        "half": 0.5,
        "normal": 1.0,
        "double": 2.0,
    }
    def score(item: tuple[str, float]) -> float:
        _, factor = item
        pulse_bpm = song_bpm * factor
        range_penalty = 0.0 if 45.0 <= pulse_bpm <= 130.0 else 2.0
        return range_penalty + abs(math.log2(max(pulse_bpm, 1e-6) / max(heartbeat_bpm, 1e-6)))
    return min(candidates.items(), key=score)[0]


def _pulse_grid(beats: np.ndarray, mode: str, beats_per_bar: int) -> np.ndarray:
    if mode == "mute" or len(beats) < 2:
        return np.asarray([], dtype=np.float64)
    if mode in {"bar", "half"}:
        stride = max(1, beats_per_bar if mode == "bar" else 2)
        return beats[np.arange(len(beats)) % stride == 0]
    if mode == "normal":
        return beats.copy()
    if mode == "double":
        values: list[float] = []
        for left, right in zip(beats[:-1], beats[1:]):
            values.extend([float(left), float((left + right) * 0.5)])
        values.append(float(beats[-1]))
        return np.asarray(values, dtype=np.float64)
    raise ValueError(f"Unsupported resolved pulse mode: {mode}")


def build_region_schedule(
    beats: np.ndarray,
    duration: float,
    default_mode: str,
    default_fit: str,
    beats_per_bar: int,
    heartbeat_start: float,
    heartbeat_end: float | None,
    edits: list[RegionEdit],
) -> list[dict[str, Any]]:
    start = max(0.0, float(heartbeat_start))
    end = min(duration, float(heartbeat_end) if heartbeat_end is not None else duration)
    if end <= start:
        return []
    boundaries = {start, end}
    for edit in edits:
        boundaries.add(max(start, edit.start_seconds))
        boundaries.add(min(end, edit.end_seconds))
    points = sorted(value for value in boundaries if start <= value <= end)
    schedule: list[dict[str, Any]] = []
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        midpoint = (left + right) * 0.5
        active = next(
            (edit for edit in edits if edit.start_seconds <= midpoint < edit.end_seconds),
            None,
        )
        mode = active.pulse_mode if active and active.pulse_mode != "inherit" else default_mode
        fit = active.fit_mode if active and active.fit_mode != "inherit" else default_fit
        label = active.label if active else "Global"
        for pulse in _pulse_grid(beats, mode, beats_per_bar):
            if left - 1e-7 <= pulse < right - 1e-7:
                schedule.append(
                    {
                        "pulse_index": 0,
                        "time_seconds": float(pulse),
                        "pulse_mode": mode,
                        "fit_mode": fit,
                        "region": label,
                    }
                )
    schedule.sort(key=lambda item: item["time_seconds"])
    deduplicated: list[dict[str, Any]] = []
    for item in schedule:
        if deduplicated and abs(item["time_seconds"] - deduplicated[-1]["time_seconds"]) < 1e-4:
            deduplicated[-1] = item
        else:
            deduplicated.append(item)
    for index, item in enumerate(deduplicated):
        item["pulse_index"] = index
    return deduplicated


def extract_heartbeat_cycles(result: dict[str, Any]) -> tuple[list[np.ndarray], int]:
    audio = np.asarray(result["cleanest_audio"], dtype=np.float32)
    sample_rate = int(result["sample_rate"])
    segment = result["cleanest_segment"]
    start = float(segment["adjusted_start_seconds"])
    end = float(segment["adjusted_end_seconds"])
    beats = np.asarray(result["beat_times"], dtype=np.float64)
    internal = beats[(beats > start + 0.025) & (beats < end - 0.025)] - start
    boundaries = np.concatenate(([0.0], internal, [len(audio) / sample_rate]))
    boundaries = np.unique(np.clip(boundaries, 0.0, len(audio) / sample_rate))
    cycles: list[np.ndarray] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        begin = int(round(left * sample_rate))
        finish = int(round(right * sample_rate))
        if finish - begin >= int(0.2 * sample_rate):
            cycles.append(_edge_fade(audio[begin:finish, None], sample_rate, 5.0))
    expected = max(1, int(segment.get("cycle_count") or 1))
    if len(cycles) < 2:
        indices = np.linspace(0, len(audio), expected + 1, dtype=int)
        cycles = [
            _edge_fade(audio[indices[i] : indices[i + 1], None], sample_rate, 5.0)
            for i in range(expected)
            if indices[i + 1] > indices[i]
        ]
    if not cycles:
        raise ValueError("No usable heartbeat cycle was found in the selected clean loop.")
    return cycles, sample_rate


def render_heartbeat_layer(
    cycles: list[np.ndarray],
    schedule: list[dict[str, Any]],
    sample_rate: int,
    channels: int,
    output_samples: int,
) -> np.ndarray:
    output = np.zeros((output_samples, channels), dtype=np.float32)
    if not schedule:
        return output
    times = np.asarray([item["time_seconds"] for item in schedule], dtype=np.float64)
    intervals = np.diff(times)
    fallback = float(np.median(intervals)) if len(intervals) else 60.0 / 75.0
    for index, item in enumerate(schedule):
        interval = float(intervals[index]) if index < len(intervals) else fallback
        target_samples = max(1, int(round(interval * sample_rate)))
        cycle = cycles[index % len(cycles)]
        if cycle.shape[1] == 1 and channels > 1:
            cycle = np.repeat(cycle, channels, axis=1)
        elif cycle.shape[1] != channels:
            cycle = np.repeat(np.mean(cycle, axis=1, keepdims=True), channels, axis=1)
        fitted = fit_cycle(cycle, target_samples, sample_rate, item["fit_mode"])
        begin = int(round(item["time_seconds"] * sample_rate))
        finish = min(output_samples, begin + len(fitted))
        if begin < output_samples and finish > begin:
            output[begin:finish] += fitted[: finish - begin]
    return output


def fit_cycle(cycle: np.ndarray, target_samples: int, sample_rate: int, mode: str) -> np.ndarray:
    if mode == "gap" and len(cycle) <= target_samples:
        output = np.zeros((target_samples, cycle.shape[1]), dtype=np.float32)
        output[: len(cycle)] = cycle
        return output
    rate = len(cycle) / max(1, target_samples)
    stretched_channels = [
        librosa.effects.time_stretch(cycle[:, channel], rate=rate)
        for channel in range(cycle.shape[1])
    ]
    output = np.zeros((target_samples, cycle.shape[1]), dtype=np.float32)
    for channel, values in enumerate(stretched_channels):
        length = min(target_samples, len(values))
        output[:length, channel] = values[:length]
    return _edge_fade(output, sample_rate, 4.0)


def region_gain_envelope(
    sample_count: int,
    sample_rate: int,
    edits: list[RegionEdit],
    field: str,
) -> np.ndarray:
    envelope = np.ones(sample_count, dtype=np.float32)
    for edit in edits:
        value_db = float(getattr(edit, field))
        if abs(value_db) < 1e-9:
            continue
        begin = max(0, min(sample_count, int(round(edit.start_seconds * sample_rate))))
        finish = max(begin, min(sample_count, int(round(edit.end_seconds * sample_rate))))
        if finish <= begin:
            continue
        target = db_to_gain(value_db)
        envelope[begin:finish] = target
        fade = min(
            (finish - begin) // 2,
            max(0, int(round(edit.fade_ms * sample_rate / 1000.0))),
        )
        if fade > 1:
            envelope[begin : begin + fade] = np.linspace(1.0, target, fade, dtype=np.float32)
            envelope[finish - fade : finish] = np.linspace(target, 1.0, fade, dtype=np.float32)
    return envelope


def analyze_loudness(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    data = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if not data.size or peak <= 1e-9:
        return {"integrated_lufs": float("-inf"), "active_lufs": float("-inf"), "peak_dbfs": -240.0}
    minimum = max(1, int(round(0.4 * sample_rate)))
    if len(data) < minimum:
        padded = np.zeros((minimum, data.shape[1]), dtype=np.float32)
        padded[: len(data)] = data
        data = padded
    meter = pyln.Meter(sample_rate, block_size=0.4)
    integrated = float(meter.integrated_loudness(data))
    blocks = np.asarray(meter.blockwise_loudness, dtype=float)
    finite = blocks[np.isfinite(blocks) & (blocks > -70.0)]
    active = float(np.percentile(finite, 75.0)) if len(finite) else integrated
    return {
        "integrated_lufs": integrated,
        "active_lufs": active,
        "peak_dbfs": float(20.0 * np.log10(max(peak, 1e-12))),
    }


def frequency_selective_duck(
    song: np.ndarray,
    heartbeat: np.ndarray,
    sample_rate: int,
    *,
    depth_db: float,
    cutoff_hz: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if depth_db <= 0 or not heartbeat.size or float(np.max(np.abs(heartbeat))) <= 1e-9:
        return song.copy(), {"depth_db": 0.0, "cutoff_hz": float(cutoff_hz)}
    trigger = np.max(np.abs(heartbeat), axis=1)
    window = max(1, int(round(0.12 * sample_rate)))
    envelope = signal.lfilter(np.ones(window) / window, [1.0], trigger)
    high = float(np.percentile(envelope[envelope > 1e-9], 95.0)) if np.any(envelope > 1e-9) else 0.0
    normalized = np.clip(envelope / max(high, 1e-9), 0.0, 1.0)
    gain = np.power(10.0, (-abs(depth_db) * np.power(normalized, 0.65)) / 20.0).astype(np.float32)
    cutoff = float(np.clip(cutoff_hz, 40.0, sample_rate * 0.45))
    sos = signal.butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    low = signal.sosfiltfilt(sos, song, axis=0).astype(np.float32)
    return (song - low + low * gain[:, None]).astype(np.float32), {
        "depth_db": float(depth_db),
        "cutoff_hz": cutoff,
        "mean_active_duck_db": float(-20.0 * np.log10(max(np.mean(gain[normalized > 0.05]), 1e-9)))
        if np.any(normalized > 0.05)
        else 0.0,
    }


def master_mix(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float,
    ceiling_dbfs: float,
) -> tuple[np.ndarray, dict[str, float]]:
    before = analyze_loudness(audio, sample_rate)
    requested = target_lufs - before["integrated_lufs"] if np.isfinite(before["integrated_lufs"]) else 0.0
    loudness_gain_db = float(np.clip(requested, -18.0, 12.0))
    output = audio * db_to_gain(loudness_gain_db)
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    ceiling = db_to_gain(ceiling_dbfs)
    peak_scale = ceiling / peak if peak > ceiling else 1.0
    output = np.asarray(output * peak_scale, dtype=np.float32)
    applied_gain_db = loudness_gain_db + 20.0 * math.log10(max(peak_scale, 1e-12))
    after = analyze_loudness(output, sample_rate)
    return output, {
        "target_lufs": float(target_lufs),
        "ceiling_dbfs": float(ceiling_dbfs),
        "input_lufs": before["integrated_lufs"],
        "requested_loudness_gain_db": loudness_gain_db,
        "peak_protection_db": float(20.0 * math.log10(max(peak_scale, 1e-12))),
        "applied_gain_db": float(applied_gain_db),
        "output_lufs": after["integrated_lufs"],
        "output_peak_dbfs": after["peak_dbfs"],
    }


def make_click_mix(
    song: np.ndarray,
    beats: np.ndarray,
    sample_rate: int,
    beats_per_bar: int,
) -> np.ndarray:
    output = np.asarray(song * 0.72, dtype=np.float32)
    length = max(1, int(round(0.035 * sample_rate)))
    t = np.arange(length) / sample_rate
    for index, beat in enumerate(beats):
        frequency = 1400.0 if index % max(1, beats_per_bar) == 0 else 900.0
        click = (0.18 * np.sin(2.0 * np.pi * frequency * t) * np.exp(-t * 75.0)).astype(np.float32)
        begin = int(round(beat * sample_rate))
        finish = min(len(output), begin + length)
        if begin < len(output) and finish > begin:
            output[begin:finish] += click[: finish - begin, None]
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    return output / max(1.0, peak / db_to_gain(-1.0))


def _edge_fade(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).copy()
    fade = min(len(output) // 2, max(1, int(round(fade_ms * sample_rate / 1000.0))))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        output[:fade] *= ramp[:, None]
        output[-fade:] *= ramp[::-1, None]
    return output


def db_to_gain(value_db: float) -> float:
    return float(10.0 ** (float(value_db) / 20.0))


def wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.asarray(audio, dtype=np.float32).clip(-1.0, 1.0),
        sample_rate,
        format="WAV",
        subtype="PCM_24",
    )
    return buffer.getvalue()


def make_zip(artifacts: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in artifacts.items():
            archive.writestr(name, value)
    return buffer.getvalue()
