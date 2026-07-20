from __future__ import annotations

import io
import json
import logging
import math
import os
import zipfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

import librosa
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import psutil
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
    pulse_min_bpm: float = 55.0
    pulse_max_bpm: float = 110.0
    timing_offset_ms: float = 0.0
    section_adaptive_strength: float = 0.65
    heartbeat_fade_in_seconds: float = 4.0
    heartbeat_fade_out_seconds: float = 5.0
    max_stretch_ratio: float = 1.18


@dataclass(frozen=True)
class HeartbeatCycle:
    audio: np.ndarray
    anchor_offset_samples: int
    active_samples: int
    source_cycle_index: int
    anchor_mode: str = "s1-onset"


PULSE_MODES = {
    "auto",
    "downbeat",
    "kick",
    "backbeat",
    "every-beat",
    "bar",
    "half",
    "normal",
    "double",
    "mute",
}
FIT_MODES = {"gap", "stretch"}


def _rss_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))


def _record_memory(trace: dict[str, float], stage: str) -> None:
    value = _rss_mb()
    trace[stage] = value
    logging.getLogger(__name__).info("heartbeat-render stage=%s rss_mb=%.1f", stage, value)


def analyze_song_bytes(
    filename: str,
    data: bytes | memoryview | BinaryIO,
    *,
    manual_bpm: float | None = None,
    manual_first_beat: float | None = None,
    manual_first_downbeat: float | None = None,
    manual_meter: int = 4,
    force_constant_grid: bool = False,
    max_duration_seconds: float | None = None,
) -> dict[str, Any]:
    audio, sample_rate, source = read_song_bytes(
        filename,
        data,
        max_duration_seconds=max_duration_seconds,
    )
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

    meter = int(np.clip(manual_meter, 2, 12))
    beat_energy, beat_low_strength = _beat_features(
        analysis_audio,
        analysis_rate,
        beat_grid,
        onset,
        hop_length,
    )
    downbeats, downbeat_confidence, downbeat_source = _infer_downbeats(
        beat_grid,
        beat_low_strength,
        meter,
        manual_first_downbeat,
    )
    overview_count = min(10000, len(mono))
    if overview_count:
        overview_indices = np.linspace(0, len(mono) - 1, overview_count, dtype=int)
        overview_values = mono[overview_indices].astype(np.float32, copy=False)
        overview_times = overview_indices.astype(np.float64) / sample_rate
    else:
        overview_values = np.asarray([], dtype=np.float32)
        overview_times = np.asarray([], dtype=np.float64)

    result = {
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
        "manual_first_downbeat": manual_first_downbeat,
        "meter": meter,
        "beat_tracking_confidence": confidence,
        "downbeat_confidence": downbeat_confidence,
        "downbeat_source": downbeat_source,
        "detected_interval_cv": interval_cv,
        "detected_beat_times_seconds": detected.tolist(),
        "beat_grid_times_seconds": beat_grid.tolist(),
        "downbeat_times_seconds": downbeats.tolist(),
        "beat_energy": beat_energy.tolist(),
        "beat_low_strength": beat_low_strength.tolist(),
        "waveform_overview_times_seconds": overview_times.tolist(),
        "waveform_overview_values": overview_values.tolist(),
        "warnings": warnings,
    }
    del audio, mono, analysis_audio
    return result


def process_music_bytes(
    song_filename: str,
    song_data: bytes | memoryview | BinaryIO,
    heartbeat_result: dict[str, Any],
    song_analysis: dict[str, Any],
    params: MixParams | None = None,
    region_edits: list[RegionEdit] | None = None,
    *,
    render_duration_seconds: float | None = None,
    output_dir: str | Path | None = None,
    export_stems: bool | None = None,
    export_debug: bool | None = None,
    create_zip: bool | None = None,
) -> dict[str, Any]:
    params = params or MixParams()
    memory_trace: dict[str, float] = {}
    _record_memory(memory_trace, "start")
    disk_output = Path(output_dir) if output_dir is not None else None
    if disk_output is not None:
        disk_output.mkdir(parents=True, exist_ok=True)
    export_stems = (disk_output is None) if export_stems is None else bool(export_stems)
    export_debug = (disk_output is None) if export_debug is None else bool(export_debug)
    create_zip = (disk_output is None) if create_zip is None else bool(create_zip)
    edits = validate_region_edits(region_edits or [], song_analysis["duration_seconds"])
    song, sample_rate, _ = read_song_bytes(song_filename, song_data)
    _record_memory(memory_trace, "song_decoded")
    full_duration = len(song) / sample_rate
    render_duration = (
        min(full_duration, max(1.0, float(render_duration_seconds)))
        if render_duration_seconds is not None
        else full_duration
    )
    target_samples = min(len(song), int(round(render_duration * sample_rate)))
    song = song[:target_samples]
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
    if heartbeat_source_rate != sample_rate:
        scale = sample_rate / heartbeat_source_rate
        cycles = [
            HeartbeatCycle(
                audio=resample_audio(cycle.audio, heartbeat_source_rate, sample_rate),
                anchor_offset_samples=int(round(cycle.anchor_offset_samples * scale)),
                active_samples=int(round(cycle.active_samples * scale)),
                source_cycle_index=cycle.source_cycle_index,
                anchor_mode=cycle.anchor_mode,
            )
            for cycle in cycles
        ]
    song_bpm = float(song_analysis["estimated_bpm"])
    heartbeat_bpm = float(
        heartbeat_result["cleanest_segment"].get("local_bpm")
        or heartbeat_result["summary"]["tempo"]["estimated_bpm"]
    )
    default_mode = params.pulse_mode
    downbeats = np.asarray(song_analysis.get("downbeat_times_seconds", []), dtype=np.float64)
    beat_energy = np.asarray(song_analysis.get("beat_energy", []), dtype=np.float32)
    active_duration = max(cycle.active_samples for cycle in cycles) / sample_rate
    schedule = build_region_schedule(
        beat_grid,
        duration,
        default_mode,
        params.fit_mode,
        params.beats_per_bar,
        params.heartbeat_start_seconds,
        params.heartbeat_end_seconds,
        edits,
        heartbeat_bpm=heartbeat_bpm,
        downbeats=downbeats,
        beat_energy=beat_energy,
        active_duration_seconds=active_duration,
        pulse_min_bpm=params.pulse_min_bpm,
        pulse_max_bpm=params.pulse_max_bpm,
        timing_offset_seconds=params.timing_offset_ms / 1000.0,
        section_adaptive_strength=params.section_adaptive_strength,
    )
    heartbeat_raw, anchor_report = render_heartbeat_layer(
        cycles,
        schedule,
        sample_rate,
        song.shape[1],
        len(song),
        max_stretch_ratio=params.max_stretch_ratio,
    )
    _record_memory(memory_trace, "heartbeat_rendered")

    song_automation = region_gain_envelope(len(song), sample_rate, edits, "song_gain_db")
    heartbeat_automation = region_gain_envelope(
        len(song), sample_rate, edits, "heartbeat_gain_db"
    )
    heartbeat_automation *= arrangement_fade_envelope(
        len(song),
        sample_rate,
        params.heartbeat_start_seconds,
        params.heartbeat_end_seconds,
        params.heartbeat_fade_in_seconds,
        params.heartbeat_fade_out_seconds,
    )
    song *= song_automation[:, None]
    heartbeat_raw *= heartbeat_automation[:, None]
    del song_automation, heartbeat_automation
    song_track = song
    heartbeat_track = heartbeat_raw

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
    stem_song_source = ducked_song.copy() if export_stems else None
    raw_mix = ducked_song
    raw_mix += heartbeat_track
    mastered, master_report = master_mix(
        raw_mix,
        sample_rate,
        target_lufs=params.master_target_lufs,
        ceiling_dbfs=params.output_ceiling_dbfs,
    )
    _record_memory(memory_trace, "mix_mastered")
    applied_master_gain = db_to_gain(master_report["applied_gain_db"])
    exported_song = exported_heartbeat = None
    if export_stems:
        assert stem_song_source is not None
        exported_song = stem_song_source * applied_master_gain
        exported_heartbeat = heartbeat_track * applied_master_gain

    _record_memory(memory_trace, "pre_export")
    timeline = pd.DataFrame(schedule)
    report = {
        "song": {
            key: value
            for key, value in song_analysis.items()
            if key
            not in {
                "audio",
                "detected_beat_times_seconds",
                "beat_grid_times_seconds",
                "waveform_overview_times_seconds",
                "waveform_overview_values",
            }
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
            "pulse_mode_resolved": "adaptive" if default_mode == "auto" else default_mode,
            "fit_mode": params.fit_mode,
            "beats_per_bar": params.beats_per_bar,
            "song_gain_db": song_gain_db,
            "heartbeat_gain_db": heartbeat_gain_db,
            "model_backed_pulse_count": int(sum(bool(item.get("model_backed", True)) for item in schedule)),
            "guide_pulse_count": int(sum(not bool(item.get("model_backed", True)) for item in schedule)),
            "guide_constraint_relaxation_count": int(
                max((item.get("guide_constraint_relaxations", 0) for item in schedule), default=0)
            ),
            "anchor_mode": "s1-onset",
            "anchor_offsets_ms": anchor_report["anchor_offsets_ms"],
            "maximum_anchor_alignment_error_ms": anchor_report["maximum_error_ms"],
            "skipped_anchor_count": anchor_report["skipped_count"],
            "regions": [asdict(edit) for edit in edits],
        },
        "ducking": duck_report,
        "master": master_report,
        "memory": {
            "rss_mb_by_stage": memory_trace,
            "peak_observed_mb": float(max(memory_trace.values(), default=0.0)),
        },
        "parameters": asdict(params),
    }
    metadata_artifacts = {
        "mix_report.json": json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
        "heartbeat_timeline.csv": timeline.to_csv(index=False).encode("utf-8"),
        "region_edits.json": json.dumps(
            [asdict(edit) for edit in edits], indent=2, ensure_ascii=False
        ).encode("utf-8"),
    }
    audio_artifacts: dict[str, np.ndarray] = {"final_mix.wav": mastered}
    if export_stems:
        assert exported_heartbeat is not None and exported_song is not None
        audio_artifacts.update(
            {
                "heartbeat_aligned.wav": exported_heartbeat,
                "song_processed.wav": exported_song,
            }
        )
    if export_debug:
        audio_artifacts["debug_click_mix.wav"] = make_click_mix(
            mastered,
            beat_grid,
            sample_rate,
            params.beats_per_bar,
        )

    result = {
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "report": report,
        "schedule": schedule,
    }
    if disk_output is None:
        artifacts = {
            **{name: wav_bytes(sample_rate, values) for name, values in audio_artifacts.items()},
            **metadata_artifacts,
        }
        result["artifacts"] = artifacts
        if create_zip:
            result["zip_bytes"] = make_zip(artifacts)
    else:
        artifact_paths: dict[str, str] = {}
        for name, values in audio_artifacts.items():
            path = disk_output / name
            sf.write(
                path,
                np.asarray(values, dtype=np.float32).clip(-1.0, 1.0),
                sample_rate,
                format="WAV",
                subtype="PCM_24",
            )
            artifact_paths[name] = str(path)
        for name, values in metadata_artifacts.items():
            path = disk_output / name
            path.write_bytes(values)
            artifact_paths[name] = str(path)
        result["artifact_paths"] = artifact_paths
        if create_zip:
            zip_path = disk_output / "heartbeat_music_project.zip"
            make_zip_from_paths(zip_path, artifact_paths)
            result["zip_path"] = str(zip_path)
    return result


def read_song_bytes(
    filename: str,
    data: bytes | memoryview | BinaryIO,
    *,
    max_duration_seconds: float | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in {"wav", "mp3"}:
        raise ValueError("Song input must be a WAV or MP3 file.")
    source_buffer = data if hasattr(data, "read") and hasattr(data, "seek") else io.BytesIO(data)
    original_position = source_buffer.tell() if hasattr(source_buffer, "tell") else 0
    try:
        source_buffer.seek(0)
        info = sf.info(source_buffer)
        duration = float(info.frames / info.samplerate) if info.samplerate else 0.0
        if max_duration_seconds is not None and duration > max_duration_seconds:
            raise ValueError(
                f"Song duration is {duration:.1f} seconds; the maximum allowed duration is "
                f"{max_duration_seconds:.0f} seconds."
            )
        source_buffer.seek(0)
        audio, sample_rate = sf.read(source_buffer, dtype="float32", always_2d=True)
    except Exception as exc:
        if isinstance(exc, ValueError) and "maximum allowed duration" in str(exc):
            raise
        raise ValueError(f"Could not decode {extension.upper()} song audio: {exc}") from exc
    finally:
        try:
            source_buffer.seek(original_position)
        except (AttributeError, OSError, ValueError):
            pass
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


def _beat_features(
    audio: np.ndarray,
    sample_rate: int,
    beats: np.ndarray,
    onset_envelope: np.ndarray,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return compact normalized full-band energy and low-frequency accents per beat."""
    values = np.asarray(audio, dtype=np.float32)
    beat_times = np.asarray(beats, dtype=np.float64)
    if not len(beat_times):
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    rms = librosa.feature.rms(y=values, frame_length=2048, hop_length=hop_length)[0]
    low_onset = librosa.onset.onset_strength(
        y=values,
        sr=sample_rate,
        hop_length=hop_length,
        fmax=min(250.0, sample_rate * 0.45),
        n_mels=32,
    )
    frames = librosa.time_to_frames(beat_times, sr=sample_rate, hop_length=hop_length)

    def sample_curve(curve: np.ndarray) -> np.ndarray:
        if not len(curve):
            return np.zeros(len(frames), dtype=np.float32)
        indices = np.clip(frames, 0, len(curve) - 1)
        return np.asarray(curve[indices], dtype=np.float32)

    full = 0.55 * sample_curve(onset_envelope) + 0.45 * sample_curve(rms)
    low = sample_curve(low_onset)

    def robust_normalize(curve: np.ndarray) -> np.ndarray:
        if not len(curve):
            return curve
        low_value, high_value = np.percentile(curve, [10.0, 90.0])
        return np.clip(
            (curve - float(low_value)) / max(float(high_value - low_value), 1e-9),
            0.0,
            1.0,
        ).astype(np.float32)

    return robust_normalize(full), robust_normalize(low)


def _infer_downbeats(
    beats: np.ndarray,
    low_strength: np.ndarray,
    meter: int,
    manual_first_downbeat: float | None,
) -> tuple[np.ndarray, float, str]:
    beat_times = np.asarray(beats, dtype=np.float64)
    meter = max(2, int(meter))
    if not len(beat_times):
        return np.asarray([], dtype=np.float64), 0.0, "unavailable"
    if manual_first_downbeat is not None:
        anchor = int(np.argmin(np.abs(beat_times - float(manual_first_downbeat))))
        return beat_times[np.arange(len(beat_times)) % meter == anchor % meter], 1.0, "manual"
    strengths = np.asarray(low_strength, dtype=np.float64)
    if len(strengths) != len(beat_times):
        strengths = np.zeros(len(beat_times), dtype=np.float64)
    phase_scores = np.asarray(
        [np.mean(strengths[phase::meter]) if len(strengths[phase::meter]) else 0.0 for phase in range(meter)],
        dtype=np.float64,
    )
    anchor_phase = int(np.argmax(phase_scores))
    sorted_scores = np.sort(phase_scores)
    margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
    confidence = float(np.clip(0.35 + margin, 0.0, 0.85))
    indices = np.arange(len(beat_times))
    return beat_times[indices % meter == anchor_phase], confidence, "low-frequency-accent"


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


def _bridge_interval_grid(
    left: float,
    right: float,
    target_interval: float,
    minimum_interval: float,
    maximum_interval: float,
) -> tuple[np.ndarray, bool]:
    gap = float(right - left)
    minimum_count = max(1, int(np.ceil(gap / max(maximum_interval, 1e-9))))
    maximum_count = max(1, int(np.floor(gap / max(minimum_interval, 1e-9))))
    relaxed = minimum_count > maximum_count
    if not relaxed:
        counts = np.arange(minimum_count, maximum_count + 1)
        count = int(counts[np.argmin(np.abs(gap / counts - target_interval))])
    else:
        count = maximum_count
    return np.linspace(left, right, count + 1, dtype=np.float64), relaxed


def build_adaptive_pulse_grid(
    beats: np.ndarray,
    heartbeat_bpm: float,
    duration: float,
    *,
    downbeats: np.ndarray | None = None,
    pulse_min_bpm: float = 55.0,
    pulse_max_bpm: float = 110.0,
    active_duration_seconds: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Choose locally suitable real beats and bridge only unsupported long gaps."""
    values = np.unique(np.asarray(beats, dtype=np.float64))
    values = values[(values >= 0.0) & (values <= duration + 1e-9)]
    if len(values) < 2:
        return values, np.ones(len(values), dtype=bool), 0
    slow_rate = max(1e-6, min(float(pulse_min_bpm), float(pulse_max_bpm)))
    fast_rate = max(slow_rate, max(float(pulse_min_bpm), float(pulse_max_bpm)))
    minimum_interval = max(60.0 / fast_rate, max(0.0, active_duration_seconds) + 0.025)
    maximum_interval = max(60.0 / slow_rate, minimum_interval)
    natural_interval = 60.0 / max(float(heartbeat_bpm), 1e-6)
    target_interval = float(np.clip(natural_interval, minimum_interval, maximum_interval))
    anchors = np.asarray(downbeats if downbeats is not None else [], dtype=np.float64)
    anchor_index = int(np.argmin(np.abs(values - anchors[0]))) if len(anchors) else 0
    relaxations = 0

    def walk(start_index: int, direction: int) -> tuple[list[float], list[bool], int]:
        current = start_index
        times = [float(values[current])]
        backed = [True]
        local_relaxations = 0
        while 0 <= current + direction < len(values):
            if direction > 0:
                candidates = np.arange(current + 1, len(values))
                deltas = values[candidates] - values[current]
            else:
                candidates = np.arange(0, current)
                deltas = values[current] - values[candidates]
            valid = np.flatnonzero(
                (deltas >= minimum_interval - 1e-9)
                & (deltas <= maximum_interval + 1e-9)
            )
            if len(valid):
                choice = int(valid[np.argmin(np.abs(np.log(np.maximum(deltas[valid], 1e-9) / target_interval)))])
                next_index = int(candidates[choice])
                times.append(float(values[next_index]))
                backed.append(True)
                current = next_index
                continue
            far = np.flatnonzero(deltas > maximum_interval + 1e-9)
            if not len(far):
                break
            next_index = int(candidates[int(far[0] if direction > 0 else far[-1])])
            left_index, right_index = sorted((current, next_index))
            bridge, relaxed = _bridge_interval_grid(
                float(values[left_index]),
                float(values[right_index]),
                target_interval,
                minimum_interval,
                maximum_interval,
            )
            local_relaxations += int(relaxed)
            inner = bridge[1:-1] if direction > 0 else bridge[-2:0:-1]
            times.extend(float(value) for value in inner)
            backed.extend(False for _ in inner)
            times.append(float(values[next_index]))
            backed.append(True)
            current = next_index
        return times, backed, local_relaxations

    forward_times, forward_backed, forward_relax = walk(anchor_index, 1)
    backward_times, backward_backed, backward_relax = walk(anchor_index, -1)
    relaxations += forward_relax + backward_relax
    times = np.asarray([*reversed(backward_times[1:]), *forward_times], dtype=np.float64)
    model_backed = np.asarray([*reversed(backward_backed[1:]), *forward_backed], dtype=bool)
    return times, model_backed, relaxations


def _role_pulse_grid(
    beats: np.ndarray,
    downbeats: np.ndarray,
    mode: str,
    meter: int,
) -> np.ndarray:
    values = np.asarray(beats, dtype=np.float64)
    if mode in {"normal", "every-beat", "double", "half", "bar", "mute"}:
        if mode == "bar" and len(downbeats):
            return np.asarray(downbeats, dtype=np.float64)
        return _pulse_grid(values, mode, meter)
    anchors = np.asarray(downbeats, dtype=np.float64)
    anchor_index = int(np.argmin(np.abs(values - anchors[0]))) if len(anchors) else 0
    positions = (np.arange(len(values)) - anchor_index) % max(2, meter)
    if mode == "downbeat":
        wanted = {0}
    elif mode == "kick":
        wanted = {0, 2} if meter == 4 else {0}
    elif mode == "backbeat":
        wanted = {1, 3} if meter == 4 else {1}
    else:
        raise ValueError(f"Unsupported pulse role: {mode}")
    return values[np.isin(positions, list(wanted))]


def build_region_schedule(
    beats: np.ndarray,
    duration: float,
    default_mode: str,
    default_fit: str,
    beats_per_bar: int,
    heartbeat_start: float,
    heartbeat_end: float | None,
    edits: list[RegionEdit],
    *,
    heartbeat_bpm: float = 75.0,
    downbeats: np.ndarray | None = None,
    beat_energy: np.ndarray | None = None,
    active_duration_seconds: float = 0.0,
    pulse_min_bpm: float = 55.0,
    pulse_max_bpm: float = 110.0,
    timing_offset_seconds: float = 0.0,
    section_adaptive_strength: float = 0.0,
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
    downbeat_values = np.asarray(downbeats if downbeats is not None else [], dtype=np.float64)
    energy_values = np.asarray(beat_energy if beat_energy is not None else [], dtype=np.float32)
    adaptive_times, adaptive_backed, adaptive_relaxations = build_adaptive_pulse_grid(
        beats,
        heartbeat_bpm,
        duration,
        downbeats=downbeat_values,
        pulse_min_bpm=pulse_min_bpm,
        pulse_max_bpm=pulse_max_bpm,
        active_duration_seconds=active_duration_seconds,
    )
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
        if mode == "auto":
            pulse_values = adaptive_times
            model_values = adaptive_backed
        else:
            pulse_values = _role_pulse_grid(beats, downbeat_values, mode, beats_per_bar)
            model_values = np.ones(len(pulse_values), dtype=bool)
        for local_index, (pulse, model_backed) in enumerate(zip(pulse_values, model_values)):
            pulse = float(pulse) + float(timing_offset_seconds)
            if left - 1e-7 <= pulse < right - 1e-7:
                if len(energy_values) == len(beats):
                    nearest = int(np.argmin(np.abs(beats - pulse)))
                    local_energy = float(np.clip(energy_values[nearest], 0.0, 1.0))
                else:
                    local_energy = 0.5
                strength = float(np.clip(section_adaptive_strength, 0.0, 1.0))
                is_downbeat = bool(
                    len(downbeat_values)
                    and np.min(np.abs(downbeat_values - pulse)) <= 0.08
                )
                sparse_threshold = 0.12 + 0.18 * strength
                if (
                    mode == "auto"
                    and strength >= 0.35
                    and local_energy < sparse_threshold
                    and local_index % 2 == 1
                    and not is_downbeat
                ):
                    continue
                velocity = 1.0 - strength * 0.30 * (1.0 - local_energy)
                schedule.append(
                    {
                        "pulse_index": 0,
                        "time_seconds": float(pulse),
                        "pulse_mode": mode,
                        "fit_mode": fit,
                        "region": label,
                        "model_backed": bool(model_backed),
                        "section_energy": local_energy,
                        "velocity": float(velocity),
                        "guide_constraint_relaxations": int(adaptive_relaxations if mode == "auto" else 0),
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


def _detect_s1_anchor(audio: np.ndarray, sample_rate: int) -> int:
    values = np.max(np.abs(np.asarray(audio, dtype=np.float32)), axis=1)
    if not len(values):
        return 0
    smooth = max(1, int(round(0.008 * sample_rate)))
    envelope = np.convolve(values, np.ones(smooth) / smooth, mode="same")
    search_end = min(len(envelope), max(smooth + 1, int(round(0.45 * sample_rate))))
    peak = int(np.argmax(envelope[:search_end]))
    onset_left = max(0, peak - int(round(0.12 * sample_rate)))
    rise = envelope[onset_left : peak + 1]
    if not len(rise):
        return peak
    baseline = float(np.quantile(rise, 0.15))
    threshold = baseline + 0.20 * max(float(envelope[peak]) - baseline, 0.0)
    below = np.flatnonzero(rise[:-1] < threshold)
    return onset_left + int(below[-1]) + 1 if len(below) else onset_left


def _active_cycle_samples(audio: np.ndarray, sample_rate: int, anchor: int) -> int:
    values = np.max(np.abs(np.asarray(audio, dtype=np.float32)), axis=1)
    if not len(values):
        return 1
    smooth = max(1, int(round(0.012 * sample_rate)))
    envelope = np.convolve(values, np.ones(smooth) / smooth, mode="same")
    baseline = float(np.quantile(envelope, 0.20))
    peak = float(np.max(envelope[max(0, anchor) :])) if anchor < len(envelope) else 0.0
    threshold = baseline + 0.08 * max(peak - baseline, 0.0)
    active = np.flatnonzero(envelope[max(0, anchor) :] > threshold)
    if not len(active):
        return min(len(audio), anchor + int(round(0.25 * sample_rate)))
    tail = anchor + int(active[-1]) + int(round(0.035 * sample_rate))
    return int(np.clip(tail, anchor + 1, len(audio)))


def extract_heartbeat_cycles(result: dict[str, Any]) -> tuple[list[HeartbeatCycle], int]:
    audio = np.asarray(result["cleanest_audio"], dtype=np.float32)
    sample_rate = int(result["sample_rate"])
    segment = result["cleanest_segment"]
    start = float(segment["adjusted_start_seconds"])
    end = float(segment["adjusted_end_seconds"])
    beats = np.asarray(result["beat_times"], dtype=np.float64)
    internal = beats[(beats > start + 0.025) & (beats < end - 0.025)] - start
    boundaries = np.concatenate(([0.0], internal, [len(audio) / sample_rate]))
    boundaries = np.unique(np.clip(boundaries, 0.0, len(audio) / sample_rate))
    cycles: list[HeartbeatCycle] = []
    for cycle_index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        begin = int(round(left * sample_rate))
        finish = int(round(right * sample_rate))
        if finish - begin >= int(0.2 * sample_rate):
            cycle_audio = _edge_fade(audio[begin:finish, None], sample_rate, 5.0)
            anchor = _detect_s1_anchor(cycle_audio, sample_rate)
            cycles.append(
                HeartbeatCycle(
                    audio=cycle_audio,
                    anchor_offset_samples=anchor,
                    active_samples=_active_cycle_samples(cycle_audio, sample_rate, anchor),
                    source_cycle_index=cycle_index,
                )
            )
    expected = max(1, int(segment.get("cycle_count") or 1))
    if len(cycles) < 2:
        indices = np.linspace(0, len(audio), expected + 1, dtype=int)
        cycles = []
        for cycle_index in range(expected):
            if indices[cycle_index + 1] <= indices[cycle_index]:
                continue
            cycle_audio = _edge_fade(
                audio[indices[cycle_index] : indices[cycle_index + 1], None],
                sample_rate,
                5.0,
            )
            anchor = _detect_s1_anchor(cycle_audio, sample_rate)
            cycles.append(
                HeartbeatCycle(
                    audio=cycle_audio,
                    anchor_offset_samples=anchor,
                    active_samples=_active_cycle_samples(cycle_audio, sample_rate, anchor),
                    source_cycle_index=cycle_index,
                )
            )
    if not cycles:
        raise ValueError("No usable heartbeat cycle was found in the selected clean loop.")
    return cycles, sample_rate


def render_heartbeat_layer(
    cycles: list[HeartbeatCycle],
    schedule: list[dict[str, Any]],
    sample_rate: int,
    channels: int,
    output_samples: int,
    *,
    max_stretch_ratio: float = 1.18,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.zeros((output_samples, channels), dtype=np.float32)
    if not schedule:
        return output, {"anchor_offsets_ms": [], "maximum_error_ms": 0.0, "skipped_count": 0}
    times = np.asarray([item["time_seconds"] for item in schedule], dtype=np.float64)
    intervals = np.diff(times)
    fallback = float(np.median(intervals)) if len(intervals) else 60.0 / 75.0
    rendered_errors: list[float] = []
    skipped = 0
    previous_cycle = -1
    for index, item in enumerate(schedule):
        interval = float(intervals[index]) if index < len(intervals) else fallback
        target_samples = max(1, int(round(interval * sample_rate)))
        costs = [
            abs(math.log(max(len(cycle.audio), 1) / target_samples))
            + (0.08 if cycle_index == previous_cycle and len(cycles) > 1 else 0.0)
            for cycle_index, cycle in enumerate(cycles)
        ]
        cycle_index = int(np.argmin(costs))
        previous_cycle = cycle_index
        cycle = cycles[cycle_index]
        cycle_audio = cycle.audio
        if cycle_audio.shape[1] == 1 and channels > 1:
            cycle_audio = np.repeat(cycle_audio, channels, axis=1)
        elif cycle_audio.shape[1] != channels:
            cycle_audio = np.repeat(np.mean(cycle_audio, axis=1, keepdims=True), channels, axis=1)
        fitted, fitted_anchor = fit_cycle(
            HeartbeatCycle(
                audio=cycle_audio,
                anchor_offset_samples=cycle.anchor_offset_samples,
                active_samples=cycle.active_samples,
                source_cycle_index=cycle.source_cycle_index,
                anchor_mode=cycle.anchor_mode,
            ),
            target_samples,
            sample_rate,
            item["fit_mode"],
            max_stretch_ratio=max_stretch_ratio,
        )
        fitted *= float(item.get("velocity", 1.0))
        target_anchor = int(round(item["time_seconds"] * sample_rate))
        begin = target_anchor - fitted_anchor
        source_begin = max(0, -begin)
        begin = max(0, begin)
        finish = min(output_samples, begin + len(fitted) - source_begin)
        if begin < output_samples and finish > begin:
            source_finish = source_begin + finish - begin
            output[begin:finish] += fitted[source_begin:source_finish]
            if source_begin <= fitted_anchor < source_finish and 0 <= target_anchor < output_samples:
                actual_anchor = begin + fitted_anchor - source_begin
                rendered_errors.append(abs(actual_anchor - target_anchor) * 1000.0 / sample_rate)
            else:
                skipped += 1
        else:
            skipped += 1
    return output, {
        "anchor_offsets_ms": [
            float(cycle.anchor_offset_samples * 1000.0 / sample_rate) for cycle in cycles
        ],
        "maximum_error_ms": float(max(rendered_errors, default=0.0)),
        "skipped_count": int(skipped),
    }


def fit_cycle(
    cycle: HeartbeatCycle,
    target_samples: int,
    sample_rate: int,
    mode: str,
    *,
    max_stretch_ratio: float = 1.18,
) -> tuple[np.ndarray, int]:
    audio = cycle.audio
    target_samples = max(1, int(target_samples))
    if mode == "gap":
        output = np.zeros((target_samples, audio.shape[1]), dtype=np.float32)
        length = min(len(audio), target_samples)
        output[:length] = audio[:length]
        return output, min(cycle.anchor_offset_samples, target_samples - 1)
    if mode != "stretch":
        raise ValueError(f"Unknown cycle fit mode: {mode}")
    limit = max(1.01, float(max_stretch_ratio))
    bounded_target = int(np.clip(target_samples, len(audio) / limit, len(audio) * limit))
    rate = len(audio) / max(1, bounded_target)
    stretched_channels = [
        librosa.effects.time_stretch(audio[:, channel], rate=rate)
        for channel in range(audio.shape[1])
    ]
    output = np.zeros((target_samples, audio.shape[1]), dtype=np.float32)
    for channel, values in enumerate(stretched_channels):
        length = min(target_samples, len(values))
        output[:length, channel] = values[:length]
    output = _edge_fade(output, sample_rate, 4.0)
    expected_anchor = int(round(cycle.anchor_offset_samples * bounded_target / max(len(audio), 1)))
    search_left = max(0, expected_anchor - int(round(0.06 * sample_rate)))
    search_right = min(len(output), expected_anchor + int(round(0.12 * sample_rate)) + 1)
    if search_right > search_left:
        detected = _detect_s1_anchor(output[search_left:search_right], sample_rate) + search_left
    else:
        detected = int(np.clip(expected_anchor, 0, len(output) - 1))
    return output, detected


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


def arrangement_fade_envelope(
    sample_count: int,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float | None,
    fade_in_seconds: float,
    fade_out_seconds: float,
) -> np.ndarray:
    """Create a smooth heartbeat-only entrance/exit envelope inside its active range."""
    envelope = np.ones(sample_count, dtype=np.float32)
    start = int(np.clip(round(float(start_seconds) * sample_rate), 0, sample_count))
    end_time = sample_count / sample_rate if end_seconds is None else float(end_seconds)
    end = int(np.clip(round(end_time * sample_rate), start, sample_count))
    envelope[:start] = 0.0
    envelope[end:] = 0.0
    fade_in = min(end - start, max(0, int(round(float(fade_in_seconds) * sample_rate))))
    fade_out = min(end - start, max(0, int(round(float(fade_out_seconds) * sample_rate))))
    if fade_in > 1:
        envelope[start : start + fade_in] *= np.sin(
            np.linspace(0.0, np.pi * 0.5, fade_in, dtype=np.float32)
        ) ** 2
    if fade_out > 1:
        envelope[end - fade_out : end] *= np.cos(
            np.linspace(0.0, np.pi * 0.5, fade_out, dtype=np.float32)
        ) ** 2
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
        return song, {"depth_db": 0.0, "cutoff_hz": float(cutoff_hz)}
    trigger = np.abs(heartbeat[:, 0]).astype(np.float32, copy=True)
    for channel in range(1, heartbeat.shape[1]):
        np.maximum(trigger, np.abs(heartbeat[:, channel]), out=trigger)
    window = max(1, int(round(0.12 * sample_rate)))
    envelope = signal.lfilter(
        np.ones(window, dtype=np.float32) / window,
        np.asarray([1.0], dtype=np.float32),
        trigger,
    ).astype(np.float32, copy=False)
    del trigger
    high = float(np.percentile(envelope[envelope > 1e-9], 95.0)) if np.any(envelope > 1e-9) else 0.0
    envelope /= max(high, 1e-9)
    np.clip(envelope, 0.0, 1.0, out=envelope)
    active_mask = envelope > 0.05
    np.power(envelope, 0.65, out=envelope)
    envelope *= -abs(depth_db) / 20.0
    np.power(10.0, envelope, out=envelope)
    gain = envelope
    cutoff = float(np.clip(cutoff_hz, 40.0, sample_rate * 0.45))
    sos = signal.butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    low = signal.sosfiltfilt(sos, song, axis=0).astype(np.float32)
    song -= low
    low *= gain[:, None]
    song += low
    mean_active_duck = (
        float(-20.0 * np.log10(max(float(np.mean(gain[active_mask])), 1e-9)))
        if np.any(active_mask)
        else 0.0
    )
    del low
    return song, {
        "depth_db": float(depth_db),
        "cutoff_hz": cutoff,
        "mean_active_duck_db": mean_active_duck,
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
    output = audio
    output *= db_to_gain(loudness_gain_db)
    peak_before_limiter = float(np.max(np.abs(output))) if output.size else 0.0
    ceiling = db_to_gain(ceiling_dbfs)
    limiter_active = peak_before_limiter > ceiling
    if limiter_active:
        output /= ceiling
        np.tanh(output, out=output)
        output *= ceiling
    after = analyze_loudness(output, sample_rate)
    peak_reduction_db = (
        after["peak_dbfs"]
        - 20.0 * math.log10(max(peak_before_limiter, 1e-12))
        if limiter_active
        else 0.0
    )
    return output, {
        "target_lufs": float(target_lufs),
        "ceiling_dbfs": float(ceiling_dbfs),
        "input_lufs": before["integrated_lufs"],
        "requested_loudness_gain_db": loudness_gain_db,
        "limiter": "tanh-soft-peak",
        "limiter_active": bool(limiter_active),
        "peak_protection_db": float(peak_reduction_db),
        "applied_gain_db": float(loudness_gain_db),
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


def make_zip_from_paths(destination: str | Path, artifacts: dict[str, str]) -> str:
    destination = Path(destination)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in artifacts.items():
            path = Path(value)
            if path.is_file():
                archive.write(path, arcname=name)
    return str(destination)
