from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def peak_normalize(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(x))) if len(x) else 0.0
    if max_abs <= peak or max_abs <= 1e-12:
        return x.astype(np.float32)
    return (x / max_abs * peak).astype(np.float32)


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.column_stack([x, x]).astype(np.float32)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1).astype(np.float32)
    return x[:, :2].astype(np.float32)


def load_audio(path: str | os.PathLike[str], target_sr: int | None = None) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        decoded, sr = librosa.load(str(path), sr=None, mono=False)
        y = decoded[:, None] if decoded.ndim == 1 else decoded.T
    if target_sr is not None and sr != target_sr:
        channels = [librosa.resample(y[:, c], orig_sr=sr, target_sr=target_sr) for c in range(y.shape[1])]
        y = np.column_stack(channels).astype(np.float32)
        sr = target_sr
    return y, int(sr)


def stretch_loop_to_bpm(loop: np.ndarray, sr: int, source_bpm: float, target_bpm: float) -> np.ndarray:
    if source_bpm <= 0 or target_bpm <= 0 or len(loop) < 16:
        return loop.astype(np.float32)
    rate = float(target_bpm / source_bpm)
    stretched_channels = []
    for c in range(loop.shape[1]):
        stretched_channels.append(librosa.effects.time_stretch(loop[:, c].astype(np.float32), rate=rate))
    n = min(len(ch) for ch in stretched_channels)
    return np.column_stack([ch[:n] for ch in stretched_channels]).astype(np.float32)


def stretch_loop_to_duration(loop: np.ndarray, sr: int, target_duration_seconds: float) -> np.ndarray:
    """Time-stretch a loop to an exact musical-grid duration without changing pitch."""
    if len(loop) < 16 or sr <= 0 or target_duration_seconds <= 0:
        return loop.astype(np.float32)
    source_duration = len(loop) / float(sr)
    rate = source_duration / float(target_duration_seconds)
    stretched_channels = [
        librosa.effects.time_stretch(loop[:, c].astype(np.float32), rate=rate)
        for c in range(loop.shape[1])
    ]
    target_samples = max(1, int(round(target_duration_seconds * sr)))
    result = np.zeros((target_samples, len(stretched_channels)), dtype=np.float32)
    for channel_index, channel in enumerate(stretched_channels):
        copy_length = min(target_samples, len(channel))
        result[:copy_length, channel_index] = channel[:copy_length]
    return result


def tile_to_length(loop: np.ndarray, target_samples: int) -> np.ndarray:
    if len(loop) == 0:
        return np.zeros((target_samples, 2), dtype=np.float32)
    repeats = int(np.ceil(target_samples / len(loop)))
    return np.tile(loop, (repeats, 1))[:target_samples].astype(np.float32)


def tile_from_offset(loop: np.ndarray, target_samples: int, offset_samples: int) -> np.ndarray:
    channels = loop.shape[1] if loop.ndim == 2 else 2
    result = np.zeros((target_samples, channels), dtype=np.float32)
    offset = max(0, min(int(offset_samples), target_samples))
    if offset < target_samples and len(loop):
        result[offset:] = tile_to_length(loop, target_samples - offset)
    return result


def generate_aligned_beat_times(duration: float, bpm: float, first_beat: float = 0.0) -> np.ndarray:
    if duration <= 0 or bpm <= 0:
        return np.array([], dtype=np.float32)
    step = 60.0 / bpm
    return np.arange(first_beat, duration + 1e-6, step, dtype=np.float32)


def export_mp3(wav_path: Path, mp3_path: Path) -> bool:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(mp3_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0 and mp3_path.exists()


def mix_heartbeat_with_song(
    song_wav: str | os.PathLike[str],
    loop_wav: str | os.PathLike[str],
    heartbeat_summary: dict[str, Any],
    song_bpm: float,
    out_dir: str | os.PathLike[str],
    heartbeat_gain_db: float = -15.0,
    first_beat_seconds: float = 0.0,
    heartbeat_beats_per_loop: int | None = None,
    output_peak: float = 0.98,
    song_beat_times: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    song, sr = load_audio(song_wav)
    song = ensure_stereo(song)
    loop, _ = load_audio(loop_wav, target_sr=sr)
    loop = ensure_stereo(loop)

    source_bpm = float(heartbeat_summary.get("best_loop", {}).get("local_bpm") or heartbeat_summary.get("tempo", {}).get("estimated_bpm") or 0.0)
    target_bpm = float(song_bpm or source_bpm or 75.0)
    source_loop_beats = int(
        heartbeat_beats_per_loop
        or heartbeat_summary.get("best_loop", {}).get("num_beats")
        or 4
    )
    target_loop_duration = source_loop_beats * 60.0 / target_bpm
    alignment_offset = max(0.0, float(first_beat_seconds))
    dynamic_grid = _validated_beat_grid(song_beat_times, duration=len(song) / sr)
    segment_durations: list[float] = []
    if len(dynamic_grid) >= source_loop_beats + 1:
        tiled, segment_durations = build_tempo_mapped_layer(
            loop,
            sr,
            len(song),
            dynamic_grid,
            source_loop_beats,
        )
        aligned_beats = dynamic_grid
        grid_mode = "dynamic_tempo_map"
    else:
        stretched = stretch_loop_to_duration(loop, sr, target_loop_duration)
        tiled = tile_from_offset(stretched, len(song), int(round(alignment_offset * sr)))
        aligned_beats = generate_aligned_beat_times(len(song) / sr, target_bpm, first_beat=alignment_offset)
        segment_durations = [target_loop_duration]
        grid_mode = "constant_grid"
    heartbeat_gain = db_to_gain(heartbeat_gain_db)
    heartbeat_bed = tiled * heartbeat_gain
    raw_mix = song + heartbeat_bed
    raw_peak = float(np.max(np.abs(raw_mix))) if len(raw_mix) else 0.0
    mixed = peak_normalize(raw_mix, peak=output_peak)
    applied_mix_gain = float(output_peak / raw_peak) if raw_peak > output_peak else 1.0

    heartbeat_layer_wav = out_dir / "heartbeat_layer.wav"
    final_wav = out_dir / "final_audio.wav"
    final_mp3 = out_dir / "final_audio.mp3"
    sf.write(str(heartbeat_layer_wav), heartbeat_bed, sr)
    sf.write(str(final_wav), mixed, sr)
    mp3_ok = export_mp3(final_wav, final_mp3)

    duration = float(len(song) / sr) if sr else 0.0
    beat_csv = out_dir / "aligned_heartbeat_beats.csv"
    pd.DataFrame({"beat_index": np.arange(len(aligned_beats)), "time_seconds": aligned_beats}).to_csv(beat_csv, index=False)

    report = {
        "song_wav": str(song_wav),
        "loop_wav": str(loop_wav),
        "sample_rate": int(sr),
        "duration_seconds": duration,
        "source_heartbeat_bpm": source_bpm,
        "target_song_bpm": target_bpm,
        "heartbeat_beats_per_loop": source_loop_beats,
        "source_loop_duration_seconds": float(len(loop) / sr),
        "target_loop_duration_seconds": target_loop_duration,
        "time_stretch_rate": float((len(loop) / sr) / target_loop_duration),
        "first_beat_seconds": alignment_offset,
        "heartbeat_gain_db": float(heartbeat_gain_db),
        "heartbeat_gain_linear": heartbeat_gain,
        "pre_limiter_peak_abs": raw_peak,
        "mix_gain_reduction_db": float(20.0 * np.log10(max(applied_mix_gain, 1e-12))),
        "final_peak_abs": float(np.max(np.abs(mixed))) if len(mixed) else 0.0,
        "mp3_exported": bool(mp3_ok),
        "aligned_beat_count": int(len(aligned_beats)),
        "grid_mode": grid_mode,
        "tempo_mapped_segment_count": len(segment_durations),
        "tempo_mapped_segment_duration_min_seconds": float(min(segment_durations)),
        "tempo_mapped_segment_duration_max_seconds": float(max(segment_durations)),
        "heartbeat_layer_wav": str(heartbeat_layer_wav),
        "final_audio_wav": str(final_wav),
        "final_audio_mp3": str(final_mp3) if mp3_ok else None,
        "aligned_heartbeat_beats_csv": str(beat_csv),
    }
    (out_dir / "mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _validated_beat_grid(values: list[float] | np.ndarray | None, duration: float) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    grid = np.asarray(values, dtype=np.float64)
    grid = grid[np.isfinite(grid)]
    grid = grid[(grid >= 0.0) & (grid <= duration + 1e-6)]
    grid = np.unique(grid)
    if len(grid) < 2 or np.any(np.diff(grid) < 0.08):
        return np.array([], dtype=np.float64)
    return grid


def build_tempo_mapped_layer(
    loop: np.ndarray,
    sr: int,
    target_samples: int,
    beat_grid: np.ndarray,
    beats_per_loop: int,
) -> tuple[np.ndarray, list[float]]:
    """Render one heartbeat loop per beat-grid block so tempo drift cannot accumulate."""
    result = np.zeros((target_samples, loop.shape[1]), dtype=np.float32)
    durations: list[float] = []
    block = max(1, int(beats_per_loop))
    for index in range(0, len(beat_grid) - 1, block):
        end_index = min(index + block, len(beat_grid) - 1)
        if end_index <= index:
            continue
        start_time = float(beat_grid[index])
        end_time = float(beat_grid[end_index])
        if end_index - index < block:
            local_period = float(np.median(np.diff(beat_grid[max(0, index - block) : end_index + 1])))
            end_time = min(target_samples / sr, start_time + block * local_period)
        duration = end_time - start_time
        if duration <= 0:
            continue
        rendered = stretch_loop_to_duration(loop, sr, duration)
        start_sample = max(0, int(round(start_time * sr)))
        end_sample = min(target_samples, start_sample + len(rendered))
        if end_sample <= start_sample:
            continue
        result[start_sample:end_sample] = rendered[: end_sample - start_sample]
        durations.append(duration)
    return result, durations
