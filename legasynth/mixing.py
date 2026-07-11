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
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
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


def tile_to_length(loop: np.ndarray, target_samples: int) -> np.ndarray:
    if len(loop) == 0:
        return np.zeros((target_samples, 2), dtype=np.float32)
    repeats = int(np.ceil(target_samples / len(loop)))
    return np.tile(loop, (repeats, 1))[:target_samples].astype(np.float32)


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
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    song, sr = load_audio(song_wav)
    song = ensure_stereo(song)
    loop, _ = load_audio(loop_wav, target_sr=sr)
    loop = ensure_stereo(loop)

    source_bpm = float(heartbeat_summary.get("best_loop", {}).get("local_bpm") or heartbeat_summary.get("tempo", {}).get("estimated_bpm") or 0.0)
    target_bpm = float(song_bpm or source_bpm or 75.0)
    stretched = stretch_loop_to_bpm(loop, sr, source_bpm, target_bpm)
    tiled = tile_to_length(stretched, len(song))
    heartbeat_gain = db_to_gain(heartbeat_gain_db)
    heartbeat_bed = tiled * heartbeat_gain
    mixed = peak_normalize(song + heartbeat_bed, peak=0.98)

    final_wav = out_dir / "final_audio.wav"
    final_mp3 = out_dir / "final_audio.mp3"
    sf.write(str(final_wav), mixed, sr)
    mp3_ok = export_mp3(final_wav, final_mp3)

    duration = float(len(song) / sr) if sr else 0.0
    aligned_beats = generate_aligned_beat_times(duration, target_bpm)
    beat_csv = out_dir / "aligned_heartbeat_beats.csv"
    pd.DataFrame({"beat_index": np.arange(len(aligned_beats)), "time_seconds": aligned_beats}).to_csv(beat_csv, index=False)

    report = {
        "song_wav": str(song_wav),
        "loop_wav": str(loop_wav),
        "sample_rate": int(sr),
        "duration_seconds": duration,
        "source_heartbeat_bpm": source_bpm,
        "target_song_bpm": target_bpm,
        "time_stretch_rate": float(target_bpm / source_bpm) if source_bpm > 0 else None,
        "heartbeat_gain_db": float(heartbeat_gain_db),
        "heartbeat_gain_linear": heartbeat_gain,
        "final_peak_abs": float(np.max(np.abs(mixed))) if len(mixed) else 0.0,
        "mp3_exported": bool(mp3_ok),
        "aligned_beat_count": int(len(aligned_beats)),
        "final_audio_wav": str(final_wav),
        "final_audio_mp3": str(final_mp3) if mp3_ok else None,
        "aligned_heartbeat_beats_csv": str(beat_csv),
    }
    (out_dir / "mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

