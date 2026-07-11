from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(video_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = float(frames / fps) if fps > 0 else 0.0
    return {
        "path": str(path),
        "filename": path.name,
        "duration": duration,
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
    }


def extract_audio(video_path: str | os.PathLike[str], wav_path: str | os.PathLike[str], sample_rate: int = 44100) -> Path:
    video_path = Path(video_path)
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path


def analyze_song_audio(wav_path: str | os.PathLike[str]) -> dict[str, Any]:
    wav_path = Path(wav_path)
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration = float(len(y) / sr) if sr else 0.0
    if len(y) == 0:
        return {
            "audio_sample_rate": int(sr),
            "audio_duration": duration,
            "estimated_song_bpm": 0.0,
            "song_beat_times_seconds": [],
            "beat_tracking_confidence": 0.0,
        }

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo_value = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if tempo_value <= 0 or len(beat_times) < 2:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_value = float(np.atleast_1d(librosa.beat.tempo(onset_envelope=onset_env, sr=sr))[0])
        beat_times = np.array([], dtype=np.float32)

    confidence = float(min(1.0, len(beat_times) / max(duration / 0.5, 1.0)))
    return {
        "audio_sample_rate": int(sr),
        "audio_duration": duration,
        "estimated_song_bpm": tempo_value,
        "song_beat_times_seconds": [round(float(v), 6) for v in beat_times],
        "beat_tracking_confidence": confidence,
    }


def prepare_video_audio(
    video_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    sample_rate: int = 44100,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "song_audio.wav"
    metadata = probe_video(video_path)
    extract_audio(video_path, audio_path, sample_rate=sample_rate)
    audio_info = analyze_song_audio(audio_path)
    metadata.update(audio_info)
    metadata["extracted_audio_path"] = str(audio_path)
    (out_dir / "video_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

