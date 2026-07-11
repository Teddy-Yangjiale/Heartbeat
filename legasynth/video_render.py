from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf


def nearest_pulse_strength(t: float, beat_times: np.ndarray, width: float = 0.18) -> float:
    if len(beat_times) == 0:
        return 0.0
    idx = int(np.searchsorted(beat_times, t))
    candidates = []
    if idx < len(beat_times):
        candidates.append(abs(float(beat_times[idx]) - t))
    if idx > 0:
        candidates.append(abs(float(beat_times[idx - 1]) - t))
    if not candidates:
        return 0.0
    d = min(candidates)
    return float(np.exp(-((d / width) ** 2)))


def load_waveform(loop_wav: str | os.PathLike[str], max_points: int = 512) -> np.ndarray:
    try:
        y, _ = sf.read(str(loop_wav), dtype="float32", always_2d=False)
    except Exception:
        return np.zeros(max_points, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) == 0:
        return np.zeros(max_points, dtype=np.float32)
    idx = np.linspace(0, len(y) - 1, max_points).astype(int)
    wave = y[idx]
    peak = float(np.max(np.abs(wave)))
    if peak > 1e-9:
        wave = wave / peak
    return wave.astype(np.float32)


def draw_waveform(frame: np.ndarray, wave: np.ndarray, phase: float, strength: float) -> None:
    h, w = frame.shape[:2]
    panel_h = max(70, int(h * 0.14))
    y_mid = h - panel_h // 2
    x0 = int(w * 0.05)
    x1 = int(w * 0.95)
    width = max(2, x1 - x0)
    cv2.rectangle(frame, (0, h - panel_h - 8), (w, h), (0, 0, 0), thickness=-1)
    overlay_alpha = 0.45
    frame[h - panel_h - 8 : h] = (frame[h - panel_h - 8 : h] * (1.0 - overlay_alpha)).astype(np.uint8)

    rolled = np.roll(wave, int(phase * len(wave)))
    xs = np.linspace(x0, x1, len(rolled)).astype(np.int32)
    amp = max(18, int(panel_h * (0.22 + 0.16 * strength)))
    ys = (y_mid - rolled * amp).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=(80, 245, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.line(frame, (x0, y_mid), (x1, y_mid), (120, 120, 120), 1, cv2.LINE_AA)


def draw_text(frame: np.ndarray, bpm: float, title_text: str, strength: float) -> None:
    h, w = frame.shape[:2]
    bpm_text = f"Heartbeat BPM {bpm:.1f}"
    cv2.putText(frame, bpm_text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, bpm_text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 255), 1, cv2.LINE_AA)
    if strength > 0.35:
        cv2.circle(frame, (w - 48, 42), int(10 + 16 * strength), (40, 40, 255), -1, cv2.LINE_AA)
    if title_text:
        y = h - max(92, int(h * 0.17))
        cv2.putText(frame, title_text[:60], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, title_text[:60], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 230, 230), 1, cv2.LINE_AA)


def apply_pulse(frame: np.ndarray, strength: float, effect_strength: float) -> np.ndarray:
    if strength <= 0:
        return frame
    gain = 1.0 + 0.35 * effect_strength * strength
    out = np.clip(frame.astype(np.float32) * gain, 0, 255)
    tint = np.zeros_like(out)
    tint[:, :, 2] = 80.0
    out = out * (1.0 - 0.18 * effect_strength * strength) + tint * (0.18 * effect_strength * strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def mux_audio(video_no_audio: Path, audio_path: Path, final_path: Path) -> None:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(video_no_audio),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-shortest",
        str(final_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def render_heartbeat_video(
    source_video: str | os.PathLike[str],
    final_audio: str | os.PathLike[str],
    loop_wav: str | os.PathLike[str],
    beat_times: list[float] | np.ndarray,
    out_dir: str | os.PathLike[str],
    heartbeat_bpm: float,
    title_text: str = "",
    effect_strength: float = 0.75,
    duration_limit: float | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "final_video.mp4"
    temp_video = out_dir / "_visual_no_audio.mp4"

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames
    if duration_limit and duration_limit > 0:
        max_frames = min(max_frames, int(duration_limit * fps)) if max_frames else int(duration_limit * fps)

    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not create video writer: {temp_video}")

    beats = np.asarray(beat_times, dtype=np.float32)
    wave = load_waveform(loop_wav)
    frame_index = 0
    while True:
        if max_frames and frame_index >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_index / fps
        pulse = nearest_pulse_strength(t, beats)
        frame = apply_pulse(frame, pulse, effect_strength)
        phase = (t * heartbeat_bpm / 60.0) % 1.0 if heartbeat_bpm > 0 else 0.0
        draw_waveform(frame, wave, phase, pulse)
        draw_text(frame, heartbeat_bpm, title_text, pulse)
        writer.write(frame)
        frame_index += 1

    writer.release()
    cap.release()
    mux_audio(temp_video, Path(final_audio), final_path)
    try:
        temp_video.unlink()
    except OSError:
        pass

    report = {
        "source_video": str(source_video),
        "final_audio": str(final_audio),
        "final_video": str(final_path),
        "fps": fps,
        "width": width,
        "height": height,
        "rendered_frames": frame_index,
        "duration_seconds": float(frame_index / fps) if fps else 0.0,
        "effect_strength": float(effect_strength),
        "title_text": title_text,
    }
    (out_dir / "video_render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

