"""Feature B: render a music video whose editing rhythm is driven by the heartbeat.

The source music video plays on a linear timeline (so it stays in sync with the
mixed audio), but on top of it we impose heartbeat-synchronous dynamics:

- a per-beat zoom + brightness "pulse" so the picture visibly breathes with the heart,
- a white flash accent on strong beats,
- harder "cut" accents (a zoom punch) every N beats, N coming from Feature A's
  style profile, so the edit feels faster when the heart is racing and calmer when
  it is at rest,
- an emotion-driven colour grade (warmth / saturation / brightness / contrast /
  vignette) taken from Feature A.

An optional diagnostic overlay (heartbeat waveform + BPM) can still be drawn.
"""

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


DEFAULT_STYLE: dict[str, Any] = {
    "warmth": 0.0,
    "saturation": 1.0,
    "brightness": 1.0,
    "contrast": 1.0,
    "vignette": 0.0,
    "pulse_intensity": 0.6,
    "flash_strength": 0.2,
    "beats_per_cut": 4,
    "grade_name": "neutral",
    "mood": "neutral",
}


# --- beat helpers ------------------------------------------------------------

def nearest_pulse_strength(t: float, beat_times: np.ndarray, width: float = 0.16) -> float:
    """Gaussian bump that peaks (1.0) exactly on the nearest beat and decays away."""
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
    return float(np.exp(-((d / max(width, 1e-4)) ** 2)))


def select_cut_beats(beat_times: np.ndarray, beats_per_cut: int) -> np.ndarray:
    """Every Nth beat becomes a hard "cut" accent."""
    beats = np.asarray(beat_times, dtype=np.float32)
    if len(beats) == 0:
        return beats
    step = max(1, int(beats_per_cut))
    return beats[::step]


# --- colour grade ------------------------------------------------------------

def precompute_vignette(width: int, height: int, strength: float) -> np.ndarray | None:
    """Radial darkening mask in [1-strength, 1], shape (H, W, 1) for broadcasting."""
    if strength <= 1e-3:
        return None
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(xs ** 2 + ys ** 2) / np.sqrt(2.0)
    mask = 1.0 - float(strength) * np.clip(radius ** 1.6, 0.0, 1.0)
    return mask.astype(np.float32)[:, :, None]


def apply_color_grade(
    frame_f: np.ndarray,
    style: dict[str, Any],
    vignette_mask: np.ndarray | None,
    extra_brightness: float = 1.0,
) -> np.ndarray:
    """Apply warmth, contrast, brightness, saturation and vignette. Input/out float32 BGR."""
    warmth = float(style.get("warmth", 0.0))
    saturation = float(style.get("saturation", 1.0))
    brightness = float(style.get("brightness", 1.0)) * float(extra_brightness)
    contrast = float(style.get("contrast", 1.0))

    out = frame_f
    # Contrast around mid-grey, then brightness.
    if abs(contrast - 1.0) > 1e-3:
        out = (out - 128.0) * contrast + 128.0
    if abs(brightness - 1.0) > 1e-3:
        out = out * brightness

    # Warmth: push red up / blue down (BGR order -> channel 2 is R, channel 0 is B).
    if abs(warmth) > 1e-3:
        out[:, :, 2] *= (1.0 + 0.25 * warmth)
        out[:, :, 0] *= (1.0 - 0.25 * warmth)

    # Cheap saturation in BGR via luma mixing (avoids two HSV conversions per frame).
    if abs(saturation - 1.0) > 1e-3:
        luma = (0.114 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.299 * out[:, :, 2])[:, :, None]
        out = luma + saturation * (out - luma)

    if vignette_mask is not None:
        out = out * vignette_mask

    return out


def zoom_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Centre zoom by `scale` (>=1) keeping the original resolution."""
    if scale <= 1.0001:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    return resized[y0:y0 + h, x0:x0 + w]


# --- overlays ----------------------------------------------------------------

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
    frame[h - panel_h - 8 : h] = (frame[h - panel_h - 8 : h] * 0.55).astype(np.uint8)
    rolled = np.roll(wave, int(phase * len(wave)))
    xs = np.linspace(x0, x1, len(rolled)).astype(np.int32)
    amp = max(18, int(panel_h * (0.22 + 0.16 * strength)))
    ys = (y_mid - rolled * amp).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=(80, 245, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.line(frame, (x0, y_mid), (x1, y_mid), (120, 120, 120), 1, cv2.LINE_AA)


def _outlined_text(frame: np.ndarray, text: str, org: tuple[int, int], scale: float, color, thickness: int = 2) -> None:
    """Dark outline then a single coloured fill -> crisp, no ghosting."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_text(frame: np.ndarray, bpm: float, title_text: str, mood: str, strength: float) -> None:
    h, w = frame.shape[:2]
    _outlined_text(frame, f"Heartbeat {bpm:.0f} BPM", (24, 44), 0.9, (60, 220, 255), 2)
    if mood and mood != "neutral":
        _outlined_text(frame, mood, (24, 76), 0.62, (235, 235, 235), 2)
    if strength > 0.35:
        cv2.circle(frame, (w - 48, 42), int(10 + 16 * strength), (40, 40, 255), -1, cv2.LINE_AA)
    if title_text:
        y = h - max(92, int(h * 0.17))
        _outlined_text(frame, title_text[:60], (24, y), 0.72, (235, 235, 235), 2)


def active_subtitle(t: float, subtitles: list[dict] | None) -> tuple[str, float]:
    """Return (text, alpha) for the lyric line active at time t, with fade in/out."""
    if not subtitles:
        return "", 0.0
    fade = 0.28
    for line in subtitles:
        start, end = float(line["start"]), float(line["end"])
        if start <= t <= end:
            a = min((t - start) / fade, (end - t) / fade, 1.0)
            return str(line.get("text", "")), float(max(0.0, a))
    return "", 0.0


def draw_subtitle(frame: np.ndarray, text: str, alpha: float) -> None:
    """Karaoke-style centred lyric with a translucent backdrop and fade."""
    if not text or alpha <= 0.02:
        return
    h, w = frame.shape[:2]
    scale = max(0.7, w / 1100.0)
    thickness = max(2, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    # CJK glyphs are wider than getTextSize reports for the Hershey font; pad by an estimate.
    tw = max(tw, int(len(text) * 26 * scale))
    x = (w - tw) // 2
    y = int(h * 0.88)
    pad = int(14 * scale)
    x0, y0 = max(0, x - pad), max(0, y - th - pad)
    x1, y1 = min(w, x + tw + pad), min(h, y + base + pad)
    box_alpha = 0.45 * alpha
    frame[y0:y1, x0:x1] = (frame[y0:y1, x0:x1] * (1.0 - box_alpha)).astype(np.uint8)
    col = int(245 * alpha)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (col, col, col), thickness, cv2.LINE_AA)


def mux_audio(video_no_audio: Path, audio_path: Path, final_path: Path) -> None:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i", str(video_no_audio),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(final_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# --- main render -------------------------------------------------------------

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
    style_profile: dict[str, Any] | None = None,
    show_overlay: bool = False,
    enable_beat_editing: bool = True,
    subtitles: list[dict] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "final_video.mp4"
    temp_video = out_dir / "_visual_no_audio.mp4"

    style = {**DEFAULT_STYLE, **(style_profile or {})}
    pulse_intensity = float(style["pulse_intensity"]) * float(effect_strength)
    flash_strength = float(style["flash_strength"]) * float(effect_strength)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames
    if duration_limit and duration_limit > 0:
        limit_frames = int(duration_limit * fps)
        max_frames = min(max_frames, limit_frames) if max_frames else limit_frames

    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not create video writer: {temp_video}")

    beats = np.asarray(beat_times, dtype=np.float32)
    cut_beats = select_cut_beats(beats, int(style["beats_per_cut"])) if enable_beat_editing else np.array([], dtype=np.float32)
    beat_period = 60.0 / heartbeat_bpm if heartbeat_bpm > 0 else 0.8
    vignette_mask = precompute_vignette(width, height, float(style["vignette"]))
    wave = load_waveform(loop_wav)

    frame_index = 0
    while True:
        if max_frames and frame_index >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_index / fps

        pulse = nearest_pulse_strength(t, beats, width=min(0.16, beat_period * 0.35)) if enable_beat_editing else 0.0
        cut = nearest_pulse_strength(t, cut_beats, width=min(0.10, beat_period * 0.25)) if enable_beat_editing else 0.0

        # Geometry: zoom breathes on every beat, punches harder on cut beats.
        scale = 1.0 + (0.04 + 0.06 * pulse_intensity) * pulse + 0.05 * cut
        work = zoom_frame(frame, scale) if scale > 1.0001 else frame

        # Colour grade + per-beat brightness pulse.
        extra_brightness = 1.0 + 0.12 * pulse_intensity * pulse
        graded = apply_color_grade(work.astype(np.float32), style, vignette_mask, extra_brightness)

        # White flash on strong / cut beats.
        flash = flash_strength * max(pulse * 0.5, cut)
        if flash > 1e-3:
            graded = graded * (1.0 - flash) + 255.0 * flash

        out_frame = np.clip(graded, 0, 255).astype(np.uint8)

        if show_overlay:
            phase = (t / beat_period) % 1.0 if beat_period > 0 else 0.0
            draw_waveform(out_frame, wave, phase, pulse)
            draw_text(out_frame, heartbeat_bpm, title_text, str(style.get("mood_zh") or style.get("mood", "")), pulse)

        sub_text, sub_alpha = active_subtitle(t, subtitles)
        if sub_text:
            draw_subtitle(out_frame, sub_text, sub_alpha)

        writer.write(out_frame)
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
        "beat_editing_enabled": bool(enable_beat_editing),
        "beats_per_cut": int(style["beats_per_cut"]),
        "cut_accent_count": int(len(cut_beats)),
        "overlay_shown": bool(show_overlay),
        "subtitle_lines": int(len(subtitles)) if subtitles else 0,
        "grade_name": style.get("grade_name"),
        "mood": style.get("mood"),
        "style_profile": style,
    }
    (out_dir / "video_render_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
