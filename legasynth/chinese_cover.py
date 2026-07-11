"""Feature C: sing the song's lyrics in Chinese and swap them in for the original vocal.

Pipeline (each stage degrades gracefully if a heavy/optional dependency is missing):

  1. separation      original song -> instrumental + rough vocal estimate
                     (karaoke centre-channel removal by default; Demucs if installed)
  2. lyrics          Chinese lyrics come from the user (plain lines or timed LRC);
                     if untimed, lines are spread across the detected vocal span
  3. melody          per-line target pitch is read from the original vocal's F0 (librosa.pyin)
  4. singing         each Chinese line is spoken by edge-tts, then time-stretched to its slot
                     and pitch-shifted onto the song's melody note -> pitched Chinese "singing"
                     (an offline sine-hum placeholder is used if edge-tts / network is unavailable)
  5. remix           synthesized Chinese vocal + instrumental -> cover_audio.wav

The cover_audio.wav is what the rest of the pipeline mixes the heartbeat into and renders.
This is an MVP: line-level pitch following, not studio singing. Syllable-level pitch and
truly "singable" translation are noted as future work in the report.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf


DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"


# --- stage 1: separation -----------------------------------------------------

def _load_stereo(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:, :2].astype(np.float32), int(sr)


def separate_vocal_instrumental(
    song_wav: str | os.PathLike[str], out_dir: Path, method: str = "auto"
) -> dict[str, Any]:
    """Split into instrumental + rough vocal estimate.

    Default is karaoke-style centre removal: anything panned dead-centre (typically
    the lead vocal) is cancelled by L-R, leaving the panned accompaniment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stereo, sr = _load_stereo(song_wav)
    left, right = stereo[:, 0], stereo[:, 1]

    used = "center_channel_removal"
    is_mono_like = float(np.mean((left - right) ** 2)) < 1e-6
    if is_mono_like:
        # Nothing to cancel: keep the original as instrumental, overlay the new vocal on top.
        instrumental = stereo.copy()
        used = "mono_passthrough_no_separation"
    else:
        side = (left - right) * 0.5  # centred vocal cancels out
        instrumental = np.stack([side, side], axis=1).astype(np.float32)

    vocal_estimate = ((left + right) * 0.5).astype(np.float32)  # centre = vocal + centred instruments

    inst_path = out_dir / "instrumental.wav"
    voc_path = out_dir / "vocal_estimate.wav"
    sf.write(str(inst_path), instrumental, sr)
    sf.write(str(voc_path), vocal_estimate, sr)
    return {
        "method": used,
        "sample_rate": sr,
        "duration_seconds": float(len(left) / sr) if sr else 0.0,
        "instrumental_path": str(inst_path),
        "vocal_estimate_path": str(voc_path),
    }


# --- stage 2: lyrics ---------------------------------------------------------

_LRC_LINE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")


def parse_lyrics(
    lyrics_text: str | None,
    lrc_text: str | None,
    vocal_estimate_path: str | os.PathLike[str],
    duration: float,
) -> list[dict[str, Any]]:
    """Return a list of {text, start, end} lines with timing in seconds."""
    if lrc_text and lrc_text.strip():
        lines = _parse_lrc(lrc_text, duration)
        if lines:
            return lines

    raw = [ln.strip() for ln in (lyrics_text or "").splitlines() if ln.strip()]
    if not raw:
        return []
    span_start, span_end = _detect_vocal_span(vocal_estimate_path, duration)
    return _distribute_lines(raw, span_start, span_end)


def _parse_lrc(lrc_text: str, duration: float) -> list[dict[str, Any]]:
    entries: list[tuple[float, str]] = []
    for line in lrc_text.splitlines():
        m = _LRC_LINE.match(line.strip())
        if m:
            t = int(m.group(1)) * 60.0 + float(m.group(2))
            text = m.group(3).strip()
            if text:
                entries.append((t, text))
    entries.sort(key=lambda e: e[0])
    lines: list[dict[str, Any]] = []
    for i, (start, text) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else min(duration, start + 4.0)
        lines.append({"text": text, "start": float(start), "end": float(max(start + 0.4, end))})
    return lines


def _detect_vocal_span(vocal_path: str | os.PathLike[str], duration: float) -> tuple[float, float]:
    try:
        y, sr = librosa.load(str(vocal_path), sr=None, mono=True)
    except Exception:
        return 0.0, duration
    if len(y) == 0:
        return 0.0, duration
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    thresh = float(np.percentile(rms, 60)) * 0.5
    active = np.where(rms > thresh)[0]
    if len(active) < 2:
        return 0.0, duration
    return float(times[active[0]]), float(min(duration, times[active[-1]]))


def _distribute_lines(raw_lines: list[str], span_start: float, span_end: float) -> list[dict[str, Any]]:
    n = len(raw_lines)
    span = max(0.5, span_end - span_start)
    slot = span / n
    lines = []
    for i, text in enumerate(raw_lines):
        start = span_start + i * slot
        end = start + slot * 0.9  # small gap between lines
        lines.append({"text": text, "start": float(start), "end": float(end)})
    return lines


# --- stage 3: melody ---------------------------------------------------------

def extract_melody_f0(vocal_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Voiced fundamental-frequency contour of the original vocal via pyin."""
    y, sr = librosa.load(str(vocal_path), sr=None, mono=True)
    if len(y) == 0:
        return {"sr": int(sr), "times": np.array([]), "f0": np.array([])}
    try:
        f0, voiced, _ = librosa.pyin(
            y, sr=sr, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C6"))
        )
    except Exception:
        return {"sr": int(sr), "times": np.array([]), "f0": np.array([])}
    times = librosa.times_like(f0, sr=sr)
    return {"sr": int(sr), "times": times, "f0": f0}


def _fold_to_singing_range(hz: float, low: float = 165.0, high: float = 440.0) -> float:
    """Octave-shift a pitch into a comfortable sung range so pitch-shifts stay small.

    pyin sometimes locks onto a bass note or drops an octave; folding avoids the
    muddy result of shifting a bright TTS voice down to ~65 Hz.
    """
    if hz <= 0:
        return hz
    while hz < low:
        hz *= 2.0
    while hz > high:
        hz /= 2.0
    return hz


def _target_pitch_for_window(melody: dict[str, Any], start: float, end: float) -> float | None:
    times, f0 = melody.get("times"), melody.get("f0")
    if times is None or len(times) == 0:
        return None
    mask = (times >= start) & (times <= end)
    segment = f0[mask]
    segment = segment[np.isfinite(segment)]
    if len(segment) == 0:
        return None
    return _fold_to_singing_range(float(np.median(segment)))


# --- stage 4: Chinese singing synthesis -------------------------------------

def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _edge_tts_to_wav(text: str, voice: str, wav_path: Path, sr: int) -> bool:
    """Synthesize `text` with edge-tts (online) and decode to a mono wav at `sr`."""
    try:
        import edge_tts
    except Exception:
        return False
    mp3_path = wav_path.with_suffix(".mp3")

    async def _run() -> None:
        await edge_tts.Communicate(text, voice).save(str(mp3_path))

    try:
        asyncio.run(_run())
        if not mp3_path.exists() or mp3_path.stat().st_size == 0:
            return False
        subprocess.run(
            [_ffmpeg(), "-y", "-i", str(mp3_path), "-ac", "1", "-ar", str(sr), str(wav_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return wav_path.exists()
    except Exception:
        return False
    finally:
        try:
            mp3_path.unlink()
        except OSError:
            pass


def _fit_segment(y: np.ndarray, sr: int, target_len: int, target_hz: float | None, source_hz_hint: float | None) -> np.ndarray:
    """Time-stretch `y` to target_len samples and pitch-shift toward target_hz."""
    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)
    # Pitch shift toward the melody note.
    if target_hz and target_hz > 0:
        src_hz = source_hz_hint
        if not src_hz or src_hz <= 0:
            try:
                f0, _, _ = librosa.pyin(y, sr=sr, fmin=80.0, fmax=500.0)
                f0 = f0[np.isfinite(f0)]
                src_hz = float(np.median(f0)) if len(f0) else None
            except Exception:
                src_hz = None
        if src_hz and src_hz > 0:
            n_steps = 12.0 * np.log2(target_hz / src_hz)
            n_steps = float(np.clip(n_steps, -18.0, 18.0))
            if abs(n_steps) > 0.3:
                try:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                except Exception:
                    pass
    # Time-stretch to fill the slot.
    cur = len(y)
    if cur > 8 and target_len > 8:
        rate = cur / target_len
        if 0.25 < rate < 4.0 and abs(rate - 1.0) > 0.05:
            try:
                y = librosa.effects.time_stretch(y, rate=rate)
            except Exception:
                pass
    # Pad/trim to exact length with short fades.
    if len(y) < target_len:
        y = np.concatenate([y, np.zeros(target_len - len(y), dtype=np.float32)])
    else:
        y = y[:target_len]
    fade = min(256, target_len // 8)
    if fade > 1:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y.astype(np.float32)


def _sine_hum(target_len: int, sr: int, target_hz: float | None) -> np.ndarray:
    """Offline placeholder 'voice' when edge-tts is unavailable: a soft vowel-like hum."""
    hz = target_hz if (target_hz and target_hz > 0) else 220.0
    t = np.arange(target_len) / sr
    tone = 0.5 * np.sin(2 * np.pi * hz * t) + 0.25 * np.sin(2 * np.pi * 2 * hz * t)
    tremolo = 1.0 + 0.15 * np.sin(2 * np.pi * 5.0 * t)
    y = (tone * tremolo).astype(np.float32)
    fade = min(512, target_len // 6)
    if fade > 1:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y * 0.4


def synthesize_chinese_vocal(
    lines: list[dict[str, Any]],
    melody: dict[str, Any],
    sr: int,
    total_len: int,
    voice: str,
    work_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    track = np.zeros(total_len, dtype=np.float32)
    tts_ok = 0
    placeholder = 0
    per_line = []
    for i, line in enumerate(lines):
        start_i = int(max(0, line["start"] * sr))
        end_i = int(min(total_len, line["end"] * sr))
        target_len = end_i - start_i
        if target_len <= sr // 10:
            continue
        target_hz = _target_pitch_for_window(melody, line["start"], line["end"])
        wav_path = work_dir / f"line_{i:03d}.wav"
        if _edge_tts_to_wav(line["text"], voice, wav_path, sr):
            y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
            seg = _fit_segment(y.astype(np.float32), sr, target_len, target_hz, None)
            tts_ok += 1
            source = "edge_tts"
        else:
            seg = _sine_hum(target_len, sr, target_hz)
            placeholder += 1
            source = "sine_placeholder"
        track[start_i:end_i] += seg[:target_len]
        per_line.append({
            "index": i, "text": line["text"], "start": round(line["start"], 3),
            "end": round(line["end"], 3), "target_hz": round(target_hz, 2) if target_hz else None,
            "source": source,
        })
    peak = float(np.max(np.abs(track))) if len(track) else 0.0
    if peak > 1e-6:
        track = track / peak * 0.85
    info = {
        "line_count": len(lines),
        "tts_lines": tts_ok,
        "placeholder_lines": placeholder,
        "voice": voice,
        "synthesis_backend": "edge_tts" if tts_ok else ("sine_placeholder" if placeholder else "none"),
        "per_line": per_line,
    }
    return track.astype(np.float32), info


# --- stage 5: orchestration --------------------------------------------------

def generate_chinese_cover(
    song_wav: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    lyrics: str | None = None,
    lrc: str | None = None,
    voice: str = DEFAULT_VOICE,
    vocal_gain_db: float = -2.0,
    instrumental_gain_db: float = -3.0,
    separation_method: str = "auto",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = separate_vocal_instrumental(song_wav, out_dir, method=separation_method)
    sr = int(sep["sample_rate"])
    duration = float(sep["duration_seconds"])

    lines = parse_lyrics(lyrics, lrc, sep["vocal_estimate_path"], duration)
    instrumental, _ = _load_stereo(sep["instrumental_path"])
    total_len = instrumental.shape[0]

    notes = {"warnings": []}
    if not lines:
        notes["warnings"].append(
            "No Chinese lyrics provided; cover audio is the instrumental only. "
            "Provide `lyrics` (plain lines) or `lrc` (timed) to hear Chinese singing."
        )
        vocal_track = np.zeros(total_len, dtype=np.float32)
        synth_info = {"line_count": 0, "tts_lines": 0, "placeholder_lines": 0,
                      "synthesis_backend": "none", "voice": voice, "per_line": []}
        melody = {"times": np.array([]), "f0": np.array([])}
    else:
        melody = extract_melody_f0(sep["vocal_estimate_path"])
        vocal_track, synth_info = synthesize_chinese_vocal(
            lines, melody, sr, total_len, voice, out_dir / "lines"
        )

    if synth_info.get("synthesis_backend") == "sine_placeholder":
        notes["warnings"].append(
            "edge-tts/network was unavailable, so a sine-hum melody placeholder was used "
            "instead of real Chinese singing. Re-run online for spoken Chinese lyrics."
        )

    voc_gain = 10.0 ** (vocal_gain_db / 20.0)
    inst_gain = 10.0 ** (instrumental_gain_db / 20.0)
    vocal_stereo = np.stack([vocal_track, vocal_track], axis=1) * voc_gain
    mixed = instrumental * inst_gain + vocal_stereo
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed / peak * 0.98

    cover_wav = out_dir / "cover_audio.wav"
    cover_vocal = out_dir / "chinese_vocal.wav"
    sf.write(str(cover_wav), mixed.astype(np.float32), sr)
    sf.write(str(cover_vocal), vocal_track.astype(np.float32), sr)

    report = {
        "song_wav": str(song_wav),
        "separation": sep,
        "lyrics_line_count": len(lines),
        "lyrics_source": "lrc" if (lrc and lrc.strip()) else ("plain_lines" if lyrics else "none"),
        "melody_voiced_frames": int(np.sum(np.isfinite(melody.get("f0", np.array([]))))) if len(melody.get("f0", [])) else 0,
        "synthesis": synth_info,
        "cover_audio_wav": str(cover_wav),
        "chinese_vocal_wav": str(cover_vocal),
        "instrumental_wav": sep["instrumental_path"],
        "lines": lines,
        "notes": notes,
        "future_work": [
            "Syllable-level pitch alignment (currently one melody note per line).",
            "Singable EN->ZH translation with syllable-count and tone constraints.",
            "Neural singing-voice synthesis (e.g. DiffSinger) for natural timbre.",
            "Demucs-based separation for cleaner instrumentals on real songs.",
        ],
    }
    (out_dir / "chinese_cover_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
