from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile


@dataclass(frozen=True)
class ProcessingParams:
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 180.0
    envelope_lowpass_hz: float = 6.0
    min_bpm: float = 40.0
    max_bpm: float = 140.0
    peak_prominence: float = 0.12
    peak_height_percentile: float = 65.0
    double_peak_suppression: float = 0.65
    target_loop_beats: int = 4
    crossfade_ms: float = 12.0
    zero_crossing_search_ms: float = 20.0
    export_envelope_hz: float = 100.0


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or "heartbeat"


def process_audio_file(path: str | os.PathLike[str], params: ProcessingParams | None = None) -> dict[str, Any]:
    path = Path(path)
    return process_audio_bytes(path.name, path.read_bytes(), params=params)


def process_wav_file(path: str | os.PathLike[str], params: ProcessingParams | None = None) -> dict[str, Any]:
    return process_audio_file(path, params=params)


def process_audio_bytes(
    filename: str, data: bytes, params: ProcessingParams | None = None
) -> dict[str, Any]:
    params = params or ProcessingParams()
    sr, raw, source_info = read_audio_bytes(filename, data)
    mono = to_mono_float(raw)
    duration = float(len(mono) / sr) if sr else 0.0

    cleaned = remove_dc(mono)
    peak = np.max(np.abs(cleaned)) if len(cleaned) else 0.0
    if peak > 0:
        cleaned_for_analysis = cleaned / peak
    else:
        cleaned_for_analysis = cleaned.copy()

    filtered = bandpass(cleaned_for_analysis, sr, params.bandpass_low_hz, params.bandpass_high_hz)
    envelope = extract_envelope(filtered, sr, params.envelope_lowpass_hz)
    bpm_info = estimate_period_from_autocorr(envelope, sr, params.min_bpm, params.max_bpm)
    beat_times = detect_beats(envelope, sr, bpm_info["period_seconds"], params)
    ibi = np.diff(beat_times)
    loop_info = choose_best_loop(beat_times, duration, params.target_loop_beats, bpm_info["period_seconds"])
    loop_audio, loop_audio_info = cut_loop(cleaned, sr, loop_info, params)

    filtered_audio = normalize_for_wav(filtered, target_peak=0.8)
    cleaned_audio = normalize_for_wav(cleaned, target_peak=0.8)
    loop_audio = normalize_for_wav(loop_audio, target_peak=0.8)

    quality = compute_quality(mono, cleaned, sr, source_info)
    tempo_summary = build_summary(
        filename=filename,
        sr=sr,
        duration=duration,
        source_info=source_info,
        quality=quality,
        bpm_info=bpm_info,
        beat_times=beat_times,
        ibi=ibi,
        loop_info={**loop_info, **loop_audio_info},
        params=params,
    )

    stem = safe_stem(filename)
    envelope_df = make_envelope_frame(envelope, sr, params.export_envelope_hz)
    beats_df = make_beats_frame(beat_times)
    ibi_df = make_ibi_frame(beat_times)
    diagnostic_png = make_diagnostic_plot(
        stem=stem,
        raw=mono,
        filtered=filtered,
        envelope=envelope,
        beat_times=beat_times,
        sr=sr,
        loop_info=loop_info,
        ibi=ibi,
        bpm_info=bpm_info,
    )

    artifacts = {
        "tempo_summary.json": json.dumps(tempo_summary, indent=2).encode("utf-8"),
        "processing_parameters.json": json.dumps(asdict(params), indent=2).encode("utf-8"),
        "diagnostic_report.md": make_markdown_report(tempo_summary).encode("utf-8"),
        "beat_times.csv": beats_df.to_csv(index=False).encode("utf-8"),
        "ibi.csv": ibi_df.to_csv(index=False).encode("utf-8"),
        "envelope.csv": envelope_df.to_csv(index=False).encode("utf-8"),
        "cleaned.wav": wav_bytes(sr, cleaned_audio),
        "filtered_detection.wav": wav_bytes(sr, filtered_audio),
        "best_loop.wav": wav_bytes(sr, loop_audio),
        "diagnostic_plot.png": diagnostic_png,
    }

    return {
        "name": filename,
        "stem": stem,
        "sample_rate": sr,
        "raw": mono,
        "cleaned": cleaned_audio,
        "filtered": filtered_audio,
        "envelope": envelope,
        "beat_times": beat_times,
        "ibi": ibi,
        "loop_audio": loop_audio,
        "summary": tempo_summary,
        "artifacts": artifacts,
        "zip_bytes": make_zip_bytes(stem, artifacts),
    }


def make_markdown_report(summary: dict[str, Any]) -> str:
    tempo = summary["tempo"]
    loop = summary["best_loop"]
    quality = summary["quality"]
    lines = [
        f"# Heartbeat Diagnostic Report: {summary['filename']}",
        "",
        "## Source",
        f"- Sample rate: {summary['sample_rate']} Hz",
        f"- Duration: {summary['duration_seconds']:.3f} s",
        f"- Channels: {summary['source']['channels']}",
        f"- Input format: {summary['source'].get('format', 'unknown')}",
        "",
        "## Signal Quality",
        f"- RMS: {quality['rms_dbfs']:.2f} dBFS",
        f"- Peak: {quality['peak_dbfs']:.2f} dBFS",
        f"- DC offset: {quality['dc_offset']:.8f}",
        f"- Clipping fraction: {quality['clipping_fraction']:.8f}",
        f"- Clipping suspected: {quality['is_clipping_suspected']}",
        "",
        "## Tempo And Beats",
        f"- Autocorrelation BPM: {tempo['estimated_bpm']:.3f}",
        f"- Autocorrelation confidence: {tempo['autocorr_confidence']:.6f}",
        f"- Detected beats: {tempo['detected_beats']}",
        f"- Median-IBI BPM: {tempo['picked_bpm_from_median_ibi']}",
        f"- IBI mean: {tempo['ibi_mean_seconds']}",
        f"- IBI std: {tempo['ibi_std_seconds']}",
        "",
        "## Selected Loop",
        f"- Method: {loop['method']}",
        f"- Start: {loop['start_seconds']:.6f} s",
        f"- End: {loop['end_seconds']:.6f} s",
        f"- Duration: {loop['duration_seconds']:.6f} s",
        f"- Beats: {loop['num_beats']}",
        f"- Local BPM: {loop['local_bpm']:.3f}",
        f"- Regularity score: {loop['regularity_score']}",
        "",
        "## Processing Parameters",
    ]
    for key, value in summary["parameters"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def process_wav_bytes(
    filename: str, data: bytes, params: ProcessingParams | None = None
) -> dict[str, Any]:
    return process_audio_bytes(filename, data, params=params)


def read_audio_bytes(filename: str, data: bytes) -> tuple[int, np.ndarray, dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".wav":
        sr, raw, info = read_wav_bytes(data)
        info["format"] = "wav"
        return sr, raw, info
    if suffix == ".mp3":
        return read_compressed_audio_bytes(filename, data)
    raise ValueError(f"Unsupported heartbeat audio type: {suffix or '<none>'}. Use .wav or .mp3.")


def read_wav_bytes(data: bytes) -> tuple[int, np.ndarray, dict[str, Any]]:
    sr, raw = wavfile.read(io.BytesIO(data))
    info = {
        "sample_rate": int(sr),
        "dtype": str(raw.dtype),
        "shape": list(raw.shape),
        "channels": int(raw.shape[1]) if raw.ndim == 2 else 1,
        "samples": int(raw.shape[0]),
    }
    return int(sr), raw, info


def read_compressed_audio_bytes(filename: str, data: bytes) -> tuple[int, np.ndarray, dict[str, Any]]:
    suffix = Path(filename).suffix.lower() or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=False)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if y.ndim == 1:
        raw = y.astype(np.float32)
        channels = 1
        samples = int(raw.shape[0])
        shape = [samples]
    else:
        raw = y.T.astype(np.float32)
        channels = int(raw.shape[1])
        samples = int(raw.shape[0])
        shape = list(raw.shape)
    info = {
        "sample_rate": int(sr),
        "dtype": "float32",
        "shape": shape,
        "channels": channels,
        "samples": samples,
        "format": suffix.lstrip("."),
    }
    return int(sr), raw, info


def to_mono_float(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 2:
        raw = raw.astype(np.float32).mean(axis=1)
    if np.issubdtype(raw.dtype, np.integer):
        max_abs = float(max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max))
        x = raw.astype(np.float32) / max_abs
    elif np.issubdtype(raw.dtype, np.floating):
        x = raw.astype(np.float32)
        finite_peak = float(np.nanmax(np.abs(x))) if len(x) else 0.0
        if finite_peak > 1.5:
            x = x / finite_peak
    else:
        raise ValueError(f"Unsupported WAV dtype: {raw.dtype}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def remove_dc(x: np.ndarray) -> np.ndarray:
    if not len(x):
        return x.copy()
    return (x - float(np.mean(x))).astype(np.float32)


def bandpass(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    if len(x) < 32:
        return x.copy()
    nyquist = sr / 2.0
    low = max(1.0, min(float(low_hz), nyquist * 0.8))
    high = max(low + 1.0, min(float(high_hz), nyquist * 0.95))
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    try:
        y = signal.sosfiltfilt(sos, x)
    except ValueError:
        y = signal.sosfilt(sos, x)
    return y.astype(np.float32)


def extract_envelope(x: np.ndarray, sr: int, lowpass_hz: float) -> np.ndarray:
    if not len(x):
        return x.copy()
    analytic = signal.hilbert(x)
    env = np.abs(analytic).astype(np.float32)
    cutoff = max(0.5, min(float(lowpass_hz), sr / 2.0 * 0.8))
    sos = signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    try:
        env = signal.sosfiltfilt(sos, env)
    except ValueError:
        env = signal.sosfilt(sos, env)
    lo = float(np.percentile(env, 10))
    hi = float(np.percentile(env, 95))
    env = (env - lo) / (hi - lo + 1e-9)
    return np.clip(env, 0.0, None).astype(np.float32)


def estimate_period_from_autocorr(
    envelope: np.ndarray, sr: int, min_bpm: float, max_bpm: float
) -> dict[str, float]:
    if len(envelope) < sr:
        fallback_period = 60.0 / 75.0
        return {
            "period_seconds": fallback_period,
            "estimated_bpm": 60.0 / fallback_period,
            "autocorr_confidence": 0.0,
        }
    x = envelope.astype(np.float64)
    x = x - np.mean(x)
    ac = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1 :]
    ac0 = float(ac[0]) if len(ac) else 0.0
    lo = max(1, int((60.0 / max_bpm) * sr))
    hi = min(len(ac) - 1, int((60.0 / min_bpm) * sr))
    if hi <= lo or ac0 <= 0:
        fallback_period = 60.0 / 75.0
        return {
            "period_seconds": fallback_period,
            "estimated_bpm": 60.0 / fallback_period,
            "autocorr_confidence": 0.0,
        }
    local = ac[lo:hi]
    lag = lo + int(np.argmax(local))
    period = float(lag / sr)
    confidence = float(max(ac[lag] / (ac0 + 1e-12), 0.0))
    return {
        "period_seconds": period,
        "estimated_bpm": float(60.0 / period),
        "autocorr_confidence": confidence,
    }


def detect_beats(
    envelope: np.ndarray, sr: int, period_seconds: float, params: ProcessingParams
) -> np.ndarray:
    if not len(envelope):
        return np.array([], dtype=np.float32)
    env = envelope.copy()
    if np.max(env) > np.min(env):
        env = (env - np.min(env)) / (np.max(env) - np.min(env))
    min_distance = max(0.25, params.double_peak_suppression * period_seconds)
    distance_samples = max(1, int(min_distance * sr))
    height = float(np.percentile(env, params.peak_height_percentile))
    peaks, _ = signal.find_peaks(
        env,
        distance=distance_samples,
        prominence=max(0.0, params.peak_prominence),
        height=height,
    )
    times = peaks.astype(np.float64) / float(sr)
    return times.astype(np.float32)


def choose_best_loop(
    beat_times: np.ndarray, duration: float, target_loop_beats: int, fallback_period: float
) -> dict[str, Any]:
    beats = np.asarray(beat_times, dtype=np.float64)
    n_intervals = max(1, int(target_loop_beats))
    if len(beats) >= n_intervals + 1:
        best: tuple[float, int, np.ndarray] | None = None
        for i in range(0, len(beats) - n_intervals):
            local = np.diff(beats[i : i + n_intervals + 1])
            if np.any(local <= 0):
                continue
            score = float(np.std(local) / (np.mean(local) + 1e-9))
            if best is None or score < best[0]:
                best = (score, i, local)
        if best is not None:
            score, i, local = best
            start = float(beats[i])
            end = float(beats[i + n_intervals])
            local_bpm = float(60.0 / np.median(local))
            return {
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
                "num_beats": n_intervals,
                "local_bpm": local_bpm,
                "ibi_std_seconds": float(np.std(local)),
                "regularity_score": score,
                "method": "lowest_ibi_variance",
            }
    start = 0.0
    end = min(duration, fallback_period * n_intervals)
    return {
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": max(0.0, end - start),
        "num_beats": n_intervals,
        "local_bpm": float(60.0 / fallback_period) if fallback_period > 0 else 0.0,
        "ibi_std_seconds": None,
        "regularity_score": None,
        "method": "fallback_autocorr_period",
    }


def cut_loop(
    x: np.ndarray, sr: int, loop_info: dict[str, Any], params: ProcessingParams
) -> tuple[np.ndarray, dict[str, Any]]:
    if not len(x):
        return x.copy(), {"adjusted_start_seconds": 0.0, "adjusted_end_seconds": 0.0, "boundary_jump": 0.0}
    start = int(max(0, min(len(x) - 1, round(loop_info["start_seconds"] * sr))))
    end = int(max(start + 1, min(len(x), round(loop_info["end_seconds"] * sr))))
    search = max(1, int(params.zero_crossing_search_ms * sr / 1000.0))
    start_adj = nearest_zero_crossing(x, start, search)
    end_adj = nearest_zero_crossing(x, end, search)
    if end_adj <= start_adj:
        start_adj, end_adj = start, end
    segment = x[start_adj:end_adj].astype(np.float32).copy()
    boundary_jump = float(abs(segment[-1] - segment[0])) if len(segment) else 0.0
    segment = apply_edge_fades(segment, sr, params.crossfade_ms)
    return segment, {
        "adjusted_start_seconds": float(start_adj / sr),
        "adjusted_end_seconds": float(end_adj / sr),
        "adjusted_duration_seconds": float(len(segment) / sr),
        "crossfade_ms": float(params.crossfade_ms),
        "boundary_jump_before_fade": boundary_jump,
    }


def nearest_zero_crossing(x: np.ndarray, center: int, radius: int) -> int:
    lo = max(0, center - radius)
    hi = min(len(x) - 1, center + radius)
    if hi <= lo:
        return center
    local = x[lo : hi + 1]
    sign_changes = np.where(np.diff(np.signbit(local)))[0]
    if len(sign_changes):
        candidates = lo + sign_changes
        return int(candidates[np.argmin(np.abs(candidates - center))])
    return int(lo + np.argmin(np.abs(local)))


def apply_edge_fades(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    if not len(x) or fade_ms <= 0:
        return x
    n = min(len(x) // 2, max(1, int(fade_ms * sr / 1000.0)))
    if n <= 1:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def compute_quality(
    raw_mono: np.ndarray, cleaned: np.ndarray, sr: int, source_info: dict[str, Any]
) -> dict[str, Any]:
    rms = float(np.sqrt(np.mean(cleaned * cleaned))) if len(cleaned) else 0.0
    peak = float(np.max(np.abs(cleaned))) if len(cleaned) else 0.0
    clipping_fraction = float(np.mean(np.abs(raw_mono) >= 0.999)) if len(raw_mono) else 0.0
    return {
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "dc_offset": float(np.mean(raw_mono)) if len(raw_mono) else 0.0,
        "clipping_fraction": clipping_fraction,
        "is_clipping_suspected": bool(clipping_fraction > 0.0005),
        "duration_seconds": float(len(raw_mono) / sr) if sr else 0.0,
        "channels": source_info["channels"],
    }


def build_summary(
    filename: str,
    sr: int,
    duration: float,
    source_info: dict[str, Any],
    quality: dict[str, Any],
    bpm_info: dict[str, float],
    beat_times: np.ndarray,
    ibi: np.ndarray,
    loop_info: dict[str, Any],
    params: ProcessingParams,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "sample_rate": int(sr),
        "duration_seconds": duration,
        "source": source_info,
        "quality": quality,
        "tempo": {
            **bpm_info,
            "detected_beats": int(len(beat_times)),
            "beat_times_seconds": [round(float(v), 6) for v in beat_times],
            "ibi_seconds": [round(float(v), 6) for v in ibi],
            "ibi_mean_seconds": float(np.mean(ibi)) if len(ibi) else None,
            "ibi_std_seconds": float(np.std(ibi)) if len(ibi) else None,
            "picked_bpm_from_median_ibi": float(60.0 / np.median(ibi)) if len(ibi) else None,
        },
        "best_loop": loop_info,
        "parameters": asdict(params),
    }


def dbfs(value: float) -> float:
    return float(20.0 * math.log10(max(value, 1e-12)))


def normalize_for_wav(x: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
    if not len(x):
        return x.astype(np.float32)
    peak = float(np.max(np.abs(x)))
    if peak <= 1e-12:
        return x.astype(np.float32)
    return (x.astype(np.float32) / peak * target_peak).clip(-1.0, 1.0)


def wav_bytes(sr: int, x: np.ndarray) -> bytes:
    y = np.asarray(x, dtype=np.float32).clip(-1.0, 1.0)
    pcm = (y * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    wavfile.write(buffer, sr, pcm)
    return buffer.getvalue()


def make_envelope_frame(envelope: np.ndarray, sr: int, export_hz: float) -> pd.DataFrame:
    if not len(envelope):
        return pd.DataFrame({"time_seconds": [], "envelope": []})
    step = max(1, int(sr / max(export_hz, 1.0)))
    idx = np.arange(0, len(envelope), step)
    return pd.DataFrame({"time_seconds": idx / sr, "envelope": envelope[idx]})


def make_beats_frame(beat_times: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"beat_index": np.arange(len(beat_times)), "time_seconds": beat_times})


def make_ibi_frame(beat_times: np.ndarray) -> pd.DataFrame:
    if len(beat_times) < 2:
        return pd.DataFrame({"interval_index": [], "start_time_seconds": [], "ibi_seconds": [], "local_bpm": []})
    ibi = np.diff(beat_times)
    return pd.DataFrame(
        {
            "interval_index": np.arange(len(ibi)),
            "start_time_seconds": beat_times[:-1],
            "ibi_seconds": ibi,
            "local_bpm": 60.0 / ibi,
        }
    )


def make_diagnostic_plot(
    stem: str,
    raw: np.ndarray,
    filtered: np.ndarray,
    envelope: np.ndarray,
    beat_times: np.ndarray,
    sr: int,
    loop_info: dict[str, Any],
    ibi: np.ndarray,
    bpm_info: dict[str, float],
) -> bytes:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
    fig.suptitle(f"Heartbeat preprocessing diagnostics: {stem}")
    plot_signal(axes[0], raw, sr, "Raw mono waveform")
    plot_signal(axes[1], filtered, sr, "Band-pass filtered detection signal")

    t_env = np.arange(len(envelope)) / sr if len(envelope) else np.array([])
    axes[2].plot(t_env, envelope, linewidth=0.8, color="#2c7fb8")
    for bt in beat_times:
        axes[2].axvline(float(bt), color="#e34a33", alpha=0.4, linewidth=0.8)
    axes[2].axvspan(loop_info["start_seconds"], loop_info["end_seconds"], color="#31a354", alpha=0.2)
    axes[2].set_title("Envelope, detected beats, and selected loop")
    axes[2].set_ylabel("Envelope")

    if len(ibi):
        axes[3].plot(beat_times[:-1], 60.0 / ibi, marker="o", linewidth=1.0, color="#756bb1")
    axes[3].axhline(bpm_info["estimated_bpm"], color="#636363", linestyle="--", linewidth=1.0)
    axes[3].set_title("Local BPM by inter-beat interval")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("BPM")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def plot_signal(ax: plt.Axes, x: np.ndarray, sr: int, title: str) -> None:
    if not len(x):
        ax.set_title(title)
        return
    max_points = 6000
    step = max(1, len(x) // max_points)
    idx = np.arange(0, len(x), step)
    ax.plot(idx / sr, x[idx], linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")


def make_zip_bytes(stem: str, artifacts: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in artifacts.items():
            zf.writestr(f"{stem}/{name}", data)
    return buffer.getvalue()


def save_result_to_dir(result: dict[str, Any], base_dir: str | os.PathLike[str]) -> Path:
    out_dir = Path(base_dir) / result["stem"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["artifacts"].items():
        (out_dir / name).write_bytes(data)
    return out_dir


def make_batch_zip(results: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            for name, data in result["artifacts"].items():
                zf.writestr(f"{result['stem']}/{name}", data)
    return buffer.getvalue()
