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


DEFAULT_EXPORT_PEAK_DBFS = 20.0 * math.log10(0.8)


@dataclass(frozen=True)
class ProcessingParams:
    bandpass_low_hz: float = 25.0
    bandpass_high_hz: float = 160.0
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
    export_peak_dbfs: float = DEFAULT_EXPORT_PEAK_DBFS
    enable_speech_suppression: bool = True
    hpss_margin: float = 2.0
    spectral_noise_percentile: float = 15.0
    spectral_reduction_strength: float = 1.15
    spectral_floor_db: float = -30.0
    beat_gate_pre_ms: float = 90.0
    beat_gate_post_ms: float = 300.0
    between_beat_attenuation_db: float = -28.0
    analysis_window_seconds: float = 6.0
    analysis_window_hop_seconds: float = 3.0
    min_consensus_windows: int = 2
    loop_candidate_count: int = 5
    enable_template_confirmation: bool = True
    template_pre_ms: float = 90.0
    template_post_ms: float = 300.0
    template_correlation_threshold: float = 0.35
    min_template_beats: int = 4


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
    filename: str,
    data: bytes,
    params: ProcessingParams | None = None,
    manual_beat_times: np.ndarray | list[float] | None = None,
    manual_loop_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    params = params or ProcessingParams()
    sr, raw, source_info = read_audio_bytes(filename, data)
    mono = to_mono_float(raw)
    duration = float(len(mono) / sr) if sr else 0.0

    dc_removed = remove_dc(mono)
    peak = np.max(np.abs(dc_removed)) if len(dc_removed) else 0.0
    if peak > 0:
        cleaned_for_analysis = dc_removed / peak
    else:
        cleaned_for_analysis = dc_removed.copy()

    band_limited = bandpass(cleaned_for_analysis, sr, params.bandpass_low_hz, params.bandpass_high_hz)
    filtered = suppress_non_heart_content(band_limited, sr, params)
    envelope = extract_envelope(filtered, sr, params.envelope_lowpass_hz)
    bpm_info, window_analysis = estimate_bpm_with_consensus(envelope, sr, params)
    candidate_beat_times = detect_beats(envelope, sr, bpm_info["period_seconds"], params)
    beat_times, template_info, template_analysis, template_waveform = confirm_beats_with_template(
        filtered, candidate_beat_times, sr, bpm_info["period_seconds"], params
    )
    normalized_manual_beats = normalize_manual_beat_times(manual_beat_times, duration)
    if normalized_manual_beats is not None:
        beat_times = normalized_manual_beats
        template_info = {
            **template_info,
            "confirmed_count": int(len(beat_times)),
            "confirmation_fraction": float(len(beat_times) / len(candidate_beat_times))
            if len(candidate_beat_times)
            else 0.0,
            "method": "manual_beat_times",
        }
    ibi = np.diff(beat_times)
    manual_loop = normalize_manual_loop_range(manual_loop_range, duration)
    if manual_loop is not None:
        loop_info = make_manual_loop_info(beat_times, manual_loop)
        loop_candidates = [{**loop_info, "rank": 1}]
    else:
        loop_info, loop_candidates = rank_loop_candidates(
            beat_times,
            envelope,
            sr,
            duration,
            params.target_loop_beats,
            bpm_info["period_seconds"],
            params.loop_candidate_count,
        )
    manual_corrections = {
        "beat_times_overridden": normalized_manual_beats is not None,
        "loop_range_overridden": manual_loop is not None,
        "manual_beat_count": int(len(normalized_manual_beats)) if normalized_manual_beats is not None else None,
        "manual_loop_range_seconds": list(manual_loop) if manual_loop is not None else None,
    }
    cleaned = apply_beat_synchronous_gate(filtered, sr, beat_times, params)
    loop_audio, loop_audio_info = cut_loop(cleaned, sr, loop_info, params)

    # Use one headroom-preserving target for every exported WAV.
    export_target_peak = peak_from_dbfs(params.export_peak_dbfs)
    filtered_audio = normalize_for_wav(cleaned, target_peak=export_target_peak)
    cleaned_audio = normalize_for_wav(cleaned, target_peak=export_target_peak)
    loop_audio = normalize_for_wav(loop_audio, target_peak=export_target_peak)

    quality = compute_quality(mono, cleaned, sr, source_info, params, beat_times)
    recording_quality = assess_recording_quality(
        mono, filtered, envelope, beat_times, sr, bpm_info, window_analysis, quality, template_info
    )
    tempo_summary = build_summary(
        filename=filename,
        sr=sr,
        duration=duration,
        source_info=source_info,
        quality=quality,
        recording_quality=recording_quality,
        bpm_info=bpm_info,
        beat_times=beat_times,
        ibi=ibi,
        loop_info={**loop_info, **loop_audio_info},
        template_info=template_info,
        manual_corrections=manual_corrections,
        params=params,
    )

    stem = safe_stem(filename)
    envelope_df = make_envelope_frame(envelope, sr, params.export_envelope_hz)
    beats_df = make_beats_frame(beat_times)
    ibi_df = make_ibi_frame(beat_times)
    window_analysis_df = pd.DataFrame(window_analysis)
    loop_candidates_df = pd.DataFrame(loop_candidates)
    template_analysis_df = pd.DataFrame(template_analysis)
    template_waveform_df = pd.DataFrame(
        {
            "relative_time_seconds": np.arange(len(template_waveform), dtype=np.float32) / sr
            - params.template_pre_ms / 1000.0,
            "normalized_amplitude": template_waveform,
        }
    )
    diagnostic_png = make_diagnostic_plot(
        stem=stem,
        raw=mono,
        filtered=filtered,
        cleaned=cleaned,
        envelope=envelope,
        beat_times=beat_times,
        sr=sr,
        loop_info=loop_info,
        ibi=ibi,
        bpm_info=bpm_info,
        template_analysis=template_analysis,
    )

    artifacts = {
        "tempo_summary.json": json.dumps(tempo_summary, indent=2).encode("utf-8"),
        "processing_parameters.json": json.dumps(asdict(params), indent=2).encode("utf-8"),
        "diagnostic_report.md": make_markdown_report(tempo_summary).encode("utf-8"),
        "beat_times.csv": beats_df.to_csv(index=False).encode("utf-8"),
        "ibi.csv": ibi_df.to_csv(index=False).encode("utf-8"),
        "envelope.csv": envelope_df.to_csv(index=False).encode("utf-8"),
        "window_analysis.csv": window_analysis_df.to_csv(index=False).encode("utf-8"),
        "loop_candidates.csv": loop_candidates_df.to_csv(index=False).encode("utf-8"),
        "template_analysis.csv": template_analysis_df.to_csv(index=False).encode("utf-8"),
        "heartbeat_template.csv": template_waveform_df.to_csv(index=False).encode("utf-8"),
        "manual_corrections.json": json.dumps(manual_corrections, indent=2).encode("utf-8"),
        "recording_quality.json": json.dumps(recording_quality, indent=2).encode("utf-8"),
        "cleaned.wav": wav_bytes(sr, cleaned_audio),
        "filtered_detection.wav": wav_bytes(sr, filtered_audio),
        "best_loop.wav": wav_bytes(sr, loop_audio),
        "diagnostic_plot.png": diagnostic_png,
    }

    return {
        "name": filename,
        "input_data": data,
        "stem": stem,
        "sample_rate": sr,
        "params": params,
        "raw": mono,
        "cleaned": cleaned_audio,
        "filtered": filtered_audio,
        "envelope": envelope,
        "beat_times": beat_times,
        "ibi": ibi,
        "window_analysis": window_analysis,
        "loop_candidates": loop_candidates,
        "template_analysis": template_analysis,
        "loop_audio": loop_audio,
        "summary": tempo_summary,
        "recording_quality": recording_quality,
        "artifacts": artifacts,
        "zip_bytes": make_zip_bytes(stem, artifacts),
    }


def make_markdown_report(summary: dict[str, Any]) -> str:
    tempo = summary["tempo"]
    loop = summary["best_loop"]
    quality = summary["quality"]
    recording_quality = summary["recording_quality"]
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
        f"- Speech/noise suppression enabled: {quality['speech_suppression_enabled']}",
        f"- Beat-window coverage: {quality['beat_window_coverage_fraction']:.3f}",
        f"- Recording quality: {recording_quality['grade']} ({recording_quality['score']:.1f}/100)",
        "",
        "## Tempo And Beats",
        f"- Autocorrelation BPM: {tempo['estimated_bpm']:.3f}",
        f"- Autocorrelation confidence: {tempo['autocorr_confidence']:.6f}",
        f"- BPM method: {tempo['method']}",
        f"- Consensus windows: {tempo['consensus_window_count']}/{tempo['window_count']}",
        f"- Template-confirmed beats: {tempo['template_confirmed_beats']}/{tempo['initial_detected_beats']}",
        f"- Template median correlation: {tempo['template_median_correlation']}",
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
        f"- Loop quality score: {format_optional_score(loop['quality_score'])}",
        f"- Manual beat correction: {summary['manual_corrections']['beat_times_overridden']}",
        f"- Manual loop correction: {summary['manual_corrections']['loop_range_overridden']}",
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


def suppress_non_heart_content(x: np.ndarray, sr: int, params: ProcessingParams) -> np.ndarray:
    """Favor short, low-frequency cardiac transients over sustained speech and room noise."""
    if not len(x) or not params.enable_speech_suppression:
        return x.astype(np.float32, copy=True)

    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(max(256, sr * 0.08))))))
    hop_length = max(64, n_fft // 4)
    if len(x) < n_fft:
        return spectral_noise_gate(x, sr, params)

    spectrum = librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop_length, center=True)
    harmonic, percussive = librosa.decompose.hpss(
        spectrum,
        kernel_size=(31, 17),
        margin=(max(1.0, params.hpss_margin), 1.0),
    )
    heart_like = librosa.istft(percussive, hop_length=hop_length, length=len(x))
    return spectral_noise_gate(heart_like, sr, params)


def spectral_noise_gate(x: np.ndarray, sr: int, params: ProcessingParams) -> np.ndarray:
    """Remove the persistent spectral floor without using a prerecorded noise sample."""
    if not len(x):
        return x.astype(np.float32, copy=True)
    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(max(256, sr * 0.08))))))
    hop_length = max(64, n_fft // 4)
    if len(x) < n_fft:
        return x.astype(np.float32, copy=True)

    spectrum = librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop_length, center=True)
    magnitude = np.abs(spectrum)
    noise_floor = np.percentile(magnitude, params.spectral_noise_percentile, axis=1, keepdims=True)
    subtraction = params.spectral_reduction_strength * noise_floor
    floor_gain = float(10.0 ** (params.spectral_floor_db / 20.0))
    gain = np.maximum(1.0 - subtraction / (magnitude + 1e-9), floor_gain)
    gated = librosa.istft(spectrum * gain, hop_length=hop_length, length=len(x))
    return gated.astype(np.float32)


def apply_beat_synchronous_gate(
    x: np.ndarray, sr: int, beat_times: np.ndarray, params: ProcessingParams
) -> np.ndarray:
    """Keep the S1/S2 region of each beat and attenuate audio between cardiac events."""
    if not len(x) or not params.enable_speech_suppression or not len(beat_times):
        return x.astype(np.float32, copy=True)

    floor_gain = float(10.0 ** (params.between_beat_attenuation_db / 20.0))
    mask = np.full(len(x), floor_gain, dtype=np.float32)
    pre = max(0, int(params.beat_gate_pre_ms * sr / 1000.0))
    post = max(1, int(params.beat_gate_post_ms * sr / 1000.0))
    ramp = max(1, min(int(0.025 * sr), (pre + post) // 5))

    for time_seconds in beat_times:
        center = int(round(float(time_seconds) * sr))
        start = max(0, center - pre)
        end = min(len(x), center + post)
        if end <= start:
            continue
        mask[start:end] = 1.0
        left_start = max(0, start - ramp)
        if start > left_start:
            mask[left_start:start] = np.maximum(
                mask[left_start:start],
                np.linspace(floor_gain, 1.0, start - left_start, endpoint=False, dtype=np.float32),
            )
        right_end = min(len(x), end + ramp)
        if right_end > end:
            mask[end:right_end] = np.maximum(
                mask[end:right_end],
                np.linspace(1.0, floor_gain, right_end - end, endpoint=False, dtype=np.float32),
            )
    return (x * mask).astype(np.float32)


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


def estimate_bpm_with_consensus(
    envelope: np.ndarray, sr: int, params: ProcessingParams
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    global_estimate = estimate_period_from_autocorr(envelope, sr, params.min_bpm, params.max_bpm)
    window_samples = max(sr, int(params.analysis_window_seconds * sr))
    hop_samples = max(sr // 2, int(params.analysis_window_hop_seconds * sr))
    windows: list[dict[str, Any]] = []

    if len(envelope) >= window_samples:
        starts = list(range(0, len(envelope) - window_samples + 1, hop_samples))
        final_start = len(envelope) - window_samples
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            segment = envelope[start : start + window_samples]
            estimate = estimate_period_from_autocorr(segment, sr, params.min_bpm, params.max_bpm)
            beats = detect_beats(segment, sr, estimate["period_seconds"], params)
            windows.append(
                {
                    "start_seconds": float(start / sr),
                    "end_seconds": float((start + len(segment)) / sr),
                    "estimated_bpm": float(estimate["estimated_bpm"]),
                    "autocorr_confidence": float(estimate["autocorr_confidence"]),
                    "detected_beats": int(len(beats)),
                    "is_consensus_inlier": False,
                }
            )

    usable = [
        item
        for item in windows
        if item["autocorr_confidence"] >= 0.08 and item["detected_beats"] >= 3
    ]
    consensus_bpm: float | None = None
    bpm_mad: float | None = None
    if len(usable) >= params.min_consensus_windows:
        values = np.asarray([item["estimated_bpm"] for item in usable], dtype=np.float64)
        consensus_bpm = float(np.median(values))
        bpm_mad = float(np.median(np.abs(values - consensus_bpm)))
        tolerance = max(4.0, 3.0 * bpm_mad)
        for item in windows:
            item["is_consensus_inlier"] = bool(
                item["autocorr_confidence"] >= 0.08
                and item["detected_beats"] >= 3
                and abs(item["estimated_bpm"] - consensus_bpm) <= tolerance
            )

    inliers = [item for item in windows if item["is_consensus_inlier"]]
    if consensus_bpm is not None and len(inliers) >= params.min_consensus_windows:
        period = 60.0 / consensus_bpm
        confidence = float(np.mean([item["autocorr_confidence"] for item in inliers]))
        estimate: dict[str, Any] = {
            "period_seconds": period,
            "estimated_bpm": consensus_bpm,
            "autocorr_confidence": confidence,
            "method": "multi_window_median_consensus",
            "global_autocorr_bpm": float(global_estimate["estimated_bpm"]),
            "global_autocorr_confidence": float(global_estimate["autocorr_confidence"]),
            "window_count": int(len(windows)),
            "consensus_window_count": int(len(inliers)),
            "window_bpm_mad": bpm_mad,
        }
    else:
        estimate = {
            **global_estimate,
            "method": "global_autocorrelation_fallback",
            "global_autocorr_bpm": float(global_estimate["estimated_bpm"]),
            "global_autocorr_confidence": float(global_estimate["autocorr_confidence"]),
            "window_count": int(len(windows)),
            "consensus_window_count": 0,
            "window_bpm_mad": None,
        }
    return estimate, windows


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


def confirm_beats_with_template(
    signal_data: np.ndarray,
    candidate_times: np.ndarray,
    sr: int,
    expected_period_seconds: float,
    params: ProcessingParams,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], np.ndarray]:
    candidates = np.asarray(candidate_times, dtype=np.float32)
    pre = max(1, int(params.template_pre_ms * sr / 1000.0))
    post = max(1, int(params.template_post_ms * sr / 1000.0))
    width = pre + post
    segments: list[np.ndarray] = []
    valid_indices: list[int] = []
    for index, time_seconds in enumerate(candidates):
        center = int(round(float(time_seconds) * sr))
        start = center - pre
        end = center + post
        if start < 0 or end > len(signal_data):
            continue
        segment = signal_data[start:end].astype(np.float32)
        segment = segment - float(np.mean(segment))
        norm = float(np.linalg.norm(segment))
        if norm <= 1e-8:
            continue
        segments.append(segment / norm)
        valid_indices.append(index)

    analysis = [
        {
            "candidate_index": int(index),
            "time_seconds": float(time_seconds),
            "correlation": None,
            "is_template_match": True,
            "is_confirmed": True,
            "decision": "not_applied",
        }
        for index, time_seconds in enumerate(candidates)
    ]
    empty_template = np.zeros(width, dtype=np.float32)
    if not params.enable_template_confirmation:
        return candidates, _template_info(False, len(candidates), len(candidates), None, "disabled"), analysis, empty_template
    if len(segments) < params.min_template_beats:
        return candidates, _template_info(True, len(candidates), len(candidates), None, "insufficient_candidates"), analysis, empty_template

    template = np.median(np.stack(segments), axis=0).astype(np.float32)
    template -= float(np.mean(template))
    template_norm = float(np.linalg.norm(template))
    if template_norm <= 1e-8:
        return candidates, _template_info(True, len(candidates), len(candidates), None, "degenerate_template"), analysis, empty_template
    template /= template_norm

    correlations = np.full(len(candidates), np.nan, dtype=np.float32)
    for source_index, segment in zip(valid_indices, segments):
        correlations[source_index] = float(np.dot(segment, template))
    valid_correlations = correlations[np.isfinite(correlations)]
    adaptive_threshold = float(np.percentile(valid_correlations, 20))
    threshold = max(float(params.template_correlation_threshold), adaptive_threshold)
    template_match_mask = np.isfinite(correlations) & (correlations >= threshold)
    expected_count = len(signal_data) / max(1, sr) / max(expected_period_seconds, 1e-6)
    minimum_retained = max(params.min_template_beats, int(math.floor(expected_count * 0.8)))
    candidate_count_is_plausible = len(candidates) <= math.ceil(expected_count * 1.10)
    if candidate_count_is_plausible:
        confirmed_mask = np.ones(len(candidates), dtype=bool)
        decision = "kept_timing_complete"
    elif int(np.sum(template_match_mask)) < minimum_retained:
        confirmed_mask = np.ones(len(candidates), dtype=bool)
        decision = "kept_fallback_would_under_detect"
    else:
        confirmed_mask = template_match_mask
        decision = "template_confirmed"

    for index, row in enumerate(analysis):
        correlation = correlations[index]
        row["correlation"] = float(correlation) if np.isfinite(correlation) else None
        row["is_template_match"] = bool(template_match_mask[index]) if np.isfinite(correlation) else False
        row["is_confirmed"] = bool(confirmed_mask[index])
        row["decision"] = decision if np.isfinite(correlation) else "edge_window_unavailable"
    confirmed = candidates[confirmed_mask]
    info = _template_info(
        True,
        len(candidates),
        len(confirmed),
        float(np.median(valid_correlations)) if len(valid_correlations) else None,
        decision,
    )
    info["correlation_threshold"] = threshold
    info["expected_beat_count"] = expected_count
    info["minimum_retained_beats"] = minimum_retained
    return confirmed.astype(np.float32), info, analysis, template


def _template_info(
    enabled: bool, candidate_count: int, confirmed_count: int, median_correlation: float | None, method: str
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "candidate_count": int(candidate_count),
        "confirmed_count": int(confirmed_count),
        "confirmation_fraction": float(confirmed_count / candidate_count) if candidate_count else 0.0,
        "median_correlation": median_correlation,
        "method": method,
    }


def normalize_manual_beat_times(
    manual_beat_times: np.ndarray | list[float] | None, duration: float
) -> np.ndarray | None:
    if manual_beat_times is None:
        return None
    values = np.asarray(manual_beat_times, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    values = np.unique(np.round(values[(values >= 0.0) & (values <= duration)], decimals=6))
    if len(values) < 2:
        raise ValueError("Manual beat correction must contain at least two valid times within the recording duration.")
    return values.astype(np.float32)


def normalize_manual_loop_range(
    manual_loop_range: tuple[float, float] | None, duration: float
) -> tuple[float, float] | None:
    if manual_loop_range is None:
        return None
    start, end = (float(manual_loop_range[0]), float(manual_loop_range[1]))
    start = float(np.clip(start, 0.0, duration))
    end = float(np.clip(end, 0.0, duration))
    if end - start < 0.1:
        raise ValueError("Manual loop end must be at least 0.1 seconds after its start.")
    return start, end


def make_manual_loop_info(beat_times: np.ndarray, loop_range: tuple[float, float]) -> dict[str, Any]:
    start, end = loop_range
    in_loop = np.asarray(beat_times)[(np.asarray(beat_times) >= start) & (np.asarray(beat_times) <= end)]
    intervals = np.diff(in_loop)
    local_bpm = float(60.0 / np.median(intervals)) if len(intervals) else 0.0
    return {
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "num_beats": int(len(in_loop)),
        "local_bpm": local_bpm,
        "ibi_std_seconds": float(np.std(intervals)) if len(intervals) else None,
        "regularity_score": float(np.std(intervals) / (np.mean(intervals) + 1e-9)) if len(intervals) else None,
        "envelope_snr_db": None,
        "quality_score": None,
        "method": "manual_loop_range",
    }


def choose_best_loop(
    beat_times: np.ndarray, duration: float, target_loop_beats: int, fallback_period: float
) -> dict[str, Any]:
    best, _ = rank_loop_candidates(
        beat_times,
        np.array([], dtype=np.float32),
        1,
        duration,
        target_loop_beats,
        fallback_period,
        1,
    )
    return best


def rank_loop_candidates(
    beat_times: np.ndarray,
    envelope: np.ndarray,
    sr: int,
    duration: float,
    target_loop_beats: int,
    fallback_period: float,
    candidate_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    beats = np.asarray(beat_times, dtype=np.float64)
    n_intervals = max(1, int(target_loop_beats))
    candidates: list[dict[str, Any]] = []
    if len(beats) >= n_intervals + 1:
        for i in range(0, len(beats) - n_intervals):
            local = np.diff(beats[i : i + n_intervals + 1])
            if np.any(local <= 0):
                continue
            regularity = float(np.std(local) / (np.mean(local) + 1e-9))
            start = float(beats[i])
            end = float(beats[i + n_intervals])
            snr_db = estimate_envelope_snr(envelope, sr, start, end, beats[i : i + n_intervals + 1])
            regularity_component = float(np.exp(-7.0 * regularity))
            snr_component = float(np.clip(snr_db / 40.0, 0.0, 1.0))
            quality_score = float(100.0 * (0.75 * regularity_component + 0.25 * snr_component))
            candidates.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": end - start,
                    "num_beats": n_intervals,
                    "local_bpm": float(60.0 / np.median(local)),
                    "ibi_std_seconds": float(np.std(local)),
                    "regularity_score": regularity,
                    "envelope_snr_db": snr_db,
                    "quality_score": quality_score,
                    "method": "regularity_and_envelope_quality",
                }
            )
        if candidates:
            candidates.sort(key=lambda item: item["quality_score"], reverse=True)
            for rank, candidate in enumerate(candidates, start=1):
                candidate["rank"] = rank
            selected = dict(candidates[0])
            selected.pop("rank", None)
            return selected, candidates[: max(1, int(candidate_limit))]

    fallback = {
        "start_seconds": 0.0,
        "end_seconds": min(duration, fallback_period * n_intervals),
        "duration_seconds": max(0.0, min(duration, fallback_period * n_intervals)),
        "num_beats": n_intervals,
        "local_bpm": float(60.0 / fallback_period) if fallback_period > 0 else 0.0,
        "ibi_std_seconds": None,
        "regularity_score": None,
        "envelope_snr_db": None,
        "quality_score": 0.0,
        "method": "fallback_autocorr_period",
    }
    return fallback, [{**fallback, "rank": 1}]


def estimate_envelope_snr(
    envelope: np.ndarray, sr: int, start_seconds: float, end_seconds: float, beat_times: np.ndarray
) -> float:
    if not len(envelope) or sr <= 0:
        return 0.0
    start = max(0, int(round(start_seconds * sr)))
    end = min(len(envelope), int(round(end_seconds * sr)))
    segment = envelope[start:end]
    if not len(segment):
        return 0.0
    baseline = float(np.percentile(segment, 30))
    peak_indices = np.clip(np.round(np.asarray(beat_times) * sr).astype(int), 0, len(envelope) - 1)
    peak_level = float(np.mean(envelope[peak_indices])) if len(peak_indices) else float(np.percentile(segment, 90))
    return float(20.0 * math.log10((peak_level + 1e-6) / (baseline + 1e-6)))


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
    raw_mono: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
    source_info: dict[str, Any],
    params: ProcessingParams,
    beat_times: np.ndarray,
) -> dict[str, Any]:
    rms = float(np.sqrt(np.mean(cleaned * cleaned))) if len(cleaned) else 0.0
    peak = float(np.max(np.abs(cleaned))) if len(cleaned) else 0.0
    clipping_fraction = float(np.mean(np.abs(raw_mono) >= 0.999)) if len(raw_mono) else 0.0
    coverage = beat_window_coverage(len(cleaned), sr, beat_times, params)
    return {
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "dc_offset": float(np.mean(raw_mono)) if len(raw_mono) else 0.0,
        "clipping_fraction": clipping_fraction,
        "is_clipping_suspected": bool(clipping_fraction > 0.0005),
        "duration_seconds": float(len(raw_mono) / sr) if sr else 0.0,
        "channels": source_info["channels"],
        "speech_suppression_enabled": bool(params.enable_speech_suppression),
        "beat_window_coverage_fraction": coverage,
        "between_beat_attenuation_db": float(params.between_beat_attenuation_db),
    }


def beat_window_coverage(
    sample_count: int, sr: int, beat_times: np.ndarray, params: ProcessingParams
) -> float:
    if not sample_count or not sr or not len(beat_times):
        return 0.0
    intervals: list[tuple[int, int]] = []
    pre = max(0, int(params.beat_gate_pre_ms * sr / 1000.0))
    post = max(1, int(params.beat_gate_post_ms * sr / 1000.0))
    for time_seconds in beat_times:
        center = int(round(float(time_seconds) * sr))
        intervals.append((max(0, center - pre), min(sample_count, center + post)))
    intervals.sort()
    covered = 0
    last_end = 0
    for start, end in intervals:
        start = max(start, last_end)
        if end > start:
            covered += end - start
            last_end = end
    return float(covered / sample_count)


def assess_recording_quality(
    raw_mono: np.ndarray,
    filtered: np.ndarray,
    envelope: np.ndarray,
    beat_times: np.ndarray,
    sr: int,
    bpm_info: dict[str, Any],
    window_analysis: list[dict[str, Any]],
    base_quality: dict[str, Any],
    template_info: dict[str, Any],
) -> dict[str, Any]:
    """Produce a conservative recording-usability score, not a medical diagnosis."""
    reasons: list[str] = []
    score = 100.0
    duration = float(len(raw_mono) / sr) if sr else 0.0
    if duration < 8.0:
        score -= 25.0
        reasons.append("Recording is shorter than 8 seconds; BPM consensus is less reliable.")
    if base_quality["is_clipping_suspected"]:
        score -= 25.0
        reasons.append("Input clipping was detected.")

    filtered_rms = float(np.sqrt(np.mean(filtered * filtered))) if len(filtered) else 0.0
    raw_rms = float(np.sqrt(np.mean(raw_mono * raw_mono))) if len(raw_mono) else 0.0
    heart_band_ratio_db = float(20.0 * math.log10((filtered_rms + 1e-9) / (raw_rms + 1e-9)))
    if heart_band_ratio_db < -24.0:
        score -= 18.0
        reasons.append("Very little energy remains in the selected heart-sound band.")

    beat_count = int(len(beat_times))
    expected_beats = duration * float(bpm_info["estimated_bpm"]) / 60.0
    if beat_count < 4 or (expected_beats >= 4 and beat_count < expected_beats * 0.45):
        score -= 20.0
        reasons.append("Too few reliable heartbeat peaks were detected.")

    consensus_windows = int(bpm_info.get("consensus_window_count", 0))
    window_count = int(bpm_info.get("window_count", 0))
    if window_count >= 2 and consensus_windows < 2:
        score -= 18.0
        reasons.append("BPM estimates disagree across recording windows.")
    elif window_count >= 2 and consensus_windows / window_count < 0.5:
        score -= 10.0
        reasons.append("Only a minority of time windows agree on BPM.")

    envelope_spread = float(np.percentile(envelope, 90) - np.percentile(envelope, 20)) if len(envelope) else 0.0
    if envelope_spread < 0.12:
        score -= 12.0
        reasons.append("Heartbeat envelope has low contrast against the residual background.")
    if template_info["enabled"] and template_info["candidate_count"] >= 4:
        if template_info["confirmation_fraction"] < 0.6:
            score -= 12.0
            reasons.append("Many candidate beats do not match the learned heartbeat template.")

    score = float(np.clip(score, 0.0, 100.0))
    if score >= 75.0:
        grade = "good"
    elif score >= 50.0:
        grade = "usable_with_caution"
    else:
        grade = "poor"
    if not reasons:
        reasons.append("No major automated recording-quality issue was detected.")
    return {
        "score": score,
        "grade": grade,
        "is_recommended_for_loop": bool(score >= 50.0 and beat_count >= 4),
        "reasons": reasons,
        "metrics": {
            "duration_seconds": duration,
            "detected_beats": beat_count,
            "expected_beats_from_bpm": expected_beats,
            "heart_band_to_raw_rms_db": heart_band_ratio_db,
            "envelope_contrast": envelope_spread,
            "window_count": window_count,
            "consensus_window_count": consensus_windows,
            "template_confirmation_fraction": template_info["confirmation_fraction"],
        },
    }


def build_summary(
    filename: str,
    sr: int,
    duration: float,
    source_info: dict[str, Any],
    quality: dict[str, Any],
    recording_quality: dict[str, Any],
    bpm_info: dict[str, float],
    beat_times: np.ndarray,
    ibi: np.ndarray,
    loop_info: dict[str, Any],
    template_info: dict[str, Any],
    manual_corrections: dict[str, Any],
    params: ProcessingParams,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "sample_rate": int(sr),
        "duration_seconds": duration,
        "source": source_info,
        "quality": quality,
        "recording_quality": recording_quality,
        "tempo": {
            **bpm_info,
            "detected_beats": int(len(beat_times)),
            "beat_times_seconds": [round(float(v), 6) for v in beat_times],
            "ibi_seconds": [round(float(v), 6) for v in ibi],
            "ibi_mean_seconds": float(np.mean(ibi)) if len(ibi) else None,
            "ibi_std_seconds": float(np.std(ibi)) if len(ibi) else None,
            "picked_bpm_from_median_ibi": float(60.0 / np.median(ibi)) if len(ibi) else None,
            "initial_detected_beats": int(template_info["candidate_count"]),
            "template_confirmed_beats": int(template_info["confirmed_count"]),
            "template_confirmation_fraction": template_info["confirmation_fraction"],
            "template_median_correlation": template_info["median_correlation"],
        },
        "template_confirmation": template_info,
        "best_loop": loop_info,
        "manual_corrections": manual_corrections,
        "parameters": asdict(params),
    }


def dbfs(value: float) -> float:
    return float(20.0 * math.log10(max(value, 1e-12)))


def format_optional_score(value: float | None) -> str:
    return "manual selection" if value is None else f"{value:.1f}/100"


def peak_from_dbfs(dbfs: float) -> float:
    """Convert a requested export peak to linear amplitude with clipping headroom."""
    safe_dbfs = min(float(dbfs), -0.1)
    return float(10.0 ** (safe_dbfs / 20.0))


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
    cleaned: np.ndarray,
    envelope: np.ndarray,
    beat_times: np.ndarray,
    sr: int,
    loop_info: dict[str, Any],
    ibi: np.ndarray,
    bpm_info: dict[str, float],
    template_analysis: list[dict[str, Any]],
) -> bytes:
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(f"Heartbeat preprocessing diagnostics: {stem}")
    plot_signal(axes[0], raw, sr, "Raw mono waveform")
    plot_signal(axes[1], filtered, sr, "Speech-suppressed detection signal (band-pass + HPSS + spectral gate)")
    plot_signal(axes[2], cleaned, sr, "Cleaned heartbeat audio (beat-synchronous soft gate applied)")

    t_env = np.arange(len(envelope)) / sr if len(envelope) else np.array([])
    axes[3].plot(t_env, envelope, linewidth=0.8, color="#2c7fb8")
    for item in template_analysis:
        color = "#e34a33" if item["is_confirmed"] else "#969696"
        style = "-" if item["is_confirmed"] else "--"
        axes[3].axvline(float(item["time_seconds"]), color=color, alpha=0.5, linewidth=0.8, linestyle=style)
    axes[3].axvspan(loop_info["start_seconds"], loop_info["end_seconds"], color="#31a354", alpha=0.2)
    axes[3].set_title("Envelope, detected beats, and selected loop")
    axes[3].set_ylabel("Envelope")

    if len(ibi):
        axes[4].plot(beat_times[:-1], 60.0 / ibi, marker="o", linewidth=1.0, color="#756bb1")
    axes[4].axhline(bpm_info["estimated_bpm"], color="#636363", linestyle="--", linewidth=1.0)
    axes[4].set_title("Local BPM by inter-beat interval")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("BPM")

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
