from __future__ import annotations

import io
import json
import math
import os
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile


DEFAULT_EXPORT_PEAK_DBFS = 20.0 * math.log10(0.8)


@dataclass(frozen=True)
class ProcessingParams:
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 250.0
    envelope_lowpass_hz: float = 6.0
    min_bpm: float = 40.0
    max_bpm: float = 140.0
    peak_prominence: float = 0.12
    peak_height_percentile: float = 65.0
    double_peak_suppression: float = 0.65
    export_envelope_hz: float = 100.0
    export_peak_dbfs: float = DEFAULT_EXPORT_PEAK_DBFS
    enable_denoising: bool = True
    spectral_noise_percentile: float = 15.0
    spectral_reduction_strength: float = 1.15
    spectral_floor_db: float = -30.0
    enable_hum_suppression: bool = True
    hum_detection_threshold_db: float = 12.0
    hum_notch_quality: float = 35.0
    enable_phase_noise_reduction: bool = True
    phase_noise_reduction_strength: float = 0.90
    phase_noise_floor_db: float = -24.0
    phase_quiet_start: float = 0.48
    phase_quiet_end: float = 0.90
    beat_gate_pre_ms: float = 90.0
    beat_gate_post_ms: float = 300.0
    between_beat_attenuation_db: float = -28.0
    enable_cycle_consistency: bool = True
    cycle_min_count: int = 6
    cycle_phase_bins: int = 160
    cycle_outlier_z: float = 3.5
    cycle_outlier_attenuation_db: float = -18.0
    cycle_core_protection_fraction: float = 0.35
    cycle_core_protection_ms: float = 45.0
    cycle_core_gate_guard_ms: float = 20.0
    analysis_window_seconds: float = 6.0
    analysis_window_hop_seconds: float = 3.0
    min_consensus_windows: int = 2
    enable_template_confirmation: bool = True
    template_pre_ms: float = 90.0
    template_post_ms: float = 300.0
    template_correlation_threshold: float = 0.35
    min_template_beats: int = 4
    cleanest_segment_beats: int = 4
    cleanest_candidate_count: int = 5
    cycle_pool_size: int = 16
    cycle_pool_min_correlation: float = 0.20
    segment_zero_crossing_search_ms: float = 20.0
    segment_edge_fade_ms: float = 12.0
    playback_loop_target_rms_dbfs: float = -17.5
    playback_loop_peak_dbfs: float = -1.0
    playback_loop_max_softclip_drive: float = 8.0
    rhythm_match_tolerance_ms: float = 120.0
    rhythm_minimum_match_fraction: float = 0.85
    rhythm_maximum_count_delta_fraction: float = 0.20
    rhythm_maximum_median_timing_error_ms: float = 80.0
    rhythm_maximum_median_ibi_error_fraction: float = 0.12
    focal_cycle_min_count: int = 6
    focal_cycle_rms_ratio_threshold: float = 2.10
    focal_cycle_peak_ratio_threshold: float = 1.90
    focal_cycle_correlation_threshold: float = -0.10
    quality_low_template_correlation: float = 0.35
    quality_high_ibi_cv: float = 0.18


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
    data: bytes | memoryview | BinaryIO,
    params: ProcessingParams | None = None,
    manual_beat_times: np.ndarray | list[float] | None = None,
    max_duration_seconds: float | None = None,
    artifact_profile: str = "full",
    create_zip: bool = True,
) -> dict[str, Any]:
    params = params or ProcessingParams()
    sr, raw, source_info = read_audio_bytes(filename, data)
    mono = to_mono_float(raw)
    duration = float(len(mono) / sr) if sr else 0.0
    if max_duration_seconds is not None:
        if max_duration_seconds <= 0:
            raise ValueError("Maximum duration must be greater than zero.")
        if duration > max_duration_seconds:
            raise ValueError(
                f"WAV duration is {duration:.1f} seconds; the maximum allowed duration is "
                f"{max_duration_seconds:.0f} seconds."
            )

    dc_removed = remove_dc(mono)
    peak = np.max(np.abs(dc_removed)) if len(dc_removed) else 0.0
    if peak > 0:
        cleaned_for_analysis = dc_removed / peak
    else:
        cleaned_for_analysis = dc_removed.copy()

    band_limited = bandpass(cleaned_for_analysis, sr, params.bandpass_low_hz, params.bandpass_high_hz)
    hum_filtered, hum_suppression = suppress_detected_hum(band_limited, sr, params)
    analysis_filtered = suppress_non_heart_content(hum_filtered, sr, params)
    envelope = extract_envelope(analysis_filtered, sr, params.envelope_lowpass_hz)
    bpm_info, window_analysis = estimate_bpm_with_consensus(envelope, sr, params)
    candidate_beat_times = detect_beats(envelope, sr, bpm_info["period_seconds"], params)
    beat_times, template_info, template_analysis, template_waveform = confirm_beats_with_template(
        analysis_filtered, candidate_beat_times, sr, bpm_info["period_seconds"], params
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
    phase_filtered, phase_noise_reduction = phase_aware_noise_reduction(
        analysis_filtered,
        sr,
        beat_times,
        params,
    )
    spectral_filtered = phase_filtered if phase_noise_reduction["applied"] else analysis_filtered
    filtered, cycle_consistency = cycle_consistency_denoise(spectral_filtered, sr, beat_times, params)
    cleaned = apply_beat_synchronous_gate(filtered, sr, beat_times, params, cycle_consistency)
    rhythm_preservation = measure_rhythm_preservation(
        cleaned,
        sr,
        beat_times,
        bpm_info["period_seconds"],
        params,
    )
    focal_cycle_contamination = measure_focal_cycle_contamination(
        spectral_filtered,
        sr,
        beat_times,
        params,
    )

    cleanest_segment, segment_candidates = rank_cleanest_heartbeat_segments(
        beat_times,
        envelope,
        cleaned,
        sr,
        duration,
        params,
        cycle_consistency,
    )
    cleanest_audio, segment_boundary = cut_cleanest_heartbeat_segment(cleaned, sr, cleanest_segment, params)
    cleanest_segment = {**cleanest_segment, **segment_boundary}
    playback_audio, playback_loudness = optimize_loop_playback_loudness(cleanest_audio, params)
    cleanest_segment["playback_loudness"] = playback_loudness

    # Use one headroom-preserving target for every exported WAV.
    export_target_peak = peak_from_dbfs(params.export_peak_dbfs)
    input_reference_audio = normalize_for_wav(mono, target_peak=export_target_peak)
    spectral_filtered_audio = normalize_for_wav(spectral_filtered, target_peak=export_target_peak)
    filtered_audio = normalize_for_wav(filtered, target_peak=export_target_peak)
    cleaned_audio = normalize_for_wav(cleaned, target_peak=export_target_peak)
    cleanest_audio = normalize_for_wav(cleanest_audio, target_peak=export_target_peak)
    s1_times = np.asarray(beat_times, dtype=np.float64)
    s2_times = detect_secondary_heart_sounds(cleaned_audio, sr, s1_times)
    heartbeat_detection_audio = make_heartbeat_detection_mix(
        cleaned_audio,
        sr,
        s1_times,
        s2_times,
    )
    cycle_pool = build_heartbeat_cycle_pool(
        cleaned_audio,
        sr,
        beat_times,
        params,
        focal_cycle_contamination,
    )
    cycle_pool_preview = concatenate_cycle_pool(cycle_pool, sr)

    quality = compute_quality(
        mono,
        spectral_filtered,
        cleaned,
        sr,
        source_info,
        params,
        beat_times,
        cycle_consistency,
        rhythm_preservation,
        focal_cycle_contamination,
    )
    recording_quality = assess_recording_quality(
        mono,
        filtered,
        envelope,
        beat_times,
        sr,
        bpm_info,
        window_analysis,
        quality,
        template_info,
        cycle_consistency,
        params,
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
        template_info=template_info,
        params=params,
    )
    tempo_summary["cleanest_segment"] = cleanest_segment
    tempo_summary["noise_reduction"] = {
        "hum_suppression": hum_suppression,
        "phase_aware": phase_noise_reduction,
    }
    tempo_summary["cycle_pool"] = {
        "selected_count": len(cycle_pool),
        "requested_count": int(params.cycle_pool_size),
        "source_cycle_indices": [int(item["source_cycle_index"]) for item in cycle_pool],
    }
    tempo_summary["heartbeat_events"] = {
        "s1_peak_times_seconds": s1_times.tolist(),
        "s2_peak_times_seconds": s2_times.tolist(),
    }

    stem = safe_stem(filename)
    if artifact_profile not in {"full", "web"}:
        raise ValueError("artifact_profile must be 'full' or 'web'.")
    preview_artifacts = {
        "input_reference.wav": wav_bytes(sr, input_reference_audio),
        "cleaned.wav": wav_bytes(sr, cleaned_audio),
        "cleanest_heartbeat_loop.wav": wav_bytes(sr, cleanest_audio),
        "heartbeat_detection_mix.wav": wav_bytes(sr, heartbeat_detection_audio),
    }
    if artifact_profile == "full":
        preview_artifacts["heartbeat_cycle_pool_preview.wav"] = wav_bytes(
            sr, cycle_pool_preview
        )
    artifacts = dict(preview_artifacts)
    if artifact_profile == "full":
        envelope_df = make_envelope_frame(envelope, sr, params.export_envelope_hz)
        beats_df = make_beats_frame(beat_times)
        ibi_df = make_ibi_frame(beat_times)
        window_analysis_df = pd.DataFrame(window_analysis)
        template_analysis_df = pd.DataFrame(template_analysis)
        segment_candidates_df = pd.DataFrame(segment_candidates)
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
            "template_analysis.csv": template_analysis_df.to_csv(index=False).encode("utf-8"),
            "heartbeat_template.csv": template_waveform_df.to_csv(index=False).encode("utf-8"),
            "recording_quality.json": json.dumps(recording_quality, indent=2).encode("utf-8"),
            "cycle_consistency.json": json.dumps(cycle_consistency, indent=2).encode("utf-8"),
            "rhythm_preservation.json": json.dumps(rhythm_preservation, indent=2).encode("utf-8"),
            "focal_cycle_contamination.json": json.dumps(
                focal_cycle_contamination, indent=2
            ).encode("utf-8"),
            "hum_suppression.json": json.dumps(hum_suppression, indent=2).encode("utf-8"),
            "phase_noise_reduction.json": json.dumps(
                phase_noise_reduction, indent=2
            ).encode("utf-8"),
            "postprocess_beat_times.csv": make_beats_frame(
                np.asarray(
                    rhythm_preservation["processed_beat_times_seconds"],
                    dtype=np.float32,
                )
            ).to_csv(index=False).encode("utf-8"),
            "cleanest_segment.json": json.dumps(cleanest_segment, indent=2).encode("utf-8"),
            "cleanest_segment_candidates.csv": segment_candidates_df.to_csv(index=False).encode("utf-8"),
            **preview_artifacts,
            "spectral_filtered.wav": wav_bytes(sr, spectral_filtered_audio),
            "filtered_detection.wav": wav_bytes(sr, filtered_audio),
            "cleanest_heartbeat_loop_loud.wav": wav_bytes(sr, playback_audio),
            "diagnostic_plot.png": diagnostic_png,
        }

    return {
        "name": filename,
        "input_data": data,
        "stem": stem,
        "sample_rate": sr,
        "params": params,
        "raw": mono,
        "spectral_filtered": spectral_filtered_audio,
        "cleaned": cleaned_audio,
        "filtered": filtered_audio,
        "envelope": envelope,
        "beat_times": beat_times,
        "ibi": ibi,
        "window_analysis": window_analysis,
        "template_analysis": template_analysis,
        "summary": tempo_summary,
        "recording_quality": recording_quality,
        "cycle_consistency": cycle_consistency,
        "rhythm_preservation": rhythm_preservation,
        "focal_cycle_contamination": focal_cycle_contamination,
        "cleanest_segment": cleanest_segment,
        "cleanest_audio": cleanest_audio,
        "cycle_pool": cycle_pool,
        "hum_suppression": hum_suppression,
        "phase_noise_reduction": phase_noise_reduction,
        "s1_times": s1_times,
        "s2_times": s2_times,
        "playback_audio": playback_audio,
        "segment_candidates": segment_candidates,
        "artifacts": artifacts,
        "zip_bytes": make_zip_bytes(stem, artifacts) if create_zip else b"",
    }


def make_markdown_report(summary: dict[str, Any]) -> str:
    tempo = summary["tempo"]
    quality = summary["quality"]
    recording_quality = summary["recording_quality"]
    rhythm = quality["rhythm_preservation"]
    focal = quality["focal_cycle_contamination"]
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
        f"- Heartbeat denoising enabled: {quality['denoising_enabled']}",
        f"- Beat-window coverage: {quality['beat_window_coverage_fraction']:.3f}",
        f"- Inter-beat noise reduction: {quality['interbeat_noise_reduction_db']} dB",
        f"- Heartbeat preservation correlation: {quality['heartbeat_preservation_correlation']}",
        f"- Post-process rhythm preserved: {rhythm['is_preserved']}",
        f"- Post-process beat match: {rhythm['matched_beat_count']}/{rhythm['expected_beat_count']} ({rhythm['matched_fraction']:.3f})",
        f"- Post-process beat count delta: {rhythm['count_delta']}",
        f"- Post-process median timing error: {rhythm['median_timing_error_ms']} ms",
        f"- Post-process median IBI error fraction: {rhythm['median_ibi_error_fraction']}",
        f"- Severe focal contaminated cycles: {focal['severe_cycle_count']}",
        f"- Maximum cycle RMS ratio to median: {focal['max_rms_ratio']}",
        f"- Maximum cycle peak ratio to median: {focal['max_peak_ratio']}",
        f"- Recording quality: {recording_quality['grade']} ({recording_quality['score']:.1f}/100)",
        f"- Needs re-recording: {recording_quality['needs_rerecording']}",
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


def read_audio_bytes(
    filename: str,
    data: bytes | memoryview | BinaryIO,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".wav":
        sr, raw, info = read_wav_bytes(data)
        info["format"] = "wav"
        return sr, raw, info
    raise ValueError(f"Unsupported heartbeat audio type: {suffix or '<none>'}. Use a WAV file.")


def read_wav_bytes(data: bytes | memoryview | BinaryIO) -> tuple[int, np.ndarray, dict[str, Any]]:
    source_buffer = data if hasattr(data, "read") and hasattr(data, "seek") else io.BytesIO(data)
    original_position = source_buffer.tell() if hasattr(source_buffer, "tell") else 0
    try:
        source_buffer.seek(0)
        sr, raw = wavfile.read(source_buffer)
    finally:
        try:
            source_buffer.seek(original_position)
        except (AttributeError, OSError, ValueError):
            pass
    info = {
        "sample_rate": int(sr),
        "dtype": str(raw.dtype),
        "shape": list(raw.shape),
        "channels": int(raw.shape[1]) if raw.ndim == 2 else 1,
        "samples": int(raw.shape[0]),
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
    """Reduce the persistent noise floor without inventing replacement heart sounds."""
    if not len(x) or not params.enable_denoising:
        return x.astype(np.float32, copy=True)

    return spectral_noise_gate(x, sr, params)


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


def suppress_detected_hum(
    x: np.ndarray,
    sr: int,
    params: ProcessingParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only persistent narrow 50/60 Hz tones that clearly exceed their neighbours."""
    values = np.asarray(x, dtype=np.float32)
    report: dict[str, Any] = {
        "enabled": bool(params.enable_hum_suppression),
        "applied": False,
        "detected_frequencies_hz": [],
        "peak_excess_db": {},
        "quality_factor": float(params.hum_notch_quality),
    }
    if not params.enable_hum_suppression or len(values) < max(256, sr // 2):
        return values.copy(), report

    nperseg = min(len(values), max(1024, min(8192, int(sr * 2.0))))
    frequencies, power = signal.welch(values, fs=sr, nperseg=nperseg)
    power_db = 10.0 * np.log10(np.maximum(power, 1e-18))
    detected: list[float] = []
    excess_by_frequency: dict[str, float] = {}
    upper = min(float(params.bandpass_high_hz), sr * 0.45)
    for base in (50.0, 60.0):
        harmonics: list[tuple[float, float]] = []
        harmonic = base
        while harmonic <= upper + 1e-9:
            center = int(np.argmin(np.abs(frequencies - harmonic)))
            neighbourhood = (frequencies >= harmonic - 4.0) & (frequencies <= harmonic + 4.0)
            notch = (frequencies >= harmonic - 0.9) & (frequencies <= harmonic + 0.9)
            reference = power_db[neighbourhood & ~notch]
            if len(reference):
                excess = float(power_db[center] - np.median(reference))
                excess_by_frequency[f"{harmonic:.1f}"] = excess
                if excess >= float(params.hum_detection_threshold_db):
                    harmonics.append((harmonic, excess))
            harmonic += base
        if harmonics and (
            harmonics[0][1] >= float(params.hum_detection_threshold_db) + 3.0
            or len(harmonics) >= 2
        ):
            detected.extend(frequency for frequency, _ in harmonics)

    output = values.astype(np.float64, copy=True)
    for frequency in sorted(set(detected)):
        b, a = signal.iirnotch(
            frequency,
            max(10.0, float(params.hum_notch_quality)),
            fs=sr,
        )
        try:
            output = signal.filtfilt(b, a, output)
        except ValueError:
            output = signal.lfilter(b, a, output)
    report.update(
        {
            "applied": bool(detected),
            "detected_frequencies_hz": [float(value) for value in sorted(set(detected))],
            "peak_excess_db": excess_by_frequency,
        }
    )
    return output.astype(np.float32), report


def phase_aware_noise_reduction(
    x: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    params: ProcessingParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate the noise PSD from quiet cardiac phases and apply a protected Wiener mask."""
    values = np.asarray(x, dtype=np.float32)
    beats = np.asarray(beat_times, dtype=np.float64)
    report: dict[str, Any] = {
        "enabled": bool(params.enable_phase_noise_reduction),
        "applied": False,
        "method": "diastolic-noise-psd-protected-wiener",
        "quiet_frame_count": 0,
        "protected_frame_count": 0,
        "quiet_mean_gain_db": 0.0,
        "protected_mean_gain_db": 0.0,
    }
    if (
        not params.enable_phase_noise_reduction
        or len(values) < 512
        or len(beats) < 4
    ):
        return values.copy(), report

    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(max(256, sr * 0.08))))))
    hop_length = max(64, n_fft // 4)
    spectrum = librosa.stft(values, n_fft=n_fft, hop_length=hop_length, center=True)
    power = np.square(np.abs(spectrum), dtype=np.float64)
    frame_times = librosa.frames_to_time(
        np.arange(spectrum.shape[1]),
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    interval_index = np.searchsorted(beats, frame_times, side="right") - 1
    valid = (interval_index >= 0) & (interval_index < len(beats) - 1)
    phases = np.zeros(len(frame_times), dtype=np.float64)
    valid_indices = interval_index[valid]
    periods = beats[valid_indices + 1] - beats[valid_indices]
    phases[valid] = (
        frame_times[valid] - beats[valid_indices]
    ) / np.maximum(periods, 1e-6)
    quiet = valid & (phases >= float(params.phase_quiet_start)) & (
        phases <= float(params.phase_quiet_end)
    )
    protected = valid & (phases <= min(0.46, float(params.phase_quiet_start)))
    if int(np.sum(quiet)) < 3:
        return values.copy(), report

    noise_power = np.median(power[:, quiet], axis=1, keepdims=True)
    clean_power = np.maximum(power - noise_power, 0.0)
    wiener = clean_power / (clean_power + noise_power + 1e-18)
    floor_gain = float(10.0 ** (params.phase_noise_floor_db / 20.0))
    np.clip(wiener, floor_gain, 1.0, out=wiener)
    strength = float(np.clip(params.phase_noise_reduction_strength, 0.0, 1.5))
    gain = 1.0 - min(strength, 1.0) * (1.0 - wiener)
    if strength > 1.0:
        gain *= np.power(wiener, strength - 1.0)
    np.clip(gain, floor_gain, 1.0, out=gain)
    # S1/S2 frames retain most of their original spectrum. Quiet frames use the
    # full mask, while transition frames interpolate between both behaviours.
    protection_mix = np.ones(len(frame_times), dtype=np.float64)
    protection_mix[protected] = 0.28
    transition = valid & ~quiet & ~protected
    protection_mix[transition] = 0.60
    gain = 1.0 - protection_mix[None, :] * (1.0 - gain)
    gain = signal.medfilt(gain, kernel_size=(1, 3))
    np.clip(gain, floor_gain, 1.0, out=gain)
    output = librosa.istft(
        spectrum * gain,
        hop_length=hop_length,
        length=len(values),
    ).astype(np.float32)

    def mean_gain_db(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(20.0 * np.log10(max(float(np.mean(gain[:, mask])), 1e-12)))

    report.update(
        {
            "applied": True,
            "quiet_frame_count": int(np.sum(quiet)),
            "protected_frame_count": int(np.sum(protected)),
            "quiet_mean_gain_db": mean_gain_db(quiet),
            "protected_mean_gain_db": mean_gain_db(protected),
            "quiet_phase_range": [
                float(params.phase_quiet_start),
                float(params.phase_quiet_end),
            ],
        }
    )
    return output, report


def cycle_consistency_denoise(
    x: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    params: ProcessingParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Attenuate cycle-local envelope outliers while protecting recurrent S1/S2 energy.

    The learned template is used only to derive an attenuation mask. It is never
    copied into the output, so this stage cannot synthesize or repeat a heartbeat.
    """
    beats = np.asarray(beat_times, dtype=np.float64)
    base_info: dict[str, Any] = {
        "enabled": bool(params.enable_cycle_consistency),
        "applied": False,
        "method": "attenuation_only_cycle_envelope",
        "reason": None,
        "cycles_available": max(0, int(len(beats) - 1)),
        "cycles_used": 0,
        "median_period_seconds": None,
        "median_cycle_correlation": None,
        "outlier_fraction": 0.0,
        "mean_gain_db_on_outliers": 0.0,
        "core_protected_fraction": 0.0,
        "core_phase_ranges": [],
    }
    if not len(x) or not params.enable_cycle_consistency:
        base_info["reason"] = "disabled_or_empty"
        return x.astype(np.float32, copy=True), base_info
    if len(beats) < params.cycle_min_count + 1:
        base_info["reason"] = "insufficient_complete_cycles"
        return x.astype(np.float32, copy=True), base_info

    periods = np.diff(beats)
    finite_periods = periods[np.isfinite(periods) & (periods > 0.2)]
    if not len(finite_periods):
        base_info["reason"] = "invalid_cycle_periods"
        return x.astype(np.float32, copy=True), base_info
    median_period = float(np.median(finite_periods))
    phase_bins = max(64, int(params.cycle_phase_bins))
    phase_grid = np.linspace(0.0, 1.0, phase_bins, dtype=np.float64)
    pre_seconds = max(0.0, params.beat_gate_pre_ms / 1000.0)

    energy_window = max(3, int(round(0.018 * sr)))
    kernel = np.ones(energy_window, dtype=np.float64) / energy_window
    energy = np.sqrt(
        np.maximum(signal.fftconvolve(np.square(x, dtype=np.float64), kernel, mode="same"), 0.0)
    )

    cycle_rows: list[dict[str, Any]] = []
    normalized_envelopes: list[np.ndarray] = []
    for index, period in enumerate(periods):
        if not np.isfinite(period) or period < 0.70 * median_period or period > 1.30 * median_period:
            continue
        start = max(0, int(round((beats[index] - pre_seconds) * sr)))
        end = min(len(x), int(round((beats[index + 1] - pre_seconds) * sr)))
        if end - start < max(32, phase_bins // 2):
            continue
        source_phase = np.linspace(0.0, 1.0, end - start, dtype=np.float64)
        warped = np.interp(phase_grid, source_phase, energy[start:end])
        scale = float(np.percentile(warped, 90))
        if scale <= 1e-9:
            continue
        normalized_envelopes.append((warped / scale).astype(np.float64))
        cycle_rows.append({"index": index, "start": start, "end": end})

    if len(normalized_envelopes) < params.cycle_min_count:
        base_info["reason"] = "insufficient_regular_cycles"
        base_info["cycles_used"] = len(normalized_envelopes)
        return x.astype(np.float32, copy=True), base_info

    stack = np.stack(normalized_envelopes)
    template = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - template), axis=0)
    robust_sigma = 1.4826 * mad + max(0.015, 0.03 * float(np.max(template)))

    template_peak = float(np.max(template))
    core = template >= max(1e-9, params.cycle_core_protection_fraction * template_peak)
    protection_bins = max(
        1,
        int(round(params.cycle_core_protection_ms / 1000.0 / max(median_period, 1e-6) * phase_bins)),
    )
    if np.any(core) and protection_bins > 0:
        core = np.convolve(core.astype(np.int16), np.ones(2 * protection_bins + 1), mode="same") > 0
    core_phase_ranges: list[list[float]] = []
    range_start: int | None = None
    for index, is_core in enumerate(np.append(core, False)):
        if is_core and range_start is None:
            range_start = index
        elif not is_core and range_start is not None:
            denominator = max(1, phase_bins - 1)
            core_phase_ranges.append(
                [float(range_start / denominator), float(min(index, phase_bins - 1) / denominator)]
            )
            range_start = None

    correlations: list[float] = []
    template_centered = template - float(np.mean(template))
    template_norm = float(np.linalg.norm(template_centered))
    gain = np.ones(len(x), dtype=np.float64)
    covered = np.zeros(len(x), dtype=bool)
    outlier_samples = np.zeros(len(x), dtype=bool)
    floor_gain = float(10.0 ** (params.cycle_outlier_attenuation_db / 20.0))

    for row, normalized in zip(cycle_rows, normalized_envelopes):
        centered = normalized - float(np.mean(normalized))
        denom = float(np.linalg.norm(centered)) * template_norm
        correlations.append(float(np.dot(centered, template_centered) / denom) if denom > 1e-9 else 0.0)

        positive_z = np.maximum((normalized - template) / robust_sigma, 0.0)
        severity = np.clip(
            (positive_z - params.cycle_outlier_z) / max(params.cycle_outlier_z, 1e-6),
            0.0,
            1.0,
        )
        # Recurrent S1/S2 bins are protected; unusually large energy can still
        # receive a very small attenuation instead of being deleted outright.
        severity[core] *= 0.15
        phase_gain = 1.0 - severity * (1.0 - floor_gain)

        start = int(row["start"])
        end = int(row["end"])
        source_phase = np.linspace(0.0, 1.0, end - start, dtype=np.float64)
        sample_gain = np.interp(source_phase, phase_grid, phase_gain)
        gain[start:end] = np.minimum(gain[start:end], sample_gain)
        covered[start:end] = True
        outlier_samples[start:end] |= sample_gain < 0.98

    smoothing = max(1, int(round(0.012 * sr)))
    if smoothing > 1:
        smooth_kernel = np.ones(smoothing, dtype=np.float64) / smoothing
        smoothed = signal.fftconvolve(gain, smooth_kernel, mode="same")
        edge = min(smoothing, len(smoothed) // 2)
        if edge:
            smoothed[:edge] = gain[:edge]
            smoothed[-edge:] = gain[-edge:]
        gain = np.clip(smoothed, floor_gain, 1.0)

    covered_count = int(np.sum(covered))
    outlier_count = int(np.sum(outlier_samples & covered))
    outlier_gains = gain[outlier_samples & covered]
    base_info.update(
        {
            "applied": True,
            "reason": "ok",
            "cycles_used": len(normalized_envelopes),
            "median_period_seconds": median_period,
            "median_cycle_correlation": float(np.median(correlations)) if correlations else None,
            "outlier_fraction": float(outlier_count / covered_count) if covered_count else 0.0,
            "mean_gain_db_on_outliers": float(20.0 * np.log10(np.mean(outlier_gains) + 1e-12))
            if len(outlier_gains)
            else 0.0,
            "core_protected_fraction": float(np.mean(core)),
            "core_phase_ranges": core_phase_ranges,
        }
    )
    return (x * gain.astype(np.float32)).astype(np.float32), base_info


def apply_beat_synchronous_gate(
    x: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    params: ProcessingParams,
    cycle_consistency: dict[str, Any] | None = None,
) -> np.ndarray:
    """Keep the S1/S2 region of each beat and attenuate audio between cardiac events."""
    if not len(x) or not params.enable_denoising or not len(beat_times):
        return x.astype(np.float32, copy=True)

    effective_attenuation_db = effective_between_beat_attenuation_db(params, cycle_consistency)
    floor_gain = float(10.0 ** (effective_attenuation_db / 20.0))
    mask = np.full(len(x), floor_gain, dtype=np.float32)
    pre = max(0, int(params.beat_gate_pre_ms * sr / 1000.0))
    post = max(1, int(params.beat_gate_post_ms * sr / 1000.0))
    ramp = max(1, min(int(0.025 * sr), (pre + post) // 5))

    def retain_window(start: int, end: int) -> None:
        start = max(0, start)
        end = min(len(x), end)
        if end <= start:
            return
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

    for time_seconds in beat_times:
        center = int(round(float(time_seconds) * sr))
        retain_window(center - pre, center + post)

    cycle_consistency = cycle_consistency or {}
    core_ranges = cycle_consistency.get("core_phase_ranges", [])
    beats = np.asarray(beat_times, dtype=np.float64)
    if cycle_consistency.get("applied") and core_ranges and len(beats) >= 2:
        median_period = float(cycle_consistency.get("median_period_seconds") or np.median(np.diff(beats)))
        # A very small guard absorbs phase-warping and sample-rounding error.
        # The recurrent template has already received the full protection
        # padding, so this must remain much smaller than
        # cycle_core_protection_ms.
        core_guard = max(1, int(round(params.cycle_core_gate_guard_ms * sr / 1000.0)))
        pre_seconds = pre / sr
        for index, period in enumerate(np.diff(beats)):
            if period < 0.70 * median_period or period > 1.30 * median_period:
                continue
            cycle_start = beats[index] - pre_seconds
            for phase_start, phase_end in core_ranges:
                # core_phase_ranges were already expanded by
                # cycle_core_protection_ms when the recurrent template was
                # created. Padding them again here preserved large pieces of
                # between-beat noise around S2.
                start = int(round((cycle_start + float(phase_start) * period) * sr)) - core_guard
                end = int(round((cycle_start + float(phase_end) * period) * sr)) + core_guard
                retain_window(start, end)
    return (x * mask).astype(np.float32)


def effective_between_beat_attenuation_db(
    params: ProcessingParams,
    cycle_consistency: dict[str, Any] | None,
) -> float:
    """Limit gate depth when the 15-second recording does not support reliable cardiac timing."""
    requested = float(params.between_beat_attenuation_db)
    info = cycle_consistency or {}
    if not info.get("applied"):
        return max(requested, -6.0)

    correlation = float(info.get("median_cycle_correlation") or 0.0)
    outlier_fraction = float(info.get("outlier_fraction") or 0.0)
    if correlation < 0.65 or outlier_fraction > 0.05:
        return max(requested, -6.0)
    if correlation < 0.75 or outlier_fraction > 0.035:
        return max(requested, -12.0)
    return requested


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


def measure_rhythm_preservation(
    cleaned: np.ndarray,
    sr: int,
    expected_beat_times: np.ndarray,
    expected_period_seconds: float,
    params: ProcessingParams,
) -> dict[str, Any]:
    """Re-detect beats after denoising and compare them with the input timing sequence."""
    expected = np.asarray(expected_beat_times, dtype=np.float64)
    period = max(float(expected_period_seconds), 1e-6)
    tolerance_seconds = min(params.rhythm_match_tolerance_ms / 1000.0, 0.25 * period)
    if not len(cleaned) or sr <= 0 or len(expected) < 2:
        return {
            "applied": False,
            "is_preserved": False,
            "reason": "insufficient_signal_or_reference_beats",
            "expected_beat_count": int(len(expected)),
            "processed_beat_count": 0,
            "matched_beat_count": 0,
            "matched_fraction": 0.0,
            "count_delta": int(-len(expected)),
            "absolute_count_delta_fraction": 1.0 if len(expected) else 0.0,
            "match_tolerance_ms": float(tolerance_seconds * 1000.0),
            "median_timing_error_ms": None,
            "p95_timing_error_ms": None,
            "median_ibi_error_fraction": None,
            "processed_beat_times_seconds": [],
            "matched_expected_times_seconds": [],
            "matched_processed_times_seconds": [],
            "thresholds": rhythm_preservation_thresholds(params),
        }

    processed_envelope = extract_envelope(cleaned, sr, params.envelope_lowpass_hz)
    candidates = detect_beats(processed_envelope, sr, period, params)
    processed, template_info, _, _ = confirm_beats_with_template(
        cleaned,
        candidates,
        sr,
        period,
        params,
    )
    processed = np.asarray(processed, dtype=np.float64)
    matched_expected, matched_processed = match_beat_sequences(
        expected,
        processed,
        tolerance_seconds,
    )
    matched_count = len(matched_expected)
    matched_fraction = float(matched_count / len(expected))
    count_delta = int(len(processed) - len(expected))
    count_delta_fraction = float(abs(count_delta) / len(expected))
    timing_errors = np.abs(matched_processed - matched_expected)
    median_timing_error_ms = (
        float(np.median(timing_errors) * 1000.0) if len(timing_errors) else None
    )
    p95_timing_error_ms = (
        float(np.percentile(timing_errors, 95) * 1000.0) if len(timing_errors) else None
    )
    if len(matched_expected) >= 2:
        expected_ibi = np.diff(matched_expected)
        processed_ibi = np.diff(matched_processed)
        valid = expected_ibi > 1e-6
        ibi_errors = np.abs(processed_ibi[valid] - expected_ibi[valid]) / expected_ibi[valid]
        median_ibi_error_fraction = float(np.median(ibi_errors)) if len(ibi_errors) else None
    else:
        median_ibi_error_fraction = None

    is_preserved = bool(
        matched_fraction >= params.rhythm_minimum_match_fraction
        and count_delta_fraction <= params.rhythm_maximum_count_delta_fraction
        and median_timing_error_ms is not None
        and median_timing_error_ms <= params.rhythm_maximum_median_timing_error_ms
        and median_ibi_error_fraction is not None
        and median_ibi_error_fraction <= params.rhythm_maximum_median_ibi_error_fraction
    )
    return {
        "applied": True,
        "is_preserved": is_preserved,
        "reason": "independent_postprocess_redetection",
        "expected_beat_count": int(len(expected)),
        "processed_candidate_count": int(len(candidates)),
        "processed_beat_count": int(len(processed)),
        "processed_template_method": template_info["method"],
        "matched_beat_count": int(matched_count),
        "matched_fraction": matched_fraction,
        "count_delta": count_delta,
        "absolute_count_delta_fraction": count_delta_fraction,
        "match_tolerance_ms": float(tolerance_seconds * 1000.0),
        "median_timing_error_ms": median_timing_error_ms,
        "p95_timing_error_ms": p95_timing_error_ms,
        "median_ibi_error_fraction": median_ibi_error_fraction,
        "processed_beat_times_seconds": [round(float(value), 6) for value in processed],
        "matched_expected_times_seconds": [round(float(value), 6) for value in matched_expected],
        "matched_processed_times_seconds": [round(float(value), 6) for value in matched_processed],
        "thresholds": rhythm_preservation_thresholds(params),
    }


def match_beat_sequences(
    expected: np.ndarray,
    processed: np.ndarray,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily create an ordered, one-to-one match between two beat sequences."""
    reference = np.sort(np.asarray(expected, dtype=np.float64))
    observed = np.sort(np.asarray(processed, dtype=np.float64))
    expected_matches: list[float] = []
    processed_matches: list[float] = []
    expected_index = 0
    processed_index = 0
    while expected_index < len(reference) and processed_index < len(observed):
        delta = observed[processed_index] - reference[expected_index]
        if abs(delta) <= tolerance_seconds:
            expected_matches.append(float(reference[expected_index]))
            processed_matches.append(float(observed[processed_index]))
            expected_index += 1
            processed_index += 1
        elif delta < -tolerance_seconds:
            processed_index += 1
        else:
            expected_index += 1
    return (
        np.asarray(expected_matches, dtype=np.float64),
        np.asarray(processed_matches, dtype=np.float64),
    )


def rhythm_preservation_thresholds(params: ProcessingParams) -> dict[str, float]:
    return {
        "minimum_match_fraction": float(params.rhythm_minimum_match_fraction),
        "maximum_absolute_count_delta_fraction": float(
            params.rhythm_maximum_count_delta_fraction
        ),
        "maximum_median_timing_error_ms": float(
            params.rhythm_maximum_median_timing_error_ms
        ),
        "maximum_median_ibi_error_fraction": float(
            params.rhythm_maximum_median_ibi_error_fraction
        ),
    }


def measure_focal_cycle_contamination(
    signal_data: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    params: ProcessingParams,
) -> dict[str, Any]:
    """Find isolated, high-energy cycles that disagree with the robust heartbeat template."""
    beats = np.asarray(beat_times, dtype=np.float64)
    pre = max(1, int(round(params.template_pre_ms * sr / 1000.0)))
    post = max(1, int(round(params.template_post_ms * sr / 1000.0)))
    rows: list[dict[str, Any]] = []
    segments: list[np.ndarray] = []
    for beat_index, time_seconds in enumerate(beats):
        center = int(round(float(time_seconds) * sr))
        start = center - pre
        end = center + post
        if start < 0 or end > len(signal_data):
            continue
        segment = signal_data[start:end].astype(np.float64)
        segment -= float(np.mean(segment))
        rms = float(np.sqrt(np.mean(np.square(segment))))
        peak = float(np.max(np.abs(segment)))
        norm = float(np.linalg.norm(segment))
        if norm <= 1e-12:
            continue
        rows.append(
            {
                "beat_index": int(beat_index),
                "time_seconds": float(time_seconds),
                "rms": rms,
                "peak": peak,
            }
        )
        segments.append((segment / norm).astype(np.float64))

    thresholds = {
        "minimum_cycle_count": int(params.focal_cycle_min_count),
        "minimum_rms_ratio": float(params.focal_cycle_rms_ratio_threshold),
        "minimum_peak_ratio": float(params.focal_cycle_peak_ratio_threshold),
        "maximum_template_correlation": float(
            params.focal_cycle_correlation_threshold
        ),
    }
    if len(segments) < params.focal_cycle_min_count:
        return {
            "applied": False,
            "reason": "insufficient_complete_cycles",
            "cycle_count": int(len(segments)),
            "severe_cycle_count": 0,
            "severe_cycle_fraction": 0.0,
            "severe_cycles": [],
            "max_rms_ratio": None,
            "max_peak_ratio": None,
            "minimum_template_correlation": None,
            "thresholds": thresholds,
            "cycles": rows,
        }

    stack = np.stack(segments)
    template = np.median(stack, axis=0)
    template -= float(np.mean(template))
    template_norm = float(np.linalg.norm(template))
    if template_norm <= 1e-12:
        return {
            "applied": False,
            "reason": "degenerate_template",
            "cycle_count": int(len(segments)),
            "severe_cycle_count": 0,
            "severe_cycle_fraction": 0.0,
            "severe_cycles": [],
            "max_rms_ratio": None,
            "max_peak_ratio": None,
            "minimum_template_correlation": None,
            "thresholds": thresholds,
            "cycles": rows,
        }
    template /= template_norm
    median_rms = max(float(np.median([row["rms"] for row in rows])), 1e-12)
    median_peak = max(float(np.median([row["peak"] for row in rows])), 1e-12)
    severe_cycles = []
    for row, segment in zip(rows, segments):
        row["rms_ratio_to_median"] = float(row["rms"] / median_rms)
        row["peak_ratio_to_median"] = float(row["peak"] / median_peak)
        row["template_correlation"] = float(np.dot(segment, template))
        row["is_severe_focal_contamination"] = bool(
            row["rms_ratio_to_median"] >= params.focal_cycle_rms_ratio_threshold
            and row["peak_ratio_to_median"] >= params.focal_cycle_peak_ratio_threshold
            and row["template_correlation"] <= params.focal_cycle_correlation_threshold
        )
        if row["is_severe_focal_contamination"]:
            severe_cycles.append(
                {
                    "beat_index": row["beat_index"],
                    "time_seconds": row["time_seconds"],
                    "rms_ratio_to_median": row["rms_ratio_to_median"],
                    "peak_ratio_to_median": row["peak_ratio_to_median"],
                    "template_correlation": row["template_correlation"],
                }
            )
    return {
        "applied": True,
        "reason": "robust_template_energy_outlier_check",
        "cycle_count": int(len(rows)),
        "severe_cycle_count": int(len(severe_cycles)),
        "severe_cycle_fraction": float(len(severe_cycles) / len(rows)),
        "severe_cycles": severe_cycles,
        "max_rms_ratio": float(max(row["rms_ratio_to_median"] for row in rows)),
        "max_peak_ratio": float(max(row["peak_ratio_to_median"] for row in rows)),
        "minimum_template_correlation": float(
            min(row["template_correlation"] for row in rows)
        ),
        "thresholds": thresholds,
        "cycles": rows,
    }


def build_heartbeat_cycle_pool(
    cleaned: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    params: ProcessingParams,
    focal_cycle_contamination: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select a varied pool of real, high-quality cycles for later music rendering.

    The pool contains only cuts from the processed recording. It does not average,
    reconstruct, or synthesize a replacement heartbeat waveform.
    """
    values = np.asarray(cleaned, dtype=np.float32)
    beats = np.asarray(beat_times, dtype=np.float64)
    if not len(values) or sr <= 0 or len(beats) < 2:
        return []

    periods = np.diff(beats)
    valid_periods = periods[np.isfinite(periods) & (periods > 0.2)]
    if not len(valid_periods):
        return []
    median_period = float(np.median(valid_periods))
    pre_seconds = max(0.0, float(params.beat_gate_pre_ms) / 1000.0)
    severe_indices = {
        int(item["beat_index"])
        for item in (focal_cycle_contamination or {}).get("severe_cycles", [])
        if "beat_index" in item
    }
    candidates: list[dict[str, Any]] = []
    signatures: list[np.ndarray] = []
    signature_size = 256

    for index, period in enumerate(periods):
        if not np.isfinite(period) or period < 0.70 * median_period or period > 1.30 * median_period:
            continue
        start_seconds = max(0.0, float(beats[index]) - pre_seconds)
        end_seconds = min(len(values) / sr, float(beats[index + 1]) - pre_seconds)
        start = int(round(start_seconds * sr))
        end = int(round(end_seconds * sr))
        if end - start < max(32, int(0.25 * sr)):
            continue
        cycle = values[start:end].astype(np.float32, copy=True)
        cycle -= float(np.mean(cycle))
        peak = float(np.max(np.abs(cycle)))
        if peak <= 1e-8:
            continue
        signature = signal.resample((cycle / peak).astype(np.float64), signature_size)
        signature -= float(np.mean(signature))
        signature_norm = float(np.linalg.norm(signature))
        if signature_norm <= 1e-12:
            continue
        signatures.append((signature / signature_norm).astype(np.float64))
        candidates.append(
            {
                "audio": cycle,
                "source_cycle_index": int(index),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "period_seconds": float(period),
                "is_focal_outlier": bool(index in severe_indices),
            }
        )

    if not candidates:
        return []

    template = np.median(np.stack(signatures), axis=0)
    template -= float(np.mean(template))
    template_norm = max(float(np.linalg.norm(template)), 1e-12)
    template /= template_norm
    for candidate, signature in zip(candidates, signatures):
        cycle = np.asarray(candidate["audio"], dtype=np.float64)
        length = len(cycle)
        heart_end = max(1, min(length, int(round(length * float(params.phase_quiet_start)))))
        quiet_start = max(0, min(length - 1, int(round(length * float(params.phase_quiet_start)))))
        quiet_end = max(quiet_start + 1, min(length, int(round(length * float(params.phase_quiet_end)))))
        heart_rms = float(np.sqrt(np.mean(np.square(cycle[:heart_end]))))
        quiet_rms = float(np.sqrt(np.mean(np.square(cycle[quiet_start:quiet_end]))))
        contrast_db = float(20.0 * np.log10((heart_rms + 1e-9) / (quiet_rms + 1e-9)))
        correlation = float(np.dot(signature, template))
        regularity = float(abs(candidate["period_seconds"] - median_period) / median_period)
        score = (
            0.55 * np.clip((correlation + 0.10) / 1.10, 0.0, 1.0)
            + 0.30 * np.clip((contrast_db + 3.0) / 24.0, 0.0, 1.0)
            + 0.15 * np.exp(-8.0 * regularity)
        )
        if candidate["is_focal_outlier"]:
            score *= 0.10
        candidate.update(
            {
                "template_correlation": correlation,
                "heart_to_quiet_db": contrast_db,
                "quality_score": float(100.0 * score),
            }
        )

    eligible = [
        item
        for item in candidates
        if not item["is_focal_outlier"]
        and item["template_correlation"] >= float(params.cycle_pool_min_correlation)
    ]
    if not eligible:
        eligible = [item for item in candidates if not item["is_focal_outlier"]] or candidates
    eligible.sort(key=lambda item: item["quality_score"], reverse=True)
    limit = max(1, int(params.cycle_pool_size))
    selected = eligible[:limit]
    # Keep playback variation deterministic while avoiding a repeated quality gradient.
    selected.sort(key=lambda item: item["source_cycle_index"])
    return selected


def detect_secondary_heart_sounds(
    audio: np.ndarray,
    sr: int,
    s1_times: np.ndarray,
) -> np.ndarray:
    """Estimate one S2 energy peak inside each complete cardiac cycle."""
    values = np.abs(np.asarray(audio, dtype=np.float32))
    beats = np.asarray(s1_times, dtype=np.float64)
    if not len(values) or len(beats) < 2 or sr <= 0:
        return np.asarray([], dtype=np.float64)
    smooth = max(1, int(round(0.018 * sr)))
    envelope = np.convolve(values, np.ones(smooth) / smooth, mode="same")
    s2_times: list[float] = []
    for left, right in zip(beats[:-1], beats[1:]):
        period = float(right - left)
        if period <= 0.2:
            continue
        start = int(np.clip(round((left + 0.16 * period) * sr), 0, len(envelope)))
        end = int(np.clip(round((left + 0.58 * period) * sr), start, len(envelope)))
        if end <= start:
            continue
        peak = start + int(np.argmax(envelope[start:end]))
        s2_times.append(float(peak / sr))
    return np.asarray(s2_times, dtype=np.float64)


def make_heartbeat_detection_mix(
    audio: np.ndarray,
    sr: int,
    s1_times: np.ndarray,
    s2_times: np.ndarray,
) -> np.ndarray:
    """Overlay distinct S1/S2 clicks for diagnosis; never use this as a mix source."""
    output = np.asarray(audio, dtype=np.float32).copy() * 0.72
    if not len(output) or sr <= 0:
        return output

    def add_click(times: np.ndarray, frequency: float, amplitude: float) -> None:
        length = max(1, int(round(0.035 * sr)))
        local = np.arange(length, dtype=np.float64) / sr
        click = (
            amplitude
            * np.sin(2.0 * np.pi * frequency * local)
            * np.exp(-local * 75.0)
        ).astype(np.float32)
        for time_seconds in np.asarray(times, dtype=np.float64):
            start = int(round(float(time_seconds) * sr))
            end = min(len(output), start + length)
            if 0 <= start < len(output) and end > start:
                output[start:end] += click[: end - start]

    add_click(s1_times, 1400.0, 0.20)
    add_click(s2_times, 620.0, 0.16)
    peak = float(np.max(np.abs(output))) if len(output) else 0.0
    return output / max(1.0, peak / peak_from_dbfs(-1.0))


def concatenate_cycle_pool(
    cycle_pool: list[dict[str, Any]],
    sr: int,
    max_cycles: int = 8,
) -> np.ndarray:
    """Create a short listen-only preview of the selected real cycles."""
    if not cycle_pool or sr <= 0:
        return np.zeros(1, dtype=np.float32)
    gap = np.zeros(max(1, int(round(0.08 * sr))), dtype=np.float32)
    parts: list[np.ndarray] = []
    for item in cycle_pool[: max(1, int(max_cycles))]:
        audio = np.asarray(item.get("audio", []), dtype=np.float32)
        if not len(audio):
            continue
        parts.extend([audio, gap])
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return normalize_for_wav(np.concatenate(parts), target_peak=peak_from_dbfs(-3.0))


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


def rank_cleanest_heartbeat_segments(
    beat_times: np.ndarray,
    envelope: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
    duration: float,
    params: ProcessingParams,
    cycle_consistency: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rank real, consecutive heartbeat cycles without reconstructing their waveform."""
    beats = np.asarray(beat_times, dtype=np.float64)
    cycle_count = max(1, int(params.cleanest_segment_beats))
    candidates: list[dict[str, Any]] = []
    heart_mask = cycle_aware_heartbeat_mask(len(cleaned), sr, beats, params, cycle_consistency)

    if len(beats) >= cycle_count + 1:
        for index in range(len(beats) - cycle_count):
            local_beats = beats[index : index + cycle_count + 1]
            local_periods = np.diff(local_beats)
            if np.any(~np.isfinite(local_periods)) or np.any(local_periods <= 0):
                continue
            start_seconds = float(local_beats[0])
            end_seconds = float(local_beats[-1])
            start = max(0, int(round(start_seconds * sr)))
            end = min(len(cleaned), int(round(end_seconds * sr)))
            if end <= start:
                continue

            local_mask = heart_mask[start:end]
            local_audio = cleaned[start:end]
            heart_rms = masked_rms(local_audio, local_mask)
            gap_rms = masked_rms(local_audio, ~local_mask)
            contrast_db = float(20.0 * math.log10((heart_rms + 1e-12) / (gap_rms + 1e-12)))
            gap_rms_dbfs = dbfs(gap_rms)
            regularity = float(np.std(local_periods) / (np.mean(local_periods) + 1e-9))
            envelope_snr_db = estimate_envelope_snr(
                envelope,
                sr,
                start_seconds,
                end_seconds,
                local_beats,
            )
            candidates.append(
                {
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": end_seconds - start_seconds,
                    "cycle_count": cycle_count,
                    "local_bpm": float(60.0 / np.median(local_periods)),
                    "ibi_std_seconds": float(np.std(local_periods)),
                    "regularity_score": regularity,
                    "envelope_snr_db": envelope_snr_db,
                    "heart_to_gap_contrast_db": contrast_db,
                    "gap_rms_dbfs": gap_rms_dbfs,
                    "method": "real_cycles_regularity_contrast_and_noise",
                }
            )

    if candidates:
        gap_levels = np.asarray([item["gap_rms_dbfs"] for item in candidates], dtype=np.float64)
        gap_best = float(np.min(gap_levels))
        gap_worst = float(np.max(gap_levels))
        for item in candidates:
            regularity_component = float(np.exp(-7.0 * item["regularity_score"]))
            contrast_component = float(np.clip(item["heart_to_gap_contrast_db"] / 30.0, 0.0, 1.0))
            envelope_component = float(np.clip(item["envelope_snr_db"] / 40.0, 0.0, 1.0))
            if gap_worst - gap_best > 1e-6:
                quiet_component = float((gap_worst - item["gap_rms_dbfs"]) / (gap_worst - gap_best))
            else:
                quiet_component = 1.0
            item["quality_score"] = float(
                100.0
                * (
                    0.35 * regularity_component
                    + 0.35 * contrast_component
                    + 0.20 * envelope_component
                    + 0.10 * quiet_component
                )
            )
        candidates.sort(key=lambda item: item["quality_score"], reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank
        selected = dict(candidates[0])
        selected.pop("rank", None)
        selected["is_fallback"] = False
        return selected, candidates[: max(1, int(params.cleanest_candidate_count))]

    fallback_end = min(duration, max(0.0, cycle_count * 60.0 / max(params.min_bpm, 1.0)))
    fallback = {
        "start_seconds": 0.0,
        "end_seconds": fallback_end,
        "duration_seconds": fallback_end,
        "cycle_count": cycle_count,
        "local_bpm": None,
        "ibi_std_seconds": None,
        "regularity_score": None,
        "envelope_snr_db": None,
        "heart_to_gap_contrast_db": None,
        "gap_rms_dbfs": None,
        "quality_score": 0.0,
        "method": "fallback_insufficient_cycles",
        "is_fallback": True,
    }
    return fallback, [{**fallback, "rank": 1}]


def estimate_envelope_snr(
    envelope: np.ndarray,
    sr: int,
    start_seconds: float,
    end_seconds: float,
    beat_times: np.ndarray,
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


def cut_cleanest_heartbeat_segment(
    x: np.ndarray,
    sr: int,
    segment_info: dict[str, Any],
    params: ProcessingParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not len(x):
        return x.copy(), {
            "adjusted_start_seconds": 0.0,
            "adjusted_end_seconds": 0.0,
            "adjusted_duration_seconds": 0.0,
            "boundary_jump_before_fade": 0.0,
        }
    start = int(max(0, min(len(x) - 1, round(float(segment_info["start_seconds"]) * sr))))
    end = int(max(start + 1, min(len(x), round(float(segment_info["end_seconds"]) * sr))))
    search = max(1, int(round(params.segment_zero_crossing_search_ms * sr / 1000.0)))
    start_adjusted = nearest_zero_crossing(x, start, search)
    end_adjusted = nearest_zero_crossing(x, end, search)
    if end_adjusted <= start_adjusted:
        start_adjusted, end_adjusted = start, end
    segment = x[start_adjusted:end_adjusted].astype(np.float32, copy=True)
    boundary_jump = float(abs(segment[-1] - segment[0])) if len(segment) else 0.0
    segment = apply_edge_fades(segment, sr, params.segment_edge_fade_ms)
    return segment, {
        "adjusted_start_seconds": float(start_adjusted / sr),
        "adjusted_end_seconds": float(end_adjusted / sr),
        "adjusted_duration_seconds": float(len(segment) / sr),
        "edge_fade_ms": float(params.segment_edge_fade_ms),
        "boundary_jump_before_fade": boundary_jump,
    }


def nearest_zero_crossing(x: np.ndarray, center: int, radius: int) -> int:
    lower = max(0, center - radius)
    upper = min(len(x) - 1, center + radius)
    if upper <= lower:
        return center
    local = x[lower : upper + 1]
    sign_changes = np.where(np.diff(np.signbit(local)))[0]
    if len(sign_changes):
        candidates = lower + sign_changes
        return int(candidates[np.argmin(np.abs(candidates - center))])
    return int(lower + np.argmin(np.abs(local)))


def apply_edge_fades(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    if not len(x) or fade_ms <= 0:
        return x
    count = min(len(x) // 2, max(1, int(round(fade_ms * sr / 1000.0))))
    if count <= 1:
        return x
    output = x.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
    output[:count] *= ramp
    output[-count:] *= ramp[::-1]
    return output


def optimize_loop_playback_loudness(
    x: np.ndarray,
    params: ProcessingParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a clearly labelled playback copy; the faithful loop remains unchanged."""
    if not len(x) or float(np.max(np.abs(x))) <= 1e-12:
        return x.astype(np.float32, copy=True), {
            "applied": False,
            "reason": "empty_or_silent",
            "is_playback_optimized": True,
        }

    target_peak = peak_from_dbfs(params.playback_loop_peak_dbfs)
    target_rms = peak_from_dbfs(params.playback_loop_target_rms_dbfs)
    normalized = x.astype(np.float64) / float(np.max(np.abs(x)))

    def compressed(drive: float) -> np.ndarray:
        if drive <= 1e-4:
            curved = normalized
        else:
            curved = np.tanh(drive * normalized) / math.tanh(drive)
        return curved * target_peak

    lower = 0.0
    upper = max(0.0, float(params.playback_loop_max_softclip_drive))
    output = compressed(lower)
    drive_used = 0.0
    if float(np.sqrt(np.mean(np.square(output)))) < target_rms and upper > 0:
        for _ in range(32):
            midpoint = 0.5 * (lower + upper)
            candidate = compressed(midpoint)
            candidate_rms = float(np.sqrt(np.mean(np.square(candidate))))
            if candidate_rms < target_rms:
                lower = midpoint
            else:
                upper = midpoint
                output = candidate
        output = compressed(upper)
        drive_used = upper

    output = np.clip(output, -target_peak, target_peak).astype(np.float32)
    output_rms = float(np.sqrt(np.mean(np.square(output))))
    return output, {
        "applied": True,
        "method": "monotonic_tanh_soft_compression",
        "is_playback_optimized": True,
        "template_replacement": False,
        "minimum_target_rms_dbfs": float(params.playback_loop_target_rms_dbfs),
        "achieved_rms_dbfs": dbfs(output_rms),
        "target_peak_dbfs": float(params.playback_loop_peak_dbfs),
        "achieved_peak_dbfs": dbfs(float(np.max(np.abs(output)))),
        "softclip_drive": float(drive_used),
        "max_softclip_drive": float(params.playback_loop_max_softclip_drive),
    }


def compute_quality(
    raw_mono: np.ndarray,
    spectral_filtered: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
    source_info: dict[str, Any],
    params: ProcessingParams,
    beat_times: np.ndarray,
    cycle_consistency: dict[str, Any],
    rhythm_preservation: dict[str, Any],
    focal_cycle_contamination: dict[str, Any],
) -> dict[str, Any]:
    rms = float(np.sqrt(np.mean(cleaned * cleaned))) if len(cleaned) else 0.0
    peak = float(np.max(np.abs(cleaned))) if len(cleaned) else 0.0
    clipping_fraction = float(np.mean(np.abs(raw_mono) >= 0.999)) if len(raw_mono) else 0.0
    core_mask = cycle_aware_heartbeat_mask(
        len(cleaned), sr, beat_times, params, cycle_consistency
    )
    coverage = float(np.mean(core_mask)) if len(core_mask) else 0.0
    interbeat_mask = ~core_mask if len(core_mask) else core_mask
    interbeat_reduction_db = rms_ratio_db(cleaned, spectral_filtered, interbeat_mask)
    heartbeat_energy_change_db = rms_ratio_db(cleaned, spectral_filtered, core_mask)
    heartbeat_preservation_correlation = masked_correlation(spectral_filtered, cleaned, core_mask)
    return {
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "dc_offset": float(np.mean(raw_mono)) if len(raw_mono) else 0.0,
        "clipping_fraction": clipping_fraction,
        "is_clipping_suspected": bool(clipping_fraction > 0.0005),
        "duration_seconds": float(len(raw_mono) / sr) if sr else 0.0,
        "channels": source_info["channels"],
        "denoising_enabled": bool(params.enable_denoising),
        "beat_window_coverage_fraction": coverage,
        "requested_between_beat_attenuation_db": float(params.between_beat_attenuation_db),
        "between_beat_attenuation_db": effective_between_beat_attenuation_db(params, cycle_consistency),
        "interbeat_noise_reduction_db": interbeat_reduction_db,
        "heartbeat_energy_change_db": heartbeat_energy_change_db,
        "heartbeat_preservation_correlation": heartbeat_preservation_correlation,
        "cycle_consistency_applied": bool(cycle_consistency.get("applied", False)),
        "cycle_outlier_fraction": float(cycle_consistency.get("outlier_fraction", 0.0)),
        "cycle_outlier_mean_gain_db": float(cycle_consistency.get("mean_gain_db_on_outliers", 0.0)),
        "rhythm_preservation": rhythm_preservation,
        "focal_cycle_contamination": focal_cycle_contamination,
        "reconstruction_policy": "attenuation_only_no_template_replacement",
    }


def beat_window_mask(
    sample_count: int, sr: int, beat_times: np.ndarray, params: ProcessingParams
) -> np.ndarray:
    mask = np.zeros(max(0, sample_count), dtype=bool)
    if not sample_count or not sr:
        return mask
    pre = max(0, int(params.beat_gate_pre_ms * sr / 1000.0))
    post = max(1, int(params.beat_gate_post_ms * sr / 1000.0))
    for time_seconds in beat_times:
        center = int(round(float(time_seconds) * sr))
        mask[max(0, center - pre) : min(sample_count, center + post)] = True
    return mask


def cycle_aware_heartbeat_mask(
    sample_count: int,
    sr: int,
    beat_times: np.ndarray,
    params: ProcessingParams,
    cycle_consistency: dict[str, Any] | None,
) -> np.ndarray:
    mask = beat_window_mask(sample_count, sr, beat_times, params)
    info = cycle_consistency or {}
    core_ranges = info.get("core_phase_ranges", [])
    beats = np.asarray(beat_times, dtype=np.float64)
    if not info.get("applied") or not core_ranges or len(beats) < 2 or not sample_count or not sr:
        return mask

    median_period = float(info.get("median_period_seconds") or np.median(np.diff(beats)))
    pre_seconds = max(0.0, params.beat_gate_pre_ms / 1000.0)
    guard = max(1, int(round(params.cycle_core_gate_guard_ms * sr / 1000.0)))
    for index, period in enumerate(np.diff(beats)):
        if period < 0.70 * median_period or period > 1.30 * median_period:
            continue
        cycle_start = beats[index] - pre_seconds
        for phase_start, phase_end in core_ranges:
            start = int(round((cycle_start + float(phase_start) * period) * sr)) - guard
            end = int(round((cycle_start + float(phase_end) * period) * sr)) + guard
            mask[max(0, start) : min(sample_count, end)] = True
    return mask


def masked_rms(x: np.ndarray, mask: np.ndarray) -> float:
    count = min(len(x), len(mask))
    if not count:
        return 0.0
    selected = mask[:count]
    if not np.any(selected):
        return 0.0
    return float(np.sqrt(np.mean(np.square(x[:count][selected], dtype=np.float64))))


def rms_ratio_db(numerator: np.ndarray, denominator: np.ndarray, mask: np.ndarray) -> float | None:
    if not len(numerator) or not len(denominator) or not len(mask) or not np.any(mask):
        return None
    count = min(len(numerator), len(denominator), len(mask))
    selected = mask[:count]
    if not np.any(selected):
        return None
    numerator_rms = float(np.sqrt(np.mean(np.square(numerator[:count][selected]))))
    denominator_rms = float(np.sqrt(np.mean(np.square(denominator[:count][selected]))))
    return float(20.0 * np.log10((numerator_rms + 1e-12) / (denominator_rms + 1e-12)))


def masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    if not len(a) or not len(b) or not len(mask) or not np.any(mask):
        return None
    count = min(len(a), len(b), len(mask))
    selected = mask[:count]
    av = a[:count][selected].astype(np.float64)
    bv = b[:count][selected].astype(np.float64)
    if len(av) < 2:
        return None
    av -= float(np.mean(av))
    bv -= float(np.mean(bv))
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom > 1e-12 else None


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
    cycle_consistency: dict[str, Any],
    params: ProcessingParams,
) -> dict[str, Any]:
    """Produce a conservative recording-usability score, not a medical diagnosis."""
    reasons: list[str] = []
    rerecord_reasons: list[str] = []
    score = 100.0
    duration = float(len(raw_mono) / sr) if sr else 0.0
    if duration < 8.0:
        score -= 25.0
        reasons.append("Recording is shorter than 8 seconds; BPM consensus is less reliable.")
        rerecord_reasons.append("The recording is too short for reliable cycle-consistency denoising.")
    if base_quality["is_clipping_suspected"]:
        score -= 25.0
        reasons.append("Input clipping was detected.")
        rerecord_reasons.append("Input clipping cannot be repaired without inventing waveform detail.")

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
        rerecord_reasons.append("Too few reliable heartbeats were found to preserve S1/S2 safely.")

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
        template_median = template_info.get("median_correlation")
        if (
            template_median is not None
            and float(template_median) < params.quality_low_template_correlation
        ):
            score -= 15.0
            reasons.append(
                "The median heartbeat-template correlation is low; recurrent noise or uncertain cycle alignment may remain."
            )

    ibi = np.diff(np.asarray(beat_times, dtype=np.float64))
    ibi_cv = (
        float(np.std(ibi) / np.mean(ibi))
        if len(ibi) >= 2 and float(np.mean(ibi)) > 1e-6
        else None
    )
    if ibi_cv is not None and ibi_cv > params.quality_high_ibi_cv:
        score -= 12.0
        reasons.append(
            "Detected heartbeat intervals vary strongly; automatic cycle alignment is less reliable."
        )

    cycle_correlation = cycle_consistency.get("median_cycle_correlation")
    cycle_outlier_fraction = float(cycle_consistency.get("outlier_fraction", 0.0))
    if cycle_consistency.get("enabled") and not cycle_consistency.get("applied"):
        score -= 10.0
        reasons.append("Too few regular cycles were available for cycle-consistency denoising.")
    if cycle_correlation is not None and float(cycle_correlation) < 0.25:
        score -= 18.0
        reasons.append("Heartbeat cycles are weakly consistent after alignment.")
        if cycle_outlier_fraction > 0.15:
            rerecord_reasons.append("Cycle inconsistency and transient contamination are both high.")
    if cycle_outlier_fraction > 0.20:
        score -= 15.0
        reasons.append("A large fraction of the recording contains non-repeating transient energy.")

    preservation = base_quality.get("heartbeat_preservation_correlation")
    if preservation is not None and float(preservation) < 0.85:
        score -= 25.0
        reasons.append("Denoising changed the protected heartbeat waveform too strongly.")
        rerecord_reasons.append("The protected heartbeat waveform could not be preserved reliably.")

    rhythm = base_quality.get("rhythm_preservation", {})
    if rhythm.get("applied") and not rhythm.get("is_preserved"):
        score -= 25.0
        reasons.append("Independent post-denoising detection did not preserve enough heartbeat timing events.")
        rerecord_reasons.append(
            "Heartbeat count or timing could not be verified after denoising; exporting it as reliable would be unsafe."
        )

    focal = base_quality.get("focal_cycle_contamination", {})
    if focal.get("applied") and int(focal.get("severe_cycle_count", 0)) > 0:
        score -= 25.0
        severe_times = ", ".join(
            f"{float(item['time_seconds']):.2f}s"
            for item in focal.get("severe_cycles", [])
        )
        reasons.append(
            "One or more isolated high-energy heartbeat windows disagree with the robust cycle template."
        )
        rerecord_reasons.append(
            "Severe focal contamination overlaps a heartbeat window"
            + (f" near {severe_times}" if severe_times else "")
            + "; it cannot be removed safely without risking a false or damaged heart sound."
        )

    score = float(np.clip(score, 0.0, 100.0))
    needs_rerecording = bool(rerecord_reasons)
    if needs_rerecording:
        grade = "needs_rerecording"
        denoising_status = "rerecord"
    elif score >= 75.0:
        grade = "good"
        denoising_status = "ok"
    elif score >= 50.0:
        grade = "usable_with_caution"
        denoising_status = "limited"
    else:
        grade = "poor"
        denoising_status = "limited"
    if not reasons:
        reasons.append("No major automated recording-quality issue was detected.")
    return {
        "score": score,
        "grade": grade,
        "denoising_status": denoising_status,
        "needs_rerecording": needs_rerecording,
        "is_safe_to_export": not needs_rerecording,
        "rerecord_reasons": rerecord_reasons,
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
            "template_median_correlation": template_info.get("median_correlation"),
            "ibi_coefficient_of_variation": ibi_cv,
            "cycle_consistency_applied": bool(cycle_consistency.get("applied", False)),
            "cycle_count": int(cycle_consistency.get("cycles_used", 0)),
            "median_cycle_correlation": cycle_correlation,
            "cycle_outlier_fraction": cycle_outlier_fraction,
            "interbeat_noise_reduction_db": base_quality.get("interbeat_noise_reduction_db"),
            "heartbeat_preservation_correlation": preservation,
            "rhythm_preservation_is_preserved": rhythm.get("is_preserved"),
            "rhythm_preservation_matched_fraction": rhythm.get("matched_fraction"),
            "rhythm_preservation_count_delta": rhythm.get("count_delta"),
            "rhythm_preservation_median_timing_error_ms": rhythm.get(
                "median_timing_error_ms"
            ),
            "rhythm_preservation_median_ibi_error_fraction": rhythm.get(
                "median_ibi_error_fraction"
            ),
            "focal_contamination_severe_cycle_count": focal.get("severe_cycle_count"),
            "focal_contamination_severe_cycle_fraction": focal.get(
                "severe_cycle_fraction"
            ),
            "focal_contamination_max_rms_ratio": focal.get("max_rms_ratio"),
            "focal_contamination_max_peak_ratio": focal.get("max_peak_ratio"),
            "focal_contamination_minimum_template_correlation": focal.get(
                "minimum_template_correlation"
            ),
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
    template_info: dict[str, Any],
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
        "parameters": asdict(params),
    }


def dbfs(value: float) -> float:
    return float(20.0 * math.log10(max(value, 1e-12)))


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
    ibi: np.ndarray,
    bpm_info: dict[str, float],
    template_analysis: list[dict[str, Any]],
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(f"Heartbeat preprocessing diagnostics: {stem}")
    plot_signal(axes[0], raw, sr, "Raw mono waveform")
    plot_signal(axes[1], filtered, sr, "Cycle-consistent signal (spectral gate + transient outlier attenuation)")
    plot_signal(axes[2], cleaned, sr, "Final denoised heartbeat (S1/S2-preserving soft gate)")

    t_env = np.arange(len(envelope)) / sr if len(envelope) else np.array([])
    axes[3].plot(t_env, envelope, linewidth=0.8, color="#2c7fb8")
    for item in template_analysis:
        color = "#e34a33" if item["is_confirmed"] else "#969696"
        style = "-" if item["is_confirmed"] else "--"
        axes[3].axvline(float(item["time_seconds"]), color=color, alpha=0.5, linewidth=0.8, linestyle=style)
    axes[3].set_title("Envelope and detected heartbeat cycles")
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


def plot_signal(ax: Any, x: np.ndarray, sr: int, title: str) -> None:
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
