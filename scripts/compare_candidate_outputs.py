from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartbeat_preprocessor.core import (
    ProcessingParams,
    bandpass,
    cycle_aware_heartbeat_mask,
    masked_correlation,
    masked_rms,
    measure_rhythm_preservation,
    process_audio_file,
    remove_dc,
    rms_ratio_db,
    to_mono_float,
)


HEART_CORRELATION_TOLERANCE = 0.02
ATTACK_CORRELATION_TOLERANCE = 0.03
SPECTRAL_DISTANCE_TOLERANCE = 0.05
MINIMUM_INTERBEAT_REDUCTION_DB = 8.0


def load_wav(path: Path) -> tuple[int, np.ndarray, dict[str, Any]]:
    sr, raw = wavfile.read(path)
    mono = to_mono_float(raw)
    channels = 1 if raw.ndim == 1 else int(raw.shape[1])
    return int(sr), mono, {
        "sample_rate": int(sr),
        "channels": channels,
        "frames": int(raw.shape[0]),
        "clipping_fraction": float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 0.0,
    }


def validate_geometry(case_id: str, items: dict[str, tuple[int, np.ndarray, dict[str, Any]]]) -> None:
    signatures = {
        (metadata["sample_rate"], metadata["channels"], metadata["frames"])
        for _, _, metadata in items.values()
    }
    if len(signatures) != 1:
        raise ValueError(f"{case_id}: reference, v1 and candidate must have identical WAV geometry")


def attack_mask(sample_count: int, sr: int, beat_times: np.ndarray) -> np.ndarray:
    mask = np.zeros(sample_count, dtype=bool)
    pre = max(1, int(round(0.025 * sr)))
    post = max(1, int(round(0.110 * sr)))
    for time_seconds in beat_times:
        center = int(round(float(time_seconds) * sr))
        mask[max(0, center - pre) : min(sample_count, center + post)] = True
    return mask


def protected_spectral_profile(
    audio: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    params: ProcessingParams,
) -> dict[str, float]:
    pre = max(1, int(round(params.template_pre_ms * sr / 1000.0)))
    post = max(1, int(round(params.template_post_ms * sr / 1000.0)))
    width = pre + post
    nfft = 1 << max(8, int(math.ceil(math.log2(max(2, width)))))
    accumulated = np.zeros(nfft // 2 + 1, dtype=np.float64)
    used = 0
    window = np.hanning(width)
    for time_seconds in beat_times:
        center = int(round(float(time_seconds) * sr))
        start = center - pre
        end = center + post
        if start < 0 or end > len(audio):
            continue
        segment = audio[start:end].astype(np.float64)
        segment -= float(np.mean(segment))
        spectrum = np.fft.rfft(segment * window, n=nfft)
        accumulated += np.square(np.abs(spectrum))
        used += 1
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sr)
    bands = ((20.0, 50.0), (50.0, 100.0), (100.0, 150.0), (150.0, 250.0))
    energies = []
    for low, high in bands:
        selected = (frequencies >= low) & (frequencies < high)
        energies.append(float(np.sum(accumulated[selected])))
    total = float(sum(energies))
    if not used or total <= 1e-20:
        return {f"{int(low)}_{int(high)}_hz": 0.0 for low, high in bands}
    return {
        f"{int(low)}_{int(high)}_hz": float(energy / total)
        for (low, high), energy in zip(bands, energies)
    }


def profile_l1_distance(reference: dict[str, float], output: dict[str, float]) -> float:
    return float(sum(abs(reference[key] - output[key]) for key in reference))


def automated_prescreen(
    v1: dict[str, Any],
    candidate: dict[str, Any],
    correlation_delta: float,
    attack_delta: float,
    spectral_distance_delta: float,
) -> dict[str, Any]:
    candidate_gap = candidate["interbeat_change_after_heart_alignment_db"]
    gates = {
        "candidate_rhythm_preserved": bool(candidate["rhythm_is_preserved"]),
        "candidate_not_clipped": bool(candidate["clipping_fraction"] <= 0.0005),
        "heart_correlation_not_regressed": bool(
            correlation_delta >= -HEART_CORRELATION_TOLERANCE
        ),
        "attack_correlation_not_regressed": bool(
            attack_delta >= -ATTACK_CORRELATION_TOLERANCE
        ),
        "spectral_profile_not_regressed": bool(
            spectral_distance_delta <= SPECTRAL_DISTANCE_TOLERANCE
        ),
        "interbeat_noise_reduced_from_reference": bool(
            candidate_gap is not None and candidate_gap <= -MINIMUM_INTERBEAT_REDUCTION_DB
        ),
    }
    return {
        **gates,
        "passes": bool(all(gates.values())),
        "policy": (
            "Gain-align on protected heart regions; require rhythm and clipping safety, "
            "no material heart/attack/spectral regression versus v1, and at least 8 dB "
            "inter-beat reduction versus the input. More silence than v1 is not required."
        ),
    }


def output_metrics(
    audio: np.ndarray,
    reference_band: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    period_seconds: float,
    heart_mask: np.ndarray,
    onset_mask: np.ndarray,
    reference_profile: dict[str, float],
    params: ProcessingParams,
    clipping_fraction: float,
) -> dict[str, Any]:
    output_band = bandpass(remove_dc(audio), sr, params.bandpass_low_hz, params.bandpass_high_hz)
    reference_heart_rms = masked_rms(reference_band, heart_mask)
    output_heart_rms = masked_rms(output_band, heart_mask)
    heart_gain = reference_heart_rms / max(output_heart_rms, 1e-12)
    aligned = output_band * heart_gain
    interbeat_mask = ~heart_mask
    rhythm = measure_rhythm_preservation(output_band, sr, beat_times, period_seconds, params)
    profile = protected_spectral_profile(aligned, sr, beat_times, params)
    return {
        "heart_gain_alignment_db": float(20.0 * np.log10(max(heart_gain, 1e-12))),
        "protected_heart_correlation": masked_correlation(reference_band, aligned, heart_mask),
        "attack_correlation": masked_correlation(reference_band, aligned, onset_mask),
        "interbeat_change_after_heart_alignment_db": rms_ratio_db(
            aligned, reference_band, interbeat_mask
        ),
        "protected_spectral_profile": profile,
        "protected_spectral_profile_l1_distance": profile_l1_distance(reference_profile, profile),
        "rhythm_is_preserved": rhythm["is_preserved"],
        "rhythm_matched_fraction": rhythm["matched_fraction"],
        "rhythm_count_delta": rhythm["count_delta"],
        "rhythm_median_timing_error_ms": rhythm["median_timing_error_ms"],
        "clipping_fraction": clipping_fraction,
    }


def compare_case(case_id: str, reference_path: Path, v1_path: Path, candidate_path: Path) -> dict[str, Any]:
    paths = {"reference": reference_path, "v1": v1_path, "candidate": candidate_path}
    loaded = {name: load_wav(path) for name, path in paths.items()}
    validate_geometry(case_id, loaded)
    sr, reference, reference_metadata = loaded["reference"]
    params = ProcessingParams()
    reference_analysis = process_audio_file(reference_path, params=params)
    beat_times = np.asarray(reference_analysis["beat_times"], dtype=np.float64)
    period_seconds = float(reference_analysis["summary"]["tempo"]["period_seconds"])
    reference_band = bandpass(
        remove_dc(reference), sr, params.bandpass_low_hz, params.bandpass_high_hz
    )
    heart_mask = cycle_aware_heartbeat_mask(
        len(reference_band),
        sr,
        beat_times,
        params,
        reference_analysis["cycle_consistency"],
    )
    onset_mask = attack_mask(len(reference_band), sr, beat_times)
    reference_profile = protected_spectral_profile(reference_band, sr, beat_times, params)
    variants = {}
    for name in ("v1", "candidate"):
        _, audio, metadata = loaded[name]
        variants[name] = output_metrics(
            audio,
            reference_band,
            sr,
            beat_times,
            period_seconds,
            heart_mask,
            onset_mask,
            reference_profile,
            params,
            metadata["clipping_fraction"],
        )

    v1 = variants["v1"]
    candidate = variants["candidate"]
    correlation_delta = float(
        (candidate["protected_heart_correlation"] or 0.0)
        - (v1["protected_heart_correlation"] or 0.0)
    )
    attack_delta = float(
        (candidate["attack_correlation"] or 0.0) - (v1["attack_correlation"] or 0.0)
    )
    spectral_distance_delta = float(
        candidate["protected_spectral_profile_l1_distance"]
        - v1["protected_spectral_profile_l1_distance"]
    )
    v1_gap = v1["interbeat_change_after_heart_alignment_db"]
    candidate_gap = candidate["interbeat_change_after_heart_alignment_db"]
    gap_reduction_advantage = (
        float(v1_gap - candidate_gap) if v1_gap is not None and candidate_gap is not None else None
    )
    prescreen = automated_prescreen(
        v1,
        candidate,
        correlation_delta,
        attack_delta,
        spectral_distance_delta,
    )
    return {
        "case_id": case_id,
        "duration_seconds": float(reference_metadata["frames"] / sr),
        "sample_rate": sr,
        "reference_beat_count": int(len(beat_times)),
        "variants": variants,
        "candidate_minus_v1": {
            "protected_heart_correlation": correlation_delta,
            "attack_correlation": attack_delta,
            "protected_spectral_profile_l1_distance": spectral_distance_delta,
            "interbeat_reduction_advantage_db": gap_reduction_advantage,
        },
        "automated_prescreen": prescreen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Objective pre-screen for full-length v1 versus candidate heartbeat denoising WAVs."
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        metavar=("ID", "REFERENCE", "V1", "CANDIDATE"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [
        compare_case(case_id, Path(reference), Path(v1), Path(candidate))
        for case_id, reference, v1, candidate in args.case
    ]
    report = {
        "case_count": len(cases),
        "all_cases_pass_automated_prescreen": bool(
            all(case["automated_prescreen"]["passes"] for case in cases)
        ),
        "human_blind_listening_still_required": True,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        rows = []
        for case in cases:
            row = {
                "case_id": case["case_id"],
                "duration_seconds": case["duration_seconds"],
                "passes": case["automated_prescreen"]["passes"],
                **case["candidate_minus_v1"],
            }
            rows.append(row)
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
