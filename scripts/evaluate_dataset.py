from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartbeat_preprocessor.core import masked_correlation, process_audio_bytes, to_mono_float, wav_bytes


def load_window(path: Path, seconds: float) -> tuple[int, np.ndarray, float, float]:
    sr, raw = wavfile.read(path)
    mono = to_mono_float(raw)
    target = max(1, int(round(seconds * sr)))
    if len(mono) <= target:
        return int(sr), mono, 0.0, float(len(mono) / sr)
    start = (len(mono) - target) // 2
    end = start + target
    return int(sr), mono[start:end], float(start / sr), float(end / sr)


def annotated_s1s2_mask(path: Path, sr: int, start_seconds: float, sample_count: int) -> np.ndarray | None:
    annotation = path.with_suffix(".tsv")
    if not annotation.exists():
        return None
    mask = np.zeros(sample_count, dtype=bool)
    with annotation.open("r", encoding="utf-8") as source:
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 3:
                continue
            try:
                begin, end, state = float(row[0]), float(row[1]), int(float(row[2]))
            except ValueError:
                continue
            if state not in (1, 3):
                continue
            left = max(0, int(round((begin - start_seconds) * sr)))
            right = min(sample_count, int(round((end - start_seconds) * sr)))
            if right > left:
                mask[left:right] = True
    return mask if np.any(mask) else None


def evaluate_file(path: Path, window_seconds: float) -> dict:
    sr, audio_segment, start_seconds, end_seconds = load_window(path, window_seconds)
    result = process_audio_bytes(path.name, wav_bytes(sr, audio_segment))
    summary = result["summary"]
    quality = summary["quality"]
    recording = summary["recording_quality"]
    cycle = result["cycle_consistency"]
    rhythm = result["rhythm_preservation"]
    cleanest_segment = result["cleanest_segment"]
    playback = cleanest_segment["playback_loudness"]
    annotation_mask = annotated_s1s2_mask(path, sr, start_seconds, len(audio_segment))
    annotated_preservation = (
        masked_correlation(result["spectral_filtered"], result["cleaned"], annotation_mask)
        if annotation_mask is not None
        else None
    )
    return {
        "file": str(path),
        "window_start_seconds": start_seconds,
        "window_end_seconds": end_seconds,
        "duration_seconds": summary["duration_seconds"],
        "sample_rate": sr,
        "estimated_bpm": summary["tempo"]["estimated_bpm"],
        "detected_beats": summary["tempo"]["detected_beats"],
        "denoising_status": recording["denoising_status"],
        "needs_rerecording": recording["needs_rerecording"],
        "quality_score": recording["score"],
        "cycle_applied": cycle["applied"],
        "cycles_used": cycle["cycles_used"],
        "median_cycle_correlation": cycle["median_cycle_correlation"],
        "cycle_outlier_fraction": cycle["outlier_fraction"],
        "rhythm_is_preserved": rhythm["is_preserved"],
        "rhythm_matched_fraction": rhythm["matched_fraction"],
        "rhythm_count_delta": rhythm["count_delta"],
        "rhythm_absolute_count_delta_fraction": rhythm["absolute_count_delta_fraction"],
        "rhythm_median_timing_error_ms": rhythm["median_timing_error_ms"],
        "rhythm_median_ibi_error_fraction": rhythm["median_ibi_error_fraction"],
        "cleanest_segment_is_fallback": cleanest_segment["is_fallback"],
        "cleanest_segment_quality_score": cleanest_segment["quality_score"],
        "cleanest_segment_contrast_db": cleanest_segment["heart_to_gap_contrast_db"],
        "cleanest_segment_gap_rms_dbfs": cleanest_segment["gap_rms_dbfs"],
        "playback_loop_rms_dbfs": playback.get("achieved_rms_dbfs"),
        "playback_loop_peak_dbfs": playback.get("achieved_peak_dbfs"),
        "interbeat_noise_reduction_db": quality["interbeat_noise_reduction_db"],
        "heartbeat_preservation_correlation": quality["heartbeat_preservation_correlation"],
        "annotated_s1s2_preservation_correlation": annotated_preservation,
        "rerecord_reasons": " | ".join(recording["rerecord_reasons"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the denoiser on fixed-length public PCG windows.")
    parser.add_argument("dataset_root", type=Path, help="Directory containing WAV files")
    parser.add_argument("--window-seconds", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=0, help="Maximum WAV files; 0 evaluates all")
    parser.add_argument("--output", type=Path, default=Path("data/generated/dataset_evaluation.json"))
    args = parser.parse_args()

    files = sorted(args.dataset_root.rglob("*.wav"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        parser.error(f"no WAV files found under {args.dataset_root}")

    rows: list[dict] = []
    errors: list[dict] = []
    for index, path in enumerate(files, start=1):
        try:
            rows.append(evaluate_file(path, args.window_seconds))
            print(f"[{index}/{len(files)}] {path.name}: {rows[-1]['denoising_status']}")
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})
            print(f"[{index}/{len(files)}] {path.name}: ERROR {exc}")

    status_counts = Counter(row["denoising_status"] for row in rows)
    numeric_fields = (
        "quality_score",
        "median_cycle_correlation",
        "cycle_outlier_fraction",
        "rhythm_matched_fraction",
        "rhythm_count_delta",
        "rhythm_absolute_count_delta_fraction",
        "rhythm_median_timing_error_ms",
        "rhythm_median_ibi_error_fraction",
        "cleanest_segment_quality_score",
        "cleanest_segment_contrast_db",
        "cleanest_segment_gap_rms_dbfs",
        "playback_loop_rms_dbfs",
        "playback_loop_peak_dbfs",
        "interbeat_noise_reduction_db",
        "heartbeat_preservation_correlation",
        "annotated_s1s2_preservation_correlation",
    )
    distributions = {}
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if row[field] is not None and np.isfinite(row[field])]
        distributions[field] = (
            {
                "count": len(values),
                "min": float(np.min(values)),
                "p05": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
            if values
            else None
        )

    annotated_low_tail = sorted(
        (
            {
                "file": row["file"],
                "denoising_status": row["denoising_status"],
                "annotated_s1s2_preservation_correlation": row[
                    "annotated_s1s2_preservation_correlation"
                ],
            }
            for row in rows
            if row["annotated_s1s2_preservation_correlation"] is not None
            and row["annotated_s1s2_preservation_correlation"] < 0.85
        ),
        key=lambda item: item["annotated_s1s2_preservation_correlation"],
    )

    report = {
        "dataset_root": str(args.dataset_root),
        "window_seconds": args.window_seconds,
        "files_requested": len(files),
        "files_processed": len(rows),
        "errors": errors,
        "status_counts": dict(status_counts),
        "rerecord_fraction": float(status_counts.get("rerecord", 0) / len(rows)) if rows else None,
        "cleanest_segment_fallback_fraction": (
            float(sum(bool(row["cleanest_segment_is_fallback"]) for row in rows) / len(rows))
            if rows
            else None
        ),
        "rhythm_failure_fraction": (
            float(sum(not bool(row["rhythm_is_preserved"]) for row in rows) / len(rows))
            if rows
            else None
        ),
        "distributions": distributions,
        "annotated_s1s2_below_0_85": annotated_low_tail,
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
