from __future__ import annotations

from pathlib import Path
from typing import Any

from heartbeat_preprocessor.core import ProcessingParams, process_audio_file, save_result_to_dir
from legasynth.jobs import append_render_revision, read_manifest, resolve_input, resolve_output, update_stage
from legasynth.mixing import mix_heartbeat_with_song
from legasynth.song_analysis import analyze_song_audio, save_song_analysis
from legasynth.stems import (
    StemSeparationUnavailable,
    extract_vocal_melody,
    separate_vocals_and_accompaniment,
)


def run_heartbeat_stage(job_root: str | Path, params: ProcessingParams | None = None) -> dict[str, Any]:
    root = Path(job_root).resolve()
    update_stage(root, "heartbeat", "running", progress=0.05, backend="pcg_consensus_v2")
    try:
        source = resolve_input(root, "heartbeat")
        result = process_audio_file(source, params=params or ProcessingParams())
        output_dir = save_result_to_dir(result, root / "heartbeat")
        summary = result["summary"]
        recording_quality = summary["recording_quality"]
        loop_validation = summary["best_loop"].get("validation", {})
        needs_review = (
            not recording_quality["is_recommended_for_loop"]
            or loop_validation.get("status") == "manual_review"
        )
        warnings = list(recording_quality.get("reasons", []))
        warnings.extend(loop_validation.get("warnings", []))
        status = "manual_review" if needs_review else "ok"
        outputs = {
            "directory": str(output_dir),
            "best_loop_wav": str(output_dir / "best_loop.wav"),
            "tempo_summary_json": str(output_dir / "tempo_summary.json"),
            "beat_times_csv": str(output_dir / "beat_times.csv"),
            "recording_quality_json": str(output_dir / "recording_quality.json"),
            "diagnostic_plot_png": str(output_dir / "diagnostic_plot.png"),
        }
        update_stage(
            root,
            "heartbeat",
            status,
            backend="pcg_consensus_v2",
            outputs=outputs,
            summary=summary,
            warnings=_meaningful_warnings(warnings),
        )
        return read_manifest(root)
    except Exception as exc:
        update_stage(root, "heartbeat", "failed", backend="pcg_consensus_v2", error=str(exc))
        raise


def run_song_stage(
    job_root: str | Path,
    *,
    bpm_override: float | None = None,
    first_beat_override: float | None = None,
    beats_per_bar: int | None = None,
) -> dict[str, Any]:
    root = Path(job_root).resolve()
    update_stage(root, "song", "running", progress=0.05, backend="beatnet_or_librosa_v2")
    try:
        source = resolve_input(root, "song")
        analysis = analyze_song_audio(
            source,
            bpm_override=bpm_override,
            first_beat_override=first_beat_override,
            beats_per_bar=beats_per_bar,
        )
        artifacts = save_song_analysis(analysis, root / "song")
        status = "manual_review" if analysis["requires_manual_review"] else "ok"
        update_stage(
            root,
            "song",
            status,
            backend=analysis.get("backend", "librosa_dynamic_beat"),
            outputs=artifacts,
            summary=analysis,
            warnings=analysis.get("warnings", []),
        )
        return read_manifest(root)
    except Exception as exc:
        update_stage(root, "song", "failed", backend="beatnet_or_librosa_v2", error=str(exc))
        raise


def run_stem_stage(job_root: str | Path, *, extract_melody: bool = False) -> dict[str, Any]:
    root = Path(job_root).resolve()
    update_stage(root, "stems", "running", progress=0.02, backend="demucs_htdemucs")
    try:
        song = resolve_input(root, "song")
        report = separate_vocals_and_accompaniment(song, root / "stems")
        update_stage(root, "stems", "ok", backend="demucs_htdemucs", outputs=report, summary={"model": "htdemucs"})
    except StemSeparationUnavailable as exc:
        update_stage(root, "stems", "unavailable", backend="demucs_htdemucs", warnings=[str(exc)], error=str(exc))
        if extract_melody:
            update_stage(root, "melody", "unavailable", backend="basic_pitch_or_pyin", warnings=["Vocal stem is unavailable."])
        return read_manifest(root)
    except Exception as exc:
        update_stage(root, "stems", "failed", backend="demucs_htdemucs", error=str(exc))
        if extract_melody:
            update_stage(root, "melody", "skipped", backend="basic_pitch_or_pyin", warnings=["Stem separation failed."])
        return read_manifest(root)

    if extract_melody:
        run_melody_stage(root)
    else:
        update_stage(root, "melody", "skipped", backend="none")
    return read_manifest(root)


def run_melody_stage(job_root: str | Path) -> dict[str, Any]:
    root = Path(job_root).resolve()
    update_stage(root, "melody", "running", progress=0.02, backend="pyin_fallback")
    try:
        vocals = resolve_output(root, "stems", "vocals_wav")
        summary = extract_vocal_melody(vocals, root / "melody")
        outputs = {
            "melody_csv": summary["melody_csv"],
            "melody_summary_json": summary["melody_summary_json"],
        }
        if summary.get("melody_midi"):
            outputs["melody_midi"] = summary["melody_midi"]
        update_stage(
            root,
            "melody",
            "ok",
            backend=summary.get("backend", "pyin_fallback"),
            outputs=outputs,
            summary=summary,
            warnings=summary.get("warnings", []),
        )
    except Exception as exc:
        update_stage(root, "melody", "failed", backend="pyin_fallback", error=str(exc))
    return read_manifest(root)


def run_remix_stage(
    job_root: str | Path,
    *,
    bpm: float | None = None,
    first_beat_seconds: float | None = None,
    heartbeat_gain_db: float = -15.0,
    allow_manual_review: bool = False,
) -> dict[str, Any]:
    root = Path(job_root).resolve()
    manifest = read_manifest(root)
    heartbeat_stage = manifest["stages"]["heartbeat"]
    song_stage = manifest["stages"]["song"]
    accepted = {"ok"} | ({"manual_review"} if allow_manual_review else set())
    if heartbeat_stage["status"] not in accepted:
        raise ValueError(f"Heartbeat stage is not verified: {heartbeat_stage['status']}")
    if song_stage["status"] not in accepted:
        raise ValueError(f"Song stage is not verified: {song_stage['status']}")

    update_stage(root, "remix", "running", progress=0.05, backend="phase_vocoder_tempo_map_v2")
    try:
        song_path = resolve_input(root, "song")
        loop_path = resolve_output(root, "heartbeat", "best_loop_wav")
        heartbeat_summary = heartbeat_stage["summary"]
        song_summary = song_stage["summary"]
        selected_bpm = float(bpm or song_summary["estimated_bpm"])
        selected_first_beat = float(
            song_summary["first_beat_seconds"] if first_beat_seconds is None else first_beat_seconds
        )
        revision = len(manifest.get("render_revisions", [])) + 1
        report = mix_heartbeat_with_song(
            song_wav=song_path,
            loop_wav=loop_path,
            heartbeat_summary=heartbeat_summary,
            song_bpm=selected_bpm,
            out_dir=root / "remix" / f"revision_{revision:03d}",
            heartbeat_gain_db=heartbeat_gain_db,
            first_beat_seconds=selected_first_beat,
            heartbeat_beats_per_loop=int(heartbeat_summary["best_loop"]["num_beats"]),
            song_beat_times=(
                song_summary.get("beat_grid_times_seconds")
                if bpm is None and first_beat_seconds is None
                else None
            ),
        )
        warnings: list[str] = []
        if heartbeat_stage["status"] == "manual_review" or song_stage["status"] == "manual_review":
            warnings.append("Rendered with manually reviewed analysis; verify alignment by listening.")
        update_stage(
            root,
            "remix",
            "ok",
            backend="phase_vocoder_tempo_map_v2",
            outputs=report,
            summary=report,
            warnings=warnings,
        )
        append_render_revision(
            root,
            {
                "revision": revision,
                "bpm": selected_bpm,
                "first_beat_seconds": selected_first_beat,
                "heartbeat_gain_db": heartbeat_gain_db,
                "outputs": report,
            },
        )
        return read_manifest(root)
    except Exception as exc:
        update_stage(root, "remix", "failed", backend="phase_vocoder_tempo_map_v2", error=str(exc))
        raise


def _meaningful_warnings(values: list[str]) -> list[str]:
    return [value for value in values if value and not value.startswith("No major automated")]
