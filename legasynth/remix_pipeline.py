from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

from heartbeat_preprocessor.core import ProcessingParams, process_audio_file, safe_stem, save_result_to_dir
from legasynth.mixing import mix_heartbeat_with_song
from legasynth.song_analysis import analyze_song_audio, save_song_analysis
from legasynth.stems import extract_vocal_melody, separate_vocals_and_accompaniment


def process_song_remix(
    heartbeat_path: str | os.PathLike[str],
    song_path: str | os.PathLike[str],
    out_root: str | os.PathLike[str],
    params: ProcessingParams | None = None,
    heartbeat_gain_db: float = -15.0,
    song_bpm_override: float | None = None,
    first_beat_override: float | None = None,
    beats_per_bar: int = 4,
    separate_stems: bool = False,
    extract_melody: bool = False,
) -> dict[str, Any]:
    """Run Stage 1 heartbeat analysis followed by song-aligned heartbeat remixing."""
    heartbeat_source = Path(heartbeat_path)
    song_source = Path(song_path)
    if not heartbeat_source.is_file():
        raise FileNotFoundError(f"Heartbeat input does not exist: {heartbeat_source}")
    if not song_source.is_file():
        raise FileNotFoundError(f"Song input does not exist: {song_source}")
    if extract_melody and not separate_stems:
        raise ValueError("extract_melody requires separate_stems=True")

    params = params or ProcessingParams(target_loop_beats=max(1, int(beats_per_bar)))
    case_name = f"{safe_stem(heartbeat_source.name)}__{safe_stem(song_source.name)}"
    case_dir = Path(out_root) / case_name
    inputs_dir = case_dir / "inputs"
    heartbeat_input_dir = inputs_dir / "heartbeat"
    song_input_dir = inputs_dir / "song"
    heartbeat_dir = case_dir / "heartbeat_analysis"
    song_dir = case_dir / "song_analysis"
    remix_dir = case_dir / "remix"
    stems_dir = case_dir / "stems"
    reports_dir = case_dir / "reports"
    for directory in (heartbeat_input_dir, song_input_dir, heartbeat_dir, song_dir, remix_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copied_heartbeat = _copy_input(heartbeat_source, heartbeat_input_dir)
    copied_song = _copy_input(song_source, song_input_dir)
    heartbeat_result = process_audio_file(copied_heartbeat, params=params)
    saved_heartbeat = save_result_to_dir(heartbeat_result, heartbeat_dir)
    heartbeat_summary = heartbeat_result["summary"]

    song_analysis = analyze_song_audio(
        copied_song,
        bpm_override=song_bpm_override,
        first_beat_override=first_beat_override,
        beats_per_bar=beats_per_bar,
    )
    song_artifacts = save_song_analysis(song_analysis, song_dir)
    mix_report = mix_heartbeat_with_song(
        song_wav=copied_song,
        loop_wav=saved_heartbeat / "best_loop.wav",
        heartbeat_summary=heartbeat_summary,
        song_bpm=float(song_analysis["estimated_bpm"]),
        out_dir=remix_dir,
        heartbeat_gain_db=heartbeat_gain_db,
        first_beat_seconds=float(song_analysis["first_beat_seconds"]),
        heartbeat_beats_per_loop=int(heartbeat_summary["best_loop"]["num_beats"]),
    )

    stem_report: dict[str, Any] | None = None
    melody_report: dict[str, Any] | None = None
    if separate_stems:
        stem_report = separate_vocals_and_accompaniment(copied_song, stems_dir)
        if extract_melody:
            melody_report = extract_vocal_melody(stem_report["vocals_wav"], stems_dir / "melody")

    report = {
        "stage": "stage2_song_aligned_heartbeat_remix",
        "case_dir": str(case_dir),
        "inputs": {"heartbeat": str(copied_heartbeat), "song": str(copied_song)},
        "heartbeat_summary": heartbeat_summary,
        "song_analysis": song_analysis,
        "song_artifacts": song_artifacts,
        "mix_report": mix_report,
        "stem_separation": stem_report,
        "vocal_melody": melody_report,
        "outputs": {
            "heartbeat_best_loop_wav": str(saved_heartbeat / "best_loop.wav"),
            "heartbeat_layer_wav": mix_report["heartbeat_layer_wav"],
            "final_mix_wav": mix_report["final_audio_wav"],
            "final_mix_mp3": mix_report["final_audio_mp3"],
            "run_report_json": str(reports_dir / "run_report.json"),
            "all_outputs_zip": str(case_dir / "all_outputs.zip"),
        },
    }
    report_path = reports_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _zip_directory(case_dir, case_dir / "all_outputs.zip")
    return report


def _copy_input(source: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _zip_directory(root: Path, destination: Path) -> Path:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in root.rglob("*"):
            if file.is_file() and file != destination:
                archive.write(file, file.relative_to(root))
    return destination
