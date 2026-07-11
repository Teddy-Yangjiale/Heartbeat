from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from heartbeat_preprocessor.core import ProcessingParams, process_wav_file, safe_stem, save_result_to_dir
from legasynth.mixing import mix_heartbeat_with_song
from legasynth.video_audio import prepare_video_audio
from legasynth.video_render import render_heartbeat_video


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def copy_input(path: str | os.PathLike[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(path)
    dst = out_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def zip_directory(root: str | os.PathLike[str], zip_path: str | os.PathLike[str] | None = None) -> Path:
    root = Path(root)
    zip_path = Path(zip_path) if zip_path else root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in root.rglob("*"):
            if file.is_file() and file != zip_path:
                zf.write(file, file.relative_to(root))
    return zip_path


def process_one(
    heartbeat_path: str | os.PathLike[str],
    video_path: str | os.PathLike[str],
    out_root: str | os.PathLike[str],
    params: ProcessingParams | None = None,
    heartbeat_gain_db: float = -15.0,
    effect_strength: float = 0.75,
    duration_limit: float | None = None,
    title_text: str = "",
) -> dict[str, Any]:
    params = params or ProcessingParams()
    heartbeat_path = Path(heartbeat_path)
    out_root = Path(out_root)
    stem = safe_stem(heartbeat_path.name)
    case_dir = out_root / stem
    input_dir = case_dir / "inputs"
    pre_dir = case_dir / "preprocessing"
    song_dir = case_dir / "song_analysis"
    mix_dir = case_dir / "audio_mix"
    video_dir = case_dir / "video"
    reports_dir = case_dir / "reports"
    for d in [input_dir, pre_dir, song_dir, mix_dir, video_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    copied_heartbeat = copy_input(heartbeat_path, input_dir)
    copied_video = copy_input(video_path, input_dir)

    pre_result = process_wav_file(copied_heartbeat, params=params)
    save_result_to_dir(pre_result, pre_dir.parent)
    flat_pre = pre_dir.parent / pre_result["stem"]
    if flat_pre != pre_dir and flat_pre.exists():
        if pre_dir.exists():
            shutil.rmtree(pre_dir)
        flat_pre.rename(pre_dir)
    heartbeat_summary = pre_result["summary"]

    video_meta = prepare_video_audio(copied_video, song_dir)
    loop_wav = pre_dir / "best_loop.wav"
    mix_report = mix_heartbeat_with_song(
        song_wav=video_meta["extracted_audio_path"],
        loop_wav=loop_wav,
        heartbeat_summary=heartbeat_summary,
        song_bpm=float(video_meta["estimated_song_bpm"] or heartbeat_summary["tempo"]["estimated_bpm"]),
        out_dir=mix_dir,
        heartbeat_gain_db=heartbeat_gain_db,
    )
    aligned_beats = pd.read_csv(mix_report["aligned_heartbeat_beats_csv"])["time_seconds"].astype(float).tolist()
    video_report = render_heartbeat_video(
        source_video=copied_video,
        final_audio=mix_report["final_audio_wav"],
        loop_wav=loop_wav,
        beat_times=aligned_beats,
        out_dir=video_dir,
        heartbeat_bpm=float(heartbeat_summary["best_loop"]["local_bpm"]),
        title_text=title_text,
        effect_strength=effect_strength,
        duration_limit=duration_limit,
    )

    run_report = {
        "heartbeat_file": str(copied_heartbeat),
        "video_file": str(copied_video),
        "case_dir": str(case_dir),
        "preprocessing": heartbeat_summary,
        "video_metadata": video_meta,
        "mix_report": mix_report,
        "video_report": video_report,
        "outputs": {
            "final_video_mp4": video_report["final_video"],
            "final_audio_wav": mix_report["final_audio_wav"],
            "final_audio_mp3": mix_report["final_audio_mp3"],
            "all_outputs_zip": str(case_dir / "all_outputs.zip"),
        },
    }
    (reports_dir / "diagnostic_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
    zip_path = zip_directory(case_dir, case_dir / "all_outputs.zip")
    run_report["outputs"]["all_outputs_zip"] = str(zip_path)
    return run_report

