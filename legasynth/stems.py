from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from legasynth.model_backends import model_backend_availability, module_available, music_worker_python


class StemSeparationUnavailable(RuntimeError):
    pass


def demucs_is_available() -> bool:
    python = music_worker_python()
    return bool(python and module_available("demucs", python))


def backend_availability() -> dict[str, Any]:
    availability = model_backend_availability()
    availability["pyin_fallback"] = True
    return availability


def separate_vocals_and_accompaniment(
    song_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    model: str = "htdemucs",
) -> dict[str, str]:
    """Run Demucs two-stem separation and return stable vocals/accompaniment paths."""
    if not demucs_is_available():
        raise StemSeparationUnavailable(
            "Demucs is not installed in the music worker. Create it with "
            "`conda env create -f environment-music.yml`, then retry."
        )
    song = Path(song_path).resolve()
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = music_worker_python()
    if python is None:
        raise StemSeparationUnavailable("Music model worker Python is unavailable.")
    command = [
        str(python),
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        model,
        "-o",
        str(output),
        str(song),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        tail = (process.stderr or process.stdout or "unknown Demucs error")[-2000:]
        raise RuntimeError(f"Demucs separation failed: {tail}")

    stem_dir = output / model / song.stem
    vocals = stem_dir / "vocals.wav"
    accompaniment = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not accompaniment.exists():
        raise RuntimeError(f"Demucs completed but expected stems were not found in {stem_dir}")
    return {"vocals_wav": str(vocals), "accompaniment_wav": str(accompaniment)}


def extract_vocal_melody(
    vocals_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    fmin_note: str = "C2",
    fmax_note: str = "C7",
    hop_length: int = 256,
    prefer_basic_pitch: bool = True,
) -> dict[str, Any]:
    """Extract vocal notes with Basic Pitch, falling back to frame-level pYIN."""
    vocals = Path(vocals_path)
    python = music_worker_python()
    if prefer_basic_pitch and python is not None and module_available("basic_pitch", python):
        worker = Path(__file__).resolve().parents[1] / "model_workers" / "melody_worker.py"
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="heartbeat_basic_pitch_") as temporary:
            worker_report = Path(temporary) / "report.json"
            process = subprocess.run(
                [str(python), str(worker), str(vocals.resolve()), str(output.resolve()), str(worker_report)],
                capture_output=True,
                text=True,
                timeout=60 * 30,
            )
            if process.returncode == 0 and worker_report.is_file():
                return json.loads(worker_report.read_text(encoding="utf-8"))

    y, sr = librosa.load(str(vocals), sr=None, mono=True)
    if not len(y) or sr <= 0:
        raise ValueError(f"Vocal stem is empty or unreadable: {vocals}")
    f0, voiced_flag, voiced_probability = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz(fmin_note)),
        fmax=float(librosa.note_to_hz(fmax_note)),
        sr=sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    voiced = np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0)
    midi = np.full(len(f0), np.nan, dtype=np.float64)
    midi[voiced] = librosa.hz_to_midi(f0[voiced])
    note_names = [librosa.midi_to_note(value) if np.isfinite(value) else "" for value in midi]
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frequency_hz": f0,
            "midi_note": midi,
            "note_name": note_names,
            "voiced": voiced,
            "voiced_probability": voiced_probability,
        }
    )
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "vocal_melody.csv"
    json_path = output / "vocal_melody_summary.json"
    frame.to_csv(csv_path, index=False)
    voiced_midi = midi[np.isfinite(midi)]
    summary = {
        "backend": "pyin_fallback",
        "vocals_path": str(vocals),
        "sample_rate": int(sr),
        "hop_length": int(hop_length),
        "frame_count": int(len(frame)),
        "voiced_frame_count": int(np.sum(voiced)),
        "voiced_fraction": float(np.mean(voiced)) if len(voiced) else 0.0,
        "median_midi_note": float(np.median(voiced_midi)) if len(voiced_midi) else None,
        "melody_csv": str(csv_path),
        "warnings": ["Basic Pitch is unavailable or failed; exported frame-level pYIN rather than note events."],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["melody_summary_json"] = str(json_path)
    return summary
