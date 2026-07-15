from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def module_available(module: str, python_executable: str | os.PathLike[str] | None = None) -> bool:
    if python_executable is None or Path(python_executable).resolve() == Path(sys.executable).resolve():
        return importlib.util.find_spec(module) is not None
    process = subprocess.run(
        [str(python_executable), "-c", f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{module}') else 1)"],
        capture_output=True,
        timeout=30,
    )
    return process.returncode == 0


def music_worker_python() -> Path | None:
    explicit = os.environ.get("HEARTBEAT_MUSIC_PYTHON")
    candidates = [
        Path(explicit) if explicit else None,
        Path(r"D:\Anaconda\envs\heartbeat-music\python.exe"),
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def beatnet_available() -> bool:
    python = music_worker_python()
    return bool(python and module_available("BeatNet", python))


def run_beatnet(audio_path: str | os.PathLike[str]) -> dict[str, Any]:
    python = music_worker_python()
    if python is None or not module_available("BeatNet", python):
        raise RuntimeError("BeatNet worker is not installed")
    worker = ROOT / "model_workers" / "beatnet_worker.py"
    with tempfile.TemporaryDirectory(prefix="heartbeat_beatnet_") as temporary:
        output_path = Path(temporary) / "beatnet.json"
        process = subprocess.run(
            [str(python), str(worker), str(Path(audio_path).resolve()), str(output_path)],
            capture_output=True,
            text=True,
            timeout=60 * 30,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"},
        )
        if process.returncode != 0 or not output_path.is_file():
            message = (process.stderr or process.stdout or "BeatNet worker failed")[-3000:]
            raise RuntimeError(message)
        return json.loads(output_path.read_text(encoding="utf-8"))


def model_backend_availability() -> dict[str, Any]:
    python = music_worker_python()
    return {
        "music_worker_python": str(python) if python else None,
        "beatnet": bool(python and module_available("BeatNet", python)),
        "demucs": bool(python and module_available("demucs", python)),
        "basic_pitch": bool(python and module_available("basic_pitch", python)),
    }
