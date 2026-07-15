from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = ("heartbeat", "song", "stems", "melody", "remix")
TERMINAL_STATUSES = {"ok", "manual_review", "rejected", "failed", "unavailable", "skipped"}
_LOCKS_GUARD = threading.Lock()
_JOB_LOCKS: dict[str, threading.RLock] = {}


def _job_lock(job_root: str | os.PathLike[str]) -> threading.RLock:
    key = str(Path(job_root).resolve()).lower()
    with _LOCKS_GUARD:
        return _JOB_LOCKS.setdefault(key, threading.RLock())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def create_manifest(
    job_root: str | os.PathLike[str],
    job_id: str,
    heartbeat_path: str | os.PathLike[str] | None = None,
    song_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    root = Path(job_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, Any] = {}
    if heartbeat_path is not None:
        inputs["heartbeat"] = describe_input(heartbeat_path, root)
    if song_path is not None:
        inputs["song"] = describe_input(song_path, root)
    manifest = {
        "schema_version": 2,
        "job_id": job_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "inputs": inputs,
        "stages": {name: blank_stage() for name in STAGES},
        "render_revisions": [],
    }
    with _job_lock(root):
        write_manifest(root, manifest)
    return manifest


def blank_stage() -> dict[str, Any]:
    return {
        "status": "pending",
        "progress": 0.0,
        "started_at": None,
        "finished_at": None,
        "backend": None,
        "outputs": {},
        "summary": {},
        "warnings": [],
        "error": None,
    }


def describe_input(path: str | os.PathLike[str], job_root: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": relative_job_path(job_root, source),
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(job_root: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(job_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(job_root: str | os.PathLike[str], manifest: dict[str, Any]) -> None:
    root = Path(job_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = utc_now()
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, root / "manifest.json")


def update_stage(
    job_root: str | os.PathLike[str],
    stage_name: str,
    status: str,
    *,
    progress: float | None = None,
    backend: str | None = None,
    outputs: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if stage_name not in STAGES:
        raise ValueError(f"Unknown job stage: {stage_name}")
    with _job_lock(job_root):
        manifest = read_manifest(job_root)
        stage = manifest["stages"][stage_name]
        stage["status"] = status
        if status == "running" and stage["started_at"] is None:
            stage["started_at"] = utc_now()
        if status in TERMINAL_STATUSES:
            stage["finished_at"] = utc_now()
            stage["progress"] = 1.0
        elif progress is not None:
            stage["progress"] = float(max(0.0, min(1.0, progress)))
        if backend is not None:
            stage["backend"] = backend
        if outputs is not None:
            stage["outputs"] = make_job_relative(job_root, outputs)
        if summary is not None:
            stage["summary"] = make_job_relative(job_root, summary)
        if warnings is not None:
            stage["warnings"] = list(warnings)
        stage["error"] = error
        write_manifest(job_root, manifest)
    return manifest


def append_render_revision(job_root: str | os.PathLike[str], revision: dict[str, Any]) -> dict[str, Any]:
    with _job_lock(job_root):
        manifest = read_manifest(job_root)
        manifest.setdefault("render_revisions", []).append(make_job_relative(job_root, revision))
        write_manifest(job_root, manifest)
    return manifest


def resolve_input(job_root: str | os.PathLike[str], input_name: str) -> Path:
    root = Path(job_root).resolve()
    manifest = read_manifest(root)
    descriptor = manifest.get("inputs", {}).get(input_name)
    if not descriptor:
        raise ValueError(f"Job has no {input_name} input")
    target = (root / descriptor["path"]).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FileNotFoundError(target)
    return target


def resolve_output(job_root: str | os.PathLike[str], stage_name: str, output_name: str) -> Path:
    root = Path(job_root).resolve()
    manifest = read_manifest(root)
    value = manifest["stages"][stage_name]["outputs"].get(output_name)
    if not value:
        raise ValueError(f"Stage {stage_name} has no output {output_name}")
    target = (root / value).resolve()
    if not target.is_relative_to(root) or not target.exists():
        raise FileNotFoundError(target)
    return target


def relative_job_path(job_root: str | os.PathLike[str], value: str | os.PathLike[str]) -> str:
    root = Path(job_root).resolve()
    target = Path(value).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path is outside job directory: {target}")
    return target.relative_to(root).as_posix()


def make_job_relative(job_root: str | os.PathLike[str], value: Any) -> Any:
    root = Path(job_root).resolve()
    if isinstance(value, dict):
        return {key: make_job_relative(root, item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_job_relative(root, item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute() and path.exists() and path.resolve().is_relative_to(root):
            return path.resolve().relative_to(root).as_posix()
    return value
