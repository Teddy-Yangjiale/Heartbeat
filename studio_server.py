from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from heartbeat_preprocessor.core import ProcessingParams
from legasynth.backend import run_heartbeat_stage, run_remix_stage, run_song_stage, run_stem_stage
from legasynth.jobs import create_manifest, new_job_id, read_manifest
from legasynth.stems import backend_availability


ROOT = Path(__file__).resolve().parent
STUDIO_DIR = ROOT / "studio"
OUTPUT_ROOT = ROOT / "outputs" / "studio"
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3"}

app = FastAPI(title="Heartbeat Studio Backend", version="0.3.0")
app.mount("/assets", StaticFiles(directory=STUDIO_DIR), name="studio-assets")


class HeartbeatAnalyzeRequest(BaseModel):
    loop_beats: int = Field(default=4, ge=1, le=32)


class SongAnalyzeRequest(BaseModel):
    bpm_override: float | None = Field(default=None, gt=20.0, le=300.0)
    first_beat_override: float | None = Field(default=None, ge=0.0)
    beats_per_bar: int | None = Field(default=None, ge=1, le=12)


class StemAnalyzeRequest(BaseModel):
    extract_melody: bool = False


class RenderRequest(BaseModel):
    bpm: float | None = Field(default=None, gt=20.0, le=300.0)
    first_beat_seconds: float | None = Field(default=None, ge=0.0)
    heartbeat_gain_db: float = Field(default=-15.0, ge=-60.0, le=6.0)
    allow_manual_review: bool = False


@app.get("/")
def studio_index() -> FileResponse:
    return FileResponse(STUDIO_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "heartbeat-studio",
        "manifest_schema": 2,
        "backends": backend_availability(),
    }


@app.post("/api/jobs")
async def create_job(
    heartbeat: UploadFile | None = File(default=None),
    song: UploadFile | None = File(default=None),
) -> dict:
    if heartbeat is None and song is None:
        raise HTTPException(status_code=422, detail="Upload at least one heartbeat or song file")
    return await _create_job(heartbeat, song)


@app.post("/api/jobs/{job_id}/heartbeat/analyze")
async def analyze_heartbeat(job_id: str, request: HeartbeatAnalyzeRequest) -> dict:
    root = _job_root(job_id, require_manifest=True)
    try:
        await run_in_threadpool(
            run_heartbeat_stage,
            root,
            ProcessingParams(target_loop_beats=request.loop_beats),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"job_id": job_id, "stage": "heartbeat", "error": str(exc)}) from exc
    return _manifest_payload(root)


@app.post("/api/jobs/{job_id}/song/analyze")
async def analyze_song(job_id: str, request: SongAnalyzeRequest) -> dict:
    root = _job_root(job_id, require_manifest=True)
    try:
        await run_in_threadpool(
            run_song_stage,
            root,
            bpm_override=request.bpm_override,
            first_beat_override=request.first_beat_override,
            beats_per_bar=request.beats_per_bar,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"job_id": job_id, "stage": "song", "error": str(exc)}) from exc
    return _manifest_payload(root)


@app.post("/api/jobs/{job_id}/stems/analyze")
async def analyze_stems(job_id: str, request: StemAnalyzeRequest) -> dict:
    root = _job_root(job_id, require_manifest=True)
    await run_in_threadpool(run_stem_stage, root, extract_melody=request.extract_melody)
    return _manifest_payload(root)


@app.post("/api/jobs/{job_id}/render")
async def render_job(job_id: str, request: RenderRequest) -> dict:
    root = _job_root(job_id, require_manifest=True)
    try:
        await run_in_threadpool(
            run_remix_stage,
            root,
            bpm=request.bpm,
            first_beat_seconds=request.first_beat_seconds,
            heartbeat_gain_db=request.heartbeat_gain_db,
            allow_manual_review=request.allow_manual_review,
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail={"job_id": job_id, "stage": "remix", "error": str(exc)}) from exc
    payload = _legacy_studio_payload(root)
    manifest = read_manifest(root)
    remix = manifest["stages"]["remix"]
    payload.update(
        {
            "revision": len(manifest.get("render_revisions", [])),
            "heartbeat_layer_url": _media_url(root, remix["outputs"]["heartbeat_layer_wav"]),
            "final_mix_wav_url": _media_url(root, remix["outputs"]["final_audio_wav"]),
            "final_mix_mp3_url": _media_url(root, remix["outputs"].get("final_audio_mp3")),
            "mix_report": remix["summary"],
        }
    )
    return payload


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _manifest_payload(_job_root(job_id, require_manifest=True))


@app.post("/api/process")
async def compatibility_process(
    heartbeat: UploadFile = File(...),
    song: UploadFile = File(...),
    bpm: float | None = Form(default=None),
    first_beat: float | None = Form(default=None),
    beats_per_bar: int | None = Form(default=None),
    loop_beats: int = Form(default=4),
    heartbeat_gain_db: float = Form(default=-15.0),
    separate_stems: bool = Form(default=False),
    extract_melody: bool = Form(default=False),
) -> dict:
    """Compatibility endpoint for the existing Studio UI.

    The manifest is persisted before analysis starts, so partial results remain
    discoverable even if a later model fails or the client disconnects.
    """
    created = await _create_job(heartbeat, song)
    job_id = created["job_id"]
    root = _job_root(job_id, require_manifest=True)
    try:
        await run_in_threadpool(run_heartbeat_stage, root, ProcessingParams(target_loop_beats=loop_beats))
        await run_in_threadpool(
            run_song_stage,
            root,
            bpm_override=bpm,
            first_beat_override=first_beat,
            beats_per_bar=beats_per_bar,
        )
        if separate_stems:
            await run_in_threadpool(run_stem_stage, root, extract_melody=extract_melody)
        else:
            from legasynth.jobs import update_stage

            update_stage(root, "stems", "skipped", backend="none")
            update_stage(root, "melody", "skipped", backend="none")
        await run_in_threadpool(
            run_remix_stage,
            root,
            bpm=bpm,
            first_beat_seconds=first_beat,
            heartbeat_gain_db=heartbeat_gain_db,
            allow_manual_review=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"job_id": job_id, "error": str(exc), "manifest_url": f"/api/jobs/{job_id}"},
        ) from exc
    return _legacy_studio_payload(root)


@app.get("/media/{job_id}/{file_path:path}")
def media(job_id: str, file_path: str) -> FileResponse:
    root = _job_root(job_id, require_manifest=True)
    target = (root / file_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(target)


async def _create_job(heartbeat: UploadFile | None, song: UploadFile | None) -> dict:
    job_id = new_job_id()
    root = _job_root(job_id, require_manifest=False)
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = await _store_optional_upload(heartbeat, upload_dir, "heartbeat")
    song_path = await _store_optional_upload(song, upload_dir, "song")
    create_manifest(root, job_id, heartbeat_path=heartbeat_path, song_path=song_path)
    return _manifest_payload(root)


async def _store_optional_upload(upload: UploadFile | None, directory: Path, name: str) -> Path | None:
    if upload is None:
        return None
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"{name}: only WAV and MP3 are supported")
    destination = directory / f"{name}{suffix}"
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            output.write(chunk)
    await upload.close()
    return destination


def _job_root(job_id: str, *, require_manifest: bool) -> Path:
    if not job_id.replace("_", "").isalnum():
        raise HTTPException(status_code=404, detail="Job not found")
    root = (OUTPUT_ROOT / job_id).resolve()
    if not root.is_relative_to(OUTPUT_ROOT.resolve()):
        raise HTTPException(status_code=404, detail="Job not found")
    if require_manifest and not (root / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail="Job not found")
    return root


def _manifest_payload(root: Path) -> dict:
    manifest = read_manifest(root)
    return {
        "job_id": manifest["job_id"],
        "manifest": manifest,
        "manifest_url": f"/api/jobs/{manifest['job_id']}",
        "stages": manifest["stages"],
    }


def _legacy_studio_payload(root: Path) -> dict:
    manifest = read_manifest(root)
    heartbeat = manifest["stages"]["heartbeat"]
    song = manifest["stages"]["song"]
    remix = manifest["stages"]["remix"]
    tracks = [
        {
            "id": "song",
            "name": "Original song",
            "kind": "song",
            "url": _media_url(root, manifest["inputs"]["song"]["path"]),
        },
        {
            "id": "heartbeat",
            "name": "Heartbeat layer",
            "kind": "heartbeat",
            "url": _media_url(root, remix["outputs"]["heartbeat_layer_wav"]),
        },
    ]
    stems = manifest["stages"]["stems"]
    for output_name, track_id, label in (
        ("vocals_wav", "vocals", "Vocals"),
        ("accompaniment_wav", "accompaniment", "Accompaniment"),
    ):
        if stems["outputs"].get(output_name):
            tracks.append(
                {
                    "id": track_id,
                    "name": label,
                    "kind": track_id,
                    "url": _media_url(root, stems["outputs"][output_name]),
                }
            )
    return {
        "job_id": manifest["job_id"],
        "analysis": song["summary"],
        "heartbeat_summary": heartbeat["summary"],
        "tracks": tracks,
        "best_loop_url": _media_url(root, heartbeat["outputs"]["best_loop_wav"]),
        "final_mix_wav_url": _media_url(root, remix["outputs"]["final_audio_wav"]),
        "final_mix_mp3_url": _media_url(root, remix["outputs"].get("final_audio_mp3")),
        "melody": manifest["stages"]["melody"]["summary"],
        "stages": manifest["stages"],
    }


def _media_url(root: Path, relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root) or not target.exists():
        raise RuntimeError(f"Missing job output: {target}")
    return f"/media/{root.name}/{quote(target.relative_to(root).as_posix())}"
