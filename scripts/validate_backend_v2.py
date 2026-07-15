from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartbeat_preprocessor.core import ProcessingParams
from legasynth.backend import (
    run_heartbeat_stage,
    run_remix_stage,
    run_song_stage,
    run_stem_stage,
)
from legasynth.jobs import create_manifest, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete backend-v2 validation pipeline.")
    parser.add_argument("heartbeat", type=Path)
    parser.add_argument("song", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-manual-review", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output.resolve()
    upload = root / "uploads"
    upload.mkdir(parents=True, exist_ok=True)
    heartbeat = upload / ("heartbeat" + args.heartbeat.suffix.lower())
    song = upload / ("song" + args.song.suffix.lower())
    shutil.copy2(args.heartbeat.resolve(), heartbeat)
    shutil.copy2(args.song.resolve(), song)
    create_manifest(root, root.name, heartbeat_path=heartbeat, song_path=song)

    run_heartbeat_stage(root, ProcessingParams(target_loop_beats=4))
    run_song_stage(root)
    run_stem_stage(root, extract_melody=True)
    run_remix_stage(root, allow_manual_review=args.allow_manual_review)

    manifest = read_manifest(root)
    print(json.dumps({name: stage["status"] for name, stage in manifest["stages"].items()}, indent=2))
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
