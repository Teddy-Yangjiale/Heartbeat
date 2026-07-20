from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from heartbeat_preprocessor.core import process_audio_bytes
from music_processor.core import MixParams, analyze_song_bytes, process_music_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real end-to-end safe-mode validation.")
    parser.add_argument("heartbeat", type=Path)
    parser.add_argument("song", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("tmp"))
    parser.add_argument("--max-song-seconds", type=float, default=300.0)
    args = parser.parse_args()

    output_dir = args.output_root / f"real_validation_{int(time.time())}"
    with args.heartbeat.open("rb") as heartbeat_source, args.song.open("rb") as song_source:
        heartbeat = process_audio_bytes(
            args.heartbeat.name,
            heartbeat_source,
            max_duration_seconds=30.0,
            artifact_profile="web",
            create_zip=False,
        )
        analysis = analyze_song_bytes(
            args.song.name,
            song_source,
            max_duration_seconds=args.max_song_seconds,
        )
        result = process_music_bytes(
            args.song.name,
            song_source,
            heartbeat,
            analysis,
            MixParams(),
            output_dir=output_dir,
            export_stems=False,
            export_debug=False,
            create_zip=False,
        )
    summary = {
        "output_dir": str(output_dir.resolve()),
        "duration_seconds": result["duration_seconds"],
        "song_bpm": analysis["estimated_bpm"],
        "beat_confidence": analysis["beat_tracking_confidence"],
        "downbeat_confidence": analysis["downbeat_confidence"],
        "pulse_count": result["report"]["render"]["pulse_count"],
        "model_backed": result["report"]["render"]["model_backed_pulse_count"],
        "guide_pulses": result["report"]["render"]["guide_pulse_count"],
        "anchor_offsets_ms": result["report"]["render"]["anchor_offsets_ms"],
        "maximum_anchor_alignment_error_ms": result["report"]["render"][
            "maximum_anchor_alignment_error_ms"
        ],
        "memory": result["report"]["memory"],
        "output_lufs": result["report"]["master"]["output_lufs"],
        "output_peak_dbfs": result["report"]["master"]["output_peak_dbfs"],
        "files": {
            name: Path(path).stat().st_size
            for name, path in result["artifact_paths"].items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
