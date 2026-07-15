from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartbeat_preprocessor.core import ProcessingParams
from legasynth.remix_pipeline import process_song_remix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a song, align the processed heartbeat best loop to its beat grid, and export a remix."
    )
    parser.add_argument("--heartbeat", required=True, help="Raw heartbeat WAV/MP3 input")
    parser.add_argument("--song", required=True, help="Song WAV/MP3 input")
    parser.add_argument("--out", default="outputs/remix", help="Output root directory")
    parser.add_argument("--beats-per-bar", type=int, default=4, help="Song meter used for exported bar labels")
    parser.add_argument("--loop-beats", type=int, default=None, help="Heartbeat intervals in one loop; defaults to beats-per-bar")
    parser.add_argument("--bpm", type=float, default=None, help="Manual song BPM override")
    parser.add_argument("--first-beat", type=float, default=None, help="Manual first-beat/phase anchor in seconds")
    parser.add_argument("--heartbeat-gain-db", type=float, default=-15.0, help="Heartbeat layer gain in dB")
    parser.add_argument("--separate-stems", action="store_true", help="Run optional Demucs vocals/accompaniment separation")
    parser.add_argument("--extract-melody", action="store_true", help="Extract pYIN melody from the separated vocal stem")
    args = parser.parse_args()

    if args.beats_per_bar < 1:
        parser.error("--beats-per-bar must be at least 1")
    loop_beats = args.loop_beats or args.beats_per_bar
    if loop_beats < 1:
        parser.error("--loop-beats must be at least 1")
    if args.extract_melody and not args.separate_stems:
        parser.error("--extract-melody requires --separate-stems")

    report = process_song_remix(
        heartbeat_path=args.heartbeat,
        song_path=args.song,
        out_root=args.out,
        params=ProcessingParams(target_loop_beats=loop_beats),
        heartbeat_gain_db=args.heartbeat_gain_db,
        song_bpm_override=args.bpm,
        first_beat_override=args.first_beat,
        beats_per_bar=args.beats_per_bar,
        separate_stems=args.separate_stems,
        extract_melody=args.extract_melody,
    )
    outputs = report["outputs"]
    print(f"Song BPM: {report['song_analysis']['estimated_bpm']:.3f}")
    print(f"First beat: {report['song_analysis']['first_beat_seconds']:.3f}s")
    print(f"Heartbeat loop: {outputs['heartbeat_best_loop_wav']}")
    print(f"Heartbeat layer: {outputs['heartbeat_layer_wav']}")
    print(f"Final WAV: {outputs['final_mix_wav']}")
    if outputs["final_mix_mp3"]:
        print(f"Final MP3: {outputs['final_mix_mp3']}")
    print(f"All outputs: {outputs['all_outputs_zip']}")


if __name__ == "__main__":
    main()
