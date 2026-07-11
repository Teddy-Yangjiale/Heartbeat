from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartbeat_preprocessor.core import ProcessingParams, process_audio_file, save_result_to_dir
from legasynth.pipeline import process_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Process heartbeat WAV/MP3 files and export Stage 1 parameters.")
    parser.add_argument("files", nargs="*", help="Input heartbeat WAV or MP3 file paths")
    parser.add_argument("--heartbeat", action="append", help="Heartbeat WAV path for full pipeline. Can be repeated.")
    parser.add_argument("--video", help="Music video MP4 path for full pipeline")
    parser.add_argument("--out", default="outputs/cli", help="Output directory")
    parser.add_argument("--loop-beats", type=int, default=4, help="Target loop length in beats")
    parser.add_argument("--heartbeat-volume-db", type=float, default=-15.0, help="Heartbeat mix gain in dB")
    parser.add_argument("--effect-strength", type=float, default=0.75, help="Video pulse effect strength, 0.0 to 1.5")
    parser.add_argument("--duration-limit", type=float, default=None, help="Optional output duration limit in seconds")
    parser.add_argument("--title", default="", help="Optional title or dedication text overlay")
    parser.add_argument("--no-emotion", action="store_true", help="Disable Feature A emotion-driven styling")
    parser.add_argument("--no-beat-editing", action="store_true", help="Disable Feature B heartbeat-driven cuts")
    parser.add_argument("--overlay", action="store_true", help="Show the diagnostic waveform/BPM overlay (off by default for a clean MV)")
    parser.add_argument("--no-subtitles", action="store_true", help="Do not burn Chinese lyric subtitles")
    parser.add_argument("--chinese-cover", action="store_true", help="Enable Feature C Chinese cover")
    parser.add_argument("--chinese-lyrics", help="Path to a UTF-8 text file with Chinese lyrics (one line per lyric line)")
    parser.add_argument("--chinese-lrc", help="Path to a timed .lrc file for the Chinese lyrics")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural", help="edge-tts Chinese voice name")
    parser.add_argument("--cover-vocal-gain-db", type=float, default=-2.0, help="Chinese cover vocal gain in dB")
    args = parser.parse_args()

    cover_options = None
    if args.chinese_cover:
        lyrics_text = Path(args.chinese_lyrics).read_text(encoding="utf-8") if args.chinese_lyrics else None
        lrc_text = Path(args.chinese_lrc).read_text(encoding="utf-8") if args.chinese_lrc else None
        cover_options = {
            "enabled": True,
            "lyrics": lyrics_text,
            "lrc": lrc_text,
            "voice": args.voice,
            "vocal_gain_db": args.cover_vocal_gain_db,
        }

    params = ProcessingParams(target_loop_beats=args.loop_beats)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.video and args.heartbeat:
        for heartbeat in args.heartbeat:
            report = process_one(
                heartbeat_path=heartbeat,
                video_path=args.video,
                out_root=out,
                params=params,
                heartbeat_gain_db=args.heartbeat_volume_db,
                effect_strength=args.effect_strength,
                duration_limit=args.duration_limit,
                title_text=args.title,
                enable_emotion=not args.no_emotion,
                enable_beat_editing=not args.no_beat_editing,
                show_overlay=args.overlay,
                enable_subtitles=not args.no_subtitles,
                chinese_cover_options=cover_options,
            )
            print(
                f"{Path(heartbeat).name}: final_video={report['outputs']['final_video_mp4']}, "
                f"final_audio={report['outputs']['final_audio_wav']}, zip={report['outputs']['all_outputs_zip']}"
            )
        return

    if args.video and not args.heartbeat:
        parser.error("--video requires at least one --heartbeat path")

    if not args.files:
        parser.error("provide WAV files for legacy mode or use --heartbeat PATH --video PATH")

    for file in args.files:
        result = process_audio_file(file, params=params)
        saved = save_result_to_dir(result, out)
        summary = result["summary"]
        print(
            f"{Path(file).name}: BPM={summary['tempo']['estimated_bpm']:.1f}, "
            f"beats={summary['tempo']['detected_beats']}, saved={saved}"
        )


if __name__ == "__main__":
    main()
