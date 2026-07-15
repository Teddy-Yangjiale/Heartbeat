from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartbeat_preprocessor.core import ProcessingParams, process_audio_file, save_result_to_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise one or more short heartbeat WAV files.")
    parser.add_argument("files", nargs="+", help="Input heartbeat WAV file paths")
    parser.add_argument("--out", default="outputs/cli", help="Output directory")
    args = parser.parse_args()

    params = ProcessingParams()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for file in args.files:
        result = process_audio_file(file, params=params)
        saved = save_result_to_dir(result, out)
        summary = result["summary"]
        print(
            f"{Path(file).name}: BPM={summary['tempo']['estimated_bpm']:.1f}, "
            f"beats={summary['tempo']['detected_beats']}, "
            f"status={summary['recording_quality']['denoising_status']}, saved={saved}"
        )


if __name__ == "__main__":
    main()
