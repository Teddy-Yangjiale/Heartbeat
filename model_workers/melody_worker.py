from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Basic Pitch 0.4 annotates with numpy.typing.NDArray, while BeatNet's
# compatible NumPy 1.20 predates that alias.  The type is not used at runtime,
# so this small worker-local shim lets both pretrained models share the
# intentionally isolated Python 3.9 environment.
import numpy as np
import numpy.typing as npt

if not hasattr(npt, "NDArray"):
    class _NDArrayCompat:
        def __class_getitem__(cls, _item):
            return np.ndarray

    npt.NDArray = _NDArrayCompat  # type: ignore[attr-defined]

from basic_pitch.inference import predict


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: melody_worker.py VOCALS_AUDIO OUTPUT_DIR REPORT_JSON")
    audio_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    report_path = Path(sys.argv[3]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, midi_data, note_events = predict(str(audio_path))
    midi_path = output_dir / "vocal_melody.mid"
    csv_path = output_dir / "vocal_note_events.csv"
    midi_data.write(str(midi_path))
    rows = []
    for event in note_events:
        start, end, pitch, amplitude, *rest = event
        pitch_bends = rest[0] if rest else []
        rows.append(
            {
                "start_seconds": float(start),
                "end_seconds": float(end),
                "duration_seconds": float(end - start),
                "midi_note": int(pitch),
                "amplitude": float(amplitude),
                "pitch_bends": json.dumps(np.asarray(pitch_bends).tolist()),
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]) if rows else [
            "start_seconds", "end_seconds", "duration_seconds", "midi_note", "amplitude", "pitch_bends"
        ])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "backend": "basic_pitch_pretrained",
        "vocals_path": str(audio_path),
        "note_event_count": len(rows),
        "melody_csv": str(csv_path),
        "melody_midi": str(midi_path),
        "warnings": [],
    }
    summary_path = output_dir / "vocal_melody_summary.json"
    summary["melody_summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
