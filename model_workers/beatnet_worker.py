from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The Windows CPU wheels for PyTorch and the pinned NumPy/SciPy stack each
# bundle Intel OpenMP.  BeatNet runs in its own short-lived process, so allow
# that process to load both runtimes and keep it single-threaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from BeatNet.BeatNet import BeatNet


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: beatnet_worker.py INPUT_AUDIO OUTPUT_JSON")
    audio_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    estimator = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False)
    result = np.asarray(estimator.process(str(audio_path)), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 2:
        raise RuntimeError(f"Unexpected BeatNet output shape: {result.shape}")
    beat_times = result[:, 0]
    beat_numbers = result[:, 1].astype(int)
    intervals = np.diff(beat_times)
    payload = {
        "backend": "beatnet_crnn_dbn",
        "beat_times_seconds": [round(float(value), 6) for value in beat_times],
        "beat_numbers": [int(value) for value in beat_numbers],
        "downbeat_times_seconds": [
            round(float(time), 6) for time, number in zip(beat_times, beat_numbers) if number == 1
        ],
        "estimated_bpm": float(60.0 / np.median(intervals)) if len(intervals) else 0.0,
        "estimated_meter": int(np.max(beat_numbers)) if len(beat_numbers) else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
