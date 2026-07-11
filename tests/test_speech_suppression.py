import io
import unittest

import numpy as np
from scipy.io import wavfile

from heartbeat_preprocessor.core import ProcessingParams, process_audio_bytes


class SpeechSuppressionTest(unittest.TestCase):
    def test_reduces_sustained_voice_between_beats(self) -> None:
        sr = 8000
        seconds = 12
        time = np.arange(sr * seconds, dtype=np.float32) / sr
        voice = 0.22 * (np.sin(2 * np.pi * 110 * time) + 0.45 * np.sin(2 * np.pi * 220 * time))
        heart = np.zeros_like(time)
        rng = np.random.default_rng(7)
        beat_starts = np.arange(0.8, seconds - 0.5, 0.8)
        for beat in beat_starts:
            for offset, scale in ((0.0, 1.0), (0.22, 0.62)):
                start = int((beat + offset) * sr)
                length = int(0.055 * sr)
                burst = rng.standard_normal(length).astype(np.float32) * np.hanning(length).astype(np.float32)
                heart[start : start + length] += scale * burst

        pcm = np.clip(voice + 0.55 * heart, -0.95, 0.95)
        buffer = io.BytesIO()
        wavfile.write(buffer, sr, (pcm * 32767).astype(np.int16))
        without_suppression = process_audio_bytes(
            "synthetic_voice_contaminated.wav",
            buffer.getvalue(),
            ProcessingParams(enable_speech_suppression=False),
        )
        with_suppression = process_audio_bytes("synthetic_voice_contaminated.wav", buffer.getvalue())

        between_beats = np.zeros_like(time, dtype=bool)
        for beat in beat_starts:
            between_beats |= (time >= beat + 0.38) & (time <= beat + 0.68)

        def rms(signal: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(signal[between_beats]))))

        reduction_db = 20.0 * np.log10(
            (rms(with_suppression["cleaned"]) + 1e-12)
            / (rms(without_suppression["cleaned"]) + 1e-12)
        )
        self.assertLess(reduction_db, -20.0)
        self.assertGreaterEqual(len(with_suppression["beat_times"]), 8)


if __name__ == "__main__":
    unittest.main()
