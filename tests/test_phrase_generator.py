from __future__ import annotations

import io
import unittest
import zipfile

import numpy as np
from scipy.io import wavfile

from legasynth.phrase_generator import PhraseRequest, generate_phrase_candidates, parse_anchor_degrees


class PhraseGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = {"tempo": {"estimated_bpm": 72.0}}
        self.beat_times = np.array([0.0, 0.82, 1.68, 2.47, 3.35, 4.18, 5.02])

    def test_generates_distinct_editable_candidates(self) -> None:
        request = PhraseRequest(anchor_degrees=(4, 2, 7), candidate_count=4)
        result = generate_phrase_candidates(self.summary, self.beat_times, request)

        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual(result["manifest"]["request"]["anchor_degrees"], (4, 2, 7))
        self.assertEqual(len({candidate["name"] for candidate in result["candidates"]}), 4)
        for candidate in result["candidates"]:
            self.assertTrue(candidate["midi_bytes"].startswith(b"MThd"))
            sample_rate, audio = wavfile.read(io.BytesIO(candidate["wav_bytes"]))
            self.assertEqual(sample_rate, 44100)
            self.assertGreater(len(audio), 1000)

    def test_generation_is_deterministic(self) -> None:
        request = PhraseRequest(anchor_degrees=(1, 5, 3), intention="Grounded")
        first = generate_phrase_candidates(self.summary, self.beat_times, request)
        second = generate_phrase_candidates(self.summary, self.beat_times, request)
        self.assertEqual(first["candidates"][0]["midi_bytes"], second["candidates"][0]["midi_bytes"])

    def test_zip_contains_preview_midi_and_manifest(self) -> None:
        result = generate_phrase_candidates(self.summary, self.beat_times, PhraseRequest(candidate_count=2))
        with zipfile.ZipFile(io.BytesIO(result["zip_bytes"])) as archive:
            names = set(archive.namelist())
        self.assertIn("candidate_manifest.json", names)
        self.assertIn("candidate_01.wav", names)
        self.assertIn("candidate_01.mid", names)

    def test_anchor_parser_rejects_out_of_range_values(self) -> None:
        self.assertEqual(parse_anchor_degrees("4, 2, 7"), (4, 2, 7))
        with self.assertRaises(ValueError):
            parse_anchor_degrees("1, 8")


if __name__ == "__main__":
    unittest.main()
