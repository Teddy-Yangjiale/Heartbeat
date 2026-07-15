import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from scripts.analyze_blind_ab_scores import METRICS, analyze, read_scores
from scripts.create_blind_ab_pack import valid_case_id, validate_case


class BlindAbPackTest(unittest.TestCase):
    def test_case_id_accepts_safe_underscores_and_hyphens(self) -> None:
        self.assertTrue(valid_case_id("sample_10s"))
        self.assertTrue(valid_case_id("subject-001"))
        self.assertFalse(valid_case_id("../answer"))
        self.assertFalse(valid_case_id("case name"))

    def test_wav_geometry_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            v1 = root / "v1.wav"
            candidate = root / "candidate.wav"
            wavfile.write(reference, 4000, np.zeros(4000, dtype=np.int16))
            wavfile.write(v1, 4000, np.zeros(4000, dtype=np.int16))
            wavfile.write(candidate, 4000, np.zeros(3900, dtype=np.int16))
            with self.assertRaisesRegex(ValueError, "identical WAV geometry"):
                validate_case("case_1", reference, v1, candidate)

    def test_incomplete_scores_cannot_be_revealed_as_a_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scores = Path(directory) / "scores.csv"
            scores.write_text(
                "case_id,variant," + ",".join(METRICS) + ",notes\n"
                "case_1,A," + ",".join("5" for _ in METRICS) + ",\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_scores(scores, {("case_1", "A"): "candidate", ("case_1", "B"): "v1"})

    def test_candidate_gate_requires_noise_gain_without_safety_regression(self) -> None:
        rows = []
        for identity, scores in (
            ("v1", [3, 3, 4, 4, 4, 4]),
            ("candidate", [4, 3, 4, 4, 4, 4]),
        ):
            rows.append(
                {
                    "case_id": "case_1",
                    "variant": "A" if identity == "candidate" else "B",
                    "identity": identity,
                    "scores": dict(zip(METRICS, scores)),
                    "notes": "",
                }
            )
        self.assertTrue(analyze(rows)["decision_gate"]["candidate_passes"])
        rows[1]["scores"]["s1s2_fullness_1_to_5"] = 3
        self.assertFalse(analyze(rows)["decision_gate"]["candidate_passes"])


if __name__ == "__main__":
    unittest.main()
