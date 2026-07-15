import hashlib
import json
import os
import unittest
from pathlib import Path

from heartbeat_preprocessor.core import process_audio_file


class RealHeartbeatRegressionTest(unittest.TestCase):
    def test_annotated_local_recordings_stay_within_validated_ranges(self) -> None:
        fixture_dir = Path(
            os.environ.get(
                "HEARTBEAT_REAL_FIXTURE_DIR",
                Path(__file__).parent / "fixtures" / "real_heartbeat",
            )
        )
        expected_path = Path(__file__).parent / "fixtures" / "heartbeat_real_expected.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        missing = [name for name in expected if not (fixture_dir / name).is_file()]
        if missing:
            self.skipTest(
                "Set HEARTBEAT_REAL_FIXTURE_DIR to the private validated recordings directory; "
                f"missing {', '.join(missing)}"
            )

        for filename, contract in expected.items():
            with self.subTest(filename=filename):
                path = fixture_dir / filename
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, contract["sha256"], "Fixture content changed")
                result = process_audio_file(path)
                summary = result["summary"]
                bpm = summary["tempo"]["estimated_bpm"]
                loop = summary["best_loop"]
                self.assertGreaterEqual(bpm, contract["global_bpm_range"][0])
                self.assertLessEqual(bpm, contract["global_bpm_range"][1])
                self.assertGreaterEqual(loop["local_bpm"], contract["loop_bpm_range"][0])
                self.assertLessEqual(loop["local_bpm"], contract["loop_bpm_range"][1])
                self.assertLessEqual(loop["period_error_fraction"], contract["max_loop_period_error"])
                self.assertEqual(loop["validation"]["status"], "ok")
                self.assertEqual(
                    summary["recording_quality"]["is_recommended_for_loop"],
                    contract["expected_recommendation"],
                )


if __name__ == "__main__":
    unittest.main()
