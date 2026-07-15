import unittest

from scripts.compare_candidate_outputs import automated_prescreen


class CandidateComparisonTest(unittest.TestCase):
    @staticmethod
    def metrics(interbeat_db: float) -> dict:
        return {
            "rhythm_is_preserved": True,
            "clipping_fraction": 0.0,
            "interbeat_change_after_heart_alignment_db": interbeat_db,
        }

    def test_faithful_candidate_can_pass_without_being_quieter_than_v1(self) -> None:
        result = automated_prescreen(
            self.metrics(-22.0),
            self.metrics(-10.0),
            correlation_delta=0.08,
            attack_delta=0.02,
            spectral_distance_delta=-0.04,
        )

        self.assertTrue(result["interbeat_noise_reduced_from_reference"])
        self.assertTrue(result["passes"])

    def test_candidate_fails_when_interbeat_reduction_is_not_meaningful(self) -> None:
        result = automated_prescreen(
            self.metrics(-4.0),
            self.metrics(-5.0),
            correlation_delta=0.05,
            attack_delta=0.01,
            spectral_distance_delta=-0.02,
        )

        self.assertFalse(result["interbeat_noise_reduced_from_reference"])
        self.assertFalse(result["passes"])


if __name__ == "__main__":
    unittest.main()
