import unittest

from lottery_sim.recommendations import Candidate, diversify_ranked_candidates


class DiversifiedCandidateTests(unittest.TestCase):
    def test_selection_is_stable_and_keeps_pairwise_distance(self):
        texts = [f"{value:05d}" for value in range(500)]
        pool = [Candidate(index, "probability", text, text, "ranked") for index, text in enumerate(texts, 1)]

        first = diversify_ranked_candidates(pool)
        second = diversify_ranked_candidates(pool)

        self.assertEqual([item.number_text for item in first], [item.number_text for item in second])
        self.assertEqual(len(first), 10)
        for index, left in enumerate(first):
            for right in first[index + 1:]:
                distance = sum(a != b for a, b in zip(left.number_text, right.number_text))
                self.assertGreaterEqual(distance, 2)

    def test_relaxes_distance_when_pool_is_too_small(self):
        pool = [
            Candidate(1, "probability", "00000", "00000", "ranked"),
            Candidate(2, "probability", "00001", "00001", "ranked"),
        ]

        selected = diversify_ranked_candidates(pool, count=2)

        self.assertEqual([item.number_text for item in selected], ["00000", "00001"])


if __name__ == "__main__":
    unittest.main()
