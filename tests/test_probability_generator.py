import math
import unittest

from lottery_sim.candidate.probability_generator import generate_probability_candidates
from lottery_sim.ml.base import ProbabilityPrediction


class ProbabilityGeneratorTests(unittest.TestCase):
    def test_candidates_are_sorted_by_joint_log_probability(self):
        rows = []
        for position in range(5):
            row = [0.01] * 10
            row[position] = 0.8
            rows.append(tuple(row))
        candidates = generate_probability_candidates(ProbabilityPrediction("test", tuple(rows), 10), 3)
        self.assertEqual(candidates[0].number_text, "01234")
        self.assertGreaterEqual(candidates[0].log_probability, candidates[1].log_probability)
        self.assertAlmostEqual(math.log(candidates[0].probability), candidates[0].log_probability)
