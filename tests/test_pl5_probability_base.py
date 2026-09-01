import unittest

from lottery_sim.ml.base import ProbabilityPrediction


class ProbabilityPredictionTests(unittest.TestCase):
    def test_normalizes_five_probability_rows(self):
        prediction = ProbabilityPrediction("test", tuple((1.0,) * 10 for _ in range(5)), 20)
        self.assertEqual(len(prediction.probabilities), 5)
        for row in prediction.probabilities:
            self.assertEqual(len(row), 10)
            self.assertAlmostEqual(sum(row), 1.0)

    def test_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            ProbabilityPrediction("test", ((0.1,) * 10,), 1)
