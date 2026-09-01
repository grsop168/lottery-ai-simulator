import unittest

import numpy as np

from lottery_sim.betting import BehavioralBettor, BetDistributionEstimator, UniformBettor
from lottery_sim.models import Draw5D


class BettingModelTests(unittest.TestCase):
    def test_uniform_contains_all_numbers_and_sums_to_one(self):
        distribution = BetDistributionEstimator(UniformBettor()).estimate(())
        self.assertEqual(distribution.probabilities.shape, (100_000,))
        self.assertAlmostEqual(float(distribution.probabilities.sum()), 1.0)
        self.assertTrue(np.allclose(distribution.probabilities, 1e-5))

    def test_behavioral_is_normalized_nonuniform_and_stable(self):
        history = [Draw5D("1", "2026-01-01", (6, 8, 9, 1, 2))]
        estimator = BetDistributionEstimator(BehavioralBettor(), random_seed=7)
        first = estimator.estimate(history)
        second = estimator.estimate(history)
        self.assertEqual(first.probabilities.shape, (100_000,))
        self.assertAlmostEqual(float(first.probabilities.sum()), 1.0)
        self.assertGreater(float(first.probabilities.max()), float(first.probabilities.min()))
        np.testing.assert_array_equal(first.probabilities, second.probabilities)


if __name__ == "__main__":
    unittest.main()
