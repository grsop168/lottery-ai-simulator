import unittest

import numpy as np

from lottery_sim.betting import BehavioralBettor, BetDistributionEstimator, UniformBettor
from lottery_sim.ml.house_min_payout import HouseMinPayoutModel
from lottery_sim.models import Draw5D
from lottery_sim.backtest.pl5_walk_forward import run_pl5_walk_forward


def history(count=20):
    return [Draw5D(str(i + 1), "2026-01-01", tuple((i + p) % 10 for p in range(5))) for i in range(count)]


class HouseMinPayoutTests(unittest.TestCase):
    def test_full_and_marginal_probabilities_are_valid(self):
        model = HouseMinPayoutModel(house_temperature=1_000_000).fit(history())
        prediction = model.predict_proba(history())
        self.assertAlmostEqual(float(model.last_house_probabilities.sum()), 1.0)
        self.assertEqual(model.last_house_probabilities.shape, (100_000,))
        for row in prediction.probabilities:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_lower_temperature_prefers_lower_expected_payout(self):
        estimator = BetDistributionEstimator(BehavioralBettor(), simulated_ticket_count=100_000)
        low = HouseMinPayoutModel(estimator, house_temperature=100_000).fit(history())
        low.predict_proba(history())
        high = HouseMinPayoutModel(estimator, house_temperature=10_000_000).fit(history())
        high.predict_proba(history())
        cold = int(np.argmin(low.last_expected_payout))
        hot = int(np.argmax(low.last_expected_payout))
        self.assertGreater(low.last_house_probabilities[cold] / low.last_house_probabilities[hot], high.last_house_probabilities[cold] / high.last_house_probabilities[hot])

    def test_uniform_market_cannot_create_house_structure(self):
        estimator = BetDistributionEstimator(UniformBettor())
        model = HouseMinPayoutModel(estimator, house_temperature=1).fit(history())
        prediction = model.predict_proba(history())
        self.assertTrue(np.allclose(model.last_house_probabilities, 1e-5))
        self.assertTrue(all(np.allclose(row, 0.1) for row in prediction.probabilities))

    def test_infinite_temperature_is_uniform(self):
        model = HouseMinPayoutModel(house_temperature=float("inf")).fit(history())
        model.predict_proba(history())
        self.assertTrue(np.allclose(model.last_house_probabilities, 1e-5))

    def test_target_and_future_changes_do_not_change_prediction(self):
        draws = history(30)
        target = 20
        first = HouseMinPayoutModel().fit(draws[:target]).predict_proba(draws[:target])
        draws[target] = Draw5D("21", "2026-01-01", (9, 9, 9, 9, 9))
        draws[target + 1] = Draw5D("22", "2026-01-01", (8, 8, 8, 8, 8))
        second = HouseMinPayoutModel().fit(draws[:target]).predict_proba(draws[:target])
        self.assertEqual(first.probabilities, second.probabilities)

    def test_house_uses_common_walk_forward_without_future_data(self):
        draws = history(30)
        result = run_pl5_walk_forward(
            draws, lambda: HouseMinPayoutModel(house_temperature=1_000_000),
            min_train_draws=20, backtest_draws=5, retrain_every=2,
            candidate_count=10, top_k_values=(10,),
        )
        self.assertEqual(result.metrics.model_name, "house")
        self.assertEqual(result.metrics.backtest_draws, 5)
        self.assertTrue(all(int(cutoff) < int(target) for cutoff, target in zip(result.training_cutoffs, result.target_issues)))


if __name__ == "__main__":
    unittest.main()
