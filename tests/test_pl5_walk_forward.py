import unittest

from lottery_sim.backtest.pl5_walk_forward import run_pl5_walk_forward
from lottery_sim.ml.random_model import RandomPl5Model
from lottery_sim.models import Draw5D


class Pl5WalkForwardTests(unittest.TestCase):
    def test_training_cutoff_is_before_target_and_retraining_interval_is_honored(self):
        draws = [Draw5D(str(index + 1), "2026-01-01", tuple((index + p) % 10 for p in range(5))) for index in range(30)]
        result = run_pl5_walk_forward(
            draws,
            lambda: RandomPl5Model(seed=9),
            min_train_draws=20,
            backtest_draws=6,
            retrain_every=3,
            candidate_count=10,
            top_k_values=(10,),
        )
        self.assertEqual(result.target_issues, ("25", "26", "27", "28", "29", "30"))
        self.assertEqual(result.retrained, (True, False, False, True, False, False))
        for cutoff, target in zip(result.training_cutoffs, result.target_issues):
            self.assertLess(int(cutoff), int(target))

    def test_future_change_does_not_affect_earlier_random_result(self):
        draws = [Draw5D(str(index + 1), "2026-01-01", (index % 10,) * 5) for index in range(25)]
        first = run_pl5_walk_forward(draws, lambda: RandomPl5Model(3), 20, 5, 2, 10, (10,))
        draws[-1] = Draw5D("25", "2026-01-01", (9,) * 5)
        second = run_pl5_walk_forward(draws, lambda: RandomPl5Model(3), 20, 5, 2, 10, (10,))
        self.assertEqual(first.training_cutoffs[:-1], second.training_cutoffs[:-1])
