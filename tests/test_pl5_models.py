import importlib.util
import unittest

from lottery_sim.ml.logistic_pl5 import LogisticPl5Model
from lottery_sim.ml.markov import MarkovPl5Model
from lottery_sim.ml.random_model import RandomPl5Model
from lottery_sim.models import Draw5D


def draws(count=80):
    return [Draw5D(str(index + 1), "2026-01-01", tuple((index * (p + 1) + p) % 10 for p in range(5))) for index in range(count)]


class Pl5ModelTests(unittest.TestCase):
    def assert_matrix(self, prediction):
        self.assertEqual(len(prediction.probabilities), 5)
        for row in prediction.probabilities:
            self.assertEqual(len(row), 10)
            self.assertAlmostEqual(sum(row), 1.0)

    def test_random_probability_is_uniform_and_candidates_are_issue_stable(self):
        model = RandomPl5Model(seed=7).fit(draws())
        prediction = model.predict_proba(draws())
        self.assertEqual(prediction.probabilities[0], (0.1,) * 10)
        self.assertEqual(model.sample_candidates("100", 10), model.sample_candidates("100", 10))
        self.assertNotEqual(model.sample_candidates("100", 10), model.sample_candidates("101", 10))

    def test_markov_probability_rows_sum_to_one(self):
        model = MarkovPl5Model(min_history=20).fit(draws())
        self.assert_matrix(model.predict_proba(draws()))

    def test_markov_training_does_not_read_future(self):
        history = draws(50)
        first = MarkovPl5Model().fit(history[:40]).predict_proba(history[:40])
        history[45] = Draw5D("46", "2026-01-01", (9, 9, 9, 9, 9))
        second = MarkovPl5Model().fit(history[:40]).predict_proba(history[:40])
        self.assertEqual(first.probabilities, second.probabilities)

    def test_logistic_adapter_outputs_probability_matrix(self):
        history = draws()
        model = LogisticPl5Model(min_history=10, epochs=1).fit(history)
        self.assert_matrix(model.predict_proba(history))


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "lightgbm is not installed")
class LightGbmPl5Tests(unittest.TestCase):
    def test_train_and_predict_five_by_ten(self):
        from lottery_sim.ml.lightgbm_model import LightGbmPl5Model
        history = draws(100)
        model = LightGbmPl5Model(min_history=10, n_estimators=5).fit(history)
        prediction = model.predict_proba(history)
        self.assertEqual((5, 10), (len(prediction.probabilities), len(prediction.probabilities[0])))
        for row in prediction.probabilities:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_target_and_future_changes_do_not_change_target_prediction(self):
        from lottery_sim.ml.lightgbm_model import LightGbmPl5Model
        history = draws(100)
        target_index = 80
        first = LightGbmPl5Model(min_history=10, n_estimators=5).fit(history[:target_index]).predict_proba(history[:target_index])
        changed = list(history)
        changed[target_index] = Draw5D("81", "2026-01-01", (9, 9, 9, 9, 9))
        changed[target_index + 1] = Draw5D("82", "2026-01-01", (8, 8, 8, 8, 8))
        second = LightGbmPl5Model(min_history=10, n_estimators=5).fit(changed[:target_index]).predict_proba(changed[:target_index])
        self.assertEqual(first.probabilities, second.probabilities)


if __name__ == "__main__":
    unittest.main()
