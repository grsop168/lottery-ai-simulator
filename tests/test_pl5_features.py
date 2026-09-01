import unittest

from lottery_sim.ml.pl5_features import build_pl5_prefix_features, build_pl5_training_rows
from lottery_sim.models import Draw5D


def draw(issue, numbers):
    return Draw5D(str(issue), "2026-01-01", tuple(numbers))


class Pl5FeatureTests(unittest.TestCase):
    def setUp(self):
        self.draws = [draw(index + 1, ((index + offset) % 10 for offset in range(5))) for index in range(12)]

    def test_one_multiclass_row_per_target_issue(self):
        rows = build_pl5_training_rows(self.draws, position=0, min_history=5)
        self.assertEqual(len(rows.features), 7)
        self.assertEqual(len(rows.targets), 7)
        self.assertEqual(rows.target_issues[0], "6")

    def test_target_and_future_changes_do_not_change_target_features(self):
        target_index = 8
        before = build_pl5_prefix_features(self.draws[:target_index], position=2)
        changed = list(self.draws)
        changed[target_index] = draw(9, (9, 9, 9, 9, 9))
        changed[target_index + 1] = draw(10, (8, 8, 8, 8, 8))
        after = build_pl5_prefix_features(changed[:target_index], position=2)
        self.assertEqual(before, after)

    def test_incremental_training_features_match_direct_prefix_builder(self):
        rows = build_pl5_training_rows(self.draws, position=1, min_history=5)
        for offset, features in enumerate(rows.features, start=5):
            self.assertEqual(features, build_pl5_prefix_features(self.draws[:offset], position=1))


if __name__ == "__main__":
    unittest.main()
