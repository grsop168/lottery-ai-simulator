import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from lottery_sim.analysis.house_sensitivity import (
    benjamini_hochberg, deterministic_sensitivity_configs,
    run_house_sensitivity, save_house_sensitivity_json,
)
from lottery_sim.models import Draw5D


def draws(count=36):
    return [Draw5D(str(i + 1), "2026-01-01", tuple((i + p) % 10 for p in range(5))) for i in range(count)]


class HouseSensitivityTests(unittest.TestCase):
    def test_benjamini_hochberg_known_values(self):
        adjusted = benjamini_hochberg((0.01, 0.04, 0.03, 0.002))
        self.assertAlmostEqual(adjusted[0], 0.02)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)
        self.assertAlmostEqual(adjusted[3], 0.008)

    def test_parameter_grid_is_reproducible(self):
        self.assertEqual(deterministic_sensitivity_configs(8), deterministic_sensitivity_configs(8))

    def test_holdout_change_does_not_change_train_ranking_and_json_is_complete(self):
        configs = deterministic_sensitivity_configs(3)
        original = draws()
        first = run_house_sensitivity(original, train_count=24, top_n=2, configs=configs, bootstrap_samples=10)
        changed = list(original)
        for index in range(24, len(changed)):
            changed[index] = Draw5D(str(index + 1), "2026-01-01", (9, 9, 9, 9, 9))
        second = run_house_sensitivity(changed, train_count=24, top_n=2, configs=configs, bootstrap_samples=10)
        self.assertEqual(first.train_results, second.train_results)
        self.assertEqual([pair.config.config_id for pair in first.selected_results], [pair.config.config_id for pair in second.selected_results])
        self.assertEqual(first.train_draws, 24)
        self.assertEqual(first.holdout_draws, 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_house_sensitivity_json(first, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for field in ("parameter_space", "train_results", "selected_results", "robust_parameter_regions", "failed_parameter_regions", "conclusion_level"):
                self.assertIn(field, payload)


if __name__ == "__main__":
    unittest.main()
