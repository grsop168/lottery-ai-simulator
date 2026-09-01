import math
import unittest

from lottery_sim.backtest.pl5_metrics import EvaluatedPrediction, calculate_pl5_metrics
from lottery_sim.ml.base import ProbabilityPrediction


class Pl5MetricsTests(unittest.TestCase):
    def test_accuracy_coverage_loss_brier_and_returns(self):
        row = (0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0)
        prediction = ProbabilityPrediction("test", (row,) * 5, 10)
        evaluated = [EvaluatedPrediction(prediction, (0, 1, 2, 3, 4), ("01234",) + tuple(f"9{i:04d}" for i in range(99)))]
        result = calculate_pl5_metrics("test", evaluated, (10, 50, 100))
        self.assertAlmostEqual(result.mean_position_accuracy, 0.2)
        self.assertAlmostEqual(result.mean_top3_coverage, 0.6)
        self.assertAlmostEqual(result.mean_top5_coverage, 1.0)
        expected_log_loss = -sum(math.log(max(row[digit], 1e-15)) for digit in range(5)) / 5
        self.assertAlmostEqual(result.log_loss, expected_log_loss)
        self.assertGreater(result.brier_score, 0)
        self.assertEqual(result.candidate_exact_coverage["top10"], 1.0)
        top10 = result.returns["top10"]
        self.assertEqual(top10.total_cost, 20)
        self.assertEqual(top10.total_payout, 100000)
        self.assertEqual(top10.profit, 99980)
        self.assertEqual(top10.return_rate, 5000)
        self.assertEqual(top10.profit_roi, 4999)
