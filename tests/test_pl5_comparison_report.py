import json
import tempfile
import unittest
from pathlib import Path

from lottery_sim.backtest.pl5_metrics import Pl5ModelMetrics, ReturnMetrics
from lottery_sim.dashboard import GameDashboard, _load_pl5_model_comparison, _render_pl5_model_comparison
from lottery_sim.reports.pl5_model_comparison import ModelComparisonResult, save_model_comparison_json


def metric():
    return Pl5ModelMetrics(
        "random", 1, (0.1,) * 5, 0.1, (0.3,) * 5, 0.3, (0.5,) * 5, 0.5,
        0.5, {str(i): 0 for i in range(6)}, 0, 0.0, 2.302585, 0.9,
        {"top10": 0.0, "top50": 0.0, "top100": 0.0},
        {"top10": ReturnMetrics(10, 20, 0, -20, 0, -1)},
    )


class Pl5ComparisonReportTests(unittest.TestCase):
    def test_json_round_trip_and_pl5_only_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model-comparison-pl5.json"
            save_model_comparison_json(ModelComparisonResult("pl5", (metric(),)), path)
            loaded = _load_pl5_model_comparison(Path(directory))
            self.assertEqual(loaded[0]["model_name"], "random")
            html = _render_pl5_model_comparison(GameDashboard("pl5", "排列五", "", "", "", "", (), {}, model_comparison=loaded))
            self.assertIn("Top10净收益率", html)
            self.assertIn("Random", html)
            self.assertEqual(_render_pl5_model_comparison(GameDashboard("ssq", "双色球", "", "", "", "", (), {})), "")
