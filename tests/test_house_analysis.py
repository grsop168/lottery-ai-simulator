import unittest

from lottery_sim.analysis.house_analysis import analyze_house_history
from lottery_sim.models import Draw5D


class HouseAnalysisTests(unittest.TestCase):
    def test_three_control_mechanisms_and_statistics(self):
        draws = [Draw5D(str(i + 1), "2026-01-01", tuple((i + p) % 10 for p in range(5))) for i in range(20)]
        result = analyze_house_history(draws, simulated_ticket_count=10_000)
        self.assertEqual(result.draw_count, 20)
        self.assertEqual(len(result.mechanisms), 3)
        self.assertEqual(len(result.draw_diagnostics), 20)
        self.assertEqual(len(result.latest_low_payout_numbers), 20)
        self.assertAlmostEqual(result.mechanisms[0].brier_score, 0.9)


if __name__ == "__main__":
    unittest.main()
