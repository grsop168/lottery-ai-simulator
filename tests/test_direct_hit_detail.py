import unittest

from lottery_sim.dashboard import _recommendation_hit_detail
from lottery_sim.games.fucai3d import Fucai3DGame
from lottery_sim.games.pl5 import PL5Game
from lottery_sim.recommendation_tracking import RecommendationRecord


def checked_record(game_code: str, numbers: str, draw_numbers: str) -> RecommendationRecord:
    return RecommendationRecord(
        game_code=game_code,
        game_name=game_code,
        history_until_issue="1",
        target_issue="2",
        rank=1,
        strategy_name="test",
        numbers=numbers,
        reason="test",
        status="checked",
        draw_numbers=draw_numbers,
    )


class DirectHitDetailTests(unittest.TestCase):
    def test_3d_repeated_digits_are_matched_by_position(self):
        summary, hits, misses = _recommendation_hit_detail(checked_record("3d", "996", "695"))

        self.assertIn("1/3", summary)
        self.assertEqual(hits, "9")
        self.assertEqual(misses, "9 6")

    def test_pl5_repeated_or_elsewhere_digits_do_not_count(self):
        summary, hits, misses = _recommendation_hit_detail(checked_record("pl5", "72191", "30129"))

        self.assertIn("1/5", summary)
        self.assertEqual(hits, "1")
        self.assertEqual(misses, "7 2 9 1")

    def test_prize_and_payout_still_require_full_positional_equality(self):
        game3d = Fucai3DGame()
        game5 = PL5Game()

        self.assertEqual(game3d.payout(game3d.validate_pick((6, 9, 5)), game3d.validate_pick((9, 9, 6))), 0)
        self.assertEqual(game5.payout(game5.validate_pick((3, 0, 1, 2, 9)), game5.validate_pick((7, 2, 1, 9, 1))), 0)


if __name__ == "__main__":
    unittest.main()
