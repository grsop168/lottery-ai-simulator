import tempfile
import unittest
from pathlib import Path

from lottery_sim.history_db import load_training_records, record_training_record


class TrainingRecordMigrationTests(unittest.TestCase):
    def test_structured_training_metadata_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            record_training_record(path, {
                "created_at": "2026-01-01", "game_code": "pl5", "game_name": "排列五",
                "action": "train", "status": "completed", "summary": "ok",
                "model_name": "lightgbm", "train_start_issue": "1", "train_end_issue": "400",
                "training_draw_count": 400, "parameters_json": '{"num_class":10}',
            })
            record = load_training_records(path)[0]
            self.assertEqual(record["model_name"], "lightgbm")
            self.assertEqual(record["training_draw_count"], 400)
