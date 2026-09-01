import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lottery_sim import dashboard


class DashboardMultiUserTests(unittest.TestCase):
    def setUp(self):
        with dashboard._DASHBOARD_JOBS_LOCK:
            dashboard._DASHBOARD_JOBS.clear()

    def test_dashboard_command_includes_user_directory_overrides(self):
        command = dashboard._dashboard_command(
            "daily",
            "ssq",
            {},
            data_dir=Path("data/normalized"),
            report_dir=Path("reports/users/alice/latest"),
            recommendation_dir=Path("data/users/alice/recommendations"),
            model_dir=Path("data/users/alice/models"),
        )

        self.assertIsNotNone(command)
        self.assertIn("-DataDir", command)
        self.assertIn("data/normalized", command)
        self.assertIn("-ReportDir", command)
        self.assertIn("reports/users/alice/latest", command)
        self.assertIn("-RecommendationDir", command)
        self.assertIn("data/users/alice/recommendations", command)
        self.assertIn("-ModelDir", command)
        self.assertIn("data/users/alice/models", command)
        self.assertIn("-FastGenerate", command)

    def test_start_dashboard_job_blocks_second_running_job_for_same_user(self):
        thread = Mock()
        with patch("lottery_sim.dashboard.threading.Thread", return_value=thread):
            first = dashboard.start_dashboard_job("daily", Path.cwd(), game_code="ssq", user_key="alice")
            second = dashboard.start_dashboard_job("daily", Path.cwd(), game_code="ssq", user_key="alice")

        self.assertEqual(first.status, "running")
        self.assertEqual(second.status, "failed")
        self.assertIn("already has a running job", "\n".join(second.output_lines))
        thread.start.assert_called_once()

    def test_user_report_path_resolves_private_and_shared_data_dirs(self):
        reports_path = Path("app-root") / "reports" / "users" / "alice" / "latest"

        self.assertEqual(
            dashboard._resolve_dashboard_data_dir(reports_path),
            Path("app-root") / "data" / "users" / "alice",
        )
        self.assertEqual(
            dashboard._resolve_recommendation_dir(reports_path),
            Path("app-root") / "data" / "users" / "alice" / "recommendations",
        )
        self.assertEqual(
            dashboard._resolve_history_data_dir(reports_path),
            Path("app-root") / "data" / "normalized",
        )


if __name__ == "__main__":
    unittest.main()
