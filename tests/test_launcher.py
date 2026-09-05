import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import launcher


class LauncherTests(unittest.TestCase):
    def test_application_root_uses_executable_parent_when_frozen(self):
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", r"C:\app\lottery-ai-simulator.exe"),
        ):
            self.assertEqual(launcher.application_root(), Path(r"C:\app"))

    def test_ensure_runtime_dirs_creates_data_and_report_dirs(self):
        root = Path.cwd() / ".test-tmp" / "launcher-runtime"
        if root.exists():
            for child in sorted(root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()
            root.rmdir()

        launcher.ensure_runtime_dirs(root)

        self.assertTrue((root / "data" / "normalized").is_dir())
        self.assertTrue((root / "data" / "users").is_dir())
        self.assertTrue((root / "reports" / "latest").is_dir())
        self.assertTrue((root / "reports" / "users").is_dir())

    def test_run_dashboard_sets_packaged_cli_exe_and_starts_fastapi(self):
        root = Path.cwd()
        serve = Mock()
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", str(root / "lottery-ai-simulator.exe")),
            patch.dict(os.environ, {}, clear=True),
            patch("launcher.dashboard_health_ok", return_value=False),
            patch("lottery_sim.fastapi_app.serve_fastapi_dashboard", serve),
        ):
            launcher.run_dashboard(root=root, host="127.0.0.1", port=8765, open_browser=True)
            self.assertEqual(os.environ["LOTTERY_CLI_EXE"], str(root / "lottery-ai-simulator.exe"))

        serve.assert_called_once_with(
            reports_dir=root / "reports" / "latest",
            host="127.0.0.1",
            port=8765,
            open_browser=True,
            repo_root=root,
        )

    def test_run_dashboard_opens_existing_service_instead_of_starting_second_server(self):
        root = Path.cwd()
        serve = Mock()
        with (
            patch("launcher.dashboard_health_ok", return_value=True),
            patch("launcher.webbrowser.open") as open_browser,
            patch("lottery_sim.fastapi_app.serve_fastapi_dashboard", serve),
        ):
            with redirect_stdout(io.StringIO()):
                launcher.run_dashboard(root=root, host="127.0.0.1", port=8765, open_browser=True)

        serve.assert_not_called()
        open_browser.assert_called_once_with("http://127.0.0.1:8765")

    def test_frozen_dashboard_startup_failure_writes_log_and_waits_for_enter(self):
        root = Path.cwd() / ".test-tmp" / "launcher-failure"
        root.mkdir(parents=True, exist_ok=True)
        log_path = root / "startup-error.log"
        if log_path.exists():
            log_path.unlink()

        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", str(root / "lottery-ai-simulator.exe")),
            patch("launcher.run_dashboard", side_effect=SystemExit(1)),
            patch("builtins.input") as wait_for_enter,
        ):
            with redirect_stdout(io.StringIO()):
                result = launcher.main([])

        self.assertEqual(result, 1)
        self.assertTrue(log_path.exists())
        self.assertIn("SystemExit: 1", log_path.read_text(encoding="utf-8"))
        wait_for_enter.assert_called_once()

    def test_run_cli_delegates_to_cli_main(self):
        cli_main = Mock(return_value=0)
        with patch("lottery_sim.cli.main", cli_main):
            result = launcher.run_cli(["dashboard", "--help"])

        self.assertEqual(result, 0)
        cli_main.assert_called_once_with(["dashboard", "--help"])


if __name__ == "__main__":
    unittest.main()
