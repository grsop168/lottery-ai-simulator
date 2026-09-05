import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from lottery_sim import dashboard_service as service
from lottery_sim.fastapi_app import create_fastapi_app


def test_health_identifies_project_and_page_reads_new_reports(tmp_path):
    reports = tmp_path / 'reports/users/admin/latest'
    reports.mkdir(parents=True)
    with patch.dict('os.environ', {'LOTTERY_ADMIN_PASSWORD': 'test-password'}):
        client = TestClient(create_fastapi_app(reports, tmp_path))
    assert client.get('/health').json()['project_id'] == service.project_id(tmp_path)
    client.post('/login', data={'username': 'admin', 'password': 'test-password'})
    report = reports / 'recommend-pl5.txt'
    with patch('lottery_sim.fastapi_app.start_dashboard_job') as generate:
        report.write_text('FIRST_REPORT')
        assert 'FIRST_REPORT' in client.get('/').text
        report.write_text('UPDATED_REPORT')
        page = client.get('/').text
        assert 'UPDATED_REPORT' in page and 'FIRST_REPORT' not in page
        generate.assert_not_called()


def test_health_rejects_another_project(tmp_path):
    opener = Mock()
    opener.open.return_value = io.BytesIO(json.dumps({'ok': True, 'server': 'fastapi', 'project_id': 'other'}).encode())
    with patch.object(service, 'build_opener', return_value=opener), patch.object(service.socket, 'create_connection') as connection:
        connection.return_value.__enter__ = Mock()
        connection.return_value.__exit__ = Mock()
        assert service.state(tmp_path) == 'occupied'


def test_conflicting_port_never_launches(tmp_path):
    with patch.object(service, 'state', return_value='occupied'), patch.object(service.subprocess, 'Popen') as launch:
        with pytest.raises(RuntimeError, match='PORT_CONFLICT'):
            service.supervise(tmp_path)
        launch.assert_not_called()


def test_existing_dashboard_is_monitored_without_second_process(tmp_path):
    with patch.object(service, 'state', side_effect=['running', 'running', 'stopped']), patch.object(service.time, 'sleep'), patch.object(service.subprocess, 'Popen') as launch:
        with pytest.raises(RuntimeError, match='retry required'):
            service.supervise(tmp_path)
        launch.assert_not_called()


def test_duplicate_supervisor_returns_success(tmp_path):
    import msvcrt
    logs = tmp_path / 'reports/logs'
    logs.mkdir(parents=True)
    with (logs / '.dashboard.lock').open('w+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        with patch.object(service, 'state', return_value='running'), patch.object(service.subprocess, 'Popen') as launch:
            assert service.supervise(tmp_path) == 0
            launch.assert_not_called()


def test_child_exit_requests_task_retry_and_uses_venv(tmp_path):
    child = Mock()
    child.poll.side_effect = [None, 7]
    child.wait.return_value = 7
    with patch.object(service, 'state', side_effect=['stopped', 'running']), patch.object(service.subprocess, 'Popen', return_value=child) as launch:
        with pytest.raises(RuntimeError, match='exited code=7'):
            service.supervise(tmp_path)
    command = launch.call_args.args[0]
    assert command[0] == str(tmp_path / '.venv/Scripts/python.exe')
    assert command[command.index('--reports') + 1] == 'reports/users/admin/latest'
    assert 'dashboard' in command
    assert launch.call_args.kwargs['cwd'] == tmp_path
    assert launch.call_args.kwargs['creationflags'] == service.subprocess.CREATE_NO_WINDOW


def test_check_failure_returns_nonzero():
    with patch.object(service, 'supervise', side_effect=RuntimeError('injected')), patch.object(service.sys, 'argv', ['dashboard_service']):
        assert service.main() == 1
