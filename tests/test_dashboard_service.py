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
    opener.open.return_value.status = 200
    with patch.object(service, 'build_opener', return_value=opener), patch.object(service.socket, 'create_connection') as connection:
        connection.return_value.__enter__ = Mock()
        connection.return_value.__exit__ = Mock()
        assert service.state(tmp_path) == 'occupied'


@pytest.mark.parametrize('current', ['occupied', 'running'])
def test_existing_listener_never_adopted(tmp_path, current):
    with patch.object(service, 'state', return_value=current), patch('lottery_sim.fastapi_app.serve_fastapi_dashboard') as serve:
        with pytest.raises(RuntimeError, match='PORT_CONFLICT'):
            service.supervise(tmp_path)
        serve.assert_not_called()


def test_duplicate_service_fails_even_when_healthy(tmp_path):
    import msvcrt
    logs = tmp_path / 'reports/logs'
    logs.mkdir(parents=True)
    with (logs / '.dashboard.lock').open('w+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        with patch.object(service, 'state', return_value='running'):
            with pytest.raises(RuntimeError, match='lock is already held'):
                service.supervise(tmp_path)


def test_server_runs_inline_and_exit_requests_retry(tmp_path):
    with patch.object(service, 'state', return_value='stopped'), patch('lottery_sim.fastapi_app.serve_fastapi_dashboard') as serve:
        with pytest.raises(RuntimeError, match='exited; retry required'):
            service.supervise(tmp_path)
    serve.assert_called_once_with(reports_dir=tmp_path / 'reports/users/admin/latest',
                                  host='127.0.0.1', port=8765, open_browser=False, repo_root=tmp_path)


@pytest.mark.parametrize('status, expected', [(200, 'running'), (201, 'occupied')])
def test_health_requires_http_200(tmp_path, status, expected):
    response = io.BytesIO(json.dumps({'ok': True, 'server': 'fastapi', 'project_id': service.project_id(tmp_path)}).encode())
    response.status = status
    opener = Mock()
    opener.open.return_value = response
    with patch.object(service, 'build_opener', return_value=opener), patch.object(service.socket, 'create_connection'):
        assert service.state(tmp_path) == expected


def test_check_failure_returns_nonzero():
    with patch.object(service, 'supervise', side_effect=RuntimeError('injected')), patch.object(service.sys, 'argv', ['dashboard_service']):
        assert service.main() == 1


def test_pythonw_missing_streams_logs_failure_and_exit(tmp_path):
    with patch.object(service.sys, 'stdout', None), patch.object(service.sys, 'stderr', None), \
            patch.object(service.sys, 'argv', ['dashboard_service']), \
            patch.object(service, 'supervise', side_effect=RuntimeError('pythonw failure')):
        assert service.run_logged(tmp_path) == 1
        assert service.sys.stdout is None and service.sys.stderr is None
    log, = (tmp_path / 'reports/logs').glob('dashboard-*.log')
    content = log.read_text(encoding='utf-8')
    assert 'SERVICE START' in content
    assert 'RuntimeError: pythonw failure' in content
    assert 'SERVICE EXIT' in content and 'code=1' in content
