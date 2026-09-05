import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from unittest.mock import patch

import pytest
from lottery_sim import auto_pl5

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path):
    data = tmp_path / 'normalized'
    data.mkdir()
    source = data / 'pl5.csv'
    source.write_text('issue,draw_date,number,source\n2026235,2026-09-02,77967,test\n')
    # Use the real schema but only write copies into pytest's temporary directory.
    from lottery_sim.recommendation_tracking import RecommendationRecord, save_recommendation_records
    original = tmp_path / 'fixture.csv'
    save_recommendation_records([RecommendationRecord(
        game_code='pl5', game_name='PL5', history_until_issue='2026235', target_issue='2026236',
        rank=i, strategy_name='fixture', numbers=f'{i:05d}', reason='test',
        generated_at='2026-09-02 22:00:00', run_id='fixture',
    ) for i in range(1, 11)], original)
    store = tmp_path / 'recommendations'
    store.mkdir()
    return data, source, store, original.read_bytes()


def test_generate_then_skip_preserves_history(tmp_path, capsys):
    data, source, store, content = fixture(tmp_path)
    history = store / 'pl5/2026234.csv'
    history.parent.mkdir()
    history.write_bytes(b'frozen historical sample')
    before = hashlib.sha256(history.read_bytes()).hexdigest()
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        if 'update-pl5' in command:
            source.write_text('issue,draw_date,number,source\n2026235,2026-09-02,77967,test\n')
            return
        stage = Path(command[command.index('-RecommendationDir') + 1])
        (stage / 'pl5').mkdir()
        (stage / 'pl5/2026236.csv').write_bytes(content)
    source.write_text('issue,draw_date,number,source\n2026234,2026-09-01,30129,test\n')
    params = dict(Game='pl5', FastGenerate=True, SkipNormalize=False, DataDir=str(data), RecommendationDir=str(store))
    real_check = auto_pl5.preflight
    with patch.object(auto_pl5, 'preflight', side_effect=lambda *a, **k: real_check(*a, now=datetime(2026, 9, 2, 22), **k)), patch.object(auto_pl5.subprocess, 'run', side_effect=child):
        auto_pl5.run(dict(params))
        target = store / 'pl5/2026236.csv'
        first = target.read_bytes()
        auto_pl5.run(dict(params))
    assert len(calls) == 3
    assert sum('update-pl5' not in c for c in calls) == 1
    assert target.read_bytes() == first == content
    assert hashlib.sha256(history.read_bytes()).hexdigest() == before
    assert 'SKIP_ALREADY_GENERATED target_issue=2026236' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        auto_pl5.publish(target, target, '2026236')


def test_waiting_and_incomplete_fail_closed(tmp_path):
    _, source, store, _ = fixture(tmp_path)
    with pytest.raises(RuntimeError, match='WAITING_FOR_NEW_DRAW'):
        auto_pl5.preflight(source, store, datetime(2026, 9, 3, 21, 30))
    target = store / 'pl5/2026236.csv'
    target.parent.mkdir()
    target.write_text('game_code,target_issue\n')
    with pytest.raises(RuntimeError, match='INCOMPLETE_EXISTING_RECORD'):
        auto_pl5.preflight(source, store)


def test_update_failure_does_not_generate(tmp_path):
    data, _, store, _ = fixture(tmp_path)
    with patch.object(auto_pl5.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'update')) as run:
        with pytest.raises(subprocess.CalledProcessError):
            auto_pl5.run(dict(DataDir=str(data), RecommendationDir=str(store)))
    assert run.call_count == 1
    assert not list(store.glob('pl5/*.csv'))


def test_fast_dry_run_excludes_heavy_commands():
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts/daily.ps1'), '-Game', 'pl5', '-FastGenerate', '-DryRun'], capture_output=True, text=True)
    assert result.returncode == 0
    for forbidden in ['[backtest-pl5]', '[compare-pl5]', '[stability-pl5]', '[backtest-ml-pl5]', '[compare-models-pl5]', '[analyze-house-pl5]', 'sensitivity', 'walk-forward']:
        assert forbidden not in result.stdout


@pytest.mark.parametrize('failure', [False, True])
def test_auto_entry_venv_logs_exit(tmp_path, failure):
    # Relocate only the fixed root for an isolated execution of the real wrapper.
    root = tmp_path / 'project'
    (root / 'scripts').mkdir(parents=True)
    (root / '.venv/Scripts').mkdir(parents=True)
    import shutil
    shutil.copyfile(sys.executable, root / '.venv/Scripts/python.exe')
    shutil.copyfile(ROOT / '.venv/pyvenv.cfg', root / '.venv/pyvenv.cfg')
    (root / 'scripts/daily.ps1').write_text("throw 'injected failure'" if failure else "python -c \"import sys; print('EXECUTED_PYTHON=' + sys.executable)\"; if ($LASTEXITCODE -ne 0) { exit 1 }; Write-Host 'SKIP_ALREADY_GENERATED target_issue=2026236'")
    script = (ROOT / 'scripts/auto_pl5.ps1').read_text(encoding='utf-8-sig').replace(str(ROOT), str(root))
    entry = root / 'scripts/auto_pl5.ps1'
    entry.write_text(script, encoding='utf-8-sig')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(entry)], capture_output=True)
    assert (result.returncode != 0) == failure
    logs = list((root / 'reports/logs').glob('*.log'))
    assert len(logs) == 1
    text = logs[0].read_text(encoding='utf-8-sig')
    assert str(root / '.venv/Scripts/python.exe') in text
    assert 'Finish:' in text and 'Elapsed seconds:' in text
    assert ('FAILURE' if failure else 'SKIP_ALREADY_GENERATED') in text
    if not failure:
        assert 'EXECUTED_PYTHON=' + str(root / '.venv/Scripts/python.exe') in text


def test_concurrent_run_fails_before_update(tmp_path):
    import msvcrt
    data, _, store, _ = fixture(tmp_path)
    with (store / '.pl5-daily.lock').open('w+b') as lock:
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        with patch.object(auto_pl5.subprocess, 'run') as child:
            with pytest.raises(OSError):
                auto_pl5.run(dict(DataDir=str(data), RecommendationDir=str(store)))
            child.assert_not_called()


def test_worker_failure_never_publishes(tmp_path):
    data, _, store, _ = fixture(tmp_path)
    with patch.object(auto_pl5, 'preflight', return_value='2026236'), patch.object(
        auto_pl5.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'worker')
    ):
        with pytest.raises(subprocess.CalledProcessError):
            auto_pl5.run(dict(DataDir=str(data), RecommendationDir=str(store), SkipNormalize=True))
    assert not list(store.glob('pl5/*.csv'))
    assert not list(store.glob('.pl5-stage-*'))


@pytest.mark.parametrize('issue', ['2026239', '2026240'])
def test_publish_inherits_live_acl_and_dashboard_reads(tmp_path, issue):
    import tempfile
    from lottery_sim import dashboard
    _, _, store, content = fixture(tmp_path)
    content = content.replace(b'2026236', issue.encode())
    live = store / 'pl5'
    live.mkdir()
    historical = live / '2026238.csv'
    historical.write_bytes(content.replace(issue.encode(), b'2026238'))
    historical_hash = hashlib.sha256(historical.read_bytes()).hexdigest()
    target = live / f'{issue}.csv'
    with tempfile.TemporaryDirectory(prefix='.pl5-stage-', dir=store) as stage:
        source = Path(stage) / 'pl5' / target.name
        source.parent.mkdir()
        source.write_bytes(content)
        auto_pl5.publish(source, target, issue)
        assert not os.path.samefile(source, target)
        assert target.read_bytes() == content
        with target.open('r', encoding='utf-8') as stream:
            assert stream.readline().startswith('game_code,')
        with target.open('r+b'):
            pass  # Verify write access without changing a byte.
        if os.name == 'nt':
            script = ('$a = Get-Acl -LiteralPath $args[0]; '
                      'if ($a.AreAccessRulesProtected -or @($a.Access | '
                      'Where-Object { -not $_.IsInherited }).Count) { exit 1 }; '
                      'Get-Content -LiteralPath $args[0] -ErrorAction Stop | Out-Null')
            subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                            '& { ' + script + ' }', str(target)], check=True)
            acl_command = ['powershell.exe', '-NoProfile', '-Command',
                           '& { (Get-Acl -LiteralPath $args[0]).Sddl }', str(target)]
            acl_before = subprocess.check_output(acl_command)
        with patch.object(dashboard, '_resolve_recommendation_dir', return_value=store), patch.object(
            dashboard, 'sync_recommendation_records'
        ):
            records = dashboard._load_dashboard_recommendation_records(tmp_path / 'reports', 'pl5')
        assert len([r for r in records if r.target_issue == issue]) == 10
        with pytest.raises(FileExistsError):
            auto_pl5.publish(source, target, issue)
        assert target.read_bytes() == content
        if os.name == 'nt':
            assert subprocess.check_output(acl_command) == acl_before
    assert hashlib.sha256(historical.read_bytes()).hexdigest() == historical_hash
    assert not list(live.glob('.pl5-publish-*'))


@pytest.mark.skipif(os.name != 'nt', reason='Windows ACL reset')
def test_acl_failure_does_not_publish(tmp_path):
    _, _, store, content = fixture(tmp_path)
    source = tmp_path / 'staged.csv'
    source.write_bytes(content)
    target = store / 'pl5/2026236.csv'
    with patch.object(auto_pl5.subprocess, 'check_call', side_effect=subprocess.CalledProcessError(1, 'icacls')):
        with pytest.raises(subprocess.CalledProcessError):
            auto_pl5.publish(source, target, '2026236')
    assert not target.exists()
    assert not list(target.parent.glob('.pl5-publish-*'))
