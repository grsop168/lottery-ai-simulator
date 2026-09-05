"""Transactional orchestration only; prediction algorithms remain in daily.ps1."""
import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, time

from lottery_sim.data_sources.pl5_17500 import load_pl5_draws_csv
from lottery_sim.issue_calendar import next_issue_from_latest_draw
from lottery_sim.recommendation_tracking import available_recommendation_draws, load_recommendation_records

ROOT = Path(__file__).resolve().parents[2]


def complete(path, target, count=10):
    records = load_recommendation_records(path)
    ranks = {r.rank for r in records}
    return set(range(1, count + 1)).issubset(ranks) and all(
        r.game_code == 'pl5' and r.target_issue == target and r.numbers
        and r.strategy_name and r.generated_at and r.run_id and r.numbers.isdigit()
        and len(r.numbers) == 5 and r.rank > 0 and int(r.history_until_issue) < int(target)
        for r in records
    )


def preflight(csv, store, now=None, extra_paths=(), count=10):
    now = now or datetime.now()
    draws = sorted(load_pl5_draws_csv(csv), key=lambda d: int(d.issue))
    if not draws:
        raise RuntimeError('WAITING_FOR_NEW_DRAW no draws')
    latest = draws[-1]
    target = next_issue_from_latest_draw('pl5', latest.issue, latest.draw_date).issue
    print(f'latest_draw_issue={latest.issue} target_issue={target}', flush=True)
    paths = [store / 'pl5' / f'{target}.csv', *extra_paths]
    for path in paths:
        if path.exists():
            if not complete(path, target, count):
                raise RuntimeError(f'INCOMPLETE_EXISTING_RECORD refusing overwrite: {path}')
            print(f'SKIP_ALREADY_GENERATED target_issue={target}', flush=True)
            return None
    ready = available_recommendation_draws(draws, as_of=now)
    if not ready or ready[-1].issue != latest.issue or (
        latest.draw_date < now.date().isoformat() and now.time() >= time(21, 30)
    ):
        raise RuntimeError('WAITING_FOR_NEW_DRAW source stale or draw not ready')
    return target


def publish(source, destination, target, count=10):
    if not complete(source, target, count):
        raise RuntimeError('INCOMPLETE_GENERATION')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Hard links retain the source DACL. Never link the private staging inode
    # directly into live: tempfile directories can have admin-only Windows ACLs.
    # Create a fresh inode in live and copy bytes only (not staging metadata).
    pending = destination.parent / f'.pl5-publish-{uuid.uuid4().hex}.tmp'
    try:
        with pending.open('xb') as output, source.open('rb') as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        if os.name == 'nt':
            # Reset only this unpublished file to its live parent's inherited
            # DACL, even when the producer runs with an elevated token.
            subprocess.check_call(['icacls.exe', str(pending), '/reset'],
                                  stdout=subprocess.DEVNULL)
        # Same-volume, atomic, no-overwrite publication after ACL repair succeeds.
        os.link(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def run(params):
    store = Path(params.get('RecommendationDir', 'data/recommendations')).resolve()
    store.mkdir(parents=True, exist_ok=True)
    import msvcrt
    shared = store.is_relative_to(ROOT / 'data')
    lock_root = ROOT / 'data' if shared else store
    with (lock_root / '.pl5-daily.lock').open('a+b') as lock:
        lock.seek(0)
        if lock.read(1) == b'':
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        csv = Path(params.get('DataDir', 'data/normalized')) / 'pl5.csv'
        before = load_pl5_draws_csv(csv)[-1].issue if csv.exists() else 'none'
        if not params.get('SkipNormalize'):
            try:
                subprocess.run([sys.executable, '-m', 'lottery_sim.cli', 'update-pl5', '--csv', str(csv)], check=True)
            except Exception:
                print('UPDATE failure', flush=True)
                raise
            after = max(load_pl5_draws_csv(csv), key=lambda d: int(d.issue)).issue
            print(f'UPDATE success previous_issue={before} latest_issue={after} advanced={int(after) > int(before) if before != "none" else True}', flush=True)
        else:
            print('UPDATE skipped SkipNormalize', flush=True)
        extra = []
        if shared:
            latest = max(load_pl5_draws_csv(csv), key=lambda d: int(d.issue))
            target = next_issue_from_latest_draw('pl5', latest.issue, latest.draw_date).issue
            extra = list((ROOT / 'data/users').glob(f'*/recommendations/pl5/{target}.csv'))
            extra.append(ROOT / 'data/recommendations/pl5' / f'{target}.csv')
        count = int(params.get('Count', 10))
        target = preflight(csv, store, extra_paths=extra, count=count)
        if target is None:
            return
        with tempfile.TemporaryDirectory(prefix='.pl5-stage-', dir=store) as stage:
            params.update(RecommendationDir=stage, SkipNormalize=True)
            command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'scripts/daily.ps1')]
            for key, value in params.items():
                if isinstance(value, bool):
                    if value:
                        command.append('-' + key)
                else:
                    command.extend(['-' + key, str(value)])
            env = dict(os.environ, LOTTERY_PL5_WORKER='1')
            print(f'GENERATE target_issue={target}', flush=True)
            subprocess.run(command, env=env, check=True)
            publish(Path(stage) / 'pl5' / f'{target}.csv', store / 'pl5' / f'{target}.csv', target, count)
            print(f'GENERATE success target_issue={target}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('params')
    args = parser.parse_args()
    try:
        run(json.loads(args.params))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
