"""Supervise the local dashboard without invoking any generation commands."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
URL = 'http://127.0.0.1:8765'


def project_id(root):
    return hashlib.sha256(str(Path(root).resolve()).casefold().encode('utf-8')).hexdigest()


def state(root=ROOT):
    try:
        with build_opener(ProxyHandler({})).open(URL + '/health', timeout=3) as response:
            health = json.load(response)
        if response.status == 200 and health.get('ok') is True and health.get('server') == 'fastapi' and health.get('project_id') == project_id(root):
            return 'running'
    except (OSError, ValueError):
        pass
    try:
        with socket.create_connection(('127.0.0.1', 8765), timeout=1):
            return 'occupied'
    except OSError:
        return 'stopped'


def supervise(root=ROOT):
    import msvcrt
    logs = root / 'reports/logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / '.dashboard.lock').open('a+b') as lock:
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Dashboard service lock is already held; refusing unowned service') from exc
        if state(root) != 'stopped':
            raise RuntimeError('PORT_CONFLICT 8765: refusing to monitor an unowned listener')
        # uvicorn blocks in this process; the scheduler owns the actual server.
        from lottery_sim.fastapi_app import serve_fastapi_dashboard
        print(f'START pid={os.getpid()} python={sys.executable}', flush=True)
        serve_fastapi_dashboard(
            reports_dir=root / 'reports/users/admin/latest',
            host='127.0.0.1', port=8765, open_browser=False, repo_root=root,
        )
        raise RuntimeError('Dashboard exited; retry required')



def main():
    try:
        if '--check' in sys.argv:
            current = state()
            print(f'Dashboard: {current}; URL: {URL}', flush=True)
            return 0 if current == 'running' else 1
        return supervise()
    except Exception:
        traceback.print_exc()
        return 1


def run_logged(root=ROOT):
    """Install file streams before uvicorn configures logging (pythonw has none)."""
    logs = root / 'reports/logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f'dashboard-{datetime.now():%Y%m%d-%H%M%S-%f}-{os.getpid()}.log'
    with log.open('a', encoding='utf-8', buffering=1) as output:
        with redirect_stdout(output), redirect_stderr(output):
            code = 1
            print(f'SERVICE START {datetime.now().isoformat()} pid={os.getpid()} python={sys.executable}', flush=True)
            try:
                code = main()
                return code
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                print(f'SERVICE EXIT {datetime.now().isoformat()} code={code}', flush=True)


if __name__ == '__main__':
    # Absolute script invocation works without activation or inherited PYTHONPATH.
    sys.path.insert(0, str(ROOT / 'src'))
    os.chdir(ROOT)
    if '--check' in sys.argv:
        sys.exit(main())
    sys.exit(run_logged())
