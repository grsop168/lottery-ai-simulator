"""Supervise the local dashboard without invoking any generation commands."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
URL = 'http://127.0.0.1:8765'


def project_id(root):
    return hashlib.sha256(str(Path(root).resolve()).casefold().encode('utf-8')).hexdigest()


def state(root=ROOT):
    try:
        with build_opener(ProxyHandler({})).open(URL + '/health', timeout=3) as response:
            health = json.load(response)
        if health.get('ok') is True and health.get('server') == 'fastapi' and health.get('project_id') == project_id(root):
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
        except OSError:
            if state(root) == 'running':
                print('ALREADY_RUNNING http://127.0.0.1:8765', flush=True)
                return 0
            raise RuntimeError('Dashboard supervisor is starting or unhealthy')
        current = state(root)
        if current == 'occupied':
            raise RuntimeError('PORT_CONFLICT 8765: not a healthy dashboard from this project')
        if current == 'running':
            print('ALREADY_RUNNING monitoring existing dashboard', flush=True)
            while state(root) == 'running':
                time.sleep(15)
            raise RuntimeError('Existing dashboard exited or became unavailable; retry required')
        python = root / '.venv/Scripts/python.exe'
        command = [str(python), '-u', '-m', 'lottery_sim.cli', 'dashboard', '--server', 'fastapi',
                   '--reports', 'reports/users/admin/latest', '--host', '127.0.0.1', '--port', '8765']
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), PYTHONIOENCODING='utf-8')
        print('START ' + subprocess.list2cmdline(command), flush=True)
        child = subprocess.Popen(command, cwd=root, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for _ in range(60):
                if child.poll() is not None:
                    raise RuntimeError(f'Dashboard startup failed exit={child.returncode}')
                if state(root) == 'running':
                    print('READY http://127.0.0.1:8765', flush=True)
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Dashboard startup health check timed out')
            code = child.wait()
            # Even a clean unexpected exit must restart this persistent service.
            raise RuntimeError(f'Dashboard exited code={code}; retry required')
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)


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


if __name__ == '__main__':
    sys.exit(main())
