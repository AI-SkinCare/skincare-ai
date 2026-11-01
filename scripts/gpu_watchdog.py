"""Monitor only a child owned by this run; no driver, clock or registry changes."""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now().astimezone().isoformat()


def save_json(path, value):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        handle.flush()


def run_probe(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', timeout=6,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f'exit {result.returncode}')
    return result.stdout.strip()


def gpu_probe():
    values = run_probe(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits', '--id=0']).split(',')
    if len(values) != 6:
        raise ValueError('unexpected GPU telemetry')
    metrics = {}
    for key, value in zip(('total_mb', 'used_mb', 'free_mb', 'temperature_c', 'utilization_pct', 'power_w'), values):
        try:
            metrics[key] = float(value.strip())
        except ValueError:
            metrics[key] = None  # Unsupported fields must not become zero/overheat alarms.
    return metrics


def event_probe(after_record=0):
    # Invoke our trusted source as a command; no machine/user execution-policy changes.
    script = (ROOT / 'scripts/probe_gpu_events.ps1').read_text(encoding='utf-8')
    command = '& {\n' + script + '\n} -AfterRecord ' + str(int(after_record))
    powershell = shutil.which('pwsh.exe') or shutil.which('powershell.exe')
    if not powershell:
        raise RuntimeError('PowerShell event collector unavailable')
    return json.loads(run_probe([powershell, '-NoProfile', '-NonInteractive', '-Command', command]))


@dataclass
class GuardPolicy:
    failures: int = 0
    event_failures: int = 0
    hot_since: float | None = None
    temperature_limit: float = 85
    hot_seconds: float = 30
    stall_seconds: float = 900

    def observe(self, tick, metrics, events, log_idle, event_ok=True):
        if events:
            return 'new_gpu_or_hardware_error'
        self.event_failures = 0 if event_ok else self.event_failures + 1
        self.failures = 0 if metrics is not None else self.failures + 1
        if self.event_failures >= 3:
            return 'event_monitor_unavailable'
        if self.failures >= 3:
            return 'gpu_probe_failed_repeatedly'
        temp = metrics.get('temperature_c') if metrics else None
        if temp is not None and temp >= self.temperature_limit:
            if self.hot_since is None:
                self.hot_since = tick
            if tick - self.hot_since >= self.hot_seconds:
                return 'sustained_high_temperature'
        else:
            self.hot_since = None
        if log_idle >= self.stall_seconds:
            return 'progress_log_stalled'
        return None


def track_owned_children(process):
    """Windows venv launchers have a second Python process that also owns CUDA."""
    if not hasattr(process, '_owned_root'):
        process._owned_descendants = {}
        if process.poll() is not None:
            return
        try:
            process._owned_root = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            return
    root = getattr(process, '_owned_root', None)
    if root is not None and root.is_running():
        try:
            for child in root.children(recursive=True):
                process._owned_descendants[(child.pid,child.create_time())] = child
        except psutil.NoSuchProcess:
            pass


def stop_owned_child(process):
    """Stop verified descendants and the Popen handle, never all Python processes."""
    track_owned_children(process)
    children = list(getattr(process,'_owned_descendants',{}).values())
    for child in reversed(children):
        try:
            if child.is_running():  # psutil checks process creation identity before kill.
                child.kill()
        except psutil.NoSuchProcess:
            pass
    if children:
        psutil.wait_procs(children,timeout=5)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


class SafetyStop(RuntimeError):
    pass


def guarded_run(command, log_path, run_dir, stage, env):
    run_dir, log_path = Path(run_dir), Path(log_path)
    events = event_probe()  # No GPU child is launched if event monitoring cannot initialise.
    metrics = gpu_probe()
    if metrics['free_mb'] is None or metrics['free_mb'] < 10000:
        raise SafetyStop(f'GPU free memory below preflight minimum: {metrics}')
    if metrics['temperature_c'] is None or metrics['temperature_c'] >= 75:
        raise SafetyStop(f'GPU temperature unavailable or above preflight minimum: {metrics}')
    record = events['latest_record']
    policy = GuardPolicy()
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n[guarded start {now()}] {command!r}\n')
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        track_owned_children(process)
        started = last_tick = last_progress = time.monotonic()
        last_size = log_path.stat().st_size
        snapshot = {'stage': stage, 'child_pid': process.pid, 'started_at': now(), 'status': 'running'}
        save_json(run_dir / 'watchdog.json', snapshot)
        try:
            while process.poll() is None:
                track_owned_children(process)
                tick = time.monotonic()
                reason = 'monitor_gap_over_120_seconds' if tick - last_tick > 120 else None
                if (run_dir / 'STOP').exists():
                    reason = 'user_requested_stop'
                last_tick = tick
                size = log_path.stat().st_size
                if size != last_size:
                    last_progress, last_size = tick, size
                gpu_error = event_error = None
                try:
                    metrics = gpu_probe()
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    metrics, gpu_error = None, str(exc)
                new_events = []
                try:
                    result = event_probe(record)
                    if result['latest_record'] < record:
                        reason = 'system_event_log_reset'
                    record = result['latest_record']
                    new_events = result['events']
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    event_error = str(exc)
                reason = reason or policy.observe(tick, metrics, new_events, tick - last_progress, event_error is None)
                snapshot.update(updated_at=now(), metrics=metrics, gpu_error=gpu_error, event_error=event_error,
                                owned_descendant_pids=[p.pid for p in getattr(process,'_owned_descendants',{}).values() if p.is_running()],
                                last_system_record=record, log_idle_seconds=round(tick-last_progress, 1),
                                elapsed_seconds=round(tick-started, 1))
                append_json(run_dir / 'gpu_telemetry.jsonl', snapshot)
                if reason:
                    snapshot.update(status='safety_stopped', reason=reason, driver_events=new_events)
                    append_json(run_dir / 'safety_events.jsonl', snapshot)
                    save_json(run_dir / 'watchdog.json', snapshot)
                    raise SafetyStop(f'{stage}: {reason}; manual diagnosis required, no automatic restart')
                save_json(run_dir / 'watchdog.json', snapshot)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            # Catch errors recorded just as the workload exits successfully.
            final_events = event_probe(record)['events']
            if final_events:
                snapshot.update(status='safety_stopped', reason='new_gpu_or_hardware_error', driver_events=final_events)
                append_json(run_dir / 'safety_events.jsonl', snapshot)
                save_json(run_dir / 'watchdog.json', snapshot)
                raise SafetyStop(f'{stage}: driver error at process exit')
            snapshot.update(status='finished', exit_code=process.returncode, updated_at=now())
            save_json(run_dir / 'watchdog.json', snapshot)
            return process.returncode
        finally:
            stop_owned_child(process)
