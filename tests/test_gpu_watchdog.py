import os
import subprocess
import sys
import json
import time
import psutil

import pytest

from scripts.gpu_watchdog import GuardPolicy, stop_owned_child
from scripts import gpu_watchdog


def test_new_error_stops_even_while_gpu_and_progress_appear_healthy():
    policy = GuardPolicy()
    assert policy.observe(10, {'temperature_c':40}, [{'event_id':153}], 0) == 'new_gpu_or_hardware_error'


def test_high_temperature_requires_sustained_readings_and_resets():
    policy = GuardPolicy()
    assert policy.observe(0, {'temperature_c':86}, [], 0) is None
    assert policy.observe(15, {'temperature_c':84}, [], 0) is None
    assert policy.observe(30, {'temperature_c':86}, [], 0) is None
    assert policy.observe(59, {'temperature_c':86}, [], 0) is None
    assert policy.observe(60, {'temperature_c':86}, [], 0) == 'sustained_high_temperature'


def test_transient_probe_failure_recovers_but_repeated_failures_stop():
    policy = GuardPolicy()
    assert policy.observe(0,None,[],0) is None
    assert policy.observe(15,{'temperature_c':None},[],0) is None
    assert policy.observe(30,None,[],0) is None
    assert policy.observe(45,None,[],0) is None
    assert policy.observe(60,None,[],0) == 'gpu_probe_failed_repeatedly'


def test_event_monitor_failure_is_not_silently_healthy():
    policy = GuardPolicy()
    for tick in (0,15):
        assert policy.observe(tick,{'temperature_c':40},[],0,event_ok=False) is None
    assert policy.observe(30,{'temperature_c':40},[],0,event_ok=False) == 'event_monitor_unavailable'


def test_long_evaluation_is_allowed_but_fifteen_minute_stall_stops():
    policy = GuardPolicy()
    assert policy.observe(600,{'temperature_c':40},[],600) is None
    assert policy.observe(900,{'temperature_c':40},[],900) == 'progress_log_stalled'


def test_only_the_owned_process_is_terminated(tmp_path):
    flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    unrelated = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],creationflags=flags)
    marker = tmp_path/'actual_pid.txt'
    owned = subprocess.Popen([sys.executable,'-c',
        'import os,sys,time;from pathlib import Path;Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)',
        str(marker)],creationflags=flags)
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:
            time.sleep(.02)
        actual = psutil.Process(int(marker.read_text()))
        stop_owned_child(owned)
        assert owned.poll() is not None
        assert not actual.is_running()
        assert unrelated.poll() is None
        stop_owned_child(owned)  # Terminated handles must not target a reused PID.
    finally:
        stop_owned_child(owned)
        stop_owned_child(unrelated)


def test_driver_incident_is_persisted_and_stops_real_guarded_child(tmp_path,monkeypatch):
    probes = iter([{'latest_record':10,'events':[]},
                   {'latest_record':11,'events':[{'event_id':153,'record_id':11}]}])
    monkeypatch.setattr(gpu_watchdog,'event_probe',lambda record=0:next(probes))
    monkeypatch.setattr(gpu_watchdog,'gpu_probe',lambda:{'free_mb':14000,'temperature_c':40})
    with pytest.raises(gpu_watchdog.SafetyStop,match='new_gpu_or_hardware_error'):
        gpu_watchdog.guarded_run([sys.executable,'-c','import time;time.sleep(30)'],tmp_path/'child.log',
                                tmp_path,'fixture',dict(os.environ))
    incident = json.loads((tmp_path/'watchdog.json').read_text(encoding='utf-8'))
    assert incident['status']=='safety_stopped'
    assert incident['driver_events'][0]['event_id']==153
    assert (tmp_path/'safety_events.jsonl').exists()
