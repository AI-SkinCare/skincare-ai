import json

from scripts import run_qwen3_comparison as runner
from scripts.gpu_watchdog import SafetyStop


def test_safety_incident_is_recorded_and_next_stage_never_runs(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    for name in (runner.CONFIG,runner.MIX_CONFIG):
        path=tmp_path/name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({'train':{'output_dir':'outputs/new'},'smoke':{'output_dir':'outputs/smoke'}}))
    run=tmp_path/'outputs/run'
    run.mkdir(parents=True)
    monkeypatch.setattr(runner,'fingerprint',lambda:{'code':'same'})
    monkeypatch.setattr(runner,'import_previous',lambda path:[])
    monkeypatch.setattr(runner,'pipeline_steps',lambda path:[
        ('smoke',['unused'],'gpu_train',None),('must_not_run',['unused'],'gpu_train',None)])
    called=[]
    def stop(command,log_path,run_dir,stage,env):
        called.append(stage)
        raise SafetyStop('fixture GPU driver error')
    monkeypatch.setattr(runner,'guarded_run',stop)
    assert runner.run_locked(run,False)==1
    state=json.loads((run/'status.json').read_text(encoding='utf-8'))
    assert called==['smoke']
    assert state['experiment_status']=='safety_stopped'
    assert 'must_not_run' not in state['stages']
