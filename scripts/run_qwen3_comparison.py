"""Serial, guarded Qwen3 model-only and auxiliary-data experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gpu_watchdog import SafetyStop, guarded_run, now, save_json
from src.llm.prepare_quality_data import digest, read_rows
from src.llm.prepare_auxiliary_data import DEFAULT_OUT

CONFIG = 'configs/sllm_qlora_v3_qwen3.yaml'
MIX_CONFIG = 'configs/sllm_qlora_v3_qwen3_mix.yaml'
PREVIOUS = ROOT/'outputs/quality_followup_20261006'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = [ROOT/CONFIG, ROOT/MIX_CONFIG, ROOT/'data/processed/sllm_v3/manifest.json',
             ROOT/'data/processed/sllm_v3/train.jsonl', ROOT/'data/processed/sllm_v3/val_eval.jsonl',
             DEFAULT_OUT/'manifest.json', DEFAULT_OUT/'train_auxiliary.jsonl',
             ROOT/'configs/filter_rules.yaml', *sorted((ROOT/'src/llm').glob('*.py')),
             *(ROOT/'scripts'/name for name in ('run_qwen3_comparison.py', 'gpu_watchdog.py', 'probe_gpu_events.ps1', 'profile_qwen3_data.py'))]
    aux = json.loads((DEFAULT_OUT/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in aux['source_sha256'].items():
        source = (ROOT/name).resolve()
        source.relative_to(ROOT)
        if sha(source)!=expected:
            raise ValueError(f'auxiliary source changed: {name}')
        paths.append(source)
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def verify_eval(path, expected_key=None):
    from src.llm.evaluate import comparison_key
    from src.llm.ingredients import IngredientExtractor
    proc = ROOT/'data/processed/sllm_v3'
    rows = read_rows(proc/'val_eval.jsonl')
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    meta = result['summary']
    dataset = json.loads((proc/'manifest.json').read_text(encoding='utf-8'))['dataset_id']
    expected_key = expected_key or comparison_key(proc, IngredientExtractor.load(proc/'ingredient_dict.json'), 'val_eval', False)
    if (meta['dataset_id'] != dataset or meta['split'] != 'val_eval' or meta['comparison_key'] != expected_key
            or meta['evaluated_ids_hash'] != digest(sorted(r['id'] for r in rows))
            or sorted(i['id'] for i in result['items']) != sorted(r['id'] for r in rows)):
        raise ValueError(f'incompatible development result: {path}')
    return meta


def import_previous(destination):
    previous = json.loads((PREVIOUS/'status.json').read_text(encoding='utf-8'))
    if previous['experiment_status'] != 'complete_development_comparison':
        raise ValueError('previous comparison incomplete')
    imported = []
    for stage in ('rule', 'base', 'v2', 'v3'):
        for name, expected_sha in previous['stages'][stage]['artifacts'].items():
            path = Path(name)
            if path.suffix != '.json':
                continue
            if sha(path) != expected_sha:
                raise ValueError(f'previous artifact changed: {path}')
            meta = verify_eval(path)
            target = destination/path.name
            if target.exists() and sha(target) != expected_sha:
                raise ValueError('cannot overwrite different imported result')
            if not target.exists():
                shutil.copy2(path, target)
            imported.append({'path':str(target),'source':str(path),'sha256':expected_sha, 'model':meta['model']})
    return imported


def pipeline_steps(run):
    config = yaml.safe_load((ROOT/CONFIG).read_text(encoding='utf-8'))
    mix_config = yaml.safe_load((ROOT/MIX_CONFIG).read_text(encoding='utf-8'))
    pure_out = ROOT/config['train']['output_dir']
    mix_out = ROOT/mix_config['train']['output_dir']
    smoke_out = ROOT/config['smoke']['output_dir']
    ev = ['-m','src.llm.evaluate','--config',CONFIG,'--output-dir',str(run/'val_eval')]
    return [
        ('profile', ['scripts/profile_qwen3_data.py',str(run/'data_profile.json')], 'cpu', run/'data_profile.json'),
        ('smoke', ['-m','src.llm.train_qlora','--config',CONFIG,'--smoke'], 'gpu_train', smoke_out/'train_info.json'),
        ('base_qwen3', ev+['--base','Qwen/Qwen3-8B'], 'gpu_eval', None),
        ('train_qwen3', ['-m','src.llm.train_qlora','--config',CONFIG], 'gpu_train', pure_out/'train_info.json'),
        ('eval_qwen3', ev+['--adapter',str(pure_out/'adapter')], 'gpu_eval', None),
        ('train_qwen3_mix', ['-m','src.llm.train_qlora','--config',MIX_CONFIG], 'gpu_train', mix_out/'train_info.json'),
        ('eval_qwen3_mix', ev+['--adapter',str(mix_out/'adapter')], 'gpu_eval', None),
        ('compare', ['-m','src.llm.compare','--eval-dir',str(run/'val_eval')], 'cpu', run/'val_eval/comparison.md'),
        ('content_review', ['-m','src.llm.export_model_review','--eval-dir',str(run/'val_eval'),
                            '--out',str(run/'model_content_review.jsonl')], 'cpu',run/'model_content_review.jsonl'),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=ROOT/'outputs/qwen3_comparison_20261008_02')
    ap.add_argument('--resume',action='store_true')
    args = ap.parse_args()
    run = args.run_dir.resolve()
    run.relative_to(ROOT/'outputs')
    run.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (ROOT/'outputs/qwen3_comparison.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        return run_locked(run,args.resume)


def run_locked(run, resume):
    expected = fingerprint()
    path = run/'status.json'
    if path.exists():
        state = json.loads(path.read_text(encoding='utf-8'))
        if not resume or state['fingerprint'] != expected:
            raise RuntimeError('preserve existing run: resume needs same code and data')
        if state['experiment_status'] == 'safety_stopped':
            raise SafetyStop('GPU safety incident requires diagnosis before a new controlled experiment')
        if state.get('experiment_status') == 'running':
            raise SafetyStop('previous process ended without a completion record; inspect host and child processes first')
        state.setdefault('recovery_history',[]).append({'at':now(),'previous_error':state.pop('error',None)})
    else:
        # Never adopt existing adapter/checkpoint files from a different experiment.
        outputs = []
        for cfg_path in (CONFIG,MIX_CONFIG):
            cfg = yaml.safe_load((ROOT/cfg_path).read_text(encoding='utf-8'))
            outputs.extend([ROOT/cfg['train']['output_dir'],ROOT/cfg['smoke']['output_dir']])
        for out in outputs:
            if out.exists():
                raise RuntimeError(f'existing model output requires a new configuration: {out}')
        state = {'started_at':now(),'fingerprint':expected,'stages':{},'experiment_status':'preparing',
                 'clinical_validity':'not_expert_verified','independent_test':'waiting_for_new_human_cases_and_review',
                 'generation_protocol':'shared: nonthinking, greedy first; one retry at temperature .3/top_p .9; max_new_tokens 2400',
                 'training_protocol':'same recommendation corpus; auxiliary arm capped at 5% rows and matches the main-only optimizer-update budget'}
    (run/'val_eval').mkdir(exist_ok=True)
    def save():
        state['updated_at'] = now()
        state['runner_pid'] = os.getpid()
        save_json(path,state)
    env = dict(os.environ,HF_HOME='D:/hf_cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    state['imported_results'] = import_previous(run/'val_eval')
    save()
    for name, command, kind, artifact in pipeline_steps(run):
        old = state['stages'].get(name,{})
        if old.get('status') == 'complete':
            if any(not Path(p).exists() or sha(p)!=h for p,h in old['artifacts'].items()):
                raise RuntimeError('completed artifact changed')
            continue
        stage = {'status':'running','started_at':now(),'command':command,'log':str(run/f'{name}.log')}
        state['stages'][name] = stage
        state.update(current_stage=name,experiment_status='running')
        save()
        try:
            if fingerprint()!=expected:
                raise ValueError('code/config/data changed; preserve this run and start a new experiment')
            before = {p:sha(p) for p in (run/'val_eval').glob('*.json')}
            if name.startswith('train_'):
                out = ROOT/yaml.safe_load((ROOT/command[-1]).read_text(encoding='utf-8'))['train']['output_dir']
                if any(out.glob('checkpoint-*')):
                    command = command+['--resume']
            stage['command'] = command
            save()
            full_command = [sys.executable,'-B',*command]
            if kind.startswith('gpu'):
                code = guarded_run(full_command,Path(stage['log']),run,name,env)
            else:
                with Path(stage['log']).open('a',encoding='utf-8') as log:
                    child = subprocess.Popen(full_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    stage['pid'] = child.pid; save()
                    code = child.wait()
            stage['exit_code'] = code
            if code:
                raise RuntimeError(f'{name} exited {code}; see {stage["log"]}')
            artifacts = [artifact] if artifact else [p for p in (run/'val_eval').glob('*.json') if p not in before or sha(p)!=before[p]]
            if kind=='gpu_eval':
                if len(artifacts)!=1:
                    raise ValueError('one new evaluation expected')
                verify_eval(artifacts[0])
            if kind=='gpu_train':
                info = json.loads(artifact.read_text(encoding='utf-8'))
                dataset = json.loads((ROOT/'data/processed/sllm_v3/manifest.json').read_text(encoding='utf-8'))['dataset_id']
                if info['base_model']!='Qwen/Qwen3-8B' or info['dataset_manifest']['dataset_id']!=dataset:
                    raise ValueError('trained model/data mismatch')
                if name=='train_qwen3_mix':
                    aux = json.loads((DEFAULT_OUT/'manifest.json').read_text(encoding='utf-8'))
                    if info['auxiliary_manifest']!=aux or info['auxiliary_samples']<=0:
                        raise ValueError('mixed adapter missing intended auxiliary data')
                adapter = ROOT/info['adapter_dir']
                if not (adapter/'adapter_model.safetensors').exists():
                    raise ValueError('trained adapter missing')
                artifacts += sorted(p for p in adapter.glob('*') if p.is_file())
            if fingerprint()!=expected:
                raise ValueError('code/config/data changed during stage')
            stage.update(status='complete',finished_at=now(),artifacts={str(p):sha(p) for p in artifacts})
            save()
        except Exception as exc:
            label = 'safety_stopped' if isinstance(exc,SafetyStop) else 'failed'
            stage.update(status=label,finished_at=now())
            state.update(experiment_status=label,error=str(exc))
            save()
            print(str(exc),flush=True)
            return 1
    state.update(experiment_status='complete_development_comparison',current_stage=None,finished_at=now())
    save()
    print('Qwen3 development comparison complete; human efficacy review and independent test still pending',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())
