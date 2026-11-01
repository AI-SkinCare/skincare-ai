"""동일 v3 개발 문항의 순차 GPU 실험. 종료 코드와 산출물을 확인한 단계만 완료로 기록한다."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/sllm_qlora_v3.yaml"
RUN = ROOT / "outputs/quality_followup_20261006"
EVAL = RUN / "val_eval"


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_fingerprint():
    files = [ROOT / CONFIG, ROOT / "data/processed/sllm_v3/manifest.json",
             *sorted((ROOT / "src/llm").glob("*.py")), ROOT / "configs/filter_rules.yaml"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def eval_artifact(paths, fingerprint):
    expected = json.loads((ROOT / "data/processed/sllm_v3/manifest.json").read_text(encoding="utf-8"))["dataset_id"]
    sys.path.insert(0, str(ROOT))
    from src.llm.prepare_quality_data import digest, read_rows
    rows = read_rows(ROOT / "data/processed/sllm_v3/val_eval.jsonl")
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("summary", {})
        if (meta.get("dataset_id") != expected or meta.get("split") != "val_eval"
                or len(data.get("items", [])) != len(rows)
                or meta.get("evaluated_ids_hash") != digest(sorted(r["id"] for r in rows))):
            raise RuntimeError(f"unexpected evaluation artifact: {path}")
    if experiment_fingerprint() != fingerprint:
        raise RuntimeError("code/config changed during experiment")


def steps():
    common = ["-m", "src.llm.evaluate", "--config", CONFIG, "--output-dir", str(EVAL)]
    return [
        ("integrity", ["-m", "src.llm.audit_quality_data"], "audit"),
        ("rule", common + ["--mock"], "eval"),
        ("base", common + ["--base", "Qwen/Qwen2.5-7B-Instruct"], "eval"),
        ("v2", common + ["--adapter", "outputs/rec_sllm_v2/adapter"], "eval"),
        ("train_v3", ["-m", "src.llm.train_qlora", "--config", CONFIG], "train"),
        ("v3", common + ["--adapter", "outputs/rec_sllm_v3/adapter"], "eval"),
        ("compare", ["-m", "src.llm.compare", "--eval-dir", str(EVAL)], "compare"),
        ("content_review", ["-m", "src.llm.export_model_review", "--eval-dir", str(EVAL)], "review"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    # Windows OS lock: 잔여 파일의 존재 여부가 아니라 실제 프로세스 소유권으로 중복 실행 방지.
    import msvcrt
    with (RUN / "pipeline.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        return run_locked(args)


def run_locked(args):
    status_path = RUN / "status.json"
    fingerprint = experiment_fingerprint()
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise RuntimeError("existing run; use --resume")
        if state["fingerprint"] != fingerprint:
            raise RuntimeError("existing run code/config differs; preserve run and choose a new experiment")
    else:
        state = {"started_at": now(), "fingerprint": fingerprint, "stages": {},
                 "experiment_status": "running", "clinical_validity": "not_expert_verified",
                 "independent_test": "waiting_for_new_human_cases_and_review"}
    env = dict(os.environ, HF_HOME=os.environ.get("HF_HOME", "D:/hf_cache"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128")

    def save():
        state["updated_at"] = now()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    save()
    for name, command, kind in steps():
        previous = state["stages"].get(name, {})
        if previous.get("status") == "complete":
            for path, sha in previous.get("artifacts", {}).items():
                if not Path(path).exists() or file_hash(Path(path)) != sha:
                    raise RuntimeError(f"completed artifact changed: {path}")
            continue
        if experiment_fingerprint() != fingerprint:
            state["experiment_status"] = "failed"
            state["error"] = "code/config changed; new run required"
            save()
            return 1
        before = set(EVAL.glob("*.json"))
        if kind == "train" and list((ROOT / "outputs/rec_sllm_v3").glob("checkpoint-*")):
            command = command + ["--resume"]
        stage = {"status": "running", "started_at": now(), "command": command, "log": str(RUN / f"{name}.log")}
        state["stages"][name] = stage
        state["current_stage"] = name
        state["experiment_status"] = "running"
        save()
        with Path(stage["log"]).open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-B", *command], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            stage["pid"] = process.pid
            save()
            return_code = process.wait()
        stage["exit_code"] = return_code
        stage["finished_at"] = now()
        try:
            if return_code != 0:
                raise RuntimeError(f"stage {name} exited {return_code}; see {stage['log']}")
            if kind == "eval":
                artifacts = sorted(set(EVAL.glob("*.json")) - before)
                if len(artifacts) != 1:
                    raise RuntimeError("one new evaluation output expected")
                eval_artifact(artifacts, fingerprint)
            elif kind == "train":
                info_path = ROOT / "outputs/rec_sllm_v3/train_info.json"
                info = json.loads(info_path.read_text(encoding="utf-8"))
                expected = json.loads((ROOT / "data/processed/sllm_v3/manifest.json").read_text(encoding="utf-8"))["dataset_id"]
                if info.get("dataset_manifest", {}).get("dataset_id") != expected:
                    raise RuntimeError("adapter dataset does not match")
                artifacts = [info_path, *sorted((ROOT / "outputs/rec_sllm_v3/adapter").glob("*"))]
                if not any(p.name == "adapter_model.safetensors" for p in artifacts):
                    raise RuntimeError("trained adapter missing")
            elif kind == "compare":
                artifacts = [EVAL / "comparison.md"]
                if not artifacts[0].exists():
                    raise RuntimeError("comparison missing")
            elif kind == "review":
                artifacts = [ROOT / "data/review_20261006/model_content_review.jsonl"]
                if not artifacts[0].exists():
                    raise RuntimeError("content review export missing")
            else:
                artifacts = [ROOT / "data/processed/sllm_v3/audit_report.json"]
            if experiment_fingerprint() != fingerprint:
                raise RuntimeError("code/config changed during stage")
            stage["artifacts"] = {str(p): file_hash(p) for p in artifacts if p.is_file()}
            stage["status"] = "complete"
        except Exception as exc:
            stage["status"] = "failed"
            state["experiment_status"] = "failed"
            state["error"] = str(exc)
            save()
            print(state["error"], flush=True)
            return 1
        save()
    state["experiment_status"] = "complete_development_comparison"
    state["current_stage"] = None
    state["finished_at"] = now()
    save()
    print("development comparison complete; independent human test remains pending", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
