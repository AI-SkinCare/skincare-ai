"""scripts/run_pipeline.py — 밤새 무인으로 도는 실행기라 재시작·건너뛰기·실패 처리를 검증한다."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
rp = importlib.util.module_from_spec(spec)
sys.modules["run_pipeline"] = rp   # dataclass가 모듈을 sys.modules에서 찾으므로 먼저 등록
spec.loader.exec_module(rp)


# ── LogWriter ──
def test_log_collapses_progress_and_crlf(tmp_path):
    p = tmp_path / "l.log"
    w = rp.LogWriter(p, progress_every=0)
    w.feed("시작\r\n")                                   # CRLF
    w.feed("  0%|    | 0/10\r 50%|##  | 5/10\r")         # 진행 중 (줄바꿈 없음)
    w.feed("100%|####| 10/10\n끝\n")
    w.close()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "시작"
    assert lines[1].startswith("[진행]") and "5/10" in lines[1]
    assert lines[2] == "100%|####| 10/10" and lines[3] == "끝"


def test_log_throttles_progress(tmp_path):
    p = tmp_path / "l.log"
    w = rp.LogWriter(p, progress_every=3600)
    for i in range(1000):
        w.feed(f"\r{i}/1000")
    w.close()
    body = p.read_text(encoding="utf-8").splitlines()
    assert sum(l.startswith("[진행]") for l in body) == 1   # 1000번 갱신 → 한 줄
    assert body[-1] == "999/1000"                           # 종료 시 마지막 상태는 남김


# ── Runner.sh ──
def _runner(tmp_path):
    return rp.Runner(tmp_path, rp.LogWriter(tmp_path / "p.log"), env=rp.child_env(), echo=False)


def test_sh_streams_output_and_returns_code(tmp_path):
    r = _runner(tmp_path)
    assert r.sh([sys.executable, "-c", "print('한글 출력'); import sys; sys.exit(3)"]) == 3
    r.log.close()
    assert "한글 출력" in (tmp_path / "p.log").read_text(encoding="utf-8")


def test_sh_missing_binary(tmp_path):
    assert _runner(tmp_path).sh(["definitely-not-a-real-binary-xyz"]) == 127


# ── run_steps: 건너뛰기 / 실패 / 재개 ──
def _steps(calls, fail_at=None):
    def mk(name):
        def run(r):
            calls.append(name)
            return 1 if name == fail_at else 0
        return rp.Step(name, name, run)
    return [mk(n) for n in ("a", "b", "c")]


def test_failure_stops_and_rerun_resumes_from_failed_step(tmp_path):
    sp, r = tmp_path / "status.json", _runner(tmp_path)
    calls = []
    assert rp.run_steps(_steps(calls, fail_at="b"), r, sp) == 1
    assert calls == ["a", "b"]                                    # c는 실행 안 됨
    st = json.loads(sp.read_text(encoding="utf-8"))
    assert st["steps"]["a"]["state"] == "done" and st["steps"]["b"]["state"] == "failed"

    calls.clear()
    assert rp.run_steps(_steps(calls), r, sp) == 0
    assert calls == ["b", "c"]                                    # a는 건너뜀


def test_interrupted_running_step_is_rerun(tmp_path):
    sp, r = tmp_path / "status.json", _runner(tmp_path)
    rp.save_status(sp, {"steps": {"a": {"state": "done"}, "b": {"state": "running"}}})  # 창을 닫은 상황
    calls = []
    rp.run_steps(_steps(calls), r, sp)
    assert calls == ["b", "c"]


@pytest.mark.parametrize("kw,expected", [
    ({"force": True}, ["a", "b", "c"]),
    ({"from_": "b"}, ["b", "c"]),
    ({"only": "c"}, ["c"]),
])
def test_rerun_options(tmp_path, kw, expected):
    sp, r = tmp_path / "status.json", _runner(tmp_path)
    rp.run_steps(_steps([]), r, sp)                               # 전부 완료 상태로 만든 뒤
    calls = []
    rp.run_steps(_steps(calls), r, sp, **kw)
    assert calls == expected


def test_unknown_step_name(tmp_path):
    assert rp.run_steps(_steps([]), _runner(tmp_path), tmp_path / "s.json", only="nope") == 2


def test_step_exception_is_recorded_as_failure(tmp_path):
    sp, r = tmp_path / "status.json", _runner(tmp_path)

    def boom(_r):
        raise RuntimeError("x")
    assert rp.run_steps([rp.Step("a", "a", boom)], r, sp) == 1
    assert json.loads(sp.read_text(encoding="utf-8"))["steps"]["a"]["state"] == "failed"


def test_real_step_plan_order():
    assert [s.name for s in rp.make_steps(1, 60)] == [
        "venv", "install", "check_env", "smoke", "eval_base", "train", "eval_finetuned", "compare"]


# ── _torch_status: capture()가 stdout+stderr를 이어 붙여도 표시 줄만 정확히 찾아야 한다 ──
def test_torch_status_ignores_trailing_stderr_noise(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    # 실제 장애 재현: import torch가 성공해도 뒤에 경고(stderr)가 더 붙어 "마지막 줄"이 값이 아니게 됨
    noisy = "__STATUS__ 2.11.0+cu128 12.8\nUserWarning: something unrelated with an = sign\n"
    monkeypatch.setattr(r, "capture", lambda cmd: (0, noisy))
    ok, why = rp._torch_status(r)
    assert ok and "2.11.0+cu128" in why and "12.8" in why


def test_torch_status_not_installed(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    monkeypatch.setattr(r, "capture", lambda cmd: (1, "ModuleNotFoundError: No module named 'torch'\n"))
    ok, why = rp._torch_status(r)
    assert not ok and "미설치" in why


def test_torch_status_cpu_only(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    monkeypatch.setattr(r, "capture", lambda cmd: (0, "__STATUS__ 2.5.0 cpu\n"))
    ok, why = rp._torch_status(r)
    assert not ok and "CPU" in why


def test_torch_status_old_cuda_build(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    monkeypatch.setattr(r, "capture", lambda cmd: (0, "__STATUS__ 2.1.0+cu121 12.1\n"))
    ok, why = rp._torch_status(r)
    assert not ok and "12.1" in why


def test_torch_status_unparseable_output_does_not_raise(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    # 표시 줄 자체가 없는 예상 밖 출력 → 예외 없이 실패로 보고해야 한다 (과거엔 여기서 ValueError로 죽었음)
    monkeypatch.setattr(r, "capture", lambda cmd: (0, "some warning with an = sign\n"))
    ok, why = rp._torch_status(r)
    assert not ok


def test_base_model_ignores_trailing_stderr_noise(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    monkeypatch.setattr(r, "capture", lambda cmd: (0, "__STATUS__ Qwen/Qwen2.5-7B-Instruct\nDeprecationWarning: x\n"))
    assert rp._base_model(r) == "Qwen/Qwen2.5-7B-Instruct"


def test_base_model_falls_back_on_failure(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    monkeypatch.setattr(r, "capture", lambda cmd: (1, "error\n"))
    assert rp._base_model(r) == "Qwen/Qwen2.5-7B-Instruct"


def test_train_resumes_only_from_incomplete_run(tmp_path, monkeypatch):
    r = _runner(tmp_path)
    cmds = []
    monkeypatch.setattr(r, "py", lambda *a: cmds.append(a) or 0)
    train = next(s for s in rp.make_steps(1, 60) if s.name == "train")
    out = tmp_path / rp.TRAIN_DIR

    train.run(r)                                                  # 체크포인트 없음 → 새 학습
    (out / "checkpoint-100").mkdir(parents=True)
    train.run(r)                                                  # 중단된 학습 → 이어서
    (out / "train_info.json").write_text("{}", encoding="utf-8")
    train.run(r)                                                  # 이미 끝난 학습을 --force로 다시 → 처음부터
    assert ["--resume" in c for c in cmds] == [False, True, False]
