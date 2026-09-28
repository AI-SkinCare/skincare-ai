"""학습·평가 전체 파이프라인 실행기 (조장 PC, Windows + RTX 5070 Ti).

실행:  레포 루트의 run_pipeline.bat 더블클릭
또는:  py scripts\\run_pipeline.py [--epochs 1] [--eval-n 60] [--from STEP] [--only STEP] [--force]
상태:  py scripts\\run_pipeline.py --status

단계  venv → install → check_env → smoke → eval_base → train → eval_finetuned → compare

- 끝난 단계는 outputs/run_logs/status.json에 기록되고, 다시 실행하면 건너뛴다.
  창을 닫았거나 PC가 꺼졌으면 다시 실행만 하면 된다. 학습은 마지막 체크포인트(100스텝마다)부터 이어진다.
- 전체 로그는 outputs/run_logs/pipeline.log (UTF-8). 진행률 표시줄은 1분에 한 줄로 요약해 남긴다.
  → Claude가 이 두 파일을 읽어 진행 상황을 확인하고 오류를 진단할 수 있다.
- 실행 중에는 Windows 절전 모드 진입을 막는다 (화면 꺼짐은 허용).
- 가상환경을 만들기 전에 실행되므로 표준 라이브러리만 사용한다.
"""
from __future__ import annotations

import argparse
import codecs
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "run_logs"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"   # RTX 50 시리즈는 CUDA 12.8 이상 빌드 필수
MIN_CUDA = (12, 8)
TRAIN_DIR = "outputs/rec_sllm_v1"


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def venv_python(root: Path) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# ───────────────────────── 로그 ─────────────────────────
class LogWriter:
    """콘솔에는 원본 그대로 보여주고, 로그 파일에는 정리해서 남긴다.

    tqdm 진행률은 '\\r'로 같은 줄을 계속 덮어써서, 그대로 저장하면 8시간 학습 로그가 수백 MB가 된다.
    그래서 완성된 줄(\\n)은 마지막 '\\r' 조각만, 진행 중인 줄은 progress_every초마다 한 번만 기록한다.
    """

    MAX_BUF = 100_000

    def __init__(self, path: Path, progress_every: float = 60.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.buf = ""
        self.every = progress_every
        self.last_progress = float("-inf")

    def line(self, text: str) -> None:
        self.f.write(text.rstrip("\r\n") + "\n")
        self.f.flush()

    def feed(self, text: str) -> None:
        self.buf += text
        while "\n" in self.buf:
            ln, self.buf = self.buf.split("\n", 1)
            seg = ln.rstrip("\r").split("\r")[-1]          # CRLF와 진행률 덮어쓰기 모두 처리
            self.line(seg)
        if "\r" in self.buf:
            segs = [s for s in self.buf.split("\r") if s.strip()]
            if segs and time.monotonic() - self.last_progress >= self.every:
                self.line("[진행] " + segs[-1].strip())
                self.last_progress = time.monotonic()
            self.buf = self.buf.rsplit("\r", 1)[-1]
        if len(self.buf) > self.MAX_BUF:
            self.line(self.buf)
            self.buf = ""

    def flush_rest(self) -> None:
        if self.buf.strip():
            self.line(self.buf.split("\r")[-1])
        self.buf = ""

    def close(self) -> None:
        self.flush_rest()
        self.f.close()


# ───────────────────────── 실행 ─────────────────────────
class Runner:
    def __init__(self, root: Path, log: LogWriter, env: dict[str, str] | None = None, echo: bool = True):
        self.root = root
        self.log = log
        self.env = env if env is not None else child_env()
        self.vpy = venv_python(root)
        self.echo = echo

    def note(self, msg: str) -> None:
        if self.echo:
            print(msg, flush=True)
        self.log.line(msg)

    def sh(self, cmd: list[str]) -> int:
        """명령을 실행하며 출력을 콘솔과 로그로 실시간 전달한다. 종료 코드를 반환."""
        self.note(f"$ {subprocess.list2cmdline(cmd)}")
        try:
            p = subprocess.Popen(cmd, cwd=self.root, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as e:
            self.note(f"실행 실패: {e}")
            return 127
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        assert p.stdout is not None
        while True:
            chunk = p.stdout.read1(65536) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
            if not chunk:
                break
            text = dec.decode(chunk)
            if self.echo:
                sys.stdout.write(text)
                sys.stdout.flush()
            self.log.feed(text)
        rest = dec.decode(b"", final=True)
        if rest:
            self.log.feed(rest)
        rc = p.wait()
        self.log.flush_rest()
        return rc

    def py(self, *args: str) -> int:
        return self.sh([str(self.vpy), *args])

    def capture(self, cmd: list[str]) -> tuple[int, str]:
        try:
            r = subprocess.run(cmd, cwd=self.root, env=self.env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, str(e)
        return r.returncode, (r.stdout or "") + (r.stderr or "")


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1", TOKENIZERS_PARALLELISM="false")
    # 16GB급 GPU에서 학습 중 파편화로 인한 CUDA OOM을 줄인다 (긴 시퀀스가 섞여 있을 때 특히 도움됨).
    # 주의: expandable_segments는 Windows CUDA 드라이버에서 지원되지 않아 조용히 무시된다
    # (torch가 "expandable_segments not supported on this platform" 경고만 찍고 아무 효과가 없음).
    # 대신 Windows에서도 동작하는 max_split_size_mb를 쓴다. 이미 설정돼 있으면 덮어쓰지 않는다.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # 7B 모델(약 15GB)이 C드라이브를 채우지 않도록
    if not env.get("HF_HOME") and os.name == "nt" and Path("D:/").exists():
        env["HF_HOME"] = "D:\\hf_cache"
    if env.get("HF_HOME"):
        Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    return env


def keep_awake(on: bool) -> None:
    """실행 중 절전 진입 방지 (Windows 전용). 프로세스가 끝나면 자동으로 해제된다."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:  # noqa: BLE001
        pass


# ───────────────────────── 단계 ─────────────────────────
@dataclass
class Step:
    name: str
    title: str
    run: Callable[[Runner], int]
    hint: str = ""


_MARK = "__STATUS__"  # stdout/stderr가 섞여도 이 표시가 붙은 줄만 찾아 파싱한다


def _marked_line(out: str, mark: str) -> str | None:
    """capture()는 stdout+stderr를 이어 붙이므로, 경고 등 다른 줄이 마지막에 올 수 있다.
    그래서 '마지막 줄'이 아니라 고유 표시가 붙은 줄을 명시적으로 찾는다."""
    for ln in reversed(out.splitlines()):
        if ln.startswith(mark):
            return ln[len(mark):].strip()
    return None


def _torch_status(r: Runner) -> tuple[bool, str]:
    rc, out = r.capture([str(r.vpy), "-c",
                         f"import torch; print('{_MARK}', torch.__version__, torch.version.cuda or 'cpu')"])
    line = _marked_line(out, _MARK) if rc == 0 else None
    if line is None:
        return False, "torch 미설치" if rc != 0 else f"torch 상태 확인 실패 (출력: {out.strip()[-200:]!r})"
    parts = line.split()
    if len(parts) < 2:
        return False, f"torch 상태 확인 실패 (출력: {line!r})"
    ver, cuda = parts[0], parts[1]
    if cuda == "cpu":
        return False, f"CPU 전용 torch {ver}"
    try:
        cuda_tuple = tuple(int(x) for x in cuda.split(".")[:2])
    except ValueError:
        return False, f"torch {ver}의 CUDA 버전을 알 수 없음 ({cuda!r})"
    if cuda_tuple < MIN_CUDA:
        return False, f"torch {ver}가 CUDA {cuda}용 (RTX 50 시리즈는 12.8 이상 필요)"
    return True, f"torch {ver} / CUDA {cuda}"


def step_venv(r: Runner) -> int:
    if r.vpy.exists():
        r.note(f"가상환경 있음: {r.vpy}")
        return 0
    v = sys.version_info[:2]
    if v < (3, 9):
        r.note(f"Python {v[0]}.{v[1]}는 너무 낮습니다. Python 3.11을 설치하세요 (python.org).")
        return 1
    if v >= (3, 14):
        r.note(f"경고: Python {v[0]}.{v[1]}는 일부 패키지 휠이 없을 수 있습니다. 설치 단계에서 실패하면 Python 3.11로 다시 시도하세요.")
    return r.sh([sys.executable, "-m", "venv", str(r.root / ".venv")])


def step_install(r: Runner) -> int:
    if rc := r.py("-m", "pip", "install", "-U", "pip"):
        return rc
    ok, why = _torch_status(r)
    r.note(f"torch 상태: {why}")
    if not ok:
        if "미설치" not in why:
            r.py("-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio")
        if rc := r.py("-m", "pip", "install", "torch", "--index-url", TORCH_INDEX):
            return rc
        ok, why = _torch_status(r)
        r.note(f"torch 상태: {why}")
        if not ok:
            return 1
    return r.py("-m", "pip", "install", "-r", "requirements-train.txt", "-r", "requirements-dev.txt")


def step_check_env(r: Runner) -> int:
    return r.py("scripts/check_env.py")


def step_smoke(r: Runner) -> int:
    return r.py("-m", "src.llm.train_qlora", "--smoke")


def _base_model(r: Runner) -> str:
    rc, out = r.capture([str(r.vpy), "-c",
                         f"import yaml; print('{_MARK}', yaml.safe_load(open('configs/sllm_qlora.yaml', encoding='utf-8'))['model']['base'])"])
    line = _marked_line(out, _MARK) if rc == 0 else None
    return line if line else "Qwen/Qwen2.5-7B-Instruct"


def _gpu_others(r: Runner) -> None:
    """학습 전에 GPU를 쓰는 다른 프로세스가 있으면 알려준다 (OOM 예방)."""
    if not shutil.which("nvidia-smi"):
        return
    rc, out = r.capture(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    procs = [l for l in out.splitlines() if l.strip()] if rc == 0 else []
    if procs:
        r.note("[주의] GPU를 사용 중인 다른 프로그램이 있습니다. 학습 중 OOM이 나면 이것들을 종료하세요:")
        for l in procs:
            r.note("   " + l)


def make_steps(epochs: float, eval_n: int) -> list[Step]:
    def eval_base(r: Runner) -> int:
        return r.py("-m", "src.llm.evaluate", "--base", _base_model(r), "--n", str(eval_n))

    def train(r: Runner) -> int:
        out = r.root / TRAIN_DIR
        done_before = (out / "train_info.json").exists()
        resume = any(out.glob("checkpoint-*")) and not done_before
        _gpu_others(r)
        # CUDA 오류로 죽으면 마지막 체크포인트에서 자동 재시도한다 (진행이 없으면 2회 연속 실패 시 중단).
        rc, stalled, last_ckpt = 1, 0, -1
        for attempt in range(1, 31):
            if resume:
                r.note(f"체크포인트 발견 → 이어서 학습합니다. (시도 {attempt})")
            rc = r.py("-m", "src.llm.train_qlora", "--epochs", str(epochs), *(["--resume"] if resume else []))
            if rc == 0:
                return 0
            ckpts = [int(p.name.split("-")[-1]) for p in out.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
            cur = max(ckpts, default=0)
            stalled = stalled + 1 if cur <= last_ckpt else 0
            last_ckpt = cur
            if not ckpts or stalled >= 2:
                break
            resume = True
            r.note(f"[재시도] 학습 실패(코드 {rc}), 체크포인트 step {cur}부터 20초 뒤 재개")
            time.sleep(20)
        return rc

    def eval_finetuned(r: Runner) -> int:
        adapter = r.root / TRAIN_DIR / "adapter"
        if not adapter.exists():
            r.note(f"어댑터가 없습니다: {adapter} (train 단계가 끝나지 않음)")
            return 1
        return r.py("-m", "src.llm.evaluate", "--adapter", f"{TRAIN_DIR}/adapter", "--n", str(eval_n))

    return [
        Step("venv", "가상환경 준비", step_venv),
        Step("install", "패키지 설치 (torch CUDA 12.8 포함 — 처음엔 10~20분)", step_install,
             "네트워크 또는 Python 버전 문제일 가능성이 큽니다."),
        Step("check_env", "GPU 환경 점검", step_check_env,
             "로그의 [문제] 목록을 확인하세요. 대부분 torch·bitsandbytes 버전 문제입니다."),
        Step("smoke", "스모크 학습 (0.5B 모델, 약 5분)", step_smoke),
        Step("eval_base", f"원본 모델 평가 (7B 약 15GB 다운로드 + {eval_n}건, 30~75분 추정)", eval_base,
             "다운로드 중 끊겼으면 다시 실행하면 이어받습니다."),
        Step("train", f"본 학습 ({epochs:g}에폭, 에폭당 3~5시간 추정)", train,
             "CUDA out of memory면 GPU를 쓰는 다른 프로그램을 끄고 다시 실행하세요. 마지막 체크포인트부터 이어집니다."),
        Step("eval_finetuned", f"파인튜닝 모델 평가 ({eval_n}건, 30~75분 추정)", eval_finetuned),
        Step("compare", "비교표 생성 (outputs/eval/comparison.md)", lambda r: r.py("-m", "src.llm.compare")),
    ]


# ───────────────────────── 상태 관리 ─────────────────────────
def load_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"steps": {}}


def save_status(path: Path, st: dict) -> None:
    st["updated"] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)   # 중간에 끊겨도 깨진 JSON이 남지 않게


def run_steps(steps: list[Step], runner: Runner, status_path: Path, *, from_: str | None = None,
              only: str | None = None, force: bool = False, run_args: dict | None = None) -> int:
    names = [s.name for s in steps]
    for n in (from_, only):
        if n and n not in names:
            runner.note(f"알 수 없는 단계: {n} (가능: {', '.join(names)})")
            return 2
    start = names.index(from_) if from_ else 0
    selected = [s for s in steps[start:] if not only or s.name == only]

    st = load_status(status_path)
    st.setdefault("steps", {})
    st["run_args"] = run_args or {}
    st["started"] = now()

    for i, s in enumerate(selected, 1):
        prev = st["steps"].get(s.name, {})
        rerun = force or bool(only) or bool(from_)
        if prev.get("state") == "done" and not rerun:
            runner.note(f"[{i}/{len(selected)}] {s.title} — 이미 완료 ({prev.get('finished')}), 건너뜀")
            continue
        banner = f"[{i}/{len(selected)}] {s.title}"
        runner.note("\n" + "=" * 70 + f"\n{banner}   (시작 {now()})\n" + "=" * 70)
        st["current"] = s.name
        st["steps"][s.name] = {"state": "running", "started": now(), "title": s.title}
        save_status(status_path, st)

        t0 = time.time()
        try:
            rc = s.run(runner)
        except Exception as e:  # noqa: BLE001 — 실행기 자체 오류도 상태에 남긴다
            runner.note(f"실행기 오류: {e!r}")
            rc = 1
        mins = round((time.time() - t0) / 60, 1)
        st["steps"][s.name].update(state="done" if rc == 0 else "failed", finished=now(), minutes=mins, returncode=rc)
        if rc != 0:
            st["current"] = None
            save_status(status_path, st)
            runner.note(f"\n[실패] {s.title} (종료 코드 {rc}, {mins}분)")
            if s.hint:
                runner.note(f"  → {s.hint}")
            runner.note("  → 다시 실행하면 완료된 단계는 건너뛰고 여기서부터 이어갑니다.")
            runner.note("  → Claude에게 '파이프라인 로그 확인해줘'라고 하면 로그를 읽고 진단합니다.")
            return rc
        runner.note(f"[완료] {s.title} ({mins}분)")
        save_status(status_path, st)

    st["current"] = None
    st["finished"] = now()
    save_status(status_path, st)
    done_msg = "\n선택한 단계 완료."
    if any(s.name == "compare" for s in selected):
        done_msg = "\n모든 단계 완료. 결과: outputs/eval/comparison.md"
    runner.note(done_msg)
    return 0


def print_status(status_path: Path) -> None:
    st = load_status(status_path)
    if not st.get("steps"):
        print("아직 실행 기록이 없습니다.")
        return
    print(f"마지막 갱신 {st.get('updated')}  현재 단계: {st.get('current') or '-'}")
    for name, s in st["steps"].items():
        print(f"  {name:15s} {s.get('state', '-'):8s} {s.get('minutes', '-')!s:>6}분  {s.get('finished') or s.get('started', '')}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="학습·평가 전체 파이프라인")
    ap.add_argument("--epochs", type=float, default=1, help="본 학습 에폭 수 (기본 1)")
    ap.add_argument("--eval-n", type=int, default=60, help="평가 표본 수 (원본·파인튜닝 모델에 같은 값 사용)")
    ap.add_argument("--from", dest="from_", help="이 단계부터 다시 실행 (완료된 단계도 재실행)")
    ap.add_argument("--only", help="이 단계만 실행")
    ap.add_argument("--force", action="store_true", help="완료된 단계도 모두 재실행")
    ap.add_argument("--status", action="store_true", help="진행 상태만 출력")
    args = ap.parse_args(argv)

    status_path = LOG_DIR / "status.json"
    if args.status:
        print_status(status_path)
        return 0

    log = LogWriter(LOG_DIR / "pipeline.log")
    runner = Runner(ROOT, log)
    runner.note("\n" + "#" * 70 + f"\n# 파이프라인 시작 {now()}  Python {platform.python_version()}  {platform.platform()}")
    runner.note(f"# 인자: epochs={args.epochs:g} eval_n={args.eval_n} from={args.from_} only={args.only} force={args.force}")
    runner.note(f"# HF_HOME={runner.env.get('HF_HOME', '(기본값)')}\n" + "#" * 70)
    if shutil.which("nvidia-smi"):
        _, out = runner.capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
                                 "--format=csv,noheader"])
        runner.note(f"# GPU: {out.strip()}")
    keep_awake(True)
    try:
        return run_steps(make_steps(args.epochs, args.eval_n), runner, status_path,
                         from_=args.from_, only=args.only, force=args.force,
                         run_args={"epochs": args.epochs, "eval_n": args.eval_n})
    except KeyboardInterrupt:
        runner.note("\n중단됨 (Ctrl+C). 다시 실행하면 이어갑니다.")
        return 130
    finally:
        keep_awake(False)
        log.close()


if __name__ == "__main__":
    sys.exit(main())
