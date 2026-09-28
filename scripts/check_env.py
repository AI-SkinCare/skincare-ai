"""학습 전 환경 점검: python scripts/check_env.py

import 성공만으로는 부족하다. RTX 50 시리즈(Blackwell, sm_120)는 CUDA 12.8 이상으로
빌드된 torch/bitsandbytes가 아니면 import는 되지만 GPU 연산에서
'no kernel image is available' 로 실패한다. 그래서 실제로 GPU에서 4-bit 연산을 한 번 돌려 본다.
"""
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

problems: list[str] = []

print(f"Python {sys.version.split()[0]}")
try:
    import torch
except ImportError:
    sys.exit("torch 미설치 → docs/ai-module.md의 설치 순서를 따르세요")
print(f"torch {torch.__version__} / CUDA 빌드 {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("CUDA를 인식하지 못했습니다. CPU용 torch가 설치됐을 가능성이 큽니다 (pip show torch 확인).")

p = torch.cuda.get_device_properties(0)
vram = p.total_memory / 1024**3
cc = (p.major, p.minor)
print(f"GPU {p.name} / VRAM {vram:.1f}GB / compute capability {cc[0]}.{cc[1]} / bf16 {torch.cuda.is_bf16_supported()}")

# 1) Blackwell(sm_120)인데 CUDA 12.8 미만 빌드인지
if cc[0] >= 12:
    cuda_ver = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
    if cuda_ver < (12, 8):
        problems.append(
            f"RTX 50 시리즈에는 CUDA 12.8 이상 빌드 torch가 필요합니다(현재 {torch.version.cuda}). "
            "docs/ai-module.md의 cu128 설치 명령으로 다시 설치하세요.")

# 2) torch 기본 연산이 실제로 GPU에서 도는지
try:
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    (a @ a).sum().item()
    print("torch GPU 연산 OK")
except Exception as e:  # noqa: BLE001
    problems.append(f"torch GPU 연산 실패: {e}")

# 3) bitsandbytes 4-bit 연산이 실제로 도는지 (QLoRA의 핵심)
try:
    import bitsandbytes as bnb
    lin = bnb.nn.Linear4bit(256, 256, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
    lin(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)).sum().item()
    print(f"bitsandbytes {bnb.__version__} 4-bit GPU 연산 OK")
except Exception as e:  # noqa: BLE001
    problems.append(f"bitsandbytes 4-bit 연산 실패: {e} → pip install -U bitsandbytes")

for mod in ("transformers", "peft", "accelerate", "fastapi"):
    try:
        print(f"{mod} {__import__(mod).__version__}")
    except ImportError:
        problems.append(f"{mod} 미설치 → pip install -r requirements-train.txt")

print("\n[권장 설정]")
# 16GB 카드는 OS 예약분 때문에 15.5~15.9GB로 보고된다
if vram >= 15:
    print("- 7B QLoRA, max_seq_len 3072 그대로 사용")
elif vram >= 10:
    print("- 7B QLoRA 가능. OOM 나면 configs/sllm_qlora.yaml의 max_seq_len을 2560으로")
else:
    print("- VRAM 부족: model.base를 Qwen/Qwen2.5-3B-Instruct로 변경 권장")
print("- 서빙: 7B 4-bit 약 5~6GB + 생성 버퍼. 회귀 모델과 같은 GPU를 쓰므로 여유 2GB 이상 확보")
print("- 학습 중에는 추론 서버를 끄세요 (같은 GPU를 두고 경쟁하면 OOM)")
for drive in ("C:\\", "D:\\"):
    try:
        print(f"- {drive} 남은 용량 {shutil.disk_usage(drive).free / 1024**3:.0f}GB")
    except OSError:
        pass
hf_home = os.environ.get("HF_HOME")
if hf_home:
    print(f"- HF_HOME = {hf_home}")
else:
    print("- HF_HOME 미설정 → 7B 모델(약 15GB)이 C드라이브에 받아집니다. $env:HF_HOME=\"D:\\hf_cache\" 권장")

if problems:
    print("\n[문제]")
    for m in problems:
        print(f"- {m}")
    sys.exit(1)
print("\n모든 점검 통과 → python -m src.llm.train_qlora --smoke 로 진행하세요.")
