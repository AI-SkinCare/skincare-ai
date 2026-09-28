@echo off
rem Re-evaluation for the presentation: finetuned model on all 201 -> base model n=60 (new scoring) -> comparison table.
rem Log: outputs\run_logs\eval_full.log
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if not defined HF_HOME if exist D:\ set HF_HOME=D:\hf_cache
set LOG=outputs\run_logs\eval_full.log
echo [%date% %time%] start finetuned n=201 >> %LOG%
.venv\Scripts\python.exe -m src.llm.evaluate --adapter outputs/rec_sllm_v1/adapter >> %LOG% 2>&1
echo [%date% %time%] finetuned exit %errorlevel% >> %LOG%
.venv\Scripts\python.exe -m src.llm.evaluate --base Qwen/Qwen2.5-7B-Instruct --n 60 >> %LOG% 2>&1
echo [%date% %time%] base exit %errorlevel% >> %LOG%
.venv\Scripts\python.exe -m src.llm.compare >> %LOG% 2>&1
echo [%date% %time%] ALL DONE >> %LOG%
