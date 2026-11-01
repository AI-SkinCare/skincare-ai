@echo off
rem v2 data + Qwen3-8B base (1 epoch) -> evaluate all 201 -> comparison table.
rem Training auto-resumes from the last checkpoint on CUDA errors (up to 15 retries).
rem Log: outputs\run_logs\v2_qwen3.log
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if not defined HF_HOME if exist D:\ set HF_HOME=D:\hf_cache
set LOG=outputs\run_logs\v2_qwen3.log
set CFG=configs\sllm_qlora_v2_qwen3.yaml
set OUT=outputs\rec_sllm_v2_qwen3
set TRIES=0

if exist %OUT%\adapter goto evaluate

:train
set RESUME=
if exist %OUT%\checkpoint-* set RESUME=--resume
echo [%date% %time%] train start (try %TRIES%) %RESUME% >> %LOG%
.venv\Scripts\python.exe -m src.llm.train_qlora --config %CFG% --epochs 1 %RESUME% >> %LOG% 2>&1
if not errorlevel 1 goto evaluate
set /a TRIES+=1
echo [%date% %time%] train failed, retry %TRIES% >> %LOG%
if %TRIES% geq 15 goto fail
timeout /t 20 /nobreak >nul
goto train

:evaluate
echo [%date% %time%] train done, eval start >> %LOG%
.venv\Scripts\python.exe -m src.llm.evaluate --config %CFG% --adapter %OUT%/adapter >> %LOG% 2>&1
echo [%date% %time%] eval exit %errorlevel% >> %LOG%
.venv\Scripts\python.exe -m src.llm.compare >> %LOG% 2>&1
echo [%date% %time%] ALL DONE >> %LOG%
exit /b 0

:fail
echo [%date% %time%] TRAIN FAILED after %TRIES% retries >> %LOG%
echo [%date% %time%] ALL DONE >> %LOG%
exit /b 1
