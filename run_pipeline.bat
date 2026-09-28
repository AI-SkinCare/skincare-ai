@echo off
rem Double-click to run the full pipeline: setup -> check -> smoke -> eval -> train -> eval -> compare.
rem Re-running skips finished steps and resumes training from the last checkpoint.
rem Options are passed through, e.g.:  run_pipeline.bat --epochs 2    run_pipeline.bat --status
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
    if not defined PY py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
    if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
)
if not defined PY (
    python -c "import sys" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.11 from https://www.python.org/downloads/ and check "Add to PATH".
    pause
    exit /b 1
)

echo Using: %PY%
echo Keep this window open. Logs: outputs\run_logs\pipeline.log
echo.
%PY% scripts\run_pipeline.py %*
echo.
pause
