# 사용: .\scripts\run_server.ps1            (mock)
#       .\scripts\run_server.ps1 -Backend sllm
param([string]$Backend = "mock", [int]$Port = 8000)
if (-not $env:AI_INTERNAL_TOKEN) { Write-Error "AI_INTERNAL_TOKEN 환경변수를 먼저 설정하세요"; exit 1 }
$env:REC_BACKEND = $Backend
$env:PYTHONUTF8 = "1"
uvicorn server.main:app --host 0.0.0.0 --port $Port
