# 실행: $env:AI_INTERNAL_TOKEN 설정 후 .\scripts\run_disease_pipeline.ps1
param(
    [int]$Port = 8000,
    [string]$ListenAddress = "127.0.0.1",
    [ValidateSet("sllm", "mock")][string]$RecBackend = "sllm",
    [string]$AdapterDir = "outputs/rec_sllm_v2/adapter",
    [string]$ClassifierDevice = "cpu",
    [ValidateRange(0.0, 1.0)][double]$Threshold = 0.6
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
if (-not $env:AI_INTERNAL_TOKEN) { throw "AI_INTERNAL_TOKEN 환경변수를 먼저 설정하세요." }
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "프로젝트 .venv를 먼저 준비하세요." }
if ($RecBackend -eq "sllm") {
    $adapterPath = if ([IO.Path]::IsPathRooted($AdapterDir)) { $AdapterDir } else { Join-Path $repoRoot $AdapterDir }
    $adapterConfig = Get-Content -LiteralPath (Join-Path $adapterPath "adapter_config.json") -Raw | ConvertFrom-Json
    if ($adapterConfig.base_model_name_or_path -ne "Qwen/Qwen2.5-7B-Instruct") {
        throw "Qwen2.5-7B용 어댑터가 아닙니다. AdapterDir을 확인하세요."
    }
    $env:REC_ADAPTER_DIR = $adapterPath
}
$env:REC_BACKEND = $RecBackend
$env:REC_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
$env:DISEASE_BACKEND = "densenet"
$env:DISEASE_WEIGHTS = Join-Path $repoRoot "model\weights\densenet121_s2.pt"
$env:DISEASE_DEVICE = $ClassifierDevice
$env:DISEASE_THRESHOLD = $Threshold.ToString([Globalization.CultureInfo]::InvariantCulture)
$env:PYTHONUTF8 = "1"
if (-not $env:HF_HOME -and (Test-Path -LiteralPath "D:\hf_cache")) { $env:HF_HOME = "D:\hf_cache" }
if ($env:HF_HOME) {
    $preferredHub = Join-Path $env:HF_HOME "hub"
    $cachedBase = "models--Qwen--Qwen2.5-7B-Instruct\refs\main"
    $configuredCacheHasBase = $env:HF_HUB_CACHE -and (Test-Path -LiteralPath (Join-Path $env:HF_HUB_CACHE $cachedBase))
    if ((Test-Path -LiteralPath (Join-Path $preferredHub $cachedBase)) -and -not $configuredCacheHasBase) {
        # PC 이전 후 다른 HF_HUB_CACHE가 남아 있어도 이미 받은 15GB 모델을 재사용한다.
        $env:HF_HUB_CACHE = $preferredHub
    }
}
Push-Location $repoRoot
try {
    & $pythonPath -m uvicorn server.main:app --host $ListenAddress --port $Port
    if ($LASTEXITCODE -ne 0) { throw "추론 서버가 종료 코드 $LASTEXITCODE 로 끝났습니다." }
} finally { Pop-Location }
