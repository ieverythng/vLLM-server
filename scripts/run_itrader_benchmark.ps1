param(
    [string]$ITraderRepo = "C:\Users\Admin\projects\iTRADER\itrader-azr",
    [string]$ITraderPython = "C:\Users\Admin\itrader\python.exe",
    [string]$ApiBaseUrl = "http://127.0.0.1:8001/v1",
    [string]$ApiModel = "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    [int]$Tasks = 16,
    [int]$BatchSize = 4,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Assert-Path($Path, $Label) {
    if (-not (Test-Path $Path)) {
        throw "$Label not found: $Path"
    }
}

function Normalize-BaseUrl([string]$Url) {
    $clean = $Url.Trim()
    if ($clean.EndsWith("/")) { $clean = $clean.TrimEnd("/") }
    return $clean
}

Assert-Path $ITraderRepo "iTRADER repo"
Assert-Path $ITraderPython "iTRADER python"

$base = Normalize-BaseUrl $ApiBaseUrl
$probeUrl = if ($base.EndsWith("/v1")) { "$base/models" } else { "$base/v1/models" }

if (-not $DryRun) {
    try {
        Invoke-WebRequest -Uri $probeUrl -UseBasicParsing -TimeoutSec 8 | Out-Null
        Write-Host "[itrader-benchmark] endpoint reachable: $probeUrl"
    } catch {
        throw "Endpoint not reachable at $probeUrl. Start vLLM gateway first."
    }
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = Join-Path $ITraderRepo "data"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$outPath = Join-Path $outDir "task_pool_vllm_$stamp.jsonl"
$qaPath = Join-Path $outDir "task_pool_vllm_$stamp.qa.json"
$candPath = Join-Path $outDir "task_pool_vllm_$stamp.candidates.jsonl"

$args = @(
    "-m", "itrader.scripts.gen_task_pool",
    "--backend", "vllm",
    "--api_model", $ApiModel,
    "--api_base_url", $base,
    "--n_tasks", "$Tasks",
    "--batch_size", "$BatchSize",
    "--max_backend_errors", "5",
    "--out", $outPath,
    "--qa_out", $qaPath,
    "--candidate_log_out", $candPath
)

Write-Host "[itrader-benchmark] python: $ITraderPython"
Write-Host "[itrader-benchmark] repo: $ITraderRepo"
Write-Host "[itrader-benchmark] out: $outPath"
Write-Host "[itrader-benchmark] cmd: $ITraderPython $($args -join ' ')"

if ($DryRun) {
    exit 0
}

Push-Location $ITraderRepo
try {
    & $ITraderPython @args
} finally {
    Pop-Location
}

Write-Host "[itrader-benchmark] complete"
Write-Host "[itrader-benchmark] qa: $qaPath"
Write-Host "[itrader-benchmark] candidates: $candPath"
