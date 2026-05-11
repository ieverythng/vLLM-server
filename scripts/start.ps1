param(
    [ValidateSet("start", "stop", "restart", "status", "dry-run", "preflight")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"

$BaseDir = Split-Path -Parent $PSScriptRoot
$Python = $null
$PythonCandidates = @()
if ($env:VLLM_PYTHON) { $PythonCandidates += $env:VLLM_PYTHON }
$PythonCandidates += (Join-Path $BaseDir ".venv311\Scripts\python.exe")
$PythonCandidates += (Join-Path $BaseDir ".venv\Scripts\python.exe")

foreach ($candidate in $PythonCandidates) {
    if ($candidate -and (Test-Path $candidate)) {
        $Python = $candidate
        break
    }
}

$Manager = Join-Path $BaseDir "vllm_manager.py"
$GatewayPidFile = Join-Path $BaseDir "gateway.pid"
$GatewayLog = Join-Path $BaseDir "gateway.log"
$GatewayErrLog = Join-Path $BaseDir "gateway.err.log"
$GatewayHost = $env:GATEWAY_HOST
if (-not $GatewayHost) { $GatewayHost = "0.0.0.0" }
$GatewayPort = $env:GATEWAY_PORT
if (-not $GatewayPort) { $GatewayPort = "8001" }
$VllmTimeout = $env:VLLM_TIMEOUT
if (-not $VllmTimeout) { $VllmTimeout = "300" }

function Log($Message) {
    Write-Host "[start.ps1] $Message"
}

function Require-Python {
    if (-not $Python -or -not (Test-Path $Python)) {
        throw "Python runtime not found. Set VLLM_PYTHON or create .venv311/.venv in repo root."
    }
}

function Test-PidRunning($PidFile) {
    if (-not (Test-Path $PidFile)) { return $false }
    $pidText = (Get-Content $PidFile -Raw).Trim()
    if ($pidText -notmatch '^\d+$') { return $false }
    return [bool](Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue)
}

function Wait-Http($Url, $Label, [int]$TimeoutSeconds) {
    Log "Waiting for $Label at $Url"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5 | Out-Null
            Log "$Label is ready"
            return
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    throw "Timeout waiting for $Label"
}

function Start-Vllm {
    Require-Python
    & $Python $Manager start --no-wait
    if ($LASTEXITCODE -ne 0) { throw "vLLM manager failed to start backend" }
    Wait-Http "http://127.0.0.1:8000/v1/models" "vLLM /v1/models" ([int]$VllmTimeout)
}

function Start-Gateway {
    Require-Python
    if (Test-PidRunning $GatewayPidFile) {
        Log "Gateway already running (PID $(Get-Content $GatewayPidFile -Raw))"
        return
    }
    Log "Starting FastAPI gateway on ${GatewayHost}:${GatewayPort}"
    $args = @("-m", "uvicorn", "server:app", "--host", $GatewayHost, "--port", $GatewayPort, "--log-level", "info")
    $proc = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $BaseDir -RedirectStandardOutput $GatewayLog -RedirectStandardError $GatewayErrLog -PassThru -WindowStyle Hidden
    Set-Content -Path $GatewayPidFile -Value $proc.Id
    Wait-Http "http://127.0.0.1:${GatewayPort}/health" "gateway /health" 60
}

function Stop-Gateway {
    if (Test-PidRunning $GatewayPidFile) {
        $pidText = (Get-Content $GatewayPidFile -Raw).Trim()
        Log "Stopping gateway (PID $pidText)"
        Stop-Process -Id ([int]$pidText) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $GatewayPidFile -Force -ErrorAction SilentlyContinue
}

function Show-Status {
    Require-Python
    & $Python $Manager status
    Write-Host "python: $Python"
    if (Test-PidRunning $GatewayPidFile) {
        Write-Host "gateway: running (PID $((Get-Content $GatewayPidFile -Raw).Trim()))"
    } else {
        Write-Host "gateway: stopped"
    }
    Write-Host "gateway_url: http://127.0.0.1:$GatewayPort"
}

switch ($Action) {
    "dry-run" {
        Require-Python
        & $Python $Manager dry-run
    }
    "preflight" {
        Require-Python
        & $Python $Manager preflight
    }
    "start" {
        Start-Vllm
        Start-Gateway
        Show-Status
    }
    "stop" {
        Stop-Gateway
        Require-Python
        & $Python $Manager stop
    }
    "restart" {
        Stop-Gateway
        Require-Python
        & $Python $Manager stop
        Start-Vllm
        Start-Gateway
        Show-Status
    }
    "status" {
        Show-Status
    }
}
