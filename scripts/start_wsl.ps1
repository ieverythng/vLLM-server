param(
    [ValidateSet("start", "stop", "restart", "status", "dry-run", "preflight")]
    [string]$Action = "preflight",
    [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"

function Convert-WindowsPathToWsl([string]$WindowsPath) {
    $full = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($full -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLower()
        $rest = $Matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert path to WSL format: $WindowsPath"
}

$BaseDir = Split-Path -Parent $PSScriptRoot
$RepoLinux = Convert-WindowsPathToWsl $BaseDir
$LinuxPython = if ($env:VLLM_WSL_PYTHON) { $env:VLLM_WSL_PYTHON } else { "/home/juanbeck/vLLM-server/venv/bin/python" }

Write-Host "[start_wsl.ps1] distro=$Distro action=$Action"
Write-Host "[start_wsl.ps1] repo=$RepoLinux"
Write-Host "[start_wsl.ps1] python=$LinuxPython"

$Cmd = "set -euo pipefail; test -x '$LinuxPython'; cd '$RepoLinux'; '$LinuxPython' vllm_manager.py $Action"
& wsl -d $Distro -- bash -lc $Cmd

if ($LASTEXITCODE -ne 0) {
    throw "WSL command failed with exit code $LASTEXITCODE"
}
