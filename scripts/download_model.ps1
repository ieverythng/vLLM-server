param(
    [string]$RepoId = "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    [string]$LocalDir = "D:\MODELS\cyankiwi\Qwen3.6-27B-AWQ-INT4"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:HF_HOME) { $env:HF_HOME = "D:\MODELS\.cache\huggingface" }

$hf = Get-Command hf -ErrorAction SilentlyContinue
if ($hf) {
    & $hf.Source download $RepoId --local-dir $LocalDir
    exit $LASTEXITCODE
}

$huggingFaceCli = Get-Command huggingface-cli -ErrorAction SilentlyContinue
if ($huggingFaceCli) {
    & $huggingFaceCli.Source download $RepoId --local-dir $LocalDir --resume-download
    exit $LASTEXITCODE
}

python -m huggingface_hub.commands.huggingface_cli download $RepoId --local-dir $LocalDir --resume-download
