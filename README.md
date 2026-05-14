# vLLM Server Gateway

WSL-first vLLM inference gateway for Hermes/Watson, iTrader, and future projects that need a reusable OpenAI-compatible model service.

The stack is intentionally split into two services:

```text
Clients / Apps
    |
    v
FastAPI gateway  http://0.0.0.0:8001
    |
    v
vLLM backend     http://127.0.0.1:8000
    |
    v
NVIDIA GPU
```

The first real vLLM-native target is `cyankiwi/Qwen3.6-27B-AWQ-INT4`, staged at 16k, 32k, and 64k context profiles. The repo keeps the existing `Qwen3.6-27B-Q3_K_M.gguf` llama.cpp setup documented as rollback.

## Setup

Use PowerShell from the repo root.

```powershell
C:\Users\Admin\itrader\python.exe -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\pip.exe install -r requirements.txt
```

For a fresh vLLM environment, the Qwen3.6 model guidance also supports:

```powershell
uv pip install vllm --torch-backend=auto
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Confirm CUDA/GPU visibility:

```powershell
nvidia-smi
```

### WSL2 Runtime (recommended for `vllm._C`)

If Windows Python reports `missing_vllm_compiled_runtime`, launch through WSL2 using a Linux venv that already contains `vllm._C`.

```powershell
.\scripts\start_wsl.ps1 preflight
.\scripts\start_wsl.ps1 dry-run
.\scripts\start_wsl.ps1 start
.\scripts\start_wsl.ps1 status
.\scripts\start_wsl.ps1 stop
```

Defaults:

- distro: `Ubuntu`
- Linux Python: `/home/juanbeck/vLLM-server/venv/bin/python`

Override the interpreter if needed:

```powershell
$env:VLLM_WSL_PYTHON = "/home/<user>/<repo>/venv/bin/python"
.\scripts\start_wsl.ps1 preflight
```

## Model Download

The preferred local model location is:

```text
D:\MODELS\cyankiwi\Qwen3.6-27B-AWQ-INT4
```

Download manually if needed:

```powershell
.\scripts\download_model.ps1
```

`models.yaml` keeps the Hugging Face model ID and prefers the local path when it exists.

## Configuration

`config.yaml` selects the active profile:

```yaml
runtime:
  active_model_profile: "qwen36_27b_awq_int4_16k"
```

Profiles live in `models.yaml`:

- `qwen36_27b_awq_int4_16k`: primary first test on 16 GB VRAM
- `qwen36_27b_awq_int4_32k`: experimental after 16k is stable
- `qwen36_27b_awq_int4_64k_risk`: risky, single-concurrency validation only
- `qwen_current_llamacpp_baseline`: documented fallback reference

The default network split is:

- vLLM backend: `127.0.0.1:8000`
- FastAPI gateway: `0.0.0.0:8001`
- Client API base URL: `http://localhost:8001/v1`

## Dry Run

Validate the active profile and print the exact vLLM command without downloading or starting the model:

```powershell
.\.venv311\Scripts\python.exe vllm_manager.py preflight
.\.venv311\Scripts\python.exe vllm_manager.py dry-run
```

Expected first profile command includes:

```text
--model D:\MODELS\cyankiwi\Qwen3.6-27B-AWQ-INT4
--quantization compressed-tensors
--max-model-len 16384
--max-num-seqs 2
--language-model-only
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

If the local model directory is missing, dry-run prints the Hugging Face ID instead.

## Launch

Recommended: WSL2 launcher (first choice)

```powershell
.\scripts\start_wsl.ps1 dry-run
.\scripts\start_wsl.ps1 preflight
.\scripts\start_wsl.ps1 start
.\scripts\start_wsl.ps1 status
.\scripts\start_wsl.ps1 stop
```

Windows launcher (fallback only):

```powershell
.\scripts\start.ps1 dry-run
.\scripts\start.ps1 preflight
.\scripts\start.ps1 start
.\scripts\start.ps1 status
.\scripts\start.ps1 stop
.\scripts\start.ps1 restart
```

Direct manager commands:

```powershell
.\.venv311\Scripts\python.exe vllm_manager.py preflight
.\.venv311\Scripts\python.exe vllm_manager.py start
.\.venv311\Scripts\python.exe vllm_manager.py status
.\.venv311\Scripts\python.exe vllm_manager.py stop
```

If preflight reports `missing_vllm_compiled_runtime`, the current interpreter cannot launch vLLM yet.

`start.sh` is retained for WSL/Linux, but PowerShell is the supported path for this repo.

## API Smoke Tests

Gateway health:

```powershell
Invoke-WebRequest http://localhost:8001/health -UseBasicParsing
Invoke-WebRequest http://localhost:8001/v1/models -UseBasicParsing
```

Chat through the gateway:

```powershell
$body = @{
  model = "default"
  user = "WatsonMain"
  messages = @(@{ role = "user"; content = "Reply with one short sentence confirming gateway routing." })
  max_tokens = 64
  temperature = 0.2
} | ConvertTo-Json -Depth 8

Invoke-RestMethod http://localhost:8001/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

iTrader JSON generation:

```powershell
$body = @{
  model = "default"
  prompt = "Return a minimal JSON object with fields symbol, task_family, horizon_bars, confidence."
  temperature = 0.0
  max_tokens = 256
  response_format = @{ type = "json_object" }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod http://localhost:8001/v1/itrader/generate `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Validation

Fast validation:

```powershell
.\.venv\Scripts\python.exe -m py_compile server.py vllm_manager.py scripts\benchmark.py scripts\dev_harness_adapter.py
.\.venv\Scripts\python.exe -m pytest
```

Full GPU validation, which can start vLLM and load the active model:

```powershell
.\.venv\Scripts\python.exe -m pytest -m gpu --run-gpu
```

The GPU path uses the configured active profile by default. Override only the smoke model with:

```powershell
$env:VLLM_SMOKE_MODEL = "cyankiwi/Qwen3.6-27B-AWQ-INT4"
```

## Benchmark Stages

Start with 16k:

```powershell
.\.venv311\Scripts\python.exe scripts\benchmark.py `
  --base-url http://localhost:8001 `
  --model cyankiwi/Qwen3.6-27B-AWQ-INT4 `
  --concurrency 1 `
  --requests 4 `
  --output reports\qwen36_27b_awq_int4_16k_c1.md
```

iTRADER task-pool benchmark against the same endpoint:

```powershell
.\scripts\run_itrader_benchmark.ps1 -Tasks 16 -BatchSize 4
```

Then increase to concurrency 2 and 4. Move to the 32k profile only after 16k is stable. Move to 64k only after 32k works, and test only concurrency 1 first.

Quick tok/s + long-context sweep (auto-resolves served model ID from `/v1/models` to avoid 404 name mismatches):

```powershell
/home/juanbeck/vLLM-server/venv/bin/python scripts/benchmark_tokps.py \
  --base-url http://127.0.0.1:8001 \
  --auto-model \
  --requests 4 \
  --concurrency 2 \
  --max-tokens 384 \
  --context-sweep \
  --output reports/benchmark-tokps.json
```

## Pass Criteria

Accept 16k as usable when:

- vLLM starts reliably.
- `/v1/models` works.
- Chat, streaming, tool calls, and iTrader JSON work.
- No OOM occurs.
- Concurrency 2 works.
- Concurrency 4 works or fails gracefully.

Accept 32k only after concurrency 1 works without repeated OOM. Treat 64k as experimental even if it starts.

## More Docs

- [Serving topology](docs/vllm-topology.md)
- [ZeroTier routing runbook (main server)](docs/zerotier-routing-runbook.md)
- [Qwen3.6 AWQ INT4 test plan](docs/qwen36-awq-int4-test-plan.md)
- [Future package handoff](docs/package-handoff.md)
- [Original GPT-5.5 handoff note](docs/HERMES_Watson_Qwen36_AWQ_INT4_Codex_Update.md)
