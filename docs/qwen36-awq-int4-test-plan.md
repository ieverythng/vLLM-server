# Qwen3.6 27B AWQ INT4 vLLM Test Plan

Primary candidate:

```text
cyankiwi/Qwen3.6-27B-AWQ-INT4
```

The model is treated as possible but tight on an RTX 5070 Ti 16 GB. Start with 16k context, then test 32k, and only attempt 64k after the lower profiles are stable. Do not start with 262k context.

## Fit Judgment

Expected first profile:

```text
Status: primary test candidate
Fit class: possible but tight
Expected good first context: 16k
Expected experimental context: 32k
Expected risky context: 64k
Do not start with: 262k
```

Reasons:

- The repository is about 20.5 GB on disk.
- Runtime KV cache pressure is the main risk on 16 GB VRAM.
- Text-only mode skips the vision encoder and frees memory.
- CPU offload and swap are enabled in the first profile.

## Quantization

Although the model name includes AWQ INT4, the config uses compressed-tensors:

```text
quant_method: compressed-tensors
format: pack-quantized
num_bits: 4
group_size: 32
```

First attempt should use:

```text
--quantization compressed-tensors
```

If vLLM rejects explicit quantization, retry without `--quantization` and allow vLLM to infer it from the model config.

## Direct vLLM Command

The active 16k profile should dry-run to a command equivalent to:

```powershell
.\.venv\Scripts\python.exe -m vllm.entrypoints.openai.api_server `
  --model D:\MODELS\cyankiwi\Qwen3.6-27B-AWQ-INT4 `
  --host 127.0.0.1 `
  --port 8000 `
  --quantization compressed-tensors `
  --dtype auto `
  --max-model-len 16384 `
  --gpu-memory-utilization 0.90 `
  --cpu-offload-gb 8 `
  --swap-space 8 `
  --max-num-seqs 2 `
  --enable-prefix-caching `
  --language-model-only `
  --reasoning-parser qwen3 `
  --enable-auto-tool-choice `
  --tool-call-parser qwen3_coder
```

Use dry-run first:

```powershell
.\.venv\Scripts\python.exe vllm_manager.py dry-run
```

## Gateway Tests

After direct vLLM works:

```powershell
.\scripts\start.ps1 start
.\scripts\start.ps1 status
Invoke-WebRequest http://localhost:8001/health -UseBasicParsing
Invoke-WebRequest http://localhost:8001/v1/models -UseBasicParsing
```

Chat through gateway:

```powershell
$body = @{
  model = "default"
  user = "WatsonMain"
  messages = @(@{ role = "user"; content = "Reply with one short sentence confirming gateway routing." })
  max_tokens = 64
  temperature = 0.2
} | ConvertTo-Json -Depth 8

Invoke-RestMethod http://localhost:8001/v1/chat/completions -Method Post -ContentType "application/json" -Body $body
```

iTrader JSON:

```powershell
$body = @{
  model = "default"
  prompt = "Return a minimal JSON object with fields symbol, task_family, horizon_bars, confidence."
  temperature = 0.0
  max_tokens = 256
  response_format = @{ type = "json_object" }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod http://localhost:8001/v1/itrader/generate -Method Post -ContentType "application/json" -Body $body
```

## Benchmark Stages

Stage 1, 16k:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --base-url http://localhost:8001 --model cyankiwi/Qwen3.6-27B-AWQ-INT4 --concurrency 1 --requests 4 --output reports\qwen36_27b_awq_int4_16k_c1.md
.\.venv\Scripts\python.exe scripts\benchmark.py --base-url http://localhost:8001 --model cyankiwi/Qwen3.6-27B-AWQ-INT4 --concurrency 2 --requests 8 --output reports\qwen36_27b_awq_int4_16k_c2.md
.\.venv\Scripts\python.exe scripts\benchmark.py --base-url http://localhost:8001 --model cyankiwi/Qwen3.6-27B-AWQ-INT4 --concurrency 4 --requests 12 --output reports\qwen36_27b_awq_int4_16k_c4.md
```

Stage 2, 32k:

1. Set `runtime.active_model_profile` to `qwen36_27b_awq_int4_32k`.
2. Restart.
3. Run concurrency 1 and 2 first.

Stage 3, 64k:

1. Set `runtime.active_model_profile` to `qwen36_27b_awq_int4_64k_risk`.
2. Restart.
3. Run concurrency 1 with only two requests first.

## Pass/Fail Criteria

16k pass:

- Server starts reliably.
- `/v1/models` works.
- Normal chat works.
- Streaming works.
- Tool-call smoke test works.
- JSON/iTrader generation works.
- No OOM.
- Concurrency 2 works.
- Concurrency 4 works or fails gracefully.

32k pass:

- Concurrency 1 works.
- Tool calling works.
- Gateway stays responsive.
- No repeated OOM.

64k pass:

- Concurrency 1 works.
- Hermes long prompt works.
- No OOM.
- Latency is not worse than the current llama.cpp 64k path by an unacceptable margin.

Do not treat 64k as production default unless it is repeatedly stable.

## Rollback

If this model fails:

- Keep the vLLM gateway changes.
- Restore the llama.cpp GGUF path for serving.
- Keep testing smaller Qwen3.6/Qwen3.5/Qwen coder AWQ/GPTQ/compressed-tensors models.
- Keep `qwen_current_llamacpp_baseline` as the fallback profile note.
