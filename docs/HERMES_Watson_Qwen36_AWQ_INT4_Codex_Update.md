# HERMES / Watson vLLM Server — Updated Qwen3.6-27B-AWQ-INT4 Test Plan

**Repository:** `https://github.com/ieverythng/vLLM-server`  
**Branch:** `feat/Watson`  
**Primary model candidate:** `cyankiwi/Qwen3.6-27B-AWQ-INT4`  
**Purpose:** update the vLLM server implementation plan so Codex can make this model the first real vLLM-native test target while keeping the existing GGUF llama.cpp setup as rollback.

---

## 1. Why this model is now the first candidate

The model page for `cyankiwi/Qwen3.6-27B-AWQ-INT4` says the repository contains Hugging Face Transformers-format weights and configuration files, and that the artifacts are compatible with Transformers, vLLM, SGLang, KTransformers, etc.

The page identifies the model as Qwen3.6-27B with:
- 27B language model parameters,
- native 262,144-token context length,
- image-text-to-text architecture with a vision encoder,
- Qwen3.5/Qwen3.6-style architecture,
- vLLM serving instructions,
- tool-call serving instructions,
- text-only serving instructions.

The repo files page lists a total model repository size of approximately 20.5 GB and four safetensor model shards:
- `model-00001-of-00004.safetensors` — 5.35 GB
- `model-00002-of-00004.safetensors` — 5.34 GB
- `model-00003-of-00004.safetensors` — 4.39 GB
- `model-00004-of-00004.safetensors` — 5.36 GB

This makes it the closest practical vLLM-native candidate to the current working llama.cpp model:

```text
Current baseline:
  Qwen3.6-27B-Q3_K_M.gguf
  llama.cpp
  around 58 / 68 layers offloaded
  around 14.8–14.9 GB VRAM
  64k context possible
  ~15 tok/s at 64k
  ~40 tok/s at 16k
```

---

## 2. Key fit judgment for RTX 5070 Ti 16GB

Treat this model as:

```text
Status: primary test candidate
Fit class: possible but tight
Expected good first context: 16k
Expected experimental context: 32k
Expected risky context: 64k
Do not start with: 262k
```

Why:
- The HF model is 20.5 GB on disk, so it is larger than available VRAM before runtime/KV-cache overhead.
- The model is quantized/compressed, so it may fit with vLLM optimizations and CPU offload.
- Long context will still create heavy KV-cache pressure.
- Text-only mode should be used first to skip the vision encoder and free memory.
- Start with `max_model_len=16384`, then test 32768, then only test 65536 after stable benchmarks.

---

## 3. Important quantization detail

Although the repo name says `AWQ-INT4`, the model config identifies the quantization method as `compressed-tensors`, with:
- `format: pack-quantized`
- `num_bits: 4`
- `group_size: 32`
- `quant_method: compressed-tensors`

Codex should not blindly force `--quantization awq`.

Preferred first attempt:

```bash
vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --host 127.0.0.1 \
  --port 8000 \
  --quantization compressed-tensors \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 8 \
  --swap-space 8 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

If this fails with quantization inference errors, try:

```bash
vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 8 \
  --swap-space 8 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Reason: vLLM may be able to infer quantization from the model config.

---

## 4. Install requirement

The HF model card recommends `vllm>=0.19.0` for Qwen3.6. The current repo requirement should therefore be updated from a loose older lower bound to a Qwen3.6-aware requirement.

Recommended change:

```text
vllm>=0.19.0
```

or, if the installed environment has compatibility constraints, document the exact pinned working version after testing.

The model card also shows a vLLM install flow based on:

```bash
uv pip install vllm --torch-backend=auto
```

For the repo, Codex can keep `requirements.txt`, but should add a note in the setup docs that a fresh environment can use the model-card recommended `uv pip install vllm --torch-backend=auto` path.

---

## 5. Config update: make this the first real profile

Add or update `models.yaml`:

```yaml
profiles:
  qwen36_27b_awq_int4_16k:
    backend: vllm
    model_id: "cyankiwi/Qwen3.6-27B-AWQ-INT4"
    quantization: "compressed-tensors"
    dtype: "auto"
    max_model_len: 16384
    gpu_memory_utilization: 0.90
    cpu_offload_gb: 8
    swap_space: 8
    max_num_seqs: 2
    enable_prefix_caching: true
    language_model_only: true
    reasoning_parser: "qwen3"
    enable_auto_tool_choice: true
    tool_call_parser: "qwen3_coder"
    notes: "Primary first test profile for RTX 5070 Ti 16GB. Start here."

  qwen36_27b_awq_int4_32k:
    backend: vllm
    model_id: "cyankiwi/Qwen3.6-27B-AWQ-INT4"
    quantization: "compressed-tensors"
    dtype: "auto"
    max_model_len: 32768
    gpu_memory_utilization: 0.92
    cpu_offload_gb: 10
    swap_space: 12
    max_num_seqs: 1
    enable_prefix_caching: true
    language_model_only: true
    reasoning_parser: "qwen3"
    enable_auto_tool_choice: true
    tool_call_parser: "qwen3_coder"
    notes: "Experimental 32k profile. Use after 16k is stable."

  qwen36_27b_awq_int4_64k_risk:
    backend: vllm
    model_id: "cyankiwi/Qwen3.6-27B-AWQ-INT4"
    quantization: "compressed-tensors"
    dtype: "auto"
    max_model_len: 65536
    gpu_memory_utilization: 0.94
    cpu_offload_gb: 12
    swap_space: 16
    max_num_seqs: 1
    enable_prefix_caching: true
    language_model_only: true
    reasoning_parser: "qwen3"
    enable_auto_tool_choice: true
    tool_call_parser: "qwen3_coder"
    notes: "Risky 64k profile. Only test after 16k and 32k pass."
```

Update `config.yaml`:

```yaml
runtime:
  active_model_profile: "qwen36_27b_awq_int4_16k"

backend:
  host: "127.0.0.1"
  port: 8000

gateway:
  host: "0.0.0.0"
  port: 8001
  log_level: "info"

inference:
  provider: "vllm"
  base_url: "http://localhost:8001/v1"
  timeout_s: 120
  structured_json_mode: true
```

---

## 6. Required vLLM manager changes

`vllm_manager.py` should build the command from the active model profile.

Expected command for the first profile:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --host 127.0.0.1 \
  --port 8000 \
  --quantization compressed-tensors \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 8 \
  --swap-space 8 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Codex should add a dry-run command:

```bash
python vllm_manager.py dry-run
```

Expected behavior:
- print the exact launch command,
- do not download the model,
- do not start vLLM,
- validate active profile fields.

---

## 7. First manual test sequence

Run this before integrating with Hermes.

### 7.1 Install / update dependencies

Fresh environment option:

```bash
uv pip install vllm --torch-backend=auto
pip install -r requirements.txt
```

Or repo-style:

```bash
pip install -r requirements.txt
```

Confirm:

```bash
python -m vllm.entrypoints.openai.api_server --help
```

or:

```bash
vllm serve --help
```

### 7.2 Direct vLLM launch, no gateway

Start with the safest profile:

```bash
vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --host 127.0.0.1 \
  --port 8000 \
  --quantization compressed-tensors \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 8 \
  --swap-space 8 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### 7.3 Check server

```bash
curl http://localhost:8000/v1/models
```

### 7.4 Chat smoke test

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "messages": [
      {
        "role": "user",
        "content": "Reply with one short sentence confirming you are running through vLLM."
      }
    ],
    "max_tokens": 64,
    "temperature": 0.2
  }'
```

### 7.5 Tool-call smoke test

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "messages": [
      {
        "role": "user",
        "content": "Call the lookup_market_data tool for AAPL price."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "lookup_market_data",
          "description": "Lookup market metadata for a symbol.",
          "parameters": {
            "type": "object",
            "properties": {
              "symbol": {"type": "string"},
              "metric": {"type": "string", "enum": ["price", "volume", "volatility", "news"]}
            },
            "required": ["symbol", "metric"],
            "additionalProperties": false
          }
        }
      }
    ],
    "tool_choice": {
      "type": "function",
      "function": {"name": "lookup_market_data"}
    },
    "max_tokens": 256,
    "temperature": 0.0
  }'
```

---

## 8. Gateway test sequence

After direct vLLM works:

```bash
./start.sh start
./start.sh status
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

Chat through gateway:

```bash
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "default",
    "user": "WatsonMain",
    "messages": [
      {
        "role": "user",
        "content": "Reply with one short sentence confirming gateway routing."
      }
    ],
    "max_tokens": 64,
    "temperature": 0.2
  }'
```

iTrader JSON generation:

```bash
curl -X POST "http://localhost:8001/v1/itrader/generate" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "default",
    "prompt": "Return a minimal JSON object with fields symbol, task_family, horizon_bars, confidence.",
    "temperature": 0.0,
    "max_tokens": 256,
    "response_format": {"type": "json_object"}
  }'
```

---

## 9. Benchmark plan with this model

Benchmark in stages.

### Stage 1: 16k

Profile:

```text
qwen36_27b_awq_int4_16k
```

Run:

```bash
python scripts/benchmark.py \
  --base-url http://localhost:8001 \
  --model cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --concurrency 1 \
  --requests 4 \
  --output reports/qwen36_27b_awq_int4_16k_c1.md
```

Then:

```bash
python scripts/benchmark.py --base-url http://localhost:8001 --model cyankiwi/Qwen3.6-27B-AWQ-INT4 --concurrency 2 --requests 8 --output reports/qwen36_27b_awq_int4_16k_c2.md
python scripts/benchmark.py --base-url http://localhost:8001 --model cyankiwi/Qwen3.6-27B-AWQ-INT4 --concurrency 4 --requests 12 --output reports/qwen36_27b_awq_int4_16k_c4.md
```

### Stage 2: 32k

Only after 16k works.

Switch:

```yaml
runtime:
  active_model_profile: "qwen36_27b_awq_int4_32k"
```

Restart:

```bash
./start.sh restart
```

Run the same benchmark at concurrency 1 and 2 first.

### Stage 3: 64k

Only after 32k works.

Switch:

```yaml
runtime:
  active_model_profile: "qwen36_27b_awq_int4_64k_risk"
```

Restart and test only:

```bash
python scripts/benchmark.py \
  --base-url http://localhost:8001 \
  --model cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --concurrency 1 \
  --requests 2 \
  --output reports/qwen36_27b_awq_int4_64k_c1_risk.md
```

---

## 10. Pass/fail criteria

### 16k pass

Accept 16k as usable if:
- server starts reliably,
- `/v1/models` works,
- normal chat works,
- streaming works,
- tool-call smoke test works,
- JSON/iTrader generation works,
- no OOM,
- VRAM remains below hard saturation,
- concurrency 2 works,
- concurrency 4 either works or fails gracefully.

### 32k pass

Accept 32k as usable if:
- concurrency 1 works,
- tool calling works,
- latency is acceptable,
- no repeated OOM,
- gateway stays responsive.

### 64k pass

Accept 64k only if:
- concurrency 1 works,
- Hermes long prompt works,
- no OOM,
- latency is not worse than the current llama.cpp 64k path by an unacceptable margin.

Do not assume 64k is production default even if it starts.

---

## 11. Rollback

If vLLM fails with this model:
- keep the current llama.cpp GGUF model as primary,
- keep the vLLM branch changes because the gateway architecture is still useful,
- test a smaller Qwen3.6/Qwen3.5/Qwen coder family model in AWQ/GPTQ/compressed-tensors format,
- update `runtime.active_model_profile` accordingly.

Current known fallback:

```text
Qwen3.6-27B-Q3_K_M.gguf via llama.cpp
```

---

## 12. Immediate Codex prompt

Paste this to Codex:

```text
You are working in https://github.com/ieverythng/vLLM-server on branch feat/Watson.

Update the repo so cyankiwi/Qwen3.6-27B-AWQ-INT4 becomes the first real vLLM-native model test target.

Implement:
1. Update requirements/docs for Qwen3.6 requiring vLLM>=0.19.0.
2. Add models.yaml with profiles:
   - qwen36_27b_awq_int4_16k
   - qwen36_27b_awq_int4_32k
   - qwen36_27b_awq_int4_64k_risk
   - qwen_current_llamacpp_baseline as documented fallback.
3. Add runtime.active_model_profile to config.yaml.
4. Split backend/gateway config if not already done.
5. Ensure inference.base_url points to http://localhost:8001/v1 for gateway use.
6. Update vllm_manager.py to build the vLLM launch command from the active model profile.
7. Add support for:
   - quantization compressed-tensors
   - cpu_offload_gb
   - swap_space
   - max_num_seqs
   - language_model_only
   - reasoning_parser
   - enable_auto_tool_choice
   - tool_call_parser
   - enable_prefix_caching
8. Add python vllm_manager.py dry-run to print the exact launch command without starting or downloading the model.
9. Add docs/qwen36-awq-int4-test-plan.md with:
   - direct vLLM serve command,
   - gateway test commands,
   - tool-call smoke test,
   - benchmark plan,
   - pass/fail criteria,
   - rollback to llama.cpp GGUF.
10. Do not actually download the model during implementation.
11. Do not remove the existing GGUF llama.cpp fallback notes.
12. Add minimal validation tests where practical.

Return:
- changed files,
- exact commands to run,
- known risks,
- next benchmark step.
```

---

## 13. Definition of done

Done when:

- `models.yaml` exists,
- active profile selects `cyankiwi/Qwen3.6-27B-AWQ-INT4`,
- dry-run prints the expected vLLM command,
- no model download happens during dry-run,
- `./start.sh start` can attempt the selected profile,
- gateway still starts on `8001`,
- direct vLLM and gateway curl tests are documented,
- benchmark plan includes 16k, 32k, and 64k stages,
- llama.cpp GGUF fallback remains documented.
