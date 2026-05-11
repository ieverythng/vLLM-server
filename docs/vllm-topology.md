# vLLM Inference Gateway Topology

This repository runs the inference stack as a two-tier service:

1. vLLM serves the model API on port `8000`.
2. FastAPI serves the gateway on port `8001` and forwards requests to vLLM.

The canonical Windows launcher is `scripts/start.ps1`.

## Serving Topology

```text
                        ZeroTier network
                               |
                               v
                    +-----------------------+
                    |  FastAPI Gateway      |
                    |  server.py            |
                    |  http://0.0.0.0:8001  |
                    |                       |
 Clients / Apps --->|  /v1/chat/completions |----+
 (Hermes, iTrader,  |  /v1/models           |    |
 Discord, Gamma)    |  /v1/itrader/generate |    |
                    +-----------------------+    |
                               |                 |
                               | local HTTP      |
                               v                 |
                    +-----------------------+    |
                    |  vLLM OpenAI server   |<---+
                    |  vllm_manager.py      |
                    |  http://127.0.0.1:8000|
                    |  /v1/models           |
                    |  /v1/chat/completions |
                    +-----------------------+
                               |
                               v
                        NVIDIA GPU / VRAM
```

The gateway is a lightweight routing and policy layer. vLLM owns token generation, KV cache, sampling, and model loading.

## Runtime Configuration

Relevant config-backed settings:

- `runtime.active_model_profile`: selected model profile from `models.yaml`
- `server.host`: gateway host bind, typically `0.0.0.0`
- `server.port`: gateway port, `8001`
- `backend.host`: vLLM backend host, typically `127.0.0.1`
- `backend.port`: vLLM backend port, `8000`
- `inference.base_url`: client-facing gateway base URL
- `auth.api_key` or `VLLM_API_KEY`: optional bearer auth

Relevant model-profile settings:

- `model_id`: Hugging Face model ID
- `local_path`: preferred local model path
- `prefer_local_path`: use the local model path when it exists
- `quantization`, `dtype`, `max_model_len`, `gpu_memory_utilization`
- `cpu_offload_gb`, `swap_space`, `max_num_seqs`
- `language_model_only`, `reasoning_parser`, `enable_auto_tool_choice`, `tool_call_parser`

## Role Routing

Role routing is controlled by the `roles:` block in `config.yaml`.

| Role | Temperature | Max tokens | Notes |
| --- | ---: | ---: | --- |
| WatsonMain | 0.7 | 8192 | General-purpose primary assistant |
| WatsonMemory | 0.3 | 4096 | Lower temperature for memory/recall |
| WatsonReviewer | 0.5 | 4096 | Review and critique workflow |
| WatsonDev | 0.7 | 8192 | Implementation and coding tasks |
| iTrader | 0.5 | 2048 | Structured market/task generation |

Routing flow:

1. The gateway reads `request.user`.
2. If it matches a known role, role defaults are applied.
3. Request-level overrides win.
4. `model: "default"` resolves to the active model profile.
5. The request is forwarded to vLLM.

## iTrader Path

`POST /v1/itrader/generate` accepts a structured prompt and returns the raw vLLM result in a `generation` envelope.

The endpoint:

- Uses a market-analysis system instruction.
- Defaults to `response_format: {"type": "json_object"}`.
- Routes `model: "default"` to the active profile model.

## Migration and Rollback

The current vLLM target is `cyankiwi/Qwen3.6-27B-AWQ-INT4`. The known fallback remains:

```text
Qwen3.6-27B-Q3_K_M.gguf via llama.cpp
```

Rollback strategy:

1. Stop this stack with `.\scripts\start.ps1 stop`.
2. Restore the llama.cpp service or launch command.
3. Point client `base_url` back to the llama.cpp-compatible endpoint.
4. Keep this gateway branch because the two-tier architecture remains useful.

## ZeroTier

ZeroTier can expose the gateway across a private mesh without opening the service publicly.

Current network ID from `config.yaml`:

```text
3b19b3a716937e29
```

Use the ZeroTier IP with port `8001` for remote clients. Keep the gateway bound to `0.0.0.0` so the ZeroTier interface can reach it.

## Files

- `README.md`: main operator runbook
- `config.yaml`: runtime and gateway/backend config
- `models.yaml`: model profiles
- `vllm_manager.py`: vLLM lifecycle and dry-run
- `server.py`: FastAPI gateway
- `scripts/start.ps1`: Windows launcher
- `scripts/benchmark.py`: workload benchmark
