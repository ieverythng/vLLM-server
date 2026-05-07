# vLLM Inference Gateway Topology

This repository now runs the inference stack as a two-tier service:

1. vLLM serves the model API on port 8000.
2. FastAPI serves the gateway on port 8001 and forwards requests to vLLM.

The launcher for both processes is `./start.sh`.

## Serving topology

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
 Discord, etc.)     |  /v1/itrader/generate |    |
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

Key idea: the gateway is a lightweight routing and policy layer. vLLM owns token generation, KV cache, sampling, and model loading.

## Environment variables

The stack is primarily configured through `config.yaml`, but `start.sh` also recognizes a few runtime environment variables:

- `GATEWAY_PORT` — gateway listen port, default `8001`
- `VLLM_TIMEOUT` — readiness timeout in seconds for `/v1/models`, default `300`

Relevant config-backed settings:

- `server.host` — vLLM host bind, typically `0.0.0.0`
- `server.port` — vLLM port, `8000`
- `server.log_level` — logging verbosity for the gateway
- `zerotier.network_id` — ZeroTier network membership ID
- `vllm.default_model` — default Hugging Face model identifier
- `vllm.quantization` — AWQ/GPTQ or other supported quantization mode
- `vllm.gpu_memory_utilization` — fraction of GPU memory allocated by vLLM
- `vllm.max_model_len` — context window size
- `vllm.dtype` — precision selection
- `vllm.tensor_parallel_size` — tensor parallel degree
- `vllm.enable_prefix_caching` — repeated prompt acceleration
- `vllm.default_sampling_params.*` — fallback generation defaults
- `roles.*` — role-specific sampling overrides
- `inference.base_url` — client-side API base URL for the gateway or vLLM endpoint
- `auth.api_key` or `VLLM_API_KEY` — optional bearer auth

## Launch commands

Use the launcher from the repository root:

```bash
./start.sh start
./start.sh status
./start.sh stop
./start.sh restart
```

What `start.sh` does:

1. Activates `venv/`
2. Confirms `nvidia-smi` is available
3. Starts vLLM through `vllm_manager.py start --no-wait`
4. Polls `http://127.0.0.1:8000/v1/models` until ready
5. Starts the FastAPI gateway on port `8001`
6. Prints process status and ZeroTier interface IPs

Direct process launch is still possible for debugging:

```bash
python vllm_manager.py start
python server.py
```

However, `server.py` will bind to the port in `config.yaml`. For the two-port deployment, prefer `start.sh`, which launches the gateway with Uvicorn on port `8001`.

## Model routing per role

Role routing is controlled by the `roles:` block in `config.yaml`.

| Role | Default model | Temperature | Max tokens | Notes |
| --- | --- | ---: | ---: | --- |
| WatsonMain | `default` | 0.7 | 8192 | General-purpose primary assistant |
| WatsonMemory | `default` | 0.3 | 4096 | Lower temperature for memory/recall style tasks |
| WatsonReviewer | `default` | 0.5 | 4096 | Review / critique workflow |
| WatsonDev | `default` | 0.7 | 8192 | Implementation and coding tasks |
| iTrader | `default` | 0.5 | 2048 | Structured market/task generation |

Routing flow:

- The gateway reads `request.user`
- If it matches a known role, the role defaults are applied
- Request-level overrides always win
- The request is then forwarded to vLLM unchanged except for the merged sampling parameters

## iTrader integration path

The gateway includes a dedicated endpoint:

- `POST /v1/itrader/generate`

This path is intended for structured generation used by the iTrader workflow.

Behavior:

- Accepts a JSON body with `prompt`, `num_tasks`, optional `model`, and optional `response_format`
- Wraps the prompt with a market-analysis system instruction
- Forces JSON-friendly generation by default
- Returns the raw vLLM result inside a `generation` envelope

Typical integration shape:

1. iTrader sends a structured prompt to the gateway
2. Gateway applies the iTrader sampling profile from `config.yaml`
3. vLLM generates the response on the shared model backend
4. iTrader parses the returned JSON payload into downstream tasks/orders

## Migration notes from llama.cpp to vLLM

The migration goal is to preserve the same application-facing API while changing the backend engine.

Main differences:

- llama.cpp typically served GGUF quantized models with a smaller feature surface
- vLLM serves OpenAI-compatible endpoints and adds stronger batching, KV cache management, and throughput scaling
- Role-based sampling stays in the gateway rather than in the backend engine

Operational changes:

- Model format may change from GGUF to HF weights or vLLM-supported quantized formats such as AWQ/GPTQ
- GPU memory planning is now handled by vLLM, not llama.cpp
- The gateway expects `/v1/models` and `/v1/chat/completions` from the backend
- Long-context support is configured via `vllm.max_model_len`

Recommended migration steps:

1. Start vLLM alongside the existing gateway path
2. Verify `/v1/models` and `/v1/chat/completions`
3. Compare output quality and latency against the old llama.cpp baseline
4. Switch `inference.base_url` to the new gateway endpoint if clients were previously pointed directly at llama.cpp
5. Retire the old llama.cpp launch path after parity is confirmed

## Rollback path to llama.cpp

Rollback is straightforward if you keep the old llama.cpp service or command available.

Rollback strategy:

1. Stop the vLLM stack: `./start.sh stop`
2. Restore the previous llama.cpp backend service or launch command on port `8000` or another OpenAI-compatible port
3. Point `inference.base_url` back to the llama.cpp endpoint
4. If needed, disable vLLM-specific config keys and restore GGUF model paths

Practical notes:

- Keep the previous llama.cpp startup command in version control or a runbook
- Preserve any model files or quantization artifacts needed by the legacy path
- If the client expects the same OpenAI schema, only the backend URL should need to change

## ZeroTier networking setup

ZeroTier allows the gateway to be reached across the private mesh network without exposing the service publicly.

Current network ID from `config.yaml`:

- `3b19b3a716937e29`

Typical setup checklist:

1. Join the host to the ZeroTier network
2. Approve the host in the ZeroTier controller if required
3. Confirm the host receives a `zt*` interface and an IPv4 address
4. Use the ZeroTier IP for remote clients when localhost access is not possible
5. Keep the gateway bound to `0.0.0.0` so the ZT interface can reach it

Useful endpoints over ZeroTier:

- Gateway: `http://<zerotier-ip>:8001`
- vLLM direct: `http://<zerotier-ip>:8000`

## Notes on ports and bindings

- vLLM remains on port `8000`
- The gateway is launched on port `8001`
- `start.sh` intentionally overrides the gateway port at launch time so the backend and front door do not collide

## Files involved

- `config.yaml` — central configuration
- `vllm_manager.py` — vLLM lifecycle management
- `server.py` — FastAPI gateway
- `start.sh` — one-command launcher for both services
