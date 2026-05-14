# ZeroTier Routing Runbook (Main vLLM Server)

This machine can act as the main inference server for anyone on the ZeroTier network.

Topology:

- vLLM backend: `127.0.0.1:8000`
- Gateway API: `0.0.0.0:8001`
- ZeroTier clients call: `http://<server-zerotier-ip>:8001/v1`

Network ID configured in `config.yaml`:

- `3b19b3a716937e29`

## 1) Start the stack (WSL-first, recommended)

From Windows PowerShell in repo root:

```powershell
.\scripts\start_wsl.ps1 preflight
.\scripts\start_wsl.ps1 dry-run
.\scripts\start_wsl.ps1 start
```

Check status:

```powershell
.\scripts\start_wsl.ps1 status
```

Stop:

```powershell
.\scripts\start_wsl.ps1 stop
```

Why WSL-first: the Linux environment reliably includes `vllm._C` runtime while native Windows Python has been flaky for this workload.

## 2) Start gateway service (if not started separately)

In WSL terminal from repo root:

```bash
/home/juanbeck/vLLM-server/venv/bin/python -m uvicorn server:app \
  --host 0.0.0.0 --port 8001 --loop asyncio --log-level info
```

Health checks:

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8001/health
curl -sS http://127.0.0.1:8001/v1/models
```

## 3) Find the server ZeroTier IP

On the server:

```bash
zerotier-cli listnetworks
ip -4 addr show | grep -E 'zt|10\.|172\.|192\.'
```

Use the ZeroTier-assigned IP in client URLs (example):

- `http://10.88.140.135:8001/v1`

## 4) Client configuration

Set clients to use gateway endpoint:

- Base URL: `http://<server-zerotier-ip>:8001/v1`
- API key: only if enabled in server `config.yaml` (`auth.api_key` / `VLLM_API_KEY`)

Example OpenAI-compatible test:

```bash
curl -sS http://<server-zerotier-ip>:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role":"user","content":"Reply with OK"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

## 5) Windows firewall and reachability

If clients cannot reach `:8001`:

1. Ensure gateway is bound to `0.0.0.0`.
2. Allow inbound TCP `8001` in Windows firewall.
3. Confirm ZeroTier network authorization is complete for server and clients.
4. Test from another node:

```bash
curl -v http://<server-zerotier-ip>:8001/health
```

## 6) Throughput + context profile guidance

Current high-context profile:

- `qwen35_9b_awq_4bit_128k`
- `max_model_len: 131072`

For stability under multi-client load:

- keep `max_num_seqs` conservative first,
- monitor VRAM with `nvidia-smi`,
- increase concurrency only after sustained healthy runs.

## 7) Troubleshooting quick checks

- Runtime probe:

```bash
/home/juanbeck/vLLM-server/venv/bin/python vllm_manager.py preflight
```

- Confirm served model IDs:

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

- Confirm gateway routing:

```bash
curl -sS http://127.0.0.1:8001/v1/models
```

If `/v1/chat/completions` returns model-not-found via gateway, verify `server.py` resolves `prefer_local_path` model IDs (WSL path mapping).
