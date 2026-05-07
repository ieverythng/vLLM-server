#!/usr/bin/env python3
"""
FastAPI Inference Gateway — Unified OpenAI-compatible API layer.

Sits between clients (Hermes, iTrader, Discord, etc.) and vLLM.
Provides:
- OpenAI-compatible /v1/chat/completions endpoint
- Model routing per role/client
- Request logging and metrics
- Structured JSON mode support
- Rate limiting and authentication
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator
import httpx
import yaml
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# ─── Config ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.resolve()

with open(BASE_DIR / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

SERVER_CFG = CONFIG.get("server", {})
INFERENCE_CFG = CONFIG.get("inference", {})
ROLES_CFG = CONFIG.get("roles", {})
VLLM_CFG = CONFIG.get("vllm", {})

# ─── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, SERVER_CFG.get("log_level", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vllm-gateway")

# ─── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="vLLM Inference Gateway",
    description="Unified OpenAI-compatible inference layer for Hermes/Watson and iTrader",
    version="1.0.0",
)

# vLLM backend URL (direct — no proxy overhead for the actual inference)
VLLM_BASE_URL = f"http://127.0.0.1:{SERVER_CFG.get('port', 8000)}"

# ─── Auth ──────────────────────────────────────────────────────────────────

API_KEY = CONFIG.get("auth", {}).get("api_key", "") or None


def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify Bearer token if API key is configured."""
    if not API_KEY:
        return  # No auth required
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ─── Request/Response Models ──────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str | List[Dict[str, Any]]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=VLLM_CFG.get("default_model", "Qwen/Qwen3.5-27B"))
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str | Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    user: Optional[str] = None  # Client identifier (role name, etc.)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: UsageInfo


# ─── Metrics ──────────────────────────────────────────────────────────────

class RequestMetrics:
    """Track request metrics for benchmarking."""

    def __init__(self):
        self.requests: List[Dict[str, Any]] = []
        self.total_requests = 0
        self.total_tokens = 0

    def record(self, entry: Dict[str, Any]):
        self.requests.append(entry)
        self.total_requests += 1
        self.total_tokens += entry.get("total_tokens", 0)

    def reset(self):
        self.requests.clear()
        self.total_requests = 0
        self.total_tokens = 0

    def summary(self) -> Dict[str, Any]:
        if not self.requests:
            return {"total_requests": 0}
        latencies = [r["latency_ms"] for r in self.requests]
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        }


metrics = RequestMetrics()

# ─── Role-based routing ──────────────────────────────────────────────────

def get_role_params(user: Optional[str]) -> Dict[str, Any]:
    """Get sampling params for a role."""
    if user and user in ROLES_CFG:
        return ROLES_CFG[user]

    defaults = VLLM_CFG.get("default_sampling_params", {})
    return {
        "temperature": defaults.get("temperature", 0.7),
        "max_tokens": defaults.get("max_tokens", 8192),
    }


def build_chat_payload(request: ChatCompletionRequest, *, stream: bool = False) -> Dict[str, Any]:
    role_params = get_role_params(request.user)
    payload = {
        "model": request.model or VLLM_CFG.get("default_model"),
        "messages": [message.model_dump(exclude_none=True) for message in request.messages],
        "temperature": request.temperature if request.temperature is not None else role_params.get("temperature", 0.7),
        "max_tokens": request.max_tokens if request.max_tokens is not None else role_params.get("max_tokens", 8192),
    }
    if stream:
        payload["stream"] = True
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop is not None:
        payload["stop"] = request.stop
    if request.tools is not None:
        payload["tools"] = request.tools
    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice
    if request.response_format is not None:
        payload["response_format"] = request.response_format
    return payload


# ─── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/v1/models")
            vllm_ok = resp.status_code == 200
    except Exception:
        vllm_ok = False

    return {
        "status": "healthy" if vllm_ok else "degraded",
        "vllm_connected": vllm_ok,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics.summary(),
    }


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """List available models (proxied from vLLM)."""
    verify_api_key(authorization)

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{VLLM_BASE_URL}/v1/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    auth_header: Optional[str] = Header(None),
):
    """OpenAI-compatible chat completions endpoint."""
    verify_api_key(auth_header)

    start_time = time.time()
    payload = build_chat_payload(request)

    async with httpx.AsyncClient(timeout=INFERENCE_CFG.get("timeout_s", 120)) as client:
        try:
            resp = await client.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            latency_ms = (time.time() - start_time) * 1000

            if resp.status_code != 200:
                logger.error(f"vLLM error: {resp.status_code} — {resp.text[:200]}")
                raise HTTPException(status_code=resp.status_code, detail=resp.text)

            result = resp.json()

            usage = result.get("usage", {})
            metrics.record({
                "timestamp": datetime.now().isoformat(),
                "model": payload["model"],
                "user": request.user or "unknown",
                "latency_ms": latency_ms,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })

            logger.info(
                f"Request completed: model={payload['model']}, "
                f"user={request.user or 'unknown'}, "
                f"latency={latency_ms:.0f}ms, "
                f"tokens={usage.get('total_tokens', '?')}"
            )

            if request.stream:
                return StreamingResponse(
                    _stream_response(result),
                    media_type="text/event-stream",
                )

            return JSONResponse(content=result)

        except httpx.TimeoutException:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Request timed out after {latency_ms:.0f}ms")
            raise HTTPException(status_code=504, detail="vLLM request timed out")


async def _stream_response(result: Dict) -> AsyncGenerator[str, None]:
    """Stream SSE response."""
    if "choices" in result:
        for choice in result["choices"]:
            delta = choice.get("delta", {})
            chunk = {
                "id": result.get("id", ""),
                "object": "chat.completion.chunk",
                "created": result.get("created", int(time.time())),
                "model": result.get("model", ""),
                "choices": [{"index": 0, "delta": delta}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(
    request: ChatCompletionRequest,
    auth_header: Optional[str] = Header(None),
):
    """Streaming chat completions — directly streams from vLLM."""
    verify_api_key(auth_header)

    ttfb_start = time.time()
    payload = build_chat_payload(request, stream=True)

    async def stream_generator():
        async with httpx.AsyncClient(timeout=INFERENCE_CFG.get("timeout_s", 120)) as client:
            async with client.stream(
                "POST",
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    logger.error(f"vLLM stream error: {resp.status_code} — {error_text[:200]}")
                    yield f"data: {{\"error\": \"{error_text.decode()[:100]}\"}}\n\n"
                    return

                logged_ttfb = False
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
                        if not logged_ttfb and line != "data: [DONE]":
                            ttfb_ms = (time.time() - ttfb_start) * 1000
                            logger.info(f"Time to first byte: {ttfb_ms:.0f}ms")
                            logged_ttfb = True

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/metrics")
async def get_metrics():
    """Get request metrics summary."""
    return metrics.summary()


@app.delete("/metrics")
async def reset_metrics():
    """Reset metrics counters."""
    metrics.reset()
    return {"status": "metrics reset"}


# ─── iTrader-specific endpoints ──────────────────────────────────────────

@app.post("/v1/itrader/generate")
async def itrader_generate(
    request: Request,
    auth_header: Optional[str] = Header(None),
):
    """
    Specialized endpoint for iTrader proposer generation.
    Accepts a structured prompt and returns validated JSON tasks.
    """
    verify_api_key(auth_header)

    body = await request.json()
    prompt = body.get("prompt", "")
    model_override = body.get("model")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a market analysis assistant. Generate trading tasks as JSON. "
                "Always respond with valid JSON matching the requested schema."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    response_format = body.get("response_format", {"type": "json_object"})

    async with httpx.AsyncClient(timeout=INFERENCE_CFG.get("timeout_s", 120)) as client:
        payload = {
            "model": model_override or VLLM_CFG.get("default_model"),
            "messages": messages,
            "temperature": body.get("temperature", 0.5),
            "max_tokens": body.get("max_tokens", 2048),
            "response_format": response_format,
        }

        resp = await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        result = resp.json()
        return JSONResponse(content={
            "status": "success",
            "model": payload["model"],
            "generation": result,
        })


# ─── Startup ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info(f"vLLM Gateway starting on {SERVER_CFG.get('host', '0.0.0.0')}:{SERVER_CFG.get('port', 8000)}")
    logger.info(f"vLLM backend: {VLLM_BASE_URL}")
    logger.info(f"Default model: {VLLM_CFG.get('default_model')}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/v1/models")
            if resp.status_code == 200:
                models = [m["id"] for m in resp.json()]
                logger.info(f"vLLM connected — models: {models}")
            else:
                logger.warning(f"vLLM not responding (status {resp.status_code})")
    except Exception as e:
        logger.warning(f"vLLM not reachable at startup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=SERVER_CFG.get("host", "0.0.0.0"),
        port=SERVER_CFG.get("port", 8000),
        log_level=SERVER_CFG.get("log_level", "info"),
    )
