#!/usr/bin/env python3
"""Quick OpenAI-compatible vLLM benchmark for throughput and context-window checks.

Features:
- Auto-discovers served model ID from /v1/models (avoids model-name mismatch 404s)
- Measures latency, completion tok/s, request tok/s, and success rate
- Optional context sweep up to large prompt sizes (including ~128k-token class runs)

Examples:
  python scripts/benchmark_tokps.py --base-url http://127.0.0.1:8001 --auto-model
  python scripts/benchmark_tokps.py --base-url http://127.0.0.1:8001 --auto-model --concurrency 4 --requests 8
  python scripts/benchmark_tokps.py --base-url http://127.0.0.1:8001 --auto-model --context-sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RunResult:
    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str | None = None

    @property
    def completion_tps(self) -> float:
        return (self.completion_tokens / self.latency_s) if self.ok and self.latency_s > 0 else 0.0

    @property
    def request_tps(self) -> float:
        return (self.total_tokens / self.latency_s) if self.ok and self.latency_s > 0 else 0.0


def normalize_base_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


async def get_model_id(client: httpx.AsyncClient, base_url: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    resp = await client.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("No models returned by /v1/models")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError("First /v1/models entry missing id")
    return str(model_id)


def build_prompt(target_tokens: int) -> str:
    # Approximation: one short word per token-ish for tokenizer stress tests.
    # Keep deterministic shape for repeatability.
    if target_tokens <= 32:
        return "Count to ten and return only the number ten."
    body = " tok" * max(1, target_tokens - 32)
    return "Repeat-ready prompt:" + body


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> RunResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise benchmark responder."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    start = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
        latency = time.perf_counter() - start
        resp.raise_for_status()
        obj = resp.json()
        usage = obj.get("usage") or {}
        return RunResult(
            ok=True,
            latency_s=latency,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )
    except Exception as exc:
        return RunResult(ok=False, latency_s=time.perf_counter() - start, prompt_tokens=0, completion_tokens=0, total_tokens=0, error=str(exc))


async def run_batch(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    requests: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> list[RunResult]:
    sem = asyncio.Semaphore(concurrency)

    async def worker() -> RunResult:
        async with sem:
            return await one_request(client, base_url, model, prompt, max_tokens, temperature, timeout_s)

    tasks = [asyncio.create_task(worker()) for _ in range(requests)]
    return [await t for t in asyncio.as_completed(tasks)]


def summarize(results: list[RunResult]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    lat = [r.latency_s for r in ok]
    comp_tps = [r.completion_tps for r in ok if r.completion_tps > 0]
    req_tps = [r.request_tps for r in ok if r.request_tps > 0]
    return {
        "count": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "avg_latency_s": statistics.fmean(lat) if lat else None,
        "p50_latency_s": statistics.median(lat) if lat else None,
        "avg_completion_tps": statistics.fmean(comp_tps) if comp_tps else None,
        "avg_request_tps": statistics.fmean(req_tps) if req_tps else None,
        "avg_prompt_tokens": statistics.fmean([r.prompt_tokens for r in ok]) if ok else None,
        "avg_completion_tokens": statistics.fmean([r.completion_tokens for r in ok]) if ok else None,
        "errors": [r.error for r in results if (not r.ok and r.error)][:5],
    }


async def main_async() -> int:
    ap = argparse.ArgumentParser(description="vLLM throughput/context benchmark")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default=None)
    ap.add_argument("--auto-model", action="store_true", help="Resolve model from /v1/models")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--context-sweep", action="store_true")
    ap.add_argument("--output", default=None, help="Write JSON report")
    args = ap.parse_args()

    base_url = normalize_base_url(args.base_url)
    limits = httpx.Limits(max_keepalive_connections=max(8, args.concurrency), max_connections=max(16, args.concurrency * 2))
    timeout = httpx.Timeout(args.timeout, connect=min(20.0, args.timeout))

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        model = await get_model_id(client, base_url, args.model if not args.auto_model else None)

        print(f"base_url={base_url}")
        print(f"model={model}")

        short_prompt = "Explain in 5 bullets how to maximize throughput while keeping output quality stable."
        baseline = await run_batch(
            client,
            base_url,
            model,
            short_prompt,
            requests=max(1, args.requests),
            concurrency=max(1, args.concurrency),
            max_tokens=max(32, args.max_tokens),
            temperature=args.temperature,
            timeout_s=args.timeout,
        )
        baseline_summary = summarize(baseline)

        report: dict[str, Any] = {
            "base_url": base_url,
            "model": model,
            "baseline": baseline_summary,
            "context_sweep": [],
        }

        if args.context_sweep:
            targets = [2048, 8192, 32768, 65536, 98304, 122880]
            for target in targets:
                prompt = build_prompt(target)
                results = await run_batch(
                    client,
                    base_url,
                    model,
                    prompt,
                    requests=1,
                    concurrency=1,
                    max_tokens=128,
                    temperature=0.0,
                    timeout_s=max(args.timeout, 300.0),
                )
                s = summarize(results)
                s["target_prompt_tokens_approx"] = target
                report["context_sweep"].append(s)
                status = "OK" if s["ok"] else "FAIL"
                print(f"context~{target}: {status} ok={s['ok']} failed={s['failed']} avg_latency_s={s['avg_latency_s']}")

    print("\nBaseline summary:")
    print(json.dumps(baseline_summary, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
