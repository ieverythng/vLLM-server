#!/usr/bin/env python3
"""Benchmark the vLLM inference gateway with realistic workloads.

Measures:
- Single-request and concurrent chat latency
- Time to first token (streaming)
- Completion/token throughput
- Structured JSON validity and iTrader-style generation reliability
- Tool-call success rate

Usage:
  python scripts/benchmark.py [--model MODEL] [--concurrency N] [--requests N]
                              [--output PATH] [--base-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Tuple

import httpx
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"
DEFAULT_REPORT_DIR = ROOT_DIR / "reports"


LONG_HERMES_SYSTEM_PROMPT = (
    "You are Hermes, a precise, reliable, and efficient coding and reasoning assistant. "
    "Follow the user's instructions exactly, think carefully, and provide concise but complete answers. "
    "When using tools, choose the minimal necessary call, validate the returned data, and explain assumptions. "
    "You should be strong at code review, debugging, planning, and structured analysis. "
    "If the task involves market analysis, trading workflows, or iTrader-style output, respond with valid JSON when requested. "
    "Never fabricate tool results. If uncertain, say so. Preserve any schema requirements exactly."
)

HERMES_USER_PROMPTS = [
    "Summarize the tradeoffs of prefix caching, tensor parallelism, and AWQ quantization in a gateway deployment.",
    "Draft a concise rollout plan for validating a vLLM gateway before production traffic is enabled.",
    "Explain how to debug a JSON schema failure in an OpenAI-compatible inference layer.",
    "Analyze this workload: 64k context, concurrent requests, long system prompt, and tool calling. Identify bottlenecks and mitigations.",
]

JSON_PROMPTS = [
    "Return a JSON object with keys: symbol, side, confidence, reason, risk_limit. Keep values short.",
    "Produce a JSON object for a trade idea with fields: symbol, timeframe, thesis, entry, stop_loss, take_profit.",
    "Return valid JSON only. Include fields: name, score, tags, summary. Use simple string values.",
]

ITRADER_PROMPTS = [
    "Generate one compact market-analysis task as JSON with fields: symbol, action, rationale, priority.",
    "Create a short trading workflow payload in JSON for a volatile large-cap stock.",
    "Return a minimal JSON task object for an iTrader-style proposal.",
]

TOOL_DESCRIPTION = (
    "Lookup current market metadata for a symbol. Use this tool when the user asks for a symbol-specific decision."
)

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "lookup_market_data",
            "description": TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "metric": {
                        "type": "string",
                        "enum": ["price", "volume", "volatility", "news"],
                    },
                },
                "required": ["symbol", "metric"],
                "additionalProperties": False,
            },
        },
    }
]


@dataclass
class RequestResult:
    ok: bool
    latency_ms: float
    ttft_ms: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    json_valid: bool = False
    tool_called: bool = False
    tool_name: Optional[str] = None
    error: Optional[str] = None

    @property
    def throughput_tok_s(self) -> float:
        seconds = self.latency_ms / 1000.0
        if seconds <= 0:
            return 0.0
        tokens = self.completion_tokens or self.total_tokens
        return tokens / seconds if tokens else 0.0


@dataclass
class SummaryStats:
    count: int
    ok: int
    failed: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    avg_ttft_ms: Optional[float]
    avg_throughput_tok_s: float
    json_valid_rate: Optional[float] = None
    tool_call_rate: Optional[float] = None


@dataclass
class BenchmarkConfig:
    base_url: str
    model: str
    concurrency: int
    num_requests: int
    timeout_s: float
    config_path: Path


class BenchmarkError(RuntimeError):
    pass


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    if value.endswith("/v1"):
        return value[:-3]
    return value


def get_default_base_url(cfg: Dict[str, Any]) -> str:
    inference = cfg.get("inference", {}) if isinstance(cfg, dict) else {}
    base_url = inference.get("base_url", "http://localhost:8000")
    return normalize_base_url(str(base_url))


def get_default_model(cfg: Dict[str, Any]) -> str:
    vllm = cfg.get("vllm", {}) if isinstance(cfg, dict) else {}
    return str(vllm.get("default_model", "Qwen/Qwen3.5-27B"))


def get_default_timeout(cfg: Dict[str, Any]) -> float:
    inference = cfg.get("inference", {}) if isinstance(cfg, dict) else {}
    return float(inference.get("timeout_s", 120))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[int(idx)]
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def extract_assistant_content(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return compact_json(content)


def extract_tool_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = payload.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            # Remove opening fence line and optional language tag.
            lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
    return stripped


def try_parse_json(text: str) -> Tuple[bool, Optional[Any]]:
    raw = strip_code_fences(text)
    try:
        return True, json.loads(raw)
    except Exception:
        return False, None


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Tuple[Dict[str, Any], float]:
    start = time.perf_counter()
    resp = await client.post(url, json=payload, timeout=timeout_s)
    latency_ms = (time.perf_counter() - start) * 1000.0
    resp.raise_for_status()
    return resp.json(), latency_ms


async def stream_ttft(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> RequestResult:
    start = time.perf_counter()
    first_token_ms: Optional[float] = None
    text_fragments: List[str] = []
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000.0
                try:
                    obj = json.loads(data)
                    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                    frag = delta.get("content")
                    if isinstance(frag, str):
                        text_fragments.append(frag)
                except Exception:
                    pass
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(ok=True, latency_ms=latency_ms, ttft_ms=first_token_ms)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(ok=False, latency_ms=latency_ms, ttft_ms=first_token_ms, error=str(exc))


async def bench_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: float,
    temperature: float = 0.2,
    max_tokens: int = 512,
    stream: bool = False,
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    user: Optional[str] = None,
) -> RequestResult:
    url = f"{base_url}/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if user is not None:
        payload["user"] = user

    if stream:
        return await stream_ttft(client, f"{base_url}/v1/chat/completions/stream", payload, timeout_s)

    start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        data = resp.json()
        text = extract_assistant_content(data)
        usage = data.get("usage") or {}
        result = RequestResult(
            ok=True,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )
        json_ok, _ = try_parse_json(text)
        result.json_valid = json_ok
        tool_calls = extract_tool_calls(data)
        if tool_calls:
            result.tool_called = True
            first = tool_calls[0] or {}
            function = first.get("function") or {}
            result.tool_name = function.get("name") if isinstance(function, dict) else None
        return result
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(ok=False, latency_ms=latency_ms, error=str(exc))


async def bench_itrader_generate(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    timeout_s: float,
    max_tokens: int = 256,
) -> RequestResult:
    url = f"{base_url}/v1/itrader/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "num_tasks": 1,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        data = resp.json()
        generation = data.get("generation") or {}
        text = extract_assistant_content(generation)
        usage = generation.get("usage") or {}
        result = RequestResult(
            ok=True,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )
        json_ok, _ = try_parse_json(text)
        result.json_valid = json_ok
        return result
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(ok=False, latency_ms=latency_ms, error=str(exc))


async def run_concurrent(
    factories: Iterable[Awaitable[RequestResult]],
    concurrency: int,
) -> List[RequestResult]:
    """Run awaitables with a concurrency cap and preserve completion order."""
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(awaitable: Awaitable[RequestResult]):
        async with semaphore:
            return await awaitable

    tasks = [asyncio.create_task(guarded(factory)) for factory in factories]
    results: List[RequestResult] = []
    for task in asyncio.as_completed(tasks):
        results.append(await task)
    return results


def summarize(results: List[RequestResult]) -> SummaryStats:
    latencies = [r.latency_ms for r in results]
    oks = [r for r in results if r.ok]
    ttfts = [r.ttft_ms for r in results if r.ttft_ms is not None]
    throughput = [r.throughput_tok_s for r in oks if r.throughput_tok_s > 0]
    json_valid_rate = None
    tool_call_rate = None
    if results:
        json_valid_rate = sum(1 for r in results if r.ok and r.json_valid) / len(results)
        tool_call_rate = sum(1 for r in results if r.ok and r.tool_called) / len(results)
    return SummaryStats(
        count=len(results),
        ok=len(oks),
        failed=len(results) - len(oks),
        avg_ms=statistics.fmean(latencies) if latencies else 0.0,
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        min_ms=min(latencies) if latencies else 0.0,
        max_ms=max(latencies) if latencies else 0.0,
        avg_ttft_ms=statistics.fmean(ttfts) if ttfts else None,
        avg_throughput_tok_s=statistics.fmean(throughput) if throughput else 0.0,
        json_valid_rate=json_valid_rate,
        tool_call_rate=tool_call_rate,
    )


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_tps(value: float) -> str:
    return f"{value:.2f}"


def render_section_table(title: str, stats: SummaryStats) -> str:
    return (
        f"### {title}\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Requests | {stats.count} |\n"
        f"| Success | {stats.ok} |\n"
        f"| Failures | {stats.failed} |\n"
        f"| Avg latency (ms) | {stats.avg_ms:.1f} |\n"
        f"| P50 latency (ms) | {stats.p50_ms:.1f} |\n"
        f"| P95 latency (ms) | {stats.p95_ms:.1f} |\n"
        f"| Min / Max latency (ms) | {stats.min_ms:.1f} / {stats.max_ms:.1f} |\n"
        f"| Avg TTFT (ms) | {fmt_ms(stats.avg_ttft_ms)} |\n"
        f"| Avg throughput (tok/s) | {fmt_tps(stats.avg_throughput_tok_s)} |\n"
        f"| JSON validity rate | {fmt_rate(stats.json_valid_rate)} |\n"
        f"| Tool-call success rate | {fmt_rate(stats.tool_call_rate)} |\n"
    )


def render_failure_samples(results: List[RequestResult], limit: int = 5) -> str:
    failures = [r for r in results if not r.ok]
    if not failures:
        return "None."
    lines = []
    for idx, failure in enumerate(failures[:limit], 1):
        lines.append(f"{idx}. latency={failure.latency_ms:.1f}ms error={failure.error or 'unknown'}")
    return "\n".join(lines)


def render_report(
    cfg: BenchmarkConfig,
    model: str,
    single: SummaryStats,
    concurrent: SummaryStats,
    json_stats: SummaryStats,
    itrader_stats: SummaryStats,
    tool_stats: SummaryStats,
    single_ttft: Optional[float],
    concurrent_ttft: Optional[float],
    json_samples: List[RequestResult],
    itrader_samples: List[RequestResult],
    tool_samples: List[RequestResult],
) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        f"# vLLM Gateway Benchmark Report ({date.today().isoformat()})",
        "",
        f"Generated: {now}",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---:|",
        f"| Base URL | {cfg.base_url} |",
        f"| Model | {model} |",
        f"| Concurrency | {cfg.concurrency} |",
        f"| Requests | {cfg.num_requests} |",
        f"| Timeout (s) | {cfg.timeout_s:.0f} |",
        "",
        "## Workloads",
        "",
        "- Hermes-style long system prompt chat completions",
        "- Hermes-style streaming chat completion for time-to-first-token measurement",
        "- iTrader-style short structured JSON generation via /v1/itrader/generate",
        "- JSON reliability checks using OpenAI-compatible response_format=json_object",
        "- Tool-calling reliability checks with a forced function schema",
        "",
        render_section_table("Hermes chat completion: single request", single),
        f"Single-request TTFT (ms): {fmt_ms(single_ttft)}",
        "",
        render_section_table("Hermes chat completion: concurrent run", concurrent),
        f"Concurrent TTFT (ms): {fmt_ms(concurrent_ttft)}",
        "",
        render_section_table("Structured JSON reliability", json_stats),
        "",
        render_section_table("iTrader-style generation", itrader_stats),
        "",
        render_section_table("Tool-call reliability", tool_stats),
        "",
        "## Notes",
        "",
        "- Throughput is computed from returned token counts divided by end-to-end latency.",
        "- JSON validity is determined by strict json.loads() parsing of the assistant payload.",
        "- Tool-call success requires a tool_calls entry with the expected tool name.",
        "",
        "## Failure samples",
        "",
        "### JSON workload failures",
        render_failure_samples(json_samples),
        "",
        "### iTrader workload failures",
        render_failure_samples(itrader_samples),
        "",
        "### Tool-call workload failures",
        render_failure_samples(tool_samples),
        "",
    ]
    return "\n".join(lines)


async def run_benchmark(cfg: BenchmarkConfig) -> Tuple[str, Dict[str, SummaryStats]]:
    limits = httpx.Limits(max_keepalive_connections=max(10, cfg.concurrency), max_connections=max(20, cfg.concurrency * 2))
    timeout = httpx.Timeout(cfg.timeout_s, connect=min(10.0, cfg.timeout_s))
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Single Hermes-style request
        single_result = await bench_chat_completion(
            client,
            cfg.base_url,
            cfg.model,
            LONG_HERMES_SYSTEM_PROMPT,
            HERMES_USER_PROMPTS[0],
            cfg.timeout_s,
            temperature=0.2,
            max_tokens=512,
            stream=False,
            user="WatsonMain",
        )

        # Streaming TTFT for the same style of workload
        single_ttft_result = await bench_chat_completion(
            client,
            cfg.base_url,
            cfg.model,
            LONG_HERMES_SYSTEM_PROMPT,
            HERMES_USER_PROMPTS[0],
            cfg.timeout_s,
            temperature=0.2,
            max_tokens=256,
            stream=True,
            user="WatsonMain",
        )

        # Concurrent Hermes-style workload with a cap
        concurrent_results = await run_concurrent(
            (
                bench_chat_completion(
                    client,
                    cfg.base_url,
                    cfg.model,
                    LONG_HERMES_SYSTEM_PROMPT,
                    HERMES_USER_PROMPTS[i % len(HERMES_USER_PROMPTS)],
                    cfg.timeout_s,
                    temperature=0.2,
                    max_tokens=512,
                    stream=False,
                    user="WatsonMain",
                )
                for i in range(cfg.num_requests)
            ),
            cfg.concurrency,
        )

        # Streaming TTFT under concurrency: run a smaller set using the same concurrency cap
        ttft_count = max(1, min(cfg.concurrency, cfg.num_requests))
        ttft_results = await run_concurrent(
            (
                bench_chat_completion(
                    client,
                    cfg.base_url,
                    cfg.model,
                    LONG_HERMES_SYSTEM_PROMPT,
                    HERMES_USER_PROMPTS[i % len(HERMES_USER_PROMPTS)],
                    cfg.timeout_s,
                    temperature=0.2,
                    max_tokens=256,
                    stream=True,
                    user="WatsonMain",
                )
                for i in range(ttft_count)
            ),
            cfg.concurrency,
        )

        # Structured JSON reliability on chat completions
        json_results = await run_concurrent(
            (
                bench_chat_completion(
                    client,
                    cfg.base_url,
                    cfg.model,
                    "You output only valid JSON, no markdown, no prose.",
                    JSON_PROMPTS[i % len(JSON_PROMPTS)],
                    cfg.timeout_s,
                    temperature=0.0,
                    max_tokens=256,
                    stream=False,
                    response_format={"type": "json_object"},
                    user="iTrader",
                )
                for i in range(cfg.num_requests)
            ),
            cfg.concurrency,
        )

        # iTrader-style generation endpoint
        itrader_results = await run_concurrent(
            (
                bench_itrader_generate(
                    client,
                    cfg.base_url,
                    cfg.model,
                    ITRADER_PROMPTS[i % len(ITRADER_PROMPTS)],
                    cfg.timeout_s,
                )
                for i in range(cfg.num_requests)
            ),
            cfg.concurrency,
        )

        # Tool-calling reliability
        tool_prompt = (
            "For the symbol AAPL, call the lookup_market_data tool with metric price and then respond briefly."
        )
        tool_results = await run_concurrent(
            (
                bench_chat_completion(
                    client,
                    cfg.base_url,
                    cfg.model,
                    "You are a function-calling assistant. Prefer tool calls whenever available.",
                    tool_prompt,
                    cfg.timeout_s,
                    temperature=0.0,
                    max_tokens=256,
                    stream=False,
                    tools=TOOL_SPEC,
                    tool_choice={"type": "function", "function": {"name": "lookup_market_data"}},
                    user="WatsonDev",
                )
                for _ in range(cfg.num_requests)
            ),
            cfg.concurrency,
        )

    single_stats = summarize([single_result])
    concurrent_stats = summarize(concurrent_results)
    json_stats = summarize(json_results)
    itrader_stats = summarize(itrader_results)
    tool_stats = summarize(tool_results)

    # Set specialized reliability rates for report visibility.
    json_stats.json_valid_rate = sum(1 for r in json_results if r.ok and r.json_valid) / len(json_results) if json_results else None
    itrader_stats.json_valid_rate = sum(1 for r in itrader_results if r.ok and r.json_valid) / len(itrader_results) if itrader_results else None
    tool_stats.tool_call_rate = sum(1 for r in tool_results if r.ok and r.tool_called and r.tool_name == "lookup_market_data") / len(tool_results) if tool_results else None

    report = render_report(
        cfg=cfg,
        model=cfg.model,
        single=single_stats,
        concurrent=concurrent_stats,
        json_stats=json_stats,
        itrader_stats=itrader_stats,
        tool_stats=tool_stats,
        single_ttft=single_ttft_result.ttft_ms,
        concurrent_ttft=statistics.fmean([r.ttft_ms for r in ttft_results if r.ttft_ms is not None]) if ttft_results else None,
        json_samples=json_results,
        itrader_samples=itrader_results,
        tool_samples=tool_results,
    )

    return report, {
        "single": single_stats,
        "concurrent": concurrent_stats,
        "json": json_stats,
        "itrader": itrader_stats,
        "tool": tool_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the vLLM inference gateway.")
    parser.add_argument("--model", help="Model name to benchmark.")
    parser.add_argument("--concurrency", type=int, help="Concurrency level for concurrent tests.")
    parser.add_argument("--requests", type=int, help="Number of requests per workload.")
    parser.add_argument("--output", help="Path to the markdown report file.")
    parser.add_argument("--base-url", default=None, help="Gateway base URL (default: http://localhost:8000 or config.yaml).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml.")
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    cfg_data = load_config(Path(args.config))

    base_url = normalize_base_url(args.base_url or get_default_base_url(cfg_data))
    model = args.model or get_default_model(cfg_data)
    concurrency = max(1, int(args.concurrency or 4))
    num_requests = max(1, int(args.requests or 8))
    timeout_s = get_default_timeout(cfg_data)

    benchmark_cfg = BenchmarkConfig(
        base_url=base_url,
        model=model,
        concurrency=concurrency,
        num_requests=num_requests,
        timeout_s=timeout_s,
        config_path=Path(args.config),
    )

    report, _stats = await run_benchmark(benchmark_cfg)

    output_path = Path(args.output) if args.output else DEFAULT_REPORT_DIR / f"benchmark-{date.today().isoformat()}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Wrote benchmark report to {output_path}")
    print()
    print(report)
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
