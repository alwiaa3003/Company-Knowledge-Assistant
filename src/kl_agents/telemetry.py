"""Per-request metrics + local tracing.

Two complementary sources:
1. ADK callbacks (callbacks.py) record agents, tools and model calls with their timings and token
   usage into a RequestRecorder for the current request.
2. ADK emits OpenTelemetry spans (invoke_agent / execute_tool / generate_content ...,
   https://adk.dev/observability/traces/). We install an in-process SpanProcessor that copies
   finished spans into the same recorder, so we get a trace id and a span timeline without any
   external collector (free, local). The same spans could be exported to Cloud Trace in production.

The current recorder lives in a ContextVar, so concurrent requests (or parallel sub-agents, which
run as asyncio tasks that inherit the context) write to the right request.
"""

from __future__ import annotations

import contextvars
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider

_current: contextvars.ContextVar["RequestRecorder | None"] = contextvars.ContextVar("kl_recorder", default=None)
_installed = False


@dataclass
class ModelCall:
    """One LLM call made by an ADK agent (or by a kb tool internally)."""

    agent: str
    start: float
    duration_s: float
    input_tokens: int
    output_tokens: int
    source: str = "adk"  # "adk" = agent model call, "kb_tool" = LLM call inside a kb tool


@dataclass
class RequestRecorder:
    """Everything we learn about one question while it runs."""

    question: str
    system: str
    t0: float = field(default_factory=time.perf_counter)
    agents: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[ModelCall] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)
    pending_model: dict[tuple, float] = field(default_factory=dict)
    pending_tool: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def rel(self, t: float) -> float:
        """Seconds since the request started."""
        return round(t - self.t0, 3)


def current() -> RequestRecorder | None:
    """The recorder of the request running in this context (None e.g. under `adk web`)."""
    return _current.get()


def start_request(question: str, system: str) -> tuple[RequestRecorder, contextvars.Token]:
    """Begin recording a request; returns (recorder, token) - pass the token to end_request."""
    install_tracing()
    rec = RequestRecorder(question=question, system=system)
    return rec, _current.set(rec)


def end_request(token: contextvars.Token) -> None:
    """Stop recording (restores the previous context value)."""
    _current.reset(token)


class _RecorderSpanProcessor(SpanProcessor):
    """Copies every finished OpenTelemetry span into the current request's recorder."""

    def on_end(self, span: ReadableSpan) -> None:
        rec = current()
        if rec is None or span.start_time is None or span.end_time is None:
            return
        ctx = span.get_span_context()
        rec.spans.append({
            "name": span.name,
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
            "start_ns": span.start_time,
            "end_ns": span.end_time,
            "duration_ms": round((span.end_time - span.start_time) / 1e6, 1),
            "attrs": {k: v for k, v in (span.attributes or {}).items()
                      if isinstance(v, (str, int, float, bool)) and len(str(v)) < 200},
        })


def install_tracing() -> None:
    """Install a global TracerProvider with our in-process processor (once per process).

    If something else already installed an SDK TracerProvider, we add our processor to it.
    """
    global _installed
    if _installed:
        return
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(_RecorderSpanProcessor())
    _installed = True


def summarize(rec: RequestRecorder, price_in: float, price_out: float, status: str,
              answer: str) -> dict[str, Any]:
    """Turn a recorder into the JSON line we log per request.

    tokens_per_sec = total output tokens / total model-call time, where model-call time is measured
    from before_model_callback (after our rate-limit wait) to after_model_callback. Calls are not
    streamed, so this is end-to-end throughput per call (network + queueing + decoding), not pure
    decoding speed. Output tokens include thinking tokens (billed as output).
    """
    total_latency = time.perf_counter() - rec.t0
    tin = sum(m.input_tokens for m in rec.model_calls)
    tout = sum(m.output_tokens for m in rec.model_calls)
    gen_time = sum(m.duration_s for m in rec.model_calls)
    trace_ids = sorted({s["trace_id"] for s in rec.spans})
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": rec.system,
        "question": rec.question,
        "status": status,
        "answer": answer,
        "agents": rec.agents,
        "tools": [t["name"] for t in rec.tools],
        "llm_calls": len(rec.model_calls),
        "llm_calls_by_agent": _count_by_agent(rec.model_calls),
        "input_tokens": tin,
        "output_tokens": tout,
        "latency_s": round(total_latency, 3),
        "generation_time_s": round(gen_time, 3),
        "tokens_per_sec": round(tout / gen_time, 1) if gen_time > 0 else None,
        "est_cost_usd": round((tin * price_in + tout * price_out) / 1_000_000, 8),
        "trace_id": trace_ids[0] if trace_ids else None,
        "errors": rec.errors,
    }


def _count_by_agent(calls: list[ModelCall]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in calls:
        key = m.agent if m.source == "adk" else f"{m.agent} (kb tool)"
        out[key] = out.get(key, 0) + 1
    return out


def write_logs(logs_dir: Path, record: dict[str, Any], rec: RequestRecorder) -> None:
    """Append the request record and its full timeline to logs/ (git-ignored)."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "agent_requests.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    timeline = {
        "trace_id": record["trace_id"], "ts": record["ts"], "system": rec.system, "question": rec.question,
        "latency_s": record["latency_s"],
        "model_calls": [{"agent": m.agent, "source": m.source, "start_s": rec.rel(m.start),
                         "duration_s": round(m.duration_s, 3), "input_tokens": m.input_tokens,
                         "output_tokens": m.output_tokens} for m in rec.model_calls],
        "tools": rec.tools,
        "spans": rec.spans,
    }
    with (logs_dir / "agent_traces.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(timeline, ensure_ascii=False) + "\n")
