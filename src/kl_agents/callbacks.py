"""ADK callbacks: rate limiting, metrics and tool bookkeeping.

Signatures follow https://adk.dev/callbacks/types-of-callbacks/ - ADK passes arguments BY KEYWORD,
so the parameter names (callback_context, llm_request, llm_response, tool, args, tool_context,
tool_response) must be exactly these. Returning None means "continue normally".
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool, ToolContext

from kb.config import load_settings
from kb.llm import get_limiter
from kl_agents import telemetry


def _model_key(ctx: CallbackContext) -> tuple:
    """Identify an in-flight model call; branch differs between parallel sub-agents."""
    return (ctx.invocation_id, ctx.agent_name, getattr(ctx, "branch", None))


def before_agent(callback_context: CallbackContext) -> Optional[Any]:
    """Record which agent runs, in order."""
    rec = telemetry.current()
    if rec is not None:
        rec.agents.append(callback_context.agent_name)
    return None


async def before_model(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """Pace model calls with the SAME limiter kb uses (GEN_RPM), then start the call timer.

    Free tier: spacing calls avoids most 429s; retries for the rest are configured on the model
    (HttpRetryOptions). We await instead of time.sleep so parallel sub-agents don't block the loop.
    """
    settings = load_settings()
    delay = get_limiter("gen", settings.gen_rpm).reserve()
    if delay > 0:
        await asyncio.sleep(delay)
    rec = telemetry.current()
    if rec is not None:
        rec.pending_model[_model_key(callback_context)] = time.perf_counter()
    return None


def after_model(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """Stop the call timer and record token usage (thinking tokens count as output)."""
    rec = telemetry.current()
    if rec is None:
        return None
    start = rec.pending_model.pop(_model_key(callback_context), None)
    if start is None:  # e.g. a second (partial) response for the same call
        return None
    meta = llm_response.usage_metadata
    tin = (meta.prompt_token_count or 0) if meta else 0
    tout = ((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)) if meta else 0
    rec.model_calls.append(telemetry.ModelCall(callback_context.agent_name, start,
                                               time.perf_counter() - start, tin, tout))
    if llm_response.error_code:
        rec.errors.append(f"{callback_context.agent_name}: {llm_response.error_code} {llm_response.error_message}")
    return None


SPECIALISTS = ("docs_agent", "table_agent")
DELEGATIONS_KEY = "delegations"


def block_repeated_delegation(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> Optional[dict]:
    """Guardrail on the coordinator: the same specialist + same request twice in one question is refused.

    Found in the eval: for an unanswerable question the coordinator re-asked docs_agent the identical
    question, doubling the searches (17 LLM calls). Returning a dict from before_tool_callback skips
    the tool and gives the LLM this dict as the result (https://adk.dev/callbacks/types-of-callbacks/).
    """
    if tool.name not in SPECIALISTS:
        return None
    request = str(args.get("request", args)).strip().lower()
    seen = tool_context.state.get(DELEGATIONS_KEY) or {}
    if seen.get("invocation_id") != tool_context.invocation_id:
        seen = {"invocation_id": tool_context.invocation_id, "calls": []}
    key = f"{tool.name}|{request}"
    if key in seen["calls"]:
        rec = telemetry.current()
        if rec is not None:
            rec.errors.append(f"blocked repeated delegation to {tool.name}")
        return {"status": "error", "message": f"{tool.name} already answered this exact request in this turn. "
                "Use its earlier answer; if it said it doesn't know, report that."}
    tool_context.state[DELEGATIONS_KEY] = {"invocation_id": seen["invocation_id"], "calls": seen["calls"] + [key]}
    return None


def before_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> Optional[dict]:
    """Start the tool timer."""
    rec = telemetry.current()
    if rec is not None:
        rec.pending_tool[tool_context.function_call_id or tool.name] = time.perf_counter()
    return None


def after_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
               tool_response: Any) -> Optional[dict]:
    """Record tool timing/status; move kb's internal LLM usage out of the response into metrics.

    Returning a dict replaces the tool result the LLM sees, so we hand back the response without
    the private `_llm_usage` field (no point spending tokens on it).
    """
    rec = telemetry.current()
    resp = dict(tool_response) if isinstance(tool_response, dict) else tool_response
    usage = resp.pop("_llm_usage", None) if isinstance(resp, dict) else None
    if rec is not None:
        start = rec.pending_tool.pop(tool_context.function_call_id or tool.name, time.perf_counter())
        status = resp.get("status") if isinstance(resp, dict) else None
        rec.tools.append({"name": tool.name, "agent": tool_context.agent_name, "start_s": rec.rel(start),
                          "duration_s": round(time.perf_counter() - start, 3), "status": status,
                          "args": {k: v for k, v in args.items() if isinstance(v, (str, int, float))}})
        if usage and usage.get("calls"):
            rec.model_calls.append(telemetry.ModelCall(
                f"{tool.name}", start, float(usage.get("latency_s", 0.0)),
                int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), source="kb_tool"))
    return resp if isinstance(resp, dict) else None
