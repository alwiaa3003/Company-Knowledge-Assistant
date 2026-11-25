"""Run one question through an ADK agent programmatically and collect answer, sources and metrics.

Run loop per https://adk.dev/runtime/ and /events/: InMemorySessionService + Runner, then iterate
`runner.run_async(...)` events. One fresh session per question unless a session id is passed (chat).
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from kb.config import load_settings
from kl_agents import telemetry
from kl_agents.builders import FINAL_KEY, build_hierarchy, build_hierarchy_no_reflection, build_single_agent

APP_NAME = "knowledge_lens"
USER_ID = "local_user"
# [file p.N] and also multi-page forms the model sometimes writes: [file p. 5, p. 9] / [file pp. 5, 9]
_CIT = re.compile(r"\[([^\[\]]+?\.(?:pdf|txt|md))\s*,?\s*"
                  r"(pp?\.?\s*\d+(?:\s*(?:,|;|&|and)\s*(?:pp?\.?\s*)?\d+)*)\s*\]", re.IGNORECASE)
_TABLE_SRC = re.compile(r"(employees\.csv \(\d+ rows? matched\))")


def build_agent(system: str):
    """'hierarchy' (coordinator + specialists + critic), 'hierarchy_noreflect' (no critic) or
    'single' (one agent, both tools)."""
    if system == "hierarchy":
        return build_hierarchy()
    if system == "hierarchy_noreflect":
        return build_hierarchy_no_reflection()
    if system == "single":
        return build_single_agent()
    raise ValueError(f"unknown system {system!r}")


def extract_sources(answer: str) -> list[dict[str, Any]]:
    """Citations in the final answer: [file p.N] and employees.csv (N rows matched)."""
    out: list[dict[str, Any]] = []
    for f, pages in _CIT.findall(answer):
        for p in re.findall(r"\d+", pages):
            item = {"file": f.strip(), "page": int(p)}
            if item not in out:
                out.append(item)
    for label in _TABLE_SRC.findall(answer):
        item = {"file": "employees.csv", "label": label}
        if item not in out:
            out.append(item)
    return out


class AgentSession:
    """A Runner + session service around one agent; reuse it for a chat (same session)."""

    def __init__(self, system: str = "hierarchy") -> None:
        self.system = system
        self.settings = load_settings()
        self.agent = build_agent(system)
        self.session_service = InMemorySessionService()
        self.runner = Runner(agent=self.agent, app_name=APP_NAME, session_service=self.session_service)

    async def new_session_id(self) -> str:
        """Create a new ADK session and return its id."""
        sid = uuid.uuid4().hex
        await self.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=sid)
        return sid

    async def ask(self, question: str, session_id: str | None = None) -> dict[str, Any]:
        """Run one question; returns {answer, sources, route, agents, tools, metrics, ...}."""
        sid = session_id or await self.new_session_id()
        rec, token = telemetry.start_request(question, self.system)
        status, last_text, error = "ok", "", None
        try:
            msg = types.Content(role="user", parts=[types.Part(text=question)])
            async for event in self.runner.run_async(user_id=USER_ID, session_id=sid, new_message=msg):
                if event.error_code:
                    status, error = "error", f"{event.error_code}: {event.error_message}"
                if event.content and event.content.parts and not event.partial:
                    text = "".join(p.text for p in event.content.parts if p.text and not p.thought)
                    if text.strip() and event.author != "user":
                        last_text = text.strip()
            session = await self.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=sid)
            answer = str((session.state.get(FINAL_KEY) if session else None) or last_text).strip()
            if not answer and status == "ok":
                status = "empty"
        except Exception as exc:  # report, never crash the CLI/eval
            status, error, answer = "error", f"{type(exc).__name__}: {exc}", ""
        finally:
            telemetry.end_request(token)
        record = telemetry.summarize(rec, self.settings.price_in_per_m, self.settings.price_out_per_m,
                                     status, answer)
        if error:
            record["errors"] = record["errors"] + [error]
        telemetry.write_logs(self.settings.logs_dir, record, rec)
        specialists = [a for a in ("docs_agent", "table_agent") if a in rec.agents]
        tool_names = {t["name"] for t in rec.tools}
        route = sorted({"docs" if n == "search_documents" else "table" for n in tool_names
                        if n in ("search_documents", "query_table")})
        return {**record, "sources": extract_sources(answer), "route": route,
                "specialists": specialists, "session_id": sid, "tool_calls": rec.tools}
