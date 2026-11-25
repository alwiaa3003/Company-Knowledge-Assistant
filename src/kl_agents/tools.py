"""ADK function tools: thin, JSON-friendly wrappers over the kb package.

ADK turns each function into a tool automatically: the function name becomes the tool name, the
docstring becomes the description the LLM reads, and type hints + defaults become the schema
(https://adk.dev/tools-custom/function-tools/). A parameter typed `ToolContext` is injected by ADK
and hidden from the LLM.

Rules followed here:
- never raise into the agent: errors come back as {"status": "error", "message": ...}
- every result has a "status" key (ADK best practice)
- tools record what they found in session state so the critic can check the draft against it
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from kb.config import load_settings
from kb.retrieve import retrieve
from kb.structured import query_table as kb_query_table

EVIDENCE_KEY = "evidence"  # list of evidence items for the current question (reset per request)
MAX_CHUNK_CHARS = 1500     # what one chunk may contribute to the agent's context


def _remember(tool_context: ToolContext | None, item: dict[str, Any]) -> None:
    """Append one evidence item to session state (read later by the critic).

    Evidence is tagged with the invocation id (one invocation = one user question), so in a chat
    session the critic never sees passages retrieved for an earlier question.
    """
    if tool_context is None:
        return
    inv = tool_context.invocation_id
    ev = tool_context.state.get(EVIDENCE_KEY) or {}
    items = list(ev.get("items", [])) if isinstance(ev, dict) and ev.get("invocation_id") == inv else []
    items.append(item)
    tool_context.state[EVIDENCE_KEY] = {"invocation_id": inv, "items": items}


def search_documents(question: str, tool_context: ToolContext, k: int = 4) -> dict[str, Any]:
    """Search the company documents (ACME employee policy handbook, the RAG paper by Lewis et al.
    2020, and the "Attention Is All You Need" Transformer paper) for passages relevant to a question.

    Use this for policies, rules, procedures, definitions and research-paper content.
    It returns raw passages, NOT an answer: you must write the answer yourself using only these
    passages, and cite each fact as [file p.N] using the file and page of the passage.

    Args:
        question: A focused search query, e.g. "bereavement leave paid days".
        k: How many passages to return (1-8). Default 4.

    Returns:
        {"status": "success", "chunks": [{"file", "page", "score", "text"}, ...]} or
        {"status": "error", "message": ...}. score is cosine similarity (higher = more relevant).
    """
    try:
        k = max(1, min(int(k), 8))
        hits = retrieve(load_settings(), question, k)
        chunks = [{"file": h["file"], "page": h["page"], "score": h["score"],
                   "text": h["text"][:MAX_CHUNK_CHARS]} for h in hits]
        _remember(tool_context, {"tool": "search_documents", "query": question, "chunks": chunks})
        return {"status": "success", "query": question, "chunks": chunks}
    except Exception as exc:  # never raise into the agent
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}


def query_table(question: str, tool_context: ToolContext) -> dict[str, Any]:
    """Answer a question about the employees table (employees.csv: one row per employee with
    columns employee_id, name, team, role, location, salary_inr, start_date, manager_id).

    Use this for counts, filters, lookups, averages, sums, min/max and rankings over employees,
    teams, salaries, locations, managers and start dates. The question is translated into a
    validated query spec and computed with pandas, so the numbers are exact.

    Args:
        question: The table question in plain English, e.g. "How many employees are in Pune?".

    Returns:
        {"status": "success", "result": [rows], "rows_matched": int, "spec": {...},
         "source": "employees.csv (N rows matched)", "rows": [row numbers]} or
        {"status": "error", "message": ...}. Report numbers exactly as in "result".
    """
    try:
        out = kb_query_table(question, phrase=False)  # the agent writes the sentence itself
        if out.get("error"):
            return {"status": "error", "message": out["error"]}
        src = out["sources"][0]
        result = {"status": "success", "question": question, "result": out["result"],
                  "rows_matched": src["rows_matched"], "spec": out["spec"],
                  "source": src["label"], "rows": src["rows"],
                  "_llm_usage": out["usage"]}  # kb's own LLM call (spec), counted in metrics
        _remember(tool_context, {"tool": "query_table", "question": question, "result": out["result"],
                                 "source": src["label"]})
        return result
    except Exception as exc:  # never raise into the agent
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
