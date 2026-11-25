"""KnowledgeLens multi-agent CLI (Project 2, Google ADK).

  python agents_cli.py ask "How many people are in the Data team, and what is the remote work policy?"
  python agents_cli.py ask "..." --system single      # baseline: one agent with both tools
  python agents_cli.py chat [--system hierarchy|single]

Every request is logged to logs/agent_requests.jsonl (metrics) and logs/agent_traces.jsonl (timeline).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import warnings
from typing import Any

LINE = "-" * 72


def print_result(r: dict[str, Any]) -> None:
    """Answer, sources, the agents/tools path and one metrics line."""
    print(LINE)
    print(r["answer"] or "(no answer)")
    print(LINE)
    if r["sources"]:
        print("Sources:")
        for s in r["sources"]:
            print(f"  - {s['file']}, page {s['page']}" if "page" in s else f"  - {s['label']}")
    else:
        print("Sources: (none)")
    print(f"Agents: {' -> '.join(r['agents'])}")
    calls = [f"{t['name']}({t['args'].get('question', t['args'].get('request', ''))[:50]!r})"
             if t["args"] else t["name"] for t in r["tool_calls"]]
    print(f"Tools:  {', '.join(calls) or '(none)'}")
    tps = f"{r['tokens_per_sec']} tok/s" if r["tokens_per_sec"] else "n/a"
    print(f"Metrics: status={r['status']} | {r['latency_s']:.1f}s total ({r['generation_time_s']:.1f}s in model calls) | "
          f"LLM calls {r['llm_calls']} {r['llm_calls_by_agent']} | tokens {r['input_tokens']}/{r['output_tokens']} | "
          f"{tps} | est ${r['est_cost_usd']:.6f} | trace {r['trace_id']}")
    if r["errors"]:
        print("Errors:", "; ".join(r["errors"]))


async def _ask(system: str, question: str) -> None:
    from kl_agents.runner import AgentSession
    print_result(await AgentSession(system).ask(question))


async def _chat(system: str) -> None:
    from kl_agents.runner import AgentSession
    session = AgentSession(system)
    sid = await session.new_session_id()
    print(f"KnowledgeLens agents ({system}). Ask about the handbook, the papers or the employees table. "
          "Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", ":q"}:
            break
        if q:
            print_result(await session.ask(q, session_id=sid))


def main() -> int:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    p = argparse.ArgumentParser(prog="agents_cli", description="KnowledgeLens multi-agent assistant (ADK)")
    sub = p.add_subparsers(dest="command", required=True)
    pa = sub.add_parser("ask", help="ask one question")
    pa.add_argument("question")
    pa.add_argument("--system", choices=["hierarchy", "hierarchy_noreflect", "single"], default="hierarchy")
    pc = sub.add_parser("chat", help="interactive loop (one ADK session)")
    pc.add_argument("--system", choices=["hierarchy", "hierarchy_noreflect", "single"], default="hierarchy")
    args = p.parse_args()
    try:
        if args.command == "ask":
            asyncio.run(_ask(args.system, args.question))
        else:
            asyncio.run(_chat(args.system))
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
