"""Readable timeline of ONE agent request: agent -> tool -> model call, with durations.

  python scripts/trace_report.py                 # latest request
  python scripts/trace_report.py --trace <id>    # a specific trace id (prefix is enough)
  python scripts/trace_report.py --list          # recent requests

Reads logs/agent_traces.jsonl (written by kl_agents.telemetry). Two views:
1. the OpenTelemetry span tree ADK emitted (invoke_agent / execute_tool / generate_content ...),
2. our callback-measured model calls and tool calls with token counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "logs" / "agent_traces.jsonl"


def load_traces(path: Path) -> list[dict[str, Any]]:
    """All recorded request timelines, oldest first."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def span_tree(spans: list[dict[str, Any]]) -> list[str]:
    """Indented span tree with start offset and duration."""
    if not spans:
        return ["  (no spans recorded)"]
    t0 = min(s["start_ns"] for s in spans)
    ids = {s["span_id"] for s in spans}
    children: dict[str | None, list[dict[str, Any]]] = {}
    for s in spans:
        parent = s["parent_id"] if s["parent_id"] in ids else None
        children.setdefault(parent, []).append(s)
    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for s in sorted(children.get(parent, []), key=lambda x: x["start_ns"]):
            start = (s["start_ns"] - t0) / 1e9
            attrs = s.get("attrs", {})
            extra = []
            for key in ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"):
                if key in attrs:
                    extra.append(f"{key.rsplit('.', 1)[-1]}={attrs[key]}")
            lines.append(f"  {start:7.2f}s  {'  ' * depth}{s['name']:<40} {s['duration_ms'] / 1000:7.2f}s"
                         f"  {' '.join(extra)}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="trace id or prefix (default: latest)")
    ap.add_argument("--list", action="store_true", help="list recent requests")
    args = ap.parse_args()
    traces = load_traces(TRACES)
    if not traces:
        print(f"No traces yet in {TRACES.name}. Run: python agents_cli.py ask \"...\"")
        return 1
    if args.list:
        for t in traces[-20:]:
            print(f"{(t['trace_id'] or '-')[:12]}  {t['ts']}  {t['system']:<9} {t['latency_s']:6.1f}s  {t['question'][:60]}")
        return 0
    t = next((x for x in reversed(traces) if args.trace and (x["trace_id"] or "").startswith(args.trace)), None) \
        if args.trace else traces[-1]
    if t is None:
        print(f"trace {args.trace!r} not found")
        return 1
    print(f"Trace {t['trace_id']}  [{t['system']}]  {t['ts']}")
    print(f"Question: {t['question']}")
    print(f"Total latency: {t['latency_s']:.2f}s\n")
    print("OpenTelemetry spans emitted by ADK (offset, name, duration):")
    print("\n".join(span_tree(t["spans"])))
    print("\nModel calls (measured by callbacks; time excludes our rate-limit wait):")
    for i, m in enumerate(t["model_calls"], 1):
        who = m["agent"] + (" [inside kb tool]" if m["source"] == "kb_tool" else "")
        print(f"  #{i:<2} at {m['start_s']:6.2f}s  {who:<32} {m['duration_s']:6.2f}s  "
              f"in={m['input_tokens']:<5} out={m['output_tokens']}")
    print("\nTool calls:")
    for tc in sorted(t["tools"], key=lambda x: x["start_s"]):
        arg = tc["args"].get("question") or tc["args"].get("request") or ""
        print(f"  at {tc['start_s']:6.2f}s  {tc['agent']:<12} -> {tc['name']:<17} {tc['duration_s']:6.2f}s  "
              f"status={tc['status']}  {arg[:60]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
