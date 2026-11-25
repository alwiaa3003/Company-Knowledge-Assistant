"""Aggregate LLM metrics over logged agent requests (logs/agent_requests.jsonl).

  python scripts/metrics_summary.py                 # all requests, grouped by system
  python scripts/metrics_summary.py --last 20       # only the most recent 20 requests

Reports per system: requests, p50/p95 latency, average cost per request, average tokens/sec,
average LLM calls, average tokens, error rate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "agent_requests.jsonl"


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank-with-interpolation percentile (p in 0..100)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * p / 100
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics for one group of request records."""
    lat = [r["latency_s"] for r in rows]
    tps = [r["tokens_per_sec"] for r in rows if r.get("tokens_per_sec")]
    errors = [r for r in rows if r["status"] != "ok"]
    return {
        "requests": len(rows),
        "p50_latency_s": percentile(lat, 50),
        "p95_latency_s": percentile(lat, 95),
        "avg_cost_usd": statistics.mean(r["est_cost_usd"] for r in rows),
        "avg_tokens_per_sec": statistics.mean(tps) if tps else float("nan"),
        "avg_llm_calls": statistics.mean(r["llm_calls"] for r in rows),
        "avg_input_tokens": statistics.mean(r["input_tokens"] for r in rows),
        "avg_output_tokens": statistics.mean(r["output_tokens"] for r in rows),
        "error_rate": len(errors) / len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=0, help="only the last N requests")
    args = ap.parse_args()
    if not LOG.exists():
        print(f"No requests logged yet ({LOG.name}).")
        return 1
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("system") in ("hierarchy", "hierarchy_noreflect", "single")]
    if args.last:
        rows = rows[-args.last:]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["system"], []).append(r)
    groups["ALL"] = rows
    header = f"{'system':<20} {'n':>4} {'p50 lat':>8} {'p95 lat':>8} {'avg $/req':>11} {'tok/s':>6} " \
             f"{'LLM calls':>9} {'in tok':>7} {'out tok':>7} {'errors':>7}"
    print(header)
    print("-" * len(header))
    for name, g in groups.items():
        if not g:
            continue
        m = summarize(g)
        print(f"{name:<20} {m['requests']:>4} {m['p50_latency_s']:>7.1f}s {m['p95_latency_s']:>7.1f}s "
              f"{m['avg_cost_usd']:>11.6f} {m['avg_tokens_per_sec']:>6.1f} {m['avg_llm_calls']:>9.1f} "
              f"{m['avg_input_tokens']:>7.0f} {m['avg_output_tokens']:>7.0f} {m['error_rate']:>6.0%}")
    print("\nLatency includes the self-imposed GEN_RPM pacing (>= 60/GEN_RPM s between LLM calls).")
    print("tok/s = output tokens / time inside model calls (non-streamed, end-to-end per call).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
