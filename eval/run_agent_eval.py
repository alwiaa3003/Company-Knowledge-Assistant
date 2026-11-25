"""Evaluate the ADK agents on eval/agent_eval.json: routing, correctness, refusals, cost.

  python eval/run_agent_eval.py --system single
  python eval/run_agent_eval.py --system hierarchy_noreflect
  python eval/run_agent_eval.py --system hierarchy
  python eval/run_agent_eval.py --report            # build eval/agent_results.md from saved runs
  python eval/run_agent_eval.py --system hierarchy --only M2,U2   # re-run some items

Scoring (objective, no LLM judge):
- routing: the set of specialists actually used (hierarchy: docs_agent/table_agent; single agent:
  which tools it called) == expected set. Scored on answerable questions (unanswerable ones may
  legitimately search first).
- docs part correct: the answer cites an expected file + page AND contains each key fact (any synonym).
- table part correct: the pandas ground truth (computed here, not hard-coded) appears in the answer.
- mixed: both parts must be correct.
- refusal: unanswerable questions must say "I don't know" and cite nothing.
Raw per-question results are saved to eval/agent_runs/<system>.json (so runs can be spread out).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "eval" / "agent_runs"
SYSTEMS = ["single", "hierarchy_noreflect", "hierarchy"]
LABELS = {"single": "(a) single agent, both tools",
          "hierarchy_noreflect": "(b) coordinator + specialists",
          "hierarchy": "(c) coordinator + specialists + critic"}


def table_truths(df: pd.DataFrame) -> dict[str, Any]:
    """Ground truth for every table check, computed directly with pandas."""
    counts = df.team.value_counts()
    return {
        "count_hyderabad": int((df.location == "Hyderabad").sum()),
        "mean_salary_sales": float(df[df.team == "Sales"].salary_inr.mean()),
        "smallest_team": str(counts.index[-1]),
        "count_data": int((df.team == "Data").sum()),
        "top3_names": df.sort_values("salary_inr", ascending=False).head(3).name.tolist(),
        "count_joined_2023": int((pd.to_datetime(df.start_date) >= "2023-01-01").sum()),
    }


def numbers_in(text: str) -> list[float]:
    """All numbers in a text ("2,724,166.67" -> 2724166.67)."""
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def table_correct(answer: str, key: str, truth: Any) -> bool:
    """Is the pandas truth present in the answer?"""
    if isinstance(truth, list):
        return all(name in answer for name in truth)
    if isinstance(truth, str):
        return re.search(rf"\b{re.escape(truth)}\b", answer) is not None
    if isinstance(truth, float) and not truth.is_integer():
        return any(abs(n - truth) <= max(1.0, truth * 0.0005) for n in numbers_in(answer))  # allow rounding
    return any(n == truth for n in numbers_in(answer))


def docs_correct(answer: str, sources: list[dict], item: dict) -> tuple[bool, bool, bool]:
    """(cited_expected_source, all_facts_present, both)."""
    doc_expect = [s for s in item["expected_sources"] if "pages" in s]
    cited = any(s.get("file") == e["file"] and s.get("page") in e["pages"] for e in doc_expect for s in sources)
    low = answer.lower()
    facts = all(any(syn.lower() in low for syn in group) for group in item.get("doc_facts", []))
    return cited, facts, cited and facts


def is_refusal(answer: str) -> bool:
    return "i don't know" in answer.lower() or "i do not know" in answer.lower()


def specialists_used(r: dict[str, Any]) -> set[str]:
    """docs/table actually used: from tool calls (works for both single agent and hierarchy)."""
    used = set()
    for t in r["tool_calls"]:
        if t["name"] in ("search_documents", "docs_agent"):
            used.add("docs")
        if t["name"] in ("query_table", "table_agent"):
            used.add("table")
    return used


def score(item: dict, r: dict, truths: dict) -> dict[str, Any]:
    """All checks for one question."""
    from kl_agents.runner import extract_sources
    ans = r["answer"] or ""
    r = {**r, "sources": extract_sources(ans)}  # re-parse citations so all systems are scored identically
    used = specialists_used(r)
    out: dict[str, Any] = {"id": item["id"], "type": item["type"], "used": sorted(used),
                           "routing_ok": None, "correct": False, "detail": ""}
    if item["type"] != "unanswerable":
        out["routing_ok"] = used == set(item["expected_specialists"])
    if item["type"] == "docs":
        cited, facts, ok = docs_correct(ans, r["sources"], item)
        out.update(correct=ok, detail=f"cited={cited} facts={facts}")
    elif item["type"] == "table":
        ok = table_correct(ans, item["truth"], truths[item["truth"]])
        out.update(correct=ok, detail=f"truth={truths[item['truth']]}")
    elif item["type"] == "mixed":
        t_ok = table_correct(ans, item["truth"], truths[item["truth"]])
        cited, facts, d_ok = docs_correct(ans, r["sources"], item)
        out.update(correct=t_ok and d_ok, detail=f"table={t_ok} docs_cited={cited} docs_facts={facts}")
    else:
        refused = is_refusal(ans) and not [s for s in r["sources"] if "page" in s]
        out.update(correct=refused, detail=f"refused={refused}")
    return out


async def run_system(system: str, items: list[dict], pause_s: float) -> list[dict[str, Any]]:
    """Run every item through a fresh session of `system`, pausing between questions."""
    from kl_agents.runner import AgentSession
    session = AgentSession(system)
    results = []
    for i, item in enumerate(items):
        if i:
            time.sleep(pause_s)  # spread load: free-tier per-minute quotas
        print(f"[{system}] {item['id']}: {item['question']}", flush=True)
        r = await session.ask(item["question"])
        keep = {k: r[k] for k in ("answer", "sources", "agents", "tools", "tool_calls", "llm_calls",
                                  "llm_calls_by_agent", "input_tokens", "output_tokens", "latency_s",
                                  "generation_time_s", "tokens_per_sec", "est_cost_usd", "status",
                                  "trace_id", "errors")}
        results.append({"id": item["id"], **keep})
        print(f"   -> {r['status']} calls={r['llm_calls']} {r['latency_s']:.1f}s | {r['answer'][:160]!r}", flush=True)
    return results


def save_runs(system: str, new: list[dict]) -> None:
    """Merge new per-question results into eval/agent_runs/<system>.json (by id)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{system}.json"
    old = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    old.update({r["id"]: r for r in new})
    path.write_text(json.dumps(list(old.values()), indent=1, ensure_ascii=False), encoding="utf-8")


def aggregate(system: str, items: list[dict], truths: dict) -> dict[str, Any] | None:
    """Score a saved run and compute the summary metrics."""
    path = RUNS / f"{system}.json"
    if not path.exists():
        return None
    runs = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    scored = [score(it, runs[it["id"]], truths) for it in items if it["id"] in runs]
    rows = [runs[s["id"]] for s in scored]

    def frac(sel: list[dict], key: str) -> str:
        vals = [s[key] for s in sel if s[key] is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"

    by_type = {t: [s for s in scored if s["type"] == t] for t in ("docs", "table", "mixed", "unanswerable")}
    return {
        "system": system, "n": len(scored), "scored": scored,
        "routing": frac(scored, "routing_ok"),
        "correct_all": frac(scored, "correct"),
        "docs": frac(by_type["docs"], "correct"), "table": frac(by_type["table"], "correct"),
        "mixed": frac(by_type["mixed"], "correct"), "refusal": frac(by_type["unanswerable"], "correct"),
        "avg_calls": statistics.mean(r["llm_calls"] for r in rows),
        "avg_latency": statistics.mean(r["latency_s"] for r in rows),
        "p95_latency": sorted(r["latency_s"] for r in rows)[max(0, round(0.95 * len(rows)) - 1)],
        "avg_in": statistics.mean(r["input_tokens"] for r in rows),
        "avg_out": statistics.mean(r["output_tokens"] for r in rows),
        "avg_cost": statistics.mean(r["est_cost_usd"] for r in rows),
        "avg_tps": statistics.mean(r["tokens_per_sec"] for r in rows if r["tokens_per_sec"]),
        "errors": sum(1 for r in rows if r["status"] != "ok"),
        "runs": runs,
    }


def report(items: list[dict], truths: dict) -> str:
    """Markdown comparison table + per-question detail for every system that has saved runs."""
    aggs = [a for a in (aggregate(s, items, truths) for s in SYSTEMS) if a]
    lines = ["| System | Routing | Correct (all) | Docs | Table | Mixed | Refusal | Avg LLM calls | "
             "Avg latency | p95 latency | Avg tokens in/out | Avg cost/req | Avg tok/s | Errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in aggs:
        lines.append(f"| {LABELS[a['system']]} | {a['routing']} | **{a['correct_all']}** | {a['docs']} | {a['table']} | "
                     f"{a['mixed']} | {a['refusal']} | {a['avg_calls']:.1f} | {a['avg_latency']:.1f}s | "
                     f"{a['p95_latency']:.1f}s | {a['avg_in']:.0f}/{a['avg_out']:.0f} | ${a['avg_cost']:.5f} | "
                     f"{a['avg_tps']:.1f} | {a['errors']} |")
    lines += ["", "Per question (✔ correct, ✘ wrong; routing = specialists used; calls = LLM calls):", ""]
    head = "| ID | Question | " + " | ".join(LABELS[a["system"]] for a in aggs) + " |"
    lines += [head, "|---|---|" + "---|" * len(aggs)]
    for it in items:
        cells = []
        for a in aggs:
            s = next((x for x in a["scored"] if x["id"] == it["id"]), None)
            r = a["runs"].get(it["id"])
            if not s or not r:
                cells.append("not run")
                continue
            route = "+".join(s["used"]) or "none"
            cells.append(f"{'✔' if s['correct'] else '✘'} {route}, {r['llm_calls']} calls, {r['latency_s']:.0f}s"
                         + ("" if s["correct"] else f" ({s['detail']})"))
        lines.append(f"| {it['id']} | {it['question']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--only", help="comma-separated item ids")
    ap.add_argument("--pause", type=float, default=10.0, help="seconds between questions")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    from kb.config import load_settings
    items = json.loads((ROOT / "eval" / "agent_eval.json").read_text(encoding="utf-8"))
    truths = table_truths(pd.read_csv(load_settings().table_file))
    if args.system:
        sel = [it for it in items if not args.only or it["id"] in args.only.split(",")]
        save_runs(args.system, asyncio.run(run_system(args.system, sel, args.pause)))
    md = report(items, truths)
    (ROOT / "eval" / "agent_results_table.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
