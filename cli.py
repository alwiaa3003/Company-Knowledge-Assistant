"""KnowledgeLens command line.

  python cli.py ingest [--chunk-size 1000] [--overlap 200]
  python cli.py ask "How many days of bereavement leave do I get?" [-k 4]
  python cli.py chat [-k 4]
  python cli.py stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kb.config import load_settings  # noqa: E402

LINE = "-" * 72


def format_sources(sources: list[dict[str, Any]]) -> str:
    """One readable line per source."""
    if not sources:
        return "  (none)"
    lines = []
    for s in sources:
        if "page" in s:
            lines.append(f"  - {s['file']}, page {s['page']}")
        else:
            rows = ", ".join(map(str, s.get("rows", [])[:10]))
            more = " ..." if s.get("rows_matched", 0) > 10 else ""
            lines.append(f"  - {s['file']}: {s.get('rows_matched', '?')} rows matched (rows {rows}{more})")
    return "\n".join(lines)


def print_answer(res: dict[str, Any]) -> None:
    """Pretty-print the dict returned by answer_question."""
    u = res["usage"]
    print(LINE)
    print(res["answer"])
    print(LINE)
    print(f"Route: {res['route']} (decided by {res.get('router', '?')})")
    if res["route"] == "table" and res.get("spec"):
        spec = {k: v for k, v in res["spec"].items() if v not in (None, [], False, "none")}
        print(f"Query spec: {spec}")
    print("Sources:\n" + format_sources(res["sources"]))
    if res.get("unverified_citations"):
        print("WARNING: answer cites pages that were not retrieved: " +
              ", ".join(f"{c['file']} p.{c['page']}" for c in res["unverified_citations"]))
    if res.get("retrieved"):
        print("Retrieved: " + ", ".join(f"{r['file']} p.{r['page']} ({r['score']:.2f})" for r in res["retrieved"]))
    print(f"Latency: {u['latency_s']:.2f}s | tokens in/out: {u['input_tokens']}/{u['output_tokens']} | "
          f"LLM calls: {u.get('llm_calls', '?')} | est. cost: ${u['est_cost_usd']:.6f}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from kb.rag import build_knowledge_base
    settings = load_settings()
    print(f"Ingesting {settings.docs_dir} (chunk_size={args.chunk_size or settings.chunk_size}, "
          f"overlap={settings.chunk_overlap if args.overlap is None else args.overlap}) ...")
    info = build_knowledge_base(settings, args.chunk_size, args.overlap)
    for key, value in info.items():
        print(f"  {key}: {value}")


def cmd_ask(args: argparse.Namespace) -> None:
    from kb.rag import answer_question
    print_answer(answer_question(args.question, k=args.k))


def cmd_chat(args: argparse.Namespace) -> None:
    from kb.rag import answer_question
    print("KnowledgeLens chat. Ask about the handbook, the papers, or the employees table. "
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
            try:
                print_answer(answer_question(q, k=args.k))
            except Exception as exc:  # keep the chat alive on a single failure
                print(f"Error: {type(exc).__name__}: {exc}")


def cmd_stats(_: argparse.Namespace) -> None:
    from kb.store import collection_stats
    for c in collection_stats(load_settings().chroma_dir):
        print(f"  {c['name']}: {c['chunks']} chunks  {c['metadata']}")


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI commands."""
    p = argparse.ArgumentParser(prog="knowledgelens", description="Company Knowledge Assistant (RAG)")
    sub = p.add_subparsers(dest="command", required=True)
    pi = sub.add_parser("ingest", help="build the vector index from data/docs")
    pi.add_argument("--chunk-size", type=int, default=None, help="characters per chunk (default from .env)")
    pi.add_argument("--overlap", type=int, default=None, help="overlapping characters (default from .env)")
    pi.set_defaults(func=cmd_ingest)
    pa = sub.add_parser("ask", help="ask one question")
    pa.add_argument("question")
    pa.add_argument("-k", type=int, default=4, help="chunks to retrieve")
    pa.set_defaults(func=cmd_ask)
    pc = sub.add_parser("chat", help="interactive question loop")
    pc.add_argument("-k", type=int, default=4)
    pc.set_defaults(func=cmd_chat)
    ps = sub.add_parser("stats", help="list collections in chroma_db")
    ps.set_defaults(func=cmd_stats)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
