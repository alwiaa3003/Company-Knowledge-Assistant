"""Retrieval evaluation: hit rate@k and MRR across chunking / top-k experiments.

Run:  .venv\\Scripts\\python.exe eval/run_eval.py            (all experiments -> eval/results.md)
      .venv\\Scripts\\python.exe eval/run_eval.py --only baseline

A "hit" at k = any of the top-k chunks comes from expected_file AND one of expected_pages.
MRR = mean of 1/rank of the first correct chunk (0 if none in the top MAX_K).
Retrieval only: no generation calls, so this is cheap. Embeddings are cached on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb import store  # noqa: E402
from kb.config import Settings, collection_name, load_settings  # noqa: E402
from kb.embeddings import embed_query  # noqa: E402
from kb.rag import build_knowledge_base  # noqa: E402

MAX_K = 10
REPORT_KS = (1, 3, 5)


@dataclass(frozen=True)
class Experiment:
    """One configuration. Only ONE variable differs from the baseline."""

    name: str
    chunk_size: int
    overlap: int
    k: int
    changed: str


EXPERIMENTS = [
    Experiment("baseline", 1000, 200, 4, "-"),
    Experiment("chunk 500", 500, 200, 4, "chunk_size 1000 -> 500"),
    Experiment("chunk 2000", 2000, 200, 4, "chunk_size 1000 -> 2000"),
    Experiment("overlap 0", 1000, 0, 4, "overlap 200 -> 0"),
    Experiment("k 2", 1000, 200, 2, "k 4 -> 2"),
    Experiment("k 8", 1000, 200, 8, "k 4 -> 8"),
]


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Read eval/eval_set.json."""
    return json.loads(path.read_text(encoding="utf-8"))


def first_hit_rank(hits: list[dict[str, Any]], item: dict[str, Any]) -> int | None:
    """1-based rank of the first chunk matching expected file + page, or None."""
    for rank, h in enumerate(hits, start=1):
        if h["file"] == item["expected_file"] and h["page"] in item["expected_pages"]:
            return rank
    return None


def ensure_collection(settings: Settings, chunk_size: int, overlap: int) -> str:
    """Build the collection for this chunking config if it doesn't exist yet."""
    name = collection_name(chunk_size, overlap)
    existing = {c["name"] for c in store.collection_stats(settings.chroma_dir)}
    if name not in existing:
        print(f"  building {name} ...")
        info = build_knowledge_base(settings, chunk_size, overlap)
        print(f"  {info['chunks']} chunks ({info['newly_embedded']} newly embedded)")
    return name


def evaluate(settings: Settings, exp: Experiment, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Run every eval question against one experiment's collection."""
    col = store.get_collection(settings.chroma_dir, ensure_collection(settings, exp.chunk_size, exp.overlap))
    ranks: list[int | None] = []
    misses = []
    for item in items:
        hits = store.query(col, embed_query(settings, item["question"]), MAX_K)
        rank = first_hit_rank(hits, item)
        ranks.append(rank)
        if rank is None or rank > exp.k:
            misses.append({"question": item["question"], "expected": f"{item['expected_file']} p.{item['expected_pages']}",
                           "rank": rank, "got": [f"{h['file'].split('.')[0][:12]} p.{h['page']}" for h in hits[:exp.k]]})
    n = len(items)
    hit_at = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (*REPORT_KS, exp.k)}
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return {"exp": exp, "hit_at": hit_at, "mrr": mrr, "misses": misses, "n": n}


def to_markdown(results: list[dict[str, Any]]) -> str:
    """Results table plus the per-experiment miss lists."""
    lines = ["| Experiment | Changed | chunk_size | overlap | k | hit@1 | hit@3 | hit@5 | **hit@k** | MRR@10 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        e, h = r["exp"], r["hit_at"]
        lines.append(f"| {e.name} | {e.changed} | {e.chunk_size} | {e.overlap} | {e.k} | {h[1]:.2f} | "
                     f"{h[3]:.2f} | {h[5]:.2f} | **{h[e.k]:.2f}** | {r['mrr']:.3f} |")
    lines.append("")
    for r in results:
        e = r["exp"]
        lines.append(f"**Misses at k={e.k} — {e.name}** ({len(r['misses'])}/{r['n']})")
        if not r["misses"]:
            lines.append("- none")
        for m in r["misses"]:
            rank = f"first correct at rank {m['rank']}" if m["rank"] else f"not in top {MAX_K}"
            lines.append(f"- {m['question']} — expected {m['expected']}, {rank}; got {', '.join(m['got'])}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run only the experiment with this name")
    parser.add_argument("--out", default=str(ROOT / "eval" / "results_table.md"))
    args = parser.parse_args()
    settings = load_settings()
    items = load_eval_set(ROOT / "eval" / "eval_set.json")
    exps = [e for e in EXPERIMENTS if not args.only or e.name == args.only]
    results = []
    for exp in exps:
        print(f"== {exp.name}")
        r = evaluate(settings, exp, items)
        print(f"   hit@1={r['hit_at'][1]:.2f} hit@3={r['hit_at'][3]:.2f} hit@5={r['hit_at'][5]:.2f} "
              f"hit@{exp.k}={r['hit_at'][exp.k]:.2f} MRR={r['mrr']:.3f}")
        results.append(r)
    md = to_markdown(results)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nWrote {args.out}\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
