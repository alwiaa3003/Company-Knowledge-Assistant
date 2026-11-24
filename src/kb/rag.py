"""Top-level entry points: build the knowledge base, and answer a question (routing included).

`answer_question` and `structured.query_table` are the two functions that become ADK tools.
"""

from __future__ import annotations

import time
from typing import Any

from kb import store
from kb.chunking import chunk_records
from kb.config import Settings, collection_name, estimate_embed_cost, load_settings
from kb.embeddings import embed_formatted, format_document
from kb.generate import answer_from_chunks
from kb.ingest import load_directory
from kb.metrics import log_request
from kb.retrieve import retrieve
from kb.router import route
from kb.structured import load_table, query_table


def _title_for(file_name: str) -> str:
    """Human-ish document title used in the embedding text, e.g. 'acme employee policy handbook'."""
    return file_name.rsplit(".", 1)[0].replace("_", " ")


def build_knowledge_base(settings: Settings, chunk_size: int | None = None,
                         overlap: int | None = None) -> dict[str, Any]:
    """Ingest data/docs -> chunk -> embed (cached) -> store in collection kb_cs{size}_ov{overlap}."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    start = time.perf_counter()
    records = load_directory(settings.docs_dir)
    chunks = chunk_records(records, chunk_size, overlap)
    formatted = [format_document(c["text"], _title_for(c["file"])) for c in chunks]
    vectors, n_new = embed_formatted(settings, formatted)
    name = collection_name(chunk_size, overlap)
    store.build_index(settings.chroma_dir, name, chunks, vectors,
                      {"chunk_size": chunk_size, "overlap": overlap, "embed_model": settings.embed_model})
    approx_tokens = sum(len(formatted[i]) for i in range(len(formatted))) // 4
    return {"collection": name, "files": len({r["file"] for r in records}), "pages": len(records),
            "chunks": len(chunks), "newly_embedded": n_new, "from_cache": len(chunks) - n_new,
            "est_embed_cost_usd_if_uncached": round(estimate_embed_cost(settings, approx_tokens), 6),
            "seconds": round(time.perf_counter() - start, 2)}


def answer_question(question: str, k: int = 4) -> dict[str, Any]:
    """Answer any question: route it, then use RAG over documents or pandas over the table.

    Returns {"answer": str, "sources": [{"file","page"}] or [{"file","rows",...}],
             "route": "docs"|"table", "usage": {input_tokens, output_tokens, latency_s, est_cost_usd}}.
    """
    settings = load_settings()
    start = time.perf_counter()
    df = load_table(settings.table_file)
    chosen, method, router_usage = route(settings, question, settings.table_file.name, list(df.columns))

    if chosen == "table":
        out = query_table(question)
        u = out["usage"]
        total = {
            "input_tokens": u["input_tokens"] + router_usage.input_tokens,
            "output_tokens": u["output_tokens"] + router_usage.output_tokens,
            "latency_s": round(time.perf_counter() - start, 3),
            "est_cost_usd": round(u["est_cost_usd"] + router_usage.est_cost_usd, 8),
            "llm_calls": u["calls"] + router_usage.calls,
        }
        return {"answer": out["answer"], "sources": out["sources"], "route": "table",
                "router": method, "spec": out.get("spec"), "result": out.get("result"), "usage": total}

    chunks = retrieve(settings, question, k)
    gen = answer_from_chunks(settings, question, chunks)
    usage = gen["usage"] + router_usage
    total = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
             "latency_s": round(time.perf_counter() - start, 3),
             "est_cost_usd": round(usage.est_cost_usd, 8), "llm_calls": usage.calls}
    log_request(settings.logs_dir, {"route": "docs", "question": question, "k": k,
                                    "collection": settings.default_collection, "usage": total,
                                    "sources": gen["sources"]})
    out = {"answer": gen["answer"], "sources": gen["sources"], "route": "docs", "router": method,
           "retrieved": [{"file": c["file"], "page": c["page"], "score": c["score"]} for c in chunks],
           "usage": total}
    if gen["unverified_citations"]:
        out["unverified_citations"] = gen["unverified_citations"]
    return out
