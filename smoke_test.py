"""Smoke test: checks that every external piece works before we build anything on top.

Run:  python smoke_test.py
Checks: (1) API key loads, (2) one Gemini generation, (3) one embedding, (4) Chroma add + query.
"""

from __future__ import annotations

import sys
import time

from google import genai
from google.genai import types

sys.path.insert(0, "src")  # lets us `import kb` without installing the package
from kb.config import Settings, estimate_cost, load_settings  # noqa: E402


def check_key() -> Settings:
    """Step 1: load settings and fail early with a clear message if the key is missing."""
    settings = load_settings()
    print("[1/4] API key loaded from .env (value not shown)")
    return settings


def check_generation(client: genai.Client, settings: Settings) -> None:
    """Step 2: one tiny generation call, with latency, tokens and estimated cost."""
    start = time.perf_counter()
    resp = client.models.generate_content(
        model=settings.gen_model,
        contents="Reply with exactly: KnowledgeLens is alive.",
    )
    latency_ms = (time.perf_counter() - start) * 1000
    usage = resp.usage_metadata
    tokens_in = usage.prompt_token_count or 0
    tokens_out = usage.candidates_token_count or 0
    cost = estimate_cost(settings, tokens_in, tokens_out)
    print(f"[2/4] Generation OK ({settings.gen_model}): {resp.text.strip()!r}")
    print(f"      latency={latency_ms:.0f} ms  in={tokens_in}  out={tokens_out}  est_cost=${cost:.6f}")


def check_embedding(client: genai.Client, settings: Settings) -> list[float]:
    """Step 3: embed one sentence and confirm the vector has the expected size."""
    result = client.models.embed_content(
        model=settings.embed_model,
        contents="title: none | text: Our refund policy allows returns within 30 days.",
        config=types.EmbedContentConfig(output_dimensionality=settings.embed_dim),
    )
    vector = result.embeddings[0].values
    assert len(vector) == settings.embed_dim, f"expected {settings.embed_dim} dims, got {len(vector)}"
    print(f"[3/4] Embedding OK ({settings.embed_model}): {len(vector)} dims, first values {vector[:3]}")
    return vector


def check_batch_embedding(settings: Settings) -> None:
    """Step 3b: embed 3 DIFFERENT texts in ONE request through the project's real batch function.

    Guards against the aggregation trap: gemini-embedding-2 merges a plain list of strings into one
    embedding. We need exactly 3 vectors, of the right size, and not all identical.
    """
    from kb.embeddings import _embed_batch, format_document

    texts = [format_document("Employees get up to five paid days of bereavement leave."),
             format_document("The Transformer uses eight attention heads."),
             format_document("Lost laptops must be reported to IT Security within four hours.")]
    vectors = _embed_batch(settings, texts)  # bypasses the cache on purpose: this tests the API
    assert len(vectors) == 3, f"expected 3 vectors, got {len(vectors)}"
    assert all(len(v) == settings.embed_dim for v in vectors), "wrong vector dimension"
    assert not (vectors[0] == vectors[1] == vectors[2]), "all 3 vectors identical -> inputs were aggregated"
    print(f"[3b]  Batch embedding OK: 3 texts -> 3 distinct vectors of {settings.embed_dim} dims")


def check_chroma(vector: list[float]) -> None:
    """Step 4: store the vector in a throwaway in-memory Chroma collection and query it back."""
    import chromadb

    client = chromadb.EphemeralClient()  # in-memory: does not touch chroma_db/
    col = client.get_or_create_collection("smoke", metadata={"hnsw:space": "cosine"})
    col.add(ids=["doc1"], embeddings=[vector], documents=["refund policy"],
            metadatas=[{"file": "smoke.txt", "page": 1}])
    hit = col.query(query_embeddings=[vector], n_results=1)
    print(f"[4/4] Chroma OK: nearest id={hit['ids'][0][0]}  distance={hit['distances'][0][0]:.4f}")


def main() -> int:
    """Run all checks in order; return a process exit code."""
    try:
        settings = check_key()
        client = genai.Client(api_key=settings.gemini_api_key)
        check_generation(client, settings)
        vector = check_embedding(client, settings)
        check_batch_embedding(settings)
        check_chroma(vector)
    except Exception as exc:  # smoke test: show any failure clearly
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
