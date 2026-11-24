"""Embeddings: turn text into vectors with the Gemini embedding model, with a disk cache.

gemini-embedding-2 has no `task_type` parameter. Instead the task goes INTO the text:
  documents -> "title: {title} | text: {content}"
  queries   -> "task: search result | query: {content}"
Using the same format for both sides would make questions and passages look less alike.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Sequence

from google.genai import types

from kb.config import Settings
from kb.llm import call_with_retry, get_client, get_limiter

BATCH_SIZE = 50  # texts per API request


def format_document(text: str, title: str | None = None) -> str:
    """Document-side prompt format for gemini-embedding-2."""
    return f"title: {title or 'none'} | text: {text}"


def format_query(text: str) -> str:
    """Query-side prompt format for gemini-embedding-2."""
    return f"task: search result | query: {text}"


def cache_key(model: str, dim: int, formatted_text: str) -> str:
    """Stable cache key: the same model + dimension + exact input text -> the same vector."""
    return hashlib.sha256(f"{model}|{dim}|{formatted_text}".encode("utf-8")).hexdigest()


def _open_cache(cache_dir: Path) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite embedding cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_dir / "embeddings.sqlite")
    conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
    return conn


def _to_blob(vec: Sequence[float]) -> bytes:
    import array
    return array.array("f", vec).tobytes()


def _from_blob(blob: bytes) -> list[float]:
    import array
    arr = array.array("f")
    arr.frombytes(blob)
    return arr.tolist()


def to_contents(texts: list[str]) -> list[types.Content]:
    """Wrap EACH text in its own Content object.

    gemini-embedding-2 treats a plain list of strings as the parts of ONE input and returns a
    single aggregated embedding. Separate Content objects -> one embedding per text (per the docs).
    """
    return [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]


def _embed_batch(settings: Settings, texts: list[str]) -> list[list[float]]:
    """One API request for up to BATCH_SIZE already-formatted texts -> one vector per text."""
    client = get_client(settings.gemini_api_key)
    config = types.EmbedContentConfig(output_dimensionality=settings.embed_dim)
    contents = to_contents(texts)
    resp = call_with_retry(
        lambda: client.models.embed_content(model=settings.embed_model, contents=contents, config=config),
        get_limiter("embed", settings.embed_rpm),
    )
    vectors = [list(e.values) for e in resp.embeddings]
    if len(vectors) != len(texts):
        raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
    return vectors


def embed_formatted(settings: Settings, formatted: list[str]) -> tuple[list[list[float]], int]:
    """Embed already-formatted texts, using the cache. Returns (vectors, n_api_texts_embedded)."""
    keys = [cache_key(settings.embed_model, settings.embed_dim, t) for t in formatted]
    with closing(_open_cache(settings.cache_dir)) as conn:
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            rows = conn.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part)
            found.update({k: _from_blob(v) for k, v in rows})
        missing = [i for i, k in enumerate(keys) if k not in found]
        for start in range(0, len(missing), BATCH_SIZE):
            idx = missing[start:start + BATCH_SIZE]
            vectors = _embed_batch(settings, [formatted[i] for i in idx])
            for i, vec in zip(idx, vectors):
                found[keys[i]] = vec
            conn.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                             [(keys[i], _to_blob(v)) for i, v in zip(idx, vectors)])
            conn.commit()  # commit per batch: a crash halfway keeps the work already paid for
    return [found[k] for k in keys], len(missing)


def embed_documents(settings: Settings, texts: list[str], titles: list[str] | None = None) -> list[list[float]]:
    """Embed document chunks (document-side format). `titles` is one title per text, optional."""
    titles = titles or [None] * len(texts)  # type: ignore[list-item]
    vectors, _ = embed_formatted(settings, [format_document(t, ti) for t, ti in zip(texts, titles)])
    return vectors


def embed_query(settings: Settings, text: str) -> list[float]:
    """Embed one user question (query-side format)."""
    vectors, _ = embed_formatted(settings, [format_query(text)])
    return vectors[0]
