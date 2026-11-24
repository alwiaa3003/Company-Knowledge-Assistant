"""Vector store: a persistent Chroma database in chroma_db/.

One collection per chunking config (kb_cs1000_ov200, kb_cs500_ov200, ...), so experiments can be
compared side by side without overwriting each other. Distance = cosine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from kb.chunking import Chunk

ADD_BATCH = 1000


def get_db(chroma_dir: Path) -> chromadb.ClientAPI:
    """Open the persistent Chroma client (creates the folder on first use)."""
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_collection(chroma_dir: Path, name: str) -> Collection:
    """Return an existing collection; raises a clear error if it hasn't been built yet."""
    db = get_db(chroma_dir)
    try:
        return db.get_collection(name)
    except Exception as exc:  # Chroma raises different types across versions
        raise RuntimeError(f"Collection '{name}' not found. Run: python cli.py ingest") from exc


def build_index(chroma_dir: Path, name: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]],
                extra_metadata: dict[str, Any] | None = None) -> Collection:
    """(Re)create collection `name` and store chunks + vectors + file/page metadata.

    The collection is dropped first, so re-ingesting never leaves stale chunks behind.
    """
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    db = get_db(chroma_dir)
    if name in [c.name for c in db.list_collections()]:
        db.delete_collection(name)
    col = db.create_collection(name, metadata={"hnsw:space": "cosine", **(extra_metadata or {})})
    for i in range(0, len(chunks), ADD_BATCH):
        part = chunks[i:i + ADD_BATCH]
        col.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=[list(v) for v in vectors[i:i + ADD_BATCH]],
            metadatas=[{"file": c["file"], "page": c["page"], "chunk_index": c["chunk_index"]} for c in part],
        )
    return col


def query(collection: Collection, vector: Sequence[float], k: int = 4) -> list[dict[str, Any]]:
    """Return the k nearest chunks as {id, text, file, page, distance}."""
    res = collection.query(query_embeddings=[list(vector)], n_results=k,
                           include=["documents", "metadatas", "distances"])
    return [
        {"id": i, "text": d, "file": m["file"], "page": int(m["page"]), "distance": float(dist)}
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def collection_stats(chroma_dir: Path) -> list[dict[str, Any]]:
    """List every collection with its chunk count and metadata."""
    db = get_db(chroma_dir)
    out = []
    for c in db.list_collections():
        col = db.get_collection(c.name)
        out.append({"name": c.name, "chunks": col.count(), "metadata": col.metadata})
    return out
