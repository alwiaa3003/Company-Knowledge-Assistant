"""Retrieval: question -> top-k most similar chunks (with their citation info)."""

from __future__ import annotations

from typing import TypedDict

from kb import store
from kb.config import Settings
from kb.embeddings import embed_query


class Retrieved(TypedDict):
    """A retrieved chunk. score = 1 - cosine distance (1.0 = identical direction)."""

    id: str
    text: str
    file: str
    page: int
    score: float


def retrieve(settings: Settings, question: str, k: int = 4, collection: str | None = None) -> list[Retrieved]:
    """Embed the question and return the k nearest chunks from `collection` (default from config)."""
    col = store.get_collection(settings.chroma_dir, collection or settings.default_collection)
    hits = store.query(col, embed_query(settings, question), k)
    return [Retrieved(id=h["id"], text=h["text"], file=h["file"], page=h["page"],
                      score=round(1.0 - h["distance"], 4)) for h in hits]
