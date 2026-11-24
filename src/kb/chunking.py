"""Chunking: split page text into overlapping pieces small enough to embed and retrieve.

Sizes are in CHARACTERS. We prefer to cut at paragraph breaks, then sentence ends, then spaces,
and never let a chunk cross a page, so every chunk cites exactly one file + page.
"""

from __future__ import annotations

from typing import Iterable, TypedDict

from kb.ingest import PageRecord

# Separators from "best" to "worst" place to cut.
_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


class Chunk(TypedDict):
    """A retrievable unit: text plus a stable id and citation metadata."""

    id: str
    text: str
    file: str
    page: int
    chunk_index: int


def _find_cut(text: str, start: int, end: int, min_size: int) -> int:
    """Return the best cut position in text[start:end], searching backwards for a separator.

    A cut must leave at least `min_size` characters, otherwise we cut hard at `end`.
    """
    window = text[start:end]
    for sep in _SEPARATORS:
        pos = window.rfind(sep)
        if pos >= min_size:
            return start + pos + len(sep)
    return end


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` characters with ~`overlap` shared chars.

    Each new chunk starts `overlap` characters before the previous chunk's end (moved forward to
    the next word boundary so we don't start mid-word).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _find_cut(text, start, end, min_size=chunk_size // 2)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        if overlap:
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return chunks


def chunk_records(records: Iterable[PageRecord], chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """Chunk every page record. Ids look like `file.pdf::p3::c0` and are stable across runs."""
    out: list[Chunk] = []
    for rec in records:
        for i, piece in enumerate(chunk_text(rec["text"], chunk_size, overlap)):
            out.append(Chunk(id=f"{rec['file']}::p{rec['page']}::c{i}", text=piece,
                             file=rec["file"], page=rec["page"], chunk_index=i))
    return out
