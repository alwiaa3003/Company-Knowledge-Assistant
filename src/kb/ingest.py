"""Ingestion: turn files on disk into page-level text records.

A record is {"text": str, "file": str, "page": int}. PDFs give one record per page (1-based);
.txt/.md files are treated as a single page 1. Empty pages are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


class PageRecord(TypedDict):
    """One page of text plus where it came from (used later for citations)."""

    text: str
    file: str
    page: int


def clean_text(text: str) -> str:
    """Lightly clean extracted text while keeping paragraph breaks.

    - joins words hyphenated across a line break ("retrie-\\nval" -> "retrieval")
    - collapses runs of spaces/tabs, trims spaces around newlines
    - keeps blank lines (paragraphs) but collapses 3+ newlines to 2
    - joins single line breaks inside a paragraph into spaces
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # A single newline that does not end a sentence/heading is just a PDF line wrap -> space.
    text = re.sub(r"(?<=[^\n.:;!?])\n(?=[a-z0-9(])", " ", text)
    return text.strip()


def load_pdf(path: Path) -> list[PageRecord]:
    """Read a PDF page by page; return cleaned, non-empty pages with 1-based page numbers."""
    reader = PdfReader(path)
    records: list[PageRecord] = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            records.append(PageRecord(text=text, file=path.name, page=i))
    return records


def load_text_file(path: Path) -> list[PageRecord]:
    """Read a .txt/.md file as a single page."""
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return [PageRecord(text=text, file=path.name, page=1)] if text else []


def load_file(path: Path) -> list[PageRecord]:
    """Dispatch on file extension. Unsupported files return an empty list."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        return load_text_file(path)
    return []


def load_directory(docs_dir: Path) -> list[PageRecord]:
    """Load every supported file in `docs_dir` (sorted, so the order is deterministic)."""
    records: list[PageRecord] = []
    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            records.extend(load_file(path))
    return records
