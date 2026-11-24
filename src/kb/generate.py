"""Generation: build a grounded prompt from retrieved chunks and ask Gemini for a cited answer."""

from __future__ import annotations

import re
from typing import Any, Sequence

from kb.config import Settings
from kb.llm import GenResult, generate
from kb.retrieve import Retrieved

IDK = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are KnowledgeLens, a company knowledge assistant.
Rules:
1. Answer ONLY using the numbered context blocks provided. Do not use outside knowledge.
2. After every sentence that uses the context, cite its block like [file p.N], copying the file name
   and page exactly as shown in the block header, e.g. [acme_employee_policy_handbook.pdf p.3].
3. If the context does not contain the answer, reply with exactly: "{IDK}"
   Do not guess and do not add citations in that case.
4. Be concise: 1-4 sentences unless the question asks for a list."""

_CITATION_RE = re.compile(r"\[([^\[\]]+?\.(?:pdf|txt|md))\s*,?\s*p\.?\s*(\d+)\]", re.IGNORECASE)


def format_context(chunks: Sequence[Retrieved]) -> str:
    """Render chunks as numbered blocks with a citation header: [1] [file p.N]."""
    return "\n\n".join(f"[{i}] [{c['file']} p.{c['page']}]\n{c['text']}" for i, c in enumerate(chunks, start=1))


def build_prompt(question: str, chunks: Sequence[Retrieved]) -> str:
    """The user-turn prompt: context first, question last (models attend well to the end)."""
    return f"Context:\n{format_context(chunks)}\n\nQuestion: {question}\nAnswer:"


def extract_citations(answer: str) -> list[dict[str, Any]]:
    """Find [file p.N] citations in the answer, de-duplicated in order of appearance."""
    seen: list[dict[str, Any]] = []
    for file, page in _CITATION_RE.findall(answer):
        item = {"file": file.strip(), "page": int(page)}
        if item not in seen:
            seen.append(item)
    return seen


def is_idk(answer: str) -> bool:
    """True if the model declined to answer."""
    return "i don't know" in answer.lower() or "i do not know" in answer.lower()


def answer_from_chunks(settings: Settings, question: str, chunks: Sequence[Retrieved]) -> dict[str, Any]:
    """Generate a grounded answer. Returns {answer, sources, retrieved, usage(Usage)}.

    `sources` = the pages the model actually cited, restricted to pages we really retrieved
    (a citation to a page we never showed the model would be a hallucinated citation).
    If the model gave an answer but forgot to cite, we fall back to all retrieved pages.
    """
    if not chunks:
        return {"answer": IDK, "sources": [], "retrieved": [], "unverified_citations": [],
                "usage": GenResult("").usage}
    result = generate(settings, build_prompt(question, chunks), system=SYSTEM_PROMPT)
    retrieved_pages = []
    for c in chunks:
        p = {"file": c["file"], "page": c["page"]}
        if p not in retrieved_pages:
            retrieved_pages.append(p)
    all_cited = extract_citations(result.text)
    unverified = [c for c in all_cited if c not in retrieved_pages]  # hallucinated citations
    if is_idk(result.text):
        sources: list[dict[str, Any]] = []
    else:
        cited = [c for c in all_cited if c in retrieved_pages]
        sources = cited or retrieved_pages
    return {"answer": result.text, "sources": sources, "retrieved": retrieved_pages,
            "unverified_citations": unverified, "usage": result.usage}
