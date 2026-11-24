"""Router: decide whether a question goes to the documents (RAG) or to the table (pandas).

Primary: a tiny Gemini classification call that sees the table's column names.
Fallback: a keyword heuristic, used if the API call fails or returns something unexpected.
"""

from __future__ import annotations

import re
from typing import Literal, Sequence

from google.genai import errors

from kb.config import Settings
from kb.llm import GeminiUnavailableError, Usage, generate

Route = Literal["docs", "table"]

_TABLE_HINTS = re.compile(
    r"\b(how many|count|number of|average|mean|total|sum|highest|lowest|max(imum)?|min(imum)?|"
    r"list (all|the)|which employees?|who (works|reports|joined|started)|salary|salaries|"
    r"headcount|employees? (in|on|at|based)|team|manager|reports? to|located|location)\b",
    re.IGNORECASE)
_DOC_HINTS = re.compile(r"\b(policy|policies|handbook|allowed|rule|paper|model|attention|retriev|"
                        r"transformer|leave|gift|expense|according to)\w*", re.IGNORECASE)


def heuristic_route(question: str, columns: Sequence[str]) -> Route:
    """Keyword fallback: table words or column names (and no doc words) -> 'table'."""
    q = question.lower()
    col_hit = any(c.lower().replace("_", " ") in q or c.lower() in q for c in columns)
    table_score = len(_TABLE_HINTS.findall(q)) + (1 if col_hit else 0)
    doc_score = len(_DOC_HINTS.findall(q))
    return "table" if table_score > doc_score else "docs"


def build_router_prompt(question: str, table_name: str, columns: Sequence[str]) -> str:
    """Classification prompt; the model must answer with one word."""
    return (
        "Route the user's question to ONE data source.\n"
        f"- table: the CSV '{table_name}' with one row per employee and columns: {', '.join(columns)}. "
        "Choose it for lookups, filters, counts, sums, averages or rankings over employees/teams/"
        "salaries/locations/managers/start dates.\n"
        "- docs: PDF documents (an employee policy handbook and research papers on RAG and Transformers). "
        "Choose it for policies, rules, procedures, definitions and research content.\n"
        f"Question: {question}\nAnswer with exactly one word: table or docs."
    )


def route(settings: Settings, question: str, table_name: str, columns: Sequence[str]) -> tuple[Route, str, Usage]:
    """Return (route, method, usage); method is 'llm' or 'heuristic'."""
    try:
        res = generate(settings, build_router_prompt(question, table_name, columns), max_output_tokens=200)
        word = res.text.strip().lower().strip(".\"' ")
        if word in ("docs", "table"):
            return word, "llm", res.usage  # type: ignore[return-value]
        return heuristic_route(question, columns), "heuristic", res.usage
    except (errors.APIError, GeminiUnavailableError, OSError) as exc:
        print(f"  [router] LLM routing failed ({type(exc).__name__}); using keyword heuristic")
        return heuristic_route(question, columns), "heuristic", Usage()
