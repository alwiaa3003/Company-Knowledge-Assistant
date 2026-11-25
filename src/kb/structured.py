"""Structured data path: answer lookup/filter/count questions over a CSV with pandas.

SAFETY: the LLM never writes code. It only fills in a JSON "query spec" (filters, group_by,
aggregation, sort, limit). We validate every field against the real column names and an allowlist
of operators, then WE execute it with plain pandas. No eval/exec anywhere.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from kb.config import Settings, load_settings
from kb.llm import Usage, generate
from kb.metrics import log_request

OPS = ("==", "!=", ">", ">=", "<", "<=", "contains", "in")
AGGS = ("none", "count", "sum", "mean", "min", "max")
MAX_ROWS_RETURNED = 25


class Filter(BaseModel):
    """One condition: <column> <op> <value>. For 'in', value is a comma-separated list."""

    column: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "contains", "in"]
    value: str


class QuerySpec(BaseModel):
    """What the LLM is allowed to ask for. Nothing else can be expressed, so nothing else can run."""

    filters: list[Filter] = Field(default_factory=list)
    group_by: Optional[str] = None
    aggregation: Literal["none", "count", "sum", "mean", "min", "max"] = "none"
    agg_column: Optional[str] = None
    select: list[str] = Field(default_factory=list)
    sort_by: Optional[str] = None
    sort_desc: bool = False
    limit: Optional[int] = None


class SpecError(ValueError):
    """The spec referenced a column/operator that doesn't exist or a value that doesn't fit."""


def is_text(series: pd.Series) -> bool:
    """True for text columns. pandas 3 uses StringDtype (not `object`) for text, so check both."""
    return pd.api.types.is_string_dtype(series) or series.dtype == object


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:  # parse ISO date columns so date comparisons work
        if is_text(df[col]) and df[col].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            df[col] = pd.to_datetime(df[col])
    return df


def load_table(path: Path) -> pd.DataFrame:
    """Load the CSV (cached until the file changes)."""
    return _load_cached(str(path), path.stat().st_mtime).copy()


def describe_table(df: pd.DataFrame, max_examples: int = 6) -> str:
    """Schema summary for the prompt: column, type, and a few example values."""
    lines = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            info = f"date, range {s.min().date()} .. {s.max().date()}"
        elif pd.api.types.is_numeric_dtype(s):
            info = f"number, range {s.min()} .. {s.max()}"
        else:
            uniq = s.dropna().unique()
            info = (f"text, values: {', '.join(map(str, uniq))}" if len(uniq) <= 12
                    else f"text, e.g. {', '.join(map(str, uniq[:max_examples]))}")
        lines.append(f"- {col}: {info}")
    return "\n".join(lines)


def _coerce(series: pd.Series, raw: str) -> Any:
    """Convert the spec's string value to the column's type (number/date/text)."""
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.Timestamp(raw)
        if pd.api.types.is_numeric_dtype(series):
            return float(raw)
    except (ValueError, TypeError) as exc:
        raise SpecError(f"value {raw!r} does not fit column type {series.dtype}") from exc
    return raw


def validate_spec(spec: QuerySpec, df: pd.DataFrame) -> None:
    """Raise SpecError if the spec mentions unknown columns or invalid combinations."""
    cols = set(df.columns)
    referenced = [f.column for f in spec.filters] + spec.select + [
        c for c in (spec.group_by, spec.agg_column, spec.sort_by) if c]
    unknown = [c for c in referenced if c not in cols]
    if unknown:
        raise SpecError(f"unknown column(s): {unknown}. Valid: {sorted(cols)}")
    for f in spec.filters:
        if f.op not in OPS:
            raise SpecError(f"operator {f.op!r} not allowed")
        if f.op in (">", ">=", "<", "<=") and is_text(df[f.column]):
            raise SpecError(f"cannot use {f.op} on text column {f.column}")
    if spec.aggregation in ("sum", "mean", "min", "max"):
        if not spec.agg_column:
            raise SpecError(f"aggregation {spec.aggregation} needs agg_column")
        if spec.aggregation in ("sum", "mean") and not pd.api.types.is_numeric_dtype(df[spec.agg_column]):
            raise SpecError(f"{spec.aggregation} needs a numeric column")
    if spec.limit is not None and spec.limit < 1:
        raise SpecError("limit must be >= 1")


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    """Apply every filter (AND). Text comparisons are case-insensitive."""
    mask = pd.Series(True, index=df.index)
    for f in filters:
        s = df[f.column]
        if f.op == "contains":
            m = s.astype(str).str.contains(f.value, case=False, regex=False)
        elif f.op == "in":
            values = [v.strip() for v in f.value.split(",") if v.strip()]
            if is_text(s):
                m = s.str.lower().isin([v.lower() for v in values])
            else:
                m = s.isin([_coerce(s, v) for v in values])
        else:
            if is_text(s):
                left, right = s.str.lower(), f.value.lower()
            else:
                left, right = s, _coerce(s, f.value)
            m = {"==": left == right, "!=": left != right, ">": left > right,
                 ">=": left >= right, "<": left < right, "<=": left <= right}[f.op]
        mask &= m
    return df[mask]


def execute_spec(spec: QuerySpec, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a validated spec. Returns (result_table, matched_rows)."""
    validate_spec(spec, df)
    matched = apply_filters(df, spec.filters)
    agg = spec.aggregation
    if agg == "none":
        result = matched[spec.select] if spec.select else matched
    elif spec.group_by:
        g = matched.groupby(spec.group_by)
        if agg == "count":
            result = g.size().reset_index(name="count")
        else:
            result = getattr(g[spec.agg_column], agg)().reset_index(name=f"{agg}_{spec.agg_column}")
    else:
        value = len(matched) if agg == "count" else getattr(matched[spec.agg_column], agg)()
        result = pd.DataFrame({agg if agg == "count" else f"{agg}_{spec.agg_column}": [value]})
    if spec.sort_by and spec.sort_by in result.columns:
        result = result.sort_values(spec.sort_by, ascending=not spec.sort_desc)
    elif agg != "none" and spec.group_by:
        # After grouping, sort_by may name a column that no longer exists; sort by the aggregate.
        result = result.sort_values(result.columns[-1], ascending=not spec.sort_desc)
    if spec.limit:
        result = result.head(spec.limit)
        if agg == "none":
            matched = result
    return result, matched


SPEC_SYSTEM = """You translate questions about a table into a JSON query spec. You never write code.
Spec fields:
- filters: list of {column, op, value}; op is one of == != > >= < <= contains in. Value is always a
  string; for 'in' use a comma-separated list; dates as YYYY-MM-DD. All filters are ANDed.
- group_by: column name or null. aggregation: none|count|sum|mean|min|max. agg_column: column for
  sum/mean/min/max. select: columns to show when aggregation is none (empty = all).
- sort_by, sort_desc, limit: optional. For "highest/top N" use sort_by + sort_desc=true + limit.
Use only the listed column names and the exact category spellings shown."""


def make_spec(settings: Settings, question: str, df: pd.DataFrame) -> tuple[QuerySpec, Usage]:
    """Ask Gemini for a QuerySpec in JSON mode and parse it with pydantic."""
    prompt = f"Table columns:\n{describe_table(df)}\n\nQuestion: {question}\nReturn the query spec."
    res = generate(settings, prompt, system=SPEC_SYSTEM, json_schema=QuerySpec)
    try:
        return QuerySpec.model_validate_json(res.text), res.usage
    except ValidationError as exc:
        raise SpecError(f"model returned an invalid spec: {exc.errors()[:2]}") from exc


def _jsonable(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> list of plain dicts (dates as strings, numpy numbers as Python numbers)."""
    return json.loads(df.head(MAX_ROWS_RETURNED).to_json(orient="records", date_format="iso"))


def phrase_answer(settings: Settings, question: str, result: pd.DataFrame, n_matched: int) -> tuple[str, Usage]:
    """Turn the computed result table into a short natural-language answer (no new facts)."""
    table = result.head(MAX_ROWS_RETURNED).to_csv(index=False, date_format="%Y-%m-%d")
    prompt = (f"Question: {question}\nComputed result ({n_matched} matching rows; table below may be "
              f"truncated to {MAX_ROWS_RETURNED} rows):\n{table}\n"
              "Answer the question in 1-3 sentences using ONLY this result. Use exact numbers. "
              "If the result is empty, say no matching rows were found.")
    res = generate(settings, prompt)
    return res.text, res.usage


def query_table(question: str, phrase: bool = True) -> dict[str, Any]:
    """Answer a question over the employees table. JSON-friendly; safe to expose as an agent tool.

    phrase=False skips the second LLM call that turns the result into a sentence ("answer" is then
    empty); agents use this because their own model writes the final answer (saves a free-tier call).

    Returns {"answer", "result": [rows], "spec": {...}, "sources": [{"file","rows"}],
             "route": "table", "usage": {...}}. On a bad spec, returns an "error" field instead of raising.
    """
    settings = load_settings()
    start = time.perf_counter()
    df = load_table(settings.table_file)
    usage = Usage()
    try:
        spec, u1 = make_spec(settings, question, df)
        usage += u1
        result, matched = execute_spec(spec, df)
    except SpecError as exc:
        return {"answer": f"I couldn't translate that into a safe table query ({exc}).", "result": [],
                "spec": None, "sources": [], "route": "table", "error": str(exc), "usage": usage.to_dict()}
    answer = ""
    if phrase:
        answer, u2 = phrase_answer(settings, question, result, len(matched))
        usage += u2
    row_numbers = [int(i) + 1 for i in matched.index]  # 1-based data row numbers in the CSV
    out = {
        "answer": answer,
        "result": _jsonable(result),
        "spec": spec.model_dump(),
        "sources": [{"file": settings.table_file.name, "rows": row_numbers[:MAX_ROWS_RETURNED],
                     "rows_matched": len(matched),
                     "label": f"{settings.table_file.name} ({len(matched)} rows matched)"}],
        "route": "table",
        "usage": {**usage.to_dict(), "latency_s": round(time.perf_counter() - start, 3)},
    }
    log_request(settings.logs_dir, {"route": "table", "question": question, "usage": out["usage"],
                                    "rows_matched": len(matched), "phrased": phrase})
    return out
