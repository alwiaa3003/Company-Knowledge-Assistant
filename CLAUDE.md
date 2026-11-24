# KnowledgeLens — Company Knowledge Assistant (RAG)

Day 1 of a 3-day prep for a Google Cloud "AI Consulting Engineer" early-career role.
The user is a beginner at RAG and wants to learn while building. On Day 2 this project
is reused as a tool inside a multi-agent system built with Google ADK, so the code must stay modular.

## Goal
A Q&A assistant that answers questions from the user's own documents and ALWAYS cites its
sources (file name + page number for PDFs, file + row for CSV).

## Requirements
1. Ingest unstructured data (PDFs / text files in `data/docs/`) AND structured data (a CSV in
   `data/tables/`, optionally SQLite).
2. Chunk the text (chunk size and overlap are parameters), create embeddings, store them in a
   local persistent Chroma database (`chroma_db/`).
3. Retrieve the top-k chunks for a question and send them to Gemini (free Google AI Studio key).
   The prompt must tell the model to answer ONLY from the context and to say "I don't know" otherwise.
4. Show the source (file name/page) for every answer.
5. Route questions: document questions → RAG; structured-data questions (lookup/filter/count
   on the CSV) → a pandas-based path. A simple router decides.
6. Eval set of ~15 question/expected-source pairs (`eval/eval_set.json`) + a script measuring
   retrieval hit rate@k. Change one parameter (chunk size, overlap or top-k) and compare in a small table.
7. CLI first (`cli.py`). Streamlit only if there is time.

## Tech
Python 3.10+ (local machine: 3.13, Windows, venv in `.venv`), chromadb, **google-genai**
(NOT the deprecated google-generativeai), pypdf, pandas, python-dotenv. Everything local and free.
Config lives in `.env` (GEMINI_API_KEY, GEN_MODEL, EMBED_MODEL).

### Model choices (checked against official docs on 2026-11-23 — re-check, they change often)
- GEN_MODEL = `gemini-3.5-flash-lite` (stable, free tier, cheap). Do NOT use `gemini-2.5-flash`
  (deprecated; auto-routed from 2026-12-01).
- EMBED_MODEL = `gemini-embedding-2`, output_dimensionality = 768.
  `gemini-embedding-2` does NOT support `task_type`. Put the task in the text instead:
  - query: `task: search result | query: {text}`
  - document: `title: {title or "none"} | text: {text}`
  Embeddings from different embedding models are incompatible → re-embed (rebuild chroma_db) when switching.
- Paid-tier prices used for cost *estimates* (USD / 1M tokens): 3.5-flash-lite $0.30 in / $2.50 out;
  gemini-embedding-2 $0.20 text input.

## Structure
```
data/docs/  data/tables/  eval/  chroma_db/
src/kb/  config.py ingest.py chunking.py embeddings.py store.py retrieve.py
         generate.py structured.py router.py rag.py
cli.py  smoke_test.py  requirements.txt  .env.example  .gitignore  README.md
```

## Design rules for ADK reuse
- Each module = small, pure, well-typed functions with docstrings. No logic at import time.
- `rag.py` exposes `answer_question(question: str, k: int = 4) -> dict` returning
  `{"answer": str, "sources": [{"file": ..., "page": ...}], "route": "docs"|"table", "usage": {...}}`.
- `structured.py` exposes `query_table(question: str) -> dict`.
- These two become ADK tools on Day 2 → simple, JSON-friendly signatures.
- Log latency, input/output tokens, and estimated cost per request.

## How to work with the user (IMPORTANT — TEACHING MODE)
- One milestone at a time: (1) setup, (2) ingestion, (3) chunking, (4) embeddings + vector store,
  (5) retrieval, (6) generation with citations, (7) structured data + router, (8) evaluation.
- For each milestone: explain the concept in simple words → show the code → explain how to test it.
- Do NOT write all the code at once. Do NOT move to the next milestone until the user says "next".
- Code-writing mode: Claude writes the files, but explains each one and what the user should learn
  from it. (Default chosen in Milestone 1; the user may switch to typing the code themselves.)
- When the user hits an error, explain WHY it happened, not just the fix.
- End every milestone with one interview-style question and wait for the answer before giving feedback.

### Override (2026-11-23, session 2)
The user explicitly OVERRODE the "one milestone at a time / wait for next" rule for one session:
build Milestones 2-8 end to end autonomously, teach through LEARNING.md instead, and only stop
when blocked (e.g. download failure, API key rejected). The teaching-mode rules above still apply
by default in future sessions unless the user says otherwise.

## Progress
- [x] Milestone 1: setup (CLAUDE.md, folders, .gitignore, .env.example, requirements, venv, smoke_test.py)
- [x] Data: 3 PDFs downloaded + verified (scripts/download_data.py), employees.csv generated
      (scripts/make_employees.py, seed 42, 60 rows, consistency-checked), scripts/inventory.py
- [x] Milestone 2: ingestion (ingest.py), tested offline: 44 pages
- [x] Milestone 3: chunking (chunking.py), tested offline: 191 chunks at 1000/200
- [x] Milestone 4: embeddings + store (embeddings.py, store.py, llm.py) — real API verified 2026-11-24
- [x] Milestone 5: retrieval (retrieve.py)
- [x] Milestone 6: generation + citations (generate.py, metrics.py) — real end-to-end verified
- [x] Milestone 7: structured + router (structured.py, router.py, rag.py, cli.py) — table eval 7/7
- [x] Milestone 8: eval run 2026-11-24; results in eval/results.md (baseline 1000/200/k4 = hit@1 1.00,
      MRR 1.000; ceiling effect, eval set needs harder questions to discriminate)
- [x] Docs: LEARNING.md (incl. results section), README.md
- Next (Day 2): wrap `answer_question` and `query_table` as Google ADK tools.

## Implementation notes (things not obvious from the code)
- gemini-embedding-2 AGGREGATES a plain list of strings into ONE embedding. Always send one
  `types.Content` per text (`embeddings.to_contents`). Keep the count assertion in `_embed_batch`.
- Free-tier embed quota (`embed_content_free_tier_requests`) counts each text in a batch; 429s are
  expected during ingestion and handled by `call_with_retry` using the server's retry delay.
- `call_with_retry` also retries network errors (httpx.TransportError etc., 5 attempts) and raises
  `GeminiUnavailableError`; never retries 400/401/403.
- Latency in CLI output is dominated by the self-imposed GEN_RPM=10 pacing (≥6 s between calls).
- Grouped table queries always sort by the aggregate if sort_by isn't a result column (bug fixed 2026-11-24).
- Never print the API key, not even partially (smoke_test no longer shows the last 4 chars).
- pandas 3 uses StringDtype for text: never test `dtype == object`; use `structured.is_text()`.
- Eval writes the raw table to eval/results_table.md; eval/results.md = table + hand-written analysis.
- `.cache/embeddings.sqlite` holds the embedding cache (keyed by model|dim|formatted text).
