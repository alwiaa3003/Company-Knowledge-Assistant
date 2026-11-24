# KnowledgeLens: Company Knowledge Assistant (RAG)

Ask questions about company documents and an employee table, and get answers that **always cite
their source**: `[file p.N]` for PDFs, `employees.csv (N rows matched)` for the table.

- **Documents** → retrieval-augmented generation: chunk → embed (`gemini-embedding-2`) → Chroma → Gemini.
- **Table** questions (count / filter / average / top-N) → Gemini writes a validated JSON query
  spec → executed with pandas (the LLM never writes or runs code).
- A **router** picks the path. Every request logs latency, tokens and estimated cost.

## Resume description
> **KnowledgeLens: Company Knowledge Assistant (Python, Gemini, Chroma, pandas)**
> - Built a retrieval-augmented Q&A system over PDFs and a CSV that cites file + page for every answer and refuses when the context lacks the answer.
> - Implemented page-aware chunking, cached and rate-limited Gemini embeddings, and a persistent Chroma index, with one collection per chunking config for A/B tests.
> - Added an LLM router plus a safe structured-data path (JSON query spec validated against the schema and executed in pandas, no code execution).
> - Wrote a 15-question retrieval benchmark (hit rate@k, MRR) and ran one-variable experiments on chunk size, overlap and top-k.
> - Designed modular, JSON-friendly functions (`answer_question`, `query_table`) for reuse as Google ADK agent tools, with per-request latency, token and cost logging.

## Setup (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
Copy-Item .env.example .env            # then paste your key from https://aistudio.google.com/apikey
python smoke_test.py                   # key, generation, embedding, Chroma
```

## Data
```powershell
python scripts/download_data.py        # 3 PDFs into data/docs (skips existing)
python scripts/make_employees.py       # data/tables/employees.csv (60 fictional rows, seed 42)
python scripts/inventory.py            # files, pages, rows
```

## CLI
```powershell
python cli.py ingest                                   # baseline index: chunk 1000, overlap 200
python cli.py ingest --chunk-size 500 --overlap 100    # another config -> collection kb_cs500_ov100
python cli.py stats                                    # list collections and chunk counts
python cli.py ask "How many days of bereavement leave do I get?"
python cli.py ask "How many people in the Data team are based in Bengaluru?"
python cli.py ask "What is the difference between RAG-Sequence and RAG-Token?" -k 6
python cli.py chat                                     # interactive; type 'exit' to quit
```
`ask` prints the answer, sources, route (and who decided it), retrieved chunks with scores,
latency, tokens and estimated cost. `CHUNK_SIZE` / `CHUNK_OVERLAP` in `.env` select which
collection `ask` uses.

## Evaluation
```powershell
python eval/run_eval.py               # retrieval experiments -> eval/results_table.md
python eval/run_table_eval.py         # router + table answers -> eval/table_results.md
```
Results and analysis: [eval/results.md](eval/results.md).

## Use from Python (and as ADK tools)
```python
import sys; sys.path.insert(0, "src")
from kb.rag import answer_question
from kb.structured import query_table
answer_question("Can I accept a $200 gift from a supplier?", k=4)
# {"answer": "...", "sources": [{"file": "...pdf", "page": 3}], "route": "docs", "usage": {...}}
```

## Folder map
```
data/docs/            source PDFs (ACME handbook, RAG paper, Attention paper)
data/tables/          employees.csv
src/kb/
  config.py           settings from .env, cost estimates
  llm.py              Gemini client, retry/backoff, rate limiting, token usage
  ingest.py           PDF/txt/md -> page records {text, file, page}
  chunking.py         page-aware chunking with overlap, stable chunk ids
  embeddings.py       document/query embedding formats, batching, SQLite cache
  store.py            Chroma persistent collections (one per chunking config)
  retrieve.py         question -> top-k chunks with scores
  generate.py         grounded prompt, citations, "I don't know"
  structured.py       JSON query spec -> validated -> pandas (query_table)
  router.py           LLM router + keyword fallback
  rag.py              build_knowledge_base, answer_question
  metrics.py          per-request JSONL log (logs/requests.jsonl)
eval/                 eval_set.json, run_eval.py, run_table_eval.py, results
scripts/              download_data.py, make_employees.py, inventory.py
cli.py                command line
smoke_test.py         environment check
LEARNING.md           concepts, pitfalls and interview questions per milestone
chroma_db/ .cache/ logs/   generated, git-ignored
```

## Models (checked 2026-11-23)
`gemini-3.5-flash-lite` for generation, `gemini-embedding-2` (768 dims) for embeddings. Model
names change often, so check the [models page](https://ai.google.dev/gemini-api/docs/models). Switching
the embedding model means re-running `ingest` (old and new vectors aren't comparable).
Free-tier note: Google may use free-tier prompts to improve its products, so don't upload confidential documents.
