# Learning KnowledgeLens: RAG from the ground up

This guide walks through the 8 milestones of the project. For each one: the idea in plain words,
where it lives in the code, one thing that commonly goes wrong, and an interview-style question.
Try to answer each question out loud **before** opening the model answer.

The whole system in one picture:

```
            question
               │
          ┌────▼────┐   "docs"   ┌──────────┐   ┌───────────┐   ┌──────────┐
          │ router  ├───────────►│ retrieve ├──►│  generate ├──►│ answer + │
          └────┬────┘            │ (Chroma) │   │ (Gemini)  │   │ [file p.N]│
               │ "table"         └────▲─────┘   └───────────┘   └──────────┘
          ┌────▼──────────┐           │ offline, once:
          │ structured.py │      ingest → chunk → embed → store
          │ LLM → JSON    │
          │ spec → pandas │
          └───────────────┘
```

---

## Milestone 1: Setup

**Concept.** Before writing any AI logic, make every external dependency prove it works: the API
key loads, one generation call succeeds, one embedding comes back with the right size, and the
vector database can store and find a vector. A *smoke test* does exactly that. Secrets live in
`.env` (never committed); settings are read in one place, so the rest of the code never touches
environment variables directly.

**Files.** [smoke_test.py](smoke_test.py), [src/kb/config.py](src/kb/config.py), [.env.example](.env.example), [.gitignore](.gitignore)

**What can go wrong.** The key is set in one terminal or folder but not where the code runs
(`.env` is read relative to the project root). The code fails late with a confusing 401/403 error.
`load_settings()` fails *early* with a clear message instead.

**Interview question.** Why does `config.py` do nothing at import time, and why does
`load_settings()` raise an error when the key is missing instead of letting the first API call fail?

### Model answer
<details><summary>Show</summary>

Importing a module should be free of side effects, so other code (tests, an ADK agent, a CLI
`--help`) can import it without needing secrets or network. Failing fast at the boundary gives one
clear, actionable error ("set GEMINI_API_KEY in .env"). Otherwise you get an HTTP error deep inside a
call stack that might be mistaken for a quota or model problem. It also avoids spending time (or
money) on work that will fail anyway.
</details>

---

## Milestone 2: Ingestion

**Concept.** Ingestion turns files into clean text *with provenance*. We read PDFs page by page
and keep `{text, file, page}` for every page, because citations need to know where each sentence
came from. Light cleaning fixes PDF artefacts: words hyphenated across lines are re-joined, extra
spaces collapsed, and single line-wraps turned into spaces, while blank lines (paragraphs) are kept.
Chunking uses those paragraph breaks later.

**Files.** [src/kb/ingest.py](src/kb/ingest.py) (`load_pdf`, `clean_text`, `load_directory`), [scripts/download_data.py](scripts/download_data.py)

**What can go wrong.** Scanned PDFs have no text layer, so `extract_text()` returns `""` and the
document silently contributes nothing. That's why the download script prints the first 300
characters of each PDF. Another trap: aggressive cleaning destroys structure. The ACME gift
*table* comes out as one value per line. Joining everything into one paragraph would make
"$76 – $250 → disclose" impossible to read. Also, de-hyphenation is imperfect: "region-specific"
split across lines becomes "regionspecific".

**Interview question.** Why do we keep page numbers at ingestion time instead of just storing
each document's full text?

### Model answer
<details><summary>Show</summary>

Provenance can't be recovered later. Once text from different pages is merged, you can no longer
say which page a fact came from. Page-level records let every chunk carry exactly one
`file + page`, which makes citations precise and lets us *evaluate* retrieval (did we fetch the
right page?). In production you'd keep even richer metadata: section title, document version,
access permissions, last-updated date.
</details>

---

## Milestone 3: Chunking

**Concept.** Embedding a whole 19-page paper as one vector would blur every topic together.
Chunking cuts text into pieces small enough to be about one thing. **chunk_size** is the
maximum characters per chunk. **overlap** is how many characters consecutive chunks share, so a
sentence on a boundary still appears whole in at least one chunk. We try to cut at the "best"
separator (paragraph break, then line break, then sentence end, then space) instead of
mid-word. A chunk never crosses a page, and each gets a stable id like
`acme_employee_policy_handbook.pdf::p3::c1`.

**Files.** [src/kb/chunking.py](src/kb/chunking.py) (`chunk_text`, `chunk_records`)

**What can go wrong.** Chunks that are too small lose context ("up to $125" without saying *what*
costs $125). Chunks that are too large mix several topics, so their vector matches nothing strongly.
Without overlap, a fact split across a boundary is half in each chunk and fully in neither.
Our "never cross a page" rule has a cost too: the ACME gift table starts on p.3 and continues
on p.4, so its header and rows end up in different chunks.

**Interview question.** You double the chunk size. What happens to the number of chunks, the
cost of each LLM answer, and retrieval precision?

### Model answer
<details><summary>Show</summary>

Roughly half as many chunks, so indexing is cheaper. But each retrieved chunk carries twice the
text, so with the same k the prompt is ~2× bigger (more input tokens, higher cost and latency).
Precision often drops because each chunk covers several topics and its embedding gets "averaged".
Recall can rise because the answer is more likely to be *somewhere* inside a big chunk. Which
effect wins depends on the documents, and that's why you measure it (Milestone 8) instead of
guessing.
</details>

---

## Milestone 4: Embeddings + vector store

**Concept.** An embedding model maps text to a vector (here 768 numbers). Texts with similar
*meaning* land close together, even with different words ("close family member dies" ≈
"bereavement leave"). We store every chunk's vector in **Chroma**, a local vector database, with
`file` and `page` as metadata, using **cosine distance** (angle between vectors).

Two details specific to `gemini-embedding-2`:
- It has **no `task_type`** parameter. The task goes into the text: documents are embedded as
  `title: <doc title> | text: <chunk>`, questions as `task: search result | query: <question>`.
  Questions and passages look different (short vs long, question vs statement), and the prefixes
  tell the model which side is which.
- We ask for **768 dimensions** instead of the default 3072: 4× less storage, small quality loss.

We also **cache** every vector on disk (SQLite, keyed by a hash of model + dimension + exact text).
Re-running ingestion or experiments costs zero API calls for text we've already embedded.
Embedding requests are **batched** (50 texts per call), **rate-limited**, and **retried with
exponential backoff** on 429/5xx.

**Files.** [src/kb/embeddings.py](src/kb/embeddings.py), [src/kb/store.py](src/kb/store.py), [src/kb/llm.py](src/kb/llm.py) (`call_with_retry`, `RateLimiter`)

**What can go wrong.** Mixing embedding models: vectors from `gemini-embedding-001` and
`gemini-embedding-2` live in different spaces, and comparing them gives nonsense *without any
error*. That's why the cache key includes the model name and dimension, and the collection
metadata records `embed_model`. Switch models → rebuild the index.

**A real bug from this project: the aggregation trap.** With `gemini-embedding-2`, passing a
plain list of strings (`contents=["a", "b", "c"]`) does **not** give three embeddings. The
docs say it "produces a single aggregated embedding for multiple inputs". Our first ingest asked for
50 vectors and got 1. The fix (from the docs): wrap each text in its own `types.Content` object
(`embeddings.to_contents`). Two safeguards caught it before bad data was stored: the count
assertion in `_embed_batch`, and a smoke test that embeds 3 different texts in one batch and checks
it gets 3 *different* vectors. Lesson: assert the shape of what an API returns, especially
after a model upgrade, because the same SDK call can change meaning between models.

The free tier also counts **each text in a batch** against the per-minute embedding request quota
(`embed_content_free_tier_requests`). A 50-text batch is "50 requests", so the 429 retry with the
server-suggested delay is what keeps ingestion going.

**Interview question.** Why do we retry 429 and 503 errors but *not* 400 or 403?

### Model answer
<details><summary>Show</summary>

429 (rate limit) and 5xx (server overloaded) are *transient*: the same request will probably
succeed if you wait, so retry with exponential backoff (plus jitter, so many clients don't retry
in lockstep), and honour the server's suggested retry delay. 400 (bad request) and 401/403
(bad or unauthorised key) are *permanent*: retrying sends the same broken request again, wastes
time and quota, and can hide the real bug.
</details>

---

## Milestone 5: Retrieval

**Concept.** At question time: embed the question (query format), ask Chroma for the **k**
nearest chunks, and return them with their score (`1 − cosine distance`, higher = more similar)
and citation info. **k** is a trade-off: more chunks = higher chance the answer is included
(recall), but a longer, noisier, more expensive prompt.

**Files.** [src/kb/retrieve.py](src/kb/retrieve.py) (`retrieve`)

**What can go wrong.** Pure vector search is weak on exact tokens: phone numbers, IDs, product
codes, rare acronyms. "What's the number for +1 (800) 555-0199?" may not retrieve the page that
contains exactly that string. That's what *hybrid search* (vector + keyword/BM25) fixes. The RAG
paper itself found BM25 beat dense retrieval on FEVER, which is entity-heavy (p.7).

**Interview question.** A user complains: "the answer is in the handbook, but the bot says it
doesn't know." How do you debug it?

### Model answer
<details><summary>Show</summary>

Separate the two halves. (1) Retrieval: look at the retrieved chunks (the CLI prints them with
scores). Is the right page in the top-k? Check whether it's in the top 10 or 20. If it's not
there, the problem is chunking or embeddings: try a different chunk size, more k, hybrid search,
or query rewriting. (2) Generation: if the right chunk *was* retrieved but the model still said
"I don't know", the prompt or the model is too strict, or the fact is split across chunks. Add
the failing question to the eval set so the fix is measured and stays fixed.
</details>

---

## Milestone 6: Generation with citations

**Concept.** We give Gemini numbered context blocks, each with a header like
`[acme_employee_policy_handbook.pdf p.8]`, plus strict rules in the system prompt: answer
**only** from the context, cite every claim as `[file p.N]`, and reply exactly
*"I don't know based on the provided documents."* when the context doesn't contain the answer.
This is **grounding**. Afterwards we parse the citations, keep only those that point to pages we
actually retrieved, and flag the rest as `unverified_citations` (a hallucinated citation).
Each request logs latency, input/output tokens (thinking tokens count as output) and an
estimated cost to `logs/requests.jsonl`.

**Files.** [src/kb/generate.py](src/kb/generate.py) (`SYSTEM_PROMPT`, `build_prompt`, `answer_from_chunks`), [src/kb/metrics.py](src/kb/metrics.py)

**What can go wrong.** The model mixes in its own training knowledge, especially for the
research papers, which it has probably read. The answer might be *correct* but not *grounded*.
Temperature 0 and strict instructions help, but only a faithfulness eval proves it (see
"production" below). The opposite failure: over-refusal, saying "I don't know" when the answer
is there but phrased differently.

**Interview question.** How would you check that an answer's citation actually supports the
claim, not just that the cited page was retrieved?

### Model answer
<details><summary>Show</summary>

We currently check that the cited page was *retrieved*. That catches invented citations, but not
"real page, wrong claim". To check *support*, split the answer into claims and test each against
the cited chunk's text. Use an NLI/entailment model or an LLM-as-judge prompt ("Does this passage
support this sentence? yes/no + quote"), and report a faithfulness score. Tools like Ragas or
Vertex AI's evaluation service implement this. Humans spot-check a sample to calibrate the judge.
</details>

---

## Milestone 7: Structured data + router

**Concept.** Vector search is the wrong tool for "how many people in Data work in Bengaluru?":
the answer isn't written in any document, it has to be **computed**. So a **router** first decides
the path. It's a tiny Gemini call that sees the CSV's column names and must answer `docs` or
`table`, with a keyword heuristic as fallback if the call fails.

The table path is built for **safety**. The LLM never writes code. It fills in a JSON *query spec*
(filters with an allowlisted operator, group_by, an aggregation from count/sum/mean/min/max,
sort, limit), enforced with a pydantic schema in Gemini's JSON mode. We validate every column
name against the real CSV, then execute the spec ourselves with ordinary pandas. No
`eval`/`exec` anywhere. A second small call phrases the computed result as a sentence, and the
source is `employees.csv (N rows matched)` plus the row numbers.

**Files.** [src/kb/router.py](src/kb/router.py), [src/kb/structured.py](src/kb/structured.py) (`QuerySpec`, `validate_spec`, `execute_spec`, `query_table`), [src/kb/rag.py](src/kb/rag.py) (`answer_question`), [scripts/make_employees.py](scripts/make_employees.py)

**What can go wrong.** (1) Letting an LLM write pandas/SQL and running it: prompt injection
becomes code execution ("ignore the question and delete files"). (2) Ambiguous questions: "How
many days of bereavement leave?" contains "how many" but belongs to the docs. A pure keyword
router gets this wrong, which is why the LLM router is primary. (3) A real bug we hit: in pandas 3,
text columns have `StringDtype`, not `object`. A check like `dtype == object` silently failed, so
`team == "data"` matched 0 rows instead of 14. Fixed with `is_text()` in `structured.py`.
(4) Another real bug, found by the eval: for "Which team has the most employees?" the model asked to
sort by `employee_id`, which doesn't exist after grouping. The sort was skipped silently, and
"top 1" became the first team alphabetically. That was Data, which is also the biggest team,
so the test **passed by luck**. We only caught it by reading the spec, not the score. A "fewest
employees" test now guards it. Lesson: a passing test is only evidence if it could have failed.
(5) Network errors (DNS `getaddrinfo failed`) have no HTTP status code at all. `call_with_retry`
retries them (5 attempts, exponential backoff) separately from 429/5xx, and never retries 400/401/403.

**Interview question.** Why use a JSON query spec instead of asking the LLM to write the pandas
code directly? What do you give up?

### Model answer
<details><summary>Show</summary>

Safety and verifiability. The spec can only express what we allow: known columns, allowlisted
operators, a fixed set of aggregations. So a malicious or confused model can't run arbitrary code,
read files or exfiltrate data. Specs are also easy to log, validate, cache and unit-test. The
cost is expressiveness: no joins, computed columns (e.g. tenure in years), OR-conditions or
nested queries, unless we extend the schema. In production you'd grow the spec deliberately, or
use text-to-SQL against a **read-only** database user with row limits and timeouts.
</details>

---

## Milestone 8: Evaluation

**Concept.** "It looks like it works" is not evidence. We wrote 15 questions whose correct
**source page** we know from reading the PDFs (8 handbook, 4 RAG paper, 3 Attention paper).
Several are paraphrased on purpose ("close family member dies" instead of "bereavement"), so
keyword overlap can't hide weak retrieval. Metrics:
- **hit rate@k**: the share of questions where at least one of the top-k chunks comes from the
  expected file *and* page.
- **MRR** (mean reciprocal rank): average of 1/rank of the first correct chunk (1st = 1.0,
  2nd = 0.5, absent = 0). It rewards putting the right chunk *first*.

Then we change **one variable at a time** against a baseline (chunk 1000, overlap 200, k 4). If
you change two at once, you can't tell which one caused the difference. This evaluates
*retrieval* only, which is cheap (no generation) and the most common failure point. The table
path has its own 6-question test with ground truth computed directly in pandas.

**Files.** [eval/eval_set.json](eval/eval_set.json), [eval/run_eval.py](eval/run_eval.py), [eval/run_table_eval.py](eval/run_table_eval.py), [eval/results.md](eval/results.md)

**What can go wrong.** An eval set written by guessing ("the policy is probably on page 5")
measures nothing. Every expected page here was checked against the extracted text. Also:
15 questions is tiny. One question = 6.7 percentage points, so small differences are noise, not
signal.

**Interview question.** Hit rate@5 is 0.93 but users still get wrong answers. How is that possible,
and what would you measure next?

### Model answer
<details><summary>Show</summary>

Hit rate only says the right page was *somewhere* in the top 5. The model can still ignore it,
get distracted by the other 4 chunks, misread a table, or blend in outside knowledge. Measure
generation next: answer correctness against reference answers, **faithfulness** (is every claim
supported by the context?), citation accuracy, and refusal behaviour on unanswerable questions.
Also check whether the eval set looks like real user questions. Mine production logs for real
queries and add the failures.
</details>

---

## Experiment results explained

Full tables: [eval/results.md](eval/results.md). The short version:

| Config (one change vs baseline) | hit@1 | hit@4 | MRR | context sent per question |
|---|---|---|---|---|
| baseline: chunk 1000, overlap 200, k 4 | 1.00 | 1.00 | 1.000 | ~837 tokens |
| chunk 500 | 0.87 | 1.00 | 0.933 | ~430 tokens |
| chunk 2000 | 0.67 | 1.00 | 0.822 | ~1,485 tokens |
| overlap 0 | 0.87 | 1.00 | 0.933 | ~789 tokens |
| k 2 / k 8 | 1.00 | 1.00 | 1.000 | ~410 / ~1,646 tokens |

**How to read it.**
1. **Ceiling effect.** The baseline is already perfect, so experiments could only get *worse*.
   With 15 questions, one question = 6.7 points, and some rank flips came down to similarity
   differences of 0.003. That's a lesson in itself: an eval that every config passes can't help you
   choose. You need harder questions, more documents (distractors) and more items.
2. **Chunk size has a sweet spot.** Too small (500): the chunk with the keywords lacks the context
   that makes it the *best* match. Too big (2000): each chunk mixes several handbook sections, so its
   vector is an average of topics. A 2000-character chunk about privacy beat the actual "lost device"
   rule. Big chunks also cost ~1.8× more input tokens.
3. **Overlap protects facts at chunk boundaries.** Removing it cost two first places.
4. **k is a cost/safety dial, not a quality dial here.** Everything was found by rank 3, so k=2 matched
   k=4 at half the tokens, and k=8 doubled the tokens for nothing. k=4 stays the default as a margin.
5. **The real problem was structural.** The gift table crosses from p.3 to p.4, and "never cross a
   page" split the header from the rows. In 3 of 4 configs, p.4 outranked p.3 for the wine
   question. No parameter fixes that. Table-aware or heading-aware chunking does.
6. **The table route and the router** scored 7/7 and 7/7 on table questions, and the LLM router sent
   all 15 doc questions to docs. The keyword *fallback* only managed 12/15: it sends "how many
   business days…" and "maximum per-person amount…" to the table. Keyword routing can't tell
   "count rows" from "a number written in a policy".

---

## What I'd improve for production

- **Hybrid search.** Combine vector similarity with keyword search (BM25) and merge the rankings
  (e.g. reciprocal rank fusion). It fixes exact-token queries (phone numbers, IDs, acronyms).
  On Google Cloud: Vertex AI Vector Search hybrid queries, or AlloyDB/Postgres with pgvector + full-text.
- **Reranking.** Retrieve 20–50 candidates cheaply, then rerank them with a cross-encoder
  (e.g. Vertex AI Ranking API) and keep the top 4. Precision goes up without a bigger prompt.
- **Metadata filters.** Filter before or while searching by document type, department, version,
  date or user permissions ("only HR policies", "only docs this user may see"). Chroma supports
  `where=` filters on the `file`/`page` metadata we already store.
- **Better chunking.** Structure-aware chunking (by heading/section), keep tables whole across
  page breaks, and attach the section title to every chunk.
- **Evaluate generation faithfulness.** LLM-as-judge or NLI checks that each claim is supported
  by its cited chunk, plus answer correctness and refusal tests (Ragas, Vertex AI Gen AI evaluation
  service). Run the evals in CI so a prompt or model change can't silently regress quality.
- **Caching.** We already cache embeddings. Add a response cache for repeated questions
  (exact match, or semantic match with a high threshold), and use context caching for long
  repeated prompts.
- **Tracing and observability.** We log latency, tokens and cost per request to JSONL. In
  production, emit OpenTelemetry spans for router → retrieve → generate (Cloud Trace), add
  dashboards and alerts on latency, cost and "I don't know" rate, and record which chunks were
  used, so bad answers can be debugged.
- **Security.** Treat retrieved text as untrusted (prompt-injection in documents), enforce
  per-user access control on retrieval, keep secrets in Secret Manager, and keep the
  no-code-execution design for structured data.
- **Scale.** A managed vector store (Vertex AI Vector Search / RAG Engine), batch embedding jobs,
  incremental re-indexing when documents change (hash per page).
