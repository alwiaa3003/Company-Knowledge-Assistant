# KnowledgeLens evaluation results

Run on 2026-11-24 with `gemini-embedding-2` (768 dims) and `gemini-3.5-flash-lite`, on 3 PDFs
(44 pages): the ACME handbook (10 pages), the RAG paper (19) and the Attention paper (15).
Eval set: 15 questions ([eval_set.json](eval_set.json)), 6 of them paraphrased.
A **hit** = a top-k chunk from the expected file *and* an expected page. MRR is computed over the top 10.
Raw output: [results_table.md](results_table.md). Reproduce: `python eval/run_eval.py`.

## 1. Retrieval experiments (one variable changed at a time)

| Experiment | Changed | chunks | hit@1 | hit@3 | hit@5 | **hit@k** | MRR@10 | avg context sent to LLM |
|---|---|---|---|---|---|---|---|---|
| **baseline** (1000 / 200 / k=4) | – | 191 | **1.00** | 1.00 | 1.00 | **1.00** | **1.000** | 3,346 chars (~837 tok) |
| chunk 500 | chunk_size 1000 → 500 | 527 | 0.87 | 1.00 | 1.00 | 1.00 | 0.933 | 1,721 chars (~430 tok) |
| chunk 2000 | chunk_size 1000 → 2000 | 92 | 0.67 | 1.00 | 1.00 | 1.00 | 0.822 | 5,939 chars (~1,485 tok) |
| overlap 0 | overlap 200 → 0 | 165 | 0.87 | 1.00 | 1.00 | 1.00 | 0.933 | 3,156 chars (~789 tok) |
| k 2 | k 4 → 2 | 191 | 1.00 | 1.00 | 1.00 | 1.00 (@2) | 1.000 | 1,638 chars (~410 tok) |
| k 8 | k 4 → 8 | 191 | 1.00 | 1.00 | 1.00 | 1.00 (@8) | 1.000 | 6,583 chars (~1,646 tok) |

Context size = average characters of the k retrieved chunks over the 15 questions (~4 chars/token).
That's the part of the prompt that changes between configs, at a paid-tier estimate of $0.30 / 1M input tokens.

### Questions that lost rank 1 (all still found within the top 3)

| Config | Question | Rank | What beat it |
|---|---|---|---|
| chunk 500 | vendor sent wine worth ~$200 (exp. p.3) | 2 | p.4, the *continuation* of the gift table ("Over $250 → decline") |
| chunk 500 | which retriever / encoders in RAG (exp. p.3) | 2 | p.2 intro sentence "accessed with a pre-trained neural retriever" |
| chunk 2000 | vendor wine (exp. p.3) | 2 | p.4 gift-table continuation |
| chunk 2000 | lost laptop on the train (exp. p.7) | 2 | p.6 chunk ending with the *start* of Section 10 "Information Security" |
| chunk 2000 | paste customer data into unapproved chatbot (exp. p.9) | 2 | p.6 privacy section ("customer… personal data") |
| chunk 2000 | anonymous reporting phone number (exp. p.9) | 2 | p.2 "Ethics Helpline … supports anonymous reporting" (no number there) |
| chunk 2000 | attention heads and head size (exp. p.5) | 3 | p.3 general architecture overview |
| overlap 0 | vendor wine (exp. p.3) | 2 | p.4 gift-table continuation (tie: both 0.726) |
| overlap 0 | anonymous reporting phone number (exp. p.9) | 2 | p.2 Ethics Helpline mention |

## 2. Router + table route ([table_results.md](table_results.md))

| Test | Result |
|---|---|
| 7 table questions: router chose `table` | **7/7** (LLM router) |
| 7 table questions: computed answer matches pandas ground truth | **7/7** |
| 15 doc questions from the eval set: router chose `docs` | **15/15** (LLM router) |
| same 15 doc questions with the keyword **fallback** router only | **12/15**: sent "how many business days…", "maximum per-person amount…", "how many paid days off…" to `table` |

The table test originally had 6 questions. Running it exposed a bug: for "Which team has the most
employees?" the model asked to sort by `employee_id`, a column that no longer exists after grouping.
The sort was silently skipped, and `limit 1` returned the first team alphabetically, "Data". That
also happens to be the biggest team, so the question **passed by luck**. The fix: grouped results
always sort by the aggregate. A 7th question, "fewest employees", was added (correct: HR, 6), and
the old code would have answered it wrongly with "Data".

## 3. End-to-end checks (`python cli.py ask …`, baseline config)

| Question | Route | Answer | Sources | Latency | Tokens in/out | Est. cost |
|---|---|---|---|---|---|---|
| How many paid days off do I get if a close family member dies? | docs | "up to five (5) paid days … bereavement leave" | handbook p.8 | 7.9 s | 1252 / 36 | $0.00047 |
| How many people in the Data team are based in Bengaluru? | table | "4" (pandas truth: 4) | employees.csv, 4 rows (1, 2, 3, 6) | 13.4 s | 640 / 141 | $0.00054 |
| What is the CEO's favourite colour? | docs | "I don't know based on the provided documents." | none | 7.4 s | 971 / 12 | $0.00032 |

Latency is mostly **our own rate limiter**: `GEN_RPM=10` spaces calls ≥ 6 s apart, and a docs answer
makes 2 calls (router + answer) while a table answer makes 3. The smoke test's single call took 0.9 s.

## 4. Analysis in plain language

**The headline: on this corpus, retrieval is too easy to separate the configs at k ≥ 3.**
The baseline found the right page *first* for all 15 questions, and every config found it within
the top 3. With a perfect baseline, an experiment can only show regressions, never improvements.
This is a *ceiling effect*. One question is worth 6.7 percentage points, and several rank flips were
decided by similarity gaps of 0.003–0.03, so the differences below are hints, not proof.

What the hints say:
- **Smaller chunks (500)** cost two rank-1s. Both losses happened where one fact was split across a
  boundary: the gift table's *header* is on p.3 and its "$76–$250" row is far from the
  "Over $250" row on p.4, and the RAG retriever is introduced on p.2 but defined on p.3. Small
  chunks contain the keywords without the context. Upside: half the prompt size (~430 vs ~837 tokens).
- **Bigger chunks (2000)** were the worst at rank 1 (0.67, MRR 0.822). A 2000-character chunk covers
  several handbook sections at once, so its embedding is an "average" of topics. A chunk about
  privacy *and* the start of information security beat the actual lost-device rule. It also sends
  ~1.8× more context to the LLM for the same k.
- **No overlap** cost two rank-1s. Without overlap, text at a chunk boundary is in only one chunk,
  and that chunk may be mostly about something else.
- **k** doesn't change the ranking, only how far down we look. Since everything is found by rank 3,
  **k=2 got the same hit rate as k=4 with ~half the context tokens**, and k=8 doubled the context
  for no retrieval gain. k=2 is fragile, though: chunk-500 and overlap-0 both had questions at rank 2.

**Best configuration:** the baseline (chunk 1000, overlap 200) has the best ranking (MRR 1.000). For
k, the cheapest setting with no measured loss is k=2, but I'd keep **k=4** as a safety margin, given
how close some similarity scores are, and because generation sometimes needs two pages (the gift
rules span p.3–4).

**The recurring failure is structural, not a parameter:** the ACME gift table crosses a page break,
and the "never cross a page" chunking rule splits its header from its rows. In 3 of 4 configs the
continuation page (p.4) outranked the right page. Fixes that would actually help: table-aware
chunking, merging a table that continues onto the next page, or attaching the section heading
("4.2 Approval Thresholds") to every chunk.

**What to do next to get a meaningful signal:** make the eval harder. Add more documents (more
distractors), more questions per document, questions whose answer needs two pages, and
look-alike questions (e.g. gift rules for *giving* vs *receiving*). Then also evaluate the
*answers* (correctness + faithfulness), not just retrieval.

## 5. Notes on the run
- Free-tier embedding quota: 429 `embed_content_free_tier_requests`. Each text inside a batch seems
  to count as one request, so 50-text batches hit the per-minute limit right away. The retry logic
  waited out 8 such 429s during the experiments (plus 3 during the first ingest), using the
  server's suggested delay of up to ~60 s. No daily quota was exhausted.
- The embedding cache means a config only embeds chunks it hasn't seen. E.g. overlap 0 needed 121
  new embeddings out of 165, because short pages produce identical chunks in both configs.
  Re-running the whole eval now costs 0 embedding calls.
