# Project 2: single agent vs. multi-agent hierarchy (real runs)

Runs on 2026-11-25 with `google-adk==2.15.0` and `gemini-3.5-flash-lite` (free AI Studio tier), on the
12 questions in [agent_eval.json](agent_eval.json): 4 doc-only, 3 table-only, 3 mixed (need both), 2 unanswerable.
Scoring is objective, with no LLM judge ([run_agent_eval.py](run_agent_eval.py)):
- docs = expected file + page cited **and** key facts present
- table = pandas ground truth present in the answer
- mixed = both parts correct
- refusal = "I don't know…" with no citation
- routing = the specialists actually used equal the expected set

One change at a time: (a) → (b) adds the sub-agent hierarchy, (b) → (c) adds the reflection step.

## Final results (v2, after the fixes described below)

| System | Routing | Correct (all) | Docs | Table | Mixed | Refusal | Avg LLM calls | Avg latency | p95 latency | Avg tokens in/out | Avg cost/req | Avg tok/s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) single agent, both tools | 10/10 | **12/12** | 4/4 | 3/3 | 3/3 | 2/2 | **2.8** | **14.2s** | **22.3s** | 3306/171 | **$0.00142** | 22.5 | 0 |
| (b) coordinator + specialists | 10/10 | **12/12** | 4/4 | 3/3 | 3/3 | 2/2 | 5.2 | 29.3s | 38.1s | 3652/230 | $0.00167 | 15.3 | 0 |
| (c) coordinator + specialists + critic | 10/10 | **12/12** | 4/4 | 3/3 | 3/3 | 2/2 | 6.1 | 34.6s | 48.9s | 4113/243 | $0.00184 | 12.7 | 0 |

Relative to (a): (b) = 1.9× LLM calls, 2.1× latency, 1.18× cost. (c) = 2.2× calls, 2.4× latency, 1.30× cost.
Cost is the paid-tier *estimate* from `kb.config` ($0.30 / $2.50 per 1M input/output tokens); the free tier itself cost $0.

**LLM calls per question type** (from the per-question table in [agent_results_table.md](agent_results_table.md)):

| Question type | (a) single | (b) hierarchy | (c) + critic |
|---|---|---|---|
| doc-only | 2 | 4 | 5 |
| table-only (incl. 1 call inside `query_table`) | 3 | 5 | 6 |
| mixed | 3 | 7 | 8 |
| unanswerable | 4 | 1 / 10 | 2 / 9 |

## v1 → v2: a bug the eval found

v1 (same questions, first version of the hierarchy) was also 12/12 correct but less efficient:

| System | Avg LLM calls | Avg latency | p95 latency | Avg tokens in/out | Avg cost/req |
|---|---|---|---|---|---|
| (b) v1 | 5.0 | 33.5s | 56.0s | 3920/227 | $0.00174 |
| (c) v1 | 6.8 | 45.2s | 69.7s | 6789/258 | $0.00268 |

On U2 ("parking reimbursement limit for the Pune office", a trap with no answer in any document), (c) v1 used
**17 LLM calls and 130 s**. The trace showed two problems:
- The coordinator re-sent the *identical* question to `docs_agent` (6 searches in total).
- `table_agent` called ADK's `transfer_to_agent`.

Fixes in v2:
- `disallow_transfer_to_parent/peers=True` on the specialists.
- A "call each specialist at most once" rule in the coordinator's instruction.
- A `before_tool_callback` guard that refuses identical repeat delegations. It never had to fire in v2.

U2 then took 9 calls in (c) and 10 in (b). The single agent was not affected by these changes and was not rerun.
v1 raw runs: [agent_runs/v1/](agent_runs/v1/).

The v1 report also first scored (a) at 11/12. That was a **scorer bug**: the answer cited
`[attention_is_all_you_need.pdf p. 5, p. 9]` (two pages in one bracket), which the citation parser didn't
read. The parser was fixed, and all systems were re-scored with the same code.

## Does the critic work? (planted-error probe, [critic_probe.py](critic_probe.py))

In all 24 real hierarchy runs (v1 + v2) the critic **approved every draft (0 revisions)**: no draft had an error to catch.
So the probe runs the real critic → reviser graph on hand-made drafts, with the real retrieved evidence:

| Planted case | Expected | Critic verdict | Final answer after revision |
|---|---|---|---|
| correct, cited | APPROVE | APPROVE ✔ | unchanged ✔ |
| wrong number ("ten days" instead of five) | REVISE | REVISE ✔ | fixed to "five (5) … [p.8]" ✔ |
| unsupported extra claim ("unused days carry over") | REVISE | REVISE ✔ | claim replaced by "I don't know … whether unused bereavement days can be carried over" (fails my strict string check, but no unsupported claim remains) |

Verdicts 3/3, strict final check 2/3. 5 LLM calls in total.

## Honest conclusion: did the extra agents help?

**On this eval set, no. They cost more for the same quality.**

- **Quality was already at the ceiling.** The single agent answered 12/12 with correct routing, including all
  3 mixed questions. It even issued both tool calls (`query_table` and `search_documents`) in **one** model turn,
  so a mixed question costs it 3 calls versus 7–8 for the hierarchy.
- **Why the hierarchy costs more.** Every delegation adds a "coordinator decides" call and a "coordinator
  merges" call on top of the specialist's own calls, all sequential. Token cost grows less than call count
  (1.18–1.30×), because each specialist sees a small, focused context.
- **The critic adds ~1 call, ~5 s and ~10% cost per request**, and changed nothing in 24 real runs. The probe shows
  it *does* catch planted errors, so it's insurance against failures this model didn't make here.
- **One place the hierarchy looked better:** U1 ("CEO's favourite colour") was refused by the coordinator in 1–2
  calls, versus 4 for the single agent, which searched both tools first. That is cheaper, but it means refusing
  without checking. Over-eager refusal is a risk of its own.
- **Where multi-agent would earn its cost:** many tools (one agent's tool list and prompt get too long),
  specialists that need different models, permissions or data access, longer multi-step tasks, or a weaker or
  cheaper drafting model whose mistakes a critic actually catches. The right next step is a harder eval set,
  where (a) fails sometimes, *before* paying for (b)/(c).

## Caveats
- 12 questions is small. One question = 8 percentage points, and all systems tied on quality.
- **Latency is dominated by our own pacing.** `GEN_RPM=10` spaces LLM calls ≥ 6 s apart (free-tier protection).
  Each model call itself took ~1–3 s. With higher quotas, latency would scale roughly with calls × per-call time.
- tok/s = output tokens ÷ time inside model calls (non-streamed). It falls with more agents because many of
  their calls are short "routing" replies with only a few output tokens.
- **Free tier during these runs:** no 429 quota errors. kb's retry wrapper logged **19 HTTP 503 ("overloaded")**
  responses (15 in v1, 4 in v2) on the LLM call inside `query_table`, all recovered by backoff. Agent model calls
  use ADK's `HttpRetryOptions`, which retries silently, so 503s there (if any) aren't counted. No request failed
  (0 errors in all 36 + 24 runs).
- **Later, in a separate end-to-end run (8 new questions, all 8 correct),** the free-tier *daily* request quota ran out:
  a 429 asked for a 37 645 s wait, and `kb.llm.call_with_retry` obeyed it, so one question took 18 h. Now fixed:
  a hinted wait over 120 s raises `QuotaExhaustedError` immediately. That run is excluded from the tables above.
