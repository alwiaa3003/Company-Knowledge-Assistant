r"""Agent definitions: the multi-agent hierarchy (with reflection) and a single-agent baseline.

ADK 2.x docs (https://adk.dev):
- LlmAgent / Agent: https://adk.dev/agents/llm-agents/
- sub-agents with mode="single_turn" are exposed to the parent as delegation tools and return their
  result to the parent: https://adk.dev/workflows/collaboration/
- graph Workflow (edges, routes, function nodes): https://adk.dev/graphs/ , /graphs/routes/
  (SequentialAgent / LoopAgent are deprecated in favour of graph workflows in ADK 2.x, and the
  installed 2.15.0 warns about it, so the reflection step is a small graph.)
- retries for Gemini calls: Gemini(retry_options=types.HttpRetryOptions(...)) as documented for 429s.

Hierarchy:
    START -> coordinator --(delegation tools)--> docs_agent  (search_documents)
                         \-------------------> table_agent (query_table)
          -> critic -> route_review --APPROVE--> finalize
                                    \-REVISE---> reviser -> finalize      (max ONE revision)
"""

from __future__ import annotations

import json
from typing import Any, Literal

from google.adk import Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.google_llm import Gemini
from google.adk.workflow import FunctionNode
from google.genai import types
from pydantic import BaseModel, Field

from kb.config import Settings, load_settings
from kl_agents import callbacks
from kl_agents.tools import EVIDENCE_KEY, query_table, search_documents

IDK = "I don't know based on the provided documents."
DRAFT_KEY = "draft"
FINAL_KEY = "final_answer"
REVIEW_KEY = "review"


def make_model(settings: Settings) -> Gemini:
    """Gemini model from GEN_MODEL, with the API key from .env and retry/backoff on 429/5xx.

    The key is passed to google.genai.Client through `client_kwargs`, so no GOOGLE_API_KEY
    environment variable is needed and .env stays unchanged.
    """
    return Gemini(
        model=settings.gen_model,
        retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0, exp_base=2.0, max_delay=60.0,
                                             http_status_codes=[429, 500, 502, 503, 504]),
        client_kwargs={"api_key": settings.gemini_api_key},
    )


def _callbacks() -> dict[str, Any]:
    """The same metric/rate-limit callbacks on every LLM agent."""
    return dict(before_agent_callback=callbacks.before_agent,
                before_model_callback=callbacks.before_model,
                after_model_callback=callbacks.after_model,
                before_tool_callback=callbacks.before_tool,
                after_tool_callback=callbacks.after_tool)


_ZERO_TEMP = types.GenerateContentConfig(temperature=0.0)


DOCS_INSTRUCTION = f"""You are docs_agent, a careful research specialist for company documents
(ACME employee policy handbook, the RAG paper by Lewis et al. 2020, the "Attention Is All You Need" paper).

Work in a ReAct loop:
  Thought: decide what you still need to find.
  Action: call search_documents with a focused query.
  Observation: read the returned passages.
Repeat only if the passages do not contain the answer. Use AT MOST 3 search_documents calls.

Then answer:
- Use ONLY facts stated in the returned passages. Never use outside knowledge.
- After each fact cite its passage as [file p.N], e.g. [acme_employee_policy_handbook.pdf p.8].
- If the passages do not contain the answer, reply exactly: "{IDK}"
- Be concise (1-4 sentences). Do not show your Thought/Action lines in the final answer."""

TABLE_INSTRUCTION = """You are table_agent, a data specialist for the employees table (employees.csv).
Call query_table ONCE with the user's table question (you may rephrase it to be precise).
Answer in 1-3 sentences using ONLY the numbers/names in the tool's "result", exactly as given, and end
with the source in parentheses exactly as the tool's "source" field, e.g. (employees.csv (4 rows matched)).
If the tool returns status "error", say you could not answer from the table and why."""

COORDINATOR_INSTRUCTION = f"""You are the coordinator of KnowledgeLens, a company knowledge assistant.
You never answer from your own knowledge. You delegate to specialists:
- docs_agent: company policies (ACME handbook), and the RAG and Transformer research papers.
- table_agent: the employees table (counts, filters, averages, rankings over employees, teams,
  salaries, locations, managers, start dates).

Steps:
1. Split the question into parts. Send each part to the right specialist with a clear, self-contained
   request. A question can need one or both specialists; call both when it has both kinds of parts.
2. Combine the specialists' answers into one final answer. Keep their citations ([file p.N]) and table
   sources (employees.csv (N rows matched)) exactly as given. Do not add facts they did not give.
3. Call each specialist AT MOST ONCE per part. Never repeat a request a specialist already answered.
   If a specialist says "{IDK}" for its part, accept it and keep that statement for that part.
   If no specialist fits the question, reply exactly: "{IDK}" """


class Review(BaseModel):
    """Structured critic verdict."""

    verdict: Literal["APPROVE", "REVISE"] = Field(description="APPROVE if the draft is fully supported")
    problems: list[str] = Field(default_factory=list, description="Each unsupported claim or missing citation")


def _evidence_text(state: Any, invocation_id: str) -> str:
    """Render the evidence the tools gathered during THIS request for the critic/reviser."""
    ev = state.get(EVIDENCE_KEY) or {}
    if not isinstance(ev, dict) or ev.get("invocation_id") != invocation_id:
        return "(no tool evidence was gathered for this question)"
    lines = []
    for item in ev.get("items", []):
        if item["tool"] == "search_documents":
            for c in item["chunks"]:
                lines.append(f"[{c['file']} p.{c['page']}] {c['text']}")
        else:
            lines.append(f"[{item['source']}] table result for '{item['question']}': "
                         f"{json.dumps(item['result'][:25], ensure_ascii=False)}")
    return "\n\n".join(lines) or "(no tool evidence was gathered for this question)"


def critic_instruction(ctx: ReadonlyContext) -> str:
    """Instruction provider: the critic sees the question, the draft and the evidence, nothing else."""
    return f"""You are a strict fact-checking critic. Check the DRAFT ANSWER against the EVIDENCE.
APPROVE only if: every factual claim is supported by the evidence, every document fact has a
[file p.N] citation that matches an evidence passage, and table numbers match the table result.
An answer that says "{IDK}" for a part with no supporting evidence is correct; APPROVE it.
Otherwise return REVISE and list each problem briefly. Do not rewrite the answer.

QUESTION: {_user_question(ctx)}

DRAFT ANSWER:
{ctx.state.get(DRAFT_KEY, '')}

EVIDENCE:
{_evidence_text(ctx.state, ctx.invocation_id)}"""


def reviser_instruction(ctx: ReadonlyContext) -> str:
    """Instruction provider for the single allowed revision."""
    review = ctx.state.get(REVIEW_KEY) or {}
    problems = "\n".join(f"- {p}" for p in review.get("problems", [])) or "- (none listed)"
    return f"""Rewrite the DRAFT ANSWER so that it fixes the PROBLEMS, using ONLY the EVIDENCE.
Remove any claim the evidence does not support. Cite document facts as [file p.N] and keep table
sources like (employees.csv (N rows matched)). If nothing in the evidence answers a part, say
"{IDK}" for that part. Output only the final answer text.

QUESTION: {_user_question(ctx)}

DRAFT ANSWER:
{ctx.state.get(DRAFT_KEY, '')}

PROBLEMS:
{problems}

EVIDENCE:
{_evidence_text(ctx.state, ctx.invocation_id)}"""


def _user_question(ctx: ReadonlyContext) -> str:
    content = ctx.user_content
    if content and content.parts:
        return " ".join(p.text for p in content.parts if p.text)
    return ""


def route_review(ctx, node_input: Any) -> Event:
    """Router node: APPROVE -> finalize, REVISE -> reviser (so at most ONE revision)."""
    review = node_input if isinstance(node_input, dict) else {}
    if hasattr(node_input, "model_dump"):
        review = node_input.model_dump()
    verdict = str(review.get("verdict", "APPROVE")).upper()
    return Event(route="REVISE" if verdict == "REVISE" else "APPROVE", output=ctx.state.get(DRAFT_KEY, ""))


def finalize(ctx, node_input: Any) -> Event:
    """Terminal node: store and show the final answer (the draft, or the revision)."""
    text = node_input if isinstance(node_input, str) else ctx.state.get(DRAFT_KEY, "")
    return Event(state={FINAL_KEY: text},
                 content=types.Content(role="model", parts=[types.Part(text=text)]))


def build_specialists(settings: Settings) -> tuple[LlmAgent, LlmAgent]:
    """docs_agent and table_agent as single_turn sub-agents (called like tools, result returned)."""
    model = make_model(settings)
    docs_agent = LlmAgent(
        name="docs_agent", model=model, mode="single_turn",
        disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
        description="Answers questions from the company documents (ACME policy handbook, RAG paper, "
                    "Transformer paper) with [file p.N] citations.",
        instruction=DOCS_INSTRUCTION, tools=[search_documents], generate_content_config=_ZERO_TEMP,
        **_callbacks())
    table_agent = LlmAgent(
        name="table_agent", model=model, mode="single_turn",
        disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
        description="Answers questions about the employees table (counts, filters, averages, rankings).",
        instruction=TABLE_INSTRUCTION, tools=[query_table], generate_content_config=_ZERO_TEMP,
        **_callbacks())
    return docs_agent, table_agent


def build_coordinator(settings: Settings, output_key: str) -> LlmAgent:
    """The coordinator LlmAgent with docs_agent + table_agent as single_turn sub-agents."""
    docs_agent, table_agent = build_specialists(settings)
    return LlmAgent(
        name="coordinator", model=make_model(settings), instruction=COORDINATOR_INSTRUCTION,
        sub_agents=[docs_agent, table_agent], output_key=output_key, generate_content_config=_ZERO_TEMP,
        **{**_callbacks(), "before_tool_callback": [callbacks.block_repeated_delegation, callbacks.before_tool]})


def build_hierarchy_no_reflection(settings: Settings | None = None) -> LlmAgent:
    """Ablation: the same coordinator + specialists, WITHOUT the critic/reviser step."""
    return build_coordinator(settings or load_settings(), output_key=FINAL_KEY)


def build_hierarchy(settings: Settings | None = None) -> Workflow:
    """Coordinator + specialists + critic/reviser reflection, as an ADK graph Workflow."""
    settings = settings or load_settings()
    model = make_model(settings)
    coordinator = build_coordinator(settings, output_key=DRAFT_KEY)
    critic = LlmAgent(
        name="critic", model=model, instruction=critic_instruction, include_contents="none",
        output_schema=Review, output_key=REVIEW_KEY, generate_content_config=_ZERO_TEMP, **_callbacks())
    reviser = LlmAgent(
        name="reviser", model=model, instruction=reviser_instruction, include_contents="none",
        generate_content_config=_ZERO_TEMP, **_callbacks())
    router = FunctionNode(func=route_review, name="route_review")
    final = FunctionNode(func=finalize, name="finalize")
    return Workflow(
        name="knowledge_lens",
        description="KnowledgeLens: coordinator -> docs/table specialists -> critic -> optional revision",
        edges=[
            ("START", coordinator, critic, router),
            (router, {"APPROVE": final, "REVISE": reviser}),
            (reviser, final),
        ],
    )


SINGLE_INSTRUCTION = f"""You are KnowledgeLens, a company knowledge assistant with two tools:
- search_documents: passages from the ACME employee policy handbook, the RAG paper and the Transformer
  paper. Use it for policies, rules and research content. Use at most 3 calls.
- query_table: exact answers about the employees table (counts, filters, averages, rankings).
A question may need one or both tools. Never answer from your own knowledge.
Write the final answer using ONLY tool results: cite document facts as [file p.N] and table facts
with the tool's source, e.g. (employees.csv (4 rows matched)). Report numbers exactly.
If the tools do not contain the answer to a part, say exactly "{IDK}" for that part. Be concise."""


def build_single_agent(settings: Settings | None = None) -> LlmAgent:
    """Baseline: ONE LlmAgent with both tools, no sub-agents, no critic."""
    settings = settings or load_settings()
    return LlmAgent(
        name="single_agent", model=make_model(settings), instruction=SINGLE_INSTRUCTION,
        tools=[search_documents, query_table], output_key=FINAL_KEY, generate_content_config=_ZERO_TEMP,
        **_callbacks())
