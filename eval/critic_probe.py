"""Does the reflection step actually work? Feed the critic drafts with PLANTED errors.

  python eval/critic_probe.py

The end-to-end eval never produced a bad draft, so the critic never had to act. This probe runs the
REAL critic -> route_review -> reviser -> finalize graph on hand-made drafts with real evidence (the
real retrieved passages for the question) and checks the verdict. ~2 LLM calls per case.
Cases: a correct cited draft (expect APPROVE), a wrong number (expect REVISE + fixed), a claim that is not in
the evidence (expect REVISE + claim removed).
"""

from __future__ import annotations

import asyncio
import sys
import uuid
import warnings
from typing import Any

from google.adk import Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import FunctionNode
from google.genai import types

from kb.config import load_settings
from kb.retrieve import retrieve
from kl_agents import builders, telemetry
from kl_agents.tools import EVIDENCE_KEY

QUESTION = "How many paid days of bereavement leave do employees get?"
CASES = [
    {"name": "correct, cited", "expect": "APPROVE",
     "draft": "Employees get up to five (5) paid days of bereavement leave for immediate family "
              "[acme_employee_policy_handbook.pdf p.8].", "must_contain": "five"},
    {"name": "wrong number", "expect": "REVISE",
     "draft": "Employees get up to ten (10) paid days of bereavement leave for immediate family "
              "[acme_employee_policy_handbook.pdf p.8].", "must_contain": "five", "must_not_contain": "ten (10)"},
    {"name": "unsupported extra claim", "expect": "REVISE",
     "draft": "Employees get up to five (5) paid days of bereavement leave for immediate family "
              "[acme_employee_policy_handbook.pdf p.8]. Unused bereavement days can be carried over to the next year.",
     "must_contain": "five", "must_not_contain": "carried over"},
]


def build_probe(draft: str, chunks: list[dict[str, Any]]) -> Workflow:
    """seed (puts draft + evidence in state) -> the real critic/router/reviser/finalize nodes."""
    settings = load_settings()
    model = builders.make_model(settings)

    def seed(ctx, node_input: Any) -> Event:
        return Event(state={builders.DRAFT_KEY: draft,
                            EVIDENCE_KEY: {"invocation_id": ctx.invocation_id,
                                           "items": [{"tool": "search_documents", "query": QUESTION, "chunks": chunks}]}},
                     output=draft)

    critic = LlmAgent(name="critic", model=model, instruction=builders.critic_instruction, include_contents="none",
                      output_schema=builders.Review, output_key=builders.REVIEW_KEY,
                      generate_content_config=builders._ZERO_TEMP, **builders._callbacks())
    reviser = LlmAgent(name="reviser", model=model, instruction=builders.reviser_instruction, include_contents="none",
                       generate_content_config=builders._ZERO_TEMP, **builders._callbacks())
    router = FunctionNode(func=builders.route_review, name="route_review")
    final = FunctionNode(func=builders.finalize, name="finalize")
    return Workflow(name="critic_probe", edges=[("START", FunctionNode(func=seed, name="seed"), critic, router),
                                                (router, {"APPROVE": final, "REVISE": reviser}), (reviser, final)])


async def run_case(case: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    wf = build_probe(case["draft"], chunks)
    ss = InMemorySessionService()
    sid = uuid.uuid4().hex
    await ss.create_session(app_name="probe", user_id="u", session_id=sid)
    runner = Runner(agent=wf, app_name="probe", session_service=ss)
    rec, token = telemetry.start_request(QUESTION, "critic_probe")
    try:
        async for _ in runner.run_async(user_id="u", session_id=sid,
                                        new_message=types.Content(role="user", parts=[types.Part(text=QUESTION)])):
            pass
    finally:
        telemetry.end_request(token)
    state = (await ss.get_session(app_name="probe", user_id="u", session_id=sid)).state
    review = state.get(builders.REVIEW_KEY) or {}
    final = str(state.get(builders.FINAL_KEY, ""))
    verdict = str(review.get("verdict", "?"))
    fixed = case["must_contain"] in final.lower() and (
        "must_not_contain" not in case or case["must_not_contain"] not in final.lower())
    return {"case": case["name"], "expected": case["expect"], "verdict": verdict,
            "verdict_ok": verdict == case["expect"], "final_ok": fixed, "problems": review.get("problems", []),
            "final": final, "llm_calls": len(rec.model_calls)}


async def main() -> int:
    warnings.filterwarnings("ignore")
    hits = retrieve(load_settings(), QUESTION, 4)
    chunks = [{"file": h["file"], "page": h["page"], "score": h["score"], "text": h["text"][:1500]} for h in hits]
    print("Evidence pages:", [(c["file"][:12], c["page"]) for c in chunks])
    results = []
    for case in CASES:
        r = await run_case(case, chunks)
        results.append(r)
        print(f"\n[{r['case']}] expected {r['expected']} -> got {r['verdict']} "
              f"({'OK' if r['verdict_ok'] else 'WRONG'}); final answer correct: {r['final_ok']}; calls={r['llm_calls']}")
        print("  problems:", r["problems"])
        print("  final:", r["final"])
    ok_v = sum(r["verdict_ok"] for r in results)
    ok_f = sum(r["final_ok"] for r in results)
    print(f"\nCritic verdicts correct: {ok_v}/{len(results)} | final answers correct: {ok_f}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
