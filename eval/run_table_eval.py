"""Table-route evaluation: does the router pick 'table', and is the computed answer correct?

Run:  .venv\\Scripts\\python.exe eval/run_table_eval.py
Ground truth is computed directly with pandas. We check the STRUCTURED result (numbers / ids),
not the wording of the natural-language answer, so the check is objective.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from kb.config import load_settings  # noqa: E402
from kb.router import route  # noqa: E402
from kb.structured import load_table, query_table  # noqa: E402


def values_in(result: list[dict[str, Any]]) -> list[Any]:
    """Every cell value in the result rows (to look for the expected number / names)."""
    return [v for row in result for v in row.values()]


def close(a: Any, b: float, tol: float = 0.5) -> bool:
    return isinstance(a, (int, float)) and abs(float(a) - b) <= tol


def build_cases(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Six questions, each with a pandas ground truth and a checker over query_table's result."""
    n_data_blr = int(((df.team == "Data") & (df.location == "Bengaluru")).sum())
    mean_platform = float(df[df.team == "Platform"].salary_inr.mean())
    biggest = df.team.value_counts()
    top3 = df.sort_values("salary_inr", ascending=False).head(3).name.tolist()
    after_2023 = int((df.start_date >= "2023-01-01").sum())
    head_sales = int(df[(df.team == "Sales") & (df.manager_id.isna())].employee_id.iloc[0])
    reports = sorted(df[df.manager_id == head_sales].name.tolist())

    def has_number(target: float) -> Callable[[dict], bool]:
        return lambda out: any(close(v, target) for v in values_in(out["result"]))

    def has_names(names: list[str], exact: bool = False) -> Callable[[dict], bool]:
        def check(out: dict) -> bool:
            got = {v for v in values_in(out["result"]) if isinstance(v, str)}
            return set(names) <= got and (not exact or len(out["result"]) == len(names))
        return check

    def biggest_team(out: dict) -> bool:
        rows = out["result"]
        return bool(rows) and biggest.index[0] in [str(v) for v in rows[0].values()]

    def smallest_team(out: dict) -> bool:
        rows = out["result"]
        return bool(rows) and biggest.index[-1] in [str(v) for v in rows[0].values()]

    return [
        {"q": "How many people in the Data team are based in Bengaluru?", "truth": n_data_blr,
         "check": has_number(n_data_blr)},
        {"q": "What is the average salary in the Platform team?", "truth": round(mean_platform),
         "check": has_number(mean_platform)},
        {"q": "Which team has the most employees?", "truth": f"{biggest.index[0]} ({biggest.iloc[0]})",
         "check": biggest_team},
        # Added after a real bug: an ignored sort made "top 1" = first team alphabetically ("Data"),
        # which is also the biggest team, so the question above passed by luck. HR is the smallest.
        {"q": "Which team has the fewest employees?", "truth": f"{biggest.index[-1]} ({biggest.iloc[-1]})",
         "check": smallest_team},
        {"q": "Who are the three highest-paid employees?", "truth": top3, "check": has_names(top3)},
        {"q": "How many staff joined on or after 1 January 2023?", "truth": after_2023,
         "check": has_number(after_2023)},
        {"q": f"List everyone who reports to employee {head_sales}.", "truth": reports,
         "check": has_names(reports, exact=True)},
    ]


def main() -> int:
    settings = load_settings()
    df = load_table(settings.table_file)
    cases = build_cases(df)
    routed_ok = correct = 0
    lines = ["| # | Question | Router | Correct | Ground truth (pandas) | System answer |", "|---|---|---|---|---|---|"]
    for i, case in enumerate(cases, start=1):
        r, method, _ = route(settings, case["q"], settings.table_file.name, list(df.columns))
        out = query_table(case["q"])
        ok_route = r == "table"
        ok = bool(out.get("result")) and case["check"](out)
        routed_ok += ok_route
        correct += ok
        print(f"[{i}] route={r} ({method}) correct={ok}\n    Q: {case['q']}\n    truth: {case['truth']}\n"
              f"    spec: {out.get('spec')}\n    answer: {out['answer']}\n")
        answer = out["answer"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {i} | {case['q']} | {r} ({method}) | {'yes' if ok else 'NO'} | {case['truth']} | {answer} |")
    n = len(cases)
    summary = f"\n**Router accuracy: {routed_ok}/{n} = {routed_ok / n:.0%} — Answer correctness: {correct}/{n} = {correct / n:.0%}**\n"
    md = "\n".join(lines) + "\n" + summary
    (ROOT / "eval" / "table_results.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
