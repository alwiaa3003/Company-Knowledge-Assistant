"""Generate a fictional, internally consistent employees table: data/tables/employees.csv.

Run:  .venv\\Scripts\\python.exe scripts/make_employees.py
Seeded (seed=42), so the file is identical every run. Consistency rules:
- each team has exactly one Head (manager_id empty); every other employee reports to someone in
  the SAME team who started earlier than they did;
- start dates are between 2015 and 2025, and nobody starts before their manager;
- salary depends on role seniority, with noise.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "tables" / "employees.csv"
SEED = 42
TEAM_SIZES = {"Data": 14, "Platform": 12, "Sales": 11, "Product": 9, "Finance": 8, "HR": 6}  # = 60
LOCATIONS = ["Bengaluru", "Hyderabad", "Pune", "Mumbai", "Delhi"]
ROLES: dict[str, list[str]] = {
    "Data": ["Data Analyst", "Data Engineer", "ML Engineer", "Senior Data Engineer"],
    "Platform": ["Software Engineer", "Site Reliability Engineer", "Senior Software Engineer"],
    "Sales": ["Account Executive", "Sales Development Rep", "Senior Account Executive"],
    "Product": ["Product Manager", "Product Designer", "Senior Product Manager"],
    "Finance": ["Financial Analyst", "Accountant", "Senior Financial Analyst"],
    "HR": ["HR Generalist", "Recruiter", "HR Business Partner"],
}
# Salary bands in INR per year: (min, max)
BANDS = {"Head": (4_500_000, 6_500_000), "Lead": (3_000_000, 4_200_000),
         "Senior": (2_200_000, 3_200_000), "Mid": (1_200_000, 2_200_000)}
# Obviously fictional names built from syllables.
FIRST = ["Zarvi", "Quillo", "Mirabex", "Tovan", "Elbrin", "Yuneth", "Korra", "Faxel", "Nimbo",
         "Orsa", "Plivo", "Delmi", "Ruvan", "Sazzi", "Wendro", "Ilka", "Brumo", "Cyndel"]
LAST = ["Testwala", "Placeholdar", "Fictiona", "Sampleton", "Mockerjee", "Dummyraj",
        "Examplekar", "Notrealan", "Madeupta", "Pretendra"]


def random_date(rng: random.Random, start: date, end: date) -> date:
    """Uniform random date in [start, end]."""
    return start + timedelta(days=rng.randint(0, (end - start).days))


def make_rows(rng: random.Random) -> list[dict[str, object]]:
    """Build all employee rows, team by team, so managers always exist before reports."""
    rows: list[dict[str, object]] = []
    used_names: set[str] = set()
    next_id = 1001

    def new_name() -> str:
        while True:
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if name not in used_names:
                used_names.add(name)
                return name

    for team, size in TEAM_SIZES.items():
        head_id = next_id
        head_start = random_date(rng, date(2015, 1, 1), date(2017, 12, 31))
        rows.append(dict(employee_id=head_id, name=new_name(), team=team, role=f"Head of {team}",
                         location=rng.choice(LOCATIONS[:2]), salary_inr=rng.randrange(*BANDS["Head"], 10_000),
                         start_date=head_start.isoformat(), manager_id=""))
        next_id += 1
        managers = [(head_id, head_start)]
        n_leads = 1 if size < 10 else 2
        for i in range(size - 1):
            is_lead = i < n_leads
            mgr_id, mgr_start = (managers[0] if is_lead else rng.choice(managers))
            start = random_date(rng, mgr_start + timedelta(days=30), date(2025, 6, 30))
            role = f"{team} Lead" if is_lead else rng.choice(ROLES[team])
            band = "Lead" if is_lead else ("Senior" if role.startswith("Senior") else "Mid")
            rows.append(dict(employee_id=next_id, name=new_name(), team=team, role=role,
                             location=rng.choice(LOCATIONS), salary_inr=rng.randrange(*BANDS[band], 10_000),
                             start_date=start.isoformat(), manager_id=mgr_id))
            if is_lead:
                managers.append((next_id, start))
            next_id += 1
    return rows


def main() -> None:
    """Write the CSV and print a short summary."""
    rows = make_rows(random.Random(SEED))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
