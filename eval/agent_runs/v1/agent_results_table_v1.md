| System | Routing | Correct (all) | Docs | Table | Mixed | Refusal | Avg LLM calls | Avg latency | p95 latency | Avg tokens in/out | Avg cost/req | Avg tok/s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) single agent, both tools | 10/10 | **12/12** | 4/4 | 3/3 | 3/3 | 2/2 | 2.8 | 14.2s | 22.3s | 3306/171 | $0.00142 | 22.5 | 0 |
| (b) coordinator + specialists | 10/10 | **12/12** | 4/4 | 3/3 | 3/3 | 2/2 | 5.0 | 33.5s | 56.0s | 3920/227 | $0.00174 | 14.5 | 0 |
| (c) coordinator + specialists + critic | 10/10 | **12/12** | 4/4 | 3/3 | 3/3 | 2/2 | 6.8 | 45.2s | 69.7s | 6789/258 | $0.00268 | 12.5 | 0 |

Per question (✔ correct, ✘ wrong; routing = specialists used; calls = LLM calls):

| ID | Question | (a) single agent, both tools | (b) coordinator + specialists | (c) coordinator + specialists + critic |
|---|---|---|---|---|
| D1 | Within how many business days must an employee disclose a conflict of interest? | ✔ docs, 2 calls, 7s | ✔ docs, 4 calls, 21s | ✔ docs, 5 calls, 30s |
| D2 | I lost my work laptop. How quickly must I report it, and to whom? | ✔ docs, 2 calls, 7s | ✔ docs, 4 calls, 30s | ✔ docs, 5 calls, 25s |
| D3 | In the RAG paper, what is the difference between RAG-Sequence and RAG-Token? | ✔ docs, 2 calls, 15s | ✔ docs, 4 calls, 20s | ✔ docs, 5 calls, 25s |
| D4 | How many attention heads does the base Transformer use, and what is the dimension of each head? | ✔ docs, 2 calls, 8s | ✔ docs, 4 calls, 20s | ✔ docs, 5 calls, 32s |
| T1 | How many employees work in Hyderabad? | ✔ table, 3 calls, 13s | ✔ table, 5 calls, 33s | ✔ table, 6 calls, 32s |
| T2 | What is the average salary in the Sales team? | ✔ table, 3 calls, 13s | ✔ table, 5 calls, 42s | ✔ table, 6 calls, 35s |
| T3 | Which team has the fewest employees? | ✔ table, 3 calls, 13s | ✔ table, 5 calls, 65s | ✔ table, 6 calls, 53s |
| M1 | How many people are in the Data team, and what does the handbook say about remote work? | ✔ docs+table, 3 calls, 25s | ✔ docs+table, 7 calls, 38s | ✔ docs+table, 9 calls, 52s |
| M2 | Who are the three highest-paid employees, and what is the per-person limit for business meals? | ✔ docs+table, 3 calls, 14s | ✔ docs+table, 7 calls, 38s | ✔ docs+table, 8 calls, 70s |
| M3 | How many employees joined on or after 1 January 2023, and how many paid days of bereavement leave do employees get? | ✔ docs+table, 3 calls, 13s | ✔ docs+table, 7 calls, 37s | ✔ docs+table, 8 calls, 49s |
| U1 | What is the CEO's favourite colour? | ✔ docs+table, 4 calls, 19s | ✔ none, 1 calls, 1s | ✔ none, 2 calls, 10s |
| U2 | What is the parking reimbursement limit for employees in the Pune office? | ✔ docs, 4 calls, 22s | ✔ docs, 7 calls, 56s | ✔ docs+table, 17 calls, 130s |
