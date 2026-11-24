| Experiment | Changed | chunk_size | overlap | k | hit@1 | hit@3 | hit@5 | **hit@k** | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | - | 1000 | 200 | 4 | 1.00 | 1.00 | 1.00 | **1.00** | 1.000 |
| chunk 500 | chunk_size 1000 -> 500 | 500 | 200 | 4 | 0.87 | 1.00 | 1.00 | **1.00** | 0.933 |
| chunk 2000 | chunk_size 1000 -> 2000 | 2000 | 200 | 4 | 0.67 | 1.00 | 1.00 | **1.00** | 0.822 |
| overlap 0 | overlap 200 -> 0 | 1000 | 0 | 4 | 0.87 | 1.00 | 1.00 | **1.00** | 0.933 |
| k 2 | k 4 -> 2 | 1000 | 200 | 2 | 1.00 | 1.00 | 1.00 | **1.00** | 1.000 |
| k 8 | k 4 -> 8 | 1000 | 200 | 8 | 1.00 | 1.00 | 1.00 | **1.00** | 1.000 |

**Misses at k=4 — baseline** (0/15)
- none

**Misses at k=4 — chunk 500** (0/15)
- none

**Misses at k=4 — chunk 2000** (0/15)
- none

**Misses at k=4 — overlap 0** (0/15)
- none

**Misses at k=2 — k 2** (0/15)
- none

**Misses at k=8 — k 8** (0/15)
- none
