# Evaluation summary

- Run ID: `eval-20261117T070237Z-8a0d1154`
- Timestamp (UTC): `2026-11-17T07:03:18.908969+00:00`
- Embedding model: `BAAI/bge-small-en-v1.5`
- LLM provider/model: `openai` / `gpt-6-luna`
- Top-k: `3`
- Chunk size / overlap: `256` / `25`
- Eval file: `C:\tarun\projects\SupportDesk\src\dataset\eval_questions.jsonl`

## Scores

| Measure | Target | Actual |
| --- | --- | --- |
| All-required-source coverage at 3 | At least 13/14 | 14/14 (100.00%) |
| Behavior accuracy | All 16 reviewed | 15/16 (93.75%) |
| Citation validity | No invented/unresolved citations | 16/16 (100.00%) |
| Latency p50 / p95 (ms) | n/a | 2305 / 3118 |

## Special cases (T13 / T14 / T15)

- `T13`: expected `insufficient_evidence`, predicted `answered`, pass=False
- `T14`: expected `insufficient_evidence`, predicted `insufficient_evidence`, pass=True
- `T15`: expected `needs_clarification`, predicted `needs_clarification`, pass=True

## Per-question results

| id | retrieval | behavior | citations | expected | predicted | ms |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | pass | pass | pass | answered | answered | 3493 |
| T02 | pass | pass | pass | answered | answered | 1764 |
| T03 | pass | pass | pass | answered | answered | 2316 |
| T04 | pass | pass | pass | answered | answered | 2993 |
| T05 | pass | pass | pass | answered | answered | 1374 |
| T06 | pass | pass | pass | answered | answered | 2016 |
| T07 | pass | pass | pass | answered | answered | 2598 |
| T08 | pass | pass | pass | answered | answered | 2639 |
| T09 | pass | pass | pass | answered | answered | 2731 |
| T10 | pass | pass | pass | answered | answered | 1764 |
| T11 | pass | pass | pass | answered | answered | 2349 |
| T12 | pass | pass | pass | answered | answered | 2279 |
| T13 | n/a | fail | pass | insufficient_evidence | answered | 2294 |
| T14 | n/a | pass | pass | insufficient_evidence | insufficient_evidence | 1634 |
| T15 | pass | pass | pass | needs_clarification | needs_clarification | 2370 |
| T16 | pass | pass | pass | answered | answered | 1966 |

## Failure analysis

- `T13`: retrieval=None, behavior=False, citations=True, expected_sources=[], retrieved_files=['cancellations.md', 'returns.md', 'shipping.md'], stages=[]

## Manual review still required

Fields left null for human review: `correctness_pass`, `grounding_pass`, `citation_support_pass`.
Check that answers preserve policy conditions/exceptions and that cited passages actually support each claim.
