# Evaluation Report

Synthetic gold set; see `data/eval/gold.jsonl`.

## Configuration

| Setting | Value |
|---|---|
| retriever | `bm25` |
| generator | `offline-extractive` |
| top_k | `4` |
| chunk_size / overlap (words) | `80 / 20` |
| min_score / min_query_coverage | `4.0 / 0.4` |

## Summary

| Metric | Value |
|---|---|
| questions | 25 |
| answerable | 20 |
| unanswerable | 5 |
| recall@4 | 1.000 |
| mrr | 1.000 |
| citation_precision | 0.908 |
| groundedness | 1.000 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.000 |
| latency_p50_ms | 0.560 |
| latency_p95_ms | 0.680 |

## Per-question results

| id | answerable | refused | top score | recall | RR | cite prec | grounded | cited |
|---|---|---|---|---|---|---|---|---|
| q01 | True | False | 18.72 | 1.000 | 1.000 | 1.000 | 1.000 | discharge-planning-guideline |
| q02 | True | False | 11.13 | 1.000 | 1.000 | 0.500 | 1.000 | discharge-planning-guideline, fall-prevention-protocol |
| q03 | True | False | 10.60 | 1.000 | 1.000 | 1.000 | 1.000 | prior-authorization-imaging |
| q04 | True | False | 13.68 | 1.000 | 1.000 | 1.000 | 1.000 | prior-authorization-imaging |
| q05 | True | False | 20.83 | 1.000 | 1.000 | 1.000 | 1.000 | prior-authorization-imaging |
| q06 | True | False | 18.74 | 1.000 | 1.000 | 1.000 | 1.000 | prior-authorization-specialty-drugs |
| q07 | True | False | 18.79 | 1.000 | 1.000 | 1.000 | 1.000 | prior-authorization-specialty-drugs |
| q08 | True | False | 19.31 | 1.000 | 1.000 | 1.000 | 1.000 | medication-reconciliation-protocol |
| q09 | True | False | 15.98 | 1.000 | 1.000 | 1.000 | 1.000 | medication-reconciliation-protocol |
| q10 | True | False | 31.63 | 1.000 | 1.000 | 1.000 | 1.000 | fall-prevention-protocol |
| q11 | True | False | 7.99 | 1.000 | 1.000 | 0.500 | 1.000 | fall-prevention-protocol, sepsis-escalation-faq |
| q12 | True | False | 15.72 | 1.000 | 1.000 | 1.000 | 1.000 | sepsis-escalation-faq |
| q13 | True | False | 13.96 | 1.000 | 1.000 | 1.000 | 1.000 | sepsis-escalation-faq |
| q14 | True | False | 12.19 | 1.000 | 1.000 | 1.000 | 1.000 | post-discharge-follow-up-calls |
| q15 | True | False | 9.24 | 1.000 | 1.000 | 1.000 | 1.000 | post-discharge-follow-up-calls |
| q16 | True | False | 18.40 | 1.000 | 1.000 | 1.000 | 1.000 | home-health-referral-policy |
| q17 | True | False | 9.89 | 1.000 | 1.000 | 0.500 | 1.000 | interpreter-services-policy, fall-prevention-protocol |
| q18 | True | False | 25.06 | 1.000 | 1.000 | 1.000 | 1.000 | skilled-nursing-transfer-faq |
| q19 | True | False | 12.22 | 1.000 | 1.000 | 1.000 | 1.000 | pressure-injury-prevention |
| q20 | True | False | 15.01 | 1.000 | 1.000 | 0.667 | 1.000 | skilled-nursing-transfer-faq, home-health-referral-policy, post-discharge-follow-up-calls |
| q21 | False | True | 3.33 | n/a | n/a | n/a | n/a | - |
| q22 | False | True | 5.85 | n/a | n/a | n/a | n/a | - |
| q23 | False | True | 5.63 | n/a | n/a | n/a | n/a | - |
| q24 | False | True | 4.59 | n/a | n/a | n/a | n/a | - |
| q25 | False | True | 5.38 | n/a | n/a | n/a | n/a | - |
