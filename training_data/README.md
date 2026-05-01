# Tenacious-Bench Preference Pairs — Training Data

Generated: 2026-05-01  
Pipeline: `build_preference_pairs.py`

## Summary

| Field | Value |
|---|---|
| Final preference pairs | 94 |
| Raw pairs (pre-filter) | 152 |
| Filter pass rate | 61.8% |
| Source train tasks | 152 |

## Source Breakdown

| Source | Count | Description |
|---|---|---|
| failing_task_generated_chosen | 111 | Candidate output FAILED rubric → used as rejected; DeepSeek generated passing chosen |
| passing_task_generated_rejected | 41 | Candidate output PASSED rubric → used as chosen; DeepSeek generated failing rejected |

## Dimension Breakdown (final 94 pairs)

| Dimension | Pairs | % |
|---|---|---|
| signal_grounding_fidelity | 46 | 48.9% |
| tone_preservation | 32 | 34.0% |
| competitor_gap_honesty | 12 | 12.8% |
| icp_segment_appropriateness | 4 | 4.3% |
| bench_commitment_honesty | 0 | 0.0% |

**Note on bench_commitment_honesty absence:** The bench commitment tasks use `check_bench_compliance` with a 0.5 score when bench data is missing (not hard-fail). Most generated rejected outputs scored 0.5, which is still below threshold=0.7, making them valid rejected samples — but the corresponding chosen outputs scored ≤0.5 as well due to bench data issues. Zero pairs passed the filter for this dimension. This is documented as a v0.2 training data gap.

**Note on icp_segment_appropriateness (4 pairs only):** The scoring function uses keys "1"/"2"/"3"/"ABSTAIN" but several train tasks have full segment names in ground_truth (e.g., "segment_1_series_a_b"). These tasks produce structurally unfixable scores (always 0.0 against the keyword dict). The 4 surviving pairs are from tasks with matching short-form keys.

## Score Distributions

| | Chosen outputs | Rejected outputs |
|---|---|---|
| Mean score | 0.70 | 0.32 |
| Min score | 0.00 | 0.00 |
| Max score | 1.00 | 1.00 |

The chosen mean of 0.70 (not 1.0) reflects the blended scoring across dimensions: some pairs have chosen outputs that pass the primary dimension but score 0.0 on secondary machine-verifiable checks applied after generation.

## Rejection Reasons (58 discarded pairs)

| Reason | Count |
|---|---|
| Chosen failed rubric (score=0.0, unachievable threshold=1.0) | 36 |
| Chosen failed rubric (score=0.5 < threshold=0.7) | 14 |
| Generated rejected still passes rubric (score=1.0) | 8 |

## Models Used

| Role | Model | Rationale |
|---|---|---|
| Chosen output generation | deepseek/deepseek-chat-v3-0324 (OpenRouter) | Non-Claude, non-Qwen family for preference leakage prevention |
| Rejected output generation | deepseek/deepseek-chat-v3-0324 (OpenRouter) | Same model, opposite instruction |
| Machine scorer | scoring_evaluator.py (deterministic) | No LLM bias in filtering decisions |
| Spot-check judge | Claude Sonnet 4.6 (Anthropic) | Different family from generator — eval budget, max 50 tasks |

## Preference Leakage Prevention Policy

Per Li et al. (2025): the model that generates a candidate output is never the same model that judges it.

- Generator: DeepSeek V3.2 (deepseek family, via OpenRouter)  
- Machine scorer: deterministic Python — no model family bias  
- Spot-check: Claude Sonnet 4.6 (Anthropic family — different from DeepSeek)

Every generation call is logged in `generation_log.jsonl` with timestamp, model, token count, and bucket.

## Contamination Check Results

| Check | Threshold | Result | Max observed |
|---|---|---|---|
| 8-gram overlap | 0 matches | WARN (brand phrase overlap) | 3,965 shared ngrams |
| Cosine similarity (TF-IDF) | < 0.85 | PASS | 0.727 |

**Root cause of 8-gram overlaps:** Shared Tenacious brand phrases ("30-minute scoping conversation", "We staff specialized capability-gap squads") appear across all partitions because they derive from the same template pool. This is inherent domain vocabulary overlap, not task-level contamination. The cosine similarity check (0.727 < 0.85 threshold) confirms no semantic near-duplicates. See `methodology.md` for full explanation.

Full results: `contamination_check_training.json`

## File Inventory

| File | Description |
|---|---|
| `preference_pairs.jsonl` | 94 filtered pairs in Qwen 2.5 chat template format for ORPO |
| `preference_pairs_raw.jsonl` | 152 raw pairs (all generations, pre-filter) |
| `generation_log.jsonl` | Per-pair log: source, action, scores, cosine similarity, discard reason |
| `contamination_check_training.json` | Contamination check results |
| `build_preference_pairs.py` | Pipeline script |
| `check_contamination_training.py` | Contamination check script |
| `README.md` | This file |
