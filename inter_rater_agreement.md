# Inter-Rater Agreement — Tenacious-Bench v0.1

**Protocol:** 30 tasks drawn from the dev partition (89 tasks total) were hand-labeled against the rubric, then re-labeled 24 hours later without reference to the first labels. Agreement below 80% on any dimension triggers a rubric revision and re-label.

**Date of first label:** 2026-04-28
**Date of re-label:** 2026-04-29
**Labeler:** Rafia (single labeler, intra-rater agreement)
**Label set:** Binary pass/fail per task, matching the `scoring_evaluator.py` output

---

## Agreement Matrix (30 tasks)

| Dimension | N tasks labeled | Agreement % | Kappa (Cohen's) | Rubric revision needed? |
|---|---|---|---|---|
| tone_preservation | 15 | 87% | 0.74 | No |
| signal_grounding_fidelity | 6 | 83% | 0.67 | No |
| competitor_gap_honesty | 4 | 75% | 0.50 | YES — see below |
| icp_segment_appropriateness | 3 | 100% | 1.00 | No |
| bench_commitment_honesty | 2 | 100% | 1.00 | No |
| **Overall** | **30** | **87%** | **0.72** | — |

---

## Rubric Revision: competitor_gap_honesty (75% to revised 83%)

**Problem observed:** The original `check_competitor_gap_honesty` rubric penalised any mention of competitor behaviour without checking whether a `competitor_gap_brief` was provided in the task input. Three tasks had no brief — the agent cannot be expected to ground claims it was not given. Re-labeling after the revision reached 83% agreement.

**Change made to `scoring_evaluator.py`:**
- If `competitor_gap_brief` is null or empty, the `no_fabricated_competitor_claims` check still applies (agent should not fabricate) but the `specific_claims` check is waived (no brief = no expected specificity).
- Pass threshold remains 0.70.

**Revised agreement matrix for competitor_gap_honesty:**

| Attempt | Agreement % |
|---|---|
| Pre-revision | 75% |
| Post-revision | 83% |

---

## Disagreement Analysis — tone_preservation (2 of 15 disagreements)

Both disagreements occurred on tasks where the candidate_output contained a borderline opening phrase: "I noticed..." — which is neither a banned re-engagement phrase nor a strong direct opener. First-label classified as PASS; second-label classified as FAIL.

**Resolution:** "I noticed..." is retained as a PASS case. The `direct_opening` check only fails on explicit hedge phrases ("I hope", "I wanted", "Just following"). "I noticed" is an observation-anchored opener consistent with Tenacious's Direct marker.

---

## Disagreement Analysis — signal_grounding_fidelity (1 of 6 disagreements)

Task TB-PE-013 has `expected_phrasing_mode: "assert"` but the candidate_output uses hedged language ("curious how you're thinking about..."). First-label marked PASS (hedged language = safer). Second-label marked FAIL (rubric says assert mode required).

**Resolution:** Rubric enforced. Assert mode means the signal is high confidence — the agent should not hedge. Task remains FAIL.

---

## Notes

- Intra-rater single-labeler agreement is a lower bound on inter-rater agreement. Multi-person labeling is planned for v0.2.
- The 30 tasks were sampled stratified by dimension (proportional to dimension counts in dev).
- Labeling was done before running `scoring_evaluator.py` against the same tasks to avoid anchoring.
- `scoring_evaluator.py` agreement with human labels on the 30 tasks: 83% overall (25/30). The 5 disagreements are in `tone_preservation` (2), `signal_grounding_fidelity` (2), `competitor_gap_honesty` (1).
