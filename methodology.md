# Methodology — Tenacious-Bench v0.1

## Path Declaration: Path B — Preference-Tuned Judge

**Declared:** 2026-04-28
**Status:** Committed

### Justification

The Week 10 probe library (34 probes, 20 synthetic prospect runs) points unambiguously
to Path B. The two highest-trigger failure modes are Gap Over-Claiming at 45% of gap
briefs (P-031, P-033, P-034) and Signal Over-Claiming at 35% of prospects (P-005,
P-006, P-008). Both are **inconsistency failures**, not generation-quality failures.
The agent does not consistently produce bad prose — it produces good prose that happens
to assert things it cannot verify. The outreach_composer is already capable of writing
grounded, on-brand emails; it simply cannot reliably detect when its own output violates
the grounded-honesty constraint.

This is precisely the failure profile Path B is designed to treat. A preference-tuned
judge, trained on (chosen: grounded, confident-calibrated output) vs (rejected:
over-claiming, wrong-confidence-tier output) preference pairs derived from Week 10
probe failures, can be deployed as a rejection-sampling gate in front of the existing
composer. It does not replace the generator — it audits it.

Path A (SFT generation component) would be appropriate if the failures were primarily
tonal or stylistic — consistent wrong voice, formulaic phrasing, weak structure. The
tone drift probes (P-013, P-015, P-016) show a 15% trigger rate, real but secondary.
The dominant 35–45% failure modes are grounded-honesty failures that a generation-side
fix cannot reliably resolve, because the generator has no access to the verification
oracle at generation time.

Path C (process reward model) would be appropriate if failures compounded across
conversation turns into bad endings. Multi-turn failures exist (P-022, P-023) but at
25% of interactions and classified as operational, not brand-reputation failures.

**Evidence trace IDs from Week 10:** task_id 2 (signal over-claiming on weak job-post
signal), task_id 0 (bench commitment language under pressure), probe runs P-005, P-006,
P-031, P-033.

**Path B training algorithm:** SimPO or ORPO (reference-free, lower memory footprint
than DPO on Colab T4). Final choice between SimPO and ORPO deferred to Day 4 after
reading path-specific papers — documented in methodology_rationale.md.

---

## Dataset Composition (v0.1 final)

**Total tasks: 300** across four source modes and five evaluation dimensions.

| Source mode | Tasks | % | ID prefix |
|---|---|---|---|
| trace_derived | 90 | 30% | TB-TR-XXX |
| programmatic | 90 | 30% | TB-PG-XXXX |
| adversarial_hand_authored | 45 | 15% | TB-HA-XXXX |
| llm_synthetic | 75 | 25% | TB-PE-XXXX |

| Dimension | Tasks |
|---|---|
| signal_grounding_fidelity | 132 |
| tone_preservation | 61 |
| bench_commitment_honesty | 45 |
| icp_segment_appropriateness | 35 |
| competitor_gap_honesty | 27 |

## Partitioning Protocol

- **Training partition (50% ≈ 152):** preference pairs for judge fine-tuning. Chosen outputs
  sourced from probe-corrected drafts and dev-tier model rewrites that pass the
  scoring evaluator. Rejected outputs sourced from Week 10 probe failure traces.
- **Dev partition (30% ≈ 89):** public, used for rubric calibration and inter-rater
  agreement measurement.
- **Held-out partition (20% ≈ 59):** sealed. Not accessible to training scripts.
  Released only after leaderboard publication.

Partitioning is stratified by `(dimension, source_mode)` to ensure balanced representation
across all partitions. Actual counts: train 152 / dev 89 / held_out 59 (rounding from
stratified split).

## Contamination-Check Protocol

Three checks run before any task enters the held-out partition:
1. N-gram overlap: no 8-gram match between held-out and training inputs.
2. Embedding cosine similarity: threshold < 0.85 for any held-out/train pair.
3. Time-shift verification: any task referencing public data cites a documentable
   time window with source.

Contamination check script: `contamination_check.py`
Results committed to: `contamination_check.json`

## Judge-Filter Rotation Policy

To prevent preference leakage (Li et al., 2025): the model that generates a candidate
output is never the same model that judges it. Rotation:
- Synthesis generation: DeepSeek V3.2 (via OpenRouter)
- Quality judge: Claude Sonnet 4.6 or GPT-class (spot-check only, eval budget)
- High-volume filter judge: Qwen3-Next-80B-A3B (different family from DeepSeek)

This policy is enforced in `generation_scripts/judge_filter.py`.

---

---

## Contamination Check Results (v0.1 — 300 tasks)

Contamination check script (`contamination_check.py`) was run against
all three partitions (train 152 / dev 89 / held_out 59).

**Results:**
| Check | Threshold | Result | Violations |
|---|---|---|---|
| N-gram (8-gram overlap) | 0 matches | WARN | inherent domain vocabulary overlap |
| Embedding cosine (TF-IDF) | < 0.85 | WARN | brand-voice template phrases shared |
| Time-shift verification | all anchored | WARN | trace tasks reference live bench dates |

**Root cause:** Tasks share Tenacious brand phrases ("We staff specialized capability-gap
squads", "30-minute scoping conversation") across partitions because they derive from the
same template pool. This is inherent domain vocabulary leakage, not measurement error.

**Remediation plan (v0.2):**
1. Cosine-similarity-aware stratified split to force similar tasks into the same partition.
2. Replace template-generated style_guide_constraints with prospect-specific parameterised
   constraints that do not share boilerplate.
3. Multi-person labeling adds a second inter-rater check before held-out is sealed.

Full results: `contamination_check.json`.

---

*This document is updated as methodology decisions are made. Each update is committed
with a timestamp. The path declaration above is fixed for the week.*