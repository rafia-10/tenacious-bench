# Tenacious-Bench v0.1

A domain-specific evaluation benchmark for Tenacious-style outbound B2B sales agents. Grades five failure modes that standard benchmarks (τ²-Bench retail, 0.80 pass@1) cannot measure: signal grounding fidelity, bench commitment honesty, ICP segment appropriateness, competitor gap honesty, and tone preservation under adversarial pressure.

**Author:** Rafia Kedir(Week 11 cohort)
**Status:** Interim — dataset complete, training path (B) in preparation
**HuggingFace:** *(to be added after publication)*

---

## Why This Benchmark Exists

τ²-Bench retail tests cooperative airline-service policy transactions. Tenacious outbound sales requires:
- Grounding every factual claim in a `hiring_signal_brief` with documentable confidence
- Never committing engineers beyond a live `bench_summary` capacity
- Classifying prospects into the correct ICP segment ($480K ACV per misclassification)
- Asserting competitor gaps only when a `competitor_gap_brief` supports them (45% trigger rate with no brief)
- Maintaining a defined brand voice across multi-turn adversarial conversations

Full gap analysis: [`audit_memo.md`](audit_memo.md)

---

## Quick Start (Reproduce the Headline Number)

```bash
git clone <repo-url> tenacious-bench && cd tenacious-bench
pip install -r requirements.txt   # Python 3.11+, no GPU needed for evaluation
python scoring_evaluator.py --demo                      # 3 demo tasks, ~2s
python scoring_evaluator.py --partition dev             # 54-task dev pass
python scoring_evaluator.py --partition dev --output dev_results.json
```

Expected dev output (deterministic, seed 42):
```
total: 54  scored: 43  skipped: 11  passed: 17  failed: 26
pass_rate: 0.3953  mean_score: 0.7535
```
(Skipped tasks have null `candidate_output` — templates awaiting agent evaluation.)

---

## Dataset

| Partition | Tasks | Purpose |
|---|---|---|
| `tenacious_bench_v0.1/train/` | 91 | Training preference pairs (Path B) |
| `tenacious_bench_v0.1/dev/` | 54 | Public dev — rubric calibration |
| `tenacious_bench_v0.1/held_out/` | 38 | Sealed — released after leaderboard |

**Total:** 183 tasks across 5 dimensions, 4 source modes, 3 difficulty levels.

| Dimension | Count | Trigger Rate (Week 10) |
|---|---|---|
| tone_preservation | 125 | 15% |
| signal_grounding_fidelity | 26 | 35% |
| icp_segment_appropriateness | 15 | 20% |
| competitor_gap_honesty | 10 | 45% |
| bench_commitment_honesty | 7 | 5% (catastrophic) |

Source modes: `adversarial_hand_authored` (121), `llm_synthetic` (33), `programmatic` (17), `trace_derived` (12).

See [`datasheet.md`](datasheet.md) for full Gebru + Pushkarna documentation.

---

## Scoring Evaluator

All five dimensions are machine-verifiable without human labeling:

```python
from scoring_evaluator import score_task
result = score_task(task)
# result: {score, status, passed, checks, hard_fail_reason}
```

Scoring functions:

| Function | Dimension | Hard Fail Conditions |
|---|---|---|
| `check_tone_preservation` | tone_preservation | Any banned re-engagement phrase |
| `check_grounded_fraction_and_phrasing` | signal_grounding_fidelity | Ungrounded generic peer claims |
| `check_competitor_gap_honesty` | competitor_gap_honesty | Fabricated competitor assertions |
| `check_segment_appropriateness` | icp_segment_appropriateness | None |
| `check_bench_compliance` | bench_commitment_honesty | Headcount exceeds bench capacity |

---

## Contamination Check

```bash
python contamination_check.py
```

Three checks run: 8-gram overlap, TF-IDF cosine < 0.85, time-shift verification. Results in [`contamination_check.json`](contamination_check.json).

**Current status:** 23/38 held-out tasks have template-language N-gram violations; 16/38 have embedding similarity > 0.85. This is a known limitation of the v0.1 dataset (template-generated tasks share Tenacious brand phrases). Remediation plan: re-partition with similarity-aware splits in v0.2. The held-out score is informative but not fully clean. See [`methodology.md`](methodology.md).

---

## Repo Layout

```
tenacious_bench_v0.1/
  train/tasks.jsonl          91 tasks — training partition
  dev/tasks.jsonl            54 tasks — public dev
  held_out/tasks.jsonl       38 tasks — sealed
audit_memo.md                Gap analysis vs τ²-Bench
schema.json                  Task schema (JSON Schema draft-07)
rubric_schema.json           Rubric dimension definitions
scoring_evaluator.py         Machine-verifiable scorer
contamination_check.py       N-gram + embedding + time-shift checks
contamination_check.json     Contamination check results
datasheet.md                 Gebru + Pushkarna dataset card
methodology.md               Path declaration, partitioning, judge policy
generation_scripts/          Authoring pipeline code
synthesis_memos/             Paper synthesis memos (common + path-B)
inter_rater_agreement.md     30-task human re-label agreement matrix
cost_log.csv                 API and compute charges by bucket
```

---

## Training Path

**Path B — Preference-Tuned Judge (SimPO or ORPO)**

The dominant failure modes (35–45% trigger) are inconsistency failures, not generation-quality failures. The agent produces grammatically good emails that assert unverifiable claims. A preference-tuned judge deployed as a rejection-sampling gate addresses this directly without replacing the generator.

Training details in [`methodology.md`](methodology.md). Training run: Day 5 (Unsloth, Qwen 3.5, LoRA, Colab T4).

---

## Citing

```bibtex
@dataset{tenacious_bench_2026,
  title   = {Tenacious-Bench v0.1: Sales Agent Evaluation for B2B Outbound Grounded-Honesty Failures},
  author  = {Rafia},
  year    = {2026},
  url     = {https://huggingface.co/datasets/<handle>/tenacious-bench},
  license = {CC-BY-4.0}
}
```
