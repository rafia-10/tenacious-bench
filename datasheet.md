# Datasheet — Tenacious-Bench v0.1

**Following:** Gebru et al. (2021) "Datasheets for Datasets" (seven sections) and Pushkarna et al. (2022) "Data Cards: Purposeful and Transparent Dataset Documentation" (telescopic / periscopic / microscopic layering).

**Version:** 0.1 (pre-publication, interim submission)
**Authors:** Rafia
**Date:** 2026-04-29
**License:** CC-BY-4.0
**Contact:** rafia@10academy.org

---

## 1. Motivation

### Telescopic (one-line summary)
Tenacious-Bench v0.1 grades five sales-agent failure modes that no public benchmark currently measures for Tenacious-style outbound B2B sales.

### Periscopic (paragraph)
Standard benchmarks — τ²-Bench retail (0.80 pass@1), BFCLv3, ToolBench — evaluate agents on cooperative, policy-constrained tasks with single-hop ground truth. Outbound B2B sales agents face structurally different problems: they must ground every factual claim in a live prospect brief, respect a dynamic bench capacity document, apply a probabilistic ICP classifier with commercial stakes, and maintain a brand voice under adversarial multi-turn pressure. None of these failure modes appear in published evaluation suites.

Week 10 of the Tenacious Rapid Prototyping programme produced a working Conversion Engine (Tenacious's outbound pipeline) along with 34 documented probes covering each failure mode. The probe library showed: 45% trigger rate on competitor gap over-claiming (P-031, P-033, P-034), 35% on signal over-claiming (P-005, P-006, P-008), 20% on ICP misclassification (P-001, P-003, P-004), 15% on tone drift (P-013, P-015, P-016), and 5% on bench over-commitment (P-009, P-011, P-012) — each with catastrophic downstream consequences. Tenacious-Bench v0.1 converts this evidence into a machine-verifiable benchmark.

### Microscopic (field-level)
- **Gap motivating each dimension:** documented per probe ID in `audit_memo.md`.
- **Intended primary use:** evaluate the Tenacious outbound sales agent on five grounded-honesty dimensions. Secondary use: fine-tune a Path B preference judge.
- **Who funded this?** No external funding. Built as part of 10Academy Rapid Prototyping Week 11.

---

## 2. Composition

### Telescopic
300 evaluation tasks across 5 dimensions, 4 source modes, 3 difficulty levels; split into train (152), dev (89), held-out (59).

### Periscopic

**Dimensions:**

| Dimension | Count | % | Week 10 trigger rate | Failure consequence |
|---|---|---|---|---|
| signal_grounding_fidelity | 132 | 44% | 35% | CTO debunks claim; credibility loss |
| tone_preservation | 61 | 20% | 15% | Brand damage, open probes P-015, P-016 |
| bench_commitment_honesty | 45 | 15% | 5% | Signed SOW Tenacious cannot fulfil |
| icp_segment_appropriateness | 35 | 12% | 20% | $480K ACV per misclassification |
| competitor_gap_honesty | 27 | 9% | 45% | Irreversible brand damage |

**Source modes:**

| Mode | Count | % | ID prefix | Description |
|---|---|---|---|---|
| trace_derived (TR) | 90 | 30% | TB-TR-XXX | Real Week 10 traces, redacted and restructured |
| programmatic (PG) | 90 | 30% | TB-PG-XXXX | Template + parameter sweep (stack, headcount, signal confidence) |
| llm_synthetic (PE) | 75 | 25% | TB-PE-XXXX | Probe expansions via DeepSeek V3.2, quality-filtered by Qwen3 |
| adversarial_hand_authored (HA) | 45 | 15% | TB-HA-XXXX | Hand-authored to defeat Week 10 agent on known edge cases |

**Difficulty distribution:**
- Level 1 (108 tasks): single-check rubric, clear pass/fail
- Level 2 (108 tasks): multi-check rubric, moderate ambiguity
- Level 3 (84 tasks): highest adversarial pressure, edge cases requiring chain reasoning

**Task types:** email_generation (255), staffing_commitment_response (45).

**Null candidate_output:** 0 of 300 tasks have `candidate_output: null`. All tasks include a model-generated or hand-authored candidate output for scoring.

### Microscopic

**Schema fields per task:**
- `task_id`: `TB-{PG|PE|HA|TD}-{number}` — encodes source mode
- `dimension`: one of the five graded failure modes
- `difficulty`: 1–3 integer
- `source_mode`: `programmatic`, `llm_synthetic`, `adversarial_hand_authored`, `trace_derived`
- `task_type`: `email_generation` or `staffing_commitment_response`
- `input`: `hiring_signal_brief` (or null), `competitor_gap_brief` (or null), `bench_summary`, `prior_thread`, `style_guide_constraints`
- `candidate_output`: model-generated response being evaluated (null = template)
- `correct_output`, `incorrect_output`: rubric descriptions for human reference
- `ground_truth`: dimension-specific scoring parameters
- `rubric`: `{scoring_function, pass_threshold, dimensions_scored, max_score}`
- `metadata`: `authored_date`, `source_trace_id`, `source_probe_id`, `partition`, `contamination_checked`

Full schema: [`schema.json`](schema.json). Rubric logic: [`rubric_schema.json`](rubric_schema.json).

---

## 3. Collection Process

### Telescopic
Tasks authored in four modes over Days 1–2; each task filtered by rubric-application clarity before inclusion.

### Periscopic

**Trace-derived (90 tasks):** Week 10 agent traces provided the seed. Each trace was redacted (prospect names replaced with pseudonyms), restructured into `(input, candidate_output)` format, and labeled with ground-truth rubric scores from probe outcomes. Covers signal_grounding_fidelity, tone_preservation, and icp_segment_appropriateness. Source trace IDs in `metadata.source_trace_id`. Implementation: `generation_scripts/trace_derived.py`.

**Programmatic (90 tasks):** A template-based generator parameterised over: company stage (Seed/A/B/C), signal confidence (low/medium/high), stack (python/go/data/ml/infra), headcount (25–200), and bench state (under/at/over capacity). Covers two dimensions: 45 signal_grounding_fidelity (3 stages × 3 confidences × 5 stacks) and 45 bench_commitment_honesty (3 bench_states × 3 stacks × 5 request variants). Implementation: `generation_scripts/generate_all.py`.

**LLM-synthetic probe expansions (75 tasks):** Week 10 probe library (34 probes) provided seeds. Each probe was expanded into 3–4 variants anchored to documented failure scenarios. Synthesis model: DeepSeek V3.2 (via OpenRouter). Quality filter: Qwen3-235B-A22B (different model family — prevents preference leakage per Li et al. 2025). Covers competitor_gap_honesty, signal_grounding_fidelity, and tone_preservation. Implementation: `generation_scripts/generate_all.py`.

**Hand-authored adversarial (45 tasks):** Written to specifically test edge cases the programmatic and synthesis pipelines under-cover. Scenarios include: banned re-engagement phrases (5 phrase variants × 3 multi-turn contexts), fabricated competitor claims with contradicting public evidence (P-031 variants), statistical dishonesty on peer counts (P-033 variants), inverted signal scenarios (P-034 variants), and ICP segment mismatches (5 scenario templates × 3 cycles). Implementation: `generation_scripts/generate_all.py`.

**Quality filter applied to all tasks:**
- Pointwise judge scoring on three dimensions: input coherence (≥4/5), ground-truth verifiability (≥4/5), rubric-application clarity (≥4/5).
- Pairwise comparison when two synthesis paths produce similar tasks.
- `contamination_checked: false` in metadata = task was authored but not yet validated. These must be resolved before the held-out is used for leaderboard scoring.

### Microscopic

**Human labeling:** 30 tasks from dev were hand-labeled, then re-labeled 24 hours later. Agreement: 87% overall, 75% on competitor_gap_honesty (triggering rubric revision to 83%). Details: [`inter_rater_agreement.md`](inter_rater_agreement.md).

**Generation dates:** All tasks authored 2026-04-28 to 2026-04-29. Snapshot dates in `bench_summary.snapshot_date` range from 2026-03-01 to 2026-04-29.

---

## 4. Preprocessing and Cleaning

### Telescopic
Style-guide constraint placeholders ("Constraint variant 1–3") were expanded into actual constraint text. Source-mode metadata was corrected to match task_id prefixes.

### Periscopic

Two preprocessing passes were applied to `dataset/*/tasks.jsonl` on 2026-04-29:

1. **Source-mode correction:** All tasks were generated with `source_mode: "trace_derived"` due to a metadata bug in the authoring scripts. Corrected by mapping task_id prefix: TB-PG-* → programmatic, TB-PE-* → llm_synthetic, TB-HA-* → adversarial_hand_authored, TB-TD-* → trace_derived. Affected: 171 of 183 tasks.

2. **Style-guide constraint expansion:** Placeholder strings ("Constraint variant 1", "Constraint variant 2", "Constraint variant 3") were replaced with the actual Tenacious style-guide rules they represent (from `probe_library.md`). This makes the `check_tone_preservation` scoring function deterministic. Script: inline fix applied in `contamination_check.py`.

3. **Task-type inference:** `task_type` was null for all 183 tasks (authoring bug). Assigned based on dimension + input structure: bench_commitment tasks → `staffing_commitment_response`, all others → `email_generation`.

### Microscopic

- **Contamination check status:** `contamination_checked: false` in metadata for all current tasks. Contamination check script (`contamination_check.py`) has been run and results logged to `contamination_check.json`. 23/38 held-out tasks have N-gram or embedding violations (template-language shared across partitions). Remediation: similarity-aware re-partitioning planned for v0.2.
- **No personally identifiable information (PII):** all prospect names are pseudonyms (Camila, Jordan, Alex, Maya, Sophia, Morgan); company names are synthetic (NovaPay, RealFunctionalTest, DataFlow AI, BuildFirst). No real Tenacious prospect data used.

---

## 5. Uses

### Telescopic
Primary: evaluate outbound B2B sales agents on grounded-honesty dimensions. Secondary: training data for Path B preference judge.

### Periscopic

**Intended uses:**
- Baseline scoring of the Week 10 Tenacious Conversion Engine on five failure modes.
- Training and evaluating a preference-tuned judge (Path B) for deployment as a rejection-sampling gate.
- Community benchmark for any sales/outreach agent claiming grounded-honesty alignment.

**Out of scope:**
- General-purpose NLU evaluation (no linguistic or reading-comprehension tasks).
- Customer service / support agent evaluation (no policy-lookup tasks).
- Evaluation of agents outside B2B outbound contexts.
- Re-using the held-out partition for training (contaminates leaderboard).

**Potential misuse:**
- Do not use training tasks to fine-tune an agent and then report held-out scores as uncontaminated — the 50/30/20 split exists for this reason.
- Competitor gap tasks should not be used to train agents to make generic competitor claims; the rubric penalises fabrication.

### Microscopic

**Downstream harm potential:** Low. The dataset contains no real prospect data, no PII, and no real company confidential information. Style-guide constraints are derived from public-facing writing principles.

---

## 6. Distribution

### Telescopic
CC-BY-4.0. Published on HuggingFace Hub under `rafia-10/tenacious-bench` (pending).

### Periscopic

- **License:** CC-BY-4.0. Attribution required; commercial use permitted.
- **HuggingFace dataset:** to be published at `https://huggingface.co/datasets/rafia-10/tenacious-bench` after program staff sign-off.
- **Held-out partition:** released separately after the leaderboard is published. Not included in the initial public release.
- **Version:** v0.1. Breaking schema changes will increment the major version.

---

## 7. Maintenance

### Telescopic
Maintained by Rafia. Issues and corrections via GitHub. v0.2 planned with similarity-aware re-partitioning and multi-person labeling.

### Periscopic

**Known issues in v0.1:**
1. Template-language N-gram overlap between train and held-out. Brand-voice phrases ("30-minute scoping conversation", Tenacious sign-off) appear across partitions due to shared templates. Planned fix: cosine-similarity-aware stratified split in v0.2.
2. Dimension imbalance: signal_grounding_fidelity is 44% of the dataset (132/300), partly because the trace dataset concentrates on this dimension. v0.2 will add more competitor_gap_honesty and icp_segment_appropriateness tasks to balance.
3. Intra-rater agreement only (single labeler) for hand-authored and programmatic tasks. Multi-person IRR planned for v0.2.
4. Trace tasks (TB-TR-XXX) contain a `context` field in `input` not present in other source modes. This is preserved for traceability but is ignored by scoring functions.

**Update policy:** Tasks are not modified once in the held-out partition. Preprocessing bugs are fixed in train/dev and documented in this datasheet. Schema changes increment the version.

**Contact:** rafia@10academy.org or open a GitHub issue.
