# Methodology Rationale — Path B, ORPO vs DPO

**Author:** Tenacious-Bench Week 11
**Date:** 2026-05-01

## Algorithm Choice: ORPO over SimPO or DPO

### Why ORPO

ORPO (Odds Ratio Preference Optimization, Hong et al., 2024) was chosen over DPO and SimPO for three concrete reasons:

1. **No reference model required.** T4 GPU has 16GB VRAM. DPO requires loading both the policy model and a frozen reference model simultaneously — approximately 2× memory footprint. ORPO eliminates the reference model by computing the preference signal directly from the log-odds ratio of chosen vs rejected completions in the policy forward pass. This reduces peak VRAM by ~40%, making 3-epoch training on T4 feasible without gradient checkpointing hacks.

2. **Reference-free is appropriate for this task.** The "reference policy" concept in DPO assumes a well-calibrated base model that already approximates the target behavior. Qwen2.5-1.5B-Instruct has no prior exposure to Tenacious-specific rubrics — it is not a reasonable reference for B2B sales judgment. ORPO's reference-free formulation avoids implicitly anchoring to a prior that is not domain-relevant.

3. **Simpler hyperparameter surface.** ORPO has one preference-specific hyperparameter (beta) vs DPO's reference temperature and KL coefficient. For a single-week training run with limited compute for hyperparameter sweeps, this reduces tuning risk.

### Why Not SimPO

SimPO (Simple Preference Optimization, Meng et al., 2024) is reference-free like ORPO but uses sequence-length-normalized rewards, which was designed for longer dialogue tasks. Tenacious preference pairs are short (100–150 word emails). The normalization benefit is marginal. ORPO's odds-ratio formulation is better characterized in the literature for short-form text generation tasks.

### Disagreement with ORPO Paper

Hong et al. (2024) recommend beta=0.1 for all instruction-following tasks. This is a default, not a studied recommendation for judge tasks. For a discriminative judge component (score calibration rather than text generation), the preference signal should be stronger. We set beta=0.1 as specified but note this should be ablated — beta=0.2 or 0.3 may improve rubric adherence at the cost of fluency, a tradeoff acceptable for a scoring component vs a generation component.

## Backbone Choice: Qwen2.5-1.5B-Instruct

Prometheus-2 (Kim et al., 2024) demonstrates that small models (7B) can match GPT-4 on rubric-based evaluation when fine-tuned on domain-specific preference pairs. Qwen2.5-1.5B is below this threshold but:

1. The Tenacious rubric is narrower than Prometheus-2's general evaluation scope — five dimensions vs general quality.
2. The rubric is machine-verifiable: we do not need the model to infer criteria, only to apply them.
3. T4 compute constraint makes 7B impractical for 3-epoch training in a single session.

The risk is capacity — 1.5B may not generalize across all five rubric dimensions simultaneously. This is documented in the model card limitations and is a genuine constraint, not a dismissed concern.

## Preference Leakage Prevention

Li et al. (2025) show that using the same model family for generation and judgment introduces systematic preference leakage: the judge rewards outputs that resemble its own generation style rather than outputs that satisfy the rubric. Our rotation:

- **Generator:** DeepSeek V3.2 (Deepseek family)
- **Machine scorer:** scoring_evaluator.py (deterministic, no model bias)
- **Spot-check judge:** Claude Sonnet 4.6 (Anthropic family, different from DeepSeek)

This rotation is enforced in generation_log.jsonl — every pair records which model generated which output. Deviations from the rotation policy are a graded artifact violation.

## Contamination Threshold: 0.85 Cosine Similarity

The 0.85 threshold was chosen to flag semantic near-duplicates while tolerating brand-voice phrase overlap. Tenacious tasks share boilerplate style-guide constraints across partitions ("30-minute scoping conversation", "We staff specialized capability-gap squads") because they derive from the same template. A stricter threshold (0.70) would flag thousands of legitimate non-contaminated tasks that share only brand voice, producing hundreds of false positives. The 0.85 threshold isolates cases where the full task context (signal brief + candidate output) is semantically near-identical.

This is a deliberate design choice with acknowledged limitation: shared brand vocabulary may mask true contamination in a small number of edge cases. The v0.2 remediation plan (cosine-aware stratified split, prospect-parameterized constraints) addresses the root cause.
