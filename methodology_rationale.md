# Methodology Rationale — Path B, ORPO vs DPO

**Author:** Tenacious-Bench Week 11
**Date:** 2026-05-01

## Why Path B (Preference-Tuned Judge)

We selected Path B over Path A (SFT) or Path C (Process RM) because the Week 10 probe library demonstrated that the dominant failures are **inconsistency failures**, not generation-quality or trajectory failures. The agent's generative prose is strong, but it hallucinates confidence against ground truth. Deploying a rejection-sampling gate addresses this directly.

**Concrete Evidence (Week 10 Traces):**
- **Trace Task 2** (signal over-claiming on weak job-post signal) and **Trace Task 4** (competitor gap fabrication) show grammatically perfect outputs that are factually ungrounded.
- **Probe Runs:** P-005, P-006, P-031, P-033 document the agent fabricating competitor ML platform teams and inflating signal. The generator has no access to the verification oracle at generation time, so SFT (Path A) cannot reliably solve this. A trained auditor (Path B) can.

## Algorithm Choice: ORPO over SimPO or DPO

### Why ORPO

ORPO (Odds Ratio Preference Optimization; Hong et al., 2024) was chosen over DPO and SimPO for three concrete reasons:

1. **No reference model required.** As demonstrated in **Hong et al. (2024, Section 2.2 and Figure 2)**, ORPO eliminates the reference model by computing the preference signal directly from the log-odds ratio of chosen vs rejected completions in the policy forward pass. This reduces peak VRAM by ~40%, making 3-epoch training on a 16GB T4 feasible without gradient checkpointing hacks.
2. **Reference-free is appropriate for this task.** The "reference policy" concept in DPO assumes a well-calibrated base model that already approximates the target behavior. Qwen2.5-1.5B-Instruct has no prior exposure to Tenacious-specific rubrics. As discussed in **Hong et al. (2024, Section 4)**, ORPO's reference-free formulation avoids implicitly anchoring to a prior that is not domain-relevant.
3. **Simpler hyperparameter surface.** ORPO relies heavily on a single preference-specific hyperparameter (beta) vs DPO's reference temperature and KL coefficient.

### Why Not SimPO

SimPO (Simple Preference Optimization; Meng et al., 2024) is reference-free like ORPO but applies sequence-length-normalized rewards (**Meng et al., 2024, Section 3.1, Eq. 4**), which optimally benefits longer dialogue tasks. Tenacious preference pairs are short (100–150 word emails). The normalization benefit is marginal compared to ORPO.

### Disagreement with ORPO Paper

**Hong et al. (2024, Section 4.3)** recommend beta=0.1 for all instruction-following tasks. However, for a discriminative judge component (score calibration rather than text generation), the preference signal should be stronger. We set beta=0.1 as specified but note this should be ablated — beta=0.2 or 0.3 may improve rubric adherence at the cost of fluency, a tradeoff acceptable for a scoring component vs a generation component.

## Backbone Choice: Qwen2.5-1.5B-Instruct

Prometheus-2 (**Kim et al., 2024, Section 5.3 and Figure 4**) demonstrates that small models (e.g., 7B parameters) can match GPT-4 on rubric-based evaluation when fine-tuned on domain-specific preference pairs. While Qwen2.5-1.5B is below the studied 7B threshold, we chose it because:

1. The Tenacious rubric is narrower than Prometheus-2's general evaluation scope — five dimensions vs general quality.
2. The rubric is machine-verifiable: we do not need the model to infer criteria, only to apply them.
3. T4 compute constraints restrict scaling to 7B.

This capacity constraint risks the 1.5B model underfitting simultaneous multi-dimension rubric generalization, which we document in our model card limitations.

## Preference Leakage Prevention

**Li et al. (2025, Section 3.2, Preference Leakage)** demonstrate that using the same model family for generation and judgment introduces systematic bias: the judge rewards outputs that resemble its own generation style. To prevent this, we rigidly rotate models:

- **Generator:** DeepSeek V3.2 (Deepseek family)
- **Machine scorer:** `scoring_evaluator.py` (deterministic, no model bias)
- **Spot-check judge:** Claude Sonnet 4.6 (Anthropic family, different from DeepSeek)

This rotation is enforced in `generation_log.jsonl`.

## Contamination Threshold: 0.85 Cosine Similarity

The 0.85 threshold was specifically chosen to flag semantic near-duplicates while tolerating brand-voice phrase overlap. Since Tenacious tasks share boilerplate style-guide constraints ("30-minute scoping conversation"), setting a stricter baseline threshold like 0.70 would trigger hundreds of false positives, rendering the score unusable. This deliberate design choice isolates tasks where the full context (signal brief + candidate output) is genuinely near-identical.
