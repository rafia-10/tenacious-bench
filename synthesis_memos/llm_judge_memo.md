# Synthesis Memo: A Survey on LLM-as-a-Judge (Gu et al., 2024–2025)

**Paper:** Gu et al., "A Survey on LLM-as-a-Judge," arXiv, latest revision 2025.
**Memo author:** Rafia
**Date completed:** 2026-04-29
**Assigned reading:** Common to all paths

---

## Core Argument

The paper surveys two decades of automated evaluation and argues that LLMs-as-judges now achieve human-level alignment on several benchmarks (MT-Bench, Chatbot Arena), but are subject to five systematic biases that can silently corrupt benchmark scores: position bias, verbosity bias, self-enhancement bias, sycophancy, and preference leakage (when the generator and judge are the same model family). The paper's prescriptions are: use ensemble judges from different families, prefer pairwise comparison over pointwise scoring for close cases, calibrate with reference answers, and rotate generator/judge families to prevent leakage.

---

## Three Design Choices I Apply Directly to Tenacious-Bench

### 1. Model-rotation to prevent preference leakage
The survey's section on preference leakage (Li et al., 2025, cited within) is the strongest constraint on my generation pipeline. For Tenacious-Bench, all synthesis generation uses DeepSeek V3.2 (via OpenRouter); quality judging uses Claude Sonnet 4.6 or Qwen3-Next-80B-A3B (different families). The `generation_scripts/judge_filter.py` enforces this rotation as a hard constraint. I agree with the survey here — the failure mode is real and measurable in my data: task TB-PE-013 shows exactly the kind of inconsistency that would be masked if the same model generated and judged it.

### 2. Pointwise → pairwise escalation for ambiguous cases
The survey recommends pairwise comparison when two tasks are close. My judge filter applies this: when two synthesised tasks share >0.80 cosine similarity, the judge is called in pairwise mode to select the more diagnostic task. I implemented this in `generation_scripts/judge_filter.py` (`pairwise_select` function). The survey's evidence (Table 3) shows pairwise outperforms pointwise for preference alignment by 6–12pp on MT-Bench.

### 3. Calibration by sampling 50 tasks for spot-check
The survey recommends calibrating a cheap bulk-filter judge against a more expensive oracle on a sample. I apply this: Qwen3-Next-80B-A3B is the high-volume filter judge (dev-tier cost); Claude Sonnet 4.6 spot-checks a 50-task sample and the agreement rate is reported in the judge filter logs. Any dimension where agreement falls below 80% triggers a rubric revision — same threshold I use for inter-rater agreement.

---

## One Specific Disagreement

**Section 4.3 — "Verbosity bias is the dominant failure mode."** The survey argues that across most evaluation tasks, judges prefer longer outputs. I disagree that this generalises to the Tenacious domain. In Tenacious sales outreach, the most common failure is *over-claiming* (adding unverifiable competitor facts, inflating bench capacity) — which often manifests as longer, more specific-sounding emails. My Week 10 evidence: probe P-031 (competitor gap over-claim) and P-005 (signal over-claim) both produced emails that were longer and more confident-sounding than correct outputs. In the Tenacious domain, verbosity bias would push judges toward the *wrong* answer. My rubric explicitly penalises ungrounded specificity regardless of output length.

**Evidence:** On 20 synthetic runs, the failing outputs in P-031/P-033 averaged 187 words vs 142 words for passing outputs. A verbosity-biased judge would assign higher scores to the wrong emails at a rate I estimate at 35–40% (matching the trigger rate).

**Design implication:** For Tenacious-Bench judge calibration, I include length-matched pairs in the 50-task spot-check sample to detect if the cheap-model judge is exhibiting verbosity bias. This is not recommended by the survey but is necessary for this domain.

---

## What I Would Add to the Survey

The survey does not address *domain-specific* judge calibration — all examples are from general NLU or chat benchmarks. A practical prescriptions section for vertical-domain benchmarks (sales, legal, medical) would strengthen the paper. The leakage section (5.2) is the most useful for practitioners; it should be expanded with a checklist format.
