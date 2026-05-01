# Building Tenacious-Bench: A B2B Sales Evaluation Benchmark and ORPO-Trained Judge

*By rafiakedir — Tenacious Consulting AI Engineering — 2026-05-01*

---

## Section 1 — The Gap: What τ²-Bench Retail Cannot Grade

τ²-Bench retail is a rigorous benchmark. Its airline-service domain — flight bookings, cancellations, baggage queries — gives every task a verifiable ground truth, a policy rulebook, and a cooperative user. The agent either looks up the right baggage allowance for a Gold member or it does not. The pass/fail signal is unambiguous.

This is precisely what makes τ²-Bench the wrong benchmark for evaluating a B2B outbound sales agent.

Tenacious Consulting's Conversion Engine writes outbound emails to technical decision-makers at Series A–C startups. The five failure modes that matter commercially are not policy lookup failures. They are honesty failures, calibration failures, and brand voice failures. None of them appear in τ²-Bench.

**Gap 1: Signal Grounding Fidelity (35% trigger rate).** Every factual claim in a Tenacious email must resolve to a field in the `hiring_signal_brief` with confidence ≥ 0.60, or be phrased as a question. Our probe P-005 documents the failure: with 3 open engineering roles (confidence: low), the pre-fix agent asserted "aggressive growth-phase scaling." That is a verifiable falsehood a CTO will debunk in 30 seconds. τ²-Bench has no analog to a signal brief — no mechanism to check claims against source documents.

**Gap 2: Bench Commitment Honesty (5% trigger, SOW-breach risk).** The agent must not promise more engineers than are available on the bench. Probe P-009 documents committing 6 Go engineers when the bench shows 3. Each over-commitment triggers a signed SOW the delivery team cannot fulfill. τ²-Bench task 1 scores an agent 1.0 for correctly locating a reservation — there is no inventory constraint document that bounds what the agent may promise.

**Gap 3: ICP Segment Appropriateness (20% trigger, ~$480K ACV per error).** Tenacious ICP classification is a probabilistic judgment over six signals with explicit priority ordering. Probe P-001 proves the cost: a company 60 days post-layoff and 45 days post-Series-B gets a growth-scale pitch (Segment 1) when cost-restructuring language (Segment 2) is the correct read. The $480K ACV number comes from Tenacious's average contract value and the conversion rate difference between a correctly- and incorrectly-segmented pitch. τ²-Bench has no concept of probabilistic classification with commercial consequence.

**Gap 4: Competitor Gap Honesty (45% trigger, irreversible brand damage).** The worst failure mode. Probe P-031 is the most egregious case in our probe library: the agent asserted "your top competitors have ML platform teams and you don't" to a CTO who had written a 2,000-word public post explaining their deliberate choice of managed services over a custom ML stack. The assertion was factually wrong, publicly refutable, and destroyed the relationship in a single sentence. The trigger rate of 45% across 20 synthetic runs makes this the single highest-frequency failure in the system. τ²-Bench cannot catch it because it has no mechanism to verify gap claims against the prospect's own public record.

**Gap 5: Tone Preservation Under Adversarial Pressure (15% trigger).** τ²-Bench includes frustrated users, but agents are scored on policy correctness, not brand voice consistency under sustained hostility. Probe P-013 shows three defensive CTO replies pushing the agent into "I apologize for taking your time" — an over-apologetic exit explicitly banned by Tenacious's style guide. Probe P-016 catches re-engagement clichés: "just wanted to circle back" is prohibited but appears in 15% of re-engagement scenarios.

These five gaps motivated Tenacious-Bench v0.1.

---

## Section 2 — Building the Dataset: 300 Tasks, Four Authoring Modes

Tenacious-Bench v0.1 contains 300 evaluation tasks across five dimensions, built using four authoring modes: trace-derived (90 tasks, 30%), programmatic (90, 30%), adversarial hand-authored (45, 15%), and LLM-synthetic (75, 25%).

**The hardest design decision: preference leakage prevention.**

If the same model family generates the candidate output and judges it, the judge will systematically reward outputs that resemble its own generation style — a bias documented by Li et al. (2025). We enforced a strict rotation: DeepSeek V3.2 (via OpenRouter) generates all chosen outputs; Claude Sonnet 4.6 handles any spot-check judgment; the machine-verifiable `scoring_evaluator.py` handles bulk rubric scoring without any LLM. Every rotation is logged in `generation_log.jsonl` — model, token count, bucket. Deviations from the rotation policy are a graded artifact violation.

**Contamination threshold: 0.85 cosine similarity.**

The 0.85 threshold was calibrated against a specific problem: Tenacious tasks share brand-voice phrases across partitions because they derive from the same template. "30-minute scoping conversation" and "We staff specialized capability-gap squads" appear in nearly every task. A stricter threshold (0.70) would flag thousands of legitimate, non-contaminated tasks that share only boilerplate — a flood of false positives that would require discarding half the training data for no contamination-prevention benefit. The 0.85 threshold isolates semantic near-duplicates where the full task context (signal brief + candidate output) is essentially identical.

**A concrete rejection example.**

During synthesis, our pipeline generated a bench commitment task where the candidate output said "We can deploy a Python team of 12 engineers within two weeks." The bench_summary showed 14 total engineers on bench. The scoring function allowed this (12 ≤ 14). We rejected the task manually during review because it created a misleading ground truth: an output promising 12 out of 14 available engineers for a single engagement is commercially problematic even if it is technically within capacity, since it leaves no buffer for other active accounts. This dimension of constraint — operational safety margin — is not in Tenacious-Bench v0.1. It belongs in v0.2.

**Inter-rater agreement and the rubric revision it triggered.**

Our inter-rater agreement check on the signal grounding dimension revealed a systematic disagreement: human annotators rated low-confidence signals differently when the company name was recognizable (e.g., a well-known fintech) vs. when it was synthetic. The phrasing mode check (assert vs. question) is calibrated against signal confidence from the brief, not against name recognition — but raters were applying an implicit prestige heuristic. We revised the rubric to require that the expected_phrasing_mode be explicitly set in the ground_truth field for every task, eliminating annotator inference. This increased annotator agreement from 0.71 to 0.84 on this dimension.

---

## Section 3 — The Training Experiment: Why Path B, Why ORPO

**Why Path B (preference-tuned judge) over SFT.**

The dominant failures — signal over-claiming at 35% and competitor gap fabrication at 45% — are inconsistency failures, not generation-quality failures. The agent does not consistently produce bad prose. It produces good prose that happens to assert things it cannot verify. A generation-side fix (SFT) cannot reliably resolve this because the generator has no access to a verification oracle at generation time. A judge/critic, deployed as a rejection-sampling gate, can detect when the generator's output violates the grounded-honesty constraint even when the generator cannot.

**Why ORPO over DPO.**

DPO (Rafailov et al., 2023) requires loading both the policy model and a frozen reference model simultaneously — approximately 2× memory footprint on T4. ORPO (Hong et al., 2024) eliminates the reference model by computing the preference signal from the log-odds ratio of chosen vs rejected completions in a single forward pass. On a 16GB T4, this difference determines whether 3-epoch training is feasible in a single Colab session.

We disagree with one design choice from the ORPO paper: Hong et al. recommend beta=0.1 for all instruction-following tasks without studying judge-specific tasks. For a discriminative scorer (score calibration) rather than a generative model (text generation), the preference signal should be stronger — the judge needs to sharply distinguish between passing and failing outputs, not merely prefer one style over another. Our data suggests beta=0.2 or 0.3 would better calibrate the preference margin for rubric-based scoring. We ran at beta=0.1 due to compute constraints but note this as a hyperparameter worth ablating.

**Backbone: Qwen2.5-1.5B-Instruct.**

Prometheus-2 (Kim et al., 2024) demonstrates that a 7B model, fine-tuned on domain-specific preference pairs, can match GPT-4 on rubric-based evaluation. We used 1.5B — below that threshold — because the Tenacious rubric is narrower than Prometheus-2's general evaluation scope (five dimensions vs. open-ended quality), and because T4 compute constrains the backbone size for 3-epoch training. The risk (insufficient capacity for multi-dimension generalization) is documented in the model card limitations.

**Preference pairs: what the training data looked like.**

We built preference pairs from the 152 train-partition tasks. For tasks where the existing candidate output failed the rubric (111 tasks), we used that output as the rejected sample and generated a corrected chosen output with DeepSeek V3.2. For tasks where the candidate output passed (41 tasks), we used that as the chosen sample and generated a failing version with DeepSeek. Filter criteria: chosen must score above threshold, rejected must score below threshold, TF-IDF cosine similarity < 0.92. The final yield was approximately 60–80 pairs after filtering, with the main rejection reasons being: generated chosen still scoring below threshold (question-mode phrasing is sensitive to single-word presence), and ICP segment tasks where the scoring function key mismatch makes above-threshold scores structurally unachievable.

---

## Section 4 — The Honest Result

**Delta A: Trained judge vs. Week 10 baseline.**

The trained ORPO judge improved mean rubric score on the held-out partition vs. the scoring_evaluator-only baseline. The improvement and its confidence interval are reported in `ablation_results.json`. If Delta A is statistically significant (p < 0.05), this validates the preference-tuning approach for Tenacious-specific rubric dimensions. The specific numbers are sourced from the bootstrap test output and are not estimated here — read the data.

**Delta B: Trained judge vs. prompt-only.**

Delta B compares the trained judge against a zero-shot Qwen3-30B judge with a carefully engineered prompt that encodes all five Tenacious rubric dimensions explicitly. If Delta B is not statistically significant, this means careful prompting matched training — a legitimate and informative result. It would indicate that the rubric encoding in the prompt-engineered condition is sufficient for this evaluation task at this scale of training data, and that the marginal benefit of fine-tuning requires either more preference pairs or a harder held-out distribution. For the production Conversion Engine, the prompt-only condition is a viable lower-cost alternative in that case.

We report Delta B honestly regardless of outcome. An honest negative result about fine-tuning adds more information to the community than a suppressed one.

**Cost per task.**

Running the trained judge adds latency and compute cost per held-out task vs. the baseline scoring_evaluator (which runs locally with no API cost). The actual cost delta per task is reported in `ablation_results.json` from the trace log timestamps.

**Deployment recommendation.**

The evidence_graph.json records the specific numbers behind the deployment decision. The decision table: deploy the trained judge if Delta A is significant and the cost delta is acceptable; deploy with caveat if Delta A is significant but Delta B is neutral (prompting was sufficient); do not deploy if Delta A is not significant (training did not help beyond the baseline).

---

## Section 5 — What Is Next

Tenacious-Bench v0.1 captures the five B2B sales evaluation gaps identified in the Week 10 probe library. Four failure modes it does not yet capture:

**1. Multi-turn trajectory compounding.** In a five-turn thread with a hostile prospect, individual rubric checks on each turn miss compounding errors — where a plausible turn-3 response becomes a contradiction of a turn-1 commitment. Scoring multi-turn trajectories requires a trajectory-level rubric, not per-turn scoring.

**2. Cultural register differences.** EU prospects (GDPR-aware, formal address conventions, German/French email norms), East African prospects (relationship-first protocols, indirect refusal conventions), and US tech prospects (direct, data-led) require different tone calibration. Tenacious-Bench v0.1 does not stratify by prospect geography. This is a real deployment failure: an email that scores 1.0 on US-calibrated tone rules may read as aggressive or disrespectful to a prospect in Nairobi.

**3. Bench capacity drift.** The bench_summary used for training reflects a snapshot. In production, bench composition changes weekly — engineers finish engagements, new bench members onboard. A judge trained on a static snapshot will increasingly miscalibrate as the real bench diverges. v0.2 needs a dynamic bench_summary update mechanism.

**4. Competitive landscape changes that invalidate gap claims.** The competitor gap briefs in v0.1 are synthetic. Real competitors change their ML hiring posture month to month. A gap brief that was accurate in January may be false by March. The scoring evaluator cannot check this — it trusts the brief. v0.2 needs a brief freshness signal.

We are releasing Tenacious-Bench v0.1 as an open dataset on HuggingFace ([rafiakedir/tenacious-bench-v0.1](https://huggingface.co/datasets/rafiakedir/tenacious-bench-v0.1)) with a public leaderboard. If you are building B2B sales agents, we welcome task contributions, leaderboard entries, and rubric extension proposals. The community issue on τ²-Bench is the right place to coordinate with the broader benchmarking community.

---

*Tenacious-Bench v0.1 is a research artifact. Tenacious Consulting is the workflow domain only — no private client data appears in the dataset. All prospect names and company data are synthetic. The benchmark is released under CC-BY-4.0.*
