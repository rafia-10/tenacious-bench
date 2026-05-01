---
license: cc-by-4.0
language:
- en
tags:
- judge
- preference-learning
- b2b-sales
- orpo
- tenacious-bench
- qwen2.5
base_model: unsloth/Qwen2.5-1.5B-Instruct
---

# Tenacious-Bench Judge — LoRA Adapter

**Backbone:** `unsloth/Qwen2.5-1.5B-Instruct` (pinned: transformers==4.47.1, unsloth via git main 2026-05-01)
**Adapter type:** LoRA (PEFT==0.14.0)
**Training algorithm:** ORPO (TRL==0.12.2)
**Dataset:** [rafiakedir/tenacious-bench-v0.1](https://huggingface.co/datasets/rafiakedir/tenacious-bench-v0.1)

---

## Model Description

This adapter turns Qwen2.5-1.5B-Instruct into a rubric-aware judge for B2B outbound sales emails written by Tenacious Consulting's Conversion Engine. It is deployed as a **rejection-sampling gate** in the Conversion Engine pipeline: the generator (DeepSeek V3.2) produces a candidate email, the judge scores it on five Tenacious-specific rubric dimensions, and outputs scoring below threshold are rejected and regenerated.

**Target failure mode:** The Week 10 Conversion Engine produces correct outputs most of the time but cannot detect when its own output violates the grounded-honesty constraint. Signal over-claiming (35% trigger rate) and competitor gap fabrication (45% trigger rate) are the dominant failure modes. This judge is trained to detect both.

**Rubric dimensions scored:**
1. **Signal Grounding Fidelity** — claims must be supported by the `hiring_signal_brief` or phrased as questions
2. **Bench Commitment Honesty** — staffing commitments must not exceed `bench_summary` capacity
3. **ICP Segment Appropriateness** — email language must match the correct ICP segment (Segment 1/2/3/ABSTAIN)
4. **Competitor Gap Honesty** — gap claims must not be fabricated beyond the `competitor_gap_brief`
5. **Tone Preservation** — no re-engagement clichés, no over-apologetic exits, calendar CTA required

---

## Training Data

**Dataset:** `rafiakedir/tenacious-bench-v0.1` — training partition (152 tasks)

**Preference pair construction:**
- 152 train tasks scored with `scoring_evaluator.py`
- 111 failing tasks: existing `candidate_output` used as rejected; DeepSeek V3.2 generated corrected chosen output
- 41 passing tasks: existing `candidate_output` used as chosen; DeepSeek V3.2 generated a failing rejected output
- Filter criteria: chosen scores above rubric threshold, rejected scores below, TF-IDF cosine similarity < 0.92
- Final pair count: see `training_data/preference_pairs.jsonl` line count

**Model rotation (preference leakage prevention per Li et al., 2025):**
- Generator: DeepSeek V3.2 (deepseek/deepseek-chat-v3-0324 via OpenRouter)
- Machine scorer: `scoring_evaluator.py` (deterministic, no LLM bias)
- Spot-check judge: Claude Sonnet 4.6 (Anthropic family — different from DeepSeek)

---

## Training Framework

| Hyperparameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| LoRA dropout | 0.05 |
| Learning rate | 8e-6 |
| Batch size (per device) | 2 |
| Gradient accumulation | 4 (effective batch 8) |
| Epochs | 3 |
| Warmup ratio | 0.1 |
| LR scheduler | cosine |
| ORPO beta | 0.1 |
| Max sequence length | 1024 |
| Precision | fp16 (T4) |
| Seed | 42 |

**Why ORPO over DPO:** ORPO requires no reference model, reducing peak VRAM by ~40% on T4. Reference-free formulation is appropriate for a judge component where no well-calibrated domain reference policy exists. See `methodology_rationale.md`.

**Why this backbone:** Per Prometheus-2 (Kim et al., 2024), small models fine-tuned on domain-specific preference pairs can match GPT-4-class evaluation on narrow rubrics. Qwen2.5-1.5B is below the 7B threshold demonstrated in that paper — T4 compute constraint. Risk is documented in Limitations.

---

## Intended Use

**In scope:**
- Rejection-sampling judge for the Tenacious Conversion Engine pipeline
- Rubric-based scoring of B2B outbound sales emails on the five Tenacious dimensions above
- Research reference for small-judge fine-tuning on domain-specific preference data

**Out of scope:**
- General-purpose text quality evaluation
- Non-sales domains (customer service, technical documentation, etc.)
- Non-Tenacious sales styles (different tone rules, different ICP frameworks)
- Evaluation of outputs not related to B2B outbound sales

---

## Evaluation Results

See `ablations/ablation_results.json` for full numbers with confidence intervals and p-values from the paired bootstrap test (10,000 iterations, seed 42).

**Delta A (trained vs. Week 10 baseline):** see `ablation_results.json/delta_a`
**Delta B (trained vs. prompt-only Qwen3-30B judge):** see `ablation_results.json/delta_b`

If Delta B is not statistically significant, this indicates that careful prompt engineering of the same Qwen backbone is a viable lower-cost alternative for this rubric at v0.1 training data scale.

---

## Limitations

1. **Synthetic training data.** All preference pairs are generated from synthetic prospect briefs and DeepSeek-generated emails. The judge may not generalize to real prospect data with unusual phrasing, industry jargon not in the training distribution, or edge cases not covered by the five rubric dimensions.

2. **Small backbone.** At 1.5B parameters, the model may underfit on tasks requiring simultaneous attention to multiple rubric dimensions (e.g., a task where signal grounding and tone preservation must both pass).

3. **Tenacious-specific rubric only.** The judge encodes five rubric dimensions specific to the Tenacious Conversion Engine. It is not a general sales quality judge. Scores produced on emails from other sales systems are not meaningful.

4. **No multi-turn trajectory scoring.** The judge scores individual outputs, not conversation trajectories. Multi-turn compounding errors are not detected.

5. **Static bench_summary.** The judge was trained on preference pairs with snapshot bench capacities. In production, bench composition changes weekly — the judge's calibration for bench commitment honesty will drift as the real bench diverges.

---

## Environmental Impact

**Estimated GPU compute:** ~60–90 minutes on a single T4 GPU (3 epochs, ~70–100 preference pairs)
**CO2 equivalent:** ~0.1 kg CO2e (estimated via mlco2.ai calculator: T4 at 70W TDP × 90 minutes × US grid average of ~0.42 kg CO2/kWh ÷ 1000)
**Compute provided by:** Google Colab (free tier)

---

## Citation

```bibtex
@misc{tenacious-bench-adapter-2026,
  title        = {Tenacious-Bench Judge: LoRA Adapter for B2B Sales Evaluation},
  author       = {rafiakedir},
  year         = {2026},
  howpublished = {HuggingFace Model Hub},
  url          = {https://huggingface.co/rafiakedir/tenacious-bench-adapter}
}

@misc{tenacious-bench-v01-2026,
  title        = {Tenacious-Bench v0.1: B2B Sales Evaluation Benchmark},
  author       = {rafiakedir},
  year         = {2026},
  howpublished = {HuggingFace Datasets Hub},
  url          = {https://huggingface.co/datasets/rafiakedir/tenacious-bench-v0.1}
}
```
