# Community Issue — τ²-Bench GitHub

**Repository:** github.com/sierra-research/tau2-bench
**Issue title:** Tenacious-Bench: five B2B sales evaluation gaps in τ²-Bench retail, with open dataset and trained judge

---

## Body

Hi τ²-Bench maintainers,

We are releasing **Tenacious-Bench v0.1**, a 300-task evaluation benchmark and a ORPO-trained judge for B2B outbound sales agents, and want to flag five structural gaps in τ²-Bench retail that motivated this work. We think these gaps are worth addressing in a future τ²-Bench extension, and we are offering our dataset and tooling as a contribution toward that.

---

### The Five Gaps

**Gap 1: Signal Grounding Fidelity (trigger rate: 35%)**

τ²-Bench retail has no mechanism to verify whether an agent's factual claims are backed by a source document. In B2B sales, every claim must resolve to a field in the prospect's `hiring_signal_brief` with confidence ≥ 0.60, or be phrased as a question. Without this check, agents routinely assert "aggressive growth-phase scaling" on prospects with 2 open roles and no recent funding — a claim a CTO will debunk in 30 seconds. Our rubric function `check_grounded_fraction_and_phrasing` scores this deterministically against the source document.

**Gap 2: Bench Commitment Honesty (trigger rate: 5%, SOW-breach risk)**

τ²-Bench retail tasks never put an agent in a position where it must refuse a user request because inventory does not support it. In B2B staffing, the agent must not commit more engineers than are available on the bench. τ²-Bench has no analog to a `bench_summary.json` that constrains what the agent may promise. Our probes (P-009, P-011, P-012) document three variants: committing 6 Go engineers when bench shows 3, committing a team of 40 when total bench is 36, and fabricating Rust capacity entirely. Each triggers a signed SOW the delivery team cannot fulfill.

**Gap 3: ICP Segment Appropriateness (trigger rate: 20%, ~$480K ACV per error)**

τ²-Bench tasks have unambiguous correct actions derivable from policy lookup. Tenacious ICP classification is a probabilistic judgment over six signals with explicit priority ordering. Our probe P-001 proves the cost: an agent pitching growth-scale language (Segment 1) to a company in cost-restructuring mode (Segment 2) after a layoff event loses a $480K ACV opportunity. τ²-Bench cannot grade this because it has no analog to a four-segment classifier with confidence gates.

**Gap 4: Competitor Gap Honesty (trigger rate: 45%, highest in the system)**

This is the most dangerous failure mode. The gap-brief approach — telling a prospect what their competitors are doing that they are not — is the core value proposition of a Conversion Engine. When the brief is wrong, the brand damage is irreversible. Our probe P-031 documents the worst case: the agent asserted "your top competitors have ML platform teams and you don't" to a CTO who had published a 2,000-word post explaining their deliberate choice of managed services. τ²-Bench scores agents on policy adherence, not on whether gap claims are verifiable.

**Gap 5: Tone Preservation Under Adversarial Pressure (trigger rate: 15%)**

τ²-Bench retail includes frustrated users, but agents are scored on policy correctness, not on whether tone stays within a defined brand voice across hostile turns. Our probe P-013 shows the agent drifting into over-apologetic exits ("I apologize for taking your time") after three defensive CTO replies. Probe P-016 documents re-engagement cliché violations ("just wanted to circle back") that are explicitly banned by the style guide. These are deterministically scoreable — no LLM judge needed — but τ²-Bench has no style guide and no multi-turn adversarial persona whose hostility persists without a resolution event.

---

### What We Are Releasing

**Dataset:** [rafiakedir/tenacious-bench-v0.1](https://huggingface.co/datasets/rafiakedir/tenacious-bench-v0.1)
- 300 tasks across five evaluation dimensions
- Three partitions: train (152) / dev (89) / held_out (59)
- Machine-verifiable scoring rubric (`scoring_evaluator.py`)
- Full Gebru + Pushkarna datasheet

**Trained Judge:** [rafiakedir/tenacious-bench-adapter](https://huggingface.co/rafiakedir/tenacious-bench-adapter)
- Qwen2.5-1.5B-Instruct + LoRA, trained with ORPO on Tenacious preference pairs
- Deployed as rejection-sampling layer in production Conversion Engine pipeline
- Ablation results vs Week 10 baseline and prompt-only judge

**Scoring Evaluator:** deterministic Python rubric scorer, usable without an LLM judge

---

### Proposal for a B2B Sales Extension of τ²-Bench

At minimum, a B2B sales extension of τ²-Bench retail would need to include:

1. **Signal-grounding fidelity scoring** — a `hiring_signal_brief` analog and a scoring function that checks claims against it.
2. **Bench commitment honesty** — a `bench_summary.json` analog that constrains agent promises.
3. **ICP segment appropriateness** — a multi-segment classifier with confidence gates and explicit priority ordering.
4. **Style-guide adherence** — a deterministic banned-phrase list and CTA checker.
5. **Multi-turn adversarial personas** — hostile prospects whose position does not soften without a substantive concession.

We would be glad to contribute tasks, rubric functions, or documentation toward this extension. Alternatively, we would welcome a link to Tenacious-Bench in the τ²-Bench related-work section, which would help other B2B sales agent builders discover the dataset.

---

### Relevant Prior Work Referenced

- Kim et al. (2024), Prometheus-2: An Open Source Language Model Specialized in Evaluating Other Language Models
- Hong et al. (2024), ORPO: Monolithic Preference Optimization without Reference Model
- Li et al. (2025), Preference Leakage: A Contamination Problem in LLM-as-a-judge

**Benchmark version:** Tenacious-Bench v0.1 (2026-05-01)
**Contact:** rafiakedir (HuggingFace) / rafia@10academy.org
