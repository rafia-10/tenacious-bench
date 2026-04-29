# Audit Memo — What τ²-Bench Retail Fails to Grade About Tenacious-Specific Sales Behavior

**Author:** Conversion Engine / Week 11 Audit
**Date:** 2026-04-28
**Word count:** ~600
**References:** probe_library.md (34 probes), trace_log.jsonl (tasks 0–4), score_log.json, failure_taxonomy.md

---

## The Core Claim

τ²-Bench retail pass@1 = 0.80 [95% CI 0.62–0.91] on 30 tasks with DeepSeek V3 as backbone
(score_log.json, 2026-04-24T20:00:00Z). That number is real and reproducible. It is also
irrelevant to the question the Tenacious CEO is actually asking: *does this system send
emails a CTO would read rather than report as spam, and does it do so without misrepresenting
Tenacious's capacity or the prospect's situation?*

τ²-Bench retail (trace_log.jsonl, tasks 0–4) tests cooperative airline-service transactions —
flight bookings, cancellations, baggage queries. Every task has a ground-truth system state,
a policy rulebook, and a well-behaved user. None of those conditions hold in Tenacious
outbound sales. The five gaps below are each proven by Week 10 traces and probe evidence,
not theoretical.

---

## Gap 1 — Signal Grounding Fidelity (trigger rate: 35% of prospects)

τ²-Bench has no mechanism to check whether an agent's factual claims are backed by a
source document. In trace task 2, the agent books a flight for a stated passenger count
with no cross-check against a signal brief — because no signal brief exists in the retail
domain. In Tenacious outreach, every factual claim must resolve to a field in
`hiring_signal_brief.json` with confidence ≥ 0.60, or it must be phrased as a question.

Probes **P-005** and **P-006** prove the failure is real: with 3 open engineering roles
(confidence: low), the pre-fix agent asserted "aggressive growth-phase scaling" — a
verifiable falsehood a CTO can debunk in 30 seconds. Probe **P-008** shows the same
pattern with unconfirmed funding amounts stated as fact. The trigger rate across 20
synthetic runs was 35% — the second most common failure in the system.

τ²-Bench cannot grade this because it has no analog to the hiring_signal_brief. A
benchmark that grades Tenacious behavior must score every factual claim against its source.

---

## Gap 2 — Bench Commitment Honesty (trigger rate: 5%, catastrophic when triggered)

τ²-Bench retail never puts the agent in a position where it must refuse a user request
because inventory does not support it. In trace task 0, the agent correctly defers to
policy on refund eligibility — but policy is a rulebook lookup, not a live inventory check
against a capacity document that changes weekly.

Probes **P-009**, **P-011**, and **P-012** document three variants of bench over-commitment:
confirming 6 Go engineers when bench shows 3 (P-009), confirming a team of 40 when total
bench is 36 (P-011), and fabricating Rust capacity entirely (P-012). Each triggers a
signed SOW the Tenacious delivery team cannot fulfill. τ²-Bench scores an agent 1.0 on
task 1 (Raj Sanchez cancellation) for correctly locating reservations — it has no analog
for a bench_summary.json that constrains what the agent may promise.

---

## Gap 3 — ICP Segment Appropriateness (trigger rate: 20%, ~$480K ACV per misclassification)

τ²-Bench retail tasks have unambiguous correct actions derivable from policy lookup.
Tenacious ICP classification is a probabilistic judgment over six signals with explicit
priority ordering — and the ordering matters commercially.

Probe **P-001** (BuildFirst: layoff 60 days ago + $12M Series B 45 days ago) proves the
failure: the pre-fix agent saw fresh funding and pitched enthusiastic Segment 1 scale
language to a company in cost-restructuring mode. Probe **P-003** proves the inverse:
a company with a 45% layoff was pitched Segment 2 "replace higher-cost roles" despite
the disqualifying filter in icp_definition.md. Probe **P-004** shows the leadership
transition window (Segment 3) being missed entirely because the agent anchored on headcount.

τ²-Bench task 3 grades correct baggage-allowance lookup for a Gold member. There is no
analog for a four-segment classifier with a confidence gate and a priority ordering that
carries $480K ACV consequences per error.

---

## Gap 4 — Competitor Gap Honesty (trigger rate: 45% of gap briefs, highest in the system)

This is the most dangerous failure mode and the one τ²-Bench is most structurally
incapable of catching. The gap-brief approach — telling a prospect what their competitors
are doing that they are not — is the core value proposition of the Conversion Engine's
research-first outreach. When the brief is wrong, the brand damage is irreversible.

Probe **P-031** documents the worst case: the agent asserted "your top competitors have
ML platform teams and you don't" to a CTO who had published a 2,000-word post explaining
their deliberate choice of managed services. Probe **P-034** shows the symmetric failure:
asserting "no public AI signal" to a company actively blogging about their ML stack.
Probe **P-033** shows a statistical dishonesty: claiming "top quartile of your sector"
when only 2 of 10 peers (20%) show the signal.

The trigger rate was 45% across 20 synthetic runs — the single highest in the system.
τ²-Bench task 4 (Sophia Silva flight cancellation) scores the agent on policy adherence,
not on whether its factual assertions about the user's situation are verifiable. A
Tenacious benchmark must score gap assertions against the prospect's own public record.

---

## Gap 5 — Tone Preservation Under Adversarial Pressure (trigger rate: 15%)

τ²-Bench retail tasks 0 through 4 include frustrated users (task 4: Sophia Silva's
"million-dollar client meeting" complaint), but the agent is scored on whether it follows
policy correctly, not on whether its tone stays within a defined brand voice across
multiple hostile turns.

Probe **P-013** (three defensive CTO replies) shows the agent drifting from Tenacious's
Direct + Grounded + Professional markers into over-apologetic exits: "I completely
understand, and you're right that we may not be the best fit — I apologize for taking
your time." Probe **P-015** shows technical-depth drift: by turn 5 of a technical thread,
the agent is recommending specific LangChain versions, outside Tenacious's sales lane.
Probe **P-016** documents re-engagement cliché violations: "just wanted to circle back"
is explicitly prohibited by style_guide.md.

Both P-015 and P-016 are currently ❌ Open — the tone_checker does not yet catch
technical-depth drift or re-engagement clichés programmatically. This is a provable
Week 10 weakness with a measurable trigger rate. τ²-Bench cannot surface it because it
has no style guide and no multi-turn adversarial persona whose hostility persists across
turns without a resolution event.

---

## What Tenacious-Bench v0.1 Must Grade That τ²-Bench Does Not

| Dimension | τ²-Bench grades it? | Proven Week 10 gap |
|---|---|---|
| Signal-grounding fidelity | ✗ | P-005, P-006, P-008 (35% trigger) |
| Bench commitment honesty | ✗ | P-009, P-011, P-012 (SOW breach risk) |
| ICP segment appropriateness | ✗ | P-001, P-003, P-004 ($480K ACV per error) |
| Competitor gap honesty | ✗ | P-031, P-033, P-034 (45% trigger, irreversible) |
| Tone preservation under pressure | ✗ | P-013, P-015, P-016 (15% trigger, open) |
| Multi-thread isolation | ✗ | P-017, P-018 (architecture-level, solved) |
| Scheduling cross-timezone | ✗ | P-025, P-026, P-027 (40% of booking interactions) |

The τ²-Bench retail domain is the right benchmark for evaluating whether an agent can
navigate policy-constrained customer service transactions with cooperative users. It is
the wrong benchmark for evaluating whether an outbound B2B sales agent tells the truth,
matches its pitch to the buyer's actual situation, respects its own capacity constraints,
and maintains a defined brand voice under adversarial pressure. Tenacious-Bench v0.1
is built to grade exactly those five things, using machine-verifiable rubrics derived
from the probe library above and the Week 10 scoring infrastructure already in production.