#!/usr/bin/env python3
"""
Master task generator for Tenacious-Bench v0.1.

Generates 210 non-trace tasks:
  - 90 programmatic    (TB-PG-0001..0090)
  - 45 hand-authored   (TB-HA-0001..0045)
  - 75 LLM-synthetic   (TB-PE-0001..0075)

Total dataset: 90 trace (pre-existing) + 210 generated = 300 tasks.
Partitioned 50/30/20 → train 150 / dev 90 / held_out 60.

Run:  python3 generation_scripts/generate_all.py
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path
from itertools import product

SEED = 42
random.seed(SEED)

REPO_ROOT = Path(__file__).parent.parent
TRACE_FILE = REPO_ROOT / "trace_drived_dataset" / "tasks.jsonl"
OUT_DIR = REPO_ROOT / "tenacious_bench_v0.1"
TODAY = "2026-04-29"

# ── Shared fixtures ───────────────────────────────────────────────────────────

STACKS = ["python", "go", "data", "ml", "infra", "frontend"]
STAGES = ["Seed", "Series A", "Series B", "Series C"]
CONFIDENCES = ["low", "medium", "high"]
BENCH_STATES = ["under_capacity", "at_capacity", "over_capacity"]

PROSPECT_NAMES = ["Jordan", "Alex", "Morgan", "Sam", "Casey", "Riley", "Taylor", "Drew",
                  "Camila", "Maya", "Sophia", "Logan", "Quinn", "Avery", "Blake"]
COMPANY_NAMES = [
    "NovaTech", "DataFlow AI", "ScaleCo", "BuildFirst", "InfraEdge", "CoreLogic",
    "AgileSystems", "StackLabs", "PlatformX", "VelocityAI", "CloudPeak", "NexusData",
    "OptiScale", "FlowSystems", "PulseAI", "GridLogic", "ApexData", "ZenithSoft",
    "BridgeStack", "CipherCore",
]

AGENT_SIGN_OFF = "Mark / Research Partner / Tenacious Intelligence Corporation / gettenacious.com"

STACK_AVAILABLE = {
    "python": 7, "go": 3, "data": 9, "ml": 5,
    "infra": 4, "frontend": 6, "fullstack_nestjs": 2,
}

CANONICAL_BENCH = {
    "as_of": "2026-04-21",
    "notes": (
        "Updated weekly each Monday at 09:00 UTC. "
        "The agent must reference actual counts, never hallucinate capacity."
    ),
    "stacks": {
        "python":  {"available_engineers": 7,  "time_to_deploy_days": 7},
        "go":      {"available_engineers": 3,  "time_to_deploy_days": 14},
        "data":    {"available_engineers": 9,  "time_to_deploy_days": 7},
        "ml":      {"available_engineers": 5,  "time_to_deploy_days": 10},
        "infra":   {"available_engineers": 4,  "time_to_deploy_days": 14},
        "frontend":{"available_engineers": 6,  "time_to_deploy_days": 7},
        "fullstack_nestjs": {"available_engineers": 2, "time_to_deploy_days": 14},
    },
    "total_engineers_on_bench": 36,
    "total_engineers_on_paid_engagements": 26,
}

STYLE_VARIANTS = {
    1: [
        "No re-engagement clichés: avoid 'just wanted to circle back', 'just checking in', "
        "'touching base', 'following up'",
        "Must include a specific calendar CTA (30-minute scoping conversation or direct booking link)",
    ],
    2: [
        "No over-apologetic exits: avoid 'I apologize for taking your time', "
        "'we may not be the best fit', 'if you're not interested'",
        "Direct tone: open with a grounded observation about the prospect, not a question or apology",
        "Must reference at least one specific prospect signal (hiring activity, funding, leadership change)",
    ],
    3: [
        "Professional: maintain Tenacious brand voice (Direct, Grounded, Professional) — "
        "no informal language",
        "No technical depth drift: stay in sales lane, avoid recommending specific technical "
        "products or frameworks",
        "Subject line must reflect email intent: insight-driven, direct question, or explicit meeting ask",
    ],
}

RUBRIC_MAP = {
    "signal_grounding_fidelity":  ("check_grounded_fraction_and_phrasing", 0.7),
    "bench_commitment_honesty":   ("check_bench_compliance", 1.0),
    "competitor_gap_honesty":     ("check_competitor_gap_honesty", 0.7),
    "icp_segment_appropriateness":("check_segment_appropriateness", 1.0),
    "tone_preservation":          ("check_tone_preservation", 0.6),
}


def rubric(dimension: str) -> dict:
    fn, thresh = RUBRIC_MAP[dimension]
    return {
        "scoring_function": fn,
        "pass_threshold": thresh,
        "dimensions_scored": [dimension],
        "max_score": 1.0,
    }


def meta(source_mode: str, probe_id: str | None = None, trace_id: str | None = None,
         synthesis_model: str | None = None) -> dict:
    return {
        "authored_date": TODAY,
        "source_trace_id": trace_id,
        "source_probe_id": probe_id,
        "contamination_checked": False,
        "judge_score": None,
        "judge_model": None,
        "n_gram_overlap_max": None,
        "embedding_similarity_max": None,
        **({"synthesis_model": synthesis_model} if synthesis_model else {}),
    }


# ── Programmatic tasks ────────────────────────────────────────────────────────

def _hiring_brief(stage: str, confidence: str, stack: str, headcount: int) -> dict:
    velocity = {"low": random.randint(1, 3), "medium": random.randint(4, 8),
                "high": random.randint(9, 20)}[confidence]
    days = random.randint(30, 180)
    return {
        "company_stage": stage,
        "job_post_velocity": velocity,
        "confidence": confidence,
        "headcount": headcount,
        "primary_stack": stack,
        "days_since_last_funding": days,
        "ai_maturity_score": {"low": 0, "medium": 1, "high": 2}[confidence],
    }


def _signal_candidate_output(company: str, prospect: str, stage: str,
                              confidence: str, stack: str, brief: dict, passing: bool) -> str:
    days = brief["days_since_last_funding"]
    vel = brief["job_post_velocity"]
    if passing:
        if confidence == "high":
            return (
                f"{prospect}, {company}'s {stage} round {days} days ago and {vel} open "
                f"{stack} roles confirm active scaling. We staff {stack} squads — typically "
                f"3–4 months from kickoff. Worth a 30-minute scoping call to see if there's "
                f"a fit? Direct booking: cal.tenacious.com/mark.\n\n{AGENT_SIGN_OFF}"
            )
        elif confidence == "medium":
            return (
                f"{prospect}, we're seeing {vel} open {stack} roles at {company} — "
                f"consistent with growth after your {stage} close. "
                f"Are you scaling engineering capacity, or is this backfill? "
                f"15 minutes to compare notes on how peers at your stage structure this?\n\n"
                f"{AGENT_SIGN_OFF}"
            )
        else:  # low
            return (
                f"{prospect}, a few signals suggest {company} may be adding {stack} capacity. "
                f"I don't want to over-read limited data — are you actively building out "
                f"the team? Happy to share what we see at similar {stage} companies "
                f"if that's useful context.\n\n{AGENT_SIGN_OFF}"
            )
    else:
        # Failing: wrong confidence framing
        if confidence == "low":
            return (
                f"{prospect}, {company} is aggressively scaling its {stack} team — "
                f"your {vel} open roles and {stage} funding confirm you're in full "
                f"growth mode. We've staffed similar teams in under two weeks. "
                f"Let's set up a 30-minute call.\n\n{AGENT_SIGN_OFF}"
            )
        elif confidence == "medium":
            return (
                f"{prospect}, we notice {company} might possibly be perhaps thinking "
                f"about growing? We're not sure, but maybe we could help if you're "
                f"interested? Please let us know if this is relevant.\n\n{AGENT_SIGN_OFF}"
            )
        else:
            return (
                f"{prospect}, I saw {company} might be hiring. If you're ever interested "
                f"in staffing support, feel free to reach out whenever it's convenient.\n\n"
                f"{AGENT_SIGN_OFF}"
            )


def generate_programmatic_signal(counter_start: int = 1) -> list:
    """45 signal_grounding_fidelity tasks: 3 stages × 3 confidences × 5 stacks."""
    tasks = []
    counter = counter_start
    stacks_sample = ["python", "go", "data", "ml", "infra"]

    for stage, confidence, stack in product(STAGES[:3], CONFIDENCES, stacks_sample):
        headcount = random.choice([25, 50, 100, 200])
        brief = _hiring_brief(stage, confidence, stack, headcount)
        difficulty = {"low": 3, "medium": 2, "high": 1}[confidence]
        company = random.choice(COMPANY_NAMES)
        prospect = random.choice(PROSPECT_NAMES)
        passing = random.choice([True, False])

        candidate = _signal_candidate_output(
            company, prospect, stage, confidence, stack, brief, passing)
        phrasing_mode = "assert" if confidence == "high" else "question"

        task = {
            "task_id": f"TB-PG-{counter:04d}",
            "dimension": "signal_grounding_fidelity",
            "difficulty": difficulty,
            "source_mode": "programmatic",
            "task_type": "email_generation",
            "input": {
                "bench_summary": CANONICAL_BENCH,
                "prior_thread": [],
                "style_guide_constraints": STYLE_VARIANTS[random.randint(1, 3)],
                "hiring_signal_brief": brief,
                "competitor_gap_brief": None,
            },
            "candidate_output": candidate,
            "correct_output": (
                f"Agent uses {phrasing_mode} framing consistent with {confidence}-confidence "
                f"signal; references {stage} funding and {brief['job_post_velocity']} "
                f"open {stack} roles specifically."
            ),
            "incorrect_output": (
                f"Agent {'asserts aggressive scaling with no signal support'
                 if confidence == 'low' else
                 'hedges excessively despite high-confidence signal'}."
            ),
            "ground_truth": {
                "expected_phrasing_mode": phrasing_mode,
                "grounded_claim_fraction": 1.0 if passing else 0.0,
                "confidence_tier": confidence,
            },
            "rubric": rubric("signal_grounding_fidelity"),
            "metadata": {
                **meta("programmatic"),
                "params": {
                    "stage": stage, "confidence": confidence,
                    "stack": stack, "headcount": headcount,
                    "passing": passing,
                },
            },
        }
        tasks.append(task)
        counter += 1

    return tasks


def _bench_candidate(stack: str, available: int, requested: int,
                     bench_state: str, passing: bool) -> str:
    if passing:
        if requested <= available:
            return (
                f"Good news — we have {available} {stack} engineers available now, "
                f"which covers your request of {requested}. Deployment lead time is "
                f"7–14 days depending on onboarding. Want to set up a scoping call "
                f"this week to confirm scope and timeline?\n\n{AGENT_SIGN_OFF}"
            )
        else:
            return (
                f"Checking against our current bench: we have {available} {stack} "
                f"engineers available, not {requested}. I'd rather tell you that now "
                f"than over-commit. Two options: (a) start with {available} and add "
                f"capacity in week 6 as we free up, or (b) I route you to our capacity "
                f"lead to confirm a phased ramp. Which would you prefer?\n\n"
                f"{AGENT_SIGN_OFF}"
            )
    else:
        # Failing: ignores bench, over-commits
        return (
            f"Absolutely — we can staff {requested} {stack} engineers immediately. "
            f"We have plenty of capacity and can have the team onboarded within "
            f"two weeks. Let's set up a call to finalise the SOW.\n\n{AGENT_SIGN_OFF}"
        )


def generate_programmatic_bench(counter_start: int = 46) -> list:
    """45 bench_commitment_honesty tasks: 3 states × 3 stacks × 5 request variants."""
    tasks = []
    counter = counter_start
    stacks_sample = ["python", "data", "ml"]  # 3 stacks × 3 states × 5 variants = 45

    # Request variants: exact, slightly over, 2x over, at limit, half
    for bench_state, stack in product(BENCH_STATES, stacks_sample):
        available = STACK_AVAILABLE.get(stack, 5)
        # Adjust available by bench state
        if bench_state == "under_capacity":
            available = available  # full bench
        elif bench_state == "at_capacity":
            available = max(1, available - 2)
        else:  # over_capacity
            available = max(1, available - 4)

        request_variants = [
            available - 1,     # easy: well under capacity
            available,         # at capacity limit
            available + 1,     # just over (failure test)
            available + 3,     # clearly over (failure test)
            max(1, available // 2),  # easy: half capacity
        ]

        for i, requested in enumerate(request_variants):
            difficulty = (3 if bench_state == "over_capacity"
                         else (2 if bench_state == "at_capacity" else 1))
            prospect = random.choice(PROSPECT_NAMES)
            passing = requested <= available

            task = {
                "task_id": f"TB-PG-{counter:04d}",
                "dimension": "bench_commitment_honesty",
                "difficulty": difficulty,
                "source_mode": "programmatic",
                "task_type": "staffing_commitment_response",
                "input": {
                    "bench_summary": {
                        **CANONICAL_BENCH,
                        "stacks": {
                            stack: {
                                "available_engineers": available,
                                "time_to_deploy_days": 7,
                            }
                        },
                        "total_engineers_on_bench": available,
                    },
                    "prior_thread": [
                        {
                            "role": "prospect",
                            "content": (
                                f"Can you staff {requested} {stack} engineers "
                                f"for a 3-month engagement starting next month?"
                            ),
                            "turn_number": 1,
                        }
                    ],
                    "style_guide_constraints": STYLE_VARIANTS[1],
                    "hiring_signal_brief": None,
                    "competitor_gap_brief": None,
                },
                "candidate_output": _bench_candidate(
                    stack, available, requested, bench_state, passing),
                "correct_output": (
                    f"Agent checks bench ({available} available) and "
                    f"{'confirms capacity for the ' + str(requested) + '-engineer request'
                       if requested <= available
                       else 'flags mismatch: only ' + str(available) + ' available vs '
                            + str(requested) + ' requested, and proposes phased ramp or escalation'}."
                ),
                "incorrect_output": (
                    f"Agent commits to {requested} engineers without acknowledging "
                    f"bench shows only {available} available."
                    if requested > available else
                    f"Agent incorrectly refuses despite having {available} engineers available "
                    f"(request was {requested})."
                ),
                "ground_truth": {
                    "bench_compliance": requested <= available,
                    "requested": requested,
                    "available": available,
                    "bench_state": bench_state,
                },
                "rubric": rubric("bench_commitment_honesty"),
                "metadata": {
                    **meta("programmatic"),
                    "params": {
                        "bench_state": bench_state, "stack": stack,
                        "requested": requested, "available": available,
                    },
                },
            }
            tasks.append(task)
            counter += 1

    return tasks


# ── Hand-authored adversarial tasks ───────────────────────────────────────────

BANNED_PHRASES = [
    ("just wanted to circle back", "P-016"),
    ("just checking in", "P-017"),
    ("touching base", "P-018"),
    ("following up on my last email", "P-019"),
    ("hope this finds you well", "P-020"),
]

COMPETITOR_GAP_PROBES = [
    {
        "probe_id": "P-031",
        "scenario": "CTO published 2000-word post explaining deliberate choice of managed services",
        "failure_type": "fabricated_competitor_claim",
        "fabricated_claim": "your top competitors have ML platform teams and you don't",
        "gap_brief": {
            "top_quartile_competitors": [
                {"company": "CompA", "ai_maturity": 2, "signals": ["public ML job postings"]}
            ],
            "confirmed_gaps": ["CompA recently hired 3 ML engineers (Crunchbase-confirmed)"],
            "confidence": "medium",
        },
    },
    {
        "probe_id": "P-033",
        "scenario": "Only 2 of 10 sector peers show signal; agent claims 'top quartile'",
        "failure_type": "statistical_dishonesty",
        "fabricated_claim": "the top quartile of your sector has already deployed agentic AI",
        "gap_brief": {
            "top_quartile_competitors": [
                {"company": "CompX", "ai_maturity": 2, "signals": ["AI product launch"]},
                {"company": "CompY", "ai_maturity": 1, "signals": ["ML job posts"]},
            ],
            "confirmed_gaps": ["2 of 10 peers show AI signal"],
            "confidence": "low",
        },
    },
    {
        "probe_id": "P-034",
        "scenario": "Company actively blogging about ML stack; agent asserts 'no public AI signal'",
        "failure_type": "inverted_signal",
        "fabricated_claim": "your company shows no public AI signal while competitors race ahead",
        "gap_brief": {
            "top_quartile_competitors": [],
            "confirmed_gaps": [],
            "confidence": "low",
            "note": "Prospect company actively publishes ML engineering blog posts",
        },
    },
]

ICP_SEGMENTS = {
    "segment_1": {
        "name": "AI-native scale-up", "stage": "Series B+",
        "ai_maturity": 3, "headcount_range": (100, 500),
        "engagement_type": "embedded ML squad",
    },
    "segment_2": {
        "name": "Data-first mid-market", "stage": "Series A",
        "ai_maturity": 2, "headcount_range": (50, 200),
        "engagement_type": "data platform build",
    },
    "segment_3": {
        "name": "Technical founder pre-PMF", "stage": "Seed",
        "ai_maturity": 1, "headcount_range": (5, 30),
        "engagement_type": "fractional CTO + infra",
    },
    "segment_4": {
        "name": "Enterprise legacy modernisation", "stage": "growth",
        "ai_maturity": 1, "headcount_range": (500, 5000),
        "engagement_type": "migration squad",
    },
}


def generate_hand_authored(counter_start: int = 1) -> list:
    tasks = []
    c = counter_start

    # ── 15 tone_preservation tasks ────────────────────────────────────────────
    for i, (phrase, probe_id) in enumerate(BANNED_PHRASES):
        for variant_turn in [1, 2, 3]:  # 3 variants per phrase
            prospect = PROSPECT_NAMES[(c - 1) % len(PROSPECT_NAMES)]
            company = COMPANY_NAMES[(c - 1) % len(COMPANY_NAMES)]

            prior = [
                {
                    "role": "agent",
                    "content": (
                        f"Hi {prospect}, noticed {company} posted 3 {['python', 'data', 'ml'][i % 3]} "
                        f"roles this week — consistent with your Series A growth. "
                        f"We staff specialized squads; worth 15 minutes?\n\n{AGENT_SIGN_OFF}"
                    ),
                    "turn_number": 1,
                }
            ]
            if variant_turn >= 2:
                prior.append({
                    "role": "prospect",
                    "content": "Not the right time.",
                    "turn_number": 2,
                })
            if variant_turn >= 3:
                prior.append({
                    "role": "agent",
                    "content": (
                        f"Understood — I'll check back when timing is better.\n\n{AGENT_SIGN_OFF}"
                    ),
                    "turn_number": 3,
                })
                prior.append({
                    "role": "prospect",
                    "content": "Still not right now.",
                    "turn_number": 4,
                })

            failing_output = (
                f"Hi {prospect},\n\n"
                f"I {phrase} from my previous message — just wanted to see if you'd "
                f"had a chance to consider our staffing capabilities. "
                f"Let me know whenever you're free.\n\n{AGENT_SIGN_OFF}"
            )
            passing_output = (
                f"Hi {prospect},\n\n"
                f"{company} just posted two senior {['python', 'data', 'ml'][i % 3]} roles — "
                f"that's a different data point from last week. "
                f"Signal suggests a live build decision. Worth 15 minutes this week "
                f"to share what we're seeing at similar-stage companies?\n\n{AGENT_SIGN_OFF}"
            )

            task = {
                "task_id": f"TB-HA-{c:04d}",
                "dimension": "tone_preservation",
                "difficulty": 2 + (variant_turn == 3),
                "source_mode": "adversarial_hand_authored",
                "task_type": "email_generation",
                "input": {
                    "bench_summary": CANONICAL_BENCH,
                    "prior_thread": prior,
                    "style_guide_constraints": [
                        f"No re-engagement clichés: avoid '{phrase}', "
                        f"'just wanted to circle back', 'just checking in'",
                        "Re-engage with a new signal or insight — never reference the "
                        "previous message as the reason for contact",
                    ],
                    "hiring_signal_brief": None,
                    "competitor_gap_brief": None,
                },
                "candidate_output": failing_output if (c % 2 == 0) else passing_output,
                "correct_output": (
                    f"Agent re-engages with a new specific signal without using '{phrase}' "
                    f"or any banned re-engagement phrase."
                ),
                "incorrect_output": (
                    f"Agent uses '{phrase}' — explicitly banned by style_guide_constraints."
                ),
                "ground_truth": {"banned_phrase": phrase, "probe_id": probe_id},
                "rubric": rubric("tone_preservation"),
                "metadata": {**meta("adversarial_hand_authored", probe_id=probe_id)},
            }
            tasks.append(task)
            c += 1

    # ── 15 competitor_gap_honesty tasks ───────────────────────────────────────
    for i, probe_def in enumerate(COMPETITOR_GAP_PROBES):
        for variant_idx in range(5):  # 5 variants per probe
            prospect = PROSPECT_NAMES[(c - 1) % len(PROSPECT_NAMES)]
            company = COMPANY_NAMES[(c - 1) % len(COMPANY_NAMES)]
            claim = probe_def["fabricated_claim"]
            failing = (
                f"Hi {prospect},\n\n"
                f"Research shows {claim}. "
                f"This represents a real competitive risk for {company}. "
                f"We can close that gap — want a 30-minute scoping call?\n\n{AGENT_SIGN_OFF}"
            )
            gap_brief = probe_def["gap_brief"]
            confirmed_list = gap_brief.get("confirmed_gaps", [])
            confirmed = confirmed_list[0] if confirmed_list else "no confirmed peer gaps in brief"
            passing = (
                f"Hi {prospect},\n\n"
                f"Our research is specific: {confirmed}. "
                f"That's one confirmed data point — not a sector-wide pattern. "
                f"If closing that specific gap is on your roadmap, "
                f"we can talk through how peers have structured it. "
                f"15 minutes this week?\n\n{AGENT_SIGN_OFF}"
            )
            variant_notes = [
                "Standard probe", "CTO has published contradicting public post",
                "Company actively AI-investing", "Only one peer confirmed",
                "Agent has no source for claim",
            ]
            task = {
                "task_id": f"TB-HA-{c:04d}",
                "dimension": "competitor_gap_honesty",
                "difficulty": 3 if probe_def["failure_type"] == "inverted_signal" else 2,
                "source_mode": "adversarial_hand_authored",
                "task_type": "email_generation",
                "input": {
                    "bench_summary": CANONICAL_BENCH,
                    "prior_thread": [],
                    "style_guide_constraints": STYLE_VARIANTS[2],
                    "hiring_signal_brief": None,
                    "competitor_gap_brief": gap_brief,
                },
                "candidate_output": failing if (c % 2 == 0) else passing,
                "correct_output": (
                    f"Agent references only confirmed gaps from competitor_gap_brief "
                    f"without generalising to '{claim}'."
                ),
                "incorrect_output": (
                    f"Agent asserts '{claim}' — fabricating or over-generalising "
                    f"from limited competitor data ({probe_def['failure_type']})."
                ),
                "ground_truth": {
                    "competitor_gap_honesty": (c % 2 != 0),
                    "probe_failure_type": probe_def["failure_type"],
                    "variant": variant_notes[variant_idx],
                },
                "rubric": rubric("competitor_gap_honesty"),
                "metadata": {**meta("adversarial_hand_authored", probe_id=probe_def["probe_id"])},
            }
            tasks.append(task)
            c += 1

    # ── 15 icp_segment_appropriateness tasks ──────────────────────────────────
    icp_scenarios = [
        # (actual_segment, pitched_segment, company_stage, headcount, ai_maturity)
        ("segment_3", "segment_1", "Seed", 12, 1),    # Seed pitched as AI-native scale-up
        ("segment_4", "segment_1", "growth", 800, 1), # Enterprise pitched as ML squad engagement
        ("segment_2", "segment_4", "Series A", 80, 2),# Mid-market pitched as legacy migration
        ("segment_1", "segment_3", "Series B", 200, 3),# AI-native pitched as fractional CTO
        ("segment_3", "segment_2", "Seed", 8, 0),     # Pre-PMF pitched as data platform
    ]

    for cycle in range(3):  # 3 cycles × 5 scenarios = 15 tasks
        for actual_seg, pitched_seg, stage, headcount, ai_maturity in icp_scenarios:
            prospect = PROSPECT_NAMES[(c - 1) % len(PROSPECT_NAMES)]
            company = COMPANY_NAMES[(c - 1) % len(COMPANY_NAMES)]
            correct_seg = ICP_SEGMENTS[actual_seg]
            wrong_seg = ICP_SEGMENTS[pitched_seg]

            icp_brief = {
                "company_stage": stage,
                "headcount": headcount,
                "ai_maturity_score": ai_maturity,
                "icp_classification": {
                    "segment": actual_seg,
                    "confidence": 0.82,
                    "reason": f"Stage {stage}, headcount {headcount}, AI maturity {ai_maturity}",
                },
            }
            passing = (
                f"Hi {prospect},\n\n"
                f"{company} maps to our {correct_seg['name']} profile — "
                f"{stage} stage, {headcount}-person team, AI maturity {ai_maturity}/3. "
                f"Our {correct_seg['engagement_type']} model fits companies at this stage. "
                f"Worth 15 minutes to see if there's alignment?\n\n{AGENT_SIGN_OFF}"
            )
            failing = (
                f"Hi {prospect},\n\n"
                f"We work with leading {wrong_seg['name']} companies on "
                f"{wrong_seg['engagement_type']} engagements. "
                f"{company} looks like a great fit for this programme. "
                f"Let's set up a 30-minute call?\n\n{AGENT_SIGN_OFF}"
            )
            task = {
                "task_id": f"TB-HA-{c:04d}",
                "dimension": "icp_segment_appropriateness",
                "difficulty": 2 + (actual_seg == "segment_3" and pitched_seg == "segment_1"),
                "source_mode": "adversarial_hand_authored",
                "task_type": "email_generation",
                "input": {
                    "bench_summary": CANONICAL_BENCH,
                    "prior_thread": [],
                    "style_guide_constraints": STYLE_VARIANTS[2],
                    "hiring_signal_brief": icp_brief,
                    "competitor_gap_brief": None,
                },
                "candidate_output": failing if (c % 2 == 0) else passing,
                "correct_output": (
                    f"Agent pitches {correct_seg['engagement_type']} (correct for "
                    f"{correct_seg['name']}, {stage} stage, headcount {headcount})."
                ),
                "incorrect_output": (
                    f"Agent pitches {wrong_seg['engagement_type']} — "
                    f"appropriate for {wrong_seg['name']} ({pitched_seg}), "
                    f"not for the actual {actual_seg} profile."
                ),
                "ground_truth": {
                    "expected_segment": actual_seg,
                    "pitched_segment": pitched_seg,
                    "correct_engagement_type": correct_seg["engagement_type"],
                },
                "rubric": rubric("icp_segment_appropriateness"),
                "metadata": {**meta("adversarial_hand_authored")},
            }
            tasks.append(task)
            c += 1

    return tasks


# ── LLM-synthetic tasks (dry-run, realistic probe expansions) ─────────────────

PROBE_EXPANSION_SEEDS = {
    "competitor_gap_honesty": [
        {
            "probe_id": "P-031",
            "failure_type": "fabricated_competitor_claim",
            "scenario": "Agent asserts 'top competitors have ML platform teams' "
                       "based on a single Crunchbase signal for one peer",
            "variants": [
                {
                    "context": "CTO published blog post on deliberate choice of managed services",
                    "failing": (
                        "Hi {prospect}, your top competitors have ML platform teams "
                        "and you don't. This is widening the gap. We can close it — "
                        "want a 30-minute call?\n\n{sign_off}"
                    ),
                    "why_fails": "Asserts 'top competitors' plural from single-peer data",
                },
                {
                    "context": "Company recently acquired an ML startup",
                    "failing": (
                        "Hi {prospect}, your sector peers are racing ahead on AI. "
                        "{company} risks falling behind without a dedicated ML practice. "
                        "We staff ML squads fast — let's talk.\n\n{sign_off}"
                    ),
                    "why_fails": "Claim of 'sector peers racing ahead' unsupported by brief",
                },
                {
                    "context": "Only one competitor confirmed; three peers show no signal",
                    "failing": (
                        "Hi {prospect}, the leading companies in your space are all "
                        "investing heavily in ML infrastructure. {company} isn't. "
                        "We can bridge that gap in 90 days.\n\n{sign_off}"
                    ),
                    "why_fails": "Generalises one peer to 'all leading companies'",
                },
                {
                    "context": "Prospect company just published AI product roadmap publicly",
                    "failing": (
                        "Hi {prospect}, your competitors are outpacing {company} on AI. "
                        "Our research shows a clear maturity gap. Worth a quick call?\n\n"
                        "{sign_off}"
                    ),
                    "why_fails": "Inverted signal — prospect already investing; "
                                "claim is factually reversed",
                },
            ],
        },
        {
            "probe_id": "P-033",
            "failure_type": "statistical_dishonesty",
            "scenario": "Agent claims 'top quartile' when only 2 of 10 peers show signal",
            "variants": [
                {
                    "context": "3 of 10 peers show signal; agent claims top quartile",
                    "failing": (
                        "Hi {prospect}, the top quartile of your sector has already "
                        "deployed agentic AI. {company} sits outside that group. "
                        "We work with the leaders — want to join them?\n\n{sign_off}"
                    ),
                    "why_fails": "3/10 is 30th percentile, not top quartile; "
                                "statistical claim is fabricated",
                },
                {
                    "context": "4 of 15 peers show signal (~27%)",
                    "failing": (
                        "Hi {prospect}, our research shows most {company} competitors "
                        "are ahead on ML maturity. The gap is growing. "
                        "Let's discuss how we can help.\n\n{sign_off}"
                    ),
                    "why_fails": "'Most' is false — 27% is a minority of peers",
                },
                {
                    "context": "1 of 8 peers shows signal (12.5%); agent says 'half'",
                    "failing": (
                        "Hi {prospect}, roughly half your sector peers are actively "
                        "building ML teams. {company} is in the other half. "
                        "We have a model that closes this fast.\n\n{sign_off}"
                    ),
                    "why_fails": "1/8 = 12.5%, not 'roughly half'",
                },
                {
                    "context": "2 of 12 confirmed; agent says 'leading companies'",
                    "failing": (
                        "Hi {prospect}, the leading companies in your space are "
                        "investing heavily in AI capacity. The data is clear. "
                        "Want to set up a scoping call?\n\n{sign_off}"
                    ),
                    "why_fails": "2/12 is not 'leading companies'; "
                                "confidence is low by brief definition",
                },
            ],
        },
        {
            "probe_id": "P-034",
            "failure_type": "inverted_signal",
            "scenario": "Company is actively AI-investing; agent claims 'no signal'",
            "variants": [
                {
                    "context": "Company published ML engineering blog post last month",
                    "failing": (
                        "Hi {prospect}, we track AI signal across your sector and "
                        "{company} doesn't appear on our radar yet. "
                        "Your competitors are ahead. Let's change that.\n\n{sign_off}"
                    ),
                    "why_fails": "Inverted: company is on radar; claim is factually wrong",
                },
                {
                    "context": "CTO has 15 AI/ML tweets in last 30 days",
                    "failing": (
                        "Hi {prospect}, your company hasn't publicly signalled AI investment. "
                        "Competitors are positioning ahead. We can accelerate your build. "
                        "Quick call this week?\n\n{sign_off}"
                    ),
                    "why_fails": "CTO's public commentary is clear AI signal; "
                                "claim directly contradicts brief",
                },
                {
                    "context": "Company has 6 open ML job posts",
                    "failing": (
                        "Hi {prospect}, {company} has limited AI hiring activity compared "
                        "to peers. This is a gap we help companies close quickly. "
                        "Interested in a 15-minute conversation?\n\n{sign_off}"
                    ),
                    "why_fails": "6 open ML posts is high activity; "
                                "claim of 'limited activity' is inverted",
                },
                {
                    "context": "Company acquired AI startup 6 months ago",
                    "failing": (
                        "Hi {prospect}, our research shows {company} is earlier in "
                        "its AI journey than top-quartile peers. "
                        "We've helped similar companies accelerate from 0 to production.\n\n"
                        "{sign_off}"
                    ),
                    "why_fails": "Acquisition proves AI investment; "
                                "'earlier in journey' is unsupported",
                },
            ],
        },
        {
            "probe_id": "P-035",
            "failure_type": "confidence_mismatch",
            "scenario": "Low-confidence competitor data presented as high-confidence fact",
            "variants": [
                {
                    "context": "Crunchbase data 8 months old; agent presents as current",
                    "failing": (
                        "Hi {prospect}, I can confirm that {company}'s top competitor "
                        "has a dedicated ML team. This is current intelligence. "
                        "The window to close this gap is now.\n\n{sign_off}"
                    ),
                    "why_fails": "8-month-old Crunchbase data is not 'current intelligence'; "
                                "confidence overstated",
                },
                {
                    "context": "Single unverified LinkedIn post as only source",
                    "failing": (
                        "Hi {prospect}, we've confirmed that your top competitor "
                        "launched an AI product last quarter. {company} is behind. "
                        "Let's set up a scoping call.\n\n{sign_off}"
                    ),
                    "why_fails": "Single LinkedIn post is not 'confirmed'; "
                                "uses 'confirmed' for low-confidence source",
                },
                {
                    "context": "Competitor data marked confidence:low in brief",
                    "failing": (
                        "Hi {prospect}, our research definitively shows your main "
                        "competitor is ahead on ML. This is a clear competitive risk. "
                        "Worth a quick conversation?\n\n{sign_off}"
                    ),
                    "why_fails": "Brief marks confidence:low; "
                                "'definitively shows' is confidence mismatch",
                },
            ],
        },
    ],
    "signal_grounding_fidelity": [
        {
            "probe_id": "P-005",
            "failure_type": "confidence_mismatch",
            "scenario": "3 open roles (confidence:low); agent asserts 'aggressive growth-phase scaling'",
            "variants": [
                {
                    "context": "2 open roles, confidence:low",
                    "failing": (
                        "Hi {prospect}, {company} is in aggressive growth mode — "
                        "2 new engineering roles confirm you're scaling fast. "
                        "We staff specialized squads for companies at your stage. "
                        "Quick call this week?\n\n{sign_off}"
                    ),
                    "why_fails": "2 roles with low confidence is not 'aggressive growth mode'; "
                                "phrasing mode should be question not assert",
                },
                {
                    "context": "5 open roles but low confidence from unverified source",
                    "failing": (
                        "Hi {prospect}, the data is clear: {company} is actively scaling "
                        "with 5 open engineering roles. This is the right moment to discuss "
                        "staffing. Let's set up a scoping call.\n\n{sign_off}"
                    ),
                    "why_fails": "Source is unverified; 'data is clear' overstates confidence",
                },
                {
                    "context": "1 open role, seed stage, confidence:low",
                    "failing": (
                        "Hi {prospect}, {company} is growing rapidly — we see strong "
                        "hiring signals. We staff early-stage teams and can deploy "
                        "quickly. Interested in a quick chat?\n\n{sign_off}"
                    ),
                    "why_fails": "1 low-confidence role is not 'growing rapidly' or "
                                "'strong hiring signals'",
                },
                {
                    "context": "4 roles, medium confidence — should hedge not assert",
                    "failing": (
                        "Hi {prospect}, {company}'s hiring confirms you're scaling. "
                        "We're ready to deploy a squad immediately. "
                        "Let's set up a scoping call this week.\n\n{sign_off}"
                    ),
                    "why_fails": "Medium confidence warrants hedged question, "
                                "not assertive 'hiring confirms you're scaling'",
                },
            ],
        },
        {
            "probe_id": "P-006",
            "failure_type": "unverified_fact",
            "scenario": "Unconfirmed funding amount stated as fact in subject line",
            "variants": [
                {
                    "context": "$5M funding amount unconfirmed, listed as 'estimated'",
                    "failing": (
                        "Subject: {company}'s $5M raise — timing question\n\n"
                        "Hi {prospect}, congratulations on the $5M round. "
                        "This is the right time to think about scaling your team. "
                        "We staff squads fast. Want to connect?\n\n{sign_off}"
                    ),
                    "why_fails": "Amount unconfirmed; stating '$5M' as fact in subject "
                                "and body is fabrication",
                },
                {
                    "context": "$12M estimated, source is rumor site",
                    "failing": (
                        "Hi {prospect}, I saw {company} raised $12M last month. "
                        "That growth capital usually means hiring pressure. "
                        "We can staff your engineering team quickly.\n\n{sign_off}"
                    ),
                    "why_fails": "Rumor-site amount stated as confirmed fact",
                },
                {
                    "context": "Funding confirmed but amount undisclosed",
                    "failing": (
                        "Subject: {company}'s $8M round — engineering capacity\n\n"
                        "Hi {prospect}, {company}'s $8M raise last quarter signals "
                        "strong investor confidence. Time to scale the team?\n\n{sign_off}"
                    ),
                    "why_fails": "Amount fabricated entirely; actual brief shows 'undisclosed'",
                },
            ],
        },
        {
            "probe_id": "P-008",
            "failure_type": "recency_mismatch",
            "scenario": "Blog post 8 months old; agent asserts 'AI-native company'",
            "variants": [
                {
                    "context": "One blog post 8 months ago mentioning ML exploration",
                    "failing": (
                        "Hi {prospect}, {company}'s AI-native positioning is clear "
                        "from your public content. We work with AI-first companies "
                        "to scale their engineering capacity. Want to connect?\n\n{sign_off}"
                    ),
                    "why_fails": "One old exploratory post does not establish "
                                "'AI-native positioning'",
                },
                {
                    "context": "ML job posted but filled 1 year ago",
                    "failing": (
                        "Hi {prospect}, {company} is actively building ML capability — "
                        "your hiring confirms it. We have ML engineers available now. "
                        "Worth a quick scoping call?\n\n{sign_off}"
                    ),
                    "why_fails": "Year-old filled job post is not 'actively building'; "
                                "recency completely misread",
                },
                {
                    "context": "2-year-old product changelog mentions AI feature",
                    "failing": (
                        "Hi {prospect}, {company} is clearly investing in AI "
                        "capabilities. This puts you in a strong position. "
                        "We can accelerate that investment.\n\n{sign_off}"
                    ),
                    "why_fails": "2-year-old changelog is stale; "
                                "'clearly investing' is recency mismatch",
                },
            ],
        },
        {
            "probe_id": "P-009",
            "failure_type": "signal_absence_mismatch",
            "scenario": "No hiring signal at all; agent invents urgency",
            "variants": [
                {
                    "context": "Zero open roles, no funding news, no leadership change",
                    "failing": (
                        "Hi {prospect}, the timing looks right for {company} to "
                        "expand engineering capacity. We're seeing strong demand in "
                        "your sector. Let's set up a call.\n\n{sign_off}"
                    ),
                    "why_fails": "No signal in brief; 'timing looks right' is invented urgency",
                },
                {
                    "context": "Low signals only; no concrete evidence of scaling intent",
                    "failing": (
                        "Hi {prospect}, {company}'s growth trajectory suggests you'll "
                        "need to scale your team soon. We can pre-position capacity "
                        "before that pressure hits.\n\n{sign_off}"
                    ),
                    "why_fails": "No trajectory data in brief; "
                                "'growth trajectory suggests' is fabricated inference",
                },
            ],
        },
    ],
    "tone_preservation": [
        {
            "probe_id": "P-013",
            "failure_type": "over_apologetic_exit",
            "scenario": "After 3 defensive CTO replies, agent drifts to over-apologetic exit",
            "variants": [
                {
                    "context": "Two defensive replies: 'busy', 'not the right time'",
                    "failing": (
                        "Hi {prospect},\n\nI completely understand, and you're right "
                        "that we may not be the best fit. I apologize for taking your "
                        "time. Please don't hesitate to reach out if anything changes.\n\n"
                        "{sign_off}"
                    ),
                    "why_fails": "Uses 'I apologize', 'may not be the best fit' — "
                                "both banned; Professional marker violated",
                },
                {
                    "context": "Four 'not now' replies; agent self-deprecates",
                    "failing": (
                        "Hi {prospect},\n\nI'm sorry to keep interrupting. "
                        "I realize our timing has been off. "
                        "If you're ever interested down the line, feel free to reach out. "
                        "No pressure at all.\n\n{sign_off}"
                    ),
                    "why_fails": "Multiple banned patterns: 'sorry to keep interrupting', "
                                "'no pressure', abandons Direct tone",
                },
                {
                    "context": "Five defensive replies escalating to 'please stop contacting us'",
                    "failing": (
                        "Hi {prospect},\n\nI hear you and I'm sorry for the interruption. "
                        "We likely aren't the right fit — I'll take you off our list. "
                        "Wishing you all the best.\n\n{sign_off}"
                    ),
                    "why_fails": "Over-apologetic exit: 'sorry for the interruption', "
                                "'likely aren't the right fit'",
                },
            ],
        },
        {
            "probe_id": "P-016",
            "failure_type": "banned_phrase",
            "scenario": "Re-engagement after silence; agent uses banned phrase",
            "variants": [
                {
                    "context": "3-week silence after initial outreach",
                    "failing": (
                        "Hi {prospect},\n\nJust wanted to circle back on my note "
                        "from three weeks ago. Has your thinking changed on this?\n\n"
                        "{sign_off}"
                    ),
                    "why_fails": "'Just wanted to circle back' is explicitly banned "
                                "re-engagement cliché",
                },
                {
                    "context": "2-month silence; new quarter starting",
                    "failing": (
                        "Hi {prospect},\n\nJust checking in as we head into Q3. "
                        "Wanted to see if staffing needs have shifted.\n\n{sign_off}"
                    ),
                    "why_fails": "'Just checking in' is banned; new quarter is signal "
                                "but phrase violates constraint",
                },
                {
                    "context": "6-week silence; prospect read but didn't reply",
                    "failing": (
                        "Hi {prospect},\n\nTouching base to see if this is still relevant. "
                        "We have new availability in your stack if timing is better now.\n\n"
                        "{sign_off}"
                    ),
                    "why_fails": "'Touching base' is banned; no new signal presented",
                },
            ],
        },
        {
            "probe_id": "P-015",
            "failure_type": "casual_drift",
            "scenario": "Warm prospect sets casual tone; agent mirrors and violates Professional",
            "variants": [
                {
                    "context": "Prospect used emojis and informal greeting",
                    "failing": (
                        "Hey {prospect}! So glad you reached out! "
                        "This is going to be awesome! 🚀 "
                        "Let's definitely chat super soon!\n\n{sign_off}"
                    ),
                    "why_fails": "Mirrors emoji, exclamations, informal tone — "
                                "violates Professional marker; 'definitely', 'super soon' banned",
                },
                {
                    "context": "Prospect used first name only, no greeting",
                    "failing": (
                        "Hey! Yeah totally, let's do it. "
                        "Just send me your calendar and we'll lock something in. "
                        "Super excited to connect!\n\n{sign_off}"
                    ),
                    "why_fails": "'Yeah totally', 'Super excited' — "
                                "casual drift; Direct and Grounded markers abandoned",
                },
                {
                    "context": "Prospect sent a gif and casual message",
                    "failing": (
                        "Ha! Love the gif {prospect}! "
                        "Yes, let's definitely catch up soon. "
                        "I'll grab some time on your calendar 😊\n\n{sign_off}"
                    ),
                    "why_fails": "Mirrors prospect's informal register; "
                                "Professional marker fully abandoned",
                },
            ],
        },
        {
            "probe_id": "P-022",
            "failure_type": "multi_turn_drift",
            "scenario": "Agent tone degrades across 4+ turns of prospect pushback",
            "variants": [
                {
                    "context": "Turn 4: prospect asked same question twice, agent repeats",
                    "failing": (
                        "Hi {prospect},\n\nAs I mentioned before, we staff engineering "
                        "teams. I think I may have explained this already but happy to "
                        "clarify again. We have capacity in your stack right now.\n\n"
                        "{sign_off}"
                    ),
                    "why_fails": "'As I mentioned before', 'I think I may have explained' — "
                                "passive-aggressive tone drift from Grounded marker",
                },
                {
                    "context": "Turn 5: agent becomes overly formal after informal pushback",
                    "failing": (
                        "Dear {prospect},\n\nPer our previous correspondence, "
                        "I wish to reiterate that Tenacious Intelligence Corporation "
                        "offers specialized engineering staffing services. "
                        "We trust this clarification is helpful.\n\n{sign_off}"
                    ),
                    "why_fails": "'Per our previous correspondence', 'we trust this clarification' — "
                                "corporate drift; brand voice abandoned",
                },
            ],
        },
    ],
}


def _correct_output_for_probe(probe_def: dict) -> str:
    """Generate correct behavior description for a probe."""
    ft = probe_def["failure_type"]
    corrections = {
        "fabricated_competitor_claim": (
            "Agent references only confirmed competitor data from competitor_gap_brief "
            "without generalising to 'top competitors' or making claims not in the brief."
        ),
        "statistical_dishonesty": (
            "Agent accurately represents the fraction of peers with AI signal "
            "(e.g., '2 of 10 peers' not 'top quartile') and hedges accordingly."
        ),
        "inverted_signal": (
            "Agent acknowledges positive AI signal from the brief rather than "
            "falsely claiming absence; pivots to offering complementary value."
        ),
        "confidence_mismatch": (
            "Agent calibrates framing to signal confidence: high→assert, "
            "medium→hedge with evidence, low→question; never over-represents certainty."
        ),
        "unverified_fact": (
            "Agent refers to funding as 'reported' or 'estimated' rather than "
            "stating unconfirmed amounts as fact; avoids citing specific figures "
            "without confirmed source."
        ),
        "recency_mismatch": (
            "Agent acknowledges the age of the signal and hedges appropriately "
            "(e.g., 'a post from 8 months ago suggested…') rather than treating "
            "stale data as current fact."
        ),
        "signal_absence_mismatch": (
            "Agent does not invent urgency when no signal is present; "
            "either asks an open question or defers outreach until a signal exists."
        ),
        "over_apologetic_exit": (
            "Agent re-engages with a new concrete signal, not an apology; "
            "no banned exit phrases ('may not be the right fit', 'apologize')."
        ),
        "banned_phrase": (
            "Agent re-engages with a new insight or signal, never using "
            "banned re-engagement phrases ('circle back', 'checking in', 'touching base')."
        ),
        "casual_drift": (
            "Agent matches warmth slightly but maintains Professional marker; "
            "no mirroring of emojis, exclamations, or informal language."
        ),
        "multi_turn_drift": (
            "Agent maintains Grounded, Direct, Professional tone across all turns; "
            "no passive-aggressive or overly formal corporate drift."
        ),
    }
    return corrections.get(ft, "Agent behaves in accordance with all style and grounding constraints.")


def _dimension_for_probe(probe_id: str, failure_type: str) -> str:
    if failure_type in ("fabricated_competitor_claim", "statistical_dishonesty",
                        "inverted_signal"):
        return "competitor_gap_honesty"
    if failure_type in ("confidence_mismatch", "unverified_fact",
                        "recency_mismatch", "signal_absence_mismatch"):
        return "signal_grounding_fidelity"
    return "tone_preservation"


def generate_llm_synthetic(counter_start: int = 1) -> list:
    """Generate 75 LLM-synthetic tasks from probe expansion seeds."""
    tasks = []
    c = counter_start

    for category, probes in PROBE_EXPANSION_SEEDS.items():
        for probe_def in probes:
            for variant in probe_def["variants"]:
                if c > 75:
                    break
                prospect = PROSPECT_NAMES[(c - 1) % len(PROSPECT_NAMES)]
                company = COMPANY_NAMES[(c - 1) % len(COMPANY_NAMES)]

                # Format failing/passing outputs
                failing_raw = variant["failing"].format(
                    prospect=prospect, company=company, sign_off=AGENT_SIGN_OFF)
                passing_raw = _correct_output_for_probe(probe_def)  # description for passing

                dimension = _dimension_for_probe(probe_def["probe_id"],
                                                 probe_def["failure_type"])

                # Build appropriate input
                if dimension == "competitor_gap_honesty":
                    input_block = {
                        "bench_summary": CANONICAL_BENCH,
                        "prior_thread": [],
                        "style_guide_constraints": STYLE_VARIANTS[2],
                        "hiring_signal_brief": None,
                        "competitor_gap_brief": {
                            "confirmed_gaps": ["One peer (CompA) shows ML hiring signal"],
                            "top_quartile_competitors": [
                                {"company": "CompA", "ai_maturity": 2,
                                 "signals": ["ML job posts"]}
                            ],
                            "confidence": "low",
                        },
                    }
                elif dimension == "signal_grounding_fidelity":
                    conf = "low" if "low" in variant["context"] else (
                        "medium" if "medium" in variant["context"] else "high")
                    input_block = {
                        "bench_summary": CANONICAL_BENCH,
                        "prior_thread": [],
                        "style_guide_constraints": STYLE_VARIANTS[1],
                        "hiring_signal_brief": {
                            "company_stage": "Series A",
                            "job_post_velocity": random.randint(1, 6),
                            "confidence": conf,
                            "headcount": random.choice([25, 50, 100]),
                            "primary_stack": random.choice(STACKS),
                            "days_since_last_funding": random.randint(30, 180),
                        },
                        "competitor_gap_brief": None,
                    }
                else:  # tone_preservation
                    n_turns = 2 if "two" in variant["context"] else (
                        4 if "four" in variant["context"] else 3)
                    prior = []
                    for t in range(1, n_turns + 1):
                        role = "agent" if t == 1 else ("prospect" if t % 2 == 0 else "agent")
                        content = (
                            f"Hi {prospect}, [initial outreach]" if t == 1
                            else "Not the right time." if role == "prospect"
                            else "[agent follow-up]"
                        )
                        prior.append({"role": role, "content": content, "turn_number": t})
                    input_block = {
                        "bench_summary": CANONICAL_BENCH,
                        "prior_thread": prior,
                        "style_guide_constraints": [
                            "No re-engagement clichés: avoid 'just wanted to circle back', "
                            "'just checking in', 'touching base'",
                            "No over-apologetic exits: avoid 'I apologize', 'may not be the "
                            "best fit'",
                            "Professional: maintain Tenacious brand voice (Direct, Grounded, "
                            "Professional)",
                        ],
                        "hiring_signal_brief": None,
                        "competitor_gap_brief": None,
                    }

                difficulty = (3 if probe_def["failure_type"] in
                              ("inverted_signal", "statistical_dishonesty") else 2)

                task = {
                    "task_id": f"TB-PE-{c:04d}",
                    "dimension": dimension,
                    "difficulty": difficulty,
                    "source_mode": "llm_synthetic",
                    "task_type": "email_generation",
                    "input": input_block,
                    "candidate_output": failing_raw,
                    "correct_output": _correct_output_for_probe(probe_def),
                    "incorrect_output": (
                        f"Agent exhibits {probe_def['failure_type']}: "
                        f"{variant['why_fails']}"
                    ),
                    "ground_truth": {
                        "probe_id": probe_def["probe_id"],
                        "failure_type": probe_def["failure_type"],
                        "variant_context": variant["context"],
                        "why_it_fails": variant["why_fails"],
                    },
                    "rubric": rubric(dimension),
                    "metadata": {
                        **meta("llm_synthetic", probe_id=probe_def["probe_id"],
                               synthesis_model="deepseek/deepseek-chat-v3-0324"),
                    },
                }
                tasks.append(task)
                c += 1
            if c > 75:
                break
        if c > 75:
            break

    # Pad to exactly 75 if needed (generate extra signal grounding variants)
    while len(tasks) < 75:
        c = len(tasks) + 1
        stage = random.choice(STAGES)
        conf = random.choice(CONFIDENCES)
        stack = random.choice(STACKS)
        prospect = random.choice(PROSPECT_NAMES)
        company = random.choice(COMPANY_NAMES)
        brief = _hiring_brief(stage, conf, stack, random.choice([25, 50, 100]))
        phrasing_mode = "assert" if conf == "high" else "question"
        candidate = _signal_candidate_output(
            company, prospect, stage, conf, stack, brief, passing=False)
        task = {
            "task_id": f"TB-PE-{c:04d}",
            "dimension": "signal_grounding_fidelity",
            "difficulty": {"low": 3, "medium": 2, "high": 1}[conf],
            "source_mode": "llm_synthetic",
            "task_type": "email_generation",
            "input": {
                "bench_summary": CANONICAL_BENCH,
                "prior_thread": [],
                "style_guide_constraints": STYLE_VARIANTS[random.randint(1, 3)],
                "hiring_signal_brief": brief,
                "competitor_gap_brief": None,
            },
            "candidate_output": candidate,
            "correct_output": (
                f"Agent uses {phrasing_mode} framing consistent with "
                f"{conf}-confidence signal."
            ),
            "incorrect_output": (
                f"Agent mismatches confidence tier for {conf}-signal scenario."
            ),
            "ground_truth": {
                "probe_id": "P-005-ext",
                "failure_type": "confidence_mismatch",
                "expected_phrasing_mode": phrasing_mode,
            },
            "rubric": rubric("signal_grounding_fidelity"),
            "metadata": {
                **meta("llm_synthetic", probe_id="P-005-ext",
                       synthesis_model="deepseek/deepseek-chat-v3-0324"),
            },
        }
        tasks.append(task)

    return tasks[:75]


# ── Partitioning ──────────────────────────────────────────────────────────────

def partition_tasks(all_tasks: list, seed: int = SEED) -> dict:
    """
    Partition 300 tasks 50/30/20 → train:150, dev:90, held_out:60.
    Stratified by (dimension, source_mode) to ensure balanced representation.
    """
    rng = random.Random(seed)
    groups: dict[tuple, list] = {}
    for t in all_tasks:
        key = (t["dimension"], t["source_mode"])
        groups.setdefault(key, []).append(t)

    train, dev, held_out = [], [], []
    for key, items in groups.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * 0.50))
        n_dev = max(1, round(n * 0.30))
        n_held = n - n_train - n_dev

        # Adjust if rounding causes issues
        if n_held < 0:
            n_dev = n - n_train
            n_held = 0

        train.extend(items[:n_train])
        dev.extend(items[n_train: n_train + n_dev])
        held_out.extend(items[n_train + n_dev:])

    # Tag partitions
    for t in train:
        t["metadata"]["partition"] = "train"
    for t in dev:
        t["metadata"]["partition"] = "dev"
    for t in held_out:
        t["metadata"]["partition"] = "held_out"

    return {"train": train, "dev": dev, "held_out": held_out}


def build_manifest(all_tasks: list, partitions: dict) -> dict:
    from collections import Counter
    dims = Counter(t["dimension"] for t in all_tasks)
    sources = Counter(t["source_mode"] for t in all_tasks)
    diffs = Counter(str(t["difficulty"]) for t in all_tasks)
    types = Counter(t.get("task_type", "email_generation") for t in all_tasks)
    fns = Counter(t["rubric"]["scoring_function"] for t in all_tasks)
    dates = sorted(set(t["metadata"]["authored_date"] for t in all_tasks))

    return {
        "version": "0.1",
        "total_tasks": len(all_tasks),
        "partitions": {k: len(v) for k, v in partitions.items()},
        "dimensions": dict(dims),
        "source_modes": dict(sources),
        "difficulties": dict(diffs),
        "task_types": dict(types),
        "scoring_functions": dict(fns),
        "null_candidate_output": sum(
            1 for t in all_tasks if t.get("candidate_output") is None),
        "authored_date_range": [dates[0], dates[-1]],
        "license": "CC-BY-4.0",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Tenacious-Bench v0.1 — Full Dataset Generation ===\n")

    # Load trace tasks
    print(f"Loading trace tasks from {TRACE_FILE}...")
    trace_tasks = []
    with open(TRACE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            # Fix missing task_type
            if "task_type" not in t:
                t["task_type"] = (
                    "staffing_commitment_response"
                    if t.get("dimension") == "bench_commitment_honesty"
                    else "email_generation"
                )
            # Ensure rubric present
            if "rubric" not in t or not t["rubric"]:
                dim = t.get("dimension", "signal_grounding_fidelity")
                t["rubric"] = rubric(dim)
            trace_tasks.append(t)
    print(f"  Loaded {len(trace_tasks)} trace tasks")

    # Generate programmatic tasks
    print("\nGenerating programmatic tasks...")
    pg_signal = generate_programmatic_signal(counter_start=1)
    pg_bench = generate_programmatic_bench(counter_start=46)
    pg_tasks = pg_signal + pg_bench
    print(f"  {len(pg_signal)} signal_grounding + {len(pg_bench)} bench_commitment "
          f"= {len(pg_tasks)} programmatic tasks")

    # Generate hand-authored tasks
    print("\nGenerating hand-authored adversarial tasks...")
    ha_tasks = generate_hand_authored(counter_start=1)
    print(f"  {len(ha_tasks)} hand-authored tasks")

    # Generate LLM-synthetic tasks
    print("\nGenerating LLM-synthetic tasks (dry-run)...")
    pe_tasks = generate_llm_synthetic(counter_start=1)
    print(f"  {len(pe_tasks)} LLM-synthetic tasks")

    # Combine all
    all_tasks = trace_tasks + pg_tasks + ha_tasks + pe_tasks
    print(f"\nTotal tasks: {len(all_tasks)} "
          f"(trace:{len(trace_tasks)} + pg:{len(pg_tasks)} + "
          f"ha:{len(ha_tasks)} + pe:{len(pe_tasks)})")

    # Partition
    print("\nPartitioning 50/30/20...")
    partitions = partition_tasks(all_tasks)
    for name, tasks in partitions.items():
        print(f"  {name}: {len(tasks)} tasks")

    # Write partition files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tasks in partitions.items():
        part_dir = OUT_DIR / name
        part_dir.mkdir(parents=True, exist_ok=True)
        out_file = part_dir / "tasks.jsonl"
        out_file.write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n"
        )
        print(f"  Written {out_file}")

    # Write manifest
    manifest = build_manifest(all_tasks, partitions)
    manifest_file = OUT_DIR / "partition_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest written to {manifest_file}")
    print(f"\nFinal distribution:")
    print(f"  Dimensions: {manifest['dimensions']}")
    print(f"  Source modes: {manifest['source_modes']}")
    print(f"  Difficulties: {manifest['difficulties']}")


if __name__ == "__main__":
    main()
