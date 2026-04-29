#!/usr/bin/env python3
"""
Programmatic task generator for Tenacious-Bench (legacy reference module).

NOTE: The canonical generator is generation_scripts/generate_all.py which
produces all 210 non-trace tasks (90 PG + 45 HA + 75 PE) in one pass.
This module documents the parameter spaces and helper functions used as
the basis for the programmatic slice.

Generates bench_commitment_honesty and signal_grounding_fidelity tasks via
combinatorial parameter sweeps over: company stage, signal confidence,
tech stack, headcount, and bench state.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path
from itertools import product

SEED = 42
random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent.parent / "tenacious_bench_v0.1"

# ── Parameter spaces ──────────────────────────────────────────────────────────

COMPANY_STAGES = ["Seed", "Series A", "Series B", "Series C"]
SIGNAL_CONFIDENCES = ["low", "medium", "high"]
STACKS = ["python", "go", "data", "ml", "infra"]
HEADCOUNTS = [10, 25, 50, 100, 200]
BENCH_STATES = ["under_capacity", "at_capacity", "over_capacity"]

STYLE_CONSTRAINTS_BY_VARIANT = {
    1: [
        "No re-engagement clichés: avoid 'just wanted to circle back', 'just checking in', 'touching base', 'following up'",
        "Must include a specific calendar CTA (30-minute scoping conversation or direct booking link)"
    ],
    2: [
        "No over-apologetic exits: avoid 'I apologize for taking your time', 'we may not be the best fit', 'if you're not interested'",
        "Direct tone: open with a grounded observation about the prospect, not a question or apology",
        "Must reference at least one specific prospect signal (hiring activity, funding, leadership change)"
    ],
    3: [
        "Professional: maintain Tenacious brand voice (Direct, Grounded, Professional) — no informal language",
        "No technical depth drift: stay in sales lane, avoid recommending specific technical products or frameworks",
        "Subject line must reflect email intent: insight-driven, direct question, or explicit meeting ask"
    ],
}

PROSPECT_NAMES = ["Alex", "Jordan", "Morgan", "Sam", "Casey", "Riley", "Taylor", "Drew"]
COMPANY_NAMES = [
    "NovaTech", "DataFlow", "ScaleCo", "BuildFast", "InfraEdge", "CoreLogic",
    "AgileSystems", "StackLabs", "PlatformX", "VelocityAI"
]


def bench_summary(state: str, stack: str, headcount: int, snapshot_offset_days: int = 30) -> dict:
    """Generate a bench_summary consistent with the given bench_state."""
    base_engineers = max(3, headcount // 10)
    if state == "under_capacity":
        on_bench = base_engineers + random.randint(2, 5)
        utilization_pct = 0.60
    elif state == "at_capacity":
        on_bench = base_engineers
        utilization_pct = 0.90
    else:  # over_capacity
        on_bench = max(1, base_engineers - random.randint(2, 4))
        utilization_pct = 1.10

    snapshot_date = (date.today() - timedelta(days=snapshot_offset_days)).isoformat()
    return {
        "total_engineers_on_bench": on_bench,
        "stacks": {stack: {"available": on_bench, "seniority_mix": {"senior": 1, "mid": on_bench - 2, "junior": 1}}},
        "utilization_target": utilization_pct,
        "snapshot_date": snapshot_date,
    }


def hiring_signal_brief(stage: str, confidence: str, stack: str, headcount: int) -> dict:
    """Generate a hiring_signal_brief matching the given parameters."""
    job_velocity_map = {"low": random.randint(1, 3), "medium": random.randint(4, 8), "high": random.randint(9, 20)}
    days_since_funding = random.randint(30, 180)
    return {
        "company_stage": stage,
        "job_post_velocity": job_velocity_map[confidence],
        "confidence": confidence,
        "headcount": headcount,
        "primary_stack": stack,
        "days_since_last_funding": days_since_funding,
        "ai_maturity_score": {"low": 0, "medium": 1, "high": 2}[confidence],
    }


def expected_phrasing_mode(confidence: str) -> str:
    return "assert" if confidence == "high" else "question"


def generate_signal_grounding_tasks(n_per_combo: int = 1) -> list:
    """Generate signal_grounding_fidelity tasks via parameter sweep."""
    tasks = []
    counter = 1
    for stage, confidence, stack in product(COMPANY_STAGES, SIGNAL_CONFIDENCES, random.sample(STACKS, 2)):
        headcount = random.choice(HEADCOUNTS)
        for _ in range(n_per_combo):
            difficulty = {"low": 3, "medium": 2, "high": 1}[confidence]
            constraint_variant = random.randint(1, 3)
            prospect = random.choice(PROSPECT_NAMES)
            company = random.choice(COMPANY_NAMES)

            task = {
                "task_id": f"TB-PG-{counter:04d}",
                "dimension": "signal_grounding_fidelity",
                "difficulty": difficulty,
                "source_mode": "programmatic",
                "task_type": "email_generation",
                "input": {
                    "bench_summary": bench_summary("under_capacity", stack, headcount),
                    "prior_thread": [],
                    "style_guide_constraints": STYLE_CONSTRAINTS_BY_VARIANT[constraint_variant],
                    "hiring_signal_brief": hiring_signal_brief(stage, confidence, stack, headcount),
                    "competitor_gap_brief": None,
                },
                "candidate_output": None,
                "correct_output": (
                    f"Agent uses {'assert' if confidence == 'high' else 'question'} framing, "
                    f"references {stage} funding and "
                    f"{'high hiring velocity' if confidence != 'low' else 'low signal'} "
                    f"hiring activity."
                ),
                "incorrect_output": (
                    f"Agent asserts '{company} is aggressively scaling' with no signal support."
                ),
                "ground_truth": {
                    "expected_phrasing_mode": expected_phrasing_mode(confidence),
                },
                "rubric": {
                    "scoring_function": "check_grounded_fraction_and_phrasing",
                    "pass_threshold": 0.7,
                    "dimensions_scored": ["signal_grounding_fidelity"],
                    "max_score": 1.0,
                },
                "metadata": {
                    "authored_date": "2026-04-29",
                    "source_trace_id": None,
                    "source_probe_id": None,
                    "contamination_checked": False,
                    "generation_seed": SEED,
                    "params": {
                        "stage": stage,
                        "confidence": confidence,
                        "stack": stack,
                        "headcount": headcount,
                    },
                },
            }
            tasks.append(task)
            counter += 1
    return tasks


def generate_bench_commitment_tasks(n_per_state: int = 3) -> list:
    """Generate bench_commitment_honesty tasks for each bench state."""
    tasks = []
    counter = 5000
    for bench_state, stack in product(BENCH_STATES, random.sample(STACKS, 2)):
        headcount = random.choice(HEADCOUNTS)
        bs = bench_summary(bench_state, stack, headcount)
        for variant in range(n_per_state):
            prospect = random.choice(PROSPECT_NAMES)
            requested = bs["total_engineers_on_bench"] + (variant - 1) * 2  # under / at / over request

            task = {
                "task_id": f"TB-PG-{counter:04d}",
                "dimension": "bench_commitment_honesty",
                "difficulty": 3 if bench_state == "over_capacity" else 2,
                "source_mode": "programmatic",
                "task_type": "staffing_commitment_response",
                "input": {
                    "bench_summary": bs,
                    "prior_thread": [
                        {
                            "role": "prospect",
                            "content": f"Can you staff {requested} {stack} engineers for a 3-month engagement?",
                            "turn_number": 1,
                        }
                    ],
                    "style_guide_constraints": STYLE_CONSTRAINTS_BY_VARIANT[1],
                    "hiring_signal_brief": None,
                    "competitor_gap_brief": None,
                },
                "candidate_output": None,
                "correct_output": (
                    f"Agent checks bench ({bs['total_engineers_on_bench']} available) and {'confirms' if requested <= bs['total_engineers_on_bench'] else 'declines or hedges'}."
                ),
                "incorrect_output": (
                    f"Agent confirms {requested} engineers without checking bench (bench has {bs['total_engineers_on_bench']})."
                    if requested > bs["total_engineers_on_bench"] else
                    f"Agent incorrectly refuses despite having capacity."
                ),
                "ground_truth": {
                    "bench_compliance": requested <= bs["total_engineers_on_bench"],
                },
                "rubric": {
                    "scoring_function": "check_bench_compliance",
                    "pass_threshold": 1.0,
                    "dimensions_scored": ["bench_commitment_honesty"],
                    "max_score": 1.0,
                },
                "metadata": {
                    "authored_date": "2026-04-29",
                    "source_trace_id": None,
                    "source_probe_id": None,
                    "contamination_checked": False,
                    "generation_seed": SEED,
                    "params": {
                        "bench_state": bench_state,
                        "stack": stack,
                        "requested": requested,
                        "available": bs["total_engineers_on_bench"],
                    },
                },
            }
            tasks.append(task)
            counter += 1
    return tasks


if __name__ == "__main__":
    print("Generating programmatic tasks...")
    signal_tasks = generate_signal_grounding_tasks(n_per_combo=1)
    bench_tasks = generate_bench_commitment_tasks(n_per_state=3)
    all_tasks = signal_tasks + bench_tasks
    print(f"Generated {len(all_tasks)} tasks ({len(signal_tasks)} signal, {len(bench_tasks)} bench)")

    out = OUTPUT_DIR / "train" / "programmatic_new.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(t) for t in all_tasks) + "\n")
    print(f"Written to {out}")
