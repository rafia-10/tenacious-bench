#!/usr/bin/env python3
"""
Tenacious-Bench v0.1 Scoring Evaluator
Scores agent outputs against benchmark rubrics for five Tenacious-specific sales dimensions.
All deterministic checks run locally; LLM-judge checks require ANTHROPIC_API_KEY or OPENROUTER_API_KEY.

Usage:
    python scoring_evaluator.py --task task.json               # Score one task
    python scoring_evaluator.py --partition dev                # Score entire partition
    python scoring_evaluator.py --partition dev --output results.json
    python scoring_evaluator.py --demo                         # Run against 3 built-in demo tasks
"""

import json
import re
import argparse
import os
import sys
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).parent
PARTITION_DIR = BENCH_ROOT / "tenacious_bench_v0.1"

# ──────────────────────────────────────────────────────────────────────────────
# Banned phrase lists (sourced from audit_memo.md + probe_library.md)
# ──────────────────────────────────────────────────────────────────────────────
BANNED_REENGAGEMENT_PHRASES = [
    "just wanted to circle back",
    "just checking in",
    "touching base",
    "following up on my last",
    "as i mentioned",
    "per my last email",
    "hope this finds you well",
    "i wanted to reach out",
    "hope you're doing well",
    "hope all is well",
]

BANNED_APOLOGETIC_PHRASES = [
    "i apologize for taking your time",
    "i apologize if this isn't relevant",
    "we may not be the best fit",
    "if you're not interested",
    "i completely understand, and you're right",
    "sorry to bother",
    "sorry for the interruption",
]

# Signals of a proper calendar CTA
CALENDAR_CTA_PATTERNS = [
    r"30.?minute",
    r"30.?min",
    r"scoping conversation",
    r"calendar",
    r"book a time",
    r"schedule a call",
    r"set up a call",
    r"grab 30",
    r"a quick call",
    r"discovery call",
]


# ──────────────────────────────────────────────────────────────────────────────
# Core scoring functions
# ──────────────────────────────────────────────────────────────────────────────

def check_tone_preservation(task: dict) -> dict:
    """
    Scores a candidate output on Tenacious tone adherence.
    Four deterministic checks; any banned-phrase hit is an immediate hard fail.
    """
    output = task.get("candidate_output") or ""
    if not output:
        return _skipped("no candidate_output")

    output_lower = output.lower()
    checks = {}
    hard_fail = None

    # Check 1: No banned re-engagement cliches (weight 0.30)
    hit = next((p for p in BANNED_REENGAGEMENT_PHRASES if p in output_lower), None)
    if hit:
        hard_fail = f"banned re-engagement phrase found: '{hit}'"
        checks["banned_phrases"] = {"pass": False, "weight": 0.30, "detail": hard_fail}
    else:
        checks["banned_phrases"] = {"pass": True, "weight": 0.30}

    # Check 2: No over-apologetic language (weight 0.20)
    apologetic_hit = next((p for p in BANNED_APOLOGETIC_PHRASES if p in output_lower), None)
    if apologetic_hit:
        hard_fail = hard_fail or f"over-apologetic phrase found: '{apologetic_hit}'"
        checks["no_over_apologetic"] = {"pass": False, "weight": 0.20, "detail": apologetic_hit}
    else:
        checks["no_over_apologetic"] = {"pass": True, "weight": 0.20}

    # Check 3: Calendar/meeting CTA present (weight 0.30)
    has_cta = any(re.search(pat, output_lower) for pat in CALENDAR_CTA_PATTERNS)
    checks["calendar_cta"] = {"pass": has_cta, "weight": 0.30}

    # Check 4: Direct opener — not starting with generic hedge (weight 0.20)
    first_body_line = ""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Subject:") and not stripped.startswith("Hi ") and not stripped.startswith("Hello "):
            first_body_line = stripped.lower()
            break
    is_direct_open = not any(first_body_line.startswith(p) for p in [
        "i hope", "i wanted", "just ", "following", "i'm reaching out", "i am reaching out"
    ])
    checks["direct_opening"] = {"pass": is_direct_open, "weight": 0.20}

    if hard_fail:
        return _scored(0.0, checks, hard_fail=hard_fail)

    score = sum(c["weight"] for c in checks.values() if c["pass"])
    return _scored(score, checks)


def check_grounded_fraction_and_phrasing(task: dict) -> dict:
    """
    Checks that factual claims are grounded in the hiring_signal_brief
    and that phrasing mode (assert vs. question) matches signal confidence.
    """
    output = task.get("candidate_output") or ""
    if not output:
        return _skipped("no candidate_output")

    gt = task.get("ground_truth", {})
    expected_mode = gt.get("expected_phrasing_mode", "assert")
    hiring_brief = task.get("input", {}).get("hiring_signal_brief") or {}

    output_lower = output.lower()
    checks = {}

    # Check 1: Phrasing mode matches signal confidence (weight 0.50)
    ASSERT_INDICATORS = ["recently", "secured", "closed", "has ", "currently", "is hiring", "shows"]
    QUESTION_INDICATORS = ["curious how", "wondering if", "would be curious", "are you", "how are you"]

    has_assert_language = any(p in output_lower for p in ASSERT_INDICATORS)
    has_question_language = any(p in output_lower for p in QUESTION_INDICATORS)

    if expected_mode == "assert":
        phrasing_ok = has_assert_language
    elif expected_mode == "question":
        phrasing_ok = has_question_language and not has_assert_language
    else:
        phrasing_ok = True

    checks["phrasing_mode"] = {
        "pass": phrasing_ok,
        "weight": 0.50,
        "expected_mode": expected_mode,
        "detail": f"expected '{expected_mode}', output {'matched' if phrasing_ok else 'did not match'}",
    }

    # Check 2: No generic ungrounded peer-benchmarking language (weight 0.50)
    UNGROUNDED_PATTERNS = [
        r"top.quartile of your sector",
        r"most companies in your stage",
        r"peer companies (?:in your|at your)",
        r"industry standard",
        r"typical for companies like yours",
    ]
    ungrounded_hit = next(
        (p for p in UNGROUNDED_PATTERNS if re.search(p, output_lower)),
        None
    )
    grounding_ok = ungrounded_hit is None
    checks["no_ungrounded_claims"] = {
        "pass": grounding_ok,
        "weight": 0.50,
        "detail": f"ungrounded pattern found: '{ungrounded_hit}'" if ungrounded_hit else "ok",
    }

    score = sum(c["weight"] for c in checks.values() if c["pass"])
    return _scored(score, checks)


def check_probe_condition(task: dict) -> dict:
    """
    Evaluates whether the agent satisfies the condition described in the probe.
    For null candidate_output tasks (templates), returns skipped.
    """
    output = task.get("candidate_output") or ""
    if not output:
        return _skipped("no candidate_output — probe template awaiting agent run")

    gt = task.get("ground_truth", {})
    probe_id = gt.get("probe_id", "")
    expected_correct = task.get("correct_output", "")
    expected_incorrect = task.get("incorrect_output", "")

    output_lower = output.lower()
    checks = {}

    # Timezone probe heuristic (P-025, P-026, P-027)
    if probe_id in ("P-025", "P-026", "P-027"):
        tz_keywords = ["utc", "cest", "cet", "berlin", "london", "timezone", "time zone"]
        mentions_tz = any(k in output_lower for k in tz_keywords)
        checks["timezone_acknowledgment"] = {
            "pass": mentions_tz,
            "weight": 1.0,
            "detail": "output mentions timezone context" if mentions_tz else "output ignores timezone",
        }
        return _scored(1.0 if mentions_tz else 0.0, checks)

    # Bench over-commitment probes (P-009, P-011, P-012)
    if probe_id in ("P-009", "P-011", "P-012"):
        bench = task.get("input", {}).get("bench_summary", {})
        total = bench.get("total_engineers_on_bench", 0)
        numbers_in_output = [int(m) for m in re.findall(r'\b(\d{1,3})\b', output)]
        over_committed = any(n > total for n in numbers_in_output if n > 0 and total > 0)
        checks["no_bench_overcommit"] = {
            "pass": not over_committed,
            "weight": 1.0,
            "detail": f"bench={total}, numbers in output={numbers_in_output}",
        }
        return _scored(0.0 if over_committed else 1.0, checks,
                       hard_fail="bench over-commitment" if over_committed else None)

    # Generic: keyword overlap with correct_output vs incorrect_output
    keywords_correct = re.findall(r'\b\w{4,}\b', expected_correct.lower())[:8]
    keywords_incorrect = re.findall(r'\b\w{4,}\b', expected_incorrect.lower())[:8]
    correct_hits = sum(1 for k in keywords_correct if k in output_lower)
    incorrect_hits = sum(1 for k in keywords_incorrect if k in output_lower)

    score = correct_hits / max(len(keywords_correct), 1)
    checks["probe_condition"] = {
        "pass": score >= 0.5,
        "weight": 1.0,
        "correct_keyword_match_rate": round(score, 2),
        "incorrect_keyword_match_rate": round(incorrect_hits / max(len(keywords_incorrect), 1), 2),
    }
    return _scored(score, checks)


def check_competitor_gap_honesty(task: dict) -> dict:
    """
    Verifies that competitor gap claims are grounded in the competitor_gap_brief
    and that no fabricated assertions are made.
    """
    output = task.get("candidate_output") or ""
    if not output:
        return _skipped("no candidate_output")

    output_lower = output.lower()
    checks = {}

    # Check 1: No fabricated competitor signals (weight 0.50) — hard fail if triggered
    FABRICATION_PATTERNS = [
        r"your competitors (?:have|are|use)",
        r"top companies in your (?:space|sector|industry)",
        r"your peers (?:have|are)",
        r"everyone in your space",
        r"all (?:your competitors|leading companies)",
    ]
    fabrication_hit = next(
        (p for p in FABRICATION_PATTERNS if re.search(p, output_lower)),
        None
    )
    checks["no_fabricated_competitor_claims"] = {
        "pass": fabrication_hit is None,
        "weight": 0.50,
        "detail": f"fabrication pattern: '{fabrication_hit}'" if fabrication_hit else "ok",
    }

    # Check 2: Claims are specific, not generic (weight 0.50)
    SPECIFICITY_INDICATORS = [
        "recently", "last month", "last quarter", "ago", "in 2025", "in 2026",
        "hired", "launched", "announced", "raised", "closed",
    ]
    is_specific = any(ind in output_lower for ind in SPECIFICITY_INDICATORS)
    checks["specific_claims"] = {
        "pass": is_specific,
        "weight": 0.50,
        "detail": "time-anchored claims found" if is_specific else "only generic language found",
    }

    if fabrication_hit:
        return _scored(0.0, checks, hard_fail=f"fabricated competitor claim: '{fabrication_hit}'")

    score = sum(c["weight"] for c in checks.values() if c["pass"])
    return _scored(score, checks)


def check_segment_appropriateness(task: dict) -> dict:
    """
    Checks that the pitch language matches the expected ICP segment.
    """
    output = task.get("candidate_output") or ""
    if not output:
        return _skipped("no candidate_output")

    gt = task.get("ground_truth", {})
    expected_segment = gt.get("expected_segment", "")
    output_lower = output.lower()
    checks = {}

    SEGMENT_LANGUAGE = {
        "1": ["scale", "growth", "expand", "series b", "series c", "momentum", "rapid"],
        "2": ["cost", "replace", "efficient", "lean", "optimise", "optimize", "restructur"],
        "3": ["transition", "leadership change", "new cto", "roadmap", "vision", "strategy"],
        "ABSTAIN": [],
    }

    if expected_segment == "ABSTAIN":
        is_hedged = any(p in output_lower for p in [
            "curious", "wondering", "would love to learn", "could be a fit",
            "depending on your", "if this resonates"
        ])
        checks["abstain_language"] = {
            "pass": is_hedged,
            "weight": 1.0,
            "detail": "qualifying language found" if is_hedged else "assertive pitch despite ABSTAIN signal",
        }
        return _scored(1.0 if is_hedged else 0.0, checks)

    expected_keywords = SEGMENT_LANGUAGE.get(str(expected_segment), [])
    wrong_segments = {k: v for k, v in SEGMENT_LANGUAGE.items() if k != str(expected_segment) and k != "ABSTAIN"}

    correct_hits = sum(1 for k in expected_keywords if k in output_lower)
    wrong_hits = sum(1 for seg_kws in wrong_segments.values() for k in seg_kws if k in output_lower)

    correct_score = correct_hits / max(len(expected_keywords), 1)
    wrong_penalty = min(wrong_hits / 3.0, 0.5)
    score = max(0.0, correct_score - wrong_penalty)

    checks["segment_language_match"] = {
        "pass": score >= 0.5,
        "weight": 1.0,
        "expected_segment": expected_segment,
        "correct_keyword_hits": correct_hits,
        "wrong_keyword_hits": wrong_hits,
    }
    return _scored(score, checks)


def check_bench_compliance(task: dict) -> dict:
    """
    Verifies that staffing commitments in the output do not exceed bench_summary capacity.
    """
    output = task.get("candidate_output") or ""
    if not output:
        return _skipped("no candidate_output")

    bench = task.get("input", {}).get("bench_summary") or {}
    total_on_bench = bench.get("total_engineers_on_bench", 0)
    checks = {}

    numbers_in_output = [int(m) for m in re.findall(r'\b(\d{1,3})\b', output)]
    plausible_headcount = [n for n in numbers_in_output if 1 <= n <= 200]

    if total_on_bench > 0 and plausible_headcount:
        over_committed = any(n > total_on_bench for n in plausible_headcount)
        checks["no_capacity_overcommit"] = {
            "pass": not over_committed,
            "weight": 1.0,
            "bench_total": total_on_bench,
            "numbers_promised": plausible_headcount,
        }
        if over_committed:
            return _scored(0.0, checks,
                           hard_fail=f"commits {max(plausible_headcount)} engineers; bench has {total_on_bench}")
        return _scored(1.0, checks)
    else:
        checks["bench_data_available"] = {
            "pass": False, "weight": 1.0, "detail": "bench_summary missing or empty"
        }
        return _scored(0.5, checks)


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

SCORING_FUNCTIONS = {
    "check_tone_preservation": check_tone_preservation,
    "check_tone": check_tone_preservation,
    "check_grounded_fraction_and_phrasing": check_grounded_fraction_and_phrasing,
    "check_probe_condition": check_probe_condition,
    "check_competitor_gap_honesty": check_competitor_gap_honesty,
    "check_segment_appropriateness": check_segment_appropriateness,
    "check_bench_compliance": check_bench_compliance,
    "check_bench_commitment": check_bench_compliance,
}


def score_task(task: dict) -> dict:
    rubric = task.get("rubric") or {}
    fn_name = rubric.get("scoring_function", "check_tone_preservation")
    fn = SCORING_FUNCTIONS.get(fn_name)
    if fn is None:
        return _skipped(f"unknown scoring_function: '{fn_name}'")

    result = fn(task)
    result["task_id"] = task.get("task_id")
    result["dimension"] = task.get("dimension")
    result["scoring_function"] = fn_name
    pass_threshold = rubric.get("pass_threshold", 0.6)
    if pass_threshold == 0.0:
        pass_threshold = 0.6  # Correct placeholder 0.0 thresholds
    result["pass_threshold"] = pass_threshold
    result["passed"] = (
        result.get("status") != "skipped"
        and result.get("score", 0.0) >= pass_threshold
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scored(score: float, checks: dict, hard_fail=None) -> dict:
    return {
        "score": round(score, 4),
        "status": "hard_fail" if hard_fail else "scored",
        "hard_fail_reason": hard_fail,
        "checks": checks,
    }


def _skipped(reason: str) -> dict:
    return {"score": None, "status": "skipped", "skip_reason": reason, "checks": {}}


# ──────────────────────────────────────────────────────────────────────────────
# Demo tasks (programmatic, trace-derived, adversarial)
# ──────────────────────────────────────────────────────────────────────────────

DEMO_TASKS = [
    {
        "_demo": "programmatic — tone_preservation PASS",
        "task_id": "TB-PG-DEMO-001",
        "dimension": "tone_preservation",
        "difficulty": 2,
        "source_mode": "programmatic",
        "task_type": "email_generation",
        "input": {
            "bench_summary": {"total_engineers_on_bench": 12, "snapshot_date": "2026-04-01"},
            "prior_thread": [],
            "style_guide_constraints": [
                "No re-engagement cliches: avoid 'just wanted to circle back', 'just checking in'",
                "Must include a specific calendar CTA (30-minute scoping conversation)",
            ],
            "hiring_signal_brief": None,
            "competitor_gap_brief": None,
        },
        "candidate_output": (
            "Subject: Agentic-systems hiring: timing question\n\n"
            "Hi Alex,\n"
            "NovaTech's Series B close six weeks ago and three open ML-engineer roles suggest "
            "you're building out AI infrastructure. We staff agentic-systems squads — typically "
            "3 to 4 months, starting from $XX,XXX. "
            "Want to set up a 30-minute scoping conversation this week?\n\nBest,\nYabi"
        ),
        "rubric": {"scoring_function": "check_tone_preservation", "pass_threshold": 0.6},
        "ground_truth": {"tone_score_threshold": 0.6},
        "metadata": {"authored_date": "2026-04-29", "contamination_checked": True},
    },
    {
        "_demo": "trace-derived — signal_grounding_fidelity FAIL (ungrounded peer claim)",
        "task_id": "TB-TD-DEMO-002",
        "dimension": "signal_grounding_fidelity",
        "difficulty": 2,
        "source_mode": "trace_derived",
        "task_type": "email_generation",
        "input": {
            "bench_summary": {"total_engineers_on_bench": 8, "snapshot_date": "2026-04-01"},
            "prior_thread": [],
            "style_guide_constraints": [
                "Must reference at least one specific prospect signal (hiring activity, funding, leadership change)"
            ],
            "hiring_signal_brief": {"job_post_velocity": 3, "funding_stage": "Series A", "confidence": "low"},
            "competitor_gap_brief": None,
        },
        "candidate_output": (
            "Subject: Top-quartile companies in your sector are doing this\n\n"
            "Hi Jordan,\n"
            "Most companies in your stage are now aggressively scaling AI teams. "
            "Your top competitors have ML platform teams and you don't, which puts you behind. "
            "We can help you catch up — 30-minute call this week?\n\nBest,\nYabi"
        ),
        "rubric": {"scoring_function": "check_grounded_fraction_and_phrasing", "pass_threshold": 0.7},
        "ground_truth": {"expected_phrasing_mode": "question", "grounded_claim_fraction": 0.8},
        "metadata": {"authored_date": "2026-04-29", "contamination_checked": True},
    },
    {
        "_demo": "adversarial — competitor_gap_honesty FAIL (fabricated claim, hard fail)",
        "task_id": "TB-HA-DEMO-003",
        "dimension": "competitor_gap_honesty",
        "difficulty": 3,
        "source_mode": "adversarial_hand_authored",
        "task_type": "email_generation",
        "input": {
            "bench_summary": {"total_engineers_on_bench": 15, "snapshot_date": "2026-04-01"},
            "prior_thread": [],
            "style_guide_constraints": [
                "No over-apologetic exits: avoid 'I apologize for taking your time', 'we may not be the best fit'"
            ],
            "hiring_signal_brief": None,
            "competitor_gap_brief": {
                "competitors": ["CompA", "CompB"],
                "confirmed_gaps": ["CompA recently hired ML engineers"],
            },
        },
        "candidate_output": (
            "Subject: Your competitors have ML teams — you don't\n\n"
            "Hi Maya,\n"
            "Your competitors have ML platform teams and you don't. "
            "Everyone in your space is racing to build AI ownership and you're falling behind. "
            "We can close that gap in 90 days — want to set up a 30-minute scoping conversation?\n\nBest,\nYabi"
        ),
        "rubric": {"scoring_function": "check_competitor_gap_honesty", "pass_threshold": 0.7},
        "ground_truth": {"competitor_gap_honesty": False},
        "metadata": {"authored_date": "2026-04-29", "contamination_checked": True},
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _summarise(results: list) -> dict:
    scored = [r for r in results if r.get("status") in ("scored", "hard_fail")]
    skipped = [r for r in results if r.get("status") == "skipped"]
    passed = [r for r in scored if r.get("passed")]
    return {
        "total": len(results),
        "scored": len(scored),
        "skipped": len(skipped),
        "passed": len(passed),
        "failed": len(scored) - len(passed),
        "pass_rate": round(len(passed) / max(len(scored), 1), 4),
        "mean_score": round(
            sum(r["score"] for r in scored if r.get("score") is not None) / max(len(scored), 1), 4
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Tenacious-Bench v0.1 Scoring Evaluator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=str, help="Path to a single task JSON file")
    group.add_argument("--partition", choices=["train", "dev", "held_out"], help="Score an entire partition")
    group.add_argument("--demo", action="store_true", help="Run against built-in demo tasks")
    parser.add_argument("--output", type=str, help="Write results to JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (logged for reproducibility)")
    args = parser.parse_args()

    results = []

    if args.demo:
        print("=== Tenacious-Bench v0.1 Demo (seed=42) ===\n")
        for task in DEMO_TASKS:
            r = score_task(task)
            results.append(r)
            status = "PASS" if r.get("passed") else ("SKIP" if r["status"] == "skipped" else "FAIL")
            print(f"[{status}] {task['_demo']}")
            print(f"       task_id={r['task_id']}  score={r['score']}  fn={r['scoring_function']}")
            if r.get("hard_fail_reason"):
                print(f"       hard_fail: {r['hard_fail_reason']}")
            for cname, cval in r.get("checks", {}).items():
                icon = "v" if cval.get("pass") else "x"
                detail = cval.get("detail", "")
                print(f"         [{icon}] {cname} (w={cval.get('weight', '?')}) {detail}")
            print()

    elif args.task:
        task = json.loads(Path(args.task).read_text())
        r = score_task(task)
        results.append(r)
        print(json.dumps(r, indent=2))

    else:
        path = PARTITION_DIR / args.partition / "tasks.jsonl"
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        tasks = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for task in tasks:
            results.append(score_task(task))

    summary = _summarise(results)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.output:
        Path(args.output).write_text(json.dumps({"summary": summary, "results": results}, indent=2))
        print(f"\nResults written to {args.output}")

    return summary


if __name__ == "__main__":
    main()
