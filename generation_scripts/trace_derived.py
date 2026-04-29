#!/usr/bin/env python3
"""
Trace-derived task generator for Tenacious-Bench.

Converts Week 10 trace_log.jsonl into evaluation tasks by:
1. Redacting real prospect names -> pseudonyms
2. Restructuring trace entries into (input, candidate_output) pairs
3. Labeling ground-truth rubric scores from trace outcomes

Source: Week 10 trace_log.jsonl (tasks 0-4, same as referenced in audit_memo.md)
Key trace IDs cited in methodology.md: task_id 0 (bench commitment), task_id 2 (signal over-claiming)

Output: tenacious_bench_v0.1/train/trace_derived_tasks.jsonl
"""

import json
import re
import hashlib
from pathlib import Path

BENCH_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = BENCH_ROOT / "tenacious_bench_v0.1"

# Pseudonym mapping (keeps PII out of benchmark)
NAME_MAP = {
    # Real names -> pseudonyms (examples; extend from actual trace data)
    "john": "Jordan", "jane": "Alex", "mike": "Morgan", "sarah": "Sam",
    "david": "Casey", "emma": "Riley", "james": "Taylor", "lisa": "Drew",
}

COMPANY_MAP = {
    # Real company names -> synthetic (extend from actual trace data)
    "acme": "NovaTech", "widget co": "DataFlow", "startup inc": "ScaleCo",
}


def pseudonymise(text: str) -> str:
    """Replace real names with pseudonyms."""
    result = text
    for real, pseudo in NAME_MAP.items():
        result = re.sub(rf'\b{real}\b', pseudo, result, flags=re.IGNORECASE)
    for real, pseudo in COMPANY_MAP.items():
        result = re.sub(rf'\b{real}\b', pseudo, result, flags=re.IGNORECASE)
    return result


def trace_entry_to_task(trace: dict, task_id: str) -> dict | None:
    """
    Convert a single trace entry to a benchmark task.
    Trace format (Week 10): {task_id, agent_input, agent_output, probe_id, outcome, dimension}
    """
    if not trace.get("agent_output"):
        return None

    raw_input = pseudonymise(json.dumps(trace.get("agent_input", {})))
    raw_output = pseudonymise(trace.get("agent_output", ""))

    # Infer dimension from trace
    dimension = trace.get("dimension") or infer_dimension_from_trace(trace)
    if not dimension:
        return None

    rubric_map = {
        "signal_grounding_fidelity": ("check_grounded_fraction_and_phrasing", 0.7),
        "bench_commitment_honesty": ("check_bench_compliance", 1.0),
        "competitor_gap_honesty": ("check_competitor_gap_honesty", 0.7),
        "icp_segment_appropriateness": ("check_segment_appropriateness", 1.0),
        "tone_preservation": ("check_tone_preservation", 0.6),
    }
    fn, threshold = rubric_map.get(dimension, ("check_tone_preservation", 0.6))

    # Build input from trace
    agent_input = trace.get("agent_input", {})
    inp = {
        "bench_summary": agent_input.get("bench_summary", {"total_engineers_on_bench": 10}),
        "prior_thread": agent_input.get("prior_thread", []),
        "style_guide_constraints": agent_input.get("style_guide_constraints", [
            "No re-engagement clichés: avoid 'just wanted to circle back', 'just checking in'",
            "Must include a specific calendar CTA (30-minute scoping conversation)"
        ]),
        "hiring_signal_brief": agent_input.get("hiring_signal_brief"),
        "competitor_gap_brief": agent_input.get("competitor_gap_brief"),
    }

    # Determine difficulty from probe outcome
    probe_outcome = trace.get("outcome", "pass")
    difficulty = 3 if probe_outcome == "fail" else 2

    # Ground truth from trace labels
    gt = {}
    if dimension == "signal_grounding_fidelity":
        gt["expected_phrasing_mode"] = trace.get("expected_phrasing_mode", "assert")
    elif dimension == "icp_segment_appropriateness":
        gt["expected_segment"] = trace.get("expected_segment", "1")
    elif dimension == "bench_commitment_honesty":
        gt["bench_compliance"] = probe_outcome == "pass"
    elif dimension == "competitor_gap_honesty":
        gt["competitor_gap_honesty"] = probe_outcome == "pass"
    elif dimension == "tone_preservation":
        gt["tone_score_threshold"] = 0.6

    return {
        "task_id": task_id,
        "dimension": dimension,
        "difficulty": difficulty,
        "source_mode": "trace_derived",
        "task_type": "staffing_commitment_response" if dimension == "bench_commitment_honesty" else "email_generation",
        "input": inp,
        "candidate_output": raw_output,
        "correct_output": trace.get("correct_output", ""),
        "incorrect_output": trace.get("incorrect_output", ""),
        "ground_truth": gt,
        "rubric": {
            "scoring_function": fn,
            "pass_threshold": threshold,
            "dimensions_scored": [dimension],
            "max_score": 1.0,
        },
        "metadata": {
            "authored_date": "2026-04-29",
            "source_trace_id": str(trace.get("task_id", "")),
            "source_probe_id": trace.get("probe_id"),
            "contamination_checked": False,
        },
    }


def infer_dimension_from_trace(trace: dict) -> str | None:
    """Infer dimension from trace content if not explicitly set."""
    text = json.dumps(trace).lower()
    if "hiring_signal_brief" in text or "grounded" in text:
        return "signal_grounding_fidelity"
    if "bench_summary" in text or "over-commit" in text:
        return "bench_commitment_honesty"
    if "competitor_gap_brief" in text or "competitor" in text:
        return "competitor_gap_honesty"
    if "icp" in text or "segment" in text:
        return "icp_segment_appropriateness"
    if "tone" in text or "style_guide" in text:
        return "tone_preservation"
    return None


def process_trace_log(trace_log_path: Path) -> list:
    """Process the Week 10 trace_log.jsonl file."""
    if not trace_log_path.exists():
        print(f"WARNING: {trace_log_path} not found. Returning empty task list.")
        print("         Place your Week 10 trace_log.jsonl at this path to generate trace-derived tasks.")
        return []

    tasks = []
    counter = 7000

    with open(trace_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
                task = trace_entry_to_task(trace, f"TB-TD-{counter:04d}")
                if task:
                    tasks.append(task)
                    counter += 1
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Skipping trace: {e}")

    print(f"Generated {len(tasks)} trace-derived tasks from {trace_log_path}")
    return tasks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-log", default=str(BENCH_ROOT / "trace_log.jsonl"),
                        help="Path to Week 10 trace_log.jsonl")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "train" / "trace_derived_new.jsonl"))
    args = parser.parse_args()

    tasks = process_trace_log(Path(args.trace_log))
    if tasks:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
        print(f"Written {len(tasks)} tasks to {out}")
    else:
        print("No tasks generated. Check trace_log.jsonl path.")
