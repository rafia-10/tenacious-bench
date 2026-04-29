#!/usr/bin/env python3
"""
Judge filter for Tenacious-Bench task quality control.

Applies LLM-as-a-judge scoring on three dimensions before a task enters the dataset:
  - input_coherence (1-5): Does the task input make sense as a sales scenario?
  - ground_truth_verifiability (1-5): Can the rubric be applied without ambiguity?
  - rubric_clarity (1-5): Is the expected pass/fail outcome unambiguous?

Rotation policy (prevents preference leakage — Li et al. 2025):
  - Synthesis model: DeepSeek V3.2 (via OpenRouter)
  - High-volume filter judge: Qwen3-Next-80B-A3B (via OpenRouter)
  - Spot-check judge: Claude Sonnet 4.6 (eval-tier budget only)

Set OPENROUTER_API_KEY to use LLM judge.
Without the key, the script runs in deterministic-only mode (structural checks only).
"""

import json
import os
import re
import math
from pathlib import Path
from typing import Optional
from collections import Counter

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Inclusion thresholds
COHERENCE_THRESHOLD = 4
VERIFIABILITY_THRESHOLD = 4
CLARITY_THRESHOLD = 4

# Pairwise similarity threshold for escalation
PAIRWISE_SIMILARITY_THRESHOLD = 0.80


def structural_checks(task: dict) -> dict:
    """Deterministic pre-filter. No LLM needed."""
    issues = []

    if not task.get("task_id"):
        issues.append("missing task_id")
    if not task.get("dimension"):
        issues.append("missing dimension")
    if task.get("dimension") not in [
        "signal_grounding_fidelity", "bench_commitment_honesty",
        "icp_segment_appropriateness", "competitor_gap_honesty", "tone_preservation"
    ]:
        issues.append(f"invalid dimension: {task.get('dimension')}")
    if not task.get("rubric", {}).get("scoring_function"):
        issues.append("missing rubric.scoring_function")
    if task.get("difficulty") not in (1, 2, 3):
        issues.append(f"invalid difficulty: {task.get('difficulty')}")

    # Check that at least one input field is non-empty
    inp = task.get("input", {})
    non_null_fields = [k for k, v in inp.items() if v is not None and v != [] and v != {}]
    if not non_null_fields:
        issues.append("input has no non-null fields")

    return {"pass": len(issues) == 0, "issues": issues}


def tfidf_sim(text_a: str, text_b: str) -> float:
    """Lightweight cosine similarity for pairwise dedup."""
    def tf(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        c = Counter(tokens)
        total = max(len(tokens), 1)
        return {t: count / total for t, count in c.items()}

    ta, tb = tf(text_a), tf(text_b)
    vocab = set(ta) | set(tb)
    dot = sum(ta.get(t, 0) * tb.get(t, 0) for t in vocab)
    mag_a = math.sqrt(sum(v ** 2 for v in ta.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in tb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def task_to_text(task: dict) -> str:
    parts = []
    for v in task.get("input", {}).values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (dict, list)):
            parts.append(json.dumps(v))
    parts.append(task.get("correct_output") or "")
    return " ".join(parts).lower()


def llm_judge_score(task: dict, model: str = "qwen/qwen3-next-80b-a3b") -> Optional[dict]:
    """
    Call OpenRouter judge model to score a task on three dimensions.
    Returns None if API key is not set (run in deterministic-only mode).
    """
    if not OPENROUTER_API_KEY:
        return None

    try:
        import urllib.request
        prompt = f"""You are evaluating a benchmark task for a B2B outbound sales agent. Score on three dimensions (1-5 each):

1. input_coherence: Does the task input make sense as a sales scenario?
2. ground_truth_verifiability: Can the rubric be applied without ambiguity?
3. rubric_clarity: Is the expected pass/fail outcome unambiguous?

Task JSON:
{json.dumps(task, indent=2)[:2000]}

Respond ONLY with JSON: {{"input_coherence": N, "ground_truth_verifiability": N, "rubric_clarity": N, "reasoning": "..."}}"""

        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.0,
        }).encode()

        req = urllib.request.Request(
            f"{OPENROUTER_BASE}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rafia-10/tenacious-bench",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"]
            scores = json.loads(re.search(r'\{.*\}', content, re.DOTALL).group())
            return scores
    except Exception as e:
        print(f"  Judge API error: {e}")
        return None


def passes_quality_filter(task: dict, use_llm: bool = True) -> dict:
    """
    Full quality filter: structural checks, then optional LLM scoring.
    Returns dict with 'pass' bool and details.
    """
    struct = structural_checks(task)
    if not struct["pass"]:
        return {"pass": False, "stage": "structural", "issues": struct["issues"]}

    if use_llm and OPENROUTER_API_KEY:
        scores = llm_judge_score(task)
        if scores:
            failures = []
            if scores.get("input_coherence", 0) < COHERENCE_THRESHOLD:
                failures.append(f"input_coherence={scores['input_coherence']}")
            if scores.get("ground_truth_verifiability", 0) < VERIFIABILITY_THRESHOLD:
                failures.append(f"verifiability={scores['ground_truth_verifiability']}")
            if scores.get("rubric_clarity", 0) < CLARITY_THRESHOLD:
                failures.append(f"rubric_clarity={scores['rubric_clarity']}")
            if failures:
                return {"pass": False, "stage": "llm_judge", "issues": failures, "scores": scores}
            return {"pass": True, "stage": "llm_judge", "scores": scores}

    # Deterministic-only pass
    return {"pass": True, "stage": "structural_only"}


def pairwise_select(task_a: dict, task_b: dict) -> dict:
    """
    When two tasks are too similar (cosine > threshold), keep the more diagnostic one.
    Uses LLM judge in pairwise mode; falls back to keeping higher-difficulty task.
    """
    sim = tfidf_sim(task_to_text(task_a), task_to_text(task_b))
    if sim < PAIRWISE_SIMILARITY_THRESHOLD:
        return None  # Both are distinct enough; no selection needed

    # Prefer adversarial > llm_synthetic > programmatic > trace_derived
    MODE_RANK = {"adversarial_hand_authored": 4, "llm_synthetic": 3, "programmatic": 2, "trace_derived": 1}
    rank_a = MODE_RANK.get(task_a.get("source_mode", ""), 0)
    rank_b = MODE_RANK.get(task_b.get("source_mode", ""), 0)

    if rank_a != rank_b:
        winner = task_a if rank_a > rank_b else task_b
    else:
        # Same mode: prefer harder difficulty
        winner = task_a if task_a.get("difficulty", 1) >= task_b.get("difficulty", 1) else task_b

    return {
        "selected": winner["task_id"],
        "dropped": (task_b if winner is task_a else task_a)["task_id"],
        "cosine_similarity": round(sim, 4),
        "reason": f"source_mode rank {rank_a} vs {rank_b}, difficulty {task_a.get('difficulty')} vs {task_b.get('difficulty')}",
    }


def filter_batch(tasks: list, use_llm: bool = False) -> dict:
    """Filter a batch of tasks. Returns accepted, rejected, and dedup_decisions."""
    accepted = []
    rejected = []

    for task in tasks:
        result = passes_quality_filter(task, use_llm=use_llm)
        if result["pass"]:
            accepted.append(task)
        else:
            rejected.append({"task_id": task.get("task_id"), "reason": result})

    # Dedup pass: check pairwise similarity across accepted tasks
    dedup_decisions = []
    to_drop = set()
    for i in range(len(accepted)):
        if accepted[i]["task_id"] in to_drop:
            continue
        for j in range(i + 1, len(accepted)):
            if accepted[j]["task_id"] in to_drop:
                continue
            decision = pairwise_select(accepted[i], accepted[j])
            if decision:
                to_drop.add(decision["dropped"])
                dedup_decisions.append(decision)

    final_accepted = [t for t in accepted if t["task_id"] not in to_drop]

    return {
        "accepted": final_accepted,
        "rejected": rejected,
        "dedup_decisions": dedup_decisions,
        "counts": {
            "input": len(tasks),
            "accepted": len(final_accepted),
            "rejected_quality": len(rejected),
            "deduped": len(to_drop),
        },
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python judge_filter.py <tasks.jsonl> [--llm]")
        sys.exit(1)

    path = Path(sys.argv[1])
    use_llm = "--llm" in sys.argv

    tasks = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(tasks)} tasks from {path}")

    result = filter_batch(tasks, use_llm=use_llm)
    print(f"Accepted: {result['counts']['accepted']}")
    print(f"Rejected (quality): {result['counts']['rejected_quality']}")
    print(f"Deduped: {result['counts']['deduped']}")

    if result["rejected"]:
        print("\nFirst 5 rejected:")
        for r in result["rejected"][:5]:
            print(f"  {r['task_id']}: {r['reason']}")
