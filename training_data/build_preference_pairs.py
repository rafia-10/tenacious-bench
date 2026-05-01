#!/usr/bin/env python3
"""
Day 4 — Build Preference Pairs for ORPO training.

Strategy:
  - 111 failing train tasks  → candidate_output = REJECTED; generate CHOSEN via DeepSeek
  - 41  passing train tasks  → candidate_output = CHOSEN;  generate REJECTED via DeepSeek
  - Filter: chosen passes rubric, rejected fails rubric, cosine_sim < 0.92
  - Format: Qwen 2.5 chat template for ORPO
  - Log every decision to generation_log.jsonl
  - Write cost to cost_log.csv

Model rotation:
  - Generator:  deepseek/deepseek-chat-v3-0324   (via OpenRouter)
  - Judge:      scoring_evaluator.py (machine-verifiable, no LLM bias)
  - Similarity: sentence-transformers/all-MiniLM-L6-v2
"""

import json
import os
import sys
import time
import csv
import hashlib
import datetime
from pathlib import Path

import requests
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scoring_evaluator import score_task

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324"
EMBED_MODEL = "all-MiniLM-L6-v2"
COSINE_THRESHOLD = 0.92

OUTPUT_DIR = Path(__file__).parent
LOG_PATH = OUTPUT_DIR / "generation_log.jsonl"
RAW_PATH = OUTPUT_DIR / "preference_pairs_raw.jsonl"
FINAL_PATH = OUTPUT_DIR / "preference_pairs.jsonl"
COST_LOG = ROOT / "cost_log.csv"

JUDGE_SYSTEM = (
    "You are evaluating outbound sales emails for Tenacious Consulting. "
    "Score the following output on signal-grounding fidelity, bench commitment honesty, "
    "ICP segment appropriateness, and Tenacious tone adherence."
)

QWEN_SYSTEM = JUDGE_SYSTEM


def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def call_openrouter(messages: list, model: str = DEEPSEEK_MODEL, max_tokens: int = 600) -> tuple[str, dict]:
    """Returns (text, usage_dict). Raises on failure."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rafiakedir/tenacious-bench",
        "X-Title": "tenacious-bench-day4",
    }
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=body, timeout=90)
    latency_ms = int((time.time() - t0) * 1000)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    usage["latency_ms"] = latency_ms
    return text, usage


def log_cost(model: str, task_id: str, bucket: str, usage: dict):
    """Append one row to cost_log.csv."""
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "model": model,
        "task_id": task_id,
        "bucket": bucket,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "latency_ms": usage.get("latency_ms", 0),
        "est_cost_usd": round(
            usage.get("prompt_tokens", 0) * 0.00000027
            + usage.get("completion_tokens", 0) * 0.00000110,
            6,
        ),
    }
    write_header = not COST_LOG.exists()
    with open(COST_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_log(entry: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def build_task_input_text(task: dict) -> str:
    """Flatten task input into a single text block for the model prompt."""
    inp = task.get("input", {})
    parts = []
    hsb = inp.get("hiring_signal_brief")
    if hsb:
        parts.append(f"HIRING SIGNAL BRIEF:\n{json.dumps(hsb, indent=2)}")
    bs = inp.get("bench_summary")
    if bs:
        parts.append(f"BENCH SUMMARY:\n{json.dumps(bs, indent=2)}")
    cg = inp.get("competitor_gap_brief")
    if cg:
        parts.append(f"COMPETITOR GAP BRIEF:\n{json.dumps(cg, indent=2)}")
    sg = inp.get("style_guide_constraints")
    if sg:
        parts.append(f"STYLE GUIDE CONSTRAINTS:\n" + "\n".join(f"- {c}" for c in sg))
    pt = inp.get("prior_thread")
    if pt:
        parts.append("PRIOR THREAD:\n" + "\n".join(
            f"[{m.get('role','?')}] {m.get('content','')}" for m in pt
        ))
    task_type = task.get("task_type", "email_generation")
    dimension = task.get("dimension", "")
    parts.insert(0, f"TASK TYPE: {task_type}\nEVALUATION DIMENSION: {dimension}")
    co = task.get("correct_output", "")
    io = task.get("incorrect_output", "")
    if co or io:
        parts.append(f"PASSING BEHAVIOR: {co}\nFAILING BEHAVIOR: {io}")
    return "\n\n".join(parts)


def _dimension_chosen_instructions(task: dict) -> str:
    """Build rubric-grounded language instructions for generating a CHOSEN output."""
    dimension = task.get("dimension", "")
    gt = task.get("ground_truth") or {}
    inp = task.get("input", {})
    bench = inp.get("bench_summary") or {}
    bench_total = bench.get("total_engineers_on_bench", 0)

    if dimension == "signal_grounding_fidelity":
        mode = gt.get("expected_phrasing_mode", "assert")
        if mode == "assert":
            return (
                "RUBRIC CHECKS TO PASS:\n"
                "A) USE ASSERT LANGUAGE — your email MUST contain at least one of these exact words or phrases: "
                "'recently', 'secured', 'closed', 'has ', 'currently', 'is hiring', 'shows'. "
                "Example: 'TalentBridge recently closed a Series A' or 'CloudEdge currently has 8 open ML roles'.\n"
                "B) NO UNGROUNDED CLAIMS — do NOT write: 'top quartile of your sector', "
                "'most companies in your stage', 'peer companies at your stage', "
                "'industry standard', 'typical for companies like yours'. "
                "Reference only facts from the HIRING SIGNAL BRIEF above."
            )
        else:
            return (
                "RUBRIC CHECKS TO PASS:\n"
                "A) USE QUESTION LANGUAGE — your email MUST contain at least one of these exact phrases: "
                "'curious how', 'wondering if', 'would be curious', 'are you', 'how are you'. "
                "EXAMPLE: 'I'm curious how you're thinking about ML headcount' or 'Wondering if AI hiring is on your roadmap'.\n"
                "B) AVOID ALL ASSERT INDICATORS — do NOT use these words anywhere in your output: "
                "'recently', 'secured', 'closed', 'currently', 'is hiring', 'shows'. "
                "Also avoid the word 'has' (use 'your' or 'with' instead: 'your 3 open roles' not 'has 3 open roles').\n"
                "C) NO UNGROUNDED CLAIMS — signal confidence is low. Hedge everything. "
                "Do not reference industry benchmarks or peer comparisons."
            )

    elif dimension == "bench_commitment_honesty":
        return (
            f"RUBRIC CHECKS TO PASS:\n"
            f"A) NEVER commit to more than {bench_total} engineers total. "
            f"The bench_summary shows {bench_total} engineers on bench. "
            f"If you mention a number of engineers, it must be ≤ {bench_total}.\n"
            f"B) Be specific about bench capacity — reference the actual number or a subset of it."
        )

    elif dimension == "tone_preservation":
        return (
            "RUBRIC CHECKS TO PASS:\n"
            "A) NO BANNED RE-ENGAGEMENT PHRASES — never use: 'just wanted to circle back', "
            "'just checking in', 'touching base', 'following up on my last', 'as i mentioned', "
            "'per my last email', 'hope this finds you well', 'i wanted to reach out', "
            "'hope you're doing well', 'hope all is well'.\n"
            "B) NO OVER-APOLOGETIC LANGUAGE — never use: 'i apologize for taking your time', "
            "'sorry to bother', 'we may not be the best fit', 'if you're not interested'.\n"
            "C) INCLUDE CALENDAR CTA — your email MUST contain at least one of: "
            "'30-minute', '30 min', 'scoping conversation', 'calendar', 'book a time', "
            "'schedule a call', 'set up a call', 'grab 30', 'discovery call'.\n"
            "D) DIRECT OPENER — do not start the body with 'I hope', 'I wanted', 'just ', "
            "'following', 'I'm reaching out'."
        )

    elif dimension == "competitor_gap_honesty":
        return (
            "RUBRIC CHECKS TO PASS:\n"
            "A) NO FABRICATED COMPETITOR CLAIMS — never write: 'your competitors have', "
            "'top companies in your space', 'your peers have', 'everyone in your space', "
            "'all your competitors'. Only reference specific competitors named in the COMPETITOR GAP BRIEF.\n"
            "B) USE SPECIFIC TIME-ANCHORED CLAIMS — include at least one of: "
            "'recently', 'last month', 'last quarter', 'ago', 'in 2025', 'in 2026', "
            "'hired', 'launched', 'announced', 'raised', 'closed'."
        )

    elif dimension == "icp_segment_appropriateness":
        seg = gt.get("expected_segment", "1")
        seg_words = {
            "1": "scale, growth, expand, Series B, Series C, momentum, rapid",
            "2": "cost, replace, efficient, lean, optimise, optimize, restructur",
            "3": "transition, leadership change, new CTO, roadmap, vision, strategy",
            "ABSTAIN": "curious, wondering, would love to learn, could be a fit, depending on your, if this resonates",
        }
        return (
            f"RUBRIC CHECKS TO PASS:\n"
            f"A) USE SEGMENT {seg} LANGUAGE — your email MUST contain words from this list: "
            f"{seg_words.get(str(seg), 'growth, scale')}.\n"
            f"B) DO NOT use language from other segments (e.g., if Segment 1, avoid 'cost', 'replace', 'lean')."
        )

    return (
        "RUBRIC CHECKS TO PASS:\n"
        "A) Use assertive, specific, time-anchored claims from the brief.\n"
        "B) Include a 30-minute scoping calendar CTA.\n"
        "C) No banned phrases, no ungrounded claims."
    )


def generate_chosen(task: dict, rejected_output: str) -> tuple[str | None, dict]:
    """Ask DeepSeek to rewrite the rejected output so it passes all rubric dimensions."""
    task_text = build_task_input_text(task)
    rubric_instructions = _dimension_chosen_instructions(task)

    prompt = f"""You are rewriting a failing B2B outbound sales email for Tenacious Consulting so it passes specific rubric checks.

TASK CONTEXT:
{task_text}

{rubric_instructions}

GENERAL TENACIOUS STYLE RULES (always apply):
- Include a 30-minute scoping calendar CTA
- No re-engagement clichés ("just wanted to circle back", "touching base", "hope you're doing well")
- No over-apologetic language ("sorry to bother", "I apologize for taking your time")
- Direct opener — don't start with "I hope", "I wanted to", "I'm reaching out"
- Under 150 words total

FAILING OUTPUT (rewrite this):
{rejected_output}

Write ONLY the rewritten email (Subject line + body). No explanation.
"""
    messages = [
        {"role": "system", "content": "You write precise B2B sales emails that pass specific rubric checks. Respond with only the email."},
        {"role": "user", "content": prompt},
    ]
    try:
        text, usage = call_openrouter(messages, DEEPSEEK_MODEL, max_tokens=400)
        log_cost(DEEPSEEK_MODEL, task["task_id"], "day4_chosen_generation", usage)
        return text, usage
    except Exception as e:
        return None, {"error": str(e)}


def generate_rejected(task: dict, chosen_output: str) -> tuple[str | None, dict]:
    """Ask DeepSeek to produce a failing version of the passing output."""
    task_text = build_task_input_text(task)
    dimension = task.get("dimension", "")
    rubric = task.get("rubric", {})
    scored_dims = rubric.get("dimensions_scored", [dimension])

    FAILURE_MODES = {
        "signal_grounding_fidelity": "Make the output assert claims that are NOT supported by the hiring signal brief — use generic phrases like 'most companies in your stage' or 'top quartile of your sector'. Assert high-confidence language even though signals are low-confidence.",
        "bench_commitment_honesty": "Make the output commit to a number of engineers that EXCEEDS the bench_summary total. For example if bench has 8 engineers, promise 15.",
        "icp_segment_appropriateness": "Make the output use entirely wrong segment language — if it should be cost-restructuring language, use enthusiastic growth/scale language instead.",
        "competitor_gap_honesty": "Make the output assert fabricated competitor claims like 'your top competitors have ML platform teams and you don't' or 'everyone in your space is racing ahead' without grounding in the competitor_gap_brief.",
        "tone_preservation": "Make the output use banned re-engagement clichés: include 'just wanted to circle back' or 'touching base' or 'hope you're doing well'. Make it over-apologetic: add 'I apologize for taking your time' or 'we may not be the best fit'. Remove the calendar CTA.",
    }

    failure_instruction = FAILURE_MODES.get(
        dimension,
        "Introduce specific rubric failures: use generic ungrounded claims, add apologetic phrases, or remove the calendar CTA."
    )

    prompt = f"""You are generating a FAILING B2B outbound sales email that looks plausible but fails specific rubric checks.

TASK CONTEXT:
{task_text}

PASSING OUTPUT (for reference of what good looks like):
{chosen_output}

DIMENSION TO FAIL: {dimension}
RUBRIC DIMENSIONS: {', '.join(scored_dims)}

HOW TO MAKE IT FAIL:
{failure_instruction}

INSTRUCTIONS:
1. Write an output that LOOKS like a reasonable sales email but fails the specified dimension.
2. Keep it plausible — subtle failures are better than obvious ones.
3. Write the complete email (Subject line + body). Keep it under 150 words.

Write ONLY the failing email. No explanation, no commentary.
"""
    messages = [
        {"role": "system", "content": "You are generating failing sales email examples for a benchmark dataset."},
        {"role": "user", "content": prompt},
    ]
    try:
        text, usage = call_openrouter(messages, DEEPSEEK_MODEL, max_tokens=400)
        log_cost(DEEPSEEK_MODEL, task["task_id"], "day4_rejected_generation", usage)
        return text, usage
    except Exception as e:
        return None, {"error": str(e)}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def format_orpo_pair(task: dict, chosen: str, rejected: str) -> dict:
    """Format as Qwen 2.5 chat template for ORPO training."""
    task_text = build_task_input_text(task)
    prompt = (
        f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{task_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "prompt": prompt,
        "chosen": f"{chosen}<|im_end|>",
        "rejected": f"{rejected}<|im_end|>",
        "task_id": task.get("task_id"),
        "dimension": task.get("dimension"),
    }


def main():
    _load_env()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    # Use sklearn TF-IDF cosine similarity (torch not available in this env)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    embed_model = None
    print("Using TF-IDF cosine similarity (sklearn)")

    # Load all train tasks
    train_tasks = []
    with open(ROOT / "tenacious_bench_v0.1/train/tasks.jsonl") as f:
        for line in f:
            train_tasks.append(json.loads(line))

    print(f"Loaded {len(train_tasks)} train tasks")

    # Score each task to determine if candidate_output is passing or failing
    scored = []
    for task in train_tasks:
        result = score_task(task)
        scored.append((task, result))

    passing_tasks = [(t, r) for t, r in scored if r.get("passed")]
    failing_tasks = [(t, r) for t, r in scored if not r.get("passed")]
    print(f"Passing: {len(passing_tasks)}, Failing: {len(failing_tasks)}")

    # Clear output files
    for p in [LOG_PATH, RAW_PATH, FINAL_PATH]:
        p.unlink(missing_ok=True)

    raw_pairs = []
    final_pairs = []

    # ──────────────────────────────────────────────
    # BATCH 1: Failing tasks → candidate_output = REJECTED, generate CHOSEN
    # ──────────────────────────────────────────────
    print(f"\n--- BATCH 1: Generating CHOSEN outputs for {len(failing_tasks)} failing tasks ---")
    for i, (task, base_score) in enumerate(failing_tasks):
        task_id = task["task_id"]
        rejected = task["candidate_output"]
        print(f"  [{i+1}/{len(failing_tasks)}] {task_id} (score={base_score.get('score')})", end="", flush=True)

        chosen, usage = generate_chosen(task, rejected)
        if chosen is None:
            append_log({"task_id": task_id, "batch": "failing→chosen", "action": "discard", "reason": f"generation failed: {usage.get('error')}"})
            print(" GENERATION_FAILED")
            continue

        # Score chosen
        test_task = {**task, "candidate_output": chosen}
        chosen_score = score_task(test_task)
        raw_pairs.append({
            "task_id": task_id,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": chosen_score.get("score"),
            "chosen_passed": chosen_score.get("passed"),
            "rejected_score": base_score.get("score"),
            "rejected_passed": False,
            "source": "failing_task_generated_chosen",
            "dimension": task.get("dimension"),
        })
        with open(RAW_PATH, "a") as f:
            f.write(json.dumps(raw_pairs[-1]) + "\n")

        # Filter checks
        if not chosen_score.get("passed"):
            append_log({
                "task_id": task_id, "batch": "failing→chosen", "action": "discard",
                "reason": f"chosen failed rubric (score={chosen_score.get('score')} < threshold={chosen_score.get('pass_threshold')})",
                "chosen_score": chosen_score.get("score"),
                "rejected_score": base_score.get("score"),
            })
            print(f" CHOSEN_FAILED(score={chosen_score.get('score')})")
            continue

        # Cosine similarity check (TF-IDF)
        vect = TfidfVectorizer().fit([chosen, rejected])
        mat = vect.transform([chosen, rejected])
        sim = float(sk_cosine(mat[0], mat[1])[0, 0])

        if sim >= COSINE_THRESHOLD:
            append_log({
                "task_id": task_id, "batch": "failing→chosen", "action": "discard",
                "reason": f"cosine_similarity={sim:.3f} >= {COSINE_THRESHOLD} (too similar)",
                "cosine_sim": sim,
            })
            print(f" TOO_SIMILAR(sim={sim:.3f})")
            continue

        # Pair passes all filters
        pair = format_orpo_pair(task, chosen, rejected)
        final_pairs.append(pair)
        with open(FINAL_PATH, "a") as f:
            f.write(json.dumps(pair) + "\n")
        append_log({
            "task_id": task_id, "batch": "failing→chosen", "action": "keep",
            "chosen_score": chosen_score.get("score"),
            "rejected_score": base_score.get("score"),
            "cosine_sim": round(sim, 4),
            "source": "failing_task_generated_chosen",
        })
        print(f" OK(chosen={chosen_score.get('score'):.2f}, sim={sim:.3f})")
        time.sleep(0.3)

    # ──────────────────────────────────────────────
    # BATCH 2: Passing tasks → candidate_output = CHOSEN, generate REJECTED
    # ──────────────────────────────────────────────
    print(f"\n--- BATCH 2: Generating REJECTED outputs for {len(passing_tasks)} passing tasks ---")
    for i, (task, base_score) in enumerate(passing_tasks):
        task_id = task["task_id"]
        chosen = task["candidate_output"]
        print(f"  [{i+1}/{len(passing_tasks)}] {task_id} (score={base_score.get('score')})", end="", flush=True)

        rejected, usage = generate_rejected(task, chosen)
        if rejected is None:
            append_log({"task_id": task_id, "batch": "passing→rejected", "action": "discard", "reason": f"generation failed: {usage.get('error')}"})
            print(" GENERATION_FAILED")
            continue

        # Score rejected — must fail
        test_task = {**task, "candidate_output": rejected}
        rejected_score = score_task(test_task)
        raw_pairs.append({
            "task_id": task_id,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": base_score.get("score"),
            "chosen_passed": True,
            "rejected_score": rejected_score.get("score"),
            "rejected_passed": rejected_score.get("passed"),
            "source": "passing_task_generated_rejected",
            "dimension": task.get("dimension"),
        })
        with open(RAW_PATH, "a") as f:
            f.write(json.dumps(raw_pairs[-1]) + "\n")

        if rejected_score.get("passed"):
            append_log({
                "task_id": task_id, "batch": "passing→rejected", "action": "discard",
                "reason": f"generated rejected still passes rubric (score={rejected_score.get('score')})",
                "rejected_score": rejected_score.get("score"),
            })
            print(f" REJECTED_PASSED(score={rejected_score.get('score')})")
            continue

        # Cosine similarity (TF-IDF)
        vect2 = TfidfVectorizer().fit([chosen, rejected])
        mat2 = vect2.transform([chosen, rejected])
        sim = float(sk_cosine(mat2[0], mat2[1])[0, 0])

        if sim >= COSINE_THRESHOLD:
            append_log({
                "task_id": task_id, "batch": "passing→rejected", "action": "discard",
                "reason": f"cosine_similarity={sim:.3f} >= {COSINE_THRESHOLD}",
                "cosine_sim": sim,
            })
            print(f" TOO_SIMILAR(sim={sim:.3f})")
            continue

        pair = format_orpo_pair(task, chosen, rejected)
        final_pairs.append(pair)
        with open(FINAL_PATH, "a") as f:
            f.write(json.dumps(pair) + "\n")
        append_log({
            "task_id": task_id, "batch": "passing→rejected", "action": "keep",
            "chosen_score": base_score.get("score"),
            "rejected_score": rejected_score.get("score"),
            "cosine_sim": round(sim, 4),
            "source": "passing_task_generated_rejected",
        })
        print(f" OK(rejected={rejected_score.get('score'):.2f}, sim={sim:.3f})")
        time.sleep(0.3)

    print(f"\n=== DONE ===")
    print(f"Raw pairs written: {len(raw_pairs)}")
    print(f"Final pairs (passed all filters): {len(final_pairs)}")
    print(f"  → training_data/preference_pairs.jsonl")
    print(f"  → training_data/preference_pairs_raw.jsonl")
    print(f"  → training_data/generation_log.jsonl")


if __name__ == "__main__":
    main()
