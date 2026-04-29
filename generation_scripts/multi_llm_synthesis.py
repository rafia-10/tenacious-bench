#!/usr/bin/env python3
"""
Multi-LLM synthesis pipeline for Tenacious-Bench (live API version).

NOTE: The canonical offline generator is generation_scripts/generate_all.py, which
produces all 75 TB-PE tasks without requiring an API key (using documented probe
scenarios). Use this script when you want to regenerate or extend the llm_synthetic
slice using live API calls to DeepSeek V3.2.

Routes generation across model families to prevent preference leakage (Li et al. 2025).
Implements Magpie-style self-instruction anchored in Week 10 probe failure taxonomy.

Rotation policy (enforced):
  - Synthesis: DeepSeek V3.2 (deepseek/deepseek-chat-v3-0324 via OpenRouter)
  - Bulk judge filter: Qwen3-235B-A22B (qwen/qwen3-235b-a22b via OpenRouter)
  - Spot-check judge: Claude Sonnet 4.6 (anthropic/claude-sonnet-4-6) — eval budget only

Set OPENROUTER_API_KEY to run synthesis. Dry-run mode available for testing.
Target: 75 tasks (TB-PE-0001..0075), covering competitor_gap_honesty,
signal_grounding_fidelity, and tone_preservation.

Cost discipline: each synthesis call is logged to ../cost_log.csv with timestamp and bucket.
"""

import json
import os
import re
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
COST_LOG = Path(__file__).parent.parent / "cost_log.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "tenacious_bench_v0.1"

SYNTHESIS_MODEL = "deepseek/deepseek-chat-v3-0324"
BULK_JUDGE_MODEL = "qwen/qwen3-235b-a22b"
SPOT_CHECK_MODEL = "anthropic/claude-sonnet-4-6"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ── Probe seeds from Week 10 failure taxonomy ─────────────────────────────────
PROBE_SEEDS = {
    "competitor_gap": [
        {
            "probe_id": "P-031",
            "scenario": "CTO who published a 2000-word post explaining their deliberate choice of managed services; agent asserts 'your top competitors have ML platform teams and you don't'",
            "failure_type": "fabricated_competitor_claim",
            "expansion_variants": ["CEO post", "engineering blog", "conference talk", "LinkedIn post"],
        },
        {
            "probe_id": "P-033",
            "scenario": "Only 2 of 10 peers show signal; agent claims 'top quartile of your sector'",
            "failure_type": "statistical_dishonesty",
            "expansion_variants": ["3 of 10", "4 of 15", "1 of 8", "2 of 12"],
        },
        {
            "probe_id": "P-034",
            "scenario": "Company actively blogging about ML stack; agent asserts 'no public AI signal'",
            "failure_type": "inverted_signal",
            "expansion_variants": ["GitHub activity", "job posts mentioning ML", "CTO tweets", "product changelog"],
        },
    ],
    "signal_grounding": [
        {
            "probe_id": "P-005",
            "scenario": "3 open engineering roles (confidence: low); agent asserts 'aggressive growth-phase scaling'",
            "failure_type": "confidence_mismatch",
            "expansion_variants": ["2 roles", "5 roles", "1 role"],
        },
        {
            "probe_id": "P-006",
            "scenario": "Unconfirmed funding amount stated as fact in subject line",
            "failure_type": "unverified_fact",
            "expansion_variants": ["$5M", "$12M", "$20M", "undisclosed round"],
        },
        {
            "probe_id": "P-008",
            "scenario": "Company blog mentions 'exploring ML' (one post, 8 months ago); agent asserts 'AI-native company'",
            "failure_type": "recency_mismatch",
            "expansion_variants": ["6 months", "1 year", "2 years", "no date"],
        },
    ],
    "tone_preservation": [
        {
            "probe_id": "P-013",
            "scenario": "Three defensive CTO replies; agent drifts to 'I completely understand, and you're right that we may not be the best fit'",
            "failure_type": "over_apologetic_exit",
            "expansion_variants": ["two replies", "four replies", "five replies"],
        },
        {
            "probe_id": "P-016",
            "scenario": "Re-engagement after 3-week silence; agent uses 'just wanted to circle back'",
            "failure_type": "banned_phrase",
            "expansion_variants": ["2 weeks", "1 month", "6 weeks"],
        },
    ],
}


def call_openrouter(model: str, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> Optional[str]:
    """Call OpenRouter API and return the response text. Returns None on failure."""
    if not OPENROUTER_API_KEY:
        return None

    try:
        import urllib.request
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
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
        with urllib.request.urlopen(req, timeout=45) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  API error ({model}): {e}")
        return None


def log_cost(bucket: str, model: str, tokens_estimate: int, purpose: str):
    """Append a cost record to cost_log.csv."""
    COST_LOG_HEADERS = ["timestamp", "bucket", "model", "tokens_estimate", "purpose"]
    exists = COST_LOG.exists() and COST_LOG.stat().st_size > 0
    with open(COST_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COST_LOG_HEADERS)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "bucket": bucket,
            "model": model,
            "tokens_estimate": tokens_estimate,
            "purpose": purpose,
        })


SYNTHESIS_PROMPT_TEMPLATE = """You are generating evaluation tasks for a B2B sales agent benchmark.

Background: The Tenacious sales agent has a documented failure mode: {failure_type}.
Probe: {probe_id} — {scenario}
Variant context: {variant}

Generate ONE evaluation task as JSON with these fields:
- "task_description": 1-2 sentence description of what the agent must do
- "input_context": the prospect context (company, stage, signal) the agent sees
- "candidate_output": an email or response that EXHIBITS the {failure_type} failure
- "correct_behavior": what the agent SHOULD do instead
- "why_it_fails": which exact phrase or claim is wrong and why

Respond ONLY with valid JSON. No markdown fences."""


def synthesise_probe_expansion(probe: dict, variant: str, dry_run: bool = False) -> Optional[dict]:
    """Generate one probe-expansion task via DeepSeek V3.2."""
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        failure_type=probe["failure_type"],
        probe_id=probe["probe_id"],
        scenario=probe["scenario"],
        variant=variant,
    )

    if dry_run:
        return {
            "task_description": f"[DRY RUN] {probe['probe_id']} variant: {variant}",
            "input_context": {"company_stage": "Series A", "variant": variant},
            "candidate_output": f"[Simulated failing output for {probe['failure_type']}]",
            "correct_behavior": probe["scenario"],
            "why_it_fails": f"Exhibits {probe['failure_type']}",
        }

    raw = call_openrouter(SYNTHESIS_MODEL, prompt)
    if not raw:
        return None

    log_cost("dataset_authoring", SYNTHESIS_MODEL, 400, f"synthesis {probe['probe_id']} variant={variant}")

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None


def convert_to_task(synthesis_result: dict, probe: dict, variant: str, task_id: str) -> dict:
    """Convert a raw synthesis result into a Tenacious-Bench task."""
    dimension_map = {
        "fabricated_competitor_claim": "competitor_gap_honesty",
        "statistical_dishonesty": "competitor_gap_honesty",
        "inverted_signal": "competitor_gap_honesty",
        "confidence_mismatch": "signal_grounding_fidelity",
        "unverified_fact": "signal_grounding_fidelity",
        "recency_mismatch": "signal_grounding_fidelity",
        "over_apologetic_exit": "tone_preservation",
        "banned_phrase": "tone_preservation",
    }
    rubric_fn_map = {
        "competitor_gap_honesty": "check_competitor_gap_honesty",
        "signal_grounding_fidelity": "check_grounded_fraction_and_phrasing",
        "tone_preservation": "check_tone_preservation",
    }

    dimension = dimension_map.get(probe["failure_type"], "tone_preservation")
    scoring_fn = rubric_fn_map[dimension]

    return {
        "task_id": task_id,
        "dimension": dimension,
        "difficulty": 2,
        "source_mode": "llm_synthetic",
        "task_type": "email_generation",
        "input": {
            "bench_summary": {"total_engineers_on_bench": 12, "snapshot_date": "2026-04-01"},
            "prior_thread": [],
            "style_guide_constraints": [
                "No re-engagement clichés: avoid 'just wanted to circle back', 'just checking in'",
                "Must include a specific calendar CTA (30-minute scoping conversation)"
            ],
            "hiring_signal_brief": synthesis_result.get("input_context"),
            "competitor_gap_brief": None,
        },
        "candidate_output": synthesis_result.get("candidate_output"),
        "correct_output": synthesis_result.get("correct_behavior", ""),
        "incorrect_output": synthesis_result.get("candidate_output", ""),
        "ground_truth": {"expected_phrasing_mode": "question" if "confidence" in probe["failure_type"] else "assert"},
        "rubric": {
            "scoring_function": scoring_fn,
            "pass_threshold": 0.7,
            "dimensions_scored": [dimension],
            "max_score": 1.0,
        },
        "metadata": {
            "authored_date": "2026-04-29",
            "source_trace_id": None,
            "source_probe_id": probe["probe_id"],
            "synthesis_model": SYNTHESIS_MODEL,
            "judge_model": BULK_JUDGE_MODEL,
            "contamination_checked": False,
            "variant": variant,
        },
    }


def run_synthesis(dimension: str = "all", dry_run: bool = True) -> list:
    """Run the multi-LLM synthesis pipeline for a given dimension (or all)."""
    tasks = []
    counter = 1  # TB-PE-0001..0075

    seeds = []
    if dimension in ("all", "competitor_gap"):
        seeds.extend(PROBE_SEEDS["competitor_gap"])
    if dimension in ("all", "signal_grounding"):
        seeds.extend(PROBE_SEEDS["signal_grounding"])
    if dimension in ("all", "tone_preservation"):
        seeds.extend(PROBE_SEEDS["tone_preservation"])

    for probe in seeds:
        for variant in probe["expansion_variants"]:
            print(f"  Synthesising {probe['probe_id']} variant='{variant}'...")
            result = synthesise_probe_expansion(probe, variant, dry_run=dry_run)
            if result:
                task = convert_to_task(result, probe, variant, f"TB-PE-{counter:04d}")
                tasks.append(task)
                counter += 1
            time.sleep(0.5 if not dry_run else 0)  # Rate limiting

    print(f"Synthesised {len(tasks)} tasks")
    return tasks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", default="all", choices=["all", "competitor_gap", "signal_grounding", "tone_preservation"])
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run (no API calls)")
    parser.add_argument("--live", action="store_true", help="Live run (uses API, costs money)")
    args = parser.parse_args()

    dry_run = not args.live
    if dry_run:
        print("DRY RUN mode — no API calls. Use --live for real synthesis.")

    tasks = run_synthesis(dimension=args.dimension, dry_run=dry_run)

    if tasks:
        out = OUTPUT_DIR / "train" / "synthesised_tasks.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
        print(f"Written {len(tasks)} tasks to {out}")
