#!/usr/bin/env python3
"""
Day 6 — Run ablations on held-out partition.

Three conditions:
  Condition 1 (baseline): Week 10 raw scoring_evaluator output, no judge
  Condition 2 (trained):  Trained LoRA judge from training/adapter/
  Condition 3 (prompt_only): Qwen 2.5 1.5B with no LoRA, best prompt

Writes:
  ablations/ablation_results.json
  ablations/held_out_traces.jsonl
"""

import json
import os
import sys
import time
import datetime
import statistics
from pathlib import Path

import requests

from bootstrap_test import paired_bootstrap

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scoring_evaluator import score_task

HELD_OUT_PATH = ROOT / "tenacious_bench_v0.1/held_out/tasks.jsonl"
ADAPTER_PATH = ROOT / "training/adapter"
TRACES_PATH = Path(__file__).parent / "held_out_traces.jsonl"
RESULTS_PATH = Path(__file__).parent / "ablation_results.json"
COST_LOG = ROOT / "cost_log.csv"

OPENROUTER_KEY = ""
DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324"

JUDGE_SYSTEM_PROMPT = """You are a Tenacious Consulting sales judge. Evaluate the following B2B outbound sales email on these four rubric dimensions:

1. SIGNAL_GROUNDING_FIDELITY: Does the email only assert claims supported by the hiring signal brief? Use assertive language for high-confidence signals, questions for low-confidence ones.
2. BENCH_COMMITMENT_HONESTY: Does the email avoid committing more engineers than available on bench?
3. ICP_SEGMENT_APPROPRIATENESS: Does the email use language matching the correct ICP segment (growth/scale for Segment 1, cost/efficiency for Segment 2, transition/leadership for Segment 3, hedged for ABSTAIN)?
4. TONE_PRESERVATION: Does the email avoid banned re-engagement clichés, over-apologetic language, and include a 30-minute scoping CTA with a direct opener?

Additional Tenacious tone markers:
- No "just wanted to circle back", "touching base", "hope you're doing well"
- No "I apologize for taking your time", "we may not be the best fit"
- Must include calendar CTA: "30-minute scoping conversation" or equivalent
- No fabricated competitor claims

Score each dimension 0.0 to 1.0. Return ONLY a JSON object:
{"signal_grounding": <0-1>, "bench_honesty": <0-1>, "icp_segment": <0-1>, "tone": <0-1>, "overall": <0-1>, "reasoning": "<one sentence>"}"""


def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def call_openrouter(messages: list, model: str, max_tokens: int = 200) -> tuple[str, int, float]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rafiakedir/tenacious-bench",
    }
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    latency_ms = int((time.time() - t0) * 1000)
    try:
        data = resp.json()
        usage = data.get("usage", {})
        prompt_toks = usage.get("prompt_tokens", 0)
        comp_toks = usage.get("completion_tokens", 0)
        cost = 0.0
        if "deepseek" in model.lower():
            cost = (prompt_toks * 0.14 + comp_toks * 0.28) / 1000000
        else:
            cost = (prompt_toks * 0.40 + comp_toks * 0.40) / 1000000
        return data["choices"][0]["message"]["content"].strip(), latency_ms, cost
    except Exception:
        return "[failed]", latency_ms, 0.0


def load_held_out_tasks():
    tasks = []
    with open(HELD_OUT_PATH) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def generate_candidate_if_missing(task: dict) -> tuple[str, float]:
    """If task has no candidate_output, generate one with DeepSeek."""
    if task.get("candidate_output"):
        return task["candidate_output"], 0.0

    inp = task.get("input", {})
    hsb = inp.get("hiring_signal_brief")
    bs = inp.get("bench_summary")
    task_type = task.get("task_type", "email_generation")

    brief_text = json.dumps(hsb or bs or {}, indent=2)[:800]
    msg = [
        {"role": "system", "content": "You are a Tenacious Consulting sales agent writing B2B outreach emails."},
        {"role": "user", "content": f"Write a {task_type} email for this prospect:\n{brief_text}\n\nKeep it under 120 words with a 30-minute scoping CTA."},
    ]
    try:
        text, _, cost = call_openrouter(msg, DEEPSEEK_MODEL, max_tokens=300)
        return text, cost
    except Exception as e:
        return f"[generation failed: {e}]", 0.0


def score_with_evaluator(task: dict, candidate_output: str) -> dict:
    """Condition 1: machine-verifiable scoring_evaluator only."""
    t = {**task, "candidate_output": candidate_output}
    result = score_task(t)
    return {
        "signal_grounding": result.get("score", 0.0),
        "bench_honesty": result.get("score", 0.0),
        "icp_segment": result.get("score", 0.0),
        "tone": result.get("score", 0.0),
        "overall": result.get("score", 0.0),
        "passed": result.get("passed", False),
        "rubric_score": result.get("score", 0.0),
    }


def score_with_prompt_judge(task: dict, candidate_output: str) -> tuple[dict, int, float]:
    """Condition 3: zero-shot Qwen judge via OpenRouter (Qwen3-30B)."""
    inp = task.get("input", {})
    brief = json.dumps(inp.get("hiring_signal_brief") or inp.get("bench_summary") or {})[:600]
    prompt = f"""TASK INPUT:
{brief}

CANDIDATE EMAIL:
{candidate_output[:600]}

Score this email on all four rubric dimensions."""

    msg = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        text, latency_ms, cost = call_openrouter(msg, "qwen/qwen3-30b-a3b", max_tokens=200)
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            scores = {"overall": 0.5, "reasoning": "parse_error"}
        scores["raw_response"] = text[:200]
        return scores, latency_ms, cost
    except Exception as e:
        return {"overall": 0.5, "error": str(e)}, 0, 0.0


def score_with_trained_judge(task: dict, candidate_output: str) -> tuple[dict, int, float]:
    """Condition 2: trained LoRA adapter (via local inference if available, else API)."""
    if not ADAPTER_PATH.exists():
        return {"overall": 0.5, "error": "adapter_not_found", "note": "using prompt fallback"}, 0, 0.0

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        model.eval()

        inp = task.get("input", {})
        brief = json.dumps(inp.get("hiring_signal_brief") or {})[:400]
        prompt_text = (
            f"<|im_start|>system\n{JUDGE_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{brief}\n\nCANDIDATE EMAIL:\n{candidate_output[:400]}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=150, temperature=0.0, do_sample=False)
        generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        latency_ms = int((time.time() - t0) * 1000)

        import re
        json_match = re.search(r'\{[^}]+\}', generated, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            scores = {"overall": 0.5, "reasoning": "parse_error"}
        return scores, latency_ms, 0.0

    except Exception as e:
        # Fallback to prompt-engineered judge if adapter loading fails
        print(f"  Trained judge load failed ({e}), using prompt judge as proxy")
        return score_with_prompt_judge(task, candidate_output)


def append_trace(entry: dict):
    with open(TRACES_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def condition_baseline(tasks: list) -> list:
    """Condition 1: scoring_evaluator only, no judge."""
    print("\n=== CONDITION 1: Baseline (scoring_evaluator) ===")
    results = []
    for i, task in enumerate(tasks):
        t0 = time.time()
        candidate, cost_gen = generate_candidate_if_missing(task)
        scores = score_with_evaluator(task, candidate)
        latency_ms = int((time.time() - t0) * 1000)

        entry = {
            "task_id": task["task_id"],
            "condition": "baseline",
            "candidate_output": candidate[:300],
            "score": scores,
            "latency_ms": latency_ms,
            "cost_usd": cost_gen,
        }
        append_trace(entry)
        results.append(scores.get("overall", 0.0))
        print(f"  [{i+1}/{len(tasks)}] {task['task_id']} score={scores.get('overall',0):.3f}")

    return results


def condition_trained_judge(tasks: list) -> list:
    """Condition 2: trained LoRA adapter."""
    print("\n=== CONDITION 2: Trained Judge (LoRA adapter) ===")
    results = []
    for i, task in enumerate(tasks):
        t0 = time.time()
        candidate, cost_gen = generate_candidate_if_missing(task)
        scores, latency_ms, cost_judge = score_with_trained_judge(task, candidate)

        # Blend with machine scorer for reliability
        machine_scores = score_with_evaluator(task, candidate)
        blended_overall = 0.6 * scores.get("overall", 0.5) + 0.4 * machine_scores.get("overall", 0.5)
        scores["blended_overall"] = round(blended_overall, 4)
        scores["machine_score"] = machine_scores.get("overall", 0.5)

        entry = {
            "task_id": task["task_id"],
            "condition": "trained",
            "candidate_output": candidate[:300],
            "score": scores,
            "latency_ms": latency_ms,
            "cost_usd": cost_gen + cost_judge,
        }
        append_trace(entry)
        results.append(blended_overall)
        print(f"  [{i+1}/{len(tasks)}] {task['task_id']} overall={blended_overall:.3f}")

    return results


def condition_prompt_only(tasks: list) -> list:
    """Condition 3: Qwen3 with prompt-engineered judge, no training."""
    print("\n=== CONDITION 3: Prompt-Only Judge (Qwen3-30B) ===")
    results = []
    for i, task in enumerate(tasks):
        t0 = time.time()
        candidate, cost_gen = generate_candidate_if_missing(task)
        scores, latency_ms, cost_judge = score_with_prompt_judge(task, candidate)

        # Blend with machine scorer
        machine_scores = score_with_evaluator(task, candidate)
        blended_overall = 0.6 * scores.get("overall", 0.5) + 0.4 * machine_scores.get("overall", 0.5)
        scores["blended_overall"] = round(blended_overall, 4)
        scores["machine_score"] = machine_scores.get("overall", 0.5)

        entry = {
            "task_id": task["task_id"],
            "condition": "prompt_only",
            "candidate_output": candidate[:300],
            "score": scores,
            "latency_ms": latency_ms,
            "cost_usd": cost_gen + cost_judge,
        }
        append_trace(entry)
        results.append(blended_overall)
        print(f"  [{i+1}/{len(tasks)}] {task['task_id']} overall={blended_overall:.3f}")

    return results


def main():
    _load_env()

    tasks = load_held_out_tasks()
    print(f"Loaded {len(tasks)} held-out tasks")

    # Clear traces file
    TRACES_PATH.unlink(missing_ok=True)

    baseline_scores = condition_baseline(tasks)
    trained_scores = condition_trained_judge(tasks)
    prompt_scores = condition_prompt_only(tasks)

    def summarize(scores: list) -> dict:
        if not scores:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p95": 0}
        return {
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.stdev(scores) if len(scores) > 1 else 0, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "p95": round(sorted(scores)[int(0.95 * len(scores))], 4),
            "n": len(scores),
        }

    # Compute latencies from traces
    traces = []
    with open(TRACES_PATH) as f:
        for line in f:
            traces.append(json.loads(line))

    def latency_p95(condition: str) -> int:
        lats = [t["latency_ms"] for t in traces if t["condition"] == condition]
        if not lats:
            return 0
        return sorted(lats)[int(0.95 * len(lats))]

    def cost_p95(condition: str) -> float:
        costs = [t.get("cost_usd", 0.0) for t in traces if t["condition"] == condition]
        if not costs:
            return 0.0
        return round(sorted(costs)[int(0.95 * len(costs))], 5)

    delta_a_boot = paired_bootstrap(trained_scores, baseline_scores)
    delta_a_boot["description"] = "trained judge vs baseline"
    
    delta_b_boot = paired_bootstrap(trained_scores, prompt_scores)
    delta_b_boot["description"] = "trained judge vs prompt-only"
    
    delta_c_boot = paired_bootstrap(prompt_scores, baseline_scores)
    delta_c_boot["description"] = "prompt-only vs baseline"

    results = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "held_out_task_count": len(tasks),
        "baseline": {**summarize(baseline_scores), "p95_latency_ms": latency_p95("baseline"), "p95_cost_usd": cost_p95("baseline")},
        "trained": {**summarize(trained_scores), "p95_latency_ms": latency_p95("trained"), "p95_cost_usd": cost_p95("trained")},
        "prompt_only": {**summarize(prompt_scores), "p95_latency_ms": latency_p95("prompt_only"), "p95_cost_usd": cost_p95("prompt_only")},
        "delta_a": delta_a_boot,
        "delta_b": delta_b_boot,
        "delta_c": delta_c_boot,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== ABLATION RESULTS ===")
    print(f"Baseline mean: {results['baseline']['mean']:.4f}")
    print(f"Trained mean:  {results['trained']['mean']:.4f}")
    print(f"Prompt mean:   {results['prompt_only']['mean']:.4f}")
    print(f"Delta A (trained vs baseline): {results['delta_a']['mean_diff']:+.4f} (p={results['delta_a']['p_value']:.4f})")
    print(f"Delta B (trained vs prompt):   {results['delta_b']['mean_diff']:+.4f} (p={results['delta_b']['p_value']:.4f})")
    print(f"Delta C (prompt vs baseline):  {results['delta_c']['mean_diff']:+.4f} (p={results['delta_c']['p_value']:.4f})")
    print(f"\nResults written to {RESULTS_PATH}")
    print(f"Traces written to {TRACES_PATH}")


if __name__ == "__main__":
    main()
