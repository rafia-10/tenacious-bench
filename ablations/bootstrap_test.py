#!/usr/bin/env python3
"""
Step 9 — Paired bootstrap significance test.
Delta A: trained vs baseline
Delta B: trained vs prompt-only
10,000 bootstrap iterations, 95% CI, p-value = fraction of samples where diff ≤ 0.
"""

import json
import random
import statistics
from pathlib import Path

TRACES_PATH = Path(__file__).parent / "held_out_traces.jsonl"
RESULTS_PATH = Path(__file__).parent / "ablation_results.json"
N_BOOTSTRAP = 10_000
SEED = 42


def load_scores_by_condition(traces_path: Path) -> dict[str, dict[str, float]]:
    by_condition = {}
    with open(traces_path) as f:
        for line in f:
            t = json.loads(line)
            cond = t["condition"]
            task_id = t["task_id"]
            score = t["score"].get("blended_overall") or t["score"].get("overall") or t["score"].get("rubric_score", 0.0)
            if cond not in by_condition:
                by_condition[cond] = {}
            by_condition[cond][task_id] = float(score)
    return by_condition


def paired_bootstrap(scores_a: list[float], scores_b: list[float], n: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """
    Paired bootstrap test: H0: mean(A) - mean(B) <= 0
    scores_a should be the 'better' condition (trained)
    scores_b is the comparison condition (baseline or prompt_only)
    Returns: mean_diff, ci_lower, ci_upper, p_value, significant
    """
    rng = random.Random(seed)
    n_pairs = len(scores_a)
    observed_diff = statistics.mean(scores_a) - statistics.mean(scores_b)

    bootstrap_diffs = []
    for _ in range(n):
        indices = [rng.randint(0, n_pairs - 1) for _ in range(n_pairs)]
        sample_a = [scores_a[i] for i in indices]
        sample_b = [scores_b[i] for i in indices]
        bootstrap_diffs.append(statistics.mean(sample_a) - statistics.mean(sample_b))

    bootstrap_diffs.sort()
    ci_lower = bootstrap_diffs[int(0.025 * n)]
    ci_upper = bootstrap_diffs[int(0.975 * n)]
    p_value = sum(1 for d in bootstrap_diffs if d <= 0) / n
    significant = p_value < 0.05

    return {
        "mean_diff": round(observed_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "significant": significant,
        "n_pairs": n_pairs,
        "n_bootstrap": n,
    }


def main():
    print("Loading ablation traces...")
    by_condition = load_scores_by_condition(TRACES_PATH)

    conditions = list(by_condition.keys())
    print(f"Conditions found: {conditions}")
    print(f"Tasks per condition: {[len(v) for v in by_condition.values()]}")

    # Align task IDs across all conditions
    all_task_ids = set.intersection(*[set(v.keys()) for v in by_condition.values()])
    print(f"Aligned task count: {len(all_task_ids)}")
    task_ids = sorted(all_task_ids)

    def get_scores(cond: str) -> list[float]:
        return [by_condition[cond].get(tid, 0.5) for tid in task_ids]

    trained = get_scores("trained")
    baseline = get_scores("baseline")
    prompt_only = get_scores("prompt_only")

    print(f"\nBaseline mean: {statistics.mean(baseline):.4f}")
    print(f"Trained mean:  {statistics.mean(trained):.4f}")
    print(f"Prompt mean:   {statistics.mean(prompt_only):.4f}")

    print("\nRunning Delta A bootstrap (trained vs baseline)...")
    delta_a = paired_bootstrap(trained, baseline)
    delta_a["description"] = "trained judge vs Week 10 baseline"
    print(f"  mean_diff={delta_a['mean_diff']:+.4f}  CI=[{delta_a['ci_lower']:.4f}, {delta_a['ci_upper']:.4f}]  p={delta_a['p_value']:.4f}  significant={delta_a['significant']}")

    print("\nRunning Delta B bootstrap (trained vs prompt-only)...")
    delta_b = paired_bootstrap(trained, prompt_only)
    delta_b["description"] = "trained judge vs prompt-engineered judge"

    if not delta_b["significant"]:
        delta_b["finding"] = "prompt_engineering_sufficient"
        delta_b["finding_explanation"] = (
            "The trained ORPO judge does not significantly outperform the prompt-only Qwen3 judge. "
            "This is a legitimate and informative result: it indicates that the rubric encoding in the "
            "prompt-engineered condition is sufficient for this evaluation task at this scale of training data. "
            "It does not invalidate the trained judge — it suggests the marginal benefit of fine-tuning is "
            "task-specific and may require more preference pairs or a harder held-out distribution. "
            "For the production Conversion Engine, the prompt-only condition is a viable lower-cost alternative."
        )
    else:
        delta_b["finding"] = "training_beats_prompting"
        delta_b["finding_explanation"] = (
            "The trained ORPO judge significantly outperforms the prompt-engineered judge. "
            "This validates the preference-tuning approach for Tenacious-specific rubric dimensions."
        )

    print(f"  mean_diff={delta_b['mean_diff']:+.4f}  CI=[{delta_b['ci_lower']:.4f}, {delta_b['ci_upper']:.4f}]  p={delta_b['p_value']:.4f}  significant={delta_b['significant']}")
    print(f"  finding: {delta_b['finding']}")

    # Update results file
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    results["delta_a"] = delta_a
    results["delta_b"] = delta_b
    results["bootstrap_seed"] = SEED
    results["bootstrap_n"] = N_BOOTSTRAP
    results["baseline"]["mean"] = round(statistics.mean(baseline), 4)
    results["trained"]["mean"] = round(statistics.mean(trained), 4)
    results["prompt_only"]["mean"] = round(statistics.mean(prompt_only), 4)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nUpdated results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
