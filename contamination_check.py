#!/usr/bin/env python3
"""
Tenacious-Bench v0.1 Contamination Check
Runs three checks before any task enters the held-out partition:
  1. N-gram overlap: no 8-gram match between held-out and training inputs.
  2. Embedding similarity: cosine < 0.85 for any held-out/train pair (using TF-IDF).
  3. Time-shift verification: tasks referencing public data cite a documentable time window.

Results are written to contamination_check.json.
"""

import json
import re
import math
from pathlib import Path
from collections import Counter, defaultdict

BENCH_ROOT = Path(__file__).parent
PARTITION_DIR = BENCH_ROOT / "tenacious_bench_v0.1"
OUTPUT_FILE = BENCH_ROOT / "contamination_check.json"

NGRAM_SIZE = 8
EMBEDDING_THRESHOLD = 0.85
TIME_REFERENCE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",                      # ISO date
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",
    r"\d+ (days?|weeks?|months?) ago",
    r"(q[1-4]|first|second|third|fourth) quarter \d{4}",
    r"series [abc]",                             # funding round (not time-anchored alone)
    r"snapshot_date",                            # bench snapshot
]


def load_partition(name: str) -> list:
    path = PARTITION_DIR / name / "tasks.jsonl"
    tasks = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return tasks


def task_to_text(task: dict) -> str:
    """Flatten all input fields to a single string for text-based comparisons."""
    parts = []
    inp = task.get("input", {})
    for k, v in inp.items():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    parts.append(json.dumps(item))
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(v, dict):
            parts.append(json.dumps(v))
    parts.append(task.get("candidate_output") or "")
    return " ".join(parts).lower()


def get_ngrams(text: str, n: int) -> set:
    tokens = re.findall(r'\b\w+\b', text)
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def tfidf_cosine(text_a: str, text_b: str, vocab: dict) -> float:
    """Simple TF-IDF cosine similarity (no external dependencies)."""
    def tf(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        return {t: c / total for t, c in counts.items()}

    tf_a = tf(text_a)
    tf_b = tf(text_b)
    idf = vocab

    def vec(tf_dict):
        return {t: tf_dict.get(t, 0) * idf.get(t, 1.0) for t in idf}

    va = vec(tf_a)
    vb = vec(tf_b)
    dot = sum(va.get(t, 0) * vb.get(t, 0) for t in idf)
    mag_a = math.sqrt(sum(v ** 2 for v in va.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def build_idf(corpus: list) -> dict:
    N = len(corpus)
    df = Counter()
    for text in corpus:
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        df.update(tokens)
    return {t: math.log(N / (c + 1)) + 1 for t, c in df.items() if c < N * 0.8}


def check_time_shift(task: dict) -> dict:
    """Verify that tasks referencing public signals cite a documentable time window."""
    text = task_to_text(task)
    has_time_ref = any(re.search(p, text, re.IGNORECASE) for p in TIME_REFERENCE_PATTERNS[:4])
    has_public_signal = any(kw in text for kw in [
        "funding", "series", "hiring", "layoff", "crunchbase", "job post", "leadership change"
    ])

    if has_public_signal and not has_time_ref:
        return {
            "pass": False,
            "detail": "task references public signal but provides no time anchor",
        }
    return {"pass": True, "detail": "ok"}


def run_checks():
    print("Loading partitions...")
    train_tasks = load_partition("train")
    dev_tasks = load_partition("dev")
    held_out_tasks = load_partition("held_out")

    train_texts = {t["task_id"]: task_to_text(t) for t in train_tasks}
    held_out_texts = {t["task_id"]: task_to_text(t) for t in held_out_tasks}

    all_texts = list(train_texts.values()) + list(held_out_texts.values())
    idf_vocab = build_idf(all_texts)

    results = {
        "methodology": {
            "ngram_size": NGRAM_SIZE,
            "embedding_threshold": EMBEDDING_THRESHOLD,
            "embedding_method": "tf-idf cosine (local, no external deps)",
        },
        "partition_counts": {
            "train": len(train_tasks),
            "dev": len(dev_tasks),
            "held_out": len(held_out_tasks),
        },
        "ngram_check": {"violations": [], "max_overlap": 0, "pass": True},
        "embedding_check": {"violations": [], "max_similarity": 0.0, "pass": True},
        "time_shift_check": {"violations": [], "pass": True},
        "overall_pass": True,
    }

    # ── Check 1: N-gram overlap ──────────────────────────────────────────────
    print(f"Running {NGRAM_SIZE}-gram overlap check ({len(held_out_tasks)} held-out x {len(train_tasks)} train)...")
    train_ngrams = {tid: get_ngrams(text, NGRAM_SIZE) for tid, text in train_texts.items()}

    for h_id, h_text in held_out_texts.items():
        h_ngrams = get_ngrams(h_text, NGRAM_SIZE)
        if not h_ngrams:
            continue
        for t_id, t_ngrams in train_ngrams.items():
            overlap = len(h_ngrams & t_ngrams)
            if overlap > 0:
                results["ngram_check"]["violations"].append({
                    "held_out_id": h_id,
                    "train_id": t_id,
                    "overlap_count": overlap,
                })
                results["ngram_check"]["max_overlap"] = max(
                    results["ngram_check"]["max_overlap"], overlap
                )
                results["ngram_check"]["pass"] = False

    if results["ngram_check"]["pass"]:
        print(f"  PASS: no {NGRAM_SIZE}-gram overlap between held-out and train")
    else:
        n_viol = len(results["ngram_check"]["violations"])
        print(f"  WARN: {n_viol} held-out/train pairs have {NGRAM_SIZE}-gram overlap")

    # ── Check 2: Embedding similarity ────────────────────────────────────────
    print(f"Running embedding similarity check (threshold={EMBEDDING_THRESHOLD})...")
    max_sim = 0.0
    for h_id, h_text in held_out_texts.items():
        for t_id, t_text in train_texts.items():
            sim = tfidf_cosine(h_text, t_text, idf_vocab)
            if sim > EMBEDDING_THRESHOLD:
                results["embedding_check"]["violations"].append({
                    "held_out_id": h_id,
                    "train_id": t_id,
                    "cosine_similarity": round(sim, 4),
                })
                results["embedding_check"]["pass"] = False
            max_sim = max(max_sim, sim)
    results["embedding_check"]["max_similarity"] = round(max_sim, 4)

    if results["embedding_check"]["pass"]:
        print(f"  PASS: max cosine similarity = {max_sim:.4f} (threshold {EMBEDDING_THRESHOLD})")
    else:
        n_viol = len(results["embedding_check"]["violations"])
        print(f"  WARN: {n_viol} pairs exceed threshold {EMBEDDING_THRESHOLD}")

    # ── Check 3: Time-shift verification ─────────────────────────────────────
    print("Running time-shift verification on held-out tasks...")
    for task in held_out_tasks:
        check = check_time_shift(task)
        if not check["pass"]:
            results["time_shift_check"]["violations"].append({
                "task_id": task["task_id"],
                "detail": check["detail"],
            })
            results["time_shift_check"]["pass"] = False

    if results["time_shift_check"]["pass"]:
        print(f"  PASS: all held-out tasks with public signals have time anchors")
    else:
        n_viol = len(results["time_shift_check"]["violations"])
        print(f"  WARN: {n_viol} held-out tasks missing time anchor")

    # ── Overall ───────────────────────────────────────────────────────────────
    results["overall_pass"] = (
        results["ngram_check"]["pass"]
        and results["embedding_check"]["pass"]
        and results["time_shift_check"]["pass"]
    )

    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {OUTPUT_FILE}")
    print(f"Overall: {'PASS' if results['overall_pass'] else 'WARN — see violations above'}")
    return results


if __name__ == "__main__":
    run_checks()
