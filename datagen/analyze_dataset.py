#!/usr/bin/env python3
"""
Dataset health analysis — diagnoses why regressions happen.

Checks:
  1. Task balance (are some tasks drowning others?)
  2. Source distribution (how much is original vs topup vs adversarial?)
  3. Quality distribution (score histogram per task)
  4. Dataset growth over time (which iterations added what)
  5. Recommendations for rebalancing

Usage:
  python analyze_dataset.py
  python analyze_dataset.py --verbose
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from utils.config import load_config

_cfg = load_config()
TRAIN_FILE = _cfg.train_file
VAL_FILE = _cfg.val_file
SCORES_FILE = _cfg.data_dir / "scores.json"


def load_examples():
    examples = []
    for path in [TRAIN_FILE, VAL_FILE]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def analyze(verbose=False):
    examples = load_examples()
    scores = {}
    if SCORES_FILE.exists():
        scores = json.loads(SCORES_FILE.read_text())

    print(f"\n{'='*70}")
    print(f"  DATASET HEALTH ANALYSIS")
    print(f"  Train: {TRAIN_FILE}")
    print(f"  Total examples: {len(examples)}")
    print(f"{'='*70}")

    # ── 1. Task balance ────────────────────────────────────────────────────
    by_task = defaultdict(list)
    for ex in examples:
        by_task[ex.get("task_id", "unknown")].append(ex)

    counts = {t: len(exs) for t, exs in sorted(by_task.items())}
    mean_count = sum(counts.values()) / max(len(counts), 1)
    max_count = max(counts.values()) if counts else 0
    min_count = min(counts.values()) if counts else 0

    print(f"\n  1. TASK BALANCE (mean={mean_count:.0f}, min={min_count}, max={max_count})")
    print(f"  {'Task':<45} {'Count':>6} {'Ratio':>7}  Bar")
    print(f"  {'─'*45} {'─'*6} {'─'*7}  {'─'*20}")
    for task, count in sorted(counts.items()):
        ratio = count / mean_count if mean_count > 0 else 0
        bar = "█" * min(20, round(ratio * 10))
        flag = " ⚠️ OVERWEIGHT" if ratio > 2.0 else " ⚠️ UNDERWEIGHT" if ratio < 0.5 else ""
        print(f"  {task:<45} {count:>6} {ratio:>6.1f}x  {bar}{flag}")

    # ── 2. Source distribution ─────────────────────────────────────────────
    by_source = defaultdict(lambda: defaultdict(int))
    for ex in examples:
        source = ex.get("source", "original")
        by_source[ex.get("task_id", "unknown")][source] += 1

    all_sources = set()
    for task_sources in by_source.values():
        all_sources.update(task_sources.keys())
    all_sources = sorted(all_sources)

    print(f"\n  2. SOURCE DISTRIBUTION")
    header = f"  {'Task':<35}" + "".join(f" {s:>10}" for s in all_sources) + f" {'Total':>8}"
    print(header)
    print(f"  {'─'*35}" + "─" * (11 * len(all_sources)) + f" {'─'*8}")
    for task in sorted(by_source.keys()):
        parts = [f" {by_source[task].get(s, 0):>10}" for s in all_sources]
        total = sum(by_source[task].values())
        print(f"  {task:<35}" + "".join(parts) + f" {total:>8}")

    # Totals
    print(f"  {'─'*35}" + "─" * (11 * len(all_sources)) + f" {'─'*8}")
    totals = []
    for s in all_sources:
        t = sum(by_source[task].get(s, 0) for task in by_source)
        totals.append(t)
    total_all = sum(totals)
    print(f"  {'TOTAL':<35}" + "".join(f" {t:>10}" for t in totals) + f" {total_all:>8}")

    # ── 3. Quality distribution ────────────────────────────────────────────
    if scores:
        print(f"\n  3. QUALITY DISTRIBUTION (from scores.json, {len(scores)} scored)")

        task_scores = defaultdict(list)
        for key, data in scores.items():
            task_id = data.get("task_id", key.split("::")[0] if "::" in key else "unknown")
            s = data.get("score", 0)
            if s > 0:
                task_scores[task_id].append(s)

        print(f"  {'Task':<35} {'Avg':>5} {'Min':>5} {'5s':>4} {'4s':>4} {'3s':>4} {'2s':>4} {'1s':>4}")
        print(f"  {'─'*35} {'─'*5} {'─'*5} {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*4}")
        for task in sorted(task_scores.keys()):
            ss = task_scores[task]
            avg = sum(ss) / len(ss)
            mn = min(ss)
            hist = [sum(1 for x in ss if round(x) == v) for v in [5, 4, 3, 2, 1]]
            print(f"  {task:<35} {avg:>5.1f} {mn:>5.0f} {hist[0]:>4} {hist[1]:>4} {hist[2]:>4} {hist[3]:>4} {hist[4]:>4}")

    # ── 4. Benchmark comparison ────────────────────────────────────────────
    state_file = _cfg.data_dir / "loop_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        history = state.get("model_history", [])
        if history:
            print(f"\n  4. BENCHMARK HISTORY")
            print(f"  {'Version':<30} {'Avg':>6} {'Tasks > 0.5':>12}")
            print(f"  {'─'*30} {'─'*6} {'─'*12}")
            for entry in sorted(history, key=lambda h: h["version"]):
                scores_dict = entry.get("scores", {})
                above_half = sum(1 for v in scores_dict.values() if v >= 0.5)
                print(f"  v{entry['version']:<28} {entry['avg_score']:>6.3f} {above_half:>8}/23")

    # ── 5. Recommendations ─────────────────────────────────────────────────
    print(f"\n  5. RECOMMENDATIONS")

    overweight = [t for t, c in counts.items() if c > mean_count * 2]
    underweight = [t for t, c in counts.items() if c < mean_count * 0.5]

    if overweight:
        print(f"\n  OVERWEIGHT tasks (>2x mean, risking catastrophic forgetting):")
        for t in overweight:
            print(f"    {t}: {counts[t]} examples (mean={mean_count:.0f})")
        print(f"    → Consider trimming to {int(mean_count * 1.5)} examples each")

    if underweight:
        print(f"\n  UNDERWEIGHT tasks (<0.5x mean, model may forget these):")
        for t in underweight:
            print(f"    {t}: {counts[t]} examples (mean={mean_count:.0f})")
        print(f"    → Consider topping up to at least {int(mean_count * 0.75)} examples each")

    if max_count > min_count * 3:
        print(f"\n  IMBALANCE WARNING: max/min ratio = {max_count/max(min_count,1):.1f}x")
        print(f"    Most: {max(counts, key=counts.get)} ({max_count})")
        print(f"    Least: {min(counts, key=counts.get)} ({min_count})")
        print(f"    → Dataset is heavily skewed. Strong tasks will regress.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    analyze(verbose=args.verbose)
