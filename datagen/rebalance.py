#!/usr/bin/env python3
"""
Rebalance the dataset — trim overweight tasks, keeping highest-scored examples.

This directly fixes catastrophic forgetting caused by data flooding:
tasks with 500 examples drown out tasks with 50, causing the model
to forget strong tasks while not learning the weak ones.

Usage:
  python -m datagen.rebalance --target 100          # trim to 100 per task
  python -m datagen.rebalance --target 100 --dry-run  # show what would change
  python -m datagen.rebalance --target 120 --min 80   # target 120, floor 80
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

from utils.config import load_config

_cfg = load_config()
TRAIN_FILE = _cfg.train_file
VAL_FILE = _cfg.val_file
SCORES_FILE = _cfg.data_dir / "scores.json"


def build_score_key(example: dict) -> str:
    """Build the same key llm_judge uses for score lookup."""
    task_id = example.get("task_id", "unknown")
    msgs = example.get("messages", [])
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    user_text = user_msgs[0]["content"][:80] if user_msgs else ""
    return f"{task_id}|{user_text}"


def get_score(example: dict, scores: dict) -> float:
    """Look up the judge score for an example. Default 3.0 if unscored."""
    key = build_score_key(example)
    # Try both key formats (pipe and double-colon)
    data = scores.get(key) or scores.get(key.replace("|", "::"))
    if data:
        return data.get("score", 3.0)
    return 3.0


def rebalance(target: int = 100, min_per_task: int = 0, dry_run: bool = False):
    scores = {}
    if SCORES_FILE.exists():
        scores = json.loads(SCORES_FILE.read_text())

    # Load all examples from both files
    all_examples = []
    for path in [TRAIN_FILE, VAL_FILE]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                all_examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Group by task
    by_task = defaultdict(list)
    for ex in all_examples:
        by_task[ex.get("task_id", "unknown")].append(ex)

    print(f"\n{'='*70}")
    print(f"  REBALANCE PLAN (target={target} per task)")
    print(f"{'='*70}")
    print(f"\n  {'Task':<40} {'Before':>7} {'After':>7} {'Change':>8}")
    print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*8}")

    rebalanced = {}
    total_before = 0
    total_after = 0
    total_trimmed = 0

    for task_id in sorted(by_task.keys()):
        examples = by_task[task_id]
        before = len(examples)
        total_before += before

        if before <= target:
            # Underweight or at target — keep all
            rebalanced[task_id] = examples
            after = before
        else:
            # Overweight — keep the highest-scored examples
            scored = [(ex, get_score(ex, scores)) for ex in examples]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Always prefer original examples over targeted/adversarial
            originals = [(ex, s) for ex, s in scored if ex.get("source", "original") == "original"]
            others = [(ex, s) for ex, s in scored if ex.get("source", "original") != "original"]

            kept = []
            # Keep all originals first (they're the proven good examples)
            for ex, s in originals:
                if len(kept) < target:
                    kept.append(ex)
            # Fill remaining slots with highest-scored generated examples
            for ex, s in others:
                if len(kept) < target:
                    kept.append(ex)

            rebalanced[task_id] = kept
            after = len(kept)

        total_after += after
        change = after - before
        flag = " ← TRIMMED" if change < 0 else ""
        print(f"  {task_id:<40} {before:>7} {after:>7} {change:>+8}{flag}")
        if change < 0:
            total_trimmed += abs(change)

    print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*8}")
    print(f"  {'TOTAL':<40} {total_before:>7} {total_after:>7} {total_after-total_before:>+8}")
    print(f"\n  Trimmed: {total_trimmed} examples removed")
    print(f"  Strategy: keep originals first, then highest-scored generated")

    if dry_run:
        print(f"\n  DRY RUN — no files modified")
        return

    # Write rebalanced dataset
    val_split = _cfg.data.val_split
    train_out = []
    val_out = []

    for task_id, examples in sorted(rebalanced.items()):
        random.shuffle(examples)
        n_val = max(2, round(len(examples) * val_split))
        val_out.extend(examples[:n_val])
        train_out.extend(examples[n_val:])

    TRAIN_FILE.write_text("\n".join(json.dumps(ex) for ex in train_out) + "\n")
    VAL_FILE.write_text("\n".join(json.dumps(ex) for ex in val_out) + "\n")

    print(f"\n  Written:")
    print(f"    Train: {len(train_out)} → {TRAIN_FILE}")
    print(f"    Val:   {len(val_out)} → {VAL_FILE}")
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Rebalance dataset by trimming overweight tasks")
    parser.add_argument("--target", type=int, default=100,
                        help="Max examples per task (default: 100)")
    parser.add_argument("--min", type=int, default=0,
                        help="Min examples per task (won't trim below this)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without modifying files")
    args = parser.parse_args()
    rebalance(target=args.target, min_per_task=args.min, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
