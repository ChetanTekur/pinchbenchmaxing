#!/usr/bin/env python3
"""
Generate training examples via real-time API (batch API workaround).

Usage:
  python scripts/gen_realtime.py --tasks task_19_spreadsheet_summary --count 3
  python scripts/gen_realtime.py --all-below 40 --count 50
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from utils.config import load_config
from datagen.task_loader import load_tasks
from datagen.dynamic_gen import (
    build_dynamic_meta_prompt, VARIATION_CONFIGS, HARD_TASKS,
    _epc, compute_dynamic_deficits, resolve_tasks,
)
from datagen.topup import parse_example, extract_json_array, count_existing

cfg = load_config()


def generate_for_task(client, task_id, task_def, n_needed, max_retries=2):
    """Generate examples for one task using real-time API."""
    epc = _epc(task_id)
    n_calls = (n_needed + epc - 1) // epc
    max_tok = 16000 if task_id in HARD_TASKS else 8192
    model = cfg.claude.generation

    all_parsed = []
    errors = 0

    for i in range(n_calls):
        variation = VARIATION_CONFIGS[i % len(VARIATION_CONFIGS)]
        prompt = build_dynamic_meta_prompt(task_id, task_def, variation, epc)

        for retry in range(max_retries + 1):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=max_tok,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
                examples = extract_json_array(raw)

                if examples:
                    for ex in examples:
                        parsed = parse_example(ex, task_id)
                        if parsed:
                            parsed["source"] = "dynamic_realtime"
                            all_parsed.append(parsed)
                    break
                else:
                    errors += 1
                    if retry < max_retries:
                        print(f"      Parse failure, retrying ({retry+1}/{max_retries})...")
                        time.sleep(2)
            except anthropic.OverloadedError:
                if retry < max_retries:
                    print(f"      Overloaded, waiting 30s...")
                    time.sleep(30)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"      Error: {e}")
                break

        # Progress
        generated_so_far = len(all_parsed)
        print(f"    [{i+1}/{n_calls}] {generated_so_far} examples so far", end="\r")

    print(f"    {len(all_parsed)} examples generated ({errors} errors)        ")
    return all_parsed


def main():
    parser = argparse.ArgumentParser(description="Generate via real-time API")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task IDs")
    parser.add_argument("--all-below", type=int, default=None,
                        help="All tasks below N examples")
    parser.add_argument("--count", type=int, default=50,
                        help="Target examples per task")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    only_tasks = None
    if args.tasks:
        only_tasks = [t.strip() for t in args.tasks.split(",")]

    tasks = resolve_tasks(only_tasks=only_tasks, all_below=args.all_below)
    counts = count_existing()

    print(f"\nTasks to generate for:")
    task_plan = {}
    for tid in sorted(tasks.keys()):
        current = counts.get(tid, 0)
        needed = max(0, args.count - current)
        if needed > 0:
            task_plan[tid] = needed
            print(f"  {tid:<40} have={current} need={needed}")

    if not task_plan:
        print("  All tasks at or above target.")
        return

    total_needed = sum(task_plan.values())
    print(f"\nTotal examples to generate: {total_needed}")
    print(f"Using real-time API ({cfg.claude.generation})\n")

    for task_id, needed in task_plan.items():
        task_def = tasks[task_id]
        print(f"\n  {task_id} — {task_def.get('name', '?')} ({needed} needed)")

        parsed = generate_for_task(client, task_id, task_def, needed)

        if parsed:
            # Split train/val
            random.shuffle(parsed)
            val_split = cfg._data.get("data", {}).get("val_split", 0.1)
            val_count = max(1, round(len(parsed) * val_split))
            val_set = parsed[:val_count]
            train_set = parsed[val_count:]

            with open(cfg.train_file, "a") as f:
                for ex in train_set:
                    f.write(json.dumps(ex) + "\n")
            with open(cfg.val_file, "a") as f:
                for ex in val_set:
                    f.write(json.dumps(ex) + "\n")

            print(f"    Saved: {len(train_set)} train + {len(val_set)} val")

    print(f"\nDone. Run: python3 -m datagen.inspect_data stats")


if __name__ == "__main__":
    main()
