#!/usr/bin/env python3
"""Compare two datasets side by side — task counts, quality, diversity."""

import json
import sys
from collections import Counter
from pathlib import Path


def count_tasks(path):
    counts = Counter()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                counts[json.loads(line).get("task_id", "?")] += 1
            except json.JSONDecodeError:
                pass
    return counts


def main():
    old = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v8_backup/data/train.jsonl"
    new = sys.argv[2] if len(sys.argv) > 2 else "/workspace/synthbench/data/train.jsonl"

    v8 = count_tasks(old)
    cur = count_tasks(new)

    print(f"\nOld: {old}")
    print(f"New: {new}\n")
    print(f"{'Task':<40} {'old':>5} {'new':>5} {'diff':>6}")
    print("-" * 58)

    all_tasks = sorted(set(list(v8.keys()) + list(cur.keys())))
    for t in all_tasks:
        a, b = v8.get(t, 0), cur.get(t, 0)
        d = b - a
        flag = "  NEW" if a == 0 else ("  GONE" if b == 0 else "")
        print(f"{t:<40} {a:>5} {b:>5} {d:>+6}{flag}")

    print(f"\n{'TOTAL':<40} {sum(v8.values()):>5} {sum(cur.values()):>5}")


if __name__ == "__main__":
    main()
