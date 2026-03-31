#!/usr/bin/env python3
"""Compare a snapshot's train.jsonl against current train.jsonl.

Usage:
  python scripts/diff_snapshot.py
  python scripts/diff_snapshot.py /path/to/snapshot/train.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

# Default: find the most recent snapshot
SNAP_DIR = Path("/workspace/synthbench/data/snapshots")
TRAIN = Path("/workspace/synthbench/data/train.jsonl")


def count_tasks(path):
    counts = Counter()
    for line in open(path):
        if line.strip():
            try:
                counts[json.loads(line).get("task_id", "")] += 1
            except json.JSONDecodeError:
                pass
    return counts


def main():
    if len(sys.argv) > 1:
        snap_train = Path(sys.argv[1])
    else:
        # Find most recent snapshot
        snapshots = sorted(SNAP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshots:
            print("No snapshots found")
            return
        snap_train = snapshots[0] / "train.jsonl"
        print(f"Using snapshot: {snapshots[0].name}")

    if not snap_train.exists():
        print(f"Not found: {snap_train}")
        return

    snap = count_tasks(snap_train)
    curr = count_tasks(TRAIN)

    print(f"\n{'Task':<40} {'Snap':>5} {'Curr':>5} {'Delta':>6}")
    print(f"{'-'*40} {'-'*5} {'-'*5} {'-'*6}")

    all_tasks = sorted(set(list(snap.keys()) + list(curr.keys())))
    for t in all_tasks:
        s, c = snap.get(t, 0), curr.get(t, 0)
        delta = s - c
        marker = " <--" if delta != 0 else ""
        print(f"{t:<40} {s:>5} {c:>5} {delta:>+6}{marker}")

    print(f"\n{'Total':<40} {sum(snap.values()):>5} {sum(curr.values()):>5} {sum(snap.values()) - sum(curr.values()):>+6}")


if __name__ == "__main__":
    main()
