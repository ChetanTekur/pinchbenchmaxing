#!/usr/bin/env python3
"""Compare current train.jsonl against v21 snapshot counts and show diffs."""
import json
import os
from pathlib import Path
from collections import Counter

WS = os.environ.get("PBM_WORKSPACE", "/workspace/synthbench")
TRAIN = Path(WS) / "data" / "train.jsonl"

# Try to find v21 snapshot
snap_file = Path(WS) / "data" / "data_snapshot_v21.json"
if not snap_file.exists():
    print(f"No v21 snapshot at {snap_file}")
    exit(1)

snap = json.loads(snap_file.read_text())
v21_counts = snap.get("per_task", {})

# Count current
curr = Counter()
for line in TRAIN.read_text().splitlines():
    if line.strip():
        try:
            curr[json.loads(line).get("task_id", "")] += 1
        except json.JSONDecodeError:
            pass

print(f"v21 snapshot total: {sum(v21_counts.values())}")
print(f"Current train total: {sum(curr.values())}")
print()
print(f"{'Task':<40} {'v21':>5} {'curr':>5} {'delta':>6}")
print("-" * 58)
for t in sorted(set(list(v21_counts.keys()) + list(curr.keys()))):
    s = v21_counts.get(t, 0)
    c = curr.get(t, 0)
    d = c - s
    m = " <--" if d != 0 else ""
    print(f"{t:<40} {s:>5} {c:>5} {d:>+6}{m}")
