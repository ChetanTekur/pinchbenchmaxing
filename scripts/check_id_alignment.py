#!/usr/bin/env python3
"""Check if TaskLoader IDs match training data task IDs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datagen.task_loader import load_tasks
from datagen.topup import count_existing

tasks = load_tasks()
counts = count_existing()

print("TaskLoader IDs vs Training Data IDs:\n")
print(f"  {'TaskLoader ID':<40} {'Data ID match?':<15} {'count':>5}")
print(f"  {'-'*40} {'-'*15} {'-'*5}")

mismatches = []
for tid in sorted(tasks.keys()):
    c = counts.get(tid, 0)
    match = "YES" if tid in counts or c > 0 else "NO"
    if match == "NO":
        mismatches.append(tid)
    print(f"  {tid:<40} {match:<15} {c:>5}")

# Also show data IDs not in TaskLoader
data_only = sorted(set(counts.keys()) - set(tasks.keys()))
if data_only:
    print(f"\nData IDs NOT in TaskLoader:")
    for tid in data_only:
        print(f"  {tid:<40} count={counts[tid]}")

if mismatches:
    print(f"\n⚠ {len(mismatches)} TaskLoader IDs have no matching data")
print()
