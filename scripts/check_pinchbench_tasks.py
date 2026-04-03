#!/usr/bin/env python3
"""Check current PinchBench task definitions against our training data.

Identifies:
- Tasks in PinchBench that we have NO training data for (new/renamed)
- Tasks we have training data for that no longer exist in PinchBench (removed/renamed)
- Task name mismatches (same ID, different name)

Usage:
  python scripts/check_pinchbench_tasks.py
"""
import json
import os
import re
from pathlib import Path
from collections import Counter

WS = os.environ.get("PBM_WORKSPACE", "/workspace/synthbench")
SKILL_DIR = Path(WS) / "skill" / "tasks"
TRAIN_FILE = Path(WS) / "data" / "train.jsonl"


def load_pinchbench_tasks():
    """Load current task definitions from PinchBench repo."""
    tasks = {}
    if not SKILL_DIR.exists():
        print(f"ERROR: PinchBench tasks not found at {SKILL_DIR}")
        return tasks

    for task_dir in sorted(SKILL_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        # Look for the task definition .md file
        md_files = list(task_dir.glob("*.md"))
        if not md_files:
            continue

        task_id = task_dir.name
        # Normalize to our format: "01_calendar" -> "task_01_calendar"
        if not task_id.startswith("task_"):
            task_id = f"task_{task_id}"

        # Read task name from frontmatter or first heading
        content = md_files[0].read_text(errors="replace")
        name = task_dir.name
        # Try frontmatter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                fm = content[3:end]
                for line in fm.splitlines():
                    if line.strip().startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip("'\"")
                        break
        # Try first heading
        if name == task_dir.name:
            m = re.search(r'^#\s+(.+)', content, re.MULTILINE)
            if m:
                name = m.group(1).strip()

        tasks[task_id] = {
            "name": name,
            "dir": str(task_dir),
            "md_file": str(md_files[0]),
            "content_preview": content[:200],
        }

    return tasks


def load_training_task_ids():
    """Get task IDs from current training data."""
    counts = Counter()
    if not TRAIN_FILE.exists():
        return counts
    for line in TRAIN_FILE.read_text().splitlines():
        if line.strip():
            try:
                counts[json.loads(line).get("task_id", "")] += 1
            except json.JSONDecodeError:
                pass
    return counts


def main():
    bench_tasks = load_pinchbench_tasks()
    train_counts = load_training_task_ids()

    print(f"\n{'='*80}")
    print(f"  PINCHBENCH TASK ALIGNMENT CHECK")
    print(f"{'='*80}")
    print(f"\n  PinchBench tasks found: {len(bench_tasks)}")
    print(f"  Training data task IDs: {len(train_counts)}")

    # Current PinchBench tasks
    print(f"\n  CURRENT PINCHBENCH TASKS:")
    print(f"  {'ID':<40} {'Name':<40} {'Train':>6}")
    print(f"  {'-'*40} {'-'*40} {'-'*6}")
    for tid in sorted(bench_tasks.keys()):
        name = bench_tasks[tid]["name"][:38]
        count = train_counts.get(tid, 0)
        marker = " NEW" if count == 0 else ""
        print(f"  {tid:<40} {name:<40} {count:>5}{marker}")

    # Tasks in training data but NOT in PinchBench
    orphaned = [t for t in train_counts if t and t not in bench_tasks]
    if orphaned:
        print(f"\n  ORPHANED TRAINING DATA (task no longer in PinchBench):")
        for tid in sorted(orphaned):
            print(f"  {tid:<40} {train_counts[tid]:>5} examples -- WASTED")

    # Tasks in PinchBench but NOT in training data
    missing = [t for t in bench_tasks if t not in train_counts or train_counts[t] == 0]
    if missing:
        print(f"\n  MISSING TRAINING DATA (PinchBench task with no examples):")
        for tid in sorted(missing):
            name = bench_tasks[tid]["name"][:60]
            print(f"  {tid:<40} {name}")

    # Summary
    print(f"\n  SUMMARY:")
    aligned = len([t for t in bench_tasks if train_counts.get(t, 0) > 0])
    print(f"    Aligned tasks:  {aligned}/{len(bench_tasks)}")
    print(f"    Missing data:   {len(missing)} tasks")
    print(f"    Orphaned data:  {len(orphaned)} tasks ({sum(train_counts[t] for t in orphaned)} wasted examples)")

    if missing or orphaned:
        print(f"\n  ACTION NEEDED:")
        if orphaned:
            print(f"    1. Remove training data for orphaned tasks (teaches wrong behavior)")
        if missing:
            print(f"    2. Generate training data for {len(missing)} new/renamed tasks")
        print(f"    3. Update task_loader.py mappings if task IDs changed")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
