#!/usr/bin/env python3
"""
Analyze benchmark failure patterns from the latest benchmark log.

Reads the benchmark log and categorizes WHY each task failed:
- Did the model not use tools at all?
- Did it use the wrong tools?
- Did it stop after one step?
- Did it hit an error and give up?
- Did it produce wrong output?

Usage:
  python scripts/analyze_failures.py /workspace/synthbench/logs/bench_ollama_qwen35-9b-clawd-v11.log
"""

import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict


def parse_log(log_path):
    """Parse benchmark log to extract per-task results and notes."""
    text = Path(log_path).read_text(errors="replace")
    tasks = []

    # Find task results
    for match in re.finditer(
        r'Agent \[.*?\] starting task: (\S+).*?'
        r'Task (\S+): ([\d.]+)/1\.0 \((\d+)%\)(.*?)(?=Agent \[|$)',
        text, re.DOTALL
    ):
        task_id = match.group(2)
        score = float(match.group(3))
        pct = int(match.group(4))
        notes_match = re.search(r'Notes: (.+?)(?:\n|$)', match.group(5))
        notes = notes_match.group(1).strip() if notes_match else ""
        tasks.append({
            "task_id": task_id,
            "score": score,
            "pct": pct,
            "notes": notes[:300],
        })

    return tasks


def categorize_failure(task):
    """Categorize why a task failed based on notes."""
    notes = task["notes"].lower()
    score = task["score"]

    if score >= 0.9:
        return "PASS"
    if score >= 0.5:
        return "PARTIAL"

    categories = []

    if "truncated" in notes or "only shows" in notes or "partial" in notes:
        categories.append("INCOMPLETE — model started but didn't finish")
    if "no evidence" in notes or "did not" in notes or "never" in notes:
        categories.append("MISSING_OUTPUT — expected file/output not created")
    if "error" in notes and ("stopped" in notes or "gave up" in notes):
        categories.append("ERROR_QUIT — hit error and gave up")
    if "wrong" in notes or "incorrect" in notes:
        categories.append("WRONG_OUTPUT — produced incorrect content")
    if "one" in notes and ("file" in notes or "read" in notes or "attempt" in notes):
        categories.append("ONE_STEP — did one action then stopped")
    if not categories:
        if score == 0:
            categories.append("ZERO — complete failure (check transcript)")
        else:
            categories.append("LOW_QUALITY — attempted but scored poorly")

    return " | ".join(categories)


def main():
    if len(sys.argv) < 2:
        # Find most recent benchmark log
        log_dir = Path(os.environ.get("PBM_WORKSPACE", "/workspace/synthbench")) / "logs"
        logs = sorted(log_dir.glob("bench_ollama_qwen35-9b-clawd-v*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            print("No benchmark logs found")
            sys.exit(1)
        log_path = str(logs[0])
        print(f"Using most recent log: {log_path}")
    else:
        log_path = sys.argv[1]

    tasks = parse_log(log_path)

    if not tasks:
        print("Could not parse any tasks from log. Trying simpler parse...")
        # Simpler parse
        text = Path(log_path).read_text(errors="replace")
        for line in text.splitlines():
            m = re.search(r'Task (\S+): ([\d.]+)/1\.0 \((\d+)%\)', line)
            if m:
                task_id = m.group(1)
                score = float(m.group(2))
                pct = int(m.group(3))
                # Find notes on next line or same line
                notes_m = re.search(r'Notes: (.+)', line)
                notes = notes_m.group(1)[:300] if notes_m else ""
                tasks.append({"task_id": task_id, "score": score, "pct": pct, "notes": notes})

    print(f"\n{'='*70}")
    print(f"  BENCHMARK FAILURE ANALYSIS")
    print(f"  {len(tasks)} tasks parsed")
    print(f"{'='*70}")

    # Group by category
    passing = []
    failing = []

    for task in tasks:
        category = categorize_failure(task)
        task["category"] = category
        if task["score"] >= 0.5:
            passing.append(task)
        else:
            failing.append(task)

    print(f"\n  PASSING ({len(passing)} tasks):")
    for t in sorted(passing, key=lambda x: -x["score"]):
        print(f"    {t['task_id']:<40} {t['pct']:>3}%")

    print(f"\n  FAILING ({len(failing)} tasks):")
    for t in sorted(failing, key=lambda x: -x["score"]):
        print(f"\n    {t['task_id']:<40} {t['pct']:>3}%")
        print(f"    Category: {t['category']}")
        if t["notes"]:
            print(f"    Notes: {t['notes'][:200]}")

    # Pattern analysis
    print(f"\n{'='*70}")
    print(f"  FAILURE PATTERNS")
    print(f"{'='*70}")
    patterns = defaultdict(list)
    for t in failing:
        for cat in t["category"].split(" | "):
            patterns[cat.strip()].append(t["task_id"])

    for pattern, tasks_list in sorted(patterns.items(), key=lambda x: -len(x[1])):
        print(f"\n  {pattern} ({len(tasks_list)} tasks):")
        for tid in tasks_list:
            print(f"    - {tid}")

    print()


if __name__ == "__main__":
    main()
