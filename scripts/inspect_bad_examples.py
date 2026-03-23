#!/usr/bin/env python3
"""
Inspect actual tool calls in training data to understand quality issues.

Runs validate_data logic per example and shows the actual tool calls
for examples that fail.

Usage:
  python scripts/inspect_bad_examples.py                     # top 5 worst tasks
  python scripts/inspect_bad_examples.py task_01_calendar 5  # specific task, 5 examples
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from datagen.validate_data import validate_example

cfg = load_config()
train_file = cfg.train_file


def main():
    task_filter = sys.argv[1] if len(sys.argv) > 1 else None
    n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Validate each example and collect failures
    bad_by_task = defaultdict(list)
    for line in train_file.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        task_id = ex.get("task_id", "?")
        issues = validate_example(ex)
        critical = [i for i in issues if i.get("severity") in ("critical", "high")]
        if critical:
            bad_by_task[task_id].append({"example": ex, "issues": critical})

    if task_filter:
        tasks_to_show = [task_filter]
    else:
        ranked = sorted(bad_by_task.items(), key=lambda x: -len(x[1]))
        tasks_to_show = [t for t, _ in ranked[:5]]

    for task_id in tasks_to_show:
        items = bad_by_task.get(task_id, [])
        if not items:
            print(f"\n{task_id}: no critical/high issues")
            continue

        print(f"\n{'='*70}")
        print(f"  {task_id} — {len(items)} bad examples")
        print(f"{'='*70}")

        # Summarize
        issue_types = defaultdict(int)
        for item in items:
            for issue in item["issues"]:
                issue_types[issue["check"]] += 1
        for check, count in sorted(issue_types.items(), key=lambda x: -x[1]):
            print(f"  {count:>4}x {check}")

        # Show N examples
        for i, item in enumerate(items[:n_examples]):
            ex = item["example"]
            msgs = ex.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "?")

            print(f"\n  --- Example {i+1} ---")
            print(f"  User: {user_msg[:150]}")
            print(f"  Issues:")
            for issue in item["issues"]:
                print(f"    ⚠ {issue['check']}: {issue['detail']}")

            print(f"  Tool calls made:")
            import re
            for msg in msgs:
                if msg["role"] == "assistant":
                    for block in re.findall(r'<tool_call>(.*?)</tool_call>', msg["content"], re.DOTALL):
                        try:
                            obj = json.loads(block.strip())
                            name = obj.get("name", "?")
                            args = obj.get("arguments", {})
                            print(f"    → {name}({json.dumps(args, default=str)[:120]})")
                        except json.JSONDecodeError:
                            print(f"    → PARSE ERROR: {block[:80]}")

    print()


if __name__ == "__main__":
    main()
